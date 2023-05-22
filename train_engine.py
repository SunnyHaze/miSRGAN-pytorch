import math
import sys
from typing import Iterable

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched

import torch.nn.functional as F

from torchvision.transforms.functional import resize

def denormalize(image):
    return image
    return (image + 1) / 2 * 255



def train_one_epoch(g_model: torch.nn.Module,
                    d_model: torch.nn.Module,
                    data_loader: Iterable, 
                    g_optimizer: torch.optim.Optimizer,
                    d_optimizer: torch.optim.Optimizer,
                    perceptual_vgg: torch.nn.Module,
                    device: torch.device,
                    epoch: int,
                    log_writer=None,
                    args=None):
    g_model.train(True)
    d_model.train(True)
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    g_optimizer.zero_grad()
    d_optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
        
    # b_prev : batch previous image......
    for data_iter_step, (b_prev, b_gt, b_next) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % 20 == 0:
            lr_sched.adjust_learning_rate(g_optimizer, data_iter_step / len(data_loader) + epoch, args)
            lr_sched.adjust_learning_rate(d_optimizer, data_iter_step / len(data_loader) + epoch, args)  
        
        # ====update discriminator====
        b_prev, b_next, b_gt = b_prev.to(device), b_next.to(device), b_gt.to(device)
        
        b_sr = g_model(b_prev, b_next) # super resoluion
        
        # Debug for https://blog.csdn.net/qq_39237205/article/details/125728708 
        b_concat = torch.concat((b_sr, b_gt)) # [fake 0 , real 1]
        
        logits_concate = d_model(b_concat)
        
        N, _ = logits_concate.shape
        n_2 = int(N/2)
        
        logits_fake = logits_concate[:n_2, :]
        logits_real = logits_concate[n_2:, :]
    
        # ----discriminator loss-----
        d_loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
        d_loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))

        d_loss_real_value = d_loss_real.item()
        d_loss_fake_value = d_loss_fake.item()
    
        d_loss = 0.5 * d_loss_real + 0.5 * d_loss_fake
    
        d_loss_value = d_loss.item()
        
        if epoch >= 20 and epoch % args.update_d_period == 0:
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
        else:
            pass


        # =====update generator====
        b_prev, b_next, b_gt = b_prev.to(device), b_next.to(device), b_gt.to(device)
        
        b_sr = g_model(b_prev, b_next) # super resoluion
        
        # Debug for https://blog.csdn.net/qq_39237205/article/details/125728708 
        b_concat = torch.concat((b_sr, b_gt)) # [fake 0 , real 1]
        
        logits_concate = d_model(b_concat)
        
        N, _ = logits_concate.shape
        n_2 = int(N/2)
        
        logits_fake = logits_concate[:n_2, :]
        logits_real = logits_concate[n_2:, :]
        
        g_gan_loss = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake)) # Ones!
    
        g_gan_loss_value = g_gan_loss.item()
        
        g_mse_loss = F.mse_loss(b_sr, b_gt)
        g_mse_loss_value = g_mse_loss.item()
    
        # ----VGG perceptual loss----
        b_gt = b_gt.repeat_interleave(repeats=3, dim=1) # B, 3, 224, 224
        b_sr = b_sr.repeat_interleave(repeats=3, dim=1) # B, 3, 224, 224
        
        gt_embedding = perceptual_vgg( (b_gt + 1) / 2)    # Norm From volume ~ (-1, 1) to VGG ~ [0, 1]
        sr_embedding = perceptual_vgg( (b_sr + 1) / 2)    # Norm From volume ~ (-1, 1) to VGG ~ [0, 1]
        g_vgg_loss = F.mse_loss(sr_embedding, gt_embedding)
        g_vgg_loss_value = g_vgg_loss.item()
        
        # ----combined Gen Loss-----
        if epoch < 20:
            beta = 0
            gamma = 0
        else:
            beta = 2e-6
            gamma = 1e-3
        g_loss = gamma * g_gan_loss + g_mse_loss + beta *  g_vgg_loss
        g_loss_value = g_loss.item()
        
        with torch.autograd.set_detect_anomaly(True):
            # G backward
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
        torch.cuda.synchronize()
        # Log to metric
        lr = g_optimizer.param_groups[0]["lr"]
        # save to log.txt
        metric_logger.update(lr=lr,
                             d_loss_real=d_loss_real_value,
                             d_loss_fake=d_loss_fake_value,
                             d_loss=d_loss_value,
                             g_gan_loss = g_gan_loss_value,
                             g_mse_loss=g_mse_loss_value,
                             g_vgg_loss=g_vgg_loss_value,
                             g_loss=g_loss_value,                      
                            )
        d_loss_real_reduce = misc.all_reduce_mean(d_loss_real_value)
        d_loss_fake_reduce = misc.all_reduce_mean(d_loss_fake_value)
        d_loss_reduce = misc.all_reduce_mean(d_loss_value)
        
        g_gan_loss_reduce = misc.all_reduce_mean(g_gan_loss_value)
        g_mse_loss_reduce = misc.all_reduce_mean(g_mse_loss_value)
        g_vgg_loss_reduce = misc.all_reduce_mean(g_vgg_loss_value)
        g_loss_reduce = misc.all_reduce_mean(g_loss_value)
        

        if log_writer is not None and (data_iter_step + 1) % 100 == 0: 
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("lr/lr", lr, epoch_1000x)
            log_writer.add_scalar("train/d_loss_real", d_loss_real_reduce , epoch_1000x)
            log_writer.add_scalar("train/d_loss_fake", d_loss_fake_reduce, epoch_1000x)
            log_writer.add_scalar("train/d_loss", d_loss_reduce, epoch_1000x)
            
            log_writer.add_scalar("train/g_gan_loss", g_gan_loss_reduce, epoch_1000x)
            log_writer.add_scalar("train/g_mse_loss", g_mse_loss_reduce, epoch_1000x)
            log_writer.add_scalar("train/g_vgg_loss", g_vgg_loss_reduce, epoch_1000x)
            log_writer.add_scalar("train/g_loss", g_loss_reduce, epoch_1000x)
            
    if log_writer is not None:
        log_writer.add_images('train_visual/b_prev', denormalize(b_prev), epoch)
        log_writer.add_images('train_visual/b_next', denormalize(b_next), epoch)
        log_writer.add_images('train_visual/b_gt', denormalize(b_gt), epoch)
        log_writer.add_images('train_visual/b_sr', denormalize(b_sr), epoch)                   
        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}