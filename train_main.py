# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

import misrgan_model

from train_engine import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('miSRGAN training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU')
    
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--update_d_period', default=5, type=int, help="How many epoch for update discriminator periodically. Can balance the discriminator and the generator during training. ")
    
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--meta_data_path', default='/root/Dataset/CASIA2.0_revised/', type=str,
                        help='meta data json file path')
    parser.add_argument('--data_path', default='/root/Dataset/CASIA1.0 dataset', type=str,
                        help='dataset path(PKL file directory)')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    # 初始化一个分布式训练的参数
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    import utils.datasets

    dataset_train = utils.datasets.sr_dataset(
        meta_data_path= args.meta_data_path,
        data_path=args.data_path
        )
    
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model_g = misrgan_model.SRGAN_g()
    model_d = misrgan_model.SRGAN_d()
    vgg_perceptual = misrgan_model.vgg19_perceptual_loss()

    model_g.to(device)
    model_d.to(device)
    vgg_perceptual.to(device)

    model_g_without_ddp = model_g
    model_d_without_ddp = model_d
    vgg_perceptual_without_ddp = vgg_perceptual
    print("Model_g = %s" % str(model_g_without_ddp))
    print("Model_d = %s" % str(model_d_without_ddp))

    eff_batch_size = args.batch_size * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model_g = torch.nn.parallel.DistributedDataParallel(model_g, device_ids=[args.gpu], find_unused_parameters=False) # TODO FindersUnusedParameters False for Acceleration
        model_g_without_ddp = model_g.module
        
        model_d = torch.nn.parallel.DistributedDataParallel(model_d, device_ids=[args.gpu], find_unused_parameters=False) # TODO FindersUnusedParameters False for Acceleration
        model_d_without_ddp = model_d.module
                       
    
    # following timm: set wd as 0 for bias and norm layers
    args.opt='AdamW'
    args.betas=(0.9, 0.999)
    args.momentum=0.9
    optimizer_g  = optim_factory.create_optimizer(args, model_g_without_ddp)
    optimizer_d  = optim_factory.create_optimizer(args, model_d_without_ddp)
    
# = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer_g)
    print(optimizer_d)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_f1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model_g,
            model_d,
            data_loader_train,
            optimizer_g,
            optimizer_d,
            vgg_perceptual,
            device,
            epoch,
            log_writer=log_writer,
            args=args
        )
        # saving checkpoint
        if args.output_dir and (epoch % 5 == 0 and epoch != 0 or epoch + 1 == args.epochs):
            save_temp = {
                "model_g": model_g_without_ddp.state_dict(),
                "model_d": model_d_without_ddp.state_dict(),
                "optimizer_g" : optimizer_g.state_dict(),
                "optimizer_d" : optimizer_d.state_dict()
            }
            output_path = os.path.join(args.output_dir, f"checkpoint-{epoch}.pt")
            torch.save(save_temp, output_path)
            
            # misc.save_model(
            #     args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            #     loss_scaler=loss_scaler, epoch=epoch)
            
        # if epoch  % args.test_period == 0 or epoch + 1 == args.epochs:
        #     test_stats = test_one_epoch(
        #         model, 
        #         data_loader = data_loader_test, 
        #         device = device, 
        #         epoch = epoch, 
        #         log_writer=log_writer,
        #         args = args
        #     )
        #     local_f1 = test_stats['relative_f1']
        #     if local_f1 > best_f1 :
        #         best_f1 = local_f1
        #         print("Best F1 = %f" % best_f1)
        #         if epoch >35:
        #             misc.save_model(
        #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #         loss_scaler=loss_scaler, epoch=epoch)
                
        log_stats =  {**{f'train_{k}': v for k, v in train_stats.items()},
                        # **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,}
        # else:
        #     pass
        #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
