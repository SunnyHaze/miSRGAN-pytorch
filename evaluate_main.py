from skimage.metrics import peak_signal_noise_ratio as psnr_f


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
from typing import Iterable

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from matplotlib import pyplot as plt
import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

import misrgan_model


def test_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    device: torch.device, 
                    ckpt_epoch: int, 
                    log_writer=None,
                    args=None):
    # prepar for output dirs
    test_path = os.path.join(args.output_dir, str(ckpt_epoch))
    
    os.makedirs(test_path, exist_ok=True)
    
    # Inference
    with torch.no_grad():
        rank = misc.get_rank()
        model.zero_grad()
        model.eval()
        metric_logger = misc.MetricLogger(delimiter="  ")
        # F1 evaluation for an Epoch during training
        sum_TP, sum_TN, sum_FP, sum_FN = 0, 0, 0, 0
        print_freq = 20
        header = 'Test_infer: [{}]'.format(ckpt_epoch)
        
        
        # Inference and save images
        name_list = []
        psnr_list = []
        psnr_sum = 0
        cnt = 0
        for data_iter_step, (b_prev, b_gt, b_next, pkl_index, in_img_index) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            b_prev, b_next, b_gt = b_prev.to(device), b_next.to(device), b_gt.to(device) # N, 1, H, W
        

            b_sr = model(b_prev, b_next) # super resoluion
            b_sr = b_sr.detach()
            N, _, _, _ = b_sr.shape
            
            for i in range(N):
                prev, gt, sr, next = b_prev[i].detach(), b_gt[i], b_sr[i].detach(), b_next[i].detach()
                combine_name = f"{pkl_index[i]}_{in_img_index[i]}.png"
                if i % 3 == 0:
                    combine = torch.concat([prev, gt, sr, next], dim = 2)
                    # print("max:", torch.max(combine))
                    # print("min:", torch.min(combine))
                    combine = combine.permute(1, 2, 0)
                    combine = torch.repeat_interleave(combine, 3, 2)
                    combine = (combine + 1) / 2
                    combine = combine.detach().cpu().numpy()
                    output_full_path = os.path.join(test_path, combine_name)
            
                    plt.imsave(output_full_path, combine , cmap='gray')
                
                local_gt = gt[0].detach().cpu().numpy()
                local_sr = sr[0].detach().cpu().numpy()
                psnr_score = psnr_f(local_gt, local_sr)
                name_list.append(combine_name)
                psnr_list.append(psnr_score)
                cnt += 1
                psnr_sum += psnr_score
                
        avg_psnr_score = psnr_sum / cnt
        avg_psnr_score_reduce = misc.all_reduce_mean(avg_psnr_score)
        metric_logger.update(avg_psnr = avg_psnr_score_reduce)
        
        with open(os.path.join(test_path, f"device{rank}_psnr.json"), "w") as f:
            results = { 
                       "name_list":name_list,
                       "psnr_list":psnr_list
                       }
            json.dump(results, f)
        
  
        metric_logger.synchronize_between_processes()          
        if log_writer is not None:
            log_writer.add_scalar('test/psnr', avg_psnr_score_reduce, ckpt_epoch)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
                

    

def get_args_parser():
    parser = argparse.ArgumentParser('IML-ViT training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--test_batch_size', default=2, type=int,)
    parser.add_argument('--checkpoint_path', default = '/root/workspace/IML-ViT/output_dir_2', type=str, help='path to vit pretrain model by MAE')
    parser.add_argument('--epochs', default=200, type=int)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Dataset parameters
    parser.add_argument('--meta_data_path', default='/root/Dataset/CASIA2.0_revised/', type=str,
                        help='meta data json file path')
    parser.add_argument('--data_path', default='/root/Dataset/CASIA1.0 dataset', type=str,
                        help='dataset path')

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

    # ---- without crop augmentation ----

    dataset_test = utils.datasets.sr_dataset(
        meta_data_path= args.meta_data_path,
        data_path=args.data_path,
        if_return_index=True
    )
    
    print(dataset_test)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_test = %s" % str(sampler_test))


    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
        
    # define the model
    model_g = misrgan_model.SRGAN_g()

    model_g.to(device)

    model_g_without_ddp = model_g

    print("Model_g = %s" % str(model_g_without_ddp))
    
    if args.distributed:
        model_g = torch.nn.parallel.DistributedDataParallel(model_g, device_ids=[args.gpu], find_unused_parameters=False) # TODO FindersUnusedParameters False for Acceleration
        model_g_without_ddp = model_g.module

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    chkpt_list = os.listdir(args.checkpoint_path)
    print(chkpt_list)
    chkpt_pair = [(int(chkpt.split('-')[1].split('.')[0]) , chkpt) for chkpt in chkpt_list if chkpt.endswith(".pt")]
    chkpt_pair.sort(key=lambda x: x[0])
    print( "sorted checkpoint pairs in the ckpt dir: ",chkpt_pair)
    for epoch , chkpt_dir in chkpt_pair:
        if chkpt_dir.endswith(".pt"):
            print("Loading checkpoint: %s" % chkpt_dir)
            ckpt = os.path.join(args.checkpoint_path, chkpt_dir)
            ckpt = torch.load(ckpt, map_location='cuda')
            model_g.module.load_state_dict(ckpt['model_g'])            
            test_stats = test_one_epoch(
                model= model_g,
                data_loader=data_loader_test,
                device=device,
                ckpt_epoch=epoch,
                log_writer=log_writer,
                args=args               
            )
                
        log_stats =  {**{f'train_{k}': v for k, v in test_stats.items()},
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


