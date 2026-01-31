import argparse
import copy
import csv
import os
import warnings

import random
import math
import numpy
import torch
import tqdm
import yaml
from torch.utils import data
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import nets.nn as nn
import utils as util
from dataset import Dataset
from dataset_eval import Dataset as Dataset_Map
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
warnings.filterwarnings("ignore")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Returns a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf."""
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1

def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn
def convert_optimizer_state_dict_to_fp16(state_dict):
    """
    Converts the state_dict of a given optimizer to FP16, focusing on the 'state' key for tensor conversions.

    This method aims to reduce storage size without altering 'param_groups' as they contain non-tensor data.
    """
    for state in state_dict["state"].values():
        for k, v in state.items():
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict

def train(args, params):
    # Model
    model = nn.yolo_v8_s(len(params['names'].values())).cuda()
    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    p = [], [], []
    for name, module in model.named_modules():    
            if 'decoder' not in name:
                if hasattr(module, 'bias') and isinstance(module.bias, torch.nn.Parameter):
                    p[2].append(module.bias)
                if isinstance(module, torch.nn.BatchNorm2d):
                    p[1].append(module.weight)
                elif hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                    p[0].append(module.weight)
    # optimizer = torch.optim.SGD(p[2], params['lr0'], params['momentum'], nesterov=True)
    # optimizer.add_param_group({'params': p[0], 'weight_decay': params['weight_decay'], 'lr': params['lr0']})
    # optimizer.add_param_group({'params': p[1], 'lr': params['lr0']})
    
    optimizer = torch.optim.AdamW(
        p[2],                   # biases (no weight decay)
        lr=params['lr0'],
        weight_decay=0.0
    )
    optimizer.add_param_group({
        'params': p[0],         # other weights (with weight decay)
        'weight_decay': params['weight_decay'],
        'lr': params['lr0'],
    })
    optimizer.add_param_group({
        'params': p[1],         # batchnorm weights (no weight decay)
        'weight_decay': 0.0,
        'lr': params['lr0'],
    })
    
         
    d = [], [], []
    for name, module in model.named_modules():
        if 'decoder' in name:
            if hasattr(module, 'bias') and isinstance(module.bias, torch.nn.Parameter):
                d[2].append(module.bias)
            if isinstance(module, torch.nn.BatchNorm2d):
                d[1].append(module.weight)
            elif hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                d[0].append(module.weight)

    # optimizer.add_param_group({'params': d[2], 'lr': params['lr1'], 'momentum': params['momentum'], 'nesterov': True})
    # optimizer.add_param_group({'params': d[0], 'weight_decay': params['weight_decay'], 'lr': params['lr1']})
    # optimizer.add_param_group({'params': d[1], 'lr': params['lr1']})
    
    optimizer.add_param_group({
        'params': d[2],         # decoder biases (no weight decay)
        'weight_decay': 0.0,
        'lr': params['lr1'],
    })
    optimizer.add_param_group({
        'params': d[0],         # decoder weights (with weight decay)
        'weight_decay': params['weight_decay'],
        'lr': params['lr1'],
    })
    optimizer.add_param_group({
        'params': d[1],         # decoder batchnorm weights (no weight decay)
        'weight_decay': 0.0,
        'lr': params['lr1'],
    })

    del p, d


    # Scheduler
    lr = one_cycle(1,params['lrf'],args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = []
    gt_filenames = []
    with open('train.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip()
            filenames.append(filename)
    with open('train_gt.txt') as gt_reader:
        for filename in gt_reader.readlines():
            filename = filename.rstrip()
            gt_filenames.append(filename)
    current_epoch = 0
    with open('training.txt', 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            current_epoch = int(line.strip())

    dataset = Dataset(filenames, gt_filenames, args.input_size, params, True)
    filenames = []
    gt_filenames = []
    with open('test.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip()
            filenames.append(filename)
    with open('test_gt.txt') as gt_reader:
        for filename in gt_reader.readlines():
            filename = filename.rstrip()
            gt_filenames.append(filename)

    valid_dataset = Dataset(filenames,gt_filenames, args.input_size, params, False)
    
    filenames = []
    with open('test_rtts.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip()
            filenames.append(filename)
    rtts_dataset = Dataset_Map(filenames, args.input_size, params, False)
    
    filenames = []
    with open('test_foggy.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip()
            filenames.append(filename)

    foggy_dataset = Dataset_Map(filenames, args.input_size, params, False)
    if args.world_size <= 1:
        sampler = None
    else:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)
    valid_loader = data.DataLoader(valid_dataset, 8, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset.collate_fn)
    rtts_loader = data.DataLoader(rtts_dataset, 8, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset_Map.collate_fn)
    foggy_loader = data.DataLoader(foggy_dataset, 8, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset_Map.collate_fn)

    if args.world_size > 1:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    amp_scale = torch.cuda.amp.GradScaler()
    start_epoch = 0

    # Start training
    if args.resume:
        ckpt = torch.load(args.model_path, map_location = 'cuda', weights_only = False)
        print("Resuming Training: Epochs ",ckpt['last_epoch'])
        optimizer.load_state_dict(ckpt["optimizer"]) 
        ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
        ema.updates = ckpt["updates"]
        model.load_state_dict(ckpt["ema"].float().state_dict())
        args.epochs = args.epochs - ckpt['last_epoch']
        amp_scale.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['last_epoch']
    else:
        if args.model_path is not None:
            print('loading pretrained weights...')
            model_dict = model.state_dict()
            # print('total model weights: ', len(model_dict))
            
            # load mixyolo
            pretrained_dict = torch.load(args.model_path, map_location = 'cuda', weights_only = False)['model'].state_dict()            
            keys = ['net.', 'fpn.', 'head.']
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if any(k.startswith(prefix) for prefix in keys) and k in model_dict and numpy.shape(model_dict[k]) == numpy.shape(v)
            }            
            
            # # load dseu weight
            # dseu_dict = torch.load(args.dseu_path, map_location = 'cuda')

            # if 'model' in dseu_dict:
            #     dseu_dict = dseu_dict['model'].state_dict()
                        
            # for k, v in dseu_dict.items():
            #     if k in model_dict and numpy.shape(model_dict[k]) == numpy.shape(v):
            #         pretrained_dict[k] = v
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            # print(f"Total weights loaded: {len(pretrained_dict)}")
            
            start_epoch = 0

            print(f'"Pretrained Loaded: " {args.model_path}')
            print("Batch Size:", args.batch_size,"Detection Learning Rate:",params['lr0'],' Detection Weight:',args.detection_weight, ' Dehazing Learning Rate', params['lr1'] ,' Dehazing Weight:', args.dehazing_weight)

    best = 0
    num_batch = len(loader)
    criterion = util.ComputeLoss(model, params)
    dehaze_criterion = util.get_dseu_loss()
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 1000)
    with open(f'weights/{current_epoch}_640_{params["lr0"]}_{params["lr1"]}_{args.detection_weight}_{args.dehazing_weight}_3dec_aug_msb.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'mAP@50', 'mAP', 'mAP_RTTS','mAP_Foggy','PSNR','SSIM','detection_loss','dehazing_loss','valid_detection_loss','valid_dehazing_loss','loss','valid_loss'])
            writer.writeheader()
        for epoch in range(args.epochs):
            model.train()

            # if args.epochs - epoch == 10:
            #     loader.dataset.mosaic = False

            m_loss = util.AverageMeter()
            m_detect_loss = util.AverageMeter()
            m_dehaze_loss = util.AverageMeter()
            if args.world_size > 1:
                sampler.set_epoch(epoch)
            p_bar = enumerate(loader)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 6) % ('epoch', 'memory', 'lr', 'det_loss', 'res_loss','loss'))
            if args.local_rank == 0:
                p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

            optimizer.zero_grad()

            for i, (samples, gt_samples, targets, _) in p_bar:
                x = i + num_batch * epoch  # number of iterations
                samples = samples.cuda().float() / 255
                gt_samples = gt_samples.cuda().float() / 255
                targets = targets.cuda()

                # Warmup
                if x <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (args.batch_size * args.world_size)]
                    accumulate = max(1, numpy.interp(x, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            fp = [params['warmup_bias_lr'], y['initial_lr'] * lr(epoch)]
                        else:
                            fp = [0.0, y['initial_lr'] * lr(epoch)]
                        y['lr'] = numpy.interp(x, xp, fp)
                        if 'momentum' in y:
                            fp = [params['warmup_momentum'], params['momentum']]
                            y['momentum'] = numpy.interp(x, xp, fp)

                # Forward
                with torch.cuda.amp.autocast():
                    outputs = model(samples)  # forward
                    detection_output = outputs[0]
                    dehazing_output = outputs[1]
                    dehaze_loss = dehaze_criterion(dehazing_output,gt_samples)

                loss = criterion(detection_output, targets)
                total_loss = args.detection_weight*loss + args.dehazing_weight*dehaze_loss
                m_loss.update(total_loss.item(), samples.size(0))
                m_detect_loss.update(loss.item(), samples.size(0))
                m_dehaze_loss.update(dehaze_loss.item(), samples.size(0))
                total_loss *= args.batch_size  # loss scaled by batch_size
                total_loss *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(total_loss).backward()
                lr_show = 0.0
                # Optimize
                if x % accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    lr_show = get_lr(optimizer)
                    optimizer.zero_grad()

                    if ema:
                        ema.update(model)

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.4g' * 4) % (f'{epoch + 1}/{args.epochs}', memory, lr_show, m_detect_loss.avg,m_dehaze_loss.avg,m_loss.avg)
                    p_bar.set_description(s)

                del loss
                del dehaze_loss
                del total_loss
                del outputs
                del detection_output
                del dehazing_output

            # Scheduler
            scheduler.step()
            start_epoch += 1
            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema, current_epoch, valid_loader)
                map_rtts = test_detect(args, params,'test_rtts.txt' ,ema.ema, current_epoch, rtts_loader)
                map_foggy = test_detect(args, params,'test_foggy.txt' ,ema.ema, current_epoch, foggy_loader)

                writer.writerow({'mAP': str(f'{last[1]:.4f}'),
                                 'epoch': str(start_epoch).zfill(3),
                                 'mAP@50': str(f'{last[0]:.6f}'),
                                 'mAP_RTTS': str(f'{map_rtts:.6f}'),
                                 'mAP_Foggy': str(f'{map_foggy:.6f}'),
                                 'loss': str(f'{m_loss.avg:.3f}'),
                                 'detection_loss': str(f'{m_detect_loss.avg:.3f}'),
                                 'dehazing_loss': str(f'{m_dehaze_loss.avg:.3f}'),
                                 'valid_loss' :str(f'{last[6]:.3f}'),
                                 'valid_detection_loss': str(f'{last[4]:.3f}'),
                                 'valid_dehazing_loss': str(f'{last[5]:.3f}'),
                                 'PSNR': str(f'{last[2]:.4f}'),
                                 'SSIM': str(f'{last[3]:.4f}')})
                f.flush()

                # Update best mAP
                if last[1] > best:
                    best = last[1]
                ckpt = {'model': copy.deepcopy(ema.ema).half()}
                # Save model
                ckpt_last = {
                                "last_epoch": start_epoch,
                                "model": model.state_dict(),  # resume and final checkpoints derive from EMA
                                "ema": copy.deepcopy(ema.ema).half(),
                                "updates": ema.updates,
                                "optimizer": convert_optimizer_state_dict_to_fp16(copy.deepcopy(optimizer.state_dict())),
                                "scaler": amp_scale.state_dict(),
                            }

                # Save last, best and delete
                torch.save(ckpt_last, './weights/last.pt')
                if best == last[1]:
                    torch.save(ckpt, './weights/best.pt')
                del ckpt

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')  # strip optimizers
        util.strip_last_optimizer('./weights/last.pt')  # strip optimizers
    # Update the contents
    with open('training.txt', 'w') as writer:
        writer.write(str(current_epoch + 1) + '\n')  # Write the updated number to the file
    torch.cuda.empty_cache()

@torch.no_grad()
def test(args, params, model=None, current_epoch=0, loader=None):
    if loader is None:
        filenames = []
        gt_filenames = []
        with open('./test.txt', 'r') as reader:
            for filename in reader.readlines():
                filename = filename.rstrip()
                filenames.append(filename)
        
        with open('./test_gt.txt', 'r') as gt_reader:
            for filename in gt_reader.readlines():
                filename = filename.rstrip()
                gt_filenames.append(filename)

        dataset = Dataset(filenames, gt_filenames, args.input_size, params, False)
        loader = data.DataLoader(dataset, 8, False, num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)
    if model is None:
        model = torch.load('./weights/best.pt', map_location='cuda', weights_only=False)['model'].float()
    m_loss = util.AverageMeter()
    m_detect_loss = util.AverageMeter()
    m_dehaze_loss = util.AverageMeter()
    criterion = util.ComputeLoss(model, params)
    dehaze_criterion = util.get_dseu_loss()
    
    model.half()
    model.eval()

    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    total_psnr, total_ssim, count = 0.0, 0.0, 0

    class_scores = {}  # Dictionary to store scores for each class

    p_bar = tqdm.tqdm(loader, desc=('%10s' * 6) % ('mAP', 'PSNR', 'SSIM', 'det_vloss', 'res_vloss', 'v_loss'))
    for samples, gt_samples, targets, shapes in p_bar:
        samples = samples.cuda()
        gt_samples = gt_samples.cuda()
        targets = targets.cuda()

        samples = samples.half()  # uint8 to fp16/32
        gt_samples = gt_samples.half()
        samples = samples / 255  # 0 - 255 to 0.0 - 1.0
        gt_samples = gt_samples / 255
        _, _, height, width = samples.shape  # batch size, channels, height, width

        # Inference
        outputs = model(samples)

        # Valid
        loss = criterion(outputs[0][0], targets)
        dehaze_loss = dehaze_criterion(outputs[1], gt_samples)
        total_loss = args.detection_weight * loss + args.dehazing_weight * dehaze_loss
        m_loss.update(total_loss.item(), samples.size(0))
        m_detect_loss.update(loss.item(), samples.size(0))
        m_dehaze_loss.update(dehaze_loss.item(), samples.size(0))

        detection_output = outputs[0][1]
        dehazing_output = outputs[1]

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height)).cuda()  # to pixels
        detection_output = util.non_max_suppression(detection_output, 0.001, 0.65)

        # Metrics
        for i, output in enumerate(detection_output):
            labels = targets[targets[:, 0] == i, 1:]
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if labels.shape[0]:
                    metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                continue

            detections = output.clone()
            util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

            # Evaluate
            if labels.shape[0]:
                tbox = labels[:, 1:5].clone()  # target boxes
                tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
                tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
                tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
                tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
                util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                correct = numpy.zeros((detections.shape[0], iou_v.shape[0]))
                correct = correct.astype(bool)

                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                correct_class = t_tensor[:, 0:1] == detections[:, 5]
                for j in range(len(iou_v)):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                        matches = matches.cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True
                correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
            metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

            out = torch.clamp((torch.clamp(dehazing_output[i][:, :shapes[i][0][0], :shapes[i][0][1]], 0, 1).mul(255)), 0, 255).byte()
            nohaze = torch.clamp(gt_samples[i][:, :shapes[i][0][0], :shapes[i][0][1]].mul(255), 0, 255).byte()
            y, gt = util.rgb_to_y(out.double()), util.rgb_to_y(nohaze.double())
            current_psnr, current_ssim = util.psnr(y, gt), util.ssim(y, gt)
            total_psnr += current_psnr.item()
            total_ssim += current_ssim.item()
            count += 1
            dehazing_img = dehazing_output[i].permute(1, 2, 0).cpu().numpy()
            dehazing_img = dehazing_img.clip(0, 255)
            gt_img = gt_samples[i].permute(1, 2, 0).cpu().numpy()

    dehazed_img_path = os.path.join('dehazed_images', f'dehazed_{current_epoch}_3dec_{args.detection_weight}_{args.dehazing_weight}_{params["lr0"]}__{params["lr1"]}.png')
    cv2.imwrite(dehazed_img_path, cv2.cvtColor(numpy.uint8(255 * dehazing_img), cv2.COLOR_RGB2BGR))
    
    del loss
    del dehaze_loss
    del total_loss

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    # Define class names
    class_names = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorbike',
        4: 'bus'
    }

    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap, ap_class = util.compute_ap(*metrics)
        # class_labels = [class_names[c] for c in ap_class.keys()]
        # ap_values = list(ap_class.values())

        # # Plotting
        # plt.figure(figsize=(10, 5))
        # bars = plt.bar(class_labels, ap_values)
        # plt.xlabel('Classes')
        # plt.ylabel('Average Precision (AP)')
        # plt.title('Average Precision (AP) for each class')
        # plt.xticks(rotation=45)
        # plt.tight_layout()

        # # Adding labels
        # for bar in bars:
        #     height = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom')

        # # Save the plot
        # plot_path = f'ap_per_class.png'
        # plt.savefig(plot_path)
        # plt.show()
    # Print results
    print('%10.3g' * 6 % (map50, total_psnr / count, total_ssim / count, m_detect_loss.avg, m_dehaze_loss.avg, m_loss.avg))
    # Plot AP for each class

    model.float()  # for training
    return map50, mean_ap, total_psnr / count, total_ssim / count, m_detect_loss.avg, m_dehaze_loss.avg, m_loss.avg

@torch.no_grad()
def test_detect(args, params,input_file='test_rtts.txt', model=None, current_epoch=0, loader=None):
    if loader is None:
        print("Reading Files ", input_file, " ...")

        filenames = []
        with open(input_file) as reader:
            for filename in reader.readlines():
                filename = filename.rstrip()
                filenames.append(filename)
        
        dataset = Dataset_Map(filenames, args.input_size, params, False)
        loader = data.DataLoader(dataset, 8, False, num_workers=8,
                                pin_memory=True, collate_fn=Dataset_Map.collate_fn)

    if model is None:
        model = torch.load('./weights/best.pt', map_location='cuda')['model'].float()
    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 1) % ('mAP'))
    for samples,  targets, shapes in p_bar:
        samples = samples.cuda()
    
        targets = targets.cuda()
        samples = samples.half()  # uint8 to fp16/32
    
        samples = samples / 255  # 0 - 255 to 0.0 - 1.0

        _, _, height, width = samples.shape  # batch size, channels, height, width

        # Inference
        outputs = model(samples)
        # Valid
        
        detection_output = outputs[0][1]

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height)).cuda()  # to pixels
        detection_output = util.non_max_suppression(detection_output, 0.001, 0.65)

        # Metrics
        for i, output in enumerate(detection_output):
            labels = targets[targets[:, 0] == i, 1:]
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if labels.shape[0]:
                    metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                continue

            detections = output.clone()
            util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

            # Evaluate
            if labels.shape[0]:
                tbox = labels[:, 1:5].clone()  # target boxes
                tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
                tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
                tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
                tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
                util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                correct = numpy.zeros((detections.shape[0], iou_v.shape[0]))
                correct = correct.astype(bool)

                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                correct_class = t_tensor[:, 0:1] == detections[:, 5]
                for j in range(len(iou_v)):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                        matches = matches.cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True
                correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
            metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    class_names = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorbike',
        4: 'bus'
    }
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap,ap_class = util.compute_ap(*metrics)
        # class_labels = [class_names[c] for c in ap_class.keys()]
        # ap_values = list(ap_class.values())

        # # Plotting
        # plt.figure(figsize=(10, 5))
        # bars = plt.bar(class_labels, ap_values)
        # plt.xlabel('Classes')
        # plt.ylabel('Average Precision (AP)')
        # plt.title('Average Precision (AP) for each class')
        # plt.xticks(rotation=45)
        # plt.tight_layout()

        # # Adding labels
        # for bar in bars:
        #     height = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom')

        # # Save the plot
        # plot_path = f'ap_per_class.png'
        # plt.savefig(plot_path)
        # plt.show()
    # Print results
    print('%10.3g' * 1 % (map50))
   
    model.float()  # for training
    return map50
CLASS_COLORS = {
    'person': (0, 255, 0),    # Green
    'bicycle': (255, 0, 0),   # Blue
    'car': (0, 0, 255),       # Red
    'motorcycle': (255, 255, 0), # Cyan
    'bus': (255, 0, 255)      # Magenta
}
@torch.no_grad()
def inference(model_path, image_path, args, params, device='cpu',apply_dehazing=True, conf_threshold=0.50):
    def load_image(args,img):
        image = cv2.imread(img)
        h, w = image.shape[:2]
        r = args.input_size / max(h, w)
        if r != 1:
            image = cv2.resize(image,
                               dsize=(int(w * r), int(h * r)),
                               interpolation=cv2.INTER_LINEAR)
        return image, (h, w)
    
    def resize(image, input_size, augment):
        # Resize and pad image while meeting stride-multiple constraints
        shape = image.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(input_size / shape[0], input_size / shape[1])
        if not augment:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        pad = int(round(shape[1] * r)), int(round(shape[0] * r))
        w = (input_size - pad[0]) / 2
        h = (input_size - pad[1]) / 2

        if shape[::-1] != pad:  # resize
            image = cv2.resize(image,
                            dsize=pad,
                           interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
        left, right = int(round(w - 0.1)), int(round(w + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border
        return image, (r, r), (w, h)

    # Create output folder if it doesn't exist
    output_folder = 'inference_output'
    os.makedirs(output_folder, exist_ok=True)

    # Preprocess the image
    image, shape = load_image(args,image_path)
    h, w = image.shape[:2]
    img_resized, ratio, pad = resize(image, args.input_size, False)
    shapes = shape, ((h / shape[0], w / shape[1]), pad)  # for COCO mAP rescaling

    img = img_resized.transpose((2, 0, 1))[::-1]
    img = numpy.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)  # Add this line to add the batch dimension
    # print(img.shape)  # This should now print torch.Size([1, 3, 640, 640])
    # Load the model
    model = torch.load(model_path, map_location=device, weights_only=False)['model'].float().to(device)
    
    # Inference
    model.eval()
    outputs = model(img)

    # Process outputs
    detection_output = outputs[0][1]
    dehazing_output = outputs[1]

    # NMS
    detection_output = util.non_max_suppression(detection_output, 0.001, 0.65)

    # Restore image
    restored_img = torch.clamp(dehazing_output[0].cpu(), 0, 1).permute(1, 2, 0).numpy()
    restored_img = (restored_img * 255).astype(numpy.uint8)
    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
    original_resized = numpy.array(img_resized)
    original_resized = cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_folder, 'original_resized.jpg'), original_resized)

    # Draw bounding boxes and labels
    detections = detection_output[0]

    if detections is not None and len(detections):
        for det in detections:
            if det[4].cpu().numpy() > conf_threshold:
                bbox = det[:4].cpu().numpy()
                conf = det[4].cpu().numpy()
                cls = int(det[5].cpu().numpy())

                x1, y1, x2, y2 = bbox.astype(int)
                
                # Get class name and color
                class_name = params['names'][cls]
                color = CLASS_COLORS.get(class_name, (0, 255, 0))  # Default to green if class not found
                
                # Draw bounding box
                cv2.rectangle(restored_img, (x1, y1), (x2, y2), color, 2)
                
                # Put class name and confidence
                label = f'{class_name} {conf:.2f}'
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(restored_img, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                cv2.putText(restored_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                if not apply_dehazing:
                    cv2.rectangle(original_resized, (x1, y1), (x2, y2), color, 2)
                
                    # Put class name and confidence
                    cv2.rectangle(original_resized, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                    cv2.putText(original_resized, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save output image
    output_path = os.path.join(output_folder, 'dehazed_with_detections.jpg')
    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, restored_img)

    print(f"Output image saved to {output_path}")
    if not apply_dehazing:
        print("False")
        cv2.imwrite(os.path.join(output_folder, 'dehazed_with_detections.jpg'), original_resized)

        return original_resized
    cv2.imwrite(os.path.join(output_folder, 'original_resized.jpg'), original_resized)
    return restored_img
    
@torch.no_grad()
def inference_test_set(model_path, args, params, test_set='test', device="cpu"):
    # Load model
    model = torch.load(model_path, map_location=device)['model'].float()
    model.half()
    model.eval()

    output_dir = f'output_{test_set}'
    os.makedirs(output_dir, exist_ok=True)

    filenames = []
    with open(f'test_{test_set}.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip()
            filenames.append(filename)

    dataset = Dataset_Map(filenames, args.input_size, params, False)
    loader = data.DataLoader(dataset, 8, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset_Map.collate_fn)

    for batch_idx, (samples, targets, shapes) in enumerate(tqdm.tqdm(loader, desc='Processing test set')):
        samples = samples.to(device).half() / 255.0  # Normalize and convert to half precision

        # Inference
        outputs = model(samples)
        dehazing_output = outputs[1]

        for i in range(samples.shape[0]):
            # Restore image
            restored_img = torch.clamp(dehazing_output[i].cpu(), 0, 1).permute(1, 2, 0).numpy()
            restored_img = (restored_img * 255).astype(numpy.uint8)
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

            # Save output image
            output_filename = f'{output_dir}/dehazed_{batch_idx}_{i}.jpg'
            cv2.imwrite(output_filename, restored_img)

    print(f"Processed images saved to {output_dir}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model_path',default='pretrain_weights/best2.pt',type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_input',type=str)
    parser.add_argument('--inference_test', action='store_true')
    parser.add_argument('--data', default='voc',type=str)
    parser.add_argument('--detection_weight', default=0.8, type=int)
    parser.add_argument('--dehazing_weight', default=0.2, type=int)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))

    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    with open(os.path.join('args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.train:
        train(args, params)
    if args.test:
        if args.data == "rtts":
            print("Evaluating Rtts")
            test_detect(args,params,'test_rtts.txt')
        if args.data == "foggy":
            print("Evaluating Foggy")
            test_detect(args,params,'test_foggy.txt')
        if args.data == "voc":
            test(args, params)
    if args.inference:
        inference(args.model_path,args.inference_input,args,params, device)
    #'./weights/best.pt'
    #'./test/VOCtest-FOG/2012_004312.jpg'
    if args.inference_test:
        inference_test_set('./weights/best.pt', args, params, test_set=args.data, device=device)

if __name__ == "__main__":
    main()



