import math
import sys
import time
import torch
import transforms as T
import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils

# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
    return x.flip(2).flip(1)

def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)

def change_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        # rgb:[0,1,2] , rbg:[0,2,1] , grb:[1,0,2] , gbr:[1,2,0] , brg:[2,0,1] , bgr:[2,1,0]
        if label == 1:
            img = img[[[0,2,1]]]
        elif label == 2:
            img = img[[[1,0,2]]]
        elif label == 3:
            img = img[[[1,2,0]]]
        elif label == 4:
            img = img[[[2,0,1]]]
        elif label == 5:
            img = img[[[2,1,0]]]
        images.append(img.unsqueeze(0))
        #print(img.shape)
    #return images
    return torch.cat(images)

def change_rgb_batch(batch, label):
	if label == 'rand':
		labels = torch.randint(6, (len(batch),), dtype=torch.long)
	elif label == 'expand':
		labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
					torch.zeros(len(batch), dtype=torch.long) + 1,
					torch.zeros(len(batch), dtype=torch.long) + 2,
					torch.zeros(len(batch), dtype=torch.long) + 3])
		batch = batch.repeat((4,1,1,1))
	else:
		assert isinstance(label, int)
		labels = torch.zeros((len(batch),), dtype=torch.long) + label
	return change_batch_with_labels(batch, labels), labels

def train_one_epoch(ssh, bb, model, optimizer, data_loader, device, epoch, print_freq):
    #print(bb)
    model.train()
    #print(bb)
    ssh.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    lr_scheduler = None
    if epoch == 1:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss().to(device)

    loss_dicts = []
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        #mine changes
        from collections import OrderedDict

        #print(len(images))
        #import numpy as np
        images_r, labels_ssh = change_rgb_batch(images, 'rand')
        #images_r = np.array(images_r)
        preprocessed_images, _ = model.transform(images_r, None)
        print(preprocessed_images.tensors.shape)
        features = bb(preprocessed_images.tensors)
        print(features.shape)
        # if isinstance(features, torch.Tensor):
        #     features = OrderedDict([(0, features)])
        features = ssh(features)
        # features = ssh[0](features)
        # print(1111111111111111111111111111)
        # print(features.shape)
        # features = ssh[1](features)
        # print(222222222222222222222222222222)
        # print(features.shape)
        # features = ssh[2](features)
        # print(33333333333333333333333333333)
        # print(features.shape)
        # features = ssh[3](features)
        # print(4444444444444444444444444444444)
        # print(features.shape)
        # features = ssh[4](features)
        # print(55555555555555555555555555555)
        # print(features.shape)
        print(features)

        for res, label in zip(features, labels_ssh):
            print(res)
            print(label)
            _, predi = res.max(0)
            if predi == label:
                print('right!')
            else:
                print('wrong!')
        loss_ssh = criterion(features, labels_ssh)
        

        #res = ssh(features)

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        losses += loss_ssh

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        # ssh_value = sum(loss for loss in loss_ssh.values())

        loss_value = losses_reduced.item()
        

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training.")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        loss_dict_reduced = {k: v.cpu() for k, v in loss_dict_reduced.items()}
        loss_dicts.append(loss_dict_reduced)
        # tb_writer.add_scalar('TRAIN/LR', lr_scheduler.get_last_lr(), epoch)
        

    return loss_dicts


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, iou_types=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(n_threads)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    if iou_types is None:
        iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    results = {}
    loss_dicts = []
    for image, targets in metric_logger.log_every(data_loader, 50, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
	#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output
               for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        results.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator, results, loss_dicts
