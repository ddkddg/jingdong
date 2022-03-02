import copy
import os
import time
from os import path as osp

import motmetrics as mm
import numpy as np
import sacred
import torch
import yaml
from sacred import Experiment
from torch.utils.data import DataLoader
from tqdm import tqdm
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.oracle_tracker import OracleTracker
from tracktor.reid.resnet import ReIDNetwork_resnet50
from tracktor.tracker import Tracker
from tracktor.utils import (evaluate_mot_accums, get_mot_accum,
                            interpolate_tracks, plot_sequence)
from tracktor.reid.config import (check_cfg, engine_run_kwargs,
                                  get_default_config, lr_scheduler_kwargs,
                                  optimizer_kwargs, reset_config)
from torchreid.utils import FeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mm.lap.default_solver = 'lap'


ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')


# @ex.config
def add_reid_config(reid_models, obj_detect_models, dataset):
    # if isinstance(dataset, str):
    #     dataset = [dataset]
    #print("*************************")
    if isinstance(reid_models, str):
        reid_models = [reid_models, ] * len(dataset)
    #print("*************************")
    # if multiple reid models are provided each is applied
    # to a different dataset
    if len(reid_models) > 1:
        assert len(dataset) == len(reid_models)
    #print("*************************")
    if isinstance(obj_detect_models, str):
        obj_detect_models = [obj_detect_models, ] * len(dataset)
    if len(obj_detect_models) > 1:
        assert len(dataset) == len(obj_detect_models)
    #print("*************************")
    return reid_models, obj_detect_models, dataset


@ex.automain
def main(module_name, name, seed, obj_detect_models, ssh_models, reid_models,
         tracker, oracle, dataset, load_results, frame_range, interpolate,
         write_images, _config, _log, _run):
    
    sacred.commands.print_config(_run)
    
    # set all seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    output_dir = osp.join(get_output_dir(module_name), name)
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')
    
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(copy.deepcopy(_config), outfile, default_flow_style=False)

    dataset = Datasets(dataset)
    reid_models, obj_detect_models, dataset = add_reid_config(reid_models, obj_detect_models, dataset)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    _log.info("Initializing object detector(s).")
    #print(obj_detect_models)
    obj_detects = []
    for obj_detect_model in obj_detect_models:
        obj_detect = FRCNN_FPN(num_classes=2)
        #print(storage)
        obj_detect_state_dict = torch.load(
            obj_detect_model, map_location=lambda storage, loc: storage)
        if 'model' in obj_detect_state_dict:
            obj_detect_state_dict = obj_detect_state_dict['model']

        ssh_state_dict = torch.load(
            ssh_models, map_location=lambda storage, loc: storage)
        if 'model' in ssh_state_dict:
            ssh_state_dict = ssh_state_dict['model']

        obj_detect.load_state_dict(obj_detect_state_dict)
        obj_detects.append(obj_detect)

        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        from torch import nn

        
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        backbone = obj_detect.backbone
        #print(backbone.state_dict())
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        bb = [backbone.body.conv1, backbone.body.bn1, backbone.body.relu, backbone.body.maxpool, backbone.body.layer1, backbone.body.layer2]
        bb = nn.Sequential(*bb)
        #import torch.optim as optim
        #optimizer_ssh = optim.SGD([p for p in bb.parameters()] + [p for p in obj_detect.backbone.parameters()], lr=0.001)

        head = copy.deepcopy([backbone.body.layer3, backbone.body.layer4])
        class ViewFlatten(nn.Module):
            def __init__(self):
                super(ViewFlatten, self).__init__()
            def forward(self, x):
                return x.view(x.size(0), -1)
        head += [nn.AvgPool2d(kernel_size=8, stride=8, padding=0), ViewFlatten(), nn.Linear(in_features=5 * 6144, out_features=6, bias=True)]

        ssh = nn.Sequential(*head)
        #print(ssh.state_dict())
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        ssh.load_state_dict(ssh_state_dict)
        #print(ssh.state_dict())
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        bb.eval()
        ssh.eval()
        obj_detect.eval()
        
        
        obj_detect.to(device) 
        ssh.to(device)
        bb.to(device)
    #print(obj_detects)



    # reid
    _log.info("Initializing reID network(s).")

    reid_networks = []
    for reid_model in reid_models:
        assert os.path.isfile(reid_model)
        reid_network = FeatureExtractor(
            model_name='resnet50_fc512',
            model_path=reid_model,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu')

        reid_networks.append(reid_network)

    # tracktor
    if oracle is not None:
        tracker = OracleTracker(
            obj_detect, reid_network, tracker, oracle)
    else:
        tracker = Tracker(obj_detect, reid_network, tracker, ssh, bb)


    time_total = 0
    num_frames = 0
    mot_accums = []

    #for seq, obj_detect, reid_network in zip(dataset, obj_detects, reid_networks):
    for seq in dataset:
        
        tracker.obj_detect = obj_detect
        tracker.reid_network = reid_network
        # import torch.optim as optim
        # optimizer_ssh = optim.SGD([p for p in bb.parameters()] + [p for p in obj_detect.backbone.parameters()], lr=0.001)
        tracker.ssh = ssh
        tracker.bb = bb
        tracker.reset()

        _log.info(f"Tracking: {seq}")

        start_frame = int(frame_range['start'] * len(seq))
        end_frame = int(frame_range['end'] * len(seq))

        seq_loader = DataLoader(torch.utils.data.Subset(seq, range(start_frame, end_frame)))
        num_frames += len(seq_loader)
        
        import torch.optim as optim
        optimizer_ssh = optim.SGD([p for p in tracker.bb.parameters()], lr=0.001)

        #print('shall be sth. here before start')
        
        results = {}
        if load_results:
            results = seq.load_results(output_dir)
        if not results:
            start = time.time()

            for frame_data in tqdm(seq_loader):
                #with torch.no_grad():

                tracker.step(frame_data, optimizer_ssh)

            results = tracker.get_results()

            time_total += time.time() - start

            _log.info(f"Tracks found: {len(results)}")
            _log.info(f"Runtime for {seq}: {time.time() - start :.2f} s.")

            if interpolate:
                results = interpolate_tracks(results)

            _log.info(f"Writing predictions to: {output_dir}")
            seq.write_results(results, output_dir)

        if seq.no_gt:
            _log.info("No GT data for evaluation available.")
        else:
            mot_accums.append(get_mot_accum(results, seq_loader))

        if write_images:
            plot_sequence(
                results,
                seq,
                osp.join(output_dir, str(dataset), str(seq)),
                write_images)

    if time_total:
        _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
                f"{time_total:.2f} s for {num_frames} frames ({num_frames / time_total:.2f} Hz)")
    if mot_accums:
        _log.info("Evaluation:")
        evaluate_mot_accums(mot_accums,
                            [str(s) for s in dataset if not s.no_gt],
                            generate_overall=True)
