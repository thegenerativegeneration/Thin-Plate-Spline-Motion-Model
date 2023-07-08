import matplotlib
matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
from frames_dataset import FramesDataset

from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.bg_motion_predictor import BGMotionPredictor
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork
import torch
from train import train
from train_avd import train_avd
from reconstruction import reconstruction
import os
from torchinfo import summary
import bitsandbytes as bnb

optimizer_choices = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'adam8bit': bnb.optim.Adam8bit,
    "adamw8bit": bnb.optim.AdamW8bit,
}

if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.9")

    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "train_avd"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--detect_anomaly", action="store_true", help="detect anomaly in autograd")


    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    if opt.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])


    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])


    bg_predictor = None
    if (config['model_params']['common_params']['bg']):
        bg_predictor = BGMotionPredictor()

    avd_network = None
    if opt.mode == "train_avd":
        avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                             **config['model_params']['avd_network_params'])

    dataset = FramesDataset(is_train=(opt.mode.startswith('train')), **config['dataset_params'])
    print("Dataset length: ", len(dataset))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    optimizer_class = optimizer_choices[config['train_params']['optimizer']]

    print("Inpainting Network:")
    summary(inpainting)
    print("Keypoint Detector:")
    summary(kp_detector)
    print("Dense Motion Network:")
    summary(dense_motion_network)
    if bg_predictor is not None:
        print("Background Predictor:")
        summary(bg_predictor)

    if opt.mode == 'train':
        print("Training...")
        train(config, inpainting, kp_detector, bg_predictor, dense_motion_network, opt.checkpoint, log_dir, dataset,
              optimizer_class=optimizer_class)
    elif opt.mode == 'train_avd':
        print("Training Animation via Disentaglement...")
        train_avd(config, inpainting, kp_detector, bg_predictor, dense_motion_network, avd_network, opt.checkpoint,
                  log_dir, dataset, optimizer_class=optimizer_class)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        #TODO: update to accelerate
        reconstruction(config, inpainting, kp_detector, bg_predictor, dense_motion_network, opt.checkpoint, log_dir, dataset)

