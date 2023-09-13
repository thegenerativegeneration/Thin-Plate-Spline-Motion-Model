import os
from argparse import ArgumentParser

import yaml

from modules.gan import MultiScaleDiscriminator
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.bg_motion_predictor import BGMotionPredictor
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork

import torch

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='path to config')
    parser.add_argument('checkpoint', help='path to checkpoint to restore')
    parser.add_argument('--target-checkpoint', '-t', default=None, help='path to checkpoint to save')
    parser.add_argument('--discriminator', '-d', action='store_true', help='save discriminator')

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)

    checkpoint = torch.load(opt.checkpoint)
    print(checkpoint.keys())

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                   **config['model_params']['common_params'])

    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])

    bg_predictor = None
    if 'bg_predictor' in checkpoint:
        bg_predictor = BGMotionPredictor()
    else:
        print("No bg_predictor in checkpoint")

    avd_network = None
    if 'avd_network' in checkpoint:
        avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                                 **config['model_params']['avd_network_params'])

    if opt.discriminator and 'discriminator' in checkpoint:
        discriminator = MultiScaleDiscriminator(scales=[1], d=64)

    target_checkpoint = opt.target_checkpoint if opt.target_checkpoint is not None else os.path.join(
        opt.checkpoint.split('.')[0] + '_model_only.pth.tar')

    inpainting.load_state_dict(checkpoint['inpainting_network'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])

    if bg_predictor is not None:
        print("Loading bg_predictor")
        bg_predictor.load_state_dict(checkpoint['bg_predictor'])
    if avd_network is not None:
        print("Loading avd_network")
        avd_network.load_state_dict(checkpoint['avd_network'])
    if opt.discriminator and 'discriminator' in checkpoint:
        print("Loading discriminator")
        discriminator.load_state_dict(checkpoint['discriminator'])

    save_dict = {
        'inpainting_network': inpainting.state_dict(),
        'kp_detector': kp_detector.state_dict(),
        'dense_motion_network': dense_motion_network.state_dict()
    }
    if bg_predictor is not None:
        save_dict['bg_predictor'] = bg_predictor.state_dict()
    if avd_network is not None:
        save_dict['avd_network'] = avd_network.state_dict()
    if opt.discriminator and 'discriminator' in checkpoint:
        save_dict['discriminator'] = discriminator.state_dict()

    print(f"Saving keys: {save_dict.keys()}")

    torch.save(save_dict, target_checkpoint)
