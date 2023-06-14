import logging
import os
from contextlib import nullcontext

import matplotlib

matplotlib.use('Agg')
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.spatial import ConvexHull
import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork
from utils import VideoReader, VideoWriter

logger = logging.getLogger("TPSMM")


def relative_kp(kp_source, kp_driving, kp_driving_initial):
    source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial['fg_kp'][0].data.cpu().numpy()).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_new = {k: v for k, v in kp_driving.items()}

    kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp'])
    kp_value_diff *= adapt_movement_scale
    kp_new['fg_kp'] = kp_value_diff + kp_source['fg_kp']

    return kp_new


def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.full_load(f)

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                   **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
    avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                             **config['model_params']['avd_network_params'])
    kp_detector.to(device)
    dense_motion_network.to(device)
    inpainting.to(device)
    avd_network.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    inpainting.load_state_dict(checkpoint['inpainting_network'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])

    inpainting.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    avd_network.eval()

    return inpainting, kp_detector, dense_motion_network, avd_network


def make_animation(source_image, driving_video_generator, inpainting_network, kp_detector, dense_motion_network,
                   avd_network, device: torch.device, mode='relative', autocast_dtype=torch.float16, autocast=False):
    assert mode in ['standard', 'relative', 'avd']
    with torch.no_grad():

        if autocast:
            autocast_context = torch.autocast(device_type=str(device), dtype=autocast_dtype)
        else:
            autocast_context = nullcontext()

        with autocast_context:
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            source = source.to(device)
            kp_source = kp_detector(source)

            first_frame = True

            for driving_frame_np in tqdm(driving_video_generator):

                driving_frame = torch.tensor(driving_frame_np[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(
                    device)
                if first_frame:
                    kp_driving_initial = kp_detector(driving_frame)
                    first_frame = False
                kp_driving = kp_detector(driving_frame)
                if mode == 'standard':
                    kp_norm = kp_driving
                elif mode == 'relative':
                    kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                          kp_driving_initial=kp_driving_initial)
                elif mode == 'avd':
                    kp_norm = avd_network(kp_source, kp_driving)
                dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                    kp_source=kp_source, bg_param=None,
                                                    dropout_flag=False)
                out = inpainting_network(source, dense_motion)

                yield np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]


def find_best_frame(source, driving, cpu):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        try:
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        except:
            pass
    return frame_num


def read_and_resize_frames(video_path, img_shape):
    reader = VideoReader(video_path)
    for frame in reader:
        resized_frame = resize(frame, img_shape)[..., :3]
        yield resized_frame

    reader.close()


def read_and_resize_frames_forward(video_path, img_shape, start_frame):
    reader = VideoReader(video_path)
    for idx, frame in enumerate(reader):
        if idx < start_frame:
            continue
        resized_frame = resize(frame, img_shape)[..., :3]
        yield resized_frame
    reader.close()


def read_and_resize_frames_backward(video_path, img_shape, end_frame):
    reader = VideoReader(video_path)

    frames = []
    for idx, frame in enumerate(reader):
        if idx > end_frame:
            break
        resized_frame = resize(frame, img_shape)[..., :3]
        frames.append(resized_frame)
    reader.close()
    return reversed(frames)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='checkpoints/vox.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='./assets/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='./assets/driving.mp4', help="path to driving video or folder of images")
    parser.add_argument("--result_video", default='./result.mp4', help="path to output. Can be file name or folder.")

    parser.add_argument("--img_shape", default="256,256", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')

    parser.add_argument("--mode", default='relative', choices=['standard', 'relative', 'avd'],
                        help="Animate mode: ['standard', 'relative', 'avd'], when use the relative mode to animate a face, use '--find_best_frame' can get better quality result")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--autocast", dest="autocast", action="store_true", help="Autocast mode.")


    opt = parser.parse_args()

    source_image = imageio.imread(opt.source_image)

    if os.path.isdir(opt.driving_video):
        fps = 30
        length = len(os.listdir(opt.driving_video))
    elif "%" in opt.driving_video:
        fps = 30
        length = len(os.path.dirname(opt.driving_video))
    else:
        reader = imageio.get_reader(opt.driving_video, mode='I')

        fps = reader.get_meta_data().get('fps', 30)
        length = int(reader.get_meta_data().get('duration', 0)) * int(fps)

        reader.close()

    if opt.cpu and opt.autocast:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16

    if opt.cpu or torch.cuda.device_count() == 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    source_image = resize(source_image, opt.img_shape)[..., :3]
    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path=opt.config,
                                                                                  checkpoint_path=opt.checkpoint,
                                                                                  device=device)


    def reversed_generator(generator):
        frames = list(generator)
        return reversed(frames)


    def append_frame_to_writer(frame, writer):
        writer.append_data(img_as_ubyte(frame))

    writer = VideoWriter(opt.result_video, mode='I', fps=fps)
    if opt.find_best_frame:
        driving_video_generator = read_and_resize_frames(opt.driving_video, opt.img_shape)
        i = find_best_frame(source_image, driving_video_generator, opt.cpu)
        driving_forward = read_and_resize_frames_forward(opt.driving_video, opt.img_shape, i)
        driving_backward = read_and_resize_frames_backward(opt.driving_video, opt.img_shape, i)

        with writer:
            # Generate and append frames for the reversed backward animation
            backward_animation = make_animation(source_image, driving_backward, inpainting, kp_detector,
                                                dense_motion_network, avd_network, device=device, mode=opt.mode,
                                                autocast_dtype=autocast_dtype, autocast=opt.autocast)

            for frame in reversed_generator(backward_animation):
                append_frame_to_writer(frame, writer)

            # Generate and append frames for forward animation, skipping the first frame
            for idx, frame in tqdm(enumerate(
                    make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network,
                                   avd_network, device=device, mode=opt.mode, autocast_dtype=autocast_dtype,
                                   autocast=opt.autocast
                                   )), total=length):
                if idx == 0:
                    continue
                append_frame_to_writer(frame, writer)
    else:
        with writer:
            driving_video_generator = read_and_resize_frames(opt.driving_video, opt.img_shape)
            for frame in tqdm(
                    make_animation(source_image, driving_video_generator, inpainting, kp_detector, dense_motion_network,
                                   avd_network, device=device, mode=opt.mode, autocast_dtype=autocast_dtype,
                                   autocast=opt.autocast
                                   ), total=length):
                append_frame_to_writer(frame, writer)
