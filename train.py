from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from modules.model import GeneratorFullModel
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_
from frames_dataset import DatasetRepeater
from tqdm import tqdm
import math
from accelerate import Accelerator
from torchview import draw_graph

torch.backends.cudnn.benchmark = True

accelerator = Accelerator()


def train(config, inpainting_network, kp_detector, bg_predictor, dense_motion_network, checkpoint, log_dir, dataset,
          optimizer_class=torch.optim.Adam
          ):
    train_params = config['train_params']
    optimizer_params = config['train_params'].get('optimizer_params', {})

    optimizer = optimizer_class(
        [{'params': list(inpainting_network.parameters()) +
                    list(dense_motion_network.parameters()) +
                    list(kp_detector.parameters()),
          'initial_lr': train_params['lr_generator']}],
        lr=train_params['lr_generator'], **optimizer_params)

    optimizer_bg_predictor = None
    if bg_predictor:
        optimizer_bg_predictor = optimizer_class(
            [{'params': bg_predictor.parameters(), 'initial_lr': train_params['lr_generator']}],
            lr=train_params['lr_generator'], betas=(0.5, 0.999), weight_decay=1e-4)

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(
            checkpoint, inpainting_network=inpainting_network, dense_motion_network=dense_motion_network,
            kp_detector=kp_detector, bg_predictor=bg_predictor,
            optimizer=optimizer, optimizer_bg_predictor=optimizer_bg_predictor)
        print('load success:', start_epoch)
        start_epoch += 1
    else:
        start_epoch = 0

    freeze_kp_detector = train_params.get('freeze_kp_detector', False)
    freeze_bg_predictor = train_params.get('freeze_bg_predictor', False)
    if freeze_kp_detector:
        print('freeze kp detector')
        kp_detector.eval()
        for param in kp_detector.parameters():
            param.requires_grad = False
    if freeze_bg_predictor:
        print('freeze bg predictor')
        bg_predictor.eval()
        for param in bg_predictor.parameters():
            param.requires_grad = False

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True,
                            num_workers=train_params['dataloader_workers'], drop_last=True)

    last_epoch = (len(dataset) // train_params['batch_size']) * (start_epoch - 1)
    last_epoch = max(last_epoch, -1)

    scheduler_optimizer = OneCycleLR(optimizer, max_lr=train_params['lr_generator'],
                                     total_steps=(len(dataset) // train_params['batch_size']) * train_params[
                                         'num_epochs'],
                                     last_epoch=last_epoch
                                     )

    scheduler_bg_predictor = None
    if bg_predictor:
        scheduler_bg_predictor = OneCycleLR(optimizer_bg_predictor, max_lr=train_params['lr_generator'],
                                            total_steps=(len(dataset) // train_params['batch_size']) * train_params[
                                                'num_epochs'],
                                            last_epoch=last_epoch
                                            )
        bg_predictor, optimizer_bg_predictor = accelerator.prepare(bg_predictor, optimizer_bg_predictor)

    generator_full = GeneratorFullModel(kp_detector, bg_predictor, dense_motion_network, inpainting_network,
                                        train_params)

    bg_start = train_params['bg_start']

    inpainting_network, kp_detector, dense_motion_network, optimizer, scheduler_optimizer, dataloader, generator_full = accelerator.prepare(
        inpainting_network, kp_detector, dense_motion_network, optimizer, scheduler_optimizer, dataloader,
        generator_full)

    if train_params.get('visualize_model', False):
        # visualize graph
        sample = next(iter(dataloader))
        draw_graph(generator_full, input_data=[sample, 100], save_graph=True, directory=log_dir,
                   graph_name='generator_full')
        draw_graph(kp_detector, input_data=[sample['driving']], save_graph=True, directory=log_dir,
                   graph_name='kp_detector')
        kp_driving = kp_detector(sample['driving'])
        kp_source = kp_detector(sample['source'])
        bg_param = bg_predictor(sample['source'], sample['driving'])
        dense_motion_param = {'source_image': sample['source'], 'kp_driving': kp_driving, 'kp_source': kp_source,
                              'bg_param': bg_param,
                              'dropout_flag': False, 'dropout_p': 0.0}
        dense_motion = dense_motion_network(**dense_motion_param)
        draw_graph(dense_motion_network, input_data=dense_motion_param, save_graph=True, directory=log_dir,
                   graph_name='dense_motion_network')
        draw_graph(inpainting_network, input_data=[sample['source'], dense_motion], save_graph=True, directory=log_dir,
                   graph_name='inpainting_network')

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                checkpoint_freq=train_params['checkpoint_freq'],
                models=[inpainting_network, dense_motion_network, kp_detector]
                ) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in tqdm(dataloader):
                losses_generator, generated = generator_full(x, epoch)
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                accelerator.backward(loss)

                clip_grad_norm_(kp_detector.parameters(), max_norm=10, norm_type=math.inf)
                clip_grad_norm_(dense_motion_network.parameters(), max_norm=10, norm_type=math.inf)
                if bg_predictor and epoch >= bg_start and not freeze_bg_predictor:
                    clip_grad_norm_(bg_predictor.parameters(), max_norm=10, norm_type=math.inf)

                optimizer.step()

                if bg_predictor and epoch >= bg_start and not freeze_bg_predictor:
                    optimizer_bg_predictor.step()
                    optimizer_bg_predictor.zero_grad()
                    scheduler_bg_predictor.step()

                optimizer.zero_grad()
                scheduler_optimizer.step()

                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                lrs = {
                    'lr_generator': scheduler_optimizer.get_last_lr()[0],
                    'lr_bg_predictor': scheduler_bg_predictor.get_last_lr()[0] if bg_predictor else 0
                }
                logger.log_iter(losses=losses, others=lrs)



            model_save = {
                'inpainting_network': accelerator.unwrap_model(inpainting_network),
                'dense_motion_network': accelerator.unwrap_model(dense_motion_network),
                'kp_detector': accelerator.unwrap_model(kp_detector),
                'optimizer': optimizer,
                'bg_predictor': accelerator.unwrap_model(bg_predictor) if bg_predictor else None,
                'optimizer_bg_predictor': optimizer_bg_predictor,
            }

            logger.log_epoch(epoch, model_save, inp=x, out=generated)
