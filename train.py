import torchinfo
from tqdm import trange
import torch
from torch.utils.data import DataLoader

from gan import MultiScaleDiscriminator, discriminator_adversarial_loss, generator_adversarial_loss
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


def train(config, inpainting_network, kp_detector, bg_predictor, dense_motion_network, checkpoint,
          log_dir, dataset,
          optimizer_class=torch.optim.Adam,
          kp_detector_checkpoint=None,
          bg_predictor_checkpoint=None,
          ):
    train_params = config['train_params']
    optimizer_params = config['train_params'].get('optimizer_params', {})

    optimizer = optimizer_class(
        [{'params': list(inpainting_network.parameters()) +
                    list(dense_motion_network.parameters()) +
                    list(kp_detector.parameters()),
          'initial_lr': train_params['lr_generator']}],
        lr=train_params['lr_generator'], **optimizer_params)

    discriminator = MultiScaleDiscriminator(scales=[1], d=64)
    optimizer_discriminator = optimizer_class(
        [{'params': list(discriminator.parameters()), 'initial_lr': train_params['lr_discriminator']}],
            lr=train_params['lr_discriminator'], **optimizer_params)



    torchinfo.summary(discriminator, input_size=(1, 3, 256, 256))

    optimizer_bg_predictor = None
    if bg_predictor:
        optimizer_bg_predictor = optimizer_class(
            [{'params': bg_predictor.parameters(), 'initial_lr': train_params['lr_generator']}],
            lr=train_params['lr_generator'], betas=(0.5, 0.999), weight_decay=1e-4)

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(
            checkpoint, inpainting_network=inpainting_network, dense_motion_network=dense_motion_network,
            kp_detector=kp_detector, bg_predictor=bg_predictor,
            optimizer=optimizer, optimizer_bg_predictor=optimizer_bg_predictor,
            discriminator=discriminator, optimizer_discriminator=optimizer_discriminator)
        print('load success:', start_epoch)
        start_epoch += 1
    else:
        start_epoch = 0

    if kp_detector_checkpoint is not None:
        kp_params = torch.load(kp_detector_checkpoint)
        kp_detector.load_state_dict(kp_params['kp_detector'])
        print('load kp detector success')
    if bg_predictor_checkpoint is not None:
        bg_params = torch.load(bg_predictor_checkpoint)
        bg_predictor.load_state_dict(bg_params['bg_predictor'])
        print('load bg predictor success')

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
    discriminator_scheduler = OneCycleLR(optimizer_discriminator, max_lr=train_params['lr_discriminator'],
                                            total_steps=(len(dataset) // train_params['batch_size']) * train_params[
                                                'num_epochs'],
                                            last_epoch=last_epoch
                                            )

    discriminator, optimizer_discriminator, discriminator_scheduler = accelerator.prepare(discriminator, optimizer_discriminator, discriminator_scheduler)

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
    model_list = [inpainting_network, dense_motion_network, discriminator]
    if bg_predictor:
        model_list.append(bg_predictor)
    if not freeze_kp_detector:
        model_list.append(kp_detector)
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                checkpoint_freq=train_params['checkpoint_freq'],
                models=model_list,
                ) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            i = 0
            for x in tqdm(dataloader):
                with (accelerator.accumulate(generator_full), accelerator.accumulate(discriminator),
                      accelerator.accumulate(inpainting_network), accelerator.accumulate(dense_motion_network),
                      accelerator.accumulate(kp_detector), accelerator.accumulate(bg_predictor)):
                    losses_generator, generated = generator_full(x, epoch)
                    disc_loss = torch.zeros(1, device=x['driving'].device)
                    gen_loss = torch.zeros(1, device=x['driving'].device)

                    if i % 2 == 0:
                        disc_pred_fake = discriminator(generated['prediction'])
                        disc_pred_real = discriminator(x['driving'])
                        for j in range(len(disc_pred_real)):  # number of scales
                            disc_loss += discriminator_adversarial_loss(disc_pred_real[j], disc_pred_fake[j])
                    else:
                        features_fake, fake_preds = discriminator.forward_with_features(generated['prediction'])
                        features_real, _ = discriminator.forward_with_features(x['driving'])
                        for k in range(len(fake_preds)):
                            gen_loss += generator_adversarial_loss(fake_preds[k])

                    losses_generator['gen'] = gen_loss

                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)



                    if i % 2 == 0:
                        accelerator.backward(disc_loss, retain_graph=True)

                        clip_grad_norm_(discriminator.parameters(), max_norm=10, norm_type=math.inf)
                        optimizer_discriminator.step()
                        optimizer_discriminator.zero_grad()

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

                    scheduler_optimizer.step()
                    optimizer.zero_grad()

                    discriminator_scheduler.step()

                    losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                    lrs = {
                        'lr_generator': scheduler_optimizer.get_last_lr()[0],
                        'lr_bg_predictor': scheduler_bg_predictor.get_last_lr()[0] if bg_predictor else 0,
                        'lr_discriminator': discriminator_scheduler.get_last_lr()[0]
                    }
                    losses['disc'] = disc_loss.mean().detach().data.cpu().numpy()
                    logger.log_iter(losses=losses, others=lrs)

                    i += 1

            model_save = {
                'inpainting_network': accelerator.unwrap_model(inpainting_network),
                'dense_motion_network': accelerator.unwrap_model(dense_motion_network),
                'kp_detector': accelerator.unwrap_model(kp_detector),
                'optimizer': optimizer,
                'bg_predictor': accelerator.unwrap_model(bg_predictor) if bg_predictor else None,
                'optimizer_bg_predictor': optimizer_bg_predictor,
                'discriminator': accelerator.unwrap_model(discriminator),
                'optimizer_discriminator': optimizer_discriminator,
            }

            logger.log_epoch(epoch, model_save, inp=x, out=generated)
