from torch.utils.data import DataLoader

import numpy as np
import re

from functools import partial

from . import models as factory_models
from .data import datasets
from .data import transforms
from .data import augmix

__all__ = ['build_model', 'build_dataloader']


def build(lib, name, params):
    return getattr(lib, name)(**params) if params is not None else getattr(lib, name)()


def build_model(name, params):
    return build(factory_models, name, params)


def build_dataloader(cfg, data_info, mode):

    # Build padder
    if cfg['transform']['pad_ratio']:
        pad = partial(transforms.pad_to_ratio,
                      ratio=cfg['transform']['pad_ratio'])
    else:
        pad = None

    # Build resizer
    if 'resize_to' not in cfg['transform'].keys() or type(cfg['transform']['resize_to']) != type(None):
        resize = transforms.resize(
            x=cfg['transform']['resize_to'][0], y=cfg['transform']['resize_to'][1])
    else:
        resize = None

    # Build cropper
    if 'crop_size' not in cfg['transform'].keys() or type(cfg['transform']['crop_size']) == type(None):
        crop = None
    else:
        crop = transforms.crop(x=cfg['transform']['crop_size'][0], y=cfg['transform']['crop_size'][1],
                               test_mode=mode != 'train')

    # Build augmenter
    if mode == 'train':
        if 'augmix' in cfg['transform'].keys():
            params = cfg['transform']['augmix']
            assert 'aug_list' in params.keys(
            ), 'When using AugMix, `aug_list` must be specified in `params`'
            params['aug_list'] = getattr(augmix, params['aug_list'])()
            data_aug = partial(augmix.augment_and_mix, **params)
        elif 'augment' in cfg['transform'].keys() and type(cfg['transform']['augment']) != type(None):
            data_aug = getattr(transforms, cfg['transform']['augment'])(
                p=cfg['transform']['probability'])
        else:
            data_aug = None
    else:
        data_aug = None

    # Build preprocessor
    if type(cfg['transform']['preprocess']) != type(None):
        preprocessor = transforms.Preprocessor(
            image_range=cfg['transform']['preprocess']['image_range'],
            input_range=cfg['transform']['preprocess']['input_range'],
            mean=cfg['transform']['preprocess']['mean'],
            sdev=cfg['transform']['preprocess']['sdev'])
    else:
        preprocessor = None

    # TODO: need some way of handling TTA
    # if mode == 'test' and cfg['test']['tta']:
    #   ...

    dset_params = cfg['dataset']['params'] if cfg['dataset']['params'] is not None else {}

    dset = getattr(datasets, cfg['dataset']['name'])(
        **data_info,
        pad=pad,
        resize=resize,
        crop=crop,
        transform=data_aug,
        preprocessor=preprocessor,
        test_mode=mode != 'train',
        **dset_params)

    if mode == 'train':
        sampler = None
        if 'sampler' in cfg['dataset'].keys():
            if type(cfg['dataset']['sampler']['name']) != type(None):
                sampler = getattr(datasets, cfg['dataset']['sampler']['name'])
                sampler = sampler(dataset=dset)
        dgen_params = {
            'batch_size': cfg['train']['batch_size'],
            'num_workers': cfg['transform']['num_workers'],
            'shuffle': True if type(sampler) == type(None) else False,
            'drop_last': True,
        }
        if type(sampler) != type(None):
            dgen_params['sampler'] = sampler
    else:
        dgen_params = {
            'batch_size': cfg[mode.replace('valid', 'evaluation')]['batch_size'],
            'num_workers': cfg['transform']['num_workers'],
            'shuffle': False,
            'drop_last': False
        }

    loader = DataLoader(dset, **dgen_params)

    return loader
