import torch
from torchvision.models.video import mc3_18, r2plus1d_18

from .resnet_3d import *
from .inception_3d import *

__all__ = ['build_models']

PRETRAINED_MODELS_3D = [{'type': 'i3d',
                         'path': "./weights/j3d_e1_l0.1374.model"},
                        {'type': 'res34',
                         'path': "./weights/res34_1cy_minaug_nonorm_e4_l0.1794.model"},
                        {'type': 'mc3_112',
                         'path': "./weights/mc3_18_112_1cy_lilaug_nonorm_e9_l0.1905.model"},
                        {'type': 'mc3_224',
                         'path': "./weights/mc3_18_112t224_1cy_lilaug_nonorm_e7_l0.1901.model"},
                        {'type': 'r2p1_112',
                         'path': './weights/r2p1_18_8_112tr_112te_e12_l0.1741.model'},
                        {'type': 'i3d',
                         'path': "./weights/i3dcutmix_e11_l0.1612.model"},
                        {'type': 'r2p1_112',
                         'path': "./weights/r2plus1dcutmix_112_e10_l0.1608.model"}]


def build_models():
    models_3d = []
    for modeldict in PRETRAINED_MODELS_3D:
        if modeldict['type'] == 'i3d':
            model = InceptionI3d(157, in_channels=3, output_method='avg_pool')
            model.replace_logits(2)
            model = model.cuda()
            model.eval()
            model.load_state_dict(torch.load(modeldict['path']))
            models_3d.append(
                {'norm': 'i3d', 'model': HFlipWrapper(model=model)})

        elif modeldict['type'] == 'res18':
            model = resnet18(num_classes=2, shortcut_type='A',
                             sample_size=224, sample_duration=32)  # , last_fc=True)
            model.load_state_dict(torch.load(modeldict['path']))
            model = model.cuda()
            model.eval()
            models_3d.append(
                {'norm': 'nil', 'model': HFlipWrapper(model=model)})

        elif modeldict['type'] == 'res34':
            model = resnet34(num_classes=2, shortcut_type='A',
                             sample_size=224, sample_duration=32)  # , last_fc=True)
            model.load_state_dict(torch.load(modeldict['path']))
            model = model.cuda()
            model.eval()
            models_3d.append(
                {'norm': 'nil', 'model': HFlipWrapper(model=model)})

        elif modeldict['type'] == 'mc3_112':
            model = mc3_18(num_classes=2, pretrained=False)
            model.load_state_dict(torch.load(modeldict['path']))
            model = model.cuda()
            model.eval()
            models_3d.append(
                {'norm': '112_imagenet', 'model': HFlipWrapper(model=model)})

        elif modeldict['type'] == 'mc3_224':
            model = mc3_18(num_classes=2, pretrained=False)
            model.load_state_dict(torch.load(modeldict['path']))
            model = model.cuda()
            model.eval()
            models_3d.append(
                {'norm': '224_imagenet', 'model': HFlipWrapper(model=model)})

        elif modeldict['type'] == 'r2p1_112':
            model = r2plus1d_18(num_classes=2, pretrained=False)
            model.load_state_dict(torch.load(modeldict['path']))
            model = model.cuda()
            model.eval()
            models_3d.append(
                {'norm': '112_imagenet', 'model': HFlipWrapper(model=model)})

        else:
            raise ValueError(f"Unknown model type {modeldict['type']}")
    return models_3d
