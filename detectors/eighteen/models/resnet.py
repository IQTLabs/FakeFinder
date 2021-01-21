# definition of resnet
import os
import torch
import torchvision.models.resnet as resnet


def init_res34_cls_model(load_path, cuda):
    model = resnet.resnet34(num_classes=2)
    load_state(load_path, model, cuda)
    if cuda:
        model.cuda()
    model.eval()
    return model


def load_state(path, model, cuda=True):
    def map_func(storage, location):
        return storage.cuda()

    if not cuda:
        map_func = torch.device('cpu')
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        have_load = set(pretrain(model, checkpoint['state_dict'], cuda))
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - have_load
        for k in missing_keys:
            print('caution: missing keys from checkpoint {}: {}'.format(path, k))
            pass
    else:
        print("=> no checkpoint found at '{}'".format(path))


def pretrain(model, state_dict, cuda):
    own_state = model.state_dict()
    have_load = []
    for name, param in state_dict.items():
        # remove "module." prefix
        name = name.replace(name.split('.')[0] + '.', '')
        if name in own_state:
            have_load.append(name)
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(name, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")
    return have_load
