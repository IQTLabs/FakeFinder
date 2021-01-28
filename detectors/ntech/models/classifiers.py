import numpy as np

import torch
import torch.nn as nn

from albumentations import Compose, SmallestMaxSize, CenterCrop, Normalize, PadIfNeeded
from albumentations.pytorch import ToTensor
from efficientnet_pytorch.model import EfficientNet, MBConvBlock

VIDEO_MODEL_BBOX_MULT = 1.5
VIDEO_MODEL_MIN_SIZE = 224
VIDEO_MODEL_CROP_HEIGHT = 224
VIDEO_MODEL_CROP_WIDTH = 192
VIDEO_FACE_MODEL_TRACK_STEP = 2
VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH = 7
VIDEO_SEQUENCE_MODEL_TRACK_STEP = 14

__all__ = ['TrackSequencesClassifier', 'TrackFacesClassifier']


class SeqExpandConv(nn.Module):
    def __init__(self, in_channels, out_channels, seq_length):
        super(SeqExpandConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(
            3, 1, 1), padding=(1, 0, 0), bias=False)
        self.seq_length = seq_length

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        x = x.view(batch_size // self.seq_length,
                   self.seq_length, in_channels, height, width)
        x = self.conv(x.transpose(1, 2).contiguous()
                      ).transpose(2, 1).contiguous()
        x = x.flatten(0, 1)
        return x


class TrackSequencesClassifier(object):
    def __init__(self, weights_path):
        model = EfficientNet.from_name(
            'efficientnet-b7', override_params={'num_classes': 1})

        for module in model.modules():
            if isinstance(module, MBConvBlock):
                if module._block_args.expand_ratio != 1:
                    expand_conv = module._expand_conv
                    seq_expand_conv = SeqExpandConv(expand_conv.in_channels, expand_conv.out_channels,
                                                    VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH)
                    module._expand_conv = seq_expand_conv
        self.model = model.cuda().eval()

        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[
                              0.229, 0.224, 0.225])
        self.transform = Compose(
            [SmallestMaxSize(VIDEO_MODEL_MIN_SIZE), CenterCrop(VIDEO_MODEL_CROP_HEIGHT, VIDEO_MODEL_CROP_WIDTH),
             normalize, ToTensor()])

        state = torch.load(
            weights_path, map_location=lambda storage, loc: storage)
        state = {key: value.float() for key, value in state.items()}
        self.model.load_state_dict(state)

    def classify(self, track_sequences):
        track_sequences = [torch.stack([self.transform(image=face)['image'] for face in sequence]) for sequence in
                           track_sequences]
        track_sequences = torch.cat(track_sequences).cuda()
        with torch.no_grad():
            track_probs = torch.sigmoid(self.model(
                track_sequences)).flatten().cpu().numpy()

        return track_probs


class TrackFacesClassifier(object):
    def __init__(self, first_weights_path, second_weights_path):
        first_model = EfficientNet.from_name(
            'efficientnet-b7', override_params={'num_classes': 1})
        self.first_model = first_model.cuda().eval()
        second_model = EfficientNet.from_name(
            'efficientnet-b7', override_params={'num_classes': 1})
        self.second_model = second_model.cuda().eval()

        first_normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.first_transform = Compose(
            [SmallestMaxSize(VIDEO_MODEL_CROP_WIDTH), PadIfNeeded(VIDEO_MODEL_CROP_HEIGHT, VIDEO_MODEL_CROP_WIDTH),
             CenterCrop(VIDEO_MODEL_CROP_HEIGHT, VIDEO_MODEL_CROP_WIDTH), first_normalize, ToTensor()])

        second_normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.second_transform = Compose(
            [SmallestMaxSize(VIDEO_MODEL_MIN_SIZE), CenterCrop(VIDEO_MODEL_CROP_HEIGHT, VIDEO_MODEL_CROP_WIDTH),
             second_normalize, ToTensor()])

        first_state = torch.load(
            first_weights_path, map_location=lambda storage, loc: storage)
        first_state = {key: value.float()
                       for key, value in first_state.items()}
        self.first_model.load_state_dict(first_state)

        second_state = torch.load(
            second_weights_path, map_location=lambda storage, loc: storage)
        second_state = {key: value.float()
                        for key, value in second_state.items()}
        self.second_model.load_state_dict(second_state)

    def classify(self, track_faces):
        first_track_faces = []
        second_track_faces = []
        for i, face in enumerate(track_faces):
            if i % 4 < 2:
                first_track_faces.append(
                    self.first_transform(image=face)['image'])
            else:
                second_track_faces.append(
                    self.second_transform(image=face)['image'])
        first_track_faces = torch.stack(first_track_faces).cuda()
        second_track_faces = torch.stack(second_track_faces).cuda()
        with torch.no_grad():
            first_track_probs = torch.sigmoid(self.first_model(
                first_track_faces)).flatten().cpu().numpy()
            second_track_probs = torch.sigmoid(self.second_model(
                second_track_faces)).flatten().cpu().numpy()
            track_probs = np.concatenate(
                (first_track_probs, second_track_probs))

        return track_probs
