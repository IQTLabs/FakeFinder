import sys
import os
import time

import cv2
import numpy as np
from collections import defaultdict

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T

sys.path.append('./external/Pytorch_Retinaface')
from .model_def import WSDAN, xception
from .face_utils import norm_crop, FaceDetector
from .external.Pytorch_Retinaface.data import cfg_re50

class video_reader:
    def __init__(self, face_detector, transform=None, frame_skip=9, face_limit=25, batch_size=25):

        self.transform = transform
        self.face_detector = face_detector

        self.batch_size = batch_size
        self.frame_skip = frame_skip
        self.face_limit = face_limit

    def process_one_frame(self, video_path):
        reader = cv2.VideoCapture(video_path)
        face_count = 0

        while True:
            for _ in range(self.frame_skip):
                reader.grab()

            success, img = reader.read()
            if not success:
                break

            boxes, landms = self.face_detector.detect(img)
            if boxes.shape[0] == 0:
                continue

            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            order = areas.argmax()

            boxes = boxes[order]
            landms = landms[order]

            # Crop faces
            landmarks = landms.numpy().reshape(5, 2).astype(np.int)
            img = norm_crop(img, landmarks, image_size=320)
            aligned = Image.fromarray(img[:, :, ::-1])

            if self.transform:
                aligned = self.transform(aligned)

            yield aligned

            # Early stop
            face_count += 1
            if face_count == self.face_limit:
                break

        reader.release()

    def process_video(self, video_path):
        batch_buf = []
        for face in self.process_one_frame(video_path):
            batch_buf.append(face)

            if len(batch_buf) == self.batch_size:
                return torch.stack(batch_buf)
        if len(batch_buf) > 0:
            return torch.stack(batch_buf)


class Ensemble:
    def __init__(self):
        face_detector = FaceDetector()
        face_detector.load_checkpoint(
            "./weights/wm/RetinaFace-Resnet50-fixed.pth")
        face_detector = face_detector
        self.reader = video_reader(face_detector, T.ToTensor())

        model1 = xception(num_classes=2, pretrained=False)
        ckpt = torch.load("./weights/wm/xception-hg-2.pth")
        model1.load_state_dict(ckpt["state_dict"])
        self.model1 = model1.cuda()
        self.model1.eval()

        model2 = WSDAN(num_classes=2, M=8, net="xception",
                       pretrained=False).cuda()
        ckpt = torch.load("./weights/wm/ckpt_x.pth")
        model2.load_state_dict(ckpt["state_dict"])
        self.model2 = model2
        self.model2.eval()

        model3 = WSDAN(num_classes=2, M=8, net="efficientnet",
                       pretrained=False).cuda()
        ckpt = torch.load("./weights/wm/ckpt_e.pth")
        model3.load_state_dict(ckpt["state_dict"])
        self.model3 = model3
        self.model3.eval()

        self.zhq_nm_avg = torch.Tensor(
            [.4479, .3744, .3473]).view(1, 3, 1, 1).cuda()
        self.zhq_nm_std = torch.Tensor(
            [.2537, .2502, .2424]).view(1, 3, 1, 1).cuda()

    def inference(self, video_path):
        with torch.no_grad():
            batch = self.reader.process_video(video_path).cuda()
            i1 = F.interpolate(batch, size=299, mode="bilinear").cuda()
            i1.sub_(0.5).mul_(2.0)
            o1 = self.model1(i1).softmax(-1)[:, 1].cpu().numpy()

            i2 = (batch.cuda() - self.zhq_nm_avg) / self.zhq_nm_std
            o2, _, _ = self.model2(i2)
            o2 = o2.softmax(-1)[:, 1].cpu().numpy()

            i3 = F.interpolate(i2, size=300, mode="bilinear")
            o3, _, _ = self.model3(i3)
            o3 = o3.softmax(-1)[:, 1].cpu().numpy()

        out = 0.2 * o1 + 0.7 * o2 + 0.1 * o3
        return np.mean(out)

    def __del__(self):
        del self.reader
        del self.model1
        del self.model2
        del self.model3
        del self.zhq_nm_avg
        del self.zhq_nm_std
 
        torch.cuda.empty_cache()
        gc.collect()
