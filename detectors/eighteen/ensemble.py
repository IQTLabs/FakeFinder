import cv2
import numpy as np
import copy
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision

from face_detect_lib.models.retinaface import RetinaFace
from face_detect_lib.layers.functions.prior_box import PriorBox
from face_detect_lib.utils.box_utils import decode_batch, decode_landm_batch, decode, decode_landm

from utils import *
from models import *

__all__ = ['Ensemble', 'pipeline_cfg']

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}


class Config:
    def __init__(self):
        self.cuda = True
        self.face_pretrained_path = './weights/mobilenetV1X0.25_pretrain.tar'
        self.face_model_path = './weights/mobilenet0.25_Final.pth'
        self.model_name = 'mobile0.25'
        self.origin_size = False
        self.confidence_threshold = 0.02
        self.top_k = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = 750
        self.target_size = 400
        self.max_size = 2150
        self.model_cfg = cfg_mnet
        self.vis_thres = 0.8


pipeline_cfg = Config()


def detect_face(img_list, detect_record):

    im_shape = img_list[0].shape
    detect_key = str(im_shape[0]) + '*' + str(im_shape[1])
    if detect_key not in detect_record:
        print(detect_key + ' not in dict')
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(pipeline_cfg.target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > pipeline_cfg.max_size:
            resize = float(pipeline_cfg.max_size) / float(im_size_max)
        im_height, im_width = int(
            im_shape[0] * resize), int(im_shape[1] * resize)
        detect_record[detect_key] = {
            'resize': resize, 'resized_h': im_height, 'resized_w': im_width}
        priorbox = PriorBox(pipeline_cfg.model_cfg,
                            image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(pipeline_cfg.device)
        detect_record[detect_key]['priors'] = priors

    # detect face
    detect_info = detect_record[detect_key]
    resize = detect_info['resize']
    resize_img_list = []
    result_dets_list = []
    batch_size = 8
    detect_nms_time = 0
    for img_idx, img in enumerate(img_list):
        if detect_info['resize'] != 1:
            img = cv2.resize(img, None, None, fx=detect_info['resize'], fy=detect_info['resize'],
                             interpolation=cv2.INTER_LINEAR)
            img = np.float32(img)
        else:
            img = np.float32(img)

        resize_img_list.append(img)
        img_idx += 1
        if img_idx % batch_size == 0 or img_idx == len(img_list):
            im_height, im_width, _ = resize_img_list[0].shape
            scale = torch.Tensor([resize_img_list[0].shape[1], resize_img_list[0].shape[0], resize_img_list[0].shape[1],
                                  resize_img_list[0].shape[0]])
            resize_img_list = np.stack(resize_img_list, axis=0)  # [n,h,w,c]
            resize_img_list -= (104, 117, 123)
            resize_img_list = resize_img_list.transpose(0, 3, 1, 2)
            resize_img_list = torch.from_numpy(resize_img_list)
            resize_img_list = resize_img_list.to(pipeline_cfg.device)
            scale = scale.to(pipeline_cfg.device)
            loc, conf, landms = pipeline_cfg.net(resize_img_list)
            priors = detect_info['priors']
            prior_data = priors.data
            boxes = decode_batch(loc.data, prior_data,
                                 pipeline_cfg.model_cfg['variance'])
            boxes = boxes * scale / resize  # [batchsize, proposals, 4]
            scores = conf[:, :, 1]  # [batchsize, proposals]

            detect_nms_begin = 0
            for per_idx in range(boxes.shape[0]):
                box, score = boxes[per_idx, :, :], scores[per_idx, :]
                inds = torch.nonzero(
                    score > pipeline_cfg.confidence_threshold)[:, 0]
                box, score = box[inds, :], score[inds]
                dets = torch.cat((box, score[:, None]), dim=1)
                keep = torchvision.ops.nms(
                    box, score, pipeline_cfg.nms_threshold)
                dets = dets[keep, :]
                dets = dets.data.cpu().numpy()
                result_dets_list.append(dets)
            resize_img_list = []
            detect_nms_end = 0
            detect_nms_time += detect_nms_end - detect_nms_begin
    return result_dets_list


def init_face_detecor():
    torch.set_grad_enabled(False)
    pipeline_cfg.net = RetinaFace(
        cfg=pipeline_cfg.model_cfg, model_path=pipeline_cfg.face_pretrained_path, phase='test')
    pipeline_cfg.net = load_model(
        pipeline_cfg.net, pipeline_cfg.face_model_path, pipeline_cfg.cuda)
    pipeline_cfg.net.eval()
    cudnn.benchmark = True
    pipeline_cfg.device = torch.device("cuda" if pipeline_cfg.cuda else "cpu")
    pipeline_cfg.net = pipeline_cfg.net.to(pipeline_cfg.device)
    return pipeline_cfg.net


def get_image_score(face_scale, cls_model, softmax_func, aligned_faces, isRGB, mean, std, isScale, isFlip=False):
    try:
        # aligned_faces #[faces,frames,H,W,C] BGR
        img_aligned_faces = aligned_faces.clone().detach()
        img_aligned_faces = img_aligned_faces.permute(
            [0, 1, 4, 2, 3])  # [faces,frames,c,h,w] BGR
        if isRGB:
            img_aligned_faces = img_aligned_faces[:, :, [2, 1, 0], :, :]
        img_frames = 35
        interval = max(1, math.ceil(img_aligned_faces.shape[1] / img_frames))
        img_aligned_faces = img_aligned_faces[:, 0::interval, :, :, :]
        img_frames = (img_aligned_faces.shape[1] // 5) * 5
        img_aligned_faces = img_aligned_faces[:, :img_frames, :, :, :]

        all_score, score = [], 0
        for face_idx in range(img_aligned_faces.shape[0]):
            one_face_aligned = img_aligned_faces[face_idx, :, :, :, :]
            one_face_aligned_mean = (
                one_face_aligned - mean) / std  # [frames,c,h,w]
            if isFlip:
                one_face_aligned_mean_flip = torch.flip(
                    one_face_aligned_mean, dims=[3])
                one_face_aligned_input = torch.cat(
                    (one_face_aligned_mean, one_face_aligned_mean_flip), dim=0)
                output = cls_model(one_face_aligned_input)
                output = (output[:img_frames, :] + output[img_frames:, :]) / 2
            else:
                output = cls_model(one_face_aligned_mean)
            output = output.view(-1, 5, 2)
            output = output.mean(1)
            output = softmax_func(output)
            output = output[:, 1].cpu().numpy()  # [6,1]
            if output[output > 0.85].shape[0] / output.shape[0] > 0.7:
                score = output[output > 0.85].mean()
            elif output[output < 0.15].shape[0] / output.shape[0] > 0.7:
                score = output[output < 0.15].mean()
            else:
                score = output.mean()
            all_score.append(score)
        all_score = np.array(all_score)
        score_max, score_min, score_avg = np.max(
            all_score), np.min(all_score), np.mean(all_score)
        if score_max > 0.9:
            score = score_max
        elif len(np.where(all_score > 0.6)[0]) == all_score.shape[0]:
            score = score_max
        elif len(np.where(all_score < 0.4)[0]) == all_score.shape[0]:
            score = score_min
        else:
            score = score_avg
        if isScale:
            if score >= 0.98 or score <= 0.02:
                score = (score - 0.5) * 0.96 + 0.5
    except Exception as e:
        print(e)
        score = -1
    return score


def get_sf_score(face_scale, cls_model, softmax_func, aligned_faces, isRGB, mean, std):
    try:
        # aligned_faces [faces,frames,H,W,C]  BGR
        sf_aligned_faces = aligned_faces.clone().detach()
        sf_aligned_faces = sf_aligned_faces.permute(
            [0, 4, 1, 2, 3])  # [faces,c,frames,h,w]
        if isRGB:
            sf_aligned_faces = sf_aligned_faces[:, [2, 1, 0], :, :, :]
        sf_aligned_faces = (sf_aligned_faces - mean) / std
        sf_output = cls_model(sf_aligned_faces)
        sf_output = softmax_func(sf_output)
        sf_output = sf_output[:, 1].cpu().numpy()
        sf_max, sf_min, sf_avg = np.max(sf_output), np.min(
            sf_output), np.mean(sf_output)
        if sf_max > 0.9:
            sf_score = sf_max
        elif len(np.where(sf_output > 0.6)[0]) == sf_output.shape[0]:
            sf_score = sf_max
        elif len(np.where(sf_output < 0.4)[0]) == sf_output.shape[0]:
            sf_score = sf_min
        else:
            sf_score = sf_avg
    except Exception as e:
        print(e)
        sf_score = -1
    return sf_score


def get_final_score(score_list, weight_list):
    final_score = 0
    assert len(score_list) == len(weight_list)
    new_score_list, new_weight_list = [], []
    for idx, score in enumerate(score_list):
        if score != -1:
            new_score_list.append(score)
            new_weight_list.append(weight_list[idx])
    new_scores, new_weights = np.array(
        new_score_list), np.array(new_weight_list)
    if len(new_weights) == 0:
        return -1
    print('new_scores:', new_scores, 'new_weights',
          new_weights / np.sum(new_weights))
    final_score = np.sum(new_scores * (new_weights / np.sum(new_weights)))
    return final_score


def get_final_score_policy(score_list, weight_list, img_start_idx, sf_weight):
    assert len(score_list) == len(weight_list)
    sf_score_list, sf_weight_list = score_list[:
                                               img_start_idx], weight_list[:img_start_idx]
    img_score_list, img_weight_list = score_list[img_start_idx:], weight_list[img_start_idx:]
    new_sf_score_list, new_sf_weight_list, new_img_score_list, new_img_weight_list = [], [], [], []
    for idx, score in enumerate(sf_score_list):
        if score != -1:
            new_sf_score_list.append(score)
            new_sf_weight_list.append(sf_weight_list[idx])

    for idx, score in enumerate(img_score_list):
        if score != -1:
            new_img_score_list.append(score)
            new_img_weight_list.append(img_weight_list[idx])
    new_sf_scores, new_sf_weights = np.array(
        new_sf_score_list), np.array(new_sf_weight_list)
    new_img_scores, new_img_weights = np.array(
        new_img_score_list), np.array(new_img_weight_list)

    sf_success, img_success = True, True
    # sf
    if new_sf_scores.shape[0] != 0:
        if len(np.where(new_sf_scores > 0.8)[0]) / new_sf_scores.shape[0] > 0.7:
            new_sf_y_scores, new_sf_y_weights = new_sf_scores[new_sf_scores >
                                                              0.8], new_sf_weights[new_sf_scores > 0.8]
            sf_score = np.sum(new_sf_y_scores *
                              (new_sf_y_weights / np.sum(new_sf_y_weights)))
        elif len(np.where(new_sf_scores < 0.2)[0]) / new_sf_scores.shape[0] > 0.7:
            new_sf_y_scores, new_sf_y_weights = new_sf_scores[new_sf_scores <
                                                              0.2], new_sf_weights[new_sf_scores < 0.2]
            sf_score = np.sum(new_sf_y_scores *
                              (new_sf_y_weights / np.sum(new_sf_y_weights)))
        else:
            sf_score = np.sum(
                new_sf_scores * (new_sf_weights / np.sum(new_sf_weights)))
    else:
        sf_success = False

    # img
    if new_img_scores.shape[0] != 0:
        if len(np.where(new_img_scores > 0.8)[0]) / new_img_scores.shape[0] > 0.7:
            new_img_y_scores, new_img_y_weights = new_img_scores[new_img_scores > 0.8], new_img_weights[
                new_img_scores > 0.8]
            img_score = np.sum(new_img_y_scores *
                               (new_img_y_weights / np.sum(new_img_y_weights)))
        elif len(np.where(new_img_scores < 0.2)[0]) / new_img_scores.shape[0] > 0.7:
            new_img_y_scores, new_img_y_weights = new_img_scores[new_img_scores < 0.2], new_img_weights[
                new_img_scores < 0.2]
            img_score = np.sum(new_img_y_scores *
                               (new_img_y_weights / np.sum(new_img_y_weights)))
        else:
            img_score = np.sum(
                new_img_scores * (new_img_weights / np.sum(new_img_weights)))
    else:
        img_success = False

    if sf_success and img_success:
        final_score = sf_score * sf_weight + (1 - sf_weight) * img_score
    elif sf_success and not img_success:
        final_score = sf_score
    elif img_success and not sf_success:
        final_score = img_score
    else:
        final_score = -1
    return final_score


def predict_batch(img_list, sf_model1, sf_model2, sf_model3, xcp_model, b3_model, res34_model, b1_model, b1long_model,
                  b1short_model, b0_model, sf_model4, softmax_func, detect_record):
    # face det
    aligned_faceses, noface_flag = detect_video_face(img_list, detect_record)
    if noface_flag == -1:
        return -1
    sf1_score, sf2_score, sf3_score, sf4_score, xcp_score, b3_score, res34_score, b1_score, b1long_score, b1short_score, b0_score = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    scale_num = len(aligned_faceses)
    # slowfast model infer
    sf_mean = torch.from_numpy(
        np.array([110.63666788 / 255, 103.16065604 / 255, 96.29023126 / 255], dtype=np.float32)).reshape(
        [1, -1, 1, 1, 1]).cuda()
    sf_std = torch.from_numpy(
        np.array([38.7568578 / 255, 37.88248729 / 255, 40.02898126 / 255], dtype=np.float32)).reshape(
        [1, -1, 1, 1, 1]).cuda()
    xcp_mean = torch.from_numpy(
        np.array([0.5, 0.5, 0.5], dtype=np.float32)).reshape([1, -1, 1, 1]).cuda()
    xcp_std = torch.from_numpy(
        np.array([0.5, 0.5, 0.5], dtype=np.float32)).reshape([1, -1, 1, 1]).cuda()
    b3_mean = torch.from_numpy(np.array(
        [0.485, 0.456, 0.406], dtype=np.float32)).reshape([1, -1, 1, 1]).cuda()
    b3_std = torch.from_numpy(np.array(
        [0.229, 0.224, 0.225], dtype=np.float32)).reshape([1, -1, 1, 1]).cuda()

    for aligned_faces in aligned_faceses:
        print('aligned_faces shape:', aligned_faces.shape)
        aligned_faces = np.float32(aligned_faces)
        face_scale, face_num, align_frames = aligned_faces.shape[
            3], aligned_faces.shape[0], aligned_faces.shape[1]
        # init scale tensor
        aligned_faces = torch.from_numpy(aligned_faces)
        if pipeline_cfg.cuda:
            aligned_faces = aligned_faces.cuda()  # [faces,frames,H,W,C]
        aligned_faces /= 255

        # xcp inference
        if face_scale == 299:
            xcp_score = get_image_score(face_scale, xcp_model, softmax_func, aligned_faces, False, xcp_mean, xcp_std,
                                        False, True)
            b3_score = get_image_score(face_scale, b3_model, softmax_func, aligned_faces, True, b3_mean, b3_std, True,
                                       False)
            b1_score = get_image_score(face_scale, b1_model, softmax_func, aligned_faces, True, b3_mean, b3_std, False,
                                       True)
            b1long_score = get_image_score(face_scale, b1long_model, softmax_func, aligned_faces, True, b3_mean, b3_std,
                                           False, False)
            b1short_score = get_image_score(face_scale, b1short_model, softmax_func, aligned_faces, True, b3_mean,
                                            b3_std, False, False)
            b0_score = get_image_score(face_scale, b0_model, softmax_func, aligned_faces, True, b3_mean, b3_std, False,
                                       True)

        if face_scale == 256:
            res34_score = get_image_score(face_scale, res34_model, softmax_func, aligned_faces, True, b3_mean, b3_std,
                                          True, True)
            sf1_score = get_sf_score(
                face_scale, sf_model1, softmax_func, aligned_faces, True, sf_mean, sf_std)
            sf2_score = get_sf_score(
                face_scale, sf_model2, softmax_func, aligned_faces, True, sf_mean, sf_std)
            sf3_score = get_sf_score(
                face_scale, sf_model3, softmax_func, aligned_faces, True, sf_mean, sf_std)
            sf4_score = get_sf_score(
                face_scale, sf_model4, softmax_func, aligned_faces, True, sf_mean, sf_std)

    score_list = [sf1_score, sf2_score, sf3_score, sf4_score, xcp_score, b3_score, res34_score, b1_score, b1long_score,
                  b1short_score, b0_score]
    print(score_list)
    sf_weight_np, img_weight_np = np.array(
        [10, 8, 4, 8]), np.array([10, 6, 4, 10, 8, 8, 7])
    sf_weight_np = sf_weight_np / np.sum(sf_weight_np) * 0.4
    img_weight_np = img_weight_np / np.sum(img_weight_np) * 0.6
    weight_np = np.concatenate((sf_weight_np, img_weight_np))
    weight_list = list(weight_np)
    print(weight_list)
    final_score = get_final_score_policy(
        score_list, weight_list, len(sf_weight_np), 0.4)
    return final_score


def detect_video_face(img_list, detect_record):
    num_frames = len(img_list)
    num_faces = 0
    face_count = {}
    img_h, img_w = img_list[0].shape[0], img_list[0].shape[1]
    face_list = []

    dets_list = detect_face(img_list, detect_record)
#     detect_face_time = detect_tmp_end - detect_tmp_begin
#     global DETECT_FACE_TIME
#     DETECT_FACE_TIME += detect_face_time
#     print('detect face time:', detect_face_time)

    for idx, img_raw in enumerate(img_list):
        # preserve only faces with confidence above threshold
        dets = dets_list[idx][np.where(dets_list[idx][:, 4] >= pipeline_cfg.vis_thres)][:, :4].astype(
            np.int64)  # [m,15]
        face_list.append(dets)
        if len(dets) not in face_count:
            face_count[len(dets)] = 0
        face_count[len(dets)] += 1

    # vote for the number of faces that most frames agree on
    max_count = 0
    for num in face_count:
        if face_count[num] > max_count:
            num_faces = num
            max_count = face_count[num]
    if num_faces <= 0:
        return None, -1

    active_faces = None
    face_tubes = []
    for frame_idx in range(num_frames):
        cur_faces = face_list[frame_idx]  #
        if len(cur_faces) <= 0:
            continue

        if active_faces is not None:
            ious = vanilla_bbox_iou_overlaps(cur_faces, active_faces)
            max_iou, max_idx = np.max(ious, axis=1), np.argmax(ious, axis=1)
            mark = [False for _ in range(len(active_faces))]
        else:
            max_iou, max_idx = None, None

        for face_idx in range(len(cur_faces)):
            # IoU threshold 0.5 for determining whether is the same person
            if max_iou is None or max_iou[face_idx] < 0.5:
                face = copy.deepcopy(cur_faces[face_idx])
                if active_faces is None:
                    active_faces = face[np.newaxis, :]
                else:
                    active_faces = np.concatenate(
                        (active_faces, face[np.newaxis, :]), axis=0)
                face_tubes.append([[frame_idx, face_idx]])
            else:
                correspond_idx = max_idx[face_idx]
                # Each face tube can only add at most one face from a frame
                if mark[correspond_idx]:
                    continue
                mark[correspond_idx] = True
                active_faces[correspond_idx] = cur_faces[face_idx]
                face_tubes[correspond_idx].append([frame_idx, face_idx])
    # Choose num_faces longest face_tubes as chosen faces
    face_tubes.sort(key=lambda tube: len(tube), reverse=True)
    if len(face_tubes) < num_faces:
        num_faces = len(face_tubes)
    num_faces = min(num_faces, 2)
    face_tubes = face_tubes[:num_faces]

    aligned_faces_img_256, aligned_faces_img_299, aligned_faces_img_320 = [], [], []
    for face_idx in range(num_faces):
        cur_face_list, source_frame_list = [], []
        # record max crop_bbox size
        tube_idx, max_size = 0, 0
        for frame_idx in range(num_frames):
            cur_face = face_tubes[face_idx][tube_idx]
            next_face = None if tube_idx == len(
                face_tubes[face_idx]) - 1 else face_tubes[face_idx][tube_idx + 1]
            # find nearest frame inside face tube
            if next_face is not None and abs(cur_face[0] - frame_idx) > abs(next_face[0] - frame_idx):
                tube_idx += 1
                cur_face = next_face
            face = copy.deepcopy(face_list[cur_face[0]][cur_face[1]])
            cur_face_list.append(face)
            source_frame_list.append(cur_face[0])

            _, _, size = get_boundingbox(face, img_w, img_h)
            if size > max_size:
                max_size = size

        # align face size
        max_size = max_size // 2 * 2
        max_size = min(max_size, img_w, img_h)

        # adjust to max face size and crop faces
        cur_faces_img_256, cur_faces_img_299, cur_faces_img_320 = [], [], []
        for frame_idx in range(num_frames):
            x1, y1, size = adjust_boundingbox(
                cur_face_list[frame_idx], img_w, img_h, max_size)
            img = img_list[source_frame_list[frame_idx]
                           ][y1:y1 + size, x1:x1 + size, :]
            img_256 = cv2.resize(
                img, (256, 256), interpolation=cv2.INTER_LINEAR)
            cur_faces_img_256.append(img_256)
            img_299 = cv2.resize(
                img, (299, 299), interpolation=cv2.INTER_LINEAR)
            cur_faces_img_299.append(img_299)

        cur_faces_numpy_256 = np.stack(
            cur_faces_img_256, axis=0)  # [num_frames, h, w, c]
        cur_faces_numpy_299 = np.stack(
            cur_faces_img_299, axis=0)  # [num_frames, h, w, c]

        aligned_faces_img_256.append(cur_faces_numpy_256)
        aligned_faces_img_299.append(cur_faces_numpy_299)

    # [num_faces, num_frames, h, w, c]
    aligned_faces_numpy_256 = np.stack(aligned_faces_img_256, axis=0)
    # [num_faces, num_frames, h, w, c]
    aligned_faces_numpy_299 = np.stack(aligned_faces_img_299, axis=0)

    return [aligned_faces_numpy_256, aligned_faces_numpy_299], 1


class Ensemble:
    def __init__(self, cls_model_ckpt: str, xcp_model_ckpt: str, slow_fast_2_ckpt: str,
                 slow_fast_3_ckpt: str, b3_model_ckpt: str, res34_model_ckpt: str, b1_model_ckpt: str,
                 b1long_model_ckpt: str, b1short_model_ckpt: str, b0_model_ckpt: str, slow_fast_4_ckpt: str,
                 frame_nums: int, cuda=True):
        self.cls_model_ckpt = cls_model_ckpt
        self.xcp_model_ckpt = xcp_model_ckpt
        self.cls_model2_ckpt = slow_fast_2_ckpt
        self.cls_model3_ckpt = slow_fast_3_ckpt
        self.cls_model4_ckpt = slow_fast_4_ckpt
        self.b3_model_ckpt = b3_model_ckpt
        self.res34_model_ckpt = res34_model_ckpt
        self.b1_model_ckpt = b1_model_ckpt
        self.b1long_model_ckpt = b1long_model_ckpt
        self.b1short_model_ckpt = b1short_model_ckpt
        self.b0_model_ckpt = b0_model_ckpt

        self.frame_nums = frame_nums
        self.cuda = cuda
        self.detect_record = {}
        self.init_model()

    def init_model(self):
        self.face_det_model = init_face_detecor()
        self.face_cls_model = init_slow_fast_model(
            self.cls_model_ckpt, self.cuda)
        self.face_cls_model2 = init_slow_fast_model(
            self.cls_model2_ckpt, self.cuda)
        self.face_cls_model3 = init_slow_fast_model(
            self.cls_model3_ckpt, self.cuda)
        self.face_cls_model4 = init_slow_fast_model(
            self.cls_model4_ckpt, self.cuda)
        self.xcp_cls_model = init_xception_cls_model(
            self.xcp_model_ckpt, self.cuda)
        self.b3_cls_model = init_b3_cls_model(self.b3_model_ckpt, self.cuda)
        self.res34_cls_model = init_res34_cls_model(
            self.res34_model_ckpt, self.cuda)
        self.b1_cls_model = init_b1_cls_model(self.b1_model_ckpt, self.cuda)
        self.b1long_cls_model = init_b1_cls_model(
            self.b1long_model_ckpt, self.cuda)
        self.b1short_cls_model = init_b1_cls_model(
            self.b1short_model_ckpt, self.cuda)
        self.b0_cls_model = init_b0_cls_model(self.b0_model_ckpt, self.cuda)

    def test_kernel_video(self, video_pth):
        post_func = nn.Softmax(dim=1)
#         init_begin = time.time()
#         self.init_model()
#         init_end = time.time()
#         print('init model time:', init_end - init_begin)
#         submission = pd.read_csv("./sample_submission.csv", dtype='unicode')

        score = 0.5
        try:
            print(video_pth)
            if video_pth.split('.')[-1] != 'mp4':
                return score
            # extract image
            print(video_pth)
            reader = cv2.VideoCapture(video_pth)
            video_cnt = reader.get(cv2.CAP_PROP_FRAME_COUNT)
            interval = max(1, math.ceil(video_cnt / self.frame_nums))
            print('video_cnt:', video_cnt, 'interval:', interval)
            count, test_count = 0, 0
            success = True
            img_list = []
            while success:
                if count % interval == 0:
                    success, image = reader.read()
                    if success:
                        img_list.append(image)
                else:
                    success = reader.grab()
                count += 1
            reader.release()
            score = predict_batch(img_list, self.face_cls_model, self.face_cls_model2, self.face_cls_model3,
                                  self.xcp_cls_model, self.b3_cls_model, self.res34_cls_model, self.b1_cls_model,
                                  self.b1long_cls_model, self.b1short_cls_model, self.b0_cls_model,
                                  self.face_cls_model4, post_func, self.detect_record)
        except Exception as e:
            print(e)
            score = -1
        print('score:', score)
        if score < 0 or score > 1:
            score = 0.5
        return score
