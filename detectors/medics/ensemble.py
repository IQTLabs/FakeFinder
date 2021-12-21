import numpy as np
import yaml
import gc

import torch
import torch.nn.functional as F
from torchvision.models.video import mc3_18, r2plus1d_18
from facenet_pytorch import MTCNN

from .cnn3d import *
from .cnn2d import *
from .constants import *
from .face_utils import *

__all__ = ['Ensemble']


def inference_3d(models_3d, faces, video):
    predictions = []
    try:
        for modeldict in models_3d:
            preds_video = []
            model = modeldict['model']

            for i_face, face in enumerate(faces):
                (frame_from, frame_to), (row_from,
                                         row_to), (col_from, col_to) = face

                x = video[frame_from:frame_to,
                          row_from:row_to + 1, col_from:col_to + 1]
                x = resize_and_square_face(x, output_size=OUTPUT_FACE_SIZE)

                if PRE_INFERENCE_CROP and PRE_INFERENCE_CROP != OUTPUT_FACE_SIZE:
                    x = center_crop_video(x, PRE_INFERENCE_CROP)

                with torch.no_grad():

                    if modeldict['norm'] == '112_imagenet':
                        x = np.array([test_transforms_114_imagenet(
                            image=frame)['image'] for frame in x])
                    elif modeldict['norm'] == '224_imagenet':
                        x = np.array([test_transforms_224_imagenet(
                            image=frame)['image'] for frame in x])

                    x = torch.from_numpy(x.transpose([3, 0, 1, 2])).float()

                    if modeldict['norm'] == 'i3d':
                        x = (x / 255.) * 2 - 1
                    elif modeldict['norm'] == 'nil':
                        pass
                    elif modeldict['norm'] == '112_imagenet':
                        pass
                    elif modeldict['norm'] == '224_imagenet':
                        pass
                    else:
                        raise ValueError(
                            f"Unknown normalisation mode {modeldict['norm']}")

                    y_pred = model(x.cuda())
                    prob0, prob1 = torch.mean(
                        torch.exp(F.log_softmax(y_pred, dim=1)), dim=0)
                    if REVERSE_PROBS:
                        prob1 = 1-prob1
                    preds_video.append(float(prob1))

            if len(preds_video) > 0:
                predictions.append(USE_FACE_FUNCTION(preds_video) * RATIO_3D)
    except:
        pass
    return predictions


def inference_2d(model_2d, faces, coords, video, loader):
    try:
        FRAMES2D = 32
        # Ian's 2D model
    #     coords = coords_by_videopath[videopath]
        preds_video = []

        for i_coord, coordinate in enumerate(coords):
            (frame_from, frame_to), (row_from,
                                     row_to), (col_from, col_to) = faces[i_coord]
            x = []
            for coord_ind, frame_number in enumerate(range(frame_from, min(frame_from+FRAMES2D, frame_to-1))):
                if coord_ind >= len(coordinate):
                    break
                x1, y1, x2, y2 = coordinate[coord_ind]
                x.append(video[frame_number, y1:y2, x1:x2])
            x = np.asarray(x)
            # Reverse back to BGR because it will get reversed to RGB when preprocessed
            # x = x[...,::-1]
            # Preprocess
            x = loader.dataset.process_video(x)
            # x = np.asarray([loader.dataset.process_image(_) for _ in x])
            # Flip every other frame
            x[:, ::2] = x[:, ::2, :, ::-1]
            # RGB reverse every 3rd frame
            # x[:,::3] = x[::-1,::3]
            with torch.no_grad():
                out = model_2d(torch.from_numpy(
                    np.ascontiguousarray(x)).unsqueeze(0).cuda())
            # out = np.median(out.cpu().numpy())
            preds_video.append(out.cpu().numpy())
        if len(preds_video) > 0:
            return USE_FACE_FUNCTION(preds_video) * RATIO_2D
        else:
            pass
    except:
        pass


class Ensemble():
    def __init__(self):
        self.mtcnn = MTCNN(margin=0, keep_all=True, post_process=False, select_largest=False,
                           device='cuda:0', thresholds=MTCNN_THRESHOLDS, factor=MMTNN_FACTOR)
        self.models_3d = build_models()
        with open('./medics/cnn2d/experiment001.yaml') as f:
            CFG = yaml.load(f, Loader=yaml.FullLoader)

            CFG['model']['params']['pretrained'] = None
            model2d = build_model(CFG['model']['name'], CFG['model']['params'])
            model2d.load_state_dict(torch.load(
                './weights/medics/SRXT50_094_VM-0.2504.PTH'))
            model2d = model2d.eval().cuda()
            self.model_2d = model2d
            self.loader = build_dataloader(
                CFG, data_info={'vidfiles': [], 'labels': []}, mode='predict')

    def inference(self, video_path):
        faces, coords = face_detection_wrapper(self.mtcnn, video_path, every_n_frames=FACE_FRAMES,
                                               facedetection_downsample=FACEDETECTION_DOWNSAMPLE,
                                               max_frames_to_load=MAX_FRAMES_TO_LOAD)
        if len(faces):
            last_frame_needed = get_last_frame_needed_across_faces(faces)
            video, rescale = load_video(video_path, every_n_frames=1, to_rgb=True, rescale=None,
                                        inc_pil=False, max_frames=last_frame_needed)
        else:
            return 0.5
        predictions = inference_3d(self.models_3d, faces, video)
        preds_2d = inference_2d(self.model_2d, faces,
                                coords, video, self.loader)
        if preds_2d:
            predictions.append(preds_2d)
        if len(predictions) > 0:
            return np.clip(np.mean(predictions), PROB_MIN, PROB_MAX)
        else:
            return 0.5

    def __del__(self):
        del self.mtcnn
        del self.models_3d
        del self.model_2d
        del self.loader
        
        torch.cuda.empty_cache()
        gc.collect()
