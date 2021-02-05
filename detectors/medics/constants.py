import numpy as np
import albumentations as A
# Face detection
MAX_FRAMES_TO_LOAD = 100
MIN_FRAMES_FOR_FACE = 30
MAX_FRAMES_FOR_FACE = 100
FACE_FRAMES = 10
MAX_FACES_HIGHTHRESH = 5
MAX_FACES_LOWTHRESH = 1
FACEDETECTION_DOWNSAMPLE = 0.25
MTCNN_THRESHOLDS = (0.8, 0.8, 0.9)  # Default [0.6, 0.7, 0.7]
MTCNN_THRESHOLDS_RETRY = (0.5, 0.5, 0.5)
MMTNN_FACTOR = 0.71  # Default 0.709 p
TWO_FRAME_OVERLAP = False
# Inference
PROB_MIN, PROB_MAX = 0.001, 0.999
REVERSE_PROBS = True
DEFAULT_MISSING_PRED = 0.5
USE_FACE_FUNCTION = np.mean

# 3D inference
RATIO_3D = 1
OUTPUT_FACE_SIZE = (256, 256)
PRE_INFERENCE_CROP = (224, 224)

# 2D
RATIO_2D = 1

test_transforms_114_imagenet = A.Compose([A.Resize(height=112, width=112),
                                          A.Normalize()])

test_transforms_224_imagenet = A.Compose([A.Normalize()])
