import numpy as np
import cv2

from .tracker.iou_tracker import track_iou

DETECTOR_STEP = 3

TRACKER_SIGMA_L = 0.3
TRACKER_SIGMA_H = 0.9
TRACKER_SIGMA_IOU = 0.3
TRACKER_T_MIN = 7

VIDEO_MODEL_BBOX_MULT = 1.5
VIDEO_MODEL_MIN_SIZE = 224
VIDEO_MODEL_CROP_HEIGHT = 224
VIDEO_MODEL_CROP_WIDTH = 192
VIDEO_FACE_MODEL_TRACK_STEP = 2
VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH = 7
VIDEO_SEQUENCE_MODEL_TRACK_STEP = 14


def get_tracks(detections):
    if len(detections) == 0:
        return []

    converted_detections = []
    frame_bbox_to_face_idx = {}
    for i, detections_per_frame in enumerate(detections):
        converted_detections_per_frame = []
        for j, (bbox, score) in enumerate(zip(detections_per_frame['boxes'], detections_per_frame['scores'])):
            bbox = tuple(bbox.tolist())
            frame_bbox_to_face_idx[(i, bbox)] = j
            converted_detections_per_frame.append(
                {'bbox': bbox, 'score': score})
        converted_detections.append(converted_detections_per_frame)

    tracks = track_iou(converted_detections, TRACKER_SIGMA_L,
                       TRACKER_SIGMA_H, TRACKER_SIGMA_IOU, TRACKER_T_MIN)
    tracks_converted = []
    for track in tracks:
        start_frame = track['start_frame'] - 1
        bboxes = np.array(track['bboxes'], dtype=np.float32)
        frame_indices = np.arange(
            start_frame, start_frame + len(bboxes)) * DETECTOR_STEP
        interp_frame_indices = np.arange(
            frame_indices[0], frame_indices[-1] + 1)
        interp_bboxes = np.zeros(
            (len(interp_frame_indices), 4), dtype=np.float32)
        for i in range(4):
            interp_bboxes[:, i] = np.interp(
                interp_frame_indices, frame_indices, bboxes[:, i])

        track_converted = []
        for frame_idx, bbox in zip(interp_frame_indices, interp_bboxes):
            track_converted.append((frame_idx, bbox))
        tracks_converted.append(track_converted)

    return tracks_converted


def extract_sequence(frames, start_idx, bbox, flip):
    frame_height, frame_width, _ = frames[start_idx].shape
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    xcenter = xmin + width / 2
    ycenter = ymin + height / 2
    width = width * VIDEO_MODEL_BBOX_MULT
    height = height * VIDEO_MODEL_BBOX_MULT
    xmin = xcenter - width / 2
    ymin = ycenter - height / 2
    xmax = xmin + width
    ymax = ymin + height

    xmin = max(int(xmin), 0)
    xmax = min(int(xmax), frame_width)
    ymin = max(int(ymin), 0)
    ymax = min(int(ymax), frame_height)

    sequence = []
    for i in range(VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH):
        face = cv2.cvtColor(frames[start_idx + i]
                            [ymin:ymax, xmin:xmax], cv2.COLOR_BGR2RGB)
        sequence.append(face)

    if flip:
        sequence = [face[:, ::-1] for face in sequence]

    return sequence


def extract_face(frame, bbox, flip):
    frame_height, frame_width, _ = frame.shape
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    xcenter = xmin + width / 2
    ycenter = ymin + height / 2
    width = width * VIDEO_MODEL_BBOX_MULT
    height = height * VIDEO_MODEL_BBOX_MULT
    xmin = xcenter - width / 2
    ymin = ycenter - height / 2
    xmax = xmin + width
    ymax = ymin + height

    xmin = max(int(xmin), 0)
    xmax = min(int(xmax), frame_width)
    ymin = max(int(ymin), 0)
    ymax = min(int(ymax), frame_height)

    face = cv2.cvtColor(frame[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2RGB)
    if flip:
        face = face[:, ::-1].copy()

    return face
