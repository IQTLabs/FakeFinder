import numpy as np
import cv2
import os
import gc

from .face_utils import get_tracks, extract_sequence, extract_face
from .models import *

PATH_PREFIX = '/weights/ntech/'
DETECTOR_WEIGHTS_PATH = os.path.join(PATH_PREFIX, 'WIDERFace_DSFD_RES152.fp16.pth')
VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH = os.path.join(PATH_PREFIX, 'efficientnet-b7_ns_seq_aa-original-mstd0.5_100k_v4_cad79a/snapshot_100000.fp16.pth')
FIRST_VIDEO_FACE_MODEL_WEIGHTS_PATH = os.path.join(PATH_PREFIX, 'efficientnet-b7_ns_aa-original-mstd0.5_large_crop_100k_v4_cad79a/snapshot_100000.fp16.pth')
SECOND_VIDEO_FACE_MODEL_WEIGHTS_PATH = os.path.join(PATH_PREFIX, 'efficientnet-b7_ns_aa-original-mstd0.5_re_100k_v4_cad79a/snapshot_100000.fp16.pth')

VIDEO_FACE_MODEL_TRACK_STEP = 2
VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH = 7
VIDEO_SEQUENCE_MODEL_TRACK_STEP = 14


class video_reader:
    def __init__(self, face_detector, video_target_fps=15, detector_step=3):
        self.video_target_fps = video_target_fps
        self.face_detector = face_detector
        self.detector_step = detector_step

    def process_video(self, video_path):
        sample = {
            'frames': [],
            'tracks': []
        }
        capture = cv2.VideoCapture(video_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            return sample

        fps = int(capture.get(cv2.CAP_PROP_FPS))
        video_step = round(fps / self.video_target_fps)
        if video_step == 0:
            return sample

        for i in range(frame_count):
            capture.grab()
            if i % video_step != 0:
                continue
            ret, frame = capture.retrieve()
            if not ret:
                continue

            sample['frames'].append(frame)

        detector_frames = sample['frames'][::self.detector_step]
        DETECTOR_BATCH_SIZE = 16
        detections = []
        for start in range(0, len(detector_frames), DETECTOR_BATCH_SIZE):
            end = min(len(detector_frames), start + DETECTOR_BATCH_SIZE)
            detections_batch = self.face_detector.detect(
                detector_frames[start:end])
            for detections_per_frame in detections_batch:
                detections.append({key: value.cpu().numpy()
                                   for key, value in detections_per_frame.items()})
        sample['tracks'] = get_tracks(detections)
        return sample


class Ensemble:
    def __init__(self):
        detector = Detector(DETECTOR_WEIGHTS_PATH)
        self.track_sequences_classifier = TrackSequencesClassifier(
            VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH)
        self.track_faces_classifier = TrackFacesClassifier(
            FIRST_VIDEO_FACE_MODEL_WEIGHTS_PATH, SECOND_VIDEO_FACE_MODEL_WEIGHTS_PATH)
        self.reader = video_reader(detector)

    def inference(self, video_path):
        sample = self.reader.process_video(video_path)
        frames, tracks = sample['frames'], sample['tracks']
        if len(frames) == 0 or len(tracks) == 0:
            return 0.5
        sequence_track_scores = []
        for track in tracks:
            track_sequences = []
            for i, (start_idx, _) in enumerate(
                    track[:-VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH + 1:VIDEO_SEQUENCE_MODEL_TRACK_STEP]):
                assert start_idx >= 0 and start_idx + \
                    VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH <= len(frames)
                _, bbox = track[i * VIDEO_SEQUENCE_MODEL_TRACK_STEP +
                                VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH // 2]
                track_sequences.append(extract_sequence(
                    frames, start_idx, bbox, i % 2 == 0))
            sequence_track_scores.append(
                self.track_sequences_classifier.classify(track_sequences))
        face_track_scores = []
        for track in tracks:
            track_faces = []
            for i, (frame_idx, bbox) in enumerate(track[::VIDEO_FACE_MODEL_TRACK_STEP]):
                face = extract_face(frames[frame_idx], bbox, i % 2 == 0)
                track_faces.append(face)
            face_track_scores.append(
                self.track_faces_classifier.classify(track_faces))

        sequence_track_scores = np.concatenate(sequence_track_scores)
        face_track_scores = np.concatenate(face_track_scores)
        track_probs = np.concatenate(
            (sequence_track_scores, face_track_scores))

        delta = track_probs - 0.5
        sign = np.sign(delta)
        pos_delta = delta > 0
        neg_delta = delta < 0
        track_probs[pos_delta] = np.clip(
            0.5 + sign[pos_delta] * np.power(abs(delta[pos_delta]), 0.65), 0.01, 0.99)
        track_probs[neg_delta] = np.clip(
            0.5 + sign[neg_delta] * np.power(abs(delta[neg_delta]), 0.65), 0.01, 0.99)
        weights = np.power(abs(delta), 1.0) + 1e-4
        video_score = float((track_probs * weights).sum() / weights.sum())

        return video_score

    def __del__(self):
        del self.track_sequences_classifier
        del self.track_faces_classifier
        del self.reader
        
        gc.collect()
