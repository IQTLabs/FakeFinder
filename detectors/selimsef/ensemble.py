import re

import torch
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video
from models.classifiers import DeepFakeClassifier

model_chpt = ['final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36',
              'final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19',
              'final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29',
              'final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31',
              'final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37',
              'final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40',
              'final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23']


class Ensemble:
    def __init__(self, frames_per_video=32, input_size=380, strategy=confident_strategy):
        model_paths = ['./weights/selimsef/{}'.format(x) for x in model_chpt]
        self.models = []
        for path in model_paths:
            model = DeepFakeClassifier(
                encoder="tf_efficientnet_b7_ns").to("cuda")
            print("loading state dict {}".format(path))
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(
                {re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
            model.eval()
            del checkpoint
            self.models.append(model.half())
        self.frames_per_video = frames_per_video
        video_reader = VideoReader()
        def video_read_fn(x): return video_reader.read_frames(
            x, num_frames=frames_per_video)
        self.face_extractor = FaceExtractor(video_read_fn)
        self.input_size = input_size
        self.strategy = strategy

    def inference(self, video_path):
        return predict_on_video(self.face_extractor, video_path,
                                self.frames_per_video, self.input_size,
                                self.models, self.strategy)
