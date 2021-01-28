import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from .dsfacedetector.face_ssd_infer import SSD

DETECTOR_THRESHOLD = 0.3
DETECTOR_MIN_SIZE = 512
DETECTOR_MAX_SIZE = 512
DETECTOR_MEAN = (104.0, 117.0, 123.0)
DETECTOR_STD = (1.0, 1.0, 1.0)

__all__ = ['Detector']


class Detector(object):
    def __init__(self, weights_path):
        self.model = SSD('test')
        self.model.cuda().eval()

        state = torch.load(
            weights_path, map_location=lambda storage, loc: storage)
        state = {key: value.float() for key, value in state.items()}
        self.model.load_state_dict(state)

        self.transform = GeneralizedRCNNTransform(
            DETECTOR_MIN_SIZE, DETECTOR_MAX_SIZE, DETECTOR_MEAN, DETECTOR_STD)
        self.transform.eval()

    def detect(self, images):
        images = torch.stack([torch.from_numpy(image).cuda()
                              for image in images])
        images = images.transpose(1, 3).transpose(2, 3).float()
        original_image_sizes = [img.shape[-2:] for img in images]
        images, _ = self.transform(images, None)
        with torch.no_grad():
            detections_batch = self.model(images.tensors).cpu().numpy()
        result = []
        for detections, image_size in zip(detections_batch, images.image_sizes):
            scores = detections[1, :, 0]
            keep_idxs = scores > DETECTOR_THRESHOLD
            detections = detections[1, keep_idxs, :]
            detections = detections[:, [1, 2, 3, 4, 0]]
            detections[:, 0] *= image_size[1]
            detections[:, 1] *= image_size[0]
            detections[:, 2] *= image_size[1]
            detections[:, 3] *= image_size[0]
            result.append({
                'scores': torch.from_numpy(detections[:, 4]),
                'boxes': torch.from_numpy(detections[:, :4])
            })

        result = self.transform.postprocess(
            result, images.image_sizes, original_image_sizes)
        return result
