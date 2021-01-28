import os
import numpy as np
import pickle
from flask import Flask, request, jsonify
from ensmble import Ensemble

DETECTOR_WEIGHTS_PATH = 'WIDERFace_DSFD_RES152.fp16.pth'
VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH = 'efficientnet-b7_ns_seq_aa-original-mstd0.5_100k_v4_cad79a/snapshot_100000.fp16.pth'
FIRST_VIDEO_FACE_MODEL_WEIGHTS_PATH = 'efficientnet-b7_ns_aa-original-mstd0.5_large_crop_100k_v4_cad79a/snapshot_100000.fp16.pth'
SECOND_VIDEO_FACE_MODEL_WEIGHTS_PATH = 'efficientnet-b7_ns_aa-original-mstd0.5_re_100k_v4_cad79a/snapshot_100000.fp16.pth'

app = Flask(__name__)

submit = Ensemble(os.path.join('./weights/', DETECTOR_WEIGHTS_PATH),
                  os.path.join(
                      './weights/', VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH),
                  os.path.join(
                      './weights/', FIRST_VIDEO_FACE_MODEL_WEIGHTS_PATH),
                  os.path.join(
                      './weights/', SECOND_VIDEO_FACE_MODEL_WEIGHTS_PATH)
                  )


@app.route('/predict', methods=['POST'])
def predict():
    video_pth = str(request.get_json(force=True)['video_path'])
    result = submit.inference(video_pth)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
