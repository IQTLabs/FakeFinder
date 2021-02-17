import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify
from ensmble import Ensemble
import boto3

DETECTOR_WEIGHTS_PATH = 'WIDERFace_DSFD_RES152.fp16.pth'
VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH = 'efficientnet-b7_ns_seq_aa-original-mstd0.5_100k_v4_cad79a/snapshot_100000.fp16.pth'
FIRST_VIDEO_FACE_MODEL_WEIGHTS_PATH = 'efficientnet-b7_ns_aa-original-mstd0.5_large_crop_100k_v4_cad79a/snapshot_100000.fp16.pth'
SECOND_VIDEO_FACE_MODEL_WEIGHTS_PATH = 'efficientnet-b7_ns_aa-original-mstd0.5_re_100k_v4_cad79a/snapshot_100000.fp16.pth'

app = Flask(__name__)

model = Ensemble(os.path.join('./weights/', DETECTOR_WEIGHTS_PATH),
                 os.path.join(
    './weights/', VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH),
    os.path.join(
    './weights/', FIRST_VIDEO_FACE_MODEL_WEIGHTS_PATH),
    os.path.join(
    './weights/', SECOND_VIDEO_FACE_MODEL_WEIGHTS_PATH)
)
BUCKET_NAME = 'ff-inbound-videos'  # replace with your bucket name

s3 = boto3.resource('s3')


@app.route('/predict', methods=['POST'])
def predict():
    video_list = str(request.get_json(force=True)['video_list'])
    predictions = []
    for video in video_list:
        score = 0.5
        try:
            s3.Bucket(BUCKET_NAME).download_file(video, video)
            score = model.inference(video.split('/')[-1])
            os.remove(video)
        except:
            pass
        predictions.append({'filename': video, 'prediction': score})

    result = pd.DataFrame(predictions)
    return result.to_json()


if __name__ == '__main__':
    app.run(host='0.0.0.0')
