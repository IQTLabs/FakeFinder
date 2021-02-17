import os
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from ensemble import Ensemble
import boto3
import sys

app = Flask(__name__)

model = Ensemble()
BUCKET_NAME = 'ff-inbound-videos'  # replace with your bucket name

s3 = boto3.resource('s3')


@app.route('/predict', methods=['POST'])
def predict():
    video_list = request.get_json(force=True)['video_list']
    predictions = []
    for filename in video_list:
        score = 0.5
        video = filename.rsplit('/',1)[-1]
        try:
            s3.Bucket(BUCKET_NAME).download_file(video, video)
            score = model.inference(video)
            os.remove(video)
        except:
            pass
        predictions.append({'filename': video, 'selimsef': score})

    result = pd.DataFrame(predictions)
    return result.to_json()


if __name__ == '__main__':
    app.run(host='0.0.0.0')
