import os
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from ensemble import Ensemble
import boto3

app = Flask(__name__)

model = Ensemble()
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
