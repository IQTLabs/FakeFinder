import os
import pandas as pd
import pickle
import flask
from flask import Flask, request, jsonify, make_response
from ensemble import Ensemble
import boto3
from pathvalidate import ValidationError, validate_filename, sanitize_filename

app = Flask(__name__)

model = Ensemble()
MODEL_NAME='selimsef'

@app.route('/healthcheck')
def starting_url():
    status_code = flask.Response(status=201)
    return status_code

@app.route('/predict', methods=['POST'])
def predict():
    video_list = request.get_json(force=True)['video_list']
    predictions = []
    video = ''
    for filename in video_list:
        score = 0.5
        try:
            validate_filename(filename)
            video = sanitize_filename(file_name, platform="auto")
            video_path = os.path.join('/uploads/', video)
            if os.path.exists(video_path):
                score = model.inference(video_path)
            else:
                return make_response(f"File {video} not found.", 400)
        except ValidationError as e:
            return make_response(f"{e}", 400)
        except:
            pass
        pred={'filename': video}
        pred[MODEL_NAME]=score
        predictions.append(pred)

    result = pd.DataFrame(predictions)
    return result.to_json()


if __name__ == '__main__':
    app.run(host='0.0.0.0')
