import pandas as pd
import pickle
from flask import Flask, request, jsonify
from ensemble import Ensemble

app = Flask(__name__)

model = Ensemble()


@app.route('/predict', methods=['POST'])
def predict():
    video_list = str(request.get_json(force=True)['video_list'])
    predictions = []
    for video in video_list:
        score = 0.5
        try:
            # BOTO call here
            # only keep video name when copying it
            score = model.inference(video.split('/')[-1])
        except:
            pass
        predictions.append({'filename': video, 'prediction': score})

    result = pd.DataFrame(predictions)
    return result.to_json()


if __name__ == '__main__':
    app.run(host='0.0.0.0')
