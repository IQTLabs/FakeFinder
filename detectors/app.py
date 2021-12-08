import os
import pandas as pd
import pickle
import flask
import json
import sys
from flask import Flask, request, jsonify, make_response
from boken.ensemble import Ensemble as boken_Ensemble
from eighteen.ensemble import Ensemble as eighteen_Ensemble
from medics.ensemble import Ensemble as medics_Ensemble
from ntech.ensemble import Ensemble as ntech_Ensemble
from selimsef.ensemble import Ensemble as selimsef_Ensemble
from wm.ensemble import Ensemble as wm_Ensemble
import GPUtil
from pathvalidate import ValidationError, validate_filename, sanitize_filename

app = Flask(__name__)

models = {
    "boken": boken_Ensemble,
    "eighteen": eighteen_Ensemble,
    "medics": medics_Ensemble,
    "ntech": ntech_Ensemble,
    "selimsef": selimsef_Ensemble,
    "wm": wm_Ensemble,
}


@app.route('/healthcheck')
def starting_url():
    status_code = flask.Response(status=201)
    return status_code

@app.route('/gpustats')
def gpustats():
    gpus = GPUtil.getGPUs()
    stats = list()
    for g in gpus:
        gpu = {
            "id":g.id,
            "name": g.name,
            "load": g.load,
            "driver": g.driver,
            "memoryUsed": g.memoryUsed,
            "memoryTotal": g.memoryTotal,
            "memoryUtilization": g.memoryUtil,
            "memoryFree": g.memoryFree,
            "temperature": g.temperature
        }
        stats.append(gpu)

    print(f"{stats}", file=sys.stderr)
    return make_response(json.dumps(stats), 200)

@app.route('/predict', methods=['POST'])
def predict():
    model_names = request.get_json(force=True)['model_names']
    video_list = request.get_json(force=True)['video_list']
    predictions = []
    for modelname in model_names:
        for filename in video_list:
            score = 0.5
            video = ''
            model = models[modelname]()
            try:
                validate_filename(filename)
                video = sanitize_filename(filename, platform="auto")
                video_path = os.path.join('/uploads/', video)
                if os.path.exists(video_path):
                    score = model.inference(video_path)
                    pred={'filename': video}
                    pred[modelname]=score
                    predictions.append(pred)
                else:
                    return make_response(f"File {video} not found.", 400)
            except ValidationError as e:
                print(f'{e}')
                return make_response(f"{e}", 400)
            except Exception as err:
                print(f'{err}')
                return make_response(f"{err}", 500)

    result = pd.DataFrame(predictions)
    return result.to_json()



if __name__ == '__main__':
    app.run(host='0.0.0.0')
