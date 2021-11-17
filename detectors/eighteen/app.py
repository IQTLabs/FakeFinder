import os
import pandas as pd
import pickle
import flask
from flask import Flask, request, jsonify, make_response
from ensemble import *
from pathvalidate import ValidationError, validate_filename, sanitize_filename

app = Flask(__name__)

chpt_dir = '/weights/eighteen'
load_slowfast_path = '{}/sf_bc_jc_44000.pth.tar'.format(chpt_dir)
load_slowfast_path2 = '{}/sf_32000.pth.tar'.format(chpt_dir)
load_slowfast_path3 = '{}/sf_16x8_bc_jc_44000.pth.tar'.format(chpt_dir)
load_slowfast_path4 = '{}/sf_trainval_52000.pth.tar'.format(chpt_dir)
load_xcp_path = '{}/xcep_bgr_58000.pth.tar'.format(chpt_dir)
load_b3_path = '{}/b3_rgb_50000.pth.tar'.format(chpt_dir)
load_res34_path = '.{}/res34_rgb_23000.pth.tar'.format(chpt_dir)
load_b1_path = '{}/b1_rgb_58000.pth.tar'.format(chpt_dir)
load_b1long_path = '{}/b1_rgb_long_alldata_66000.pth.tar'.format(chpt_dir)
load_b1short_path = '{}/b1_rgb_alldata_58000.pth.tar'.format(chpt_dir)
load_b0_path = '{}/b0_rgb_58000.pth.tar'.format(chpt_dir)

frame_nums = 160
model = Ensemble(load_slowfast_path, load_xcp_path, load_slowfast_path2, load_slowfast_path3, load_b3_path,
                 load_res34_path, load_b1_path,
                 load_b1long_path, load_b1short_path, load_b0_path, load_slowfast_path4, frame_nums,
                 cuda=pipeline_cfg.cuda)
MODEL_NAME='eighteen'

@app.route('/healthcheck')
def starting_url():
    status_code = flask.Response(status=201)
    return status_code

@app.route('/predict', methods=['POST'])
def predict():
    video_list = request.get_json(force=True)['video_list']
    predictions = []
    for filename in video_list:
        score = 0.5
        video = ''
        try:
            validate_filename(filename)
            video = sanitize_filename(filename, platform="auto")
            video_path = os.path.join('/uploads/', video)
            if os.path.exists(video_path):
                score = model.inference(video_path)
                pred={'filename': video}
                pred[MODEL_NAME]=score
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
