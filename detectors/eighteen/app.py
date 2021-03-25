import os
import pandas as pd
import pickle
import flask
from flask import Flask, request, jsonify
from ensemble import *
import boto3

app = Flask(__name__)


BUCKET_NAME = 'ff-inbound-videos'  # replace with your bucket name

s3 = boto3.resource('s3')

chpt_dir = './weights'
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
        video = filename.rsplit('/',1)[-1]
        try:
            s3.Bucket(BUCKET_NAME).download_file(video, video)
            score = model.inference(video)
            os.remove(video)
        except:
            pass
        predictions.append({'filename': video, 'eighteen': score})

    result = pd.DataFrame(predictions)
    return result.to_json()


if __name__ == '__main__':
    app.run(host='0.0.0.0')
