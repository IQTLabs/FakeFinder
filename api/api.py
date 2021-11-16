from flask import Flask, render_template, jsonify, make_response
from flask_restx import Api, Resource, fields, reqparse
from pathvalidate import ValidationError, validate_filename, sanitize_filename
from werkzeug.datastructures import FileStorage
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
import json
import time
import os
import sys
import threading

class ProgressPercentage(object):

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()

app = Flask(__name__)
api = Api(app, version='1.0', title='FakeFinder API',
    description='FakeFinder API',
)

upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)


ns = api.namespace('fakefinder', description='FakeFinder operations')

ffmodel = api.model('FakeFinder', {
    'modelName': fields.String(required=True, description='Name of the model to run inference against')
})


class FakeFinderDAO(object):
    def __init__(self):
        self.ffmodels = []

    def get(self, modelName):
        for ffmodel in self.ffmodels:
            if ffmodel['modelName'] == modelName:
                return ffmodel
        api.abort(404, "FakeFinder model {} doesn't exist".format(modelName))

    def create(self, data):
        ffmodel = data
        self.ffmodels.append(ffmodel)
        return ffmodel

    def update(self, modelName, data):
        ffmodel = self.get(modelName)
        ffmodel.update(data)
        return ffmodel

    def delete(self, modelName):
        ffmodel = self.get(modelName)
        self.ffmodels.remove(ffmodel)


DAO = FakeFinderDAO()

def StartAWSColdInstance(model_name):
    url = 'http://'+ model_name + ':5000/predict'

@ns.route('/')
class FakeFinderPost(Resource):
    @ns.doc('get_fakefinder_models')
    def get(self):
        models = ["selimsef", "eighteen", "medics", "ntech", "wm", "boken"]
        print(models)
        return jsonify( { 'models': models } )


    @ns.doc('create_fakefinder_inference_task')
    @ns.expect(ffmodel)
    def post(self):
        try:
            '''Create a new task'''
            # request payload can be a list or a dictionary
            print(type(api.payload))
            agg_response = []
            if type(api.payload) is list:
                 # loop through the list for each of the selected models
                 for r in api.payload:
                    url = 'http://' + r['modelName'] + ':5000/predict'

                    headers = {'Content-type': 'application/json; charset=UTF-8'}
                    video_list = list()
                    if type(r['s3Location']) is not list:
                        video_list.append(r['s3Location'])
                    else:
                        video_list = r['s3Location']

                    response = requests.post(url, json={'video_list': video_list}, headers=headers)
                    agg_response.append(response.json())   

            print(json.dumps(agg_response))
        except Exception as e:
            make_response(e.message,500)
        return make_response(jsonify(json.dumps(agg_response)), 200)

@api.route('/upload/')
@api.expect(upload_parser)
class Upload(Resource):
    def post(self):
        args = upload_parser.parse_args()
        print(args)
        uploaded_file = args['file']  # This is FileStorage instance
        print(f'{uploaded_file.filename}')
        try:
            validate_filename(uploaded_file.filename, platform="auto")
            sanitized_filename = sanitize_filename(uploaded_file.filename, platform="auto")
            if not os.path.exists('./uploads'):
                os.makedirs('./uploads')
            file_path = os.path.join("./uploads", sanitized_filename) # path where file can be saved
            chunks = uploaded_file.read()
            with open(file_path, 'wb') as f:
                f.write(chunks)
        except ValidationError as e:
            return make_response(f"{e}", 400)
        except Exception as err:
            return {
                'statusCode': 500,
                'body': json.dumps(err)
            }

if __name__ == '__main__':
    app.run(threaded=True, debug=True, host='0.0.0.0')
