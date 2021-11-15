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
    'batchMode': fields.Boolean(required=True, default=False, description='Set this field to true if processing video/image files in batch mode. If requests are coming from a UI this should be set to false.'),
    'alwaysOn': fields.Boolean(required=False, default=True, description='Set this field to true if starting/stopping ec2 instances'),
    's3Location': fields.String(required=True, description='Image/Video S3 location. If uploading the file ththe value should be bucket name.'),
    'modelName': fields.String(required=True, description='Name of the model to run inference against'),
    'splitRequests': fields.Boolean(required=False, default=False, description='Split the request containing a list of videos to multiple requests containing single video.'),
    'numSplitRequests': fields.Integer(required=False, default=1, description='Number of splits of the list containing videos.')
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

def UploadFile(file_name, file_content):
    sanitized_filename = None
    try:
        validate_filename(file_name, platform="auto")
        sanitized_filename = sanitize_filename(file_name, platform="auto")
        if not os.path.exists('./uploads'):
            os.makedirs('./uploads')
        file_path = os.path.join("./uploads", sanitized_filename) # path where file can be saved
        print(f'{file_path}')
        file_content.save(file_path)
    except ValidationError as e:
        return make_response(f"{e}", 400)
    except Exception as err:
        return {
            'statusCode': 500,
            'body': json.dumps(err)
        }

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
                 # if split requests is true then send one file per request.
                 if r['splitRequests'] is True and r['batchMode'] is False:
                    splits = r['numSplitRequests']
                    if type(r['s3Location']) is not list:
                       # convert dict to list
                       s3Location_list = list(r['s3Location'].items())
                       final = np.array_split(s3Location_list, splits)
                    else:
                       final = np.array_split(r['s3Location'], splits)
                    print(final)
                    for i, loc in enumerate(final, start=1):
                        print(loc)
                        response = requests.post(url, json={'video_list': loc.tolist()}, headers=headers)
                        agg_response.append(response.json())
                 elif r['splitRequests'] is True and r['batchMode'] is True:
                    print("Split requests for batch mode concurrent processing")
                    # Spawn cold ec2 instance concurrently and send requests.
                    threads= []
                    splits = r['numSplitRequests']
                    if type(r['s3Location']) is not list:
                       # convert dict to list
                       s3Location_list = list(r['s3Location'].items())
                       final = np.array_split(s3Location_list, splits)
                    else:
                       final = np.array_split(r['s3Location'], splits)
                    print(final)
                    with ThreadPoolExecutor(max_workers=20) as executor:
                         for i in range(splits):
                             threads.append(executor.submit(StartAWSColdInstance, r['modelName']))
                         for i, task in enumerate(as_completed(threads)):
                             print("Task " + str(i))
                             print(task.result())
                             response = requests.post(task.result(), json={'video_list': final[i].tolist()}, headers=headers)
                             #yield response.json()
                             agg_response.append(response.json())  
                 else:
                    if type(r['s3Location']) is list:
                        response = requests.post(url, json={'video_list': r['s3Location']}, headers=headers)
                    else:
                        response = requests.post(url, json={'video_list': [r['s3Location']]}, headers=headers)
                    agg_response.append(response.json())
        elif type(api.payload) is dict:
                 # request contains a single model
                 url = 'http://' + api.payload['modelName'] + ':5000/predict'
                 headers = {'Content-type': 'application/json; charset=UTF-8'}
                 # if split requests is true then send one file per request.
                 if api.payload['splitRequests'] is True and api.payload['batchMode'] is False:
                    print("Split requests")
                    splits = api.payload['numSplitRequests']
                    if type(api.payload['s3Location']) is not list:
                       # convert dict to list
                       s3Location_list = list(api.payload['s3Location'].items())
                       final = np.array_split(s3Location_list, splits)
                    else:
                       final = np.array_split(api.payload['s3Location'], splits)
                    print(final)
                    for i, loc in enumerate(final, start=1):
                        print(loc)
                        response = requests.post(url, json={'video_list': loc.tolist()}, headers=headers)
                        agg_response.append(response.json())
                 elif api.payload['splitRequests'] is True and api.payload['batchMode'] is True:
                    print("Split requests for batch mode concurrent processing")
                    threads= []
                    splits = api.payload['numSplitRequests']
                    if type(api.payload['s3Location']) is not list:
                       # convert dict to list
                       s3Location_list = list(api.payload['s3Location'].items())
                       final = np.array_split(s3Location_list, splits)
                    else:
                       final = np.array_split(api.payload['s3Location'], splits)
                    print(final)
                    with ThreadPoolExecutor(max_workers=20) as executor:
                         for i in range(splits):
                             threads.append(executor.submit(StartAWSColdInstance, api.payload['modelName']))
                         for i, task in enumerate(as_completed(threads)):
                             print("Task " + str(i))
                             print(task.result())
                             response = requests.post(task.result(), json={'video_list': final[i].tolist()}, headers=headers)
                             #yield response.json()
                             agg_response.append(response.json())
                 else:
                    if type(api.payload['s3Location']) is list:
                        response = requests.post(url, json={'video_list': api.payload['s3Location']}, headers=headers)
                    else:
                        response = requests.post(url, json={'video_list': [api.payload['s3Location']]}, headers=headers)
                    agg_response.append(response.json())
        print(json.dumps(agg_response))
        return make_response(jsonify(json.dumps(agg_response)), 200)

@api.route('/upload/')
@api.expect(upload_parser)
class Upload(Resource):
    def post(self):
        args = upload_parser.parse_args()
        print(args)
        uploaded_file = args['file']  # This is FileStorage instance
        print(f'{uploaded_file.filename}')
        return UploadFile(uploaded_file.filename, uploaded_file)


if __name__ == '__main__':
    app.run(threaded=True, debug=True, host='0.0.0.0')
