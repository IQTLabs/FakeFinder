from flask import Flask, render_template, jsonify, make_response
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import boto3
from botocore.exceptions import ClientError
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
upload_parser.add_argument('bucket', required=True)


ns = api.namespace('fakefinder', description='FakeFinder operations')

client = boto3.client('ec2', region_name='us-east-1')
ec2 = boto3.resource('ec2', region_name='us-east-1')
s3_client = boto3.client('s3', region_name='us-east-1')
ecr = boto3.client('ecr', region_name='us-east-1')

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

ColdInstanceIds = []
WarmInstanceIds = []

def StartAWSColdInstance(model_name):
    instance_tag = cold_instance_ids[model_name]
    print(instance_tag)
    with open('userdata.txt', 'r') as file:
         data = file.read().format(detector = instance_tag, detector_weights = model_name)
    print(data)
    response = client.run_instances(ImageId='ami-072519eedc1730252', UserData=str(data), MaxCount=1,MinCount=1, SubnetId='subnet-0c2eace77b3b213bb', SecurityGroupIds=['sg-002b99dc572b09491'], InstanceType='g4dn.xlarge', IamInstanceProfile={'Arn': 'arn:aws:iam::352435047368:instance-profile/WorkerNode' }, InstanceInitiatedShutdownBehavior='terminate', BlockDeviceMappings=[{'DeviceName': '/dev/xvda', 'Ebs':{'VolumeSize': 200, 'DeleteOnTermination': True}}], KeyName='fakefinder-apiserver')
    instance_id = response['Instances'][0]['InstanceId']
    print(instance_id)
    ColdInstanceIds.append(instance_id)
    instance = ec2.Instance(instance_id) 
    instance.wait_until_running()
    print("Wait till instance status is ok")
    waiter = client.get_waiter('instance_status_ok')
    waiter.wait(InstanceIds=[instance_id,],)
    print("Instance private ip address")
    print(instance.private_ip_address)
    url = 'http://'+ instance.private_ip_address + ':5000/predict'
    #print("Sleeping for 20 seconds for stabilization")
    #time.sleep(20)
    print("Verifying that model service is ready")
    while True:
          try:
              time.sleep(20)
              healthcheck_response = requests.get('http://'+ instance.private_ip_address + ':5000/healthcheck') 
              print(healthcheck_response.status_code)
              if healthcheck_response.status_code == 201:
                 print("Inference service is ready")
                 break
          except:
              print("Inference service is not ready")
    return url

def TerminateAWSColdInstance():
    for instance_id in ColdInstanceIds:
        response = client.terminate_instances(InstanceIds=[instance_id,],)
        print(response)

def GetUrlFromAWSInstance(model_name):
    instance_id = warm_instance_ids[model_name]
    instance = ec2.Instance(instance_id)
    url = 'http://'+ instance.private_ip_address + ':5000/predict'
    return url 

def StartAWSWarmInstance(model_name):
    instance_id = warm_instance_ids[model_name]
    response = client.start_instances(InstanceIds=[instance_id,],)
    print(response)

    instance = ec2.Instance(instance_id)
    WarmInstanceIds.append(instance_id)
    print("Wait till instance starts running")
    instance.wait_until_running()
    print("Wait till instance status is ok")
    waiter = client.get_waiter('instance_status_ok')
    waiter.wait(InstanceIds=[instance_id,],)
    print("Instance private ip address")
    print(instance.private_ip_address)
    url = 'http://'+ instance.private_ip_address + ':5000/predict'
    return url

def StopAWSWarmInstance():
    for instance_id in WarmInstanceIds:
        response = client.stop_instances(InstanceIds=[instance_id,],)
        print(response)

def UploadFileToS3(file_name, file_content, bucket):
    s3_client = boto3.client('s3')
    file_path = os.path.join("./test_upload", file_name) # path where file can be saved
    file_content.save(file_path)
    # Upload the file
    try:
            s3_client.upload_file(file_name, bucket, Callback=ProgressPercentage(file_name))
            return "s3://"+ bucket + "/" + file_name
    except ClientError as e:
            logging.error(e)
    

# warm aws instances to support ui/batch mode
with open("models.json") as jsonfile:
     warm_instance_ids = json.load(jsonfile)

# cold aws instances to support batch mode
with open("images.json") as jsonfile:
     cold_instance_ids = json.load(jsonfile)

@ns.route('/')
class FakeFinderPost(Resource):
    @ns.doc('get_fakefinder_models')
    def get(self):
        print(list(warm_instance_ids.keys()))
        return jsonify( { 'models': list(warm_instance_ids.keys()) } )


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
                 if r['alwaysOn'] is False and r['batchMode'] is False:
                    print("Bringing up warm static instances")
                    url = StartAWSWarmInstance(r['modelName'])
                 elif r['alwaysOn'] is True and r['batchMode'] is False:
                    print("Using alwaysOn static instances")
                    url = GetUrlFromAWSInstance(r['modelName'])

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
                 if api.payload['alwaysOn'] is False and api.payload['batchMode'] is False:
                    print("Bringing up warm static instances")
                    url = StartAWSWarmInstance(api.payload['modelName'])
                 elif api.payload['alwaysOn'] is True and api.payload['batchMode'] is False:
                    print("Using alwaysOn static instances")
                    url = GetUrlFromAWSInstance(api.payload['modelName'])
                 
                 headers = {'Content-type': 'application/json; charset=UTF-8'}
                 # if split requests is true then send one file per request.
                 if api.payload['splitRequests'] is True and api.payload['batchMode'] is False:
                    print("Split requests for warm/always on instances")
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
                    # Spawn cold ec2 instance concurrently and send requests.
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
      finally:
        print("Terminate/Stop Instances")
        time.sleep(10)
        StopAWSWarmInstance()
        WarmInstanceIds.clear()
        TerminateAWSColdInstance()
        ColdInstanceIds.clear()

@api.route('/uploadS3/')
@api.expect(upload_parser)
class UploadS3(Resource):
    def post(self):
        args = upload_parser.parse_args()
        print(args)
        bucket = args['bucket']
        uploaded_file = args['file']  # This is FileStorage instance
        uploaded_file.save("./tmp/" + uploaded_file.filename)
        try:
            s3_client.upload_file("./tmp/" + uploaded_file.filename, bucket,  uploaded_file.filename, Callback=ProgressPercentage("./tmp/" + uploaded_file.filename))
            return "s3://"+ bucket + "/" + uploaded_file.filename, 201
        except ClientError as e:
            logging.error(e)
            return 400

if __name__ == '__main__':
    app.run(threaded=True, debug=True, host='0.0.0.0')
