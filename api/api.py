from flask import Flask, render_template, jsonify, make_response
from flask_restx import Api, Resource, fields
import requests
import boto3
from botocore.exceptions import ClientError
import numpy as np
import json

app = Flask(__name__)
api = Api(app, version='1.0', title='FakeFinder API',
    description='FakeFinder API',
)

ns = api.namespace('fakefinder', description='FakeFinder operations')

client = boto3.client('ec2', region_name='us-east-1')
ec2 = boto3.resource('ec2', region_name='us-east-1')
s3_client = boto3.client('s3', region_name='us-east-1')


ffmodel = api.model('FakeFinder', {
    'batchMode': fields.Boolean(required=True, default=False, description='Set this field to true if processing video/image files in batch mode. If requests are coming from a UI this should be set to false.'),
    'alwaysOn': fields.Boolean(required=False, default=True, description='Set this field to true if starting/stopping ec2 instances'),
    's3Location': fields.String(required=True, description='Image/Video S3 location. If uploading the file ththe value should be bucket name.'),
    'modelName': fields.String(required=True, description='Name of the model to run inference against'),
    'splitRequests': fields.Boolean(required=False, default=False, description='Split the request containing a list of videos to multiple requests containing single video.'),
    'numSplitRequests': fields.Integer(required=False, default=1, description='Number of splits of the list containing videos.'),
    'uploadFile': fields.Boolean(required=False, default=False, description='Upload the file to S3 bucket if needed.'),
    'uploadFilePath': fields.String(required=False, description='Path to upload a file or directory.')
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
    print("Wait till instance starts running")
    instance.wait_until_running()
    print("Wait till instance status is ok")
    waiter = client.get_waiter('instance_status_ok')
    waiter.wait(InstanceIds=[instance_id,],)
    print("Instance private ip address")
    print(instance.private_ip_address)
    url = 'http://'+ instance.private_ip_address + ':5000/predict'
    return url

def StopAWSWarmInstance(model_name):
    instance_id = warm_instance_ids[model_name]
    response = client.stop_instances(InstanceIds=[instance_id,],)
    print(response)

    instance = ec2.Instance(instance_id)
    print("Wait till instance stops running")
    instance.wait_until_stopped()

def UploadFileToS3(file_name, bucket, object_name=None):
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True



# warm aws instances to support ui
with open("models.json") as jsonfile:
     warm_instance_ids = json.load(jsonfile)

@ns.route('/')
class FakeFinderPost(Resource):
    @ns.doc('get_fakefinder_models')
    def get(self):
        print(list(warm_instance_ids.keys()))
        return jsonify( { 'models': list(warm_instance_ids.keys()) } )


    @ns.doc('create_fakefinder_inference_task')
    @ns.expect(ffmodel)
    #@ns.marshal_with(ffmodel, code=201)
    def post(self):
        '''Create a new task'''
        # request payload can be a list or a dictionary
        print(type(api.payload))
        agg_response = []
        if type(api.payload) is list:
             # loop through the list for each of the selected models
             for r in api.payload:
                 if r['uploadFile'] is True:
                    file_name = r['uploadFilePath']
                    UploadFileToS3(file_name, r['s3Location'])
                 if r['batchMode'] is True:
                      print("implement batchmode")
                      # Bring up the cold ec2 instances
                 if r['alwaysOn'] is False:
                     url = StartAWSWarmInstance(r['modelName'])
                 else:
                     url = GetUrlFromAWSInstance(r['modelName'])

                 headers = {'Content-type': 'application/json; charset=UTF-8'}
                 # if split requests is true then send one file per request.
                 if r['splitRequests'] is True:
                    splits = r['numSplitRequests']
                    if type(r['s3Location']) is not list:
                       # convert dict to list
                       s3Location_list = list(r['s3Location'].items())
                       final = np.array_split(s3Location_list, splits)
                    else:
                       final = np.array_split(r['s3Location'], splits)
                    print(final)
                    for loc in final:
                        response = requests.post(url, json={'video_list': loc}, headers=headers)
                 else:
                    if type(r['s3Location']) is list:
                        response = requests.post(url, json={'video_list': r['s3Location']}, headers=headers)
                    else:
                        response = requests.post(url, json={'video_list': [r['s3Location']]}, headers=headers)
                 agg_response.append(response.json())
        elif type(api.payload) is dict:
             if api.payload['batchMode'] is True:
                  print("implement batchmode")
             else:
                  # request contains a single model
                  url = model_url_dict[api.payload['modelName']]
                  headers = {'Content-type': 'application/json; charset=UTF-8'}
                  print(api.payload['s3Location'])
                  # if split requests is true then send one file per request.
                  if api.payload['splitRequests'] is True:
                     if type(api.payload['s3Location']) is list:
                        for loc in api.payload['s3Location']:
                            # TODO: If uploadFile is marked true, then upload file to S3
                            print(loc)
                            response = requests.post(url, json={'video_list': [loc]}, headers=headers)
                     else:
                        # TODO: If uploadFile is marked true, then upload file to S3
                        print(api.payload['s3Location'])
                        response = requests.post(url, json={'video_list': [api.payload['s3Location']]}, headers=headers)
                  else:
                     # TODO: If uploadFile is marked true, then upload file to S3
                     if type(api.payload['s3Location']) is list:
                        response = requests.post(url, json={'video_list': api.payload['s3Location']}, headers=headers)
                     else:
                       response = requests.post(url, json={'video_list': [api.payload['s3Location']]}, headers=headers)
        print(response.json())
        #StopAWSWarmInstance(r['modelName'])
        #print(response.text)
        #return DAO.create(api.payload), 201
        #return response, 201
        print(json.dumps(agg_response))
        return make_response(jsonify(json.dumps(agg_response)), 200)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
