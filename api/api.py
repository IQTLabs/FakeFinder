from flask import Flask, render_template
from flask_restx import Api, Resource, fields
import requests

app = Flask(__name__)
api = Api(app, version='1.0', title='FakeFinder API',
    description='FakeFinder API',
)

ns = api.namespace('fakefinder', description='FakeFinder operations')

ffmodel = api.model('FakeFinder', {
    's3Location': fields.String(required=True, description='Image/Video S3 location'),
    'modelName': fields.String(required=True, description='Name of the model to run inference against'),
    'splitRequests': fields.Boolean(required=False, default=False, description='Split the request containing a list of videos to multiple requests containing single video.'),
    'uploadFile': fields.Boolean(required=False, default=False, description='Upload the file to S3 bucket if needed.')
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

model_url_dict = {
  "selimsef": "http://192.168.105.17:5000/predict",
  "eighteen": "http://192.168.105.17:5000/predict",
  "medics": "http://192.168.105.17:5000/predict",
  "ntech": "http://192.168.105.17:5000/predict",
  "wm" : "http://192.168.105.17:5000/predict"
}

@ns.route('/')
class FakeFinderPost(Resource):
    @ns.doc('create_fakefinder_inference_task')
    @ns.expect(ffmodel)
    @ns.marshal_with(ffmodel, code=201)
    def post(self):
        '''Create a new task'''
        #url = 'http://192.168.105.17:5000/predict'
        # request payload can be a list or a dictionary
        print(type(api.payload))
        if type(api.payload) is list:
             # loop through the list for each of the selected models
             for r in api.payload:
                 print(r['modelName'])
                 url = model_url_dict[r['modelName']]
                 headers = {'Content-type': 'application/json; charset=UTF-8'}
                 print(r['s3Location'])
                 # if split requests is true then send one file per request.
                 if r['splitRequests'] is True:
                    if type(r['s3Location']) is list:
                       for loc in r['s3Location']:
                           # TODO: If uploadFile is marked true, then upload file to S3
                           print(loc)
                           response = requests.post(url, json={'video_list': loc}, headers=headers)
                    else:
                       # TODO: If uploadFile is marked true, then upload file to S3
                       print(r['s3Location'])
                       response = requests.post(url, json={'video_list': r['s3Location']}, headers=headers)
        elif type(api.payload) is dict:
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
                       response = requests.post(url, json={'video_list': loc}, headers=headers)
                else:
                       # TODO: If uploadFile is marked true, then upload file to S3
                       print(api.payload['s3Location'])
                       response = requests.post(url, json={'video_list': api.payload['s3Location']}, headers=headers)
        print(response.json())
        print(response.text)
        return DAO.create(api.payload), 201


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
