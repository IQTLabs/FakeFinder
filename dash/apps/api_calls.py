
import botocore
from botocore.exceptions import ClientError
import boto3
import logging
import requests
import json


# Build an inference model request
def BuildInferenceRequest(filename='', bucket='', model_list=[]):

    s3_file_loc = 's3://{}/{}'.format(bucket, filename)
    request_list = []
    for model_name in model_list:
        # Each model request takes dict form
        model_request_dict = {
                              "batchMode": False,
                              "alwaysOn": True,
                              "s3Location": [s3_file_loc],
                              "modelName": model_name,
                              "splitRequests": False,
                              "numSplitRequests": 0,
                              "uploadFile": False
                             }
        # Formal request is list of dicts
        request_list.append(model_request_dict)

    logging.info('Inference request list:')
    logging.info(request_list)
    return request_list


# Function to grab model list from FF API
def GetModelList(url='', debug=False):
    # For test purposes
    if debug:
        return {'models': ['selimsef', 
                           'eighteen', 
                           'medics',
                           'boken',
                           'wm']}

    # Make request to get model names
    model_list_request = requests.get(url)
    model_list = model_list_request.json()
    return model_list


# Function to submit a inference request
def SubmitInferenceRequest(url='', dict_list=[], debug=False):
    if debug:
        return [{"filename": {"0": "real_abajdarwnl.mp4"}, "wm": {"0": 0.0460696332}}, 
                {"filename": {"0": "real_abajdarwnl.mp4"}, "selimsef": {"0": 0.0113754272}}, 
                {"filename": {"0": "real_abajdarwnl.mp4"}, "medics": {"0": 0.0450550006}}, 
                {"filename": {"0": "real_abajdarwnl.mp4"}, "boken": {"0": 0.9460702622}}]
                #{"filename": {"0": "real_abajdarwnl.mp4"}, "boken": {"0": 0.0460702622}}]

    # Make inference request to API
    inference_request = requests.post(url=url, json=dict_list)
    inference_results_str = inference_request.json()
    inference_results = json.loads(inference_results_str)

    logging.info('Inference results:')
    logging.info(inference_results)
    return inference_results


# Check if file exists in s3 bucket
def CheckFileExistsS3(file_name='', bucket=[], object_name=None):
    import time
    time.sleep(3)
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3', region_name='us-east-1')
    try:
        response = s3_client.head_object(Bucket=bucket, Key=object_name)
    except ClientError as e:
        # Not found
        logging.error(e)
        return False
    return True
    

# Upload file to s3 bucket call
def UploadFileToS3(file_name='', bucket=[], object_name=None):
    import time
    time.sleep(2)

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3', region_name='us-east-1')

    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False

    return True


