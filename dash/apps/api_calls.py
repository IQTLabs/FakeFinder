import logging
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

from .definitions import FF_URL


# Build an inference model request
def BuildInferenceRequest(filename='', model_list=[]):

    request_list = []
    for model_name in model_list:
        # Each model request takes dict form
        model_request_dict = {
                              "batchMode": False,
                              "alwaysOn": True,
                              "location": [filename],
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
        import time
        time.sleep(2)
        return [{"filename": {"0": "real_abajdarwnl.mp4"}, "wm": {"0": 0.0460696332}}, 
                {"filename": {"0": "real_abajdarwnl.mp4"}, "selimsef": {"0": 0.0113754272}}, 
                {"filename": {"0": "real_abajdarwnl.mp4"}, "medics": {"0": 0.0450550006}}, 
                {"filename": {"0": "real_abajdarwnl.mp4"}, "boken": {"0": 0.9460702622}}]
                #{"filename": {"0": "real_abajdarwnl.mp4"}, "boken": {"0": 0.0460702622}}]

    ## Make multithreaded inference request(s) to API
    inference_threads = []
    inference_results = []
    with ThreadPoolExecutor(max_workers=len(dict_list)) as executor:
        for idict in dict_list:
            inference_threads.append(executor.submit(requests.post, url=url, json=[idict]))

        for task in as_completed(inference_threads):
            task_result = task.result()
            # Removes [ ] from output string
            result_str = task_result.json()[1:-1]
            result_json = json.loads(result_str)
            inference_results.append(result_json)

    logging.info('Inference results:')
    logging.info(inference_results)
    return inference_results


def UploadFile(file_name=''):
    print(f'uploading {file_name}')
    url = urljoin(FF_URL, '/upload/')
    try:
        with open(file_name, 'rb') as f:
            files = {'file': f}
            r = requests.post(url, files=files)
    except Exception as e:
        print(f'{e}')
        logging.error(e)
        return False

    return True
