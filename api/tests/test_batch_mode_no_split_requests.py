import requests
import json
import pytest


def test_batch_mode_boken():
    url = 'http://localhost:5000/fakefinder/'

    # Additional headers.
    headers = {'Content-Type': 'application/json' }

    # Body
    payload = {"batchMode": True, 
               "alwaysOn": False,
               "s3Location": "s3://ff-inbound-videos/4000.mp4",
               "modelName": "boken",
               "splitRequests": False,
               "numSplitRequests": 0,
              }

    # convert dict to json string by json.dumps() for body data.
    resp = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200

    # print response full body as text
    print(resp.json())

@pytest.mark.skip(reason="no way of currently testing this")
def test_batch_mode_selimsef():
    url = 'http://localhost:5000/fakefinder/'

    # Additional headers.
    headers = {'Content-Type': 'application/json' }

    # Body
    payload = {"batchMode": True,
               "alwaysOn": False,
               "s3Location": "s3://ff-inbound-videos/4000.mp4",
               "modelName": "selimsef",
               "splitRequests": False,
               "numSplitRequests": 0,
              }

    # convert dict to json string by json.dumps() for body data. 
    resp = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200

    # print response full body as text
    print(resp.json())

@pytest.mark.skip(reason="no way of currently testing this")
def test_batch_mode_medics():
    url = 'http://localhost:5000/fakefinder/'

    # Additional headers.
    headers = {'Content-Type': 'application/json' }

    # Body
    payload = {"batchMode": True,
               "alwaysOn": False,
               "s3Location": "s3://ff-inbound-videos/4000.mp4",
               "modelName": "medics",
               "splitRequests": False,
               "numSplitRequests": 0,
              }

    # convert dict to json string by json.dumps() for body data. 
    resp = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200

    # print response full body as text
    print(resp.json())

@pytest.mark.skip(reason="no way of currently testing this")
def test_batch_mode_wm():
    url = 'http://localhost:5000/fakefinder/'

    # Additional headers.
    headers = {'Content-Type': 'application/json' }

    # Body
    payload = {"batchMode": True,
               "alwaysOn": False,
               "s3Location": "s3://ff-inbound-videos/4000.mp4",
               "modelName": "wm",
               "splitRequests": False,
               "numSplitRequests": 0,
              }

    # convert dict to json string by json.dumps() for body data. 
    resp = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200

    # print response full body as text
    print(resp.json())

@pytest.mark.skip(reason="no way of currently testing this")
def test_batch_mode_eighteen():
    url = 'http://localhost:5000/fakefinder/'

    # Additional headers.
    headers = {'Content-Type': 'application/json' }

    # Body
    payload = {"batchMode": True,
               "alwaysOn": False,
               "s3Location": "s3://ff-inbound-videos/4000.mp4",
               "modelName": "eighteen",
               "splitRequests": False,
               "numSplitRequests": 0,
              }

    # convert dict to json string by json.dumps() for body data. 
    resp = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200

    # print response full body as text
    print(resp.json())
