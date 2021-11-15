import requests
import json
import pytest

url = 'http://0.0.0.0:5000/fakefinder/'
headers = {'Content-Type': 'application/json' }

@pytest.mark.skip(reason="no way of currently testing this")
def test_batch_mode_ntech():
    # Body
    payload = {"batchMode": True,
               "alwaysOn": False,
               "s3Location": ["s3://ff-inbound-videos/4000.mp4", "s3://ff-inbound-videos/4001.mp4", "s3://ff-inbound-videos/4002.mp4", "s3://ff-inbound-videos/4003.mp4", "s3://ff-inbound-videos/4004.mp4", "s3://ff-inbound-videos/4005.mp4"],
               "modelName": "ntech",
               "splitRequests": True,
               "numSplitRequests": 2,
              }

    # convert dict to json string by json.dumps() for body data.
    resp = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200

    # print response full body as text
    print(resp.json())

@pytest.mark.parametrize('num_splits', [2, 4, 6, 10])
def test_batch_mode_boken(num_splits):
    # Body
    payload = {"batchMode": True,
               "alwaysOn": False,
               "s3Location": ["s3://ff-inbound-videos/4000.mp4", "s3://ff-inbound-videos/4001.mp4", "s3://ff-inbound-videos/4002.mp4", "s3://ff-inbound-videos/4003.mp4", "s3://ff-inbound-videos/4004.mp4", "s3://ff-inbound-videos/4005.mp4", "s3://ff-inbound-videos/4006.mp4", "s3://ff-inbound-videos/4007.mp4", "s3://ff-inbound-videos/4008.mp4", "s3://ff-inbound-videos/4009.mp4"],
               "modelName": "boken",
               "splitRequests": True,
               "numSplitRequests": num_splits,
              }

    # convert dict to json string by json.dumps() for body data.
    resp = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200

    # print response full body as text
    print(resp.json())

@pytest.mark.skip(reason="no way of currently testing this")
def test_batch_mode_medics():
    # Body
    payload = {"batchMode": True,
               "alwaysOn": False,
               "s3Location": ["s3://ff-inbound-videos/4000.mp4", "s3://ff-inbound-videos/4001.mp4", "s3://ff-inbound-videos/4002.mp4", "s3://ff-inbound-videos/4003.mp4", "s3://ff-inbound-videos/4004.mp4", "s3://ff-inbound-videos/4005.mp4"],
               "modelName": "medics",
               "splitRequests": True,
               "numSplitRequests": 2,
              }

    # convert dict to json string by json.dumps() for body data.
    resp = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200

    # print response full body as text
    print(resp.json())

@pytest.mark.skip(reason="no way of currently testing this")
def test_batch_mode_wm():
    # Body
    payload = {"batchMode": True,
               "alwaysOn": False,
               "s3Location": ["s3://ff-inbound-videos/4000.mp4", "s3://ff-inbound-videos/4001.mp4", "s3://ff-inbound-videos/4002.mp4", "s3://ff-inbound-videos/4003.mp4", "s3://ff-inbound-videos/4004.mp4", "s3://ff-inbound-videos/4005.mp4"],
               "modelName": "wm",
               "splitRequests": True,
               "numSplitRequests": 2,
              }

    # convert dict to json string by json.dumps() for body data.
    resp = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200

    # print response full body as text
    print(resp.json())

@pytest.mark.skip(reason="no way of currently testing this")
def test_batch_mode_eighteen():
    # Body
    payload = {"batchMode": True,
               "alwaysOn": False,
               "s3Location": ["s3://ff-inbound-videos/4000.mp4", "s3://ff-inbound-videos/4001.mp4", "s3://ff-inbound-videos/4002.mp4", "s3://ff-inbound-videos/4003.mp4", "s3://ff-inbound-videos/4004.mp4", "s3://ff-inbound-videos/4005.mp4"],
               "modelName": "eighteen",
               "splitRequests": True,
               "numSplitRequests": 2,
              }

    # convert dict to json string by json.dumps() for body data.
    resp = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200

    # print response full body as text
    print(resp.json())
