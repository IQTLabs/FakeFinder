import requests
import json

def test_batch_mode_selimsef():
    url = 'http://localhost:5000/fakefinder/'

    # Additional headers.
    headers = {'Content-Type': 'application/json' }

    # Body
    payload = {"batchMode": True,
               "alwaysOn": False,
               "s3Location": ["s3://ff-inbound-videos/4000.mp4", "s3://ff-inbound-videos/4001.mp4", "s3://ff-inbound-videos/4002.mp4", "s3://ff-inbound-videos/4003.mp4", "s3://ff-inbound-videos/4004.mp4", "s3://ff-inbound-videos/4005.mp4", "s3://ff-inbound-videos/4006.mp4", "s3://ff-inbound-videos/4007.mp4", "s3://ff-inbound-videos/4008.mp4", "s3://ff-inbound-videos/4009.mp4"],
               "modelName": "selimsef",
               "splitRequests": True,
               "numSplitRequests": 2,
              }

    # convert dict to json string by json.dumps() for body data. 
    resp = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200

    # print response full body as text
    print(resp.json())

def test_batch_list_mode_selimsef():
    url = 'http://localhost:5000/fakefinder/'

    # Additional headers.
    headers = {'Content-Type': 'application/json' }

    # Body
    payload = [{"batchMode": True,
               "alwaysOn": False,
               "s3Location": ["s3://ff-inbound-videos/4000.mp4", "s3://ff-inbound-videos/4001.mp4", "s3://ff-inbound-videos/4002.mp4", "s3://ff-inbound-videos/4003.mp4", "s3://ff-inbound-videos/4004.mp4", "s3://ff-inbound-videos/4005.mp4", "s3://ff-inbound-videos/4006.mp4", "s3://ff-inbound-videos/4007.mp4", "s3://ff-inbound-videos/4008.mp4", "s3://ff-inbound-videos/4009.mp4"],
               "modelName": "selimsef",
               "splitRequests": True,
               "numSplitRequests": 2,
              }]

    # convert dict to json string by json.dumps() for body data. 
    resp = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200

    # print response full body as text
    print(resp.json())
