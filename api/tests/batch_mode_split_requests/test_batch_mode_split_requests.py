import requests
import json

def test_batch_mode_ntech():
    url = 'http://localhost:5000/fakefinder/'
    
    # Additional headers.
    headers = {'Content-Type': 'application/json' } 

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

def test_batch_mode_selimsef():
    url = 'http://localhost:5000/fakefinder/'

    # Additional headers.
    headers = {'Content-Type': 'application/json' }

    # Body
    payload = {"batchMode": True,
               "alwaysOn": False,
               "s3Location": ["s3://ff-inbound-videos/4000.mp4", "s3://ff-inbound-videos/4001.mp4", "s3://ff-inbound-videos/4002.mp4", "s3://ff-inbound-videos/4003.mp4", "s3://ff-inbound-videos/4004.mp4", "s3://ff-inbound-videos/4005.mp4"],
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

def test_batch_mode_medics():
    url = 'http://localhost:5000/fakefinder/'

    # Additional headers.
    headers = {'Content-Type': 'application/json' }

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

def test_batch_mode_wm():
    url = 'http://localhost:5000/fakefinder/'

    # Additional headers.
    headers = {'Content-Type': 'application/json' }

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

def test_batch_mode_eighteen():
    url = 'http://localhost:5000/fakefinder/'

    # Additional headers.
    headers = {'Content-Type': 'application/json' }

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

