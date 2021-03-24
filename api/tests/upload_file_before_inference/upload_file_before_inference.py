import requests
import json


def test_upload_file_batch_mode_no_split_ntech():

    url = 'http://localhost:5000/fakefinder/'

    # Additional headers.
    headers = {'Content-Type': 'multipart/form-data' }

    # Body
    payload = {"batchMode": False,
               "alwaysOn": False,
               "s3Location": "ff-inbound-videos",
               "modelName": "ntech",
               "splitRequests": False,
               "numSplitRequests": 0,
               "uploadFile": True,
               "uploadFilePath": "./file_test_4000.mp4"
              }

    files = {'file': ("file_test_4000.mp4", open("./file_test_4000.mp4", 'rb'), 'video/mp4')}

    # convert dict to json string by json.dumps() for body data. 
    resp = requests.post(url, files=files, headers=headers, data=payload)

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200

    # print response full body as text
    print(resp.json())

if __name__ == '__main__':
   test_upload_file_batch_mode_no_split_ntech()
'''
def test_upload_folder_batch_mode_no_split_ntech():

    url = 'http://localhost:5000/fakefinder/'

    # Additional headers.
    headers = {'Content-Type': 'application/json' }

    # Body
    payload = {"batchMode": False,
               "alwaysOn": False,
               "s3Location": "ff-inbound-videos",
               "modelName": "ntech",
               "splitRequests": False,
               "numSplitRequests": 0,
               "uploadFile": True,
               "uploadFilePath": "./test_folder"
              }

    # convert dict to json string by json.dumps() for body data. 
    resp = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 200

    # print response full body as text
    print(resp.json())
'''
