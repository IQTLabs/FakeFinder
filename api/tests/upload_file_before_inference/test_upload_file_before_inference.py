import requests
import json


def test_upload_file_batch_mode_no_split_ntech():

    url = 'http://localhost:5000/uploadS3/'

    # Additional headers.
    headers = {'accept': 'application/json'}

    files = {"file": ("file_test_4000.mp4", open("./file_test_4000.mp4", "rb"))}

    resp = requests.post('http://localhost:5000/uploadS3/?bucket=ff-inbound-videos', headers=headers, files=files)

    print(resp.json())

    # Validate response headers and body contents, e.g. status code.
    assert resp.status_code == 201

