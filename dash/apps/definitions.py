import os

# Local apps directory
APPS_DIR = os.path.dirname(os.path.abspath(__file__))

# Local app directory
REPO_DIR = os.path.split(APPS_DIR)[0]

# Data directory
DATA_DIR = os.path.join(REPO_DIR, 'data')

# S3 Bucket Name
BUCKET_NAME = 'ff-inbound-videos'

# FakeFinder API URL
FF_URL = 'http://192.168.105.56:5000/fakefinder/'

