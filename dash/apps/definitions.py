import os

# Local apps directory
APPS_DIR = os.path.dirname(os.path.abspath(__file__))

# Local app directory
REPO_DIR = os.path.split(APPS_DIR)[0]

# Data directory
DATA_DIR = os.path.join(REPO_DIR, 'data')

# Static directory local name
STATIC_DIRNAME = 'static'
STATIC_FULLPATH = os.path.join(APPS_DIR, STATIC_DIRNAME)
#STATIC_FULLPATH = os.path.join(os.getcwd(), 'apps', STATIC_DIRNAME)
if not os.path.exists(STATIC_FULLPATH):
    os.makedirs(STATIC_FULLPATH)

# S3 Bucket Name
BUCKET_NAME = os.getenv('S3_BUCKETNAME')

# FakeFinder API URL
FF_URL = os.getenv('FF_API_URL')

