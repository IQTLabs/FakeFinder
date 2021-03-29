# FakeFinder Dash-App

Dash application for interacting with FakeFinder API inference system.


# Setting up the Dash App

## For server usage

In order to start the dash web application, you'll need to setup another AWS EC2 instance.
We suggest using a `t2.small` instance running Ubuntu 18.04.

* The Instance will need to be assigned an ApiNode IAM role with AmazonEC2FullAccess and AmazonS3FullAccess
* The instance will need to be assigned a security group with the following inbound and outbound rules.
  1.  Inbound SSH access on port 22
  2.  Inbound HTTP access on port 80
  3.  Inbound Custom TCP access on port 8050

Once launched, take note of the IP address and ssh into the machine.
Checkout the git repository, move into the `dash` directory. and build the docker container:
```
git clone https://github.com/IQTLabs/FakeFinder.git
cd FakeFinder/dash
```

You'll need to change the `BUCKET_NAME` and `FF_URL` variables in the `apps/definitions.py` file
to point to your bucket and API url, respectively.

After doing so, the dash app server can be built by the following commands:
```
chmod +x build_docker.sh
./build_docker.sh
```
This may take a few minutes to prep all the necessary requirements.
The build and launch will be complete when you see the following terminal output:
```
Successfully tagged dash:beta
Dash is running on http://0.0.0.0:8050/

 * Serving Flask app "app" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
```

You should then be able to point a browser to the dash app's IP address to access the web application.



## For local usage (primarily for web app debugging)

First setup a standalone virtual environment:
```
virtualenv -p /usr/local/bin/python3 venv
source venv/bin/activate
```

Next, clone the repo and install the required packages:
```
git clone https://github.com/IQTLabs/FakeFinder.git
cd dash
pip install -r requirements.txt
```

To run the app locally, one first must switch the commenting on the final two lines of `index.py`
and switch the commenting on the third and fourth lines of `config.py`.

Finally, launch the app process via `python index.py` 
and then point a browser to [http://127.0.0.1:8050/](http://127.0.0.1:8050/).
