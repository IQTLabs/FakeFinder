## Detector template
This directory includes the basic components used to incorporate a new detector into the system.  In general terms the procedure is as follows:
- Add the implementation for your specific detector and wrap it in the ensemble class.
- Build and test the docker image and container.
- Add the container to the docker-compose.yml file.

The following section covers these steps in more detail.
### Adding a new detector

1. Create a new detector folder by copying [FakeFinder/detector/detector_template](https://github.com/IQTLabs/FakeFinder/tree/template/detectors/detector_template) to FakeFinder/detectors/<your_detector>. You will be working inside this directory:
```
cd detectors/<your_detector>
```
`<your_detector>` should be a short, single-word name for your detector (e.g. selimsef or ntech).

2. Implement the new detector in this directory.  Wrap it into an ensemble class using [Ensemble.py](https://github.com/IQTLabs/FakeFinder/blob/template/detectors/detector_template/ensemble.py).  The basic ensemble class includes two methods:
    1. The ```__init__``` method should initialize the model and load any necessary weights from the local weights directory.  Ideally this ensemble class should load all necessary weights, but this can occur in step 3 if so desired.
    2. The ```inference``` method will be called to make the predictions.  This method should expect to receive the path to a video, run the forensics pipeline for that algorithm, and return the score.  Currently we assigned the following labels:
        - 1: fake/manipualted
        - 0: real.
3. Make any necesary changes to app.py.  This file will be used to initialize a flask application used by the service to respond to requests.
4. Generate a requirements file “requirements.txt” listing the packages and version required to recreate the environment used for the new detector.  For example you may use the pipreqs package:
```
pip install pipreqs
pipreqs .
```
5. Edit the [docker file](https://github.com/IQTLabs/FakeFinder/blob/template/detectors/detector_template/Dockerfile) with any additional steps needed to reproduce your working environment.  The template file we have included will:
    - Create a basic linux + python3 system
    - Install the packages in the requirements file
    - Expose port 5000 for external access to/from the API
    - Run app.py to start up the flask application
You can test the dockerfile build by replacing the last line:
```
CMD ["python3","app.py"]
```
with
```
CMD ["/bin/bash"]
```
This will change the state of the docker container to provide a shell for experimentation instead of running the flask app.  Once the container has been tested return to the original:
```
CMD ["python3","app.py"]
```
You can also override the default `python3 app.py` command by running the container with the following command:
```
docker run -it --entrypoint=/bin/bash <your_image_name> -i
```
Test to make sure that an instance of the `ensemble` class performs inference without error, and that running `app.py` initializes an instance of `ensemble` withour error.

6. Push the code to your remote repository for use later.  Make sure that your weights are available in the ./weights/<your_detector>/ directory so that it can be correctly mounted when running the container.
