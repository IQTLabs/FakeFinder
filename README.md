# FakeFinder

## Available Detectors

| Name      | Input type |
| ----------- | ----------- |
| [selimsef](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/selimsef)      | video (mp4)       |
| [wm](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/wm)   | video (mp4)        |
| [ntech](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/ntech)   | video (mp4)        |
| [eighteen](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/eighteen)   | video (mp4)        |
| [medics](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/medics)   | video (mp4)        |
| [boken](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/boken)   | video (mp4)        |

## Usage Instructions

### Model Weights

Model weights are available in an S3 bucket named `ffweights` located in `us-east-1`.  To obtain the weights locally, install the [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) and run the following command:

```
aws s3 sync <path_to_weight_directory> s3://ffweights
```
This will create a top level directory called `weights`, with sub-directories for each detector.
### Docker Images

This repo contains Dockerfiles for the Dash App, API server, and each of the implemented detectors.  

To build images for each of the detectors, switch to the directory for the given detector and run the following command
```
docker build -t ${DETECTOR_NAME} .
``` 
where `DETECTOR_NAME` is the name of the detector (listed above).  By default, containers run from these images start a Flask app that serves the detector for inference.  To use GPUs in these containers you will need to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker), and utilize the NVIDA runtime when running the container:

```
docker run --runtime=nvidia -v <path_to_weight_directory>/weights/${DETECTOR_NAME}/:/workdir/weights  -d -p 5000:5000 ${DETECTOR_NAME}
```
### Configuring AWS EC2 instances and S3 Buckets

Currently, FakeFinder uses S3

### Using the API Server
#TODO:  Fill in with instructions from Mona.