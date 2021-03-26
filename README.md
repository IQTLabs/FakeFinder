# FakeFinder: Sifting out deepfakes in the wild
The FakeFinder project builds upon the work done at IQT Labs in competing in the Facebook Deepfake Detection Challenge (DFDC).  FakeFinder builds a modular, scalable and extensible framework for evaluating various deepfake detection models. The toolkit provides a web application as well as API access for integration into existing media forensic workflow and applications. To illustrate the functionality in FakeFinder we have included implementations of six existing, open source Deepfake detectors as well as a [template](./detectors/detector_template/) exemplifying how new algorithms can be easily added to the system.  

<img src="./images/Fake_video_inference.gif" width="600" />

## Table of contents
1. [Overview](#overview)
2. [Available Detectors](#detectors)
3. [Reproducing the Tool](#building)
4. [Usage Instruction](#usage)

## Overview <a name="overview"></a>

We have included [instructions](#building) to reproduce the system as we have built it, using the [AWS](https://aws.amazon.com/) ecosystem (EC2, S3, EFS and ECR).  The current tool accomodates two possible workflows:
### Small jobs: response time
The default behavior when using the Dash-App.  This work flow prioritizes availability by using **warm** (existing BUT stopped EC2 instances) or **hot** (existing AND running EC2 instances) virtual machines to run the inference on videos to be queried. 
<img src="./images/small_jobs.png" alt="drawing" width="750"/>

### Large jobs: scalability
This is the default behavior when calling the system through the API and is intended for cases when a large amount of files need to be queried.  In this workflow you can specify the number of replicas of each worker and split the files to be tested between them.  This workflow leverages **cold** (don't currently exist) virtual machines.  Using pre-built images on a container registry we can scale the inference workflow to as many instances as is required, accelerating the number of files analyzed per second at the cost of a larger start-up time.
<img src="./images/batch_jobs.png" alt="drawing" width="575"/>

## Available Detectors <a name="detectors"></a>

Although designed for extensability, the current toolkit includes implementations for six detectors open sourced from the [DeepFake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge)(DFDC) and the [DeeperForensics Challenge 2020](https://competitions.codalab.org/competitions/25228)(DFC).  
  The detectors included are:
  
| Name      | Input type | Challenge | Description |
| ----------- | ----------- | ----------- | ----------- |
| [selimsef](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/selimsef)      | video (mp4)       |  DFDC<sup>1  | [Model Card](https://github.com/IQTLabs/FakeFinder/blob/readme_work/model_cards/SelimsefCard.pdf) |
| [wm](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/wm)   | video (mp4)        |  DFDC<sup>1  | [Model Card](https://github.com/IQTLabs/FakeFinder/blob/readme_work/model_cards/WMCard.pdf) |
| [ntech](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/ntech)   | video (mp4)        |  DFDC<sup>1  | [Model Card](https://github.com/IQTLabs/FakeFinder/blob/readme_work/model_cards/NtechCard.pdf) |
| [eighteen](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/eighteen)   | video (mp4)        |  DFDC<sup>1  | [Model Card](https://github.com/IQTLabs/FakeFinder/blob/readme_work/model_cards/EighteenCard.pdf) |
| [medics](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/medics)   | video (mp4)        |  DFDC<sup>1  | [Model Card](https://github.com/IQTLabs/FakeFinder/blob/readme_work/model_cards/MedicsCard.pdf) |
| [boken](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/boken)   | video (mp4)        |  DFC<sup>2  | Model Card |

Additionally, we have included template code and instructions for adding a new detector to the system in the [detector template folder](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/detector_template).

As part of the inplementation we have evaluated the current models against the test sets provided by both ([1](https://ai.facebook.com/datasets/dfdc/), [2](https://github.com/EndlessSora/DeeperForensics-1.0/tree/master/dataset)) competitions after they closed. The following figure shows the True Positive Rate (TPR), False Positive Rate (FPR) and final accuracy (Acc) for all six models against these data.  We have also included the average binary cross entropy (LogLoss) whcih was ultimately used to score the competition.

<img src="./images/all_results.png" alt="drawing" width="900"/>

We have also measured the correlation between the six detectors over all of the evaulation dataset, shown in the following figure (Note: a correlation > 0.7 is considered a strong correlation)

<img src="./images/correlations.png" alt="drawing" width="500"/>

## Reproducing the Tool <a name="building"></a>

We built FakeFinder using several components of the AWS ecosystem:

1.  [Elastic File System](https://aws.amazon.com/efs/) (EFS) for storing model weights in a quickly accessible format.

2.  [Elastic Compute Cloud](https://aws.amazon.com/ec2/?ec2-whats-new.sort-by=item.additionalFields.postDateTime&ec2-whats-new.sort-order=desc) (EC2) instances for doing inference, as well as hosting the API server and Dash app.

3.  [Simple Storage Service](https://aws.amazon.com/s3/) (S3) for storing videos uploaded using the API server or Dash app and accessed by the inference instances.
4.  [Elastic Container Registry](https://aws.amazon.com/ecr/) for storing container images for each detector that can so they can be easily deployed to new instances when the tool is operating in the scalable mode.

We also made use of Docker for building and running containers, and Flask for the API server and serving models for inference.  Here we provide instructions on reproducing the FakeFinder architecture on AWS. Also, AWS command line interface will be used in multiple steps, so it must be installed first (instructions [here](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)).  All of the following steps should be done in the same AWS account.

### Setting up the S3 Bucket

You will need to create an S3 bucket that will store videos that are uploaded by the Dash app or API server, and store the bucket name as `S3_bucket_name`.

### Model Weights and EFS directory

Model weights are available in an S3 bucket named `ffweights` located in `us-east-1`.  To obtain the weights, run the following command in the location where you want them downloaded.

```
aws s3 sync <path_to_weight_directory> s3://ffweights
```
This will create a top level directory called `weights`, with sub-directories for each detector.  This can either be done on every EC2 instance that will be hosting a detector container, or the weights can be downloaded into an EFS directory that is mounted on each EC2 instance.  See [here](https://docs.aws.amazon.com/efs/latest/ug/wt1-getting-started.html) for instructions on creating a file system on Amazon EFS.  You will have to mount the file system at least once to download the weights onto it, and take note of the file system's IP address.


### Docker Images and Elastic Container Registry

This repo contains Dockerfiles for the Dash App, API server, and each of the implemented detectors.  You will need to clone this repository, build images for each detector, and push them to an AWS ECR as part of reproducing FakeFinder.  Follow the instructions [here](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-console.html) to create an image repository on AWS ECR, and take note of your repository's URI.  You will also need to login and authenticate your docker client before pushing images.

To build images for each of the detectors do the following steps for each detector:

Switch to the directory for the given detector 
```
cd <path_to_repository>/FakeFinder/detectors/${DETECTOR_NAME}
```
where `DETECTOR_NAME` is the name of the detector (listed above).  Inside `app.py` in that directory, the variable `BUCKET_NAME` is defined (usually on line 12).  Replace the default value with the name of the S3 bucket you created and save the changes.

Next, run the following command
```
docker build -t ${DETECTOR_IMAGE_NAME} .
``` 
Where `DETECTOR_IMAGE_NAME` is a descriptive name for the image (usually just the same as `DETECTOR_NAME`).  Next, tag the image you just built so it can be pushed to the repository:

```
docker tag ${DETECTOR_IMAGE_NAME}:latest ${ECR_URI}:${DETECTOR_IMAGE_NAME}
```
And then push the image to your ECR repository:

```
docker push ${ECR_URI}:${DETECTOR_IMAGE_NAME}
```
#TODO mona:  Fill in where users update the new image names (models.json?) and ECR URI.

You can also test these images locally.  To use GPUs in these containers you will need to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker), and utilize the NVIDA runtime when running the container.  By default, containers run from these images start a Flask app that serves the detector for inference, but you can overwrite this with the `--entrypoint` flag.  The following command will run a container for the specified detector and launch a bash shell within it.

```
docker run --runtime=nvidia -v <path_to_weight_directory>/weights/${DETECTOR_NAME}/:/workdir/weights --entrypoint=/bin/bash -it -p 5000:5000 ${DETECTOR_IMAGE_NAME}
```
### Configuring AWS EC2 instances and IAM roles

Once you have setup your S3 Bucket, EFS directory, and built and pushed all the model images to your ECR you can configure EC2 instances for FakeFinder's warm/hot operating mode.  Instructions for reproducing the cold mode can be found in the `/api` directory.  We use `g4dn.xlarge` instances running the Deep Learning AMI (Ubuntu 18.04) Version 41.0 for inference.

* Instances will need to be assigned an IAM role with AmazonEC2ContainerRegistryReadOnly and AmazonS3ReadOnlyAccess
* Instances will need to be assigned a security group with the following inbound and outbound rules.
  1.  Inbound SSH access on port 22
  2.  Inbound TCP access on port 5000
  3.  Allow all outbound traffic on all ports.

You will need to initialize an EC2 instance and execute the following steps once for each detector.

1.  Create a directory and mount your EFS drive with the weights into it.
```
mkdir /data

sudo mount -t nfs4 -o nfsvers=4.1, rsize=1048576, wsize=1048576, hard, timeo=600, retrans=2, noresvport ${EFS_IP}:/ /data
```

2.  Add the EFS drive to `/etc/fstab` to make it persist over reboots

```
sudo chmod a+w /etc/fstab

echo "${EFS_IP}:/ /data  nfs     nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport     0       0" >> /etc/fstab

sudo chmod a-w /etc/fstab
```
3.  Pull the image from your ECR
```
docker pull ${ECR_URI}:${DETECTOR_IMAGE_NAME}
```
4.  Run the container with the `--restart unless-stopped` option so that the container automatically starts when the instance does.

```
docker run --runtime=nvidia --restart unless-stopped -v /data/weights/${DETECTOR_NAME}/:/workdir/weights  -d -p 5000:5000 ${DETECTOR_IMAGE_NAME}
```

5.  Stop the instance, but _not_ the container
```
sudo shutdown -h
```
6.  Take note of the instance ID and replace the value in `FakeFinder/api/models.json` with the key corresponding to the detector.

#TODO Mona:  Instructions for starting the API server, what IAM roles for the API server should look like, and changes to api.py for people reproducing the tool.  Put that in the api directory readme


### Setting up the Dash App

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

You should then be able to point a browser to the dash app's IP address and access the web application.


## Usage Instructions <a name="usage"></a>

### Using the Dash app

#TODO zig

### Using the API layer

#TODO Mona
