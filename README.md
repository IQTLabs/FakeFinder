# FakeFinder: Sifting out deepfakes in the wild
The FakeFinder project builds upon the work done at IQT Labs in competing in the Facebook Deepfake Detection Challenge (DFDC).  FakeFinder builds a modular, scalable and extensible framework for evaluating various deepfake detection models. The toolkit provides a web application as well as API access for integration into existing media forensic workflow and applications. To illustrate the functionality in FakeFinder we have included implementations of six existing, open source Deepfake detectors as well as a [template](./detectors/detector_template/) exemplifying how new algorithms can be easily added to the system.

<a name='inferencevideo'>
<img src="./images/Fake_video_inference.gif" width="600" />
</a>

## Table of contents
1. [Overview](#overview)
2. [Available Detectors](#detectors)
3. [Reproducing the Tool](#building)
4. [Usage Instruction](#usage)

## Overview <a name="overview"></a>

We have included [instructions](#building) to reproduce the system as we have built it, using [Docker](https://www.docker.com/) containers with [Compose](https://docs.docker.com/compose/) or [Kubernetes](https://kubernetes.io/).

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
| [boken](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/boken)   | video (mp4)        |  DFC<sup>2  | [Model Card](https://github.com/IQTLabs/FakeFinder/blob/readme_work/model_cards/BokenCard.pdf) |

Additionally, we have included template code and instructions for adding a new detector to the system in the [detector template folder](https://github.com/IQTLabs/FakeFinder/tree/main/detectors/detector_template).

As part of the inplementation we have evaluated the current models against the test sets provided by both ([1](https://ai.facebook.com/datasets/dfdc/), [2](https://github.com/EndlessSora/DeeperForensics-1.0/tree/master/dataset)) competitions after they closed. The following figure shows the True Positive Rate (TPR), False Positive Rate (FPR) and final accuracy (Acc) for all six models against these data.  We have also included the average binary cross entropy (LogLoss) whcih was ultimately used to score the competition.

<img src="./images/all_results.png" alt="drawing" width="900"/>

We have also measured the correlation between the six detectors over all of the evaulation dataset, shown in the following figure (Note: a correlation > 0.7 is considered a strong correlation)

<img src="./images/correlations.png" alt="drawing" width="500"/>

## Reproducing the Tool <a name="building"></a>

We built FakeFinder using Docker for building and running containers, and Flask for the API server and serving models for inference.  Here we provide instructions on reproducing the FakeFinder architecture.  There are a few prerequisites:

### GPU Host

The different detectors require the use of a GPU. We've tested against an AWS EC2 instance type of g4dn.xlarge using the Deep Learning AMI (Ubuntu 18.04) Version 52.0.

### Clone the Repository

Some of the detectors use submodules so use the following command to clone the FakeFinder repo.
```
git clone --recurse-submodules -j8 https://github.com/IQTLabs/FakeFinder
cd FakeFinder
```

### Model Weights

To access the weights for each model run the following commands:

```
mkdir weights
cd weights
wget -O boken.tar.gz https://github.com/IQTLabs/FakeFinder/releases/download/weights/boken.tar.gz
tar -xvzf boken.tar.gz
wget -O eighteen.tar.gz https://github.com/IQTLabs/FakeFinder/releases/download/weights/eighteen.tar.gz
tar -xvzf eighteen.tar.gz
wget -O medics.tar.gz https://github.com/IQTLabs/FakeFinder/releases/download/weights/medics.tar.gz
tar -xvzf medics.tar.gz
wget -O ntech.tar.gz https://github.com/IQTLabs/FakeFinder/releases/download/weights/ntech.tar.gz
tar -xvzf ntech.tar.gz
wget -O selimsef.tar.gz https://github.com/IQTLabs/FakeFinder/releases/download/weights/selimsef.tar.gz
tar -xvzf selimsef.tar.gz
wget -O wm.tar.gz https://github.com/IQTLabs/FakeFinder/releases/download/weights/wm.tar.gz
tar -xvzf wm.tar.gz
```

This will create a top level directory called `weights`, with sub-directories for each detector.

FakeFinder can now be started with either Compose or Kubernetes

### Start with Compose

```
docker-compose up -d --build
```

### Start with Kubernetes

```
TODO
```

You should then be able to point a browser to the host's IP address to access the web application.


## Usage Instructions <a name="usage"></a>

### Using the Dash App
The above [example](#inferencevideo) demonstrates using the Inference Tool section of the web app.
Users can upload a video by clicking on the *Upload* box of the *Input Video* section.
The dropdown menu autopopulates upon upload completion, and users can play the video via a series of controls.
There is the ability to change the volume, playback speed and current location within the video file.

In the *Inference* section of the page, users may select from the deep learning models available through the API.
After checking the boxes of requested models, the *Submit* button will call the API to run inference for each model.
The results are returned in the table, which includes an assignment of Real or Fake based on the model probability output,
as well as graphically, found below the table.
The bar braph presents the confidence of the video being real or fake for each model and for submissions with more than one model,
an average confidence score is also presented.
