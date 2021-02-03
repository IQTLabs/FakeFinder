## Team: NTech-Lab
This folder contains the neceseray code to run inference using the DeepFake Detection (DFDC) solution by team NTech Lab.  For more details on the project please visit the GitHub [repository](https://github.com/NTech-Lab/deepfake-detection-challenge).

### Requirements

The following python packages are required:
 
- albumentations==0.5.2
- numpy==1.18.1
- torchvision==0.5.0
- efficientnet_pytorch==0.6.3
- opencv_python==4.2.0.34
- matplotlib==3.1.3
- torch==1.4.0
- Flask==1.1.2

They can be installed by running `pip install -r requirements.txt`.

Additionally, `opencv` requires the following system libraries `libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1`

### Using Docker

To start a container that has all the requirements for using the model first build the image by running the following command in this directory:

```
docker build -t ntech_i .
```
To use GPUs in the container you will need to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).  To run the models, a directory with the pretrained model weights  will need to be mounted in the container's `workdir/weights` directory.  

```
docker run --runtime=nvidia -it -v <path to weight directory>:/workdir/weights  ntech_i
```  

### Usage instructions

Place the relevant model weights in the [weights](./weights) directory.
``` python
from ensemble import Ensemble
submit = Ensemble(detector_weights_path, video_sequence_weights_path,
       	 	first_video_face_weights_path, second_video_face_weights_path)
prediction = submit.inference(video_path)
```
