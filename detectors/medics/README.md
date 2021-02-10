## Team: TheMedics
This folder contains the neceseray code to run inference using the DeepFake Detection (DFDC) solution by team TheMedics.  For more details on the project please visit the GitHub [repository](https://github.com/jphdotam/DFDC).

### Requirements

The following python packages are required:

- facenet_pytorch==2.5.1
- opencv_python==4.2.0.34
- numpy==1.18.1
- torchvision==0.5.0
- scipy==1.4.1
- scikit_image==0.18.1
- pretrainedmodels==0.7.4
- Cython==0.29.21
- decord==0.4.2
- einops==0.3.0
- faiss==1.5.3
- lmdb==1.0.0
- mmcv==1.2.6
- mmdet==2.9.0
- scikit_learn==0.24.1
- pandas==1.0.3
- albumentations==0.5.2 

They can be installed by running `pip install -r requirements.txt`

Additionally, `opencv` requires the following system libraries `libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1` and `mmcv` requires `build-essential`

### Using Docker

To start a container that has all the requirements for using the model first build the image by running the following command in this directory:

```
docker build -t medics_i .
```
To use GPUs in the container you will need to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).  To run the models, a directory with the pretrained model weights  will need to be mounted in the container's `workdir/weights` directory.  

```
docker run --runtime=nvidia -it -v <path to weight directory>:/workdir/weights  medics_i
```  

### Usage instructions

Place the relevant model weights in the [weights](./weights) directory.
``` python
from ensemble import Ensemble
submit = Ensemble()
prediction = submit.inference(video_path)
```
