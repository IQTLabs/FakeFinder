## Team: WM
This folder contains the neceseray code to run inference using the DeepFake Detection (DFDC) solution by team WM.  For more details on the project please visit the GitHub [repository](https://github.com/cuihaoleo/kaggle-dfdc).

### Requirements

Python requirements can be found in `requirements.txt` and installed with

```
pip install -r requirements.txt
```
`opencv` requires the following system libraries `libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1`

### External dependencies
External code dependencies are provided as git submdules in the [external](./external) directory.  To fetch these run:
```bash
git submodule init && git submodule update
```
### Using Docker

To start a container that has all the requirements for using the model first build the image by running the following command in this directory:

```
docker build -t wm_i .
```
To use GPUs in the container you will need to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).  To run the models, a directory with the pretrained model weights  will need to be mounted in the container's `workdir/weights` directory.  

```
docker run --runtime=nvidia -it -v <path to weight directory>:/workdir/weights  wm_i
```  

### Usage instructions

Place the relevant model weights in the [weights](./weights) directory.
``` python
from ensemble import Ensemble
submit = Ensemble()
prediction = submit.inference(video_path)
```
