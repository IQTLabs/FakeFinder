## Team: Selim Seferbekov 
This folder contains the neceseray code to run inference using the DeepFake Detection (DFDC) solution by team Selim Seferebekov.  For more details on the project please visit the GitHub [repository](https://github.com/selimsef/dfdc_deepfake_challenge).

### Requirements
 
- torchvision==0.5.0
- torch==1.4.0
- numpy==1.18.1
- timm==0.3.4
- opencv_python==4.2.0.34
- albumentations==0.5.2
- facenet_pytorch==2.5.1
- Flask==1.1.2
- Pillow==8.1.0

### Using Docker

To start a container that has all the requirements for using the model first build the image by running the following command in this directory:

```
docker build -t selimsef_i .
```
To use GPUs in the container you will need to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). To run the container in interactive mode use the following command.

```
docker run --runtime=nvidia -it selimsef_i
``` 

To run the models, the pretrained model weights will need to be placed in the containers `workdir/weights` directory.  This can be done by running the [download_weights.sh](./download_weights.sh) before building the image, or inside the container, or by mounting a volume containing the weights to the container at runtime:

```
docker run --runtime=nvidia -it -v <path to weight directory>:/workdir/weights  selimsef_i
```    


### Usage instructions

``` python
from ensemble import Ensemble
submit = Ensemble()
prediction = submit.inference(video_path)
