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

### Usage instructions

Place the relevant model weights in the [weights](./weights) directory.  These can be obtained by running [download_weights.sh](./download_weights.sh) script.
``` python
from ensemble import Ensemble
submit = Ensemble()
prediction = submit.inference(video_path)
