## Team: NTech-Lab
This folder contains the neceseray code to run inference using the DeepFake Detection (DFDC) solution by team NTech Lab.  For more details on the project please visit the GitHub [repository](https://github.com/NTech-Lab/deepfake-detection-challenge).

### Requirements
 
- albumentations==0.5.2
- numpy==1.18.1
- torchvision==0.5.0
- efficientnet_pytorch==0.6.3
- opencv_python==4.2.0.34
- matplotlib==3.1.3
- torch==1.4.0
- Flask==1.1.2

### Usage instructions

Place the relevant model weights in the [weights](./weights) directory.
``` python
from ensemble import Ensemble
submit = Ensemble(detector_weights_path, video_sequence_weights_path,
       	 	first_video_face_weights_path, second_video_face_weights_path)
prediction = submit.inference(video_path)
```