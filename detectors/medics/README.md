## Team: TheMedics
This folder contains the neceseray code to run inference using the DeepFake Detection (DFDC) solution by team TheMedics.  For more details on the project please visit the GitHub [repository](https://github.com/jphdotam/DFDC).

### Requirements

The following python packages are required:
 
- facenet_pytorch==2.5.1
- albumentations==0.5.2
- torch==1.4.0
- pandas==1.0.3
- opencv_python==4.2.0.34
- numpy==1.18.1
- torchvision==0.5.0
- matplotlib==3.1.3
- scipy==1.4.1
- scikit_image==0.18.1
- pretrainedmodels==0.7.4
- tqdm==4.44.1
- Cython==0.29.21
- decord==0.4.2
- einops==0.3.0
- factory==1.2
- faiss==1.5.3
- lmdb==1.0.0
- mmcv==1.2.6
- mmdet==2.9.0
- Pillow==8.1.0
- PyYAML==5.4.1
- scikit_learn==0.24.1
- skimage==0.0
- terminaltables==3.1.0

They can be installed by running `pip install -r requirements.txt`.

Additionally, `opencv` requires the following system libraries `libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1`


### Usage instructions

Place the relevant model weights in the [weights](./weights) directory.
``` python
from ensemble import Ensemble
submit = Ensemble()
prediction = submit.inference(video_path)
```
