## Team: WM
This folder contains the neceseray code to run inference using the DeepFake Detection (DFDC) solution by team Selim Seferebekov.  For more details on the project please visit the GitHub [repository](https://github.com/cuihaoleo/kaggle-dfdc).

### Requirements
 
- torch==1.4.0
- scikit_image==0.18.1
- six==1.14.0
- tqdm==4.44.1
- numpy==1.18.1
- scipy==1.4.1
- torchvision==0.5.0
- opencv_python==4.2.0.34
- absl==0.0
- bbox==0.9.2
- Cython==0.29.21
- ipython==7.19.0
- Pillow==8.1.0
- pytest==6.2.2
- skimage==0.0
- sotabencheval==0.0.38
- tensorflow==2.4.1
- Flask==1.1.2

### External dependencies
External code dependencies are provided as git submdules in the [external](./external) directory.  To fetch these run:
```bash
git submodule init && git submodule update
```

### Usage instructions

Place the relevant model weights in the [weights](./weights) directory.
``` python
from ensemble import Ensemble
submit = Ensemble()
prediction = submit.inference(video_path)
```