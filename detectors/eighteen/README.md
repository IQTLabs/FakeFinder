## Team: TheMedics
This folder contains the neceseray code to run inference using the DeepFake Detection (DFDC) solution by team TheMedics.  For more details on the project please visit the GitHub [repository](https://github.com/jphdotam/DFDC).

### Requirements
 
- easydict==1.8
- torch==1.1.0
- torchvision==0.3.0
- opencv_python==3.4.2.17
- albumentations==0.4.5
- numpy==1.16.4
- Pillow==7.1.2
- PyYAML==5.3.1
- tensorboardX==2.0
- pandas==1.0.4
- pydocs==0.2


To run the models, the pretrained model weights will need to be placed in the containers `workdir/weights` directory.  This can be done by running the [download_weights.sh](./download_weights.sh) before building the image, or inside the container, or by mounting a volume containing the weights to the container at runtime:

```
docker run --runtime=nvidia -it -v <path to weight directory>:/workdir/weights  selimsef_i
```    


### Usage instructions

``` python
from ensemble import Ensemble
model = Ensemble(load_slowfast_path, load_xcp_path, load_slowfast_path2, load_slowfast_path3, load_b3_path,
                 load_res34_path, load_b1_path,
                 load_b1long_path, load_b1short_path, load_b0_path, load_slowfast_path4, frame_nums,
                 cuda=pipeline_cfg.cuda)
		 
prediction = model.inference(video_path)
```
