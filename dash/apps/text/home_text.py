


markdown_text_intro = '''
DeepFakes are becoming a problem to our national interests.
We're here to help address it.

## The Toolkit

The FakeFinder toolkit makes accessible the top five winning models to Facebook's
DeepFake Detection Challenge (DFDC), which are listed and described below.
Users can upload a video (or point to a directory containing many videos),
choose from any or all of the deepfake detection models,
and FakeFinder will output relevant scores & metrics for deepfake identification.

Provided a video file (depicted below), FakeFinder will extract the video and audio
for ingestion into separate inference models.
The video frames are fed first into an inference pipeline, with some model dependent 
preprocessing steps.
The resulting imagery is then fed to the requested inference models and confidence
scores are returned to the user.

'''


markdown_text_body = '''

## DeepFake Detection

### The Problem
Generation and manipulation of media using machine learning 
have advanced significantly over the past six years 
because of developments and improvements in generative adversarial 
networks (GAN).
While originally designed for image synthesis, GANs have since
been used for other data modalities such as 
audio and text.

As a result of both their the broad applicability, GAN have also 
gained notoriety as capable of distorting or generating media with potential
for significant impact on consumers of that media.
A factor further confounding such concerns is the ease with which 
available software tools can be used by non-experts.
Of course, the existence of manipulated media intended to deceive is 
not a new phenomenon.
However, the machine learning methodology used to do so is 
constantly evolving and improving, necessitating the development of 
tools aimed at identifying when such methods have been used.


### The Models
Since the completion of the DFDC, the organizers have released
details of both the dataset and large meta-analysis of the submitted
solutions.
The top five submissions which are included in FakeFinder are described as follows:
 * First place used a different face detection algorithm ([MTCNN](http://arxiv.org/abs/1604.02878)) and an [EfficientNet](http://arxiv.org/abs/1905.11946) for feature encoding. The authors removed structured parts of faces during training as a form of data augmentation.
 * Second place used the Xception architecture for frame-by-frame feature extraction, and a [WS-DAN](http://arxiv.org/abs/1901.09891) model for augmentation.
 * Third place used an ensemble of EfficientNets in addition to using data augmentation during training.
 * Foruth place used an ensemble of different video models: EfficientNet, Xception, ResNet, and a [SlowFast](http://arxiv.org/abs/1812.03982) video-based network.
 * Fifth place also used MTCNN for face detection combined with an ensemble of seven models, including three 3-D Convolutional Neural Networks.


### The Dataset

For this toolkit, we focused on the publicly available
dataset provided by Facebook, i.e. the 
[DeepFake Detection Challenge](https://ai.facebook.com/datasets/dfdc/) dataset.
This represents the largest publicly available face swap video
dataset, with over 100k total clips sourced from nearly 3.5k paid actors,
with both video and audio shot in a variety of natural settings.
The source data consisted of 
* more than 48k videos shot in high definition
* an average video length of roughly 70 seconds
* a total of more that 38 days worth of footage.

The DFDC source media was manipulated using a series of state-of-the-art
(SOTA) techniques to swap both the imagery and/or the audio.
Faces were swapped using a series of previous works, such as DeepFake Autoencoder (DFAE),
a frame-based morphable mask (MM) model \cite{mmdeepfake}, a Neural Talking Heads (NTH) \cite{nth} model,
Face Swapping GAN (FSGAN) \cite{fsgan} and a StyleGAN \cite{stylegan} modified to swap 
faces between a source and target frame.  A subset of the swapped faces were 
postprocessed and fed through a sharpening filter to increase their perceptual quality.

In addition to swapping faces, the dataset also includes examples where the audio was swapped 
from both real and DeepFake videos, using the TTS \cite{tts} skins voice conversion method 
to generate synthetic voices from the source transcripts.  
However, these manipulations were not ultimately considered "DeepFakes" in the scope of the competition.
Most of the details regarding the manipulation were kept hidden until after the 
competition had been closed and the final scores released.  


'''


