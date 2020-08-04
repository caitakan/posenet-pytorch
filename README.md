## Using Posenet to extract keypoints frame by frame and estimate the keyframes for Given video.

This repo is for generate the keypoints and keyframes for SimplyDance (https://github.com/caitakan/SimplyDance)

It forked from https://github.com/rwightman/posenet-pytorch, for more details, please refer back to the original readme file


### Install Posenet-Pytorch

A suitable Python 3.x environment with a recent version of PyTorch is required. Development and testing was done with Python 3.7.1 and PyTorch 1.0 w/ CUDA10 from Conda.

If you want to use the webcam demo, a pip version of opencv (`pip install python-opencv=3.4.5.20`) is required instead of the conda version. Anaconda's default opencv does not include ffpmeg/VideoCapture support. The python bindings for OpenCV 4.0 currently have a broken impl of drawKeypoints so please force install a 3.4.x version.

A fresh conda Python 3.6/3.7 environment with the following installs should suffice: 
```
conda install -c pytorch pytorch cudatoolkit
pip install requests opencv-python==3.4.5.20
```

### Usage for generate keypoints and keyframes for SimplyDance

In linux shell, install ffmpeg:

```
sudo apt update
sudo apt install ffmpeg
```

Run the keypoints and keyframe extraction, in the shell, it set to 20fps, the result write to ./video_keypoints.json ./video_keyframes.json

```
sh process_video.sh  yourvideo.mp4 ./video_keypoints.json ./video_keyframes.json
```

### Credits

Posenet-Pytorch at https://github.com/rwightman/posenet-pytorch

The original model, weights, code, etc. was created by Google and can be found at https://github.com/tensorflow/tfjs-models/tree/master/posenet

This port and my work is in no way related to Google.

The Python conversion code that started me on my way was adapted from the CoreML port at https://github.com/infocom-tpo/PoseNet-CoreML


