# CRNCIS-Pytorch
Context-refined neural cell instance segmentation, ISBI2019

Please cite this article as:
Jingru Yi, Pengxiang Wu, Qiaoying Huang, Hui Qu, Daniel J. Hoeppner, Dimitris N. Metaxas, Context-refined neural cell instance segmentation, ISBI (2019)

# Dependencies
Ubuntu 14.04, python 3.6.4, pytorch 0.4.1, opencv-python 3.4.1.15  

# Implementation Details
To accelerate the training process, we trained the detection and segmentation modules separately. In particular,the weights of the detection module are frozen when training the segmentation module.