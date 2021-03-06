# FastCapsNet
An optimized implementation of the CapsNet from GRAM-AI that do not required TorchNet and Visdom. 

## Requirements
The present package is written in **Python 3.8**. In order to run a full capacity, the user should have an **Nvidia GPU** with **CUDA 10.2** installed. Also, the following package are required to execute our test.
```
-Pytorch 1.6
-Torchvision 0.7
-TQDM
```

## Major optimization
1) In the primary capsule we replace the list of convolution by a single convulation with num_capsules times more out_channels and we reshape it.
2) In the high capsules we replace tranpose the num_capsules dimension with the vector dimension of the prior vector.
3) In the high capsules we directly used the softmax function of pytorch.
