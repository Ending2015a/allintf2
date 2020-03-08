# Pix2Pix

Tensorflow 2.0 implementation
Implemented Date: 2020/03/01

Paper: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
Authors: Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros


## Results:

#### Generated from training images (100 epochs)
![](https://raw.githubusercontent.com/Ending2015a/allintf2/master/cg/pix2pix/samples/final_train.png)

#### Generated from testing images (100 epochs)
![](https://raw.githubusercontent.com/Ending2015a/allintf2/master/cg/pix2pix/samples/final_test.png)

## Requirements:
```
tensorflow-gpu>=2.2.0
dill
matplotlib
``` 

## How to Use:
* Trainv
```
./train
```
* Test
```
./test
```
* Inference
```
./inference
```
