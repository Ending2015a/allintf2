# CycleGAN

Tensorflow 2.0 implementation<br>
Implemented Date: 2020/03/07

Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)<br>
Authors: Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros

## Results
* UNet256 (187 epochs)
![](https://github.com/Ending2015a/allintf2/blob/master/cg/CycleGAN/samples/unet256_187.png)

* ResNet9Blocks (125 epochs)
![](https://github.com/Ending2015a/allintf2/blob/master/cg/CycleGAN/samples/resnet9_125.png)


## Requirements
```
tensorflow>=2.2.0
dill
matplotlib
```

## How to Use:
* Train
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
