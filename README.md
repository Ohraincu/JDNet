# JDNet：Joint Self-Attention and Scale-Aggregation for Self-Calibrated Deraining Network

[Cong Wang](https://supercong94.wixsite.com/supercong94)\*, [Yutong Wu](https://github.com/Ohraincu)\*, [Zhixun Su](http://faculty.dlut.edu.cn/ZhixunSu/zh_CN/index/759047/list/index.htm) †, Junyang Chen <\* Both authors contributed equally to this research. † Corresponding author.>

This work has been accepted by ACM'MM 2020. [\[Arxiv\]](https://arxiv.org/abs/2008.02763) 

## Abstract

In the field of multimedia, single image deraining is a basic preprocessing work, which can greatly improve the visual effect of subsequent high-level tasks in rainy conditions. In this paper, we propose an effective algorithm, called JDNet, to solve the single image deraining problem and conduct the segmentation and detection task for applications. Specifically, considering the important information on multi-scale features, we propose a Scale-Aggregation module to learn the features with different scales. Simultaneously, Self-Attention module is introduced to match or outperform their convolutional counterparts, which allows the feature aggregation to adapt to each channel. Furthermore, to improve the basic convolutional feature transformation process of Convolutional Neural Networks (CNNs), Self-Calibrated convolution is applied to build long-range spatial and inter-channel dependencies around each spatial location that explicitly expand fields-of-view of each convolutional layer through internal communications and hence enriches the output features. By designing the Scale-Aggregation and Self-Attention modules with Self-Calibrated convolution skillfully, the proposed model has better deraining results both on real-world and synthetic datasets. Extensive experiments are con- ducted to demonstrate the superiority of our method compared with state-of-the-art methods.

<div align=center>
<img src="https://github.com/Ohraincu/JDNet/blob/master/fig/overall.png" width="80%" height="80%">

Fig：The architecture of Joint Network for deraining (JDNet).
</div>

## Requirements

- CUDA 9.0
- Python 3.6 (or later)
- Pytorch 1.1.0
- Torchvision 0.3.0
- OpenCV

## Dataset

* Rain12  [[dataset](http://yu-li.github.io/paper/li_cvpr16_rain.zip)]
* Rain100L [[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
* Rain100H [[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
* Rain1200 [[dataset](https://github.com/hezhangsprinter/DID-MDN)] 
* Real-world images (waiting for update!)

## Training
All training and testing experiments are in a folder called [code](https://github.com/Ohraincu/JDNet/tree/master/code).
```
cd ./code
```
|--code  

    |--ablation  
        |--r1  
            |--config 
            |--models
        |--r2
            |--config 
            |--models

    |--base 
        |--rain100H 
            |--config 
            |--models
        |--rain100L
            |--config 
            |--models
        |--rain1200
            |--config 
            |--models   

    |--diff_loss
        |--mae
            |--config 
            |--models
        |--mse
            |--config 
            |--models

## Testing

### Numerical Experiments

### Visual Results



