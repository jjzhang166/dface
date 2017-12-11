<div align=center>
<img src="http://affluent.oss-cn-hangzhou.aliyuncs.com/html/images/dface_logo.png" width="350">
</div>

-----------------
# DFace • [![License](http://pic.dface.io/apache2.svg)](https://opensource.org/licenses/Apache-2.0)


| **`Linux CPU`** | **`Linux GPU`** | **`Mac OS CPU`** | **`Windows CPU`** |
|-----------------|---------------------|------------------|-------------------|
| [![Build Status](http://pic.dface.io/pass.svg)](http://pic.dface.io/pass.svg) | [![Build Status](http://pic.dface.io/pass.svg)](http://pic.dface.io/pass.svg) | [![Build Status](http://pic.dface.io/pass.svg)](http://pic.dface.io/pass.svg) | [![Build Status](http://pic.dface.io/pass.svg)](http://pic.dface.io/pass.svg) |


**基于多任务卷积网络(MTCNN)和Center-Loss的多人实时人脸检测和人脸识别系统。**


[Github项目地址](https://github.com/kuaikuaikim/DFace)  

[Slack 聊天组](https://dfaceio.slack.com/)  



**DFace** 是个开源的深度学习人脸检测和人脸识别系统。所有功能都采用　**[pytorch](https://github.com/pytorch/pytorch)**　框架开发。pytorch是一个由facebook开发的深度学习框架，它包含了一些比较有趣的高级特性，例如自动求导，动态构图等。DFace天然的继承了这些优点，使得它的训练过程可以更加简单方便，并且实现的代码可以更加清晰易懂。
DFace可以利用CUDA来支持GPU加速模式。我们建议尝试linux GPU这种模式，它几乎可以实现实时的效果。
所有的灵感都来源于学术界最近的一些研究成果，例如　[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878) 和 [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)


**MTCNN 结构**　　

![mtcnn](http://affluent.oss-cn-hangzhou.aliyuncs.com/html/images/mtcnn_st.png)


** 如果你对DFace感兴趣并且想参与到这个项目中, 以下TODO是一些需要实现的功能，我定期会更新，它会实时展示一些需要开发的清单。提交你的fork request,我会用issues来跟踪和反馈所有的问题。也可以加DFace的官方Q群 681403076 也可以加本人微信 jinkuaikuai005 **

###  TODO(需要开发的功能)
- 基于center loss 或者triplet loss原理开发人脸对比功能，模型采用ResNet inception v2. 该功能能够比较两张人脸图片的相似性。具体可以参考 [Paper](https://arxiv.org/abs/1503.03832)和[FaceNet](https://github.com/davidsandberg/facenet)
- 反欺诈功能，根据光线，质地等人脸特性来防止照片攻击，视频攻击，回放攻击等。具体可参考LBP算法和SVM训练模型。
- 3D人脸反欺诈。
- mobile移植，根据ONNX标准把pytorch训练好的模型迁移到caffe2,一些numpy算法改用c++实现。
- Tensor RT移植，高并发。
- Docker支持，gpu版

## 安装
DFace主要有两大模块，人脸检测和人脸识别。我会提供所有模型训练和运行的详细步骤。你首先需要构建一个pytorch和cv2的python环境，我推荐使用Anaconda来设置一个独立的虚拟环境。


### 依赖
* cuda 8.0
* anaconda
* pytorch
* torchvision
* cv2
* matplotlib

在这里我提供了一个anaconda的环境依赖文件environment.yml，它能方便你构建自己的虚拟环境。

```shell
conda env create -f path/to/environment.yml
```

### 人脸识别和检测

如果你对mtcnn模型感兴趣，以下过程可能会帮助到你。

#### 训练mtcnn模型

MTCNN主要有三个网络，叫做**PNet**, **RNet** 和 **ONet**。因此我们的训练过程也需要分三步先后进行。为了更好的实现效果，当前被训练的网络都将依赖于上一个训练好的网络来生成数据。所有的人脸数据集都来自　**[WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)** 和 **[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)**。WIDER FACE仅提供了大量的人脸边框定位数据，而CelebA包含了人脸关键点定位数据。


* 生成PNet训练数据和标注文件

```shell
python src/prepare_data/gen_Pnet_train_data.py --dataset_path {your dataset path} --anno_file {your dataset original annotation path}
```
* 乱序合并标注文件

```shell
python src/prepare_data/assemble_pnet_imglist.py
```

* 训练PNet模型


```shell
python src/train_net/train_p_net.py
```
* 生成ＲNet训练数据和标注文件

```shell
python src/prepare_data/gen_Rnet_train_data.py --dataset_path {your dataset path} --anno_file {your dataset original annotation path} --pmodel_file {yout PNet model file trained before}
```
* 乱序合并标注文件

```shell
python src/prepare_data/assemble_rnet_imglist.py
```

* 训练RNet模型

```shell
python src/train_net/train_r_net.py
```

* 生成ONet训练数据和标注文件

```shell
python src/prepare_data/gen_Onet_train_data.py --dataset_path {your dataset path} --anno_file {your dataset original annotation path} --pmodel_file {yout PNet model file trained before} --rmodel_file {yout RNet model file trained before}
```

* 生成ONet的人脸关键点训练数据和标注文件

```shell
python src/prepare_data/gen_landmark_48.py
```

* 乱序合并标注文件(包括人脸关键点)

```shell
python src/prepare_data/assemble_onet_imglist.py
```

* 训练ONet模型

```shell
python src/train_net/train_o_net.py
```

#### 测试人脸检测
```shell
python test_image.py
```    

### 人脸对比  

@TODO 根据center loss实现人脸识别

##　测试效果  
![mtcnn](http://affluent.oss-cn-hangzhou.aliyuncs.com/html/images/dface_demoall.PNG)  


### QQ交流群(模型获取请加群)  

#### 681403076 
 
![](http://affluent.oss-cn-hangzhou.aliyuncs.com/html/images/dfaceqqsm.png)

#### 本人微信  

##### jinkuaikuai005  

![](http://affluent.oss-cn-hangzhou.aliyuncs.com/html/images/perqr.jpg)  



## License

[Apache License 2.0](LICENSE)

