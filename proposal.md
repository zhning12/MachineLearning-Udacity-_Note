猫狗大战开题报告
========
---

项目背景
----
---
猫狗大战是2013年kaggle上举行的一个娱乐性竞赛项目[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition#description)。它是一个深度学习的图像二分类问题，需要根据图片输入将猫和狗区分开来。由于我在接下来的学习中将会选择图像处理作为进一步方向，所以我选择了这个项目来加深自己对深度学习和图像识别的认知。


---

问题描述
----
---
这个项目会使用深度学习方法识别一张图片是猫还是狗。

- 输入：一张彩色图片
- 输出：是猫还是狗

---

数据集和输入
------

---

猫狗大战的数据集可以从kaggle官网上下载：

 - [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

这里共有25000张图片，其中猫、狗图片各占一半，即各有12500张图片是猫，12500张图片是狗。并且其中会有一定的异常数据，即难以分辨的数据，这些数据需要尽可能使用预训练模型进行处理（删除），以免其对训练结果造成影响，举几个可能是异常数据的例子：

<img  src="https://img-blog.csdn.net/20180628154929571?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pobmluZzEyTA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70"   height="200px" >

另外，还有一个比较好的数据集可以作为扩充数据集或是做检测/分割问题：

 - [The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)

<img  src="https://img-blog.csdn.net/20180509184154616?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pobmluZzEyTA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70"   height="200px"><img  src="https://img-blog.csdn.net/20180509184141678?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pobmluZzEyTA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70"  height="200px">

这里应该先对图片中的猫、狗的图片作数据标注，然后打乱他们的次序进行学习和训练。
kaggle上下载的数据分为train和test两部分，其中train部分的数据用来训练模型，test部分的数据在训练完成后用来测试模型。test包含12500张图片，但是这些图片并没有进行分类。

---

解决方法描述
----
---

在这个项目中，我准备使用TensorFlow, keras，OpenCV来完成该项目，其中会使用aws来更快的训练数据。

可能会尝试的模型：

- VGG16
- ResNet50
- Xception
- Inception V3

分别简单介绍下其中两个模型：

 

 1.  VGG[1]：于2014年夺得ImageNet的定位第一和分类第二，在用于解决ImageNet中的1000类图像分类和定位问题的比赛中，深度最深的两组16和19层的VGGNet网络模型在分类和定位任务上的效果最好。VGG的特点：
	-  小卷积核。作者将卷积核全部替换为3x3（极少用了1x1）；
	-  小池化核。相比AlexNet的3x3的池化核，VGG全部为2x2的池化核；
	-  层数更深特征图更宽。基于前两点外，由于卷积核专注于扩大通道数、池化专注于缩小宽和高，使得模型架构上更深更宽的同时，计算量的增加放缓；
	-  全连接转卷积。网络测试阶段将训练阶段的三个全连接替换为三个卷积，测试重用训练时的参数，使得测试得到的全卷积网络因为没有全连接的限制，因而可以接收任意宽或高为的输入。
 2. Inception V3：Inception为Google开源的CNN模型，是基于大型图像数据库ImageNet中的数据训练而成。
  - 没有大的卷积核，用两个3x3代替5x5，两个3x1代替3x3。其计算成本远低于VGGNet或更高性能的其他模型。
  - 但是，Inception架构的复杂性使得更难以对网络进行更改，如果单纯的放大架构，大部分的计算收益会立即丢失。

---

评估标准
----
- 这里会把前面train的数据按照4:1分为训练集和验证集，当训练结束后会用验证集来检测训练的结果。然后再在测试集中得到评分来作为对模型性能的评估标准。
- 使用LogLoss函数来作为评估指标：

	$$LogLoss=-\frac{1}{n}\sum\limits_{i=1}^{n}[y_ilog(\hat{y_i})+(1-y_i)log(1-\hat(y_i)]$$
> $\hat(y)$：预测结果
y：正确归类的图片
n：样本个数

- LogLoss损失函数的分值越低，则意味着模型的表现越好
- 此次毕业项目的通过标准是：在kaggle猫狗大赛中能够进入前10%

---


项目设计
----

- 项目所给出的图片中，各个图片的尺寸大小不一，因此在使用过程中应该首先统一图片尺寸
- keras中提供了在ImageNet中进行预训练过的模型Xception预训练模型，使用迁移学习模型如VGG16、ResNet50, Inception V3等进行训练和模型融合。
- 训练前需要先对训练集中的异常数据进行清洗，大概挑选后会发现有一些图片中猫狗难以辨识，甚至是完全没有猫狗。这里在上面也进行了展示。
- 清洗完异常数据后就，将train的数据分为训练集和验证集，可以用迁移学习模型对训练集进行训练了。这些模型因为已经在相当大的图片数据中进行了训练，因而可以提高新模型的学习效率。
- 选取表现最好的模型
- 提交kaggle获取分数，检查是否符合要求

---
引用
----
[1]深度学习VGG模型核心拆解
https://blog.csdn.net/qq_40027052/article/details/79015827