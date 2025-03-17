# LearnOpenCV2D

Learning OpenCV **2D Part** By Official Documents

系统过一遍官方文档：[https://docs.opencv.org/4.x/index.html](https://docs.opencv.org/4.x/index.html)，个人笔记，因此会很跳跃，其中简单的使用不会提，比如正常的读存图片、数据格式等等。

20240719 更新：先不对各个函数做具体详细说明，重点关注各个函数的作用是什么？我要解决某个问题时应该使用什么函数？

20240722 更新：对于常用函数还是要说明，比如计算直方图这些基础函数。

20240808 更新：对于一些涉及到调整参数的函数，需要介绍一下原理。比如 HOG 特征、Harris 角点等。

20240820 更新：3D 部分不涉及，OpenCV 3D 的模块: ovis, rapid, rgbd, sfm, stereo, structured_light, surface_matching, viz

20240917 更新：有几个模块不再调研: superres（超分模块，但用的是光流方法，在 AI 下已经是昨日黄花，还是用 dnn 更香）、cvv（用于 Debug，但是我看了一下，传统的 debug 更好）

第一章：基础知识

- [X] [基础滤波函数](./docs/1.1.md)
- [X] [金字塔函数](./docs/1.2.md)
- [X] [形态学](./docs/1.3.md)
- [X] [形态学的应用](./docs/1.4.md)
- [X] [变换矩阵](./docs/1.5.md)
- [X] [阈值处理](./docs/1.6.md)
- [X] [直方图](./docs/1.7.md)
- [X] [二维几何和轮廓](./docs/1.8.md)
- [X] [轮廓匹配](./docs/1.9.md)
- [X] [模板匹配](./docs/1.b.md)
- [X] [图片质量评价](./docs/1.a.md)
- [X] [杂项](./docs/1.misc.md)

第二章：特征提取

- [X] [特征点三部曲：提取；描述；匹配](./docs/2.1.md)
- [X] [更详细的角点函数](./docs/2.2.md)
- [X] [HOG 特征](./docs/2.3.md)
- [X] [傅里叶描述子](./docs/2.7.md)
- [X] [特别的特征：边缘检测](./docs/2.4.md)
- [X] [特别的特征：Ridgets](./docs/2.8.md)
- [X] [特别的特征：直线特征](./docs/2.5.md)
- [X] [特别的特征：人脸特征](./docs/2.6.md)

第三章：目标分类、识别、检索

针对特定物体的检测

- [X] [霍尔变换检测直线和圆](./docs/3.1.md)
- [X] [常见检测：检测椭圆、斑点](./docs/3.c.md)
- [X] [常见检测：条形码二维码](./docs/3.2.md)
- [X] [常见检测：文字（text 模块）](./docs/3.6.md)
- [X] [常见检测：人脸识别](./docs/3.7.md)

基于数据的检测

- [X] [数据学习的开始：HOG+SVM](./docs/3.3.md)
- [X] [数据学习的提升：BoW 词袋模型](./docs/3.4.md)
- [X] [数据学习的提升：级联分类器](./docs/3.5.md)
- [X] [WalBoost 分类器（xobject 模块）（未成功）](./docs/3.8.md)
- [X] [多目标识别（dpm 模块）](./docs/3.9.md)

其他检测

- [X] [EdgeBoxes](./docs/3.a.md)
- [X] [SelectriveSearchSegmentation](./docs/3.b.md)

第四章：图片分割、前后景分离

- [X] [简单图像是分割：直接用形态学膨胀腐蚀](./docs/4.5.md)
- [X] [交互式前后背景分离：GrabCut, WaterShed, AlphaMatting](./docs/4.1.md)
- [X] [视频帧前后背景分离: BackGroundSubtractor](./docs/4.2.md)
- [X] [HfsSegment（hfs 模块）](./docs/4.3.md)
- [X] [GraphSegment（ximgproc 模块）](./docs/4.4.md)

第五章：图像配准、图片拼接

- [X] [整体配准一：feature-based（reg、ecc）](./docs/5.1.md)
- [X] [整体配准二：pixel-based（特征点、稀疏光流）](./docs/5.2.md)
- [X] [局部配准：MeshWarp](./docs/5.4.md)
- [X] [不同亮度的配准（HDR MTB）](./docs/5.3.md)
- [X] [从图像配准到图像拼接：简单介绍](./docs/5.5.md)
- [X] [从图像配准到图像拼接：细节介绍](./docs/5.6.md)

第六章：光流和目标追踪

- [X] [基础知识-光流：稀疏和稠密](./docs/6.3.md)
- [X] [追踪：MeanShift, CamShift](./docs/6.1.md)
- [X] [追踪：tracker 模块](./docs/6.4.md)
- [X] [追踪：motempl 模块](./docs/6.5.md)

第七章：相机相关

- [ ] [相机标定](./docs/7.1.md)
- [X] [ISP流程: AWB](./docs/7.2.md)
- [X] [ISP流程: CCM](./docs/7.3.md)

第八章：传统方法下的图像增强

- [X] [去噪](./docs/8.1.md)
- [X] [超分](./docs/8.2.md)
- [X] [对比度增强](./docs/8.4.md)

第零章：杂项

- [X] [使用长短帧融合的经典 HDR 方法](./docs/0.1.md)
- [X] [泊松融合，也可用于色彩或亮度更平滑的变换](./docs/0.2.md)
- [X] [图像转为不同风格](./docs/0.3.md)
- [X] [Image Hash，图像相似度](./docs/0.4.md)
- [X] [视网膜模型，可用于对比度增强、ToneMapping、运动检测](./docs/0.5.md)
- [X] [显著性检测](./docs/0.6.md)
- [X] [SuperPixels](./docs/0.7.md)
- [X] [PeiLinNormalization](./docs/0.8.md)
- [X] [Radon Transform(拉东变换)](./docs/0.9.md)
- [X] [Phase Unwarping](./docs/0.a.md)
- [X] [视频防抖：videostab 模块](./docs/0.b.md)
- [X] [Phase Correlate，相位配准](./docs/0.c.md)
- [X] [彩色转灰色时保持对比度](https://docs.opencv.org/4.x/d4/d32/group__photo__decolor.html)
- [X] [图像修复一](https://docs.opencv.org/4.x/d7/d8b/group__photo__inpaint.html) [、图像修复二](https://docs.opencv.org/4.x/de/daa/group__xphoto.html#ga1a7f584b7e6b10d830c4ac3bb12b4b73)

一些不好归纳的记录：

1. 提取确定的前景：使用 **DistanceTansform**，该函数在 [1.3](./docs/1.3.md) 中有介绍
2. 去高亮：使用 **illuminationChange**，[泊松融合](./docs/0.2.md) 那里