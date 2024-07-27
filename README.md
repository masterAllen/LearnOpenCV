# LearnOpenCV

Learning OpenCV By Official Documents

系统过一遍官方文档：[https://docs.opencv.org/4.x/index.html](https://docs.opencv.org/4.x/index.html)，个人笔记，因此会很跳跃，其中简单的使用不会提，比如正常的读存图片、数据格式等等。

20240719 更新：先不对各个函数做具体详细说明，重点关注各个函数的作用是什么？我要解决某个问题时应该使用什么函数？

20240722 更新：对于常用函数还是要说明，比如计算直方图这些基础函数。

第一章：基础知识

- [X] [滤波函数](./docs/1.1.md)
- [X] [金字塔函数](./docs/1.2.md)
- [X] [形态学操作](./docs/1.3.md)
- [X] [变换矩阵](./docs/1.4.md)
- [X] [阈值处理](./docs/1.5.md)
- [X] [直方图](./docs/1.6.md)
- [X] [轮廓和二维几何](./docs/1.7.md)
- [X] [模板匹配](./docs/1.8.md)
- [ ] [分类](./docs/1.9.md)
- [ ] [杂项](./docs/1.misc.md)

第二章：特征提取

- [X] [特征点三部曲：提取；描述；匹配](./docs/2.1.md)
- [X] [更详细的角点函数](./docs/2.2.md)
- [X] [HOG 特征](./docs/2.3.md)
- [X] [特别的特征：边缘检测](./docs/2.4.md)

第三章：目标分类/识别/检索

- [X] [霍尔变换检测直线和圆](./docs/3.1.md)
- [X] [常见检测：斑点、条形码二维码](./docs/3.2.md)
- [X] [从 SVM 到 BOW 词袋](./docs/3.3.md)
- [ ] [级联分类器](./docs/3.4.md)

第四章：前后景分离

- [X] [GrabCut, WaterShed](./docs/4.1.md)
- [X] [视频帧: BackGroundSubtractor](./docs/4.2.md)
- [ ] [Alpha matting]()

第五章：图像配准

- [ ] [整体配准：特征点法 or 迭代收敛](./docs/5.1.md)
- [ ] [局部配准：光流法](./docs/5.2.md)

第六章：目标追踪

- [ ] [MeanShift, CamShift](./docs/6.1.md)
- [X] [利用图像配准](./docs/6.2.md)
- [X] [光流法](./docs/6.3.md)
- [X] [Tracker](./docs/6.4.md)

<!-- 第七章：相机标定

第八章：图像拼接

第九章：图像恢复：去噪、去抖、超分
- [ ] [常见去噪]

第十章：对比度

第十章：Aruco

第九章：杂项
- [ ] [使用长短帧融合的经典 HDR 方法]
- [ ] [图像补全]
- [ ] [彩色转灰色，保持对比度]
- [ ] [图像风格化：水彩、卡通] -->

## TODO

1.7 中还有 createGeneralizedHoughBallard, createGeneralizedHoughGuil

https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html 中有一个 phaseCorrelate 函数

内部的原理：

| 章节 | 知识点                                                 |
| ---- | ------------------------------------------------------ |
| 1.1  | RANSAC                                                 |
| 1.1  | pyrMeanShiftFiltering                                  |
| 1.3  | 形态学更多应用                                         |
| 1.6  | EMD距离指标                                            |
| 1.7  | findContoursLinksRuns 内部原理                         |
| 2.1  | 各个特征点提取、描述、匹配具体的内部原理               |
| 2.2  | 亚像素角点                                             |
| 3.1  | GrabCut 原论文                                         |
| 3.2  | 每个 BackGroundSubtractor 的原理，去 OpenCV 查参考文献 |

函数具体的细节：

- HoughCircles
- HOGDescriptor 类里面的各种函数
