# LearnOpenCV

Learning OpenCV By Official Documents

系统过一遍官方文档：[https://docs.opencv.org/4.x/index.html](https://docs.opencv.org/4.x/index.html)，个人笔记，因此会很跳跃，其中简单的使用不会提，比如正常的读存图片、数据格式等等。

20240719 更新：先不对各个函数做具体详细说明，重点关注各个函数的作用是什么？我要解决某个问题时应该使用什么函数？

第一章：基础知识

- [X] [滤波函数](./docs/1.1.md)
- [X] [金字塔函数](./docs/1.2.md)
- [X] [形态学操作](./docs/1.3.md)
- [X] [变换矩阵](./docs/1.4.md)
- [X] [阈值处理](./docs/1.5.md)
- [X] [直方图](./docs/1.6.md)
- [X] [二维几何](./docs/1.7.md)
- [ ] [模板匹配](./docs/1.8.md)
- [ ] [杂项](./docs/1.misc.md)

第二章：特征提取、目标识别

- [X] [特征点三部曲：提取；描述；匹配](./docs/2.1.md)
- [X] [更详细的角点函数](./docs/2.2.md)
- [ ] [特别的特征：边缘检测](./docs/2.3.md)
- [ ] [霍尔变换检测直线和圆](./docs/2.4.md)

第三章：前后景分离

- [X] [GrabCut, WaterShed](./docs/3.1.md)
- [ ] [视频帧: BackGroundSubtractor](./docs/3.2.md)
- [ ] [Alpha matting]()

第四章：图像配准

## TODO

1.7 中还有 createGeneralizedHoughBallard, createGeneralizedHoughGuil

https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html 中有一个 phaseCorrelate 函数

内部的原理：

| 章节 | 知识点                                   |
| ---- | ---------------------------------------- |
| 1.1  | RANSAC                                   |
| 1.1  | pyrMeanShiftFiltering                    |
| 1.3  | 形态学更多应用                           |
| 1.6  | EMD距离指标                              |
| 1.7  | findContoursLinksRuns 内部原理           |
| 2.1  | 各个特征点提取、描述、匹配具体的内部原理 |
| 2.2  | 亚像素角点                               |
| 3.1  | GrabCut                               |

函数具体的细节：

- HoughCircles
