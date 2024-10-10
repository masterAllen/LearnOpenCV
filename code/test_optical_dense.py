import numpy as np
import cv2


# # 将每个点二维光流向量转为长度和角度，然后分别赋予其强度和颜色的属性。这个没有实际含义，只是便于查看效果
# def denseflow_toimg(flow): 
#     mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#     hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
#     hsv[..., 1] = 255
#     hsv[..., 0] = ang * 180 / np.pi /2
#     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# cap = cv2.VideoCapture('./video/vtest.avi')

# # 第一张图片
# ret, old_frame = cap.read()
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# # Dense Flow
# # 有些可以直接用类似 cv2.calcOpticalFlowXXX() 的方法计算的，但还是建一个对象后调用 calc 更成体系
# # 一共有两种创建方式，直接调用 createXXX 或 XXX类的 create方法，都可以
# # 这后面的 bool 值，表示是否当前对象计算是不是直接传入灰色图片，为了方便后面调用准备的，没啥特别的...
# flows = [
#     (cv2.DISOpticalFlow.create(), True),
#     (cv2.FarnebackOpticalFlow.create(), True),
#     # (cv2.optflow.DualTVL1OpticalFlow.create(), True)
#     # (cv2.optflow.createOptFlow_SimpleFlow(), True),
#     # (cv2.optflow.createOptFlow_DeepFlow(), True),
#     # (cv2.optflow.createOptFlow_DenseRLOF(), False),
#     # (cv2.optflow.createOptFlow_PCAFlow(), True),
#     # (cv2.optflow.createOptFlow_SparseToDense(), True),
# ]

# while True:
#     ret, now_frame = cap.read()
#     if now_frame is None:
#         break

#     now_gray = cv2.cvtColor(now_frame, cv2.COLOR_BGR2GRAY)

#     for i, (nowflow, cangray) in enumerate(flows):
#         if cangray:
#             flowimg = nowflow.calc(old_gray, now_gray, None,)
#         else:
#             flowimg = nowflow.calc(old_frame, now_frame, None,)
#         # now_result = denseflow_toimg(flowimg)
#         # cv2.imshow(f'{i}', now_result)

#         print(flowimg.shape)
#         exit(0)

#     # Now update the previous frame and previous points
#     old_gray, old_frame = now_gray, now_frame

#     cv2.waitKey(1)

# cv2.destroyAllWindows()

import numpy as np
import cv2

# interpolator = cv2.ximgproc.createRICInterpolator()
interpolator = cv2.ximgproc.createEdgeAwareInterpolator()

frame1 = np.array([[255,  0], [  0,  0], [  0,255]], dtype=np.uint8)
frame2 = np.array([[  0,255], [  0,  0], [255,  0]], dtype=np.uint8)
from_points = np.array([[0,0],[1,2]], dtype=np.float32)
to_points =   np.array([[1,0],[0,2]], dtype=np.float32)

dense_flow = interpolator.interpolate(frame1, from_points, frame2, to_points)