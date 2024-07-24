import cv2
import numpy as np

# 这个就是 NMS 代码，可以直接跳过不看
# 感谢文章：https://blog.csdn.net/submarineas/article/details/124512591
def NMS(boxes, threshold):
    if len(boxes) == 0:		# 边界判断，如果没有检测到任何目标，返回空列表，即不做nms
        return []
    
    boxes = np.array(boxes).astype("float")	# 将numpy中的每个元素转换成float类型

    x1 = boxes[:,0]  # 左上角顶点的横坐标
    y1 = boxes[:,1]	 # 左上角顶点的纵坐标
    w1 = boxes[:,2]  # 矩形框的宽
    h1 = boxes[:,3]  # 矩形框的高
    x2 = x1 + w1  # 右下角顶点横坐标的集合
    y2 = y1 + h1  # 纵坐标的集合
    
    area = (w1 + 1) * (h1 + 1)  # 计算每个矩形框的面积，这里分别加1是为了让IOU匹配不出现0的情况
    temp = []
    
    idxs = np.argsort(h1)	# 将 h1 中的元素从小到大排序并返回每个元素在 h1 中的下标
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        temp.append(i)   
        
        x1_m = np.maximum(x1[i], x1[idxs[:last]])	# 将其他矩形框的左上角横坐标两两比较
        y1_m = np.maximum(y1[i], y1[idxs[:last]])   # 其他矩形框的左上角纵坐标两两比较
        """两个矩形框重叠的部分是矩形，这一步的目的是为了找到这个重叠矩形的左上角顶点"""
        
        x2_m = np.minimum(x2[i], x2[idxs[:last]])	
        y2_m = np.minimum(y2[i], y2[idxs[:last]])
        """目的是为了找出这个重叠矩形的右下角顶点"""

        w = np.maximum(0, x2_m - x1_m + 1)	# 计算矩形的宽
        h = np.maximum(0, y2_m - y1_m + 1)	# 计算矩形的高
        """剔除掉没有相交的矩形，因为两个矩形框相交，则 x2_m - x1_m + 1 和 y2_m - y1_m + 1 大于零，如果两个矩形框不相交则这两个值小于零"""
        
        over = (w * h) / area[idxs[:last]]	# 计算重叠矩形面积和 area 中的面积的比值
        
        idxs = np.delete(idxs, np.concatenate(([last],  	
            np.where(over > threshold)[0])))  	# 剔除重叠的矩形框

    return boxes[temp].astype("int")


hog = cv2.HOGDescriptor()

# opencv的 默认的 hog+svm行人检测器
hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
capture = cv2.VideoCapture(cv2.samples.findFileOrKeep('./video/vtest.avi'))

while True:
    ret, frame = capture.read()

    # Detect people in the image，这个函数决定了检测情况，尤其是 winStride 和 scale 参数
    (rects, weights) = hog.detectMultiScale(frame)
    # (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # 用 Non-maximum Suppression 去掉多余的框
    rects = NMS(rects, threshold=0.5)

    for (y, x, w, h) in rects:
        cv2.rectangle(frame, (y, x), (y + w, x + h), (0, 255, 0), 2)

    cv2.imshow("hog-detector", frame)
    cv2.waitKey(1)
