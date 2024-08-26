import numpy as np
import cv2

# 画出光流，其实就是传入旧的点和新的点，把他们连起来
def draw_opt(frame, old_points, new_points, colors, mask):
    img = np.copy(frame)
    nowmask = np.copy(mask)
    for i, (new, old) in enumerate(zip(old_points, new_points)):
        a,b = new.astype(int).ravel()
        c,d = old.astype(int).ravel()

        # mask 是前面的帧已经连上的线，现在我们只要把当前帧要连接的线连接起来即可
        nowmask = cv2.line(nowmask, (a,b), (c,d), colors[i].tolist(), 2)
        img = cv2.circle(img, (a,b), 3, colors[i].tolist(),-1)
    img = cv2.add(img, nowmask)
    return img, nowmask


cap = cv2.VideoCapture('./video/vtest.avi')

# params for ShiTomasi corner detection
feature_params = {
    'maxCorners': 100,
    'qualityLevel': 0.3,
    'minDistance': 7,
    'blockSize': 7,
}

# Parameters for lucas kanade optical flow
lk_params = {
    'winSize': (15,15),
    'maxLevel': 2,
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
}

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_points0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
old_points1 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create some random colors
colors = np.random.randint(0, 255, (100,3))
mask0 = np.zeros_like(old_frame)
mask1 = np.zeros_like(old_frame)

while True:
    ret, now_frame = cap.read()
    if now_frame is None:
        break

    now_gray = cv2.cvtColor(now_frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    # 另一种：cv2.SparseOpticalFlow(**lk_params).calc(old_gray, now_gray, old_points1, None)
    new_points0, st0, err0 = cv2.calcOpticalFlowPyrLK(old_gray, now_gray, old_points0, None, **lk_params)
    # new_points1, st1, err1 = cv2.optflow.calcOpticalFlowSparseRLOF(old_frame, now_frame, old_points1, None)

    # Select good points
    old_points0 = old_points0[st0==1]
    new_points0 = new_points0[st0==1]

    # old_points1 = old_points1[st1==1]
    # new_points1 = new_points1[st1==1]

    # 画图
    img0, mask0 = draw_opt(now_frame, old_points0, new_points0, colors, mask0)
    # img1, mask1 = draw_opt(now_frame, old_points1, new_points1, colors, mask1)

    cv2.imshow('LK', img0)
    # cv2.imshow('RLOF', img1)

    old_gray, old_frame = now_gray, now_frame
    old_points0 = new_points0.reshape(-1, 1, 2)
    # old_points1 = new_points1.reshape(-1, 1, 2)

    cv2.waitKey(1)


cv2.destroyAllWindows()
cap.release()

