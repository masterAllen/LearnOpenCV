import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def mask_img(img, mask):
    if len(mask.shape) == 2:
        mask = mask[:, :, np.newaxis]
    img = img * mask
    return img

def draw_lines(img, points):
    return_img = np.copy(img)
    y0, x0 = points[-1]
    for y1, x1 in points:
        cv2.line(return_img, (y0,x0), (y1,x1), [0,255,0])
        y0, x0 = y1, x1

    for y0, x0 in points:
        return_img[x0, y0] = [255,0,0]
    return return_img
    


# 显示多张图片
def show_images(imgs, colnum=4, scale=6):
    if colnum > len(imgs):
        colnum = len(imgs)

    rownum = math.ceil(len(imgs) / colnum)

    # 计算 figsize
    maxrow = max([img.shape[0] for _,img in imgs])
    maxcol = max([img.shape[1] for _,img in imgs])

    rowpx, colpx = (maxrow*rownum, maxcol*colnum)
    if rowpx > colpx:
        rowpx, colpx = (math.ceil(scale*rowpx/colpx), scale)
    else:
        rowpx, colpx = (scale, math.ceil(scale*colpx/rowpx))

    # 画图
    _, axes = plt.subplots(rownum, colnum, figsize=(colpx, rowpx))
    if rownum == 1 and colnum == 1:
        axes = np.array([axes], dtype=object)
    axes = axes.reshape(rownum, colnum)
    for i in range(rownum):
        for j in range(colnum):
            axes[i, j].set_axis_off()

            if i*colnum + j < len(imgs):
                title, img = imgs[i*colnum + j]
                axes[i, j].set_title(title)

                if len(img.shape) > 2:
                    axes[i, j].imshow(img[..., ::-1])
                else:
                    axes[i, j].imshow(img, cmap="gray")

    plt.show()