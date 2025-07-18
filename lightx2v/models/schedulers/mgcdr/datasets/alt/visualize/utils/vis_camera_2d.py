# Standard Library
import os

# Import from third library
import cv2
import random



COLORS = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
          (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
          (238, 130, 238), (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0),
          (255, 239, 213), (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222),
          (65, 105, 225), (173, 255, 47), (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133),
          (148, 0, 211), (255, 99, 71), (144, 238, 144), (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0),
          (189, 183, 107), (255, 255, 224), (128, 128, 128), (105, 105, 105), (64, 224, 208), (205, 133, 63),
          (0, 128, 128), (72, 209, 204), (139, 69, 19),
          (255, 245, 238), (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255),
          (176, 224, 230), (0, 250, 154), (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139),
          (143, 188, 143), (255, 0, 0), (240, 128, 128), (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42),
          (178, 34, 34), (175, 238, 238), (255, 248, 220), (218, 165, 32), (255, 250, 240), (253, 245, 230),
          (244, 164, 96), (210, 105, 30)]


def do_mosaic(frame, box, neighbor=9):
    x = box.left
    y = box.top
    w = box.width
    h = box.height
    """
    马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，这样可以模糊细节，但是可以保留大体的轮廓。
    :param frame: opencv frame
    :param int x :  马赛克左顶点
    :param int y:  马赛克右顶点
    :param int w:  马赛克宽
    :param int h:  马赛克高
    :param int neighbor:  马赛克每一块的宽
    """
    fh, fw = frame.shape[0], frame.shape[1]
    if (y + h > fh) or (x + w > fw):
        return
    for i in range(0, h - neighbor, neighbor):  # 关键点0 减去neightbour 防止溢出
        for j in range(0, w - neighbor, neighbor):
            rect = [j + x, i + y, neighbor, neighbor]
            color = frame[i + y][j + x].tolist()  # 关键点1 tolist
            left_up = (rect[0], rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)  # 关键点2 减去一个像素
            cv2.rectangle(frame, left_up, right_down, color, -1)


def plot_one_box(img, box, color=None, label=None, score=None, line_thickness=None):
    ptLeftTop = (int(box[0]), int(box[1]))
    ptRightBottom = (int(box[2]), int(box[3]))
    
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]

    c1 = ptLeftTop
    c2 = ptRightBottom
    cv2.rectangle(img, ptLeftTop, ptRightBottom, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(str(label), 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, str(label), (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img,
                        str(score), (c1[0], c1[1] + 30),
                        0,
                        tl / 3, [225, 255, 255],
                        thickness=tf,
                        lineType=cv2.LINE_AA)  # noqa
    return img

