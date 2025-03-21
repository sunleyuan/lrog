import cv2
import numpy as np


def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat


# def init_vis_image(goal_name, action, legend):
#     vis_image = np.ones((537, 1165, 3)).astype(np.uint8) * 255
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 1
#     color = (20, 20, 20)  # BGR
#     thickness = 2

#     text = "Observations" 
#     textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
#     textX = (640 - textsize[0]) // 2 + 15
#     textY = (50 + textsize[1]) // 2
#     vis_image = cv2.putText(vis_image, text, (textX, textY),
#                             font, fontScale, color, thickness,
#                             cv2.LINE_AA)

#     text = "Find {}  Action {}".format(goal_name, str(action))
#     textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
#     textX = 640 + (480 - textsize[0]) // 2 + 30
#     textY = (50 + textsize[1]) // 2
#     vis_image = cv2.putText(vis_image, text, (textX, textY),
#                             font, fontScale, color, thickness,
#                             cv2.LINE_AA)

#     # draw outlines
#     color = [100, 100, 100]
#     vis_image[49, 15:655] = color
#     vis_image[49, 670:1150] = color
#     vis_image[50:530, 14] = color
#     vis_image[50:530, 655] = color
#     vis_image[50:530, 669] = color
#     vis_image[50:530, 1150] = color
#     vis_image[530, 15:655] = color
#     vis_image[530, 670:1150] = color

#     # # draw legend
#     # lx, ly, _ = legend.shape
#     # vis_image[537:537 + lx, 155:155 + ly, :] = legend

#     return vis_image

def init_vis_image(goal_name, action,legend):
#     print("goal_name=",goal_name)
    # no legend
    # vis_image = np.ones((537, 1165, 3)).astype(np.uint8) * 255
    # with legend
    # vis_image = np.ones((655, 1165, 3)).astype(np.uint8) * 255
    # 初始化显示窗口，留出更多空间在顶部绘制文本
    vis_image = np.ones((655, 1760, 3), dtype=np.uint8) * 255



    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "RGB with semantic segmentation" 
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Depth" 
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640+640+550 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

#     text = "Find {}  Action {}".format(goal_name, str(action))
    text = "Find {} ".format(goal_name)

    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 1280 + (480 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

 

    # plt.show()
    # draw outlines
    color = [100, 100, 100]
    # vis_image[49, 15:655] = color
    # vis_image[49, 670:1150] = color
    # vis_image[50:530, 14] = color
    # vis_image[50:530, 655] = color
    # vis_image[50:530, 669] = color
    # vis_image[50:530, 1150] = color
    # vis_image[530, 15:655] = color
    # vis_image[530, 670:1150] = color


    #     # 第一个图像的边界线
    # vis_image[49, 15:655] = color  # 顶部
    # vis_image[530, 15:655] = color # 底部
    # vis_image[50:530, 14] = color  # 左边
    # vis_image[50:530, 655] = color # 右边

    # # 第二个图像的边界线
    # vis_image[49, 660:1300] = color   # 顶部
    # vis_image[530, 660:1300] = color  # 底部
    # vis_image[50:530, 659] = color    # 左边
    # vis_image[50:530, 1300] = color   # 右边

    # # 第三个图像的边界线
    # vis_image[49, 1305:1784] = color  # 顶部
    # vis_image[530, 1305:1784] = color # 底部
    # vis_image[50:530, 1304] = color   # 左边
    # vis_image[50:530, 1784] = color   # 右边


    # draw legend
    # lx, ly, _ = legend.shape
    lx, ly, _ = legend.shape
    vis_image[537:537 + lx, 155:155 + ly, :] = legend

    return vis_image

