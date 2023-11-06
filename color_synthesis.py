import cv2
import numpy as np
import matplotlib.pyplot as plt



color_method = [cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_JET, cv2.COLORMAP_WINTER,
                cv2.COLORMAP_RAINBOW, cv2.COLORMAP_OCEAN, cv2.COLORMAP_SUMMER, cv2.COLORMAP_SPRING,
                cv2.COLORMAP_COOL, cv2.COLORMAP_HSV, cv2.COLORMAP_SPRING, cv2.COLORMAP_PINK, cv2.COLORMAP_HOT]

def synthesis(image_name, output_name, method, histogram=False):
    image = cv2.imread(image_name, 0)
    row = image.shape[0]
    col = image.shape[1]
    count=np.zeros(256, np.uint16)
    for i in range(row):
        for j in range(col):
            count[image[i, j]] += 1

    if histogram:
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, 256), count, label='gray')
        plt.title("Gray distribution histogram")
        plt.xlabel("value")
        plt.ylabel("count")
        plt.legend(loc="lower left")
        plt.savefig(output_name+'灰度分布图.jpg')
        # plt.show()

    label = np.zeros((row,col), np.uint8)
    Minimum = []
    num = 0
    if method < 12:
        image_color = cv2.applyColorMap(image, cv2.COLORMAP_HSV)
        cv2.imwrite(output_name+'output.jpg', image_color)
    else:
        for i in range(3, 253, 1):
            if count[i] == min(count[i-3: i+3]):
                Minimum.append(i)
                num += 1

        if num < 1:
            print("未找到足够的极小值点，无法进行彩色合成")
            return 0
        else:
            for i in range(num - 1):
                label[image > Minimum[i]] = i+1
            image_color = np.zeros((row, col, 3), np.uint8)

            for i in range(num):
                image_color[label == i] = np.random.randint(0, 255, 3)

            cv2.imwrite(output_name+'output.jpg', image_color)

            return 1

