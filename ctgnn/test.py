import cv2
from matplotlib import pyplot as plt
import os 
import random
import numpy as np
img = ['00000006.png', '00000007.png', '00000003.png', '00000012.png', '00501005.png']
for i in img:
    path = os.path.join('/mnt/data0/qh/Sewer/Train/', i)
    image = cv2.imread(path)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    v1 = np.clip(cv2.add(1*v, 30), 0, 255)
    hsv_img = np.uint8(cv2.merge((h, s, v1)))
    image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    v2 = np.clip(cv2.add(2*v, 20), 0 ,255)
    image = np.uint8(cv2.merge((h, s, v2)))
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # im = np.power(image, 1.2)

    save_path = os.path.join('./', i.split('.')[0])
    # cv2.imwrite(save_path, im)
    plt.imshow(im)
    plt.show()
    plt.savefig(save_path)
