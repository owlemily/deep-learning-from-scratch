# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('../dataset/cactus.png') # 이미지 읽어오기
#이미지의 적절한 경로를 설정한다.

plt.imshow(img)

plt.show()
