
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os
import glob
#from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from joblib import Parallel, delayed

file=r"C:\Users\suikou\Downloads\20180725_162800-0.jpg"
src = cv2.imread(file, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
retval, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
image, contours, hierarchy = cv2.findContours(
    bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#image, contours, hierarchy = cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
newimg = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imshow('image', newimg)
cv2.waitKey(0)

detect_count = 0
for i in range(0, len(contours)):
    area = cv2.contourArea(contours[i])
    if len(contours[i]) > 0:
        rect = contours[i]
        x, y, w, h = cv2.boundingRect(rect)
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
        detect_count = detect_count + 1
cv2.imshow('image', src)
cv2.waitKey(0)

