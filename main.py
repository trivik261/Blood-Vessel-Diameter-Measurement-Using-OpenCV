import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import vessel_select

def diaepc(point,im1):
    rect=cv2.minAreaRect(point)
    result="Diameter predicted="+str(min(rect[1][0],rect[1][1]))
    print("\n",result,"\n")
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(im1,[box],0,(0,0,255),2)
    cv2.imshow(result, im1)
    cv2.waitKey()

def diameter(img):

    g = img[:,:,1]
    point, dl,im1 = vessel_select.select(img)
    diaepc(point,im1)

filename=input("Enter Filename:\n")
img = cv2.imread(filename)
diameter(img)
