#-*- coding:utf-8 -*-
import os
from PIL import Image

#imgDir="/data/diary_cover_pic/"
#imgFoldName="train"
imgDir="/Users/edz/Desktop/"
imgFoldName="images"
imgs = os.listdir(imgDir+imgFoldName)

imgNum = len(imgs)
for i in range (imgNum):
    try:
        img = Image.open(imgDir+imgFoldName+"/"+imgs[i])
    except:
        print("error image:%s"%imgs[i])
        os.remove(imgDir+imgFoldName+"/"+imgs[i])