# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:57:27 2018

display image and check annotations in xml file 

@author: Wen Wen
"""
import os
import cv2

from lxml import etree
import xml.etree.ElementTree as ET

if __name__ == '__main__':
    imagepath='for_yolo_training/turn_1_1400/images/'
    xmlpath='for_yolo_training/turn_1_1400/annotations/'
    savepath='for_yolo_training/turn_1_1400/imgwithbbx/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    imagelist=os.listdir(imagepath)
    xmllist=os.listdir(xmlpath)

    for i in imagelist:
        img=cv2.imread(imagepath+i)
        tree = ET.parse(xmlpath+i.replace('jpg','xml'))
        root = tree.getroot()
        width=int(float(root.find('size').find('width').text))
        height=int(float(root.find('size').find('height').text))
        for obj in root.iter('object'):
            xmin=int(float(obj.find('bndbox').find('xmin').text))
            xmax=int(float(obj.find('bndbox').find('xmax').text))
            ymin=int(float(obj.find('bndbox').find('ymin').text))
            ymax=int(float(obj.find('bndbox').find('ymax').text))
            img=cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),3)
        
        cv2.imwrite(savepath+i.replace('.jpg','_withbbx.jpg'),img)
        
        


""" End of File """
