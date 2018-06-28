# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:08:49 2018

Get the bird eyes view by using Inverse Perspective Mapping,
for each video folder, we'll have different extrinsic matrices.

@author: Wen Wen
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

IMAGE_H = 480
IMAGE_W = 640



if __name__=='__main__':

    src = np.float32([[233, 283], [370, 289], [51, 354], [492, 368]])
    dst = np.float32([[265, 283], [375, 283], [256, 440], [375, 440]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation
    
    filepath='./'
    filename='2.jpg'
    img = cv2.imread(filepath+filename) # Read the test img
    img = img[0:IMAGE_H, 0:IMAGE_W] # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
    #img_inv = cv2.warpPerspective(warped_img, Minv, (IMAGE_W, IMAGE_H)) # Inverse transformation
    plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
    plt.show()
    cv2.imwrite(filepath+filename.split('.')[0]+'_after_IPM.jpg', warped_img)
    Mappingdict={'M':M.tolist(),'Minv':Minv.tolist()}
    with open(filepath+filename.split('.')[0]+'_MappingMatrix.json','w') as savefile:
        savefile.write(json.dumps(Mappingdict, sort_keys = True, indent = 4))
