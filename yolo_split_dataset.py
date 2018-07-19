# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:49:34 2018

discard images without car detected.
the detector is yolo

@author: Wen Wen
"""

import argparse
import cv2
import os
import time
from darkflow.net.build import TFNet
import numpy as np
import json
import shutil

if __name__=='__main__':
    # pass the parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU', type=float, default=0.8, help="select the GPU to be used (default 1.0)")
    parser.add_argument('--gpuName', type=str, default="/device:gpu:0", help="select the GPU to be used (default to use GPU 0)")
    parser.add_argument('--file_path', type=str, 
                        default='test/leading_car/', 
                        help="File path of input data (default 'D:/Private Manager/Personal File/U of Ottawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/testframes/')")
    
    args = parser.parse_args()
    

    
    # set the work directory 
    filepath=args.file_path
    folderdict=os.listdir(filepath)
    
    # initialize the darknet
    option={
            'model':'cfg/yolo.cfg',
            'load':'bin/yolov2.weights',
            'threshold':0.3,
            'gpu': args.GPU,
            'gpuName': args.gpuName
            }
    tfnet=TFNet(option)    
    print('In processing......')
    
    # begin auto-labelling
    starttime=time.time()
    for i in folderdict:
        imagepath=filepath+i+'/'
        imagedict=os.listdir(imagepath)
        #np.random.shuffle(imagedict)
        if not os.path.exists(imagepath+'disgard/'):
            os.makedirs(imagepath+'disgard/')
        
        result=[]
        imgindex=0
        annotationdict={}
        for imagename in imagedict:
            if 'bbx' not in imagename and ('png' in imagename or 'jpg' in imagename):
                img=cv2.imread(imagepath+imagename)
                
                # skip the broken images
                if img is None:
                    del img
                    # move the file to be disgarded into a new folder, keep the useful untouched
                    shutil.move(imagepath+imagename, imagepath+'disgard/'+imagename)
                    continue
                #img=cv2.resize(img,(100,50))
                result.append(tfnet.return_predict(img))
                if len(result[0])==0:
                    # no positive detection, move the image into new folder
                    annotationdict[imagename]=[]
                    
                    # move the file to be disgarded into a new folder, keep the useful untouched
                    shutil.move(imagepath+imagename, imagepath+'disgard/'+imagename)
                    
                del img
            # clear the result list for current image
            result=[]
            
    # show the total time spent
    endtime=time.time()
    print('total time:'+str(endtime-starttime)+' seconds')
    
""" End of the file """