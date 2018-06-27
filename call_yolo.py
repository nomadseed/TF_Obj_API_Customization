# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:28:07 2018

call yolo detection

@author: Wen Wen
"""

import cv2
import os
import time
from darkflow.net.build import TFNet
import numpy as np
import json

if __name__=='__main__':
    
    option={
            'model':'cfg/yolo.cfg',
            'load':'bin/yolov2.weights',
            'threshold':0.3
            }

    tfnet=TFNet(option)
    #filepath='D:/Private Manager/Personal File/U of Ottawa/Lab works/2018 winter/Vehicle Detect Cityscape/vehicleBlockChannels_trainvaltest'
    #filepath='D:/Private Manager/Personal File/U of Ottawa/Lab works/2018 winter/YOLO3/darkflow-master/sample_img/'
    filepath='D:/Private Manager/Personal File/U of Ottawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/1208/'
    imagedict=os.listdir(filepath)
    np.random.shuffle(imagedict)
    
    result=[]
    tp=0
    fp=0
    tn=0
    fn=0
    imgindex=0
    nonpositive=True
    starttime=time.time()
    annotationdict={}
    for imagename in imagedict:
        if 'png' in imagename:
            img=cv2.imread(filepath+imagename)
            
            #img=cv2.resize(img,(100,50))
            result.append(tfnet.return_predict(img))
            annotationdict[imagename]=[]
            # draw the bounding boxes
            for i in range(len(result[0])):
                if result[0][i]['label'] in 'car truck bus van vehicle':
                    annodict={}
                    annodict['id']='0'
                    annodict['category']='leading'
                    annodict['image']=imagename
                    annodict['image_width']=img.shape[1]
                    annodict['image_height']=img.shape[0]
                    annodict['shape']=['Box',1]
                    annodict['label']='Car'
                    annodict['x']=result[0][i]['topleft']['x']
                    annodict['y']=result[0][i]['topleft']['y']
                    annodict['width']=result[0][i]['bottomright']['x']-annodict['x']
                    annodict['height']=result[0][i]['bottomright']['y']-annodict['y']
                    
                    annotationdict[imagename].append(annodict)
                    #img=cv2.rectangle(img,tl,br,(0,0,255),5)
            #cv2.imwrite(filepath+'/savedimage/'+imagename,img)
            
            """
            # for roc on bounding boxes
            # check for the positive detection first
            for i in range(len(result[0])):
                if result[0][i]['label'] in 'car truck bus van vehicle' and 'car' in imagename: # TP
                    tp+=1
                    nonpositive=False
                    break
                elif result[0][i]['label'] in 'car truck bus van vehicle' and 'car' not in imagename: # FP
                    fp+=1
                    nonpositive=False
                    break
            # if no positive detection
            if nonpositive and 'car' not in imagename: # TN
                    tn+=1
            elif nonpositive and 'car' in imagename: # FN
                    fn+=1
            """
        
        result=[]
        nonpositive=True
        #imgindex+=1
        #if imgindex>10:
            #break
    with open(filepath+'annotation_'+filepath.split('/')[-2]+'.json','w') as savefile:
        savefile.write(json.dumps(annotationdict, sort_keys = True, indent = 4))
        
    print('TP:',tp,' FP:',fp,' TN:',tn,' FN:',fn)
    # show the total time spent
    endtime=time.time()
    print('total time:'+str(endtime-starttime)+' seconds')
    
""" End of the file """