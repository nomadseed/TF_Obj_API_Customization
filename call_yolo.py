# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:28:07 2018

call yolov2 for detection, make bounding boxes for all kinds of cars, save the 
bounding boxes into json file that VIVALab Annotator can read

example: python3 call_yolo.py --file_path ./some/folder/ --GPU 0.9

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
    parser.add_argument('--GPU', type=float, default=0.9, help="select the GPU to be used (default 1.0)")
    parser.add_argument('--file_path', type=str, 
                        default='D:/Private Manager/Personal File/U of Ottawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/testframes/', 
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
            'gpu': args.GPU
            }
    tfnet=TFNet(option)
    
    # begin auto-labelling
    starttime=time.time()
    for i in folderdict:
        imagepath=filepath+i+'/'
        imagedict=os.listdir(imagepath)
        #np.random.shuffle(imagedict)
    
        result=[]

        imgindex=0
        nonpositive=True
        
        annotationdict={}
        for imagename in imagedict:
            if 'png' in imagename:
                img=cv2.imread(imagepath+imagename)
                
                # skip the broken images
                if img is None:
                    del img
                    continue
                #img=cv2.resize(img,(100,50))
                result.append(tfnet.return_predict(img))
                if len(result[0])==0:
                    # no positive detection, move the image into new folder
                    annotationdict[imagename]=[]
                    
                    # move the file to be disgarded into a new folder, keep the useful untouched
                    #shutil.move(imagepath+imagename, imagepath+'disgard/'+imagename)
                
                else:
                    #save the annotation into json file
                    annotationdict[imagename]=[]
                    for i in range(len(result[0])):
                        if result[0][i]['label'] in 'car truck bus van vehicle':
                            annodict={}
                            annodict['id']=i
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
                del img
            # clear the result list for current image
            result=[]
            nonpositive=True
            #imgindex+=1
            #if imgindex>10:
                #break
            
        # after done save all the annotation into json file, save the file
        with open(imagepath+'annotation_'+imagepath.split('/')[-2]+'.json','w') as savefile:
            savefile.write(json.dumps(annotationdict, sort_keys = True, indent = 4))
        
    #print('TP:',tp,' FP:',fp,' TN:',tn,' FN:',fn)
    # show the total time spent
    endtime=time.time()
    print('total time:'+str(endtime-starttime)+' seconds')
    
""" End of the file """