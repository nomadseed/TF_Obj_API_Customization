# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:28:07 2018

call yolov2 for detection, make bounding boxes for all kinds of cars, save the 
bounding boxes into json file that VIVALab Annotator can read

example: python3 yolo_leading_car_detection.py --file_path ./some/folder/ --GPU 0.9

@author: Wen Wen
"""
import argparse
import cv2
import os
import time
from darkflow.net.build import TFNet
import json

def classifier(x,y,width,height,threshold=0.5, strip_x1=312,strip_x2=328):
    """
    input the information of bounding box (topleft, bottomright), get the possible
    category of the detected object. the method used here could be some meta-algorithm
    like SVM or Decision Tree.
    
    categories using: 'leading','sideways'
    
    threshold is the overlapping area of bbx and vertical strip, divided by overlapping 
    area between vertical strip and horizontal extension of bbx
    
    if the overlapping percentage is above threshold, return 'leading', else 
    return 'sideways'
    
    """
    
    x1=x
    x2=x+width
    category=''
    if threshold<=0 or threshold>1:
        threshold=0.5
    
    if x1>strip_x1 and x2<strip_x2:
        category='leading'
    elif x2-strip_x1>(strip_x2-strip_x1)*threshold and x1-strip_x2<-(strip_x2-strip_x1)*threshold:
        category='leading'
    else:
        category='sideways'
    
    return category

if __name__=='__main__':
    # pass the parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU', type=float, default=0.8, help="select the GPU to be used (default 0.8)")
    parser.add_argument('--file_path', type=str, 
                        default='test/leading_car/', 
                        help="File path of input data (default 'D:/Private Manager/Personal File/U of Ottawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/testframes/')")
    parser.add_argument('--draw_image', type=bool, default=False, help="draw image for debug (default False)")
    
    args = parser.parse_args()
    

    
    # set the work directory 
    filepath=args.file_path
    drawflag=args.draw_image
    folderdict=os.listdir(filepath)
    
    # initialize the darknet
    option={
            'model':'cfg/yolo.cfg',
            'load':'bin/yolov2.weights',
            'threshold':0.3,
            'gpu': args.GPU
            }
    tfnet=TFNet(option)    
    print('In processing......')
    
    # begin auto-labelling
    starttime=time.time()
    for i in folderdict:
        imagepath=filepath+i+'/'
        if not os.path.exists(imagepath+'leadingdetect/'):
            os.makedirs(imagepath+'leadingdetect/')
            
        imagedict=os.listdir(imagepath)
        #np.random.shuffle(imagedict)
    
        result=[]
        imgindex=0
        annotationdict={}
        for imagename in imagedict:
            if 'leading' not in imagename and ('png' in imagename or 'jpg' in imagename):
                img=cv2.imread(imagepath+imagename)
                
                # skip the broken images
                if img is None:
                    del img
                    continue
                #img=cv2.resize(img,(100,50))
                result.append(tfnet.return_predict(img))
                if len(result[0])==0:
                    # no positive detection, move the image into new folder
                    annotationdict[imagename]={}
                    
                    # move the file to be disgarded into a new folder, keep the useful untouched
                    #shutil.move(imagepath+imagename, imagepath+'disgard/'+imagename)
                
                else:
                    # create annotation for json file
                    annotationdict[imagename]={}
                    annotationdict[imagename]['name']=imagename
                    annotationdict[imagename]['width']=img.shape[1]
                    annotationdict[imagename]['height']=img.shape[0]
                    annotationdict[imagename]['annotations']=[]
                    
                    # save result about vehicles
                    for i in range(len(result[0])):
                        if result[0][i]['label'].lower() in 'car truck bus':
                            annodict={}
                            annodict['id']=i
                            annodict['shape']=['Box',1]
                            annodict['label']=result[0][i]['label']
                            annodict['x']=result[0][i]['topleft']['x']
                            annodict['y']=result[0][i]['topleft']['y']
                            annodict['width']=result[0][i]['bottomright']['x']-annodict['x']
                            annodict['height']=result[0][i]['bottomright']['y']-annodict['y']
                            category=classifier(annodict['x'],annodict['y'],annodict['width'],annodict['height'])
                            annodict['category']=category # decide the category with the cordinates of bbx
                            
                            annotationdict[imagename]['annotations'].append(annodict)
                            
                            if drawflag:
                                # for debugging, save the images with bounding boxes
                                tl=(result[0][i]['topleft']['x'],result[0][i]['topleft']['y'])
                                br=(result[0][i]['bottomright']['x'],result[0][i]['bottomright']['y'])
                                if annodict['category']=='leading':
                                    img=cv2.rectangle(img,tl,br,(0,255,0),2) # green
                                elif annodict['category']=='sideways':
                                    img=cv2.rectangle(img,tl,br,(0,0,255),2) # red
                    if drawflag:
                        cv2.imwrite(imagepath+'leadingdetect/'+imagename.split('.')[0]+'_leadingdetect.jpg',img) # don't save it in png!!!

                del img
            # clear the result list for current image
            result=[]

            
        # after done save all the annotation into json file, save the file
        with open(imagepath+'annotation_'+imagepath.split('/')[-2]+'.json','w') as savefile:
            savefile.write(json.dumps(annotationdict, sort_keys = True, indent = 4))
        
    # show the total time spent
    endtime=time.time()
    print('total time:'+str(endtime-starttime)+' seconds')
    
""" End of the file """