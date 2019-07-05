# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:07:30 2019

@author: Wen Wen
"""

import numpy as np
import cv2


def getClass1Score(x,classcode=1):
    return x['scores'][classcode]

def sortByScore(scores,boxes):
    """
    sort detections by class 1 score,
    there are two outputs, one for array, one for dict
    
    Args:
        scores: a list of scores of boxes, format [N,q], N is the number of
            boxes, q is number of class +1, e.g. q=2 if have only one class
        boxes: represents a list of bounding boxes, format [N,4]
            N is the number of boxes, for each box it has 4 parameters
            [y_min, x_min, y_max, x_max]
    Returns:
        boxlist: a new boxlist in array format, format [N,6], in each box it 
            has [y_min, x_min, y_max, x_max, highest_score, predicted_class]
        fullboxlist: a new boxlist in list & dict format, has more details than
            the output "boxlist"
    """
    
    fullboxlist=[]
    for i in range(len(scores)):
        boxdict={}
        boxdict['scores']=scores[i]
        boxdict['y_min']=boxes[i][0]
        boxdict['x_min']=boxes[i][1]
        boxdict['y_max']=boxes[i][2]
        boxdict['x_max']=boxes[i][3]
        fullboxlist.append(boxdict)
        
    fullboxlist.sort(key=getClass1Score, reverse=True)
    boxlist=[]
    for boxdict in fullboxlist:
        # if class 0 has highest prob, find second highest as class
        class_code=np.where(boxdict['scores']==np.amax(boxdict['scores']))[0][0]
        if class_code==0:
            class_code = 1+ np.where(boxdict['scores'][1:]==np.amax(boxdict['scores'][1:]))[0][0]

        
        boxlist.append([boxdict['y_min'],
                          boxdict['x_min'],
                          boxdict['y_max'],
                          boxdict['x_max'],
                          boxdict['scores'][class_code],
                          class_code
                          ])
    
    return boxlist,fullboxlist


def getIoU(bbx_benchmark,bbx_detect):
    """
    calculate Intersection over Union of two bounding boxes
    return 0 if no intersection

    Args:
        bbx_benchmark: format {'xmin':_, 'ymin':_, 'xmax':_, 'ymax':_, ...others}
        bbx_detect: format {'xmin':_, 'ymin':_, 'xmax':_, 'ymax':_, ...others}
    """
    
    # get the cordinates of intersecting square
    x_inter_1=max(bbx_benchmark[1],bbx_detect[1])
    y_inter_1=max(bbx_benchmark[0],bbx_detect[0])
    x_inter_2=min(bbx_benchmark[3],bbx_detect[3])
    y_inter_2=min(bbx_benchmark[2],bbx_detect[2])
# =============================================================================
#     x_inter_1=max(bbx_benchmark['xmin'],bbx_detect['xmin'])
#     y_inter_1=max(bbx_benchmark['ymin'],bbx_detect['ymin'])
#     x_inter_2=min(bbx_benchmark['xmax'],bbx_detect['xmax'])
#     y_inter_2=min(bbx_benchmark['ymax'],bbx_detect['ymax'])
# =============================================================================
    
    # get intersect area
    inter_area = max(0, x_inter_2 - x_inter_1) * max(0, y_inter_2 - y_inter_1)
    
    # get bbx area
    benchmark_area = (bbx_benchmark[2]-bbx_benchmark[0]) * (bbx_benchmark[3]-bbx_benchmark[1])
    detect_area=(bbx_detect[2]-bbx_detect[0]) * (bbx_detect[3]-bbx_detect[1])
    
    # calculate IoU
    iou = inter_area / float(benchmark_area + detect_area - inter_area)
    
    return iou

def greedyNonMaximumSupression(boxlist,clipthresh=0.05,IOUthresh=0.5):
    """
    this function is for greedy non-maximum supression
    
    
    """
    NMSed_list=[]
    if len(boxlist)==0 or clipthresh>1:
        return NMSed_list
    
    # keep every box with largest score while doesn't overlap with all the other
    # boxes
    NMSed_list.append(boxlist[0])
    for i in range(1,len(boxlist)):
        keepflag=True
        
        if boxlist[i][4]<clipthresh:
            break # break when score of current box is lower than thresh
        else:
            #print('----NMS--{}----'.format(i))
            for j in range(len(NMSed_list)):
                iou=getIoU(boxlist[i],NMSed_list[j])
                #print(iou)
                if iou>IOUthresh:
                    keepflag=False
                    break
            if keepflag:
                NMSed_list.append(boxlist[i])
    
    return NMSed_list
    

if __name__=='__main__':
    boxes=np.load('boxes.npy')[0]
    scores=np.load('scores.npy')[0]
    
    boxlist,fullboxlist = sortByScore(scores,boxes)
    
    #private function started with "__"
    img=cv2.imread('D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/debug/bdd100k/images/100k/val/b1ceb32e-3f481b43_crop.jpg')
    #img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    clipthresh=0.2
    IOUthresh=0.5
    w=640
    h=480
    for i in range(len(boxlist)):
        if boxlist[i][5]<clipthresh:
            break
        tl=(int(boxlist[i][1]*w),int(boxlist[i][0]*h))
        br=(int(boxlist[i][3]*w),int(boxlist[i][2]*h))
        img_raw=cv2.rectangle(img,tl,br,(0,255,0),1) # green

    cv2.imwrite('D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/debug/bdd100k/images/100k/val/leadingdetect/beforeNMS.jpg',
                img_raw) # don't save it in png!!!
    
    NMSed_list = greedyNonMaximumSupression(boxlist,clipthresh=clipthresh,IOUthresh=IOUthresh)
    
    img=cv2.imread('D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/debug/bdd100k/images/100k/val/b1ceb32e-3f481b43_crop.jpg')
    
    for i in range(len(NMSed_list)):
        tl=(int(NMSed_list[i][1]*w),int(NMSed_list[i][0]*h))
        br=(int(NMSed_list[i][3]*w),int(NMSed_list[i][2]*h))
        img_NMS=cv2.rectangle(img,tl,br,(0,255,0),1) # green

    cv2.imwrite('D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/debug/bdd100k/images/100k/val/leadingdetect/afterNMS.jpg',
                img_NMS) # don't save it in png!!!
    