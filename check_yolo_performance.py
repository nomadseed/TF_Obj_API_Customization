# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:52:06 2018

given the benchmark annotation, compare the result of yolo with it, return 
TP/FP/TN/FN, as well as percision and recall.


@author: Wen Wen
"""

import argparse
import cv2
import os
import time
import numpy as np
import json

def getIoU(bbx_benchmark,bbx_detect):
    """
    calculate Intersection over Union of two bounding boxes
    return 0 if no intersection
    
    """
    
    # get the cordinates of intersecting square
    x_inter_1=max(bbx_benchmark['x'],bbx_detect['x'])
    y_inter_1=max(bbx_benchmark['y'],bbx_detect['y'])
    x_inter_2=min(bbx_benchmark['x']+bbx_benchmark['width'],bbx_detect['x']+bbx_detect['width'])
    y_inter_2=min(bbx_benchmark['y']+bbx_benchmark['height'],bbx_detect['y']+bbx_detect['height'])
    
    # get intersect area
    inter_area = max(0, x_inter_2 - x_inter_1) * max(0, y_inter_2 - y_inter_1)
    
    # get bbx area
    benchmark_area = bbx_benchmark['width'] * bbx_benchmark['height']
    detect_area=bbx_detect['width'] * bbx_detect['height']
    
    # calculate IoU
    iou = inter_area / float(benchmark_area + detect_area - inter_area)
    
    # for debugging, check the result
    '''
    print('benchmark:(x1,y1)=',bbx_benchmark['x'],bbx_benchmark['y'],' (x2,y2)=',bbx_benchmark['x']+bbx_benchmark['width'],bbx_benchmark['y']+bbx_benchmark['height'])
    print('detect:(x1,y1)=',bbx_detect['x'],bbx_detect['y'],'(x2,y2)=',bbx_detect['x']+bbx_detect['width'],bbx_detect['y']+bbx_detect['height'])
    print('intersection area:',inter_area)
    print('benchmark area:',benchmark_area)
    print('detect area:',detect_area)
    print('IoU=',iou,'\n')
    '''
    
    return iou

def checkSingleImage(annos_benchmark,annos_detect,totalperformance,IOUthresh):
    # check every detected bbx, add the result to currentperformance
    performance=totalperformance
    # special case 1: no detected, but benchmark has cars, all false negative
    if len(annos_detect)==0:
        if len(annos_benchmark)!=0:
            for bbx in annos_benchmark:
                if bbx['category'].lower()=='leading':
                    performance['leading']['fn']+=1
                    performance['overall']['fn']+=1
                elif bbx['category'].lower()=='sideways':
                    # no benchmark for leading car, tn
                    performance['overall']['fn']+=1
                    
    # special case 2: no benchmark, but detected, all false positive
    elif len(annos_benchmark)==0:
        if len(annos_detect)!=0:
            for bbx in annos_detect:
                if bbx['category'].lower()=='leading':
                    performance['leading']['fp']+=1
                    performance['overall']['fp']+=1
                elif bbx['category'].lower()=='sideways':
                    # no detection for leading car, tn
                    performance['overall']['fp']+=1
        
    # common case: both benchmark and detected file has bbx, calculate IoU
    else:
        detect_poplist=annos_detect.copy()
        for bbx_benchmark in annos_benchmark:
            # calculate IoU bbx in detect and bbx in benchmark
            cnt=0
            totaldetect=len(annos_detect)
            for i in range(len(annos_detect)):
                iou=getIoU(bbx_benchmark,annos_detect[i])
                
                # true positive
                if iou>=IOUthresh:
                    if bbx_benchmark['category'].lower()==annos_detect[i]['category'].lower():
                        performance['leading']['tp']+=1
                        performance['overall']['tp']+=1
                    else:
                        if annos_detect[i]['category'].lower()=='leading': 
                            # fp for leading car, tp for all cars
                            performance['leading']['fp']+=1
                            performance['overall']['tp']+=1
                        elif annos_detect[i]['category'].lower()=='sideways':
                            # fn for leading car, tp for all cars
                            performance['leading']['fn']+=1
                            performance['overall']['tp']+=1       
                    detect_poplist.pop(i)
                    break # go to next benchmark if already have one match
                
                # searched in anno_detect, but no match
                if cnt==totaldetect:
                    # already looked for all anno_detect but no match
                    if bbx_benchmark['category'].lower()=='leading':
                        performance['leading']['fn']+=1
                        performance['overall']['fn']+=1
                    elif bbx_benchmark['category'].lower()=='sideways':
                        # tn for leading car
                        performance['overall']['fn']+=1
                cnt+=1
        
        # all bbx in benchmark are searched, matching detection are poped
        # if there are some more detecion, they are fp
        #print(len(detect_poplist))
        for bbx_detect in detect_poplist:
            if bbx_detect['category'].lower()=='leading':
                performance['leading']['fp']+=1
                performance['overall']['fp']+=1
            elif bbx_detect['category'].lower()=='sideways':
                performance['overall']['fp']+=1
        
    
    return performance

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
                        default='D:/Private Manager/Personal File/U of Ottawa/Lab works/2018 winter/YOLO3/darkflow-master/test/leading_car/', 
                        help="File path of input data")
    parser.add_argument('--IoU',type=float,default=0.5,help='percentage of IoU')
    args = parser.parse_args()
    
    filepath=args.file_path
    folderdict=os.listdir(filepath)
    IOUthresh=args.IoU
    
    # initial the numbers for performance
    totalperformance={'leading':{'tp':0, 'fp':0, 'tn':0, 'fn':0},
                      'overall':{'tp':0, 'fp':0, 'tn':0, 'fn':0}}
    
    for foldername in folderdict:
        jsonpath=filepath+foldername+'/'
        # load the json files
        benchmark=json.load(open(jsonpath+'annotation_'+foldername+'_with_leading.json'))
        detected=json.load(open(jsonpath+'annotation_'+foldername+'.json'))
        
        for imgname in detected:
            # if not detected
            if len(detected[imgname])==0:
                annos_detect={}
            else:
                annos_detect=detected[imgname]['annotations']
            
            # if no such a benchmark
            if benchmark.get(imgname)==None:
                annos_benchmark={}
            else:
                annos_benchmark=benchmark[imgname]['annotations']
            
            # calculate performance
            totalperformance=checkSingleImage(annos_benchmark,annos_detect,totalperformance,IOUthresh)
    
    # calculate precision, recall and missrate
    precision_leading=totalperformance['leading']['tp']/(totalperformance['leading']['tp']+totalperformance['leading']['fp'])
    precision_overall=totalperformance['overall']['tp']/(totalperformance['overall']['tp']+totalperformance['overall']['fp'])
    
    recall_leading=totalperformance['leading']['tp']/(totalperformance['leading']['tp']+totalperformance['leading']['fn'])
    recall_overall=totalperformance['overall']['tp']/(totalperformance['overall']['tp']+totalperformance['overall']['fn'])
    
    missrate_leading=totalperformance['leading']['fn']/(totalperformance['leading']['tp']+totalperformance['leading']['fn'])
    missrate_overall=totalperformance['overall']['fn']/(totalperformance['overall']['tp']+totalperformance['overall']['fn'])
    
    print('IoU threshold:',IOUthresh,'\n')
    
    print('overall performance on detecting cars:')
    print('precision:',precision_overall)
    print('recall:',recall_overall)
    print('miss rate:',missrate_overall,'\n')
    
    print('performance on detecting leading cars:')
    print('precision:',precision_leading)
    print('recall:',recall_leading)
    print('miss rate:',missrate_leading,'\n')
    
    
    
    
    
    
""" End of file """