# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:24:11 2020

combine heavy and light network

we check the prediction performance only, not the fps, cause fps is affected by
hardware condition severely.

@author: Wen Wen
"""

import os
import json
import argparse
import numpy as np

from IoU_tools import getIoU

def loadJsonResults(filepath, annotationflag=True, jsonlabel='detection'):
    """
    load prediction results in json format
    
    """
    detectdist={}
    filelist=os.listdir(filepath)
    print('loading precise and rough detection results')
    if annotationflag:
        for filename in filelist:
            if jsonlabel in filename and '.json' in filename:               
                    # loading detection results
                    detectdist[filename]=json.load(open(os.path.join(filepath,filename)))
            #print('{} loading completed'.format(filename))
    
    return detectdist





def boxPrediction(previous_boxes, current_detection, order=2, debug=False) :
    """
    predict the current location with previous boxes
    previous_boxes=[...box_3, box_t-2, box_t-1]
    
    """
    # compute
    if len(previous_boxes)!=order+1:
        raise ValueError('not enough boxes in previous frames, \
                         unable to make prediction')
    
    if order==2:
        # this is only for fixed sample rate, loading time stamps is better
        x1=0;x2=1;x3=2;x4=3 
        
        prediction_result={} # save format:[w,h,cx,cy]
        cate={0:'width',
              1:'height',
              2:'x',
              3:'y'}
        for j in range(4): # 4 cates
            y1= previous_boxes[0][cate[j]]
            y2= previous_boxes[1][cate[j]]
            y3= previous_boxes[2][cate[j]]
            
            if y1==y2 and y2==y3:
                y4=y1 # if not moved
            else:
                y4= solve2ndFunc(y1,y2,y3,x1,x2,x3,x4)    
            prediction_result[cate[j]]=y4
            
            # print prediction result for debug
            if debug:
                print('{}:{},{},{}->{}'.format(cate[j][0],y1,y2,y3,y4))
    return prediction_result
    
def havingLeadingOrNot(annosdict):
    """
    check if an image has leading vehicle in it
    
    """
    annos=annosdict['annotations']
    if len(annos)==0:
        return False # no annotations, return false
    
    # has annotations
    for i in annos:
        if i['category']=='leading':
            return True
    
    # looped over all the annotations and no leading found
    return False

def solve2ndFunc(y1,y2,y3,
                 x1,x2,x3,x4):
    """
    二次函数三点式表达
    y=ax^2+bx+c
    given (x1,y1)(x2,y2)(x3,y3), and x4, solve y4
        y1-y2-b(x1-x2)
    a= ------------------
        x1^2-x2^2
        (x2^2-x3^2)(y1-y2)-(x1^2-x2^2)(y2-y3)
    b= ---------------------------------------
        (x2^2-x3^2)(x1-x2)-(x1^2-x2^2)(x2-x3)
    c= y1-ax1^2-bx1
    
    y4=ax4^2+bx4+c
    
    """                            
    b=((pow(x2,2)-pow(x3,2))*(y1-y2)-(pow(x1,2)-pow(x2,2))*(y2-y3))\
        / ((pow(x2,2)-pow(x3,2))*(x1-x2)-(pow(x1,2)-pow(x2,2))*(x2-x3))
    a=(y1-y2-b*(x1-x2)) / (pow(x1,2)-pow(x2,2)) 
    c= y1-a*pow(x1,2)-b*x1
    
    y4=a*pow(x4,2)+b*x4+c
    
    return y4
    
def combineDetectionResult(precise_detection, rough_detection,
                            max_refine_frames = 20, prediction_fun_order=2,
                            IoU_thresh=0.5):
    """
    combine the detection results
    
    """
    # create new dict of detection, use hard copy for creating the format faster
    new_detection_dict=precise_detection.copy()
    
    for jsonname in new_detection_dict:
        video_len= len(new_detection_dict[jsonname])
        count=0

        for imgname in new_detection_dict[jsonname]:
            if count<=prediction_fun_order:
                # skip the first few frames
                count=count+1
                continue
            else:
                count=count+1
                currentframe=imgname.split('.')[0].split('_')[-1]
                previousimg=imgname.replace(currentframe,str(int(currentframe)-1).zfill(5))
                leading_in_pre_flag=havingLeadingOrNot(rough_detection[jsonname][previousimg])
                if not leading_in_pre_flag:
                    # if previous frame doesn't has leading, use precise detection    
                    new_detection_dict[jsonname][imgname]=precise_detection[jsonname][imgname]
                else:    
                    # if previous frame has leading, check if prediction is feasible
                    prediction_feasible_flag=True
                    while_count=prediction_fun_order+1 # check order+1 previous frames
                    while(while_count):
                        previousimg=imgname.replace(currentframe,str(int(currentframe)-(while_count)).zfill(5))
                        while_count=while_count-1
                        # use new result, instead of raw result
                        leading_in_pre_flag=havingLeadingOrNot(new_detection_dict[jsonname][previousimg])
                        if not leading_in_pre_flag:
                            prediction_feasible_flag=False
                            #print('due to {}, prediction not feasible'.format(previousimg))
                            break
                    
                    if not prediction_feasible_flag:
                        # if not feasible, use rough detection
                        new_detection_dict[jsonname][imgname]=rough_detection[jsonname][imgname]
                    else:
                        # if prediction is feasible, run prediction, refine the rough detection
                        # prepare boxes in previous frames, and rough detection in current frame
                        previous_boxes=[]
                        while_count=prediction_fun_order+1
                        while(while_count):
                            previousimg=imgname.replace(currentframe,str(int(currentframe)-(while_count)).zfill(5))
                            while_count=while_count-1
                            # box format: [w,h,cx,cy]
                            for i in new_detection_dict[jsonname][previousimg]['annotations']:
                                if i['category']=='leading':
                                    previous_boxes.append(i)
                        
                        for i in rough_detection[jsonname][imgname]['annotations']:
                            if i['category']=='leading':
                                current_detection=i

                        prediction_result = boxPrediction(
                                                previous_boxes, 
                                                current_detection, 
                                                order=prediction_fun_order)
                        is_same_vehicle = (getIoU(current_detection,prediction_result)>IoU_thresh)
                        
                        # merge the current detection & prediction with 
                        # confidence score as weight
                        if is_same_vehicle:
                            merged = mergeTwoResultWithConfidence(
                                         res=current_detection, 
                                         addi=prediction_result,
                                         debug=False)
                        
                            # use the merged result as the final detection result
                            # use all the sideways
                            new_detection_dict[jsonname][imgname]=rough_detection[jsonname][imgname]
                            # change the detection of the leading one
                            for i, box in zip(range(len(new_detection_dict[jsonname][imgname]['annotations'])),
                                              new_detection_dict[jsonname][imgname]['annotations']):
                                if box['category']=='leading':
                                    new_detection_dict[jsonname][imgname]['annotations'][i]['width']=merged['width']
                                    new_detection_dict[jsonname][imgname]['annotations'][i]['height']=merged['height']
                                    new_detection_dict[jsonname][imgname]['annotations'][i]['x']=merged['x']
                                    new_detection_dict[jsonname][imgname]['annotations'][i]['y']=merged['y']

    return new_detection_dict

def mergeTwoResultWithConfidence(res, addi, debug=True):
    """
    args:
        res: a rough detection result, full annotation boxes format
            use result['width'] for width, result['score'] for confidence, etc
        addi: additional prediction result, no confidence. use addi['width']
            for width, etc
    
    """
    conf=res['score']
    
    w = round(conf*res['width'] + (1-conf)*addi['width'])
    h = round(conf*res['height'] + (1-conf)*addi['height'])
    x = round(conf*res['x'] + (1-conf)*addi['x'])
    y = round(conf*res['y'] + (1-conf)*addi['y'])
    
    if debug:
        print('w change: {}'.format(w/res['width']))
        print('h change: {}'.format(h/res['height']))
        print('x change: {}'.format(x/res['x']))
        print('y change: {}'.format(y/res['y']))
    
    return {'width':w, 'height': h, 'x':x, 'y':y}

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/heavy_and_light/combined', 
                        help="File path of input data, also the path to save figures")
    parser.add_argument('--title_attach',type=str,default='SSD-strip-150 and SSD-strip-300',
                        help='model name for charts')
    parser.add_argument('--heavynet_path',type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/heavy_and_light/ssd_opt_300',
                        help='path of precise detection result')
    parser.add_argument('--lightnet_path',type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/heavy_and_light/ssd_opt_150',
                        help='path of rough detection result')
    parser.add_argument('--IoUthresh',type=float,default=0.5,
                        help='IoU threshold for choosing positive predictions')
    parser.add_argument('--prediction_func_order',type=int,default=2,
                        help='the order of prediction function')
    args = parser.parse_args()
    
    # parse the arguments
    savepath    = args.save_path
    heavypath   = args.heavynet_path
    lightpath   = args.lightnet_path
    titleattach = args.title_attach
    IoUthresh   = args.IoUthresh
    order       = args.prediction_func_order
    
    # load .json detection result of 150 and 300 resolution, dict form is better than list
    heavy_detection_dict = loadJsonResults(heavypath)
    light_detection_dict = loadJsonResults(lightpath)

    # read detection frame by frame from 300 or 150 detection
    # no need for sudomemory for previous boxes, read number from dict
    new_detection_dict = combineDetectionResult(
                            precise_detection = heavy_detection_dict,
                            rough_detection = light_detection_dict,
                            max_refine_frames = 20,
                            prediction_fun_order=order,
                            IoU_thresh=IoUthresh)
    
    # save the new result as the detection result of "combine heavy and light" method
    detection_dict_for_save={}
    for jsonname in new_detection_dict:
        for imgname in new_detection_dict[jsonname]:
            detection_dict_for_save[imgname] = new_detection_dict[jsonname][imgname]
        
    with open(os.path.join(savepath,'combine_heavy_light_order_{}.json'.format(order)),'w') as savefile:
        savefile.write(json.dumps(detection_dict_for_save, sort_keys = True, indent = 4))    
    
# =============================================================================
#     # save the new result separately for evaluation
#     for jsonname in new_detection_dict:
#         detection_dict_for_save={}
#         for imgname in new_detection_dict[jsonname]:
#             detection_dict_for_save[imgname] = new_detection_dict[jsonname][imgname]
#         with open(os.path.join(savepath,'{}'.format(jsonname)),'w') as savefile:
#             savefile.write(json.dumps(detection_dict_for_save, sort_keys = True, indent = 4)) 
#     
#     
# =============================================================================
    
    
    
    