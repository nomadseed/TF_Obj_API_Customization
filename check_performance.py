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

import bbox_lib.BoundingBoxes as BoundingBoxes
import bbox_lib.BoundingBox as BoundingBox
import bbox_lib.Evaluator as Evaluator
import bbox_lib.utils as utils

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

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

def _get_tp_fp_fn_extern(predictionlist, groundtruthlist, totalperformance, 
                         confidencethresh=0.3,IOUthresh=0.5, rejectsize=0):
    """
    use external bbox lib to get the tp, fp and fn
    
    args:
        predictionlist: list for prediction results
        groundtruthlist: list for groundtruch boxes
        totalperformance: output dictionary
        confidencethresh: confidence threshold, all the boxes
            with larger confidences will be counted
        IOUthresh: IOU threshold
        rejectsize: all the predictions smaller than this size
            will be removed
    
    """
    performance = totalperformance
    sizethresh=[32,96] # TF obj API threshlist to divide boxes into S,M,and L objects
    #sizethresh=[22,75] # a threshlist to divide boxes into S,M,and L objects
    
    bboxlist = BoundingBoxes.BoundingBoxes()
    bboxlist_S = BoundingBoxes.BoundingBoxes()
    bboxlist_M = BoundingBoxes.BoundingBoxes()
    bboxlist_L = BoundingBoxes.BoundingBoxes()
    #detections
    for anno in predictionlist:
        if anno['width']>=rejectsize and anno['height']>=rejectsize:
            if anno['score']>confidencethresh:
                bbox=BoundingBox.BoundingBox(imageName='',classId=1,
                                                  x=anno['x'],y=anno['y'],
                                                  w=anno['width'],h=anno['height'],
                                                  bbType=utils.BBType.Detected,
                                                  classConfidence=anno['score'])
                bboxlist.addBoundingBox(bbox)
                if max(anno['width'],anno['height'])<sizethresh[0]:
                    bboxlist_S.addBoundingBox(bbox)
                elif max(anno['width'],anno['height'])<sizethresh[1]:
                    bboxlist_M.addBoundingBox(bbox)
                else:
                    bboxlist_L.addBoundingBox(bbox)
    #groundtruth
    for anno in groundtruthlist:
        bbox=BoundingBox.BoundingBox(imageName='',classId=1,
                                          x=anno['x'],y=anno['y'],
                                          w=anno['width'],h=anno['height'],
                                          bbType=utils.BBType.GroundTruth)
        bboxlist.addBoundingBox(bbox)
        if max(anno['width'],anno['height'])<sizethresh[0]:
                bboxlist_S.addBoundingBox(bbox)
        elif max(anno['width'],anno['height'])<sizethresh[1]:
            bboxlist_M.addBoundingBox(bbox)
        else:
            bboxlist_L.addBoundingBox(bbox)
            
    evaluator=Evaluator.Evaluator()
    rec = evaluator.GetPascalVOCMetrics(bboxlist,IOUthresh)
    rec_S = evaluator.GetPascalVOCMetrics(bboxlist_S,IOUthresh)
    rec_M = evaluator.GetPascalVOCMetrics(bboxlist_M,IOUthresh)
    rec_L = evaluator.GetPascalVOCMetrics(bboxlist_L,IOUthresh)
    if len(rec)==0:
        return performance
    
    performance['overall']['tp']+=rec[0]['total TP']
    performance['overall']['fp']+=rec[0]['total FP']
    performance['overall']['fn']+=rec[0]['total positives']-rec[0]['total TP']
    
    if len(rec_S)!=0:
        performance['small']['tp']+=rec_S[0]['total TP']
        performance['small']['fp']+=rec_S[0]['total FP']
        performance['small']['fn']+=rec_S[0]['total positives']-rec_S[0]['total TP']
    if len(rec_M)!=0:
        performance['medium']['tp']+=rec_M[0]['total TP']
        performance['medium']['fp']+=rec_M[0]['total FP']
        performance['medium']['fn']+=rec_M[0]['total positives']-rec_M[0]['total TP']
    if len(rec_L)!=0:
        performance['large']['tp']+=rec_L[0]['total TP']
        performance['large']['fp']+=rec_L[0]['total FP']
        performance['large']['fn']+=rec_L[0]['total positives']-rec_L[0]['total TP']

    return performance

def _get_tp_fp_fn(predictionlist, groundtruthlist, totalperformance, confidencethresh=0.3,IOUthresh=0.5):
    '''
    validates if the GT and Predictions have larger than IOU 0.5, and are one to one mapping.
    '''
    performance=totalperformance
    
    tp = 0
    fp = 0
    fn = 0
    
    plist=[i for i in predictionlist if i['score']>confidencethresh]
    glist=groundtruthlist
    
    pg = np.zeros([len(plist), len(glist)])
    for i in range(len(plist)):
        for j in range(len(glist)):
            pg[i, j] = getIoU(plist[i], glist[j])
            if pg[i, j] < IOUthresh: # IOU THRESHOLD
                pg[i, j] = 0
    # matrix of IOUs with IOUs = 0 or IOU > thr(0.5)
    validity = True
    while pg.shape[0]>0 and pg.shape[1]>0:
        idr = 0
        while True:
            idc = np.argmax(pg[idr,:])
            if np.max(pg[idr,:]) == 0: # FP: Prediction not matching any GT
                pg = np.delete(pg, idr, axis=0 )#delete prediction
                fp+=1
                break    
            if np.max(pg[:,idc]) == 0: # FN: GT not matching any prediction
                fn+=1
                pg = np.delete(pg, idc, axis=1)
                break
            if idr == np.argmax(pg[:,idc]): # TP: Matching - remove idr & idc
                pg = np.delete(pg, idr, axis=0) #delete prediction
                pg = np.delete(pg, idc, axis=1) #delete gt
                tp +=1
                break
            else:
                idr = np.argmax(pg[:,idc])
    fp += pg.shape[0]
    fn += pg.shape[1]
    
    performance['overall']['tp']+=tp
    performance['overall']['fp']+=fp
    performance['overall']['fn']+=fn
    
    return performance


def checkSingleImage(imgname,annos_benchmark,annos_detect,totalperformance,IOUthresh):
    # check every detected bbx, add the result to currentperformance
    performance=totalperformance
    
    # remove all the null object (none-car objects)
    annos_detect=[i for i in annos_detect if i['label']!='null']
    
    # special case 1: no detected, but benchmark has cars, all false negative
    if len(annos_detect)==0:
        if len(annos_benchmark)!=0:
            for bbx in annos_benchmark:
                if bbx['category'].lower()=='leading':
                    performance['leading']['fn']+=1
                    performance['overall']['fn']+=1
                else:
                    # no benchmark for leading car, tn
                    performance['overall']['fn']+=1
                    
    # special case 2: no benchmark, but detected, all false positive
    elif len(annos_benchmark)==0:
        if len(annos_detect)!=0:
            for bbx in annos_detect:
                if bbx['category'].lower()=='leading':
                    performance['leading']['fp']+=1
                    performance['overall']['fp']+=1
                else:
                    # no detection for leading car, tn
                    performance['overall']['fp']+=1
        
    # common case: both benchmark and detected file has bbx, calculate IoU
    else:
        benchlist=[] # to store the matched bbx
        detectlist=[] # to store the matched bbx
        for i in range(len(annos_benchmark)):
            # calculate IoU bbx in detect and bbx in benchmark            
            for j in range(len(annos_detect)):
                iou=getIoU(annos_benchmark[i],annos_detect[j])
                # true positive
                if iou>=IOUthresh:
                    if annos_benchmark[i]['category'].lower()==annos_detect[j]['category'].lower():
                        performance['overall']['tp']+=1 # tn for leading
                        if annos_detect[j]['category'].lower()=='leading':
                            performance['leading']['tp']+=1
                        
                    else:
                        # tp for overall
                        performance['overall']['tp']+=1
                        if annos_detect[j]['category'].lower()=='leading': 
                            # fp for leading car
                            performance['leading']['fp']+=1
                        else:
                            # fn for leading car
                            performance['leading']['fn']+=1
                    
                    # mark the matched bbx in benchmark
                    benchlist.append(i)  
                    detectlist.append(j)
                    # go to next benchmark if already have one match
        '''
        print('imgname',imgname)
        print('bench',benchlist,'bench len',len(annos_benchmark))
        print('detect',detectlist,'detect len',len(annos_detect),'\n')
        '''
        
        # find the unmatched in benchmark, fn
        for i in range(len(annos_benchmark)):
            # if matched, skip
            if i in benchlist:
                continue
            else:
                performance['overall']['fn']+=1
                if annos_benchmark[i]['category']=='leading':
                    # miss a leading car in detection
                    performance['leading']['fn']+=1
        
        # find the unmatched in detect, fp
        for i in range(len(annos_detect)):
            # if matched, skip
            if i in detectlist:
                continue
            else:
                performance['overall']['fp']+=1
                if annos_detect[i]['category']=='leading':
                    # miss a leading car in detection
                    performance['leading']['fp']+=1
                
    
    return performance

def plotPrecisionRecall(performancedict,key='overall', 
                        modelname='', savepath=None):
    
    if savepath is None:
        raise ValueError('a savepath for figures must be provided')
        return 0
    
    precision_list=[]
    recall_list=[]

    for thresh in performancedict:
        precision_list.append(performancedict[thresh][key]['precision'])
        recall_list.append(performancedict[thresh][key]['recall'])
    
    chartaxis = [0.0,1.0,0.0,1.0]
    
    plt.figure(figsize=(6,6),dpi=100)
    plt.axis(chartaxis)
    plt.plot(recall_list,precision_list,'r.-',label='Overall')
    plt.title(modelname+ ' ' + key +' precision vs recall')
    
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc=1)
    plt.savefig(os.path.join(savepath,key+'_precision_vs_recall.png'))
    plt.show()

SETUP={
       'SSD_strip_300_gt22':{
               'file_path':'D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k/detection results/ssd_mobilenet_opt_300_gt22/',
               'model_name':'ssd_mobilenet_opt_300_gt22',
               'detected_path':'D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k/detection results/ssd_mobilenet_opt_300_gt22/annotation_val_detection.json',
               'benchmark_path':'D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k/labels/bdd100k_labels_images_val_VIVA_format_crop_gt22.json',
               },
       'combine_heavy_light':{
               'file_path':'D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/heavy_and_light/combined',
               'model_name':'combine SSD-strip-300 & 150',
               'detected_path':'D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/heavy_and_light/combined/combine_heavy_light_order_2.json',
               'benchmark_path':'D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/heavy_and_light/gt/Part4_ACC_groundtruth.json'
               },
       'ssd_opt_vnx_300':{
               'file_path':'D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/heavy_and_light/ssd_opt_vnx_finetune',
               'model_name':'SSD-strip-300 on viewnyx',
               'detected_path':'D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/heavy_and_light/ssd_opt_vnx_finetune/ssd_opt_vnx_finetune.json',
               'benchmark_path':'D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/heavy_and_light/gt/Part4_ACC_groundtruth.json'
               },
       'ssd_opt_vnx_150':{
               'file_path':'D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/heavy_and_light/ssd_opt_150_gt22',
               'model_name':'SSD-strip-150 on viewnyx',
               'detected_path':'D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/heavy_and_light/ssd_opt_150_gt22/ssd_opt_150_gt22.json',
               'benchmark_path':'D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/heavy_and_light/gt/Part4_ACC_groundtruth.json'
               }
       
       }


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='ssd_opt_vnx_150',
                        help='model name for charts')
    parser.add_argument('--IoUthresh',type=float,default=0.5,
                        help='IoU threshold for choosing positive predictions')
    
    args = parser.parse_args()
    model=args.model
    IOUthresh       = args.IoUthresh
    
    filepath        = SETUP[model]['file_path']
    modelname       = SETUP[model]['model_name']
    detectpath      = SETUP[model]['detected_path']
    benchmarkpath   = SETUP[model]['benchmark_path']
    
    folderdict      = os.listdir(filepath)
    totalperformance={}
    threshlist=[0.0001, 0.01, 0.02, 0.03, 0.04, 
                0.05, 0.1, 0.15, 0.2, 0.25, 
                0.3, 0.35, 0.4, 0.45, 0.5, 
                0.55, 0.6, 0.65, 0.7, 0.75, 
                0.8, 0.85, 0.9, 0.95, 0.99] #25 numbers in total
    precision_overall_list=[]
    precision_leading_list=[]
    recall_overall_list=[]
    recall_leading_list=[]
    MRs_overall_list=[]
    MRs_leading_list=[]
    
    # load json files
    if detectpath=='none' and benchmarkpath=='none':
        for foldername in folderdict:
            jsonpath=os.path.join(filepath,foldername)
            # load the json files
            if not os.path.exists(os.path.join(jsonpath,'annotationfull_'+foldername+'.json')):
                continue   
            else:
                benchmark=json.load(open(os.path.join(jsonpath,'annotationfull_'+foldername+'.json')))
                detected=json.load(open(os.path.join(jsonpath,'annotation_'+foldername+'_'+modelname+'.json')))
    else:# detection result and benchmark pathes are specified
        benchmark=json.load(open(benchmarkpath))
        detected=json.load(open(detectpath))
    
    # change IoU threshold from 0 to 1.0, interval 0.05
    if 'gt22' in modelname:
        rejectsize=22
    else:
        rejectsize=0
        
    for confidence_thresh in threshlist:
    
        # initial the numbers for performance
        totalperformance[confidence_thresh]={'leading':{'tp':0, 'fp':0, 'tn':0, 'fn':0},
                                            'overall':{'tp':0, 'fp':0, 'tn':0, 'fn':0},
                                            'large':{'tp':0, 'fp':0, 'tn':0, 'fn':0},
                                            'medium':{'tp':0, 'fp':0, 'tn':0, 'fn':0},
                                            'small':{'tp':0, 'fp':0, 'tn':0, 'fn':0}}
                
# =============================================================================
#         for imgname in detected:
#             # if not detected
#             if len(detected[imgname])==0:
#                 annos_detect={}
#             else:
#                 annos_detect=detected[imgname]['annotations']
#         
#             # if no such a benchmark
#             if benchmark.get(imgname.split('_')[-1])==None:
#                 print('failed getting benchmark {}'.format(imgname))
#                 annos_benchmark={}
#             else:
#                 annos_benchmark=benchmark[imgname.split('_')[-1]]['annotations']
#             
#             
#             # calculate performance
#             totalperformance[confidence_thresh]=_get_tp_fp_fn(annos_detect,
#                             annos_benchmark,
#                             totalperformance[confidence_thresh],
#                             confidence_thresh,
#                             IOUthresh)
#             #totalperformance[confidence_thresh]=checkSingleImage(imgname,annos_benchmark,annos_detect,totalperformance[confidence_thresh],confidence_thresh)
# =============================================================================
  
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
        
            totalperformance[confidence_thresh]=_get_tp_fp_fn_extern(
                             annos_detect,
                             annos_benchmark,
                             totalperformance[confidence_thresh],
                             confidence_thresh,
                             IOUthresh,
                             rejectsize=rejectsize)

        # calculate precision, recall and missrate
        for key in totalperformance[confidence_thresh]:
            if totalperformance[confidence_thresh][key]['tp']+totalperformance[confidence_thresh][key]['fp']==0:
                #print('tp+fp=0, no leading label in benchmark, pass precision of leading')
                precision=0
            else:
                precision=totalperformance[confidence_thresh][key]['tp']/(totalperformance[confidence_thresh][key]['tp']+totalperformance[confidence_thresh][key]['fp'])
            
            if totalperformance[confidence_thresh][key]['tp']+totalperformance[confidence_thresh][key]['fn']==0:
                recall=0
            else:
                recall=totalperformance[confidence_thresh][key]['tp']/(totalperformance[confidence_thresh][key]['tp']+totalperformance[confidence_thresh][key]['fn'])
            
            totalperformance[confidence_thresh][key]['precision']=precision
            totalperformance[confidence_thresh][key]['recall']=recall
        
        
        print('Threshold {} done processing'.format(confidence_thresh))
        
    
    # save the performance into json file
    with open(os.path.join(filepath,'performance.json'),'w') as savefile:
        savefile.write(json.dumps(totalperformance, sort_keys = True, indent = 4))
    
    
    # plot the precision, recall and MRs over all the cars and leading cars
    
    plotPrecisionRecall(totalperformance, key='overall', 
                        modelname=modelname, savepath=filepath)
    plotPrecisionRecall(totalperformance, key='small', 
                        modelname=modelname, savepath=filepath)
    plotPrecisionRecall(totalperformance, key='medium', 
                        modelname=modelname, savepath=filepath)
    plotPrecisionRecall(totalperformance, key='large', 
                        modelname=modelname, savepath=filepath)
    
   
# =============================================================================
#     
#     chartaxis = [0.0,1.0,0.0,1.0]
#     
#     plt.figure(figsize=(6,6),dpi=100)
#     plt.axis(chartaxis)
#     plt.plot(threshlist,precision_leading_list,'b.-',label='Leading')
#     plt.plot(threshlist,precision_overall_list,'r.-',label='Overall')
#     plt.title(modelname+' '+partname+' precision')
#     plt.xlabel('Confidence threshold')
#     plt.legend(loc=1)
#     plt.savefig(os.path.join(filepath,'precision.png'))
#     plt.show()
#     
#     plt.figure(figsize=(6,6),dpi=100)
#     plt.axis(chartaxis)
#     plt.plot(threshlist,recall_leading_list,'bs-', label='Leading')
#     plt.plot(threshlist,recall_overall_list,'rs-',label='Overall')
#     plt.title(modelname+' '+partname+' recall')
#     plt.xlabel('Confidence threshold')
#     plt.legend(loc=1)
#     plt.savefig(os.path.join(filepath,'recall.png'))
#     plt.show()
#     
#     
#     #precision over recall of different confidence
#     plt.figure(figsize=(6,6),dpi=100)
#     plt.axis(chartaxis)
#     plt.plot(recall_leading_list,precision_leading_list,'bv-', label='Leading')
#     plt.plot(recall_overall_list,precision_overall_list,'rv-',label='Overall')
#     plt.title(modelname+' '+partname+'precision vs recall')
#     plt.xlabel('recall')
#     plt.ylabel('precision')
#     plt.legend(loc=1)
#     plt.savefig(os.path.join(filepath,'precision_vs_recall.png'))
#     plt.show()
# =============================================================================
    
    
""" End of file """