# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:38:46 2020

IoU heat map

given a set of kmeans centroids, calculate the IoU of all the possible 
boxes in detection, compute the mean IoU and then draw the heat map for
visualy judging which set of centroids is better

@author: Wen Wen
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import check_performance as chp


savedKmeanList={'ssd-strip-gt22':[
                    [16,13],[23,15],[18,20],
                    [25,20],[50,23],[33,38],
                    [51,47],[68,68],[43,65],[110,34],
                    [115,107],[153,63],[99,78],[78,101],
                    [146,126],[200,161],[216,96],[128,170],
                    [221,230],[180,280],[291,174],[271,260],[251,195]                    
                    ],
                'ssd-regular-kmeans':[
                    [6,6],
                    [18,13],[11,10],
                    [33,27],[29,20],[17,22],
                    [33,51],[71,20],[41,35],
                    [52,52],[55,77],[114,22],[71,60],
                    [84,84],[159,57],[120,134],[165,113],[109,63],[236,77],
                            [120,97],[231,161],[153,178],[83,112]
                    ],
                'ssd-regular-kmeans-gt22':[
                    [17,15],
                    [26,20],[36,30],[22,29],[30,44],
                    [65,59],[44,61],[57,83],[48,43],[79,26],
                    [80,81],[130,34],[100,61],[105,93],[139,80],[83,115],
                    [180,65],[250,80],[125,160],[171,203],[175,131],[128,119],
                    [246,155]
                    ],
                # the original anchors are acquired from 'multiple_grid_anchor_generator.py'
                # basesize=256
                # num_layers=6,
                # min_scale=0.2,
                # max_scale=0.95
                # aspect_ratios=(1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3)
                'ssd-origin':[
                    [51, 51], [89, 89], [128, 128], [166, 166], [204, 204], 
                    [243, 243], [72, 36], [126, 63], [180, 90], [235, 117], 
                    [289, 144], [343, 171], [88, 29], [155, 51], [221, 73], 
                    [288, 96], [354, 118], [421, 140], [36, 72], [63, 126], 
                    [90, 180], [117, 235], [144, 289], [171, 343], [29, 88], 
                    [51, 155], [73, 221], [96, 288], [118, 354], [140, 421]
                    ],
                'ssd-GPDF-gt22':[
                    [24,8], [45,14], [90,29], [ 150,48], [ 225,72], 
                    [ 450,144], [ 15,12], [ 29,23], [ 57,46], [ 95,76], 
                    [ 143,114], [ 285,228], [ 12,15], [ 23,29], [ 45,58], 
                    [ 75,96], [ 113,144], [ 225,288], [ 10,18], [ 19,34], 
                    [ 38,68], [ 64,113], [ 96,169], [ 192,338], [ 9,20], 
                    [ 17,38], [ 34,76], [ 57,127], [ 85,191], [ 170,382]
                    ]
                }


def load_json_annotations(filepath, jsonlabel):
    annotationdict={}
    folderdict=os.listdir(filepath)
    for foldername in folderdict:
        jsonpath=os.path.join(filepath,foldername)
        if '.' in jsonpath:
            continue
        # load the json files
        jsondict=os.listdir(jsonpath)
        for jsonname in jsondict:
            if jsonlabel in jsonname and '.json'in jsonname:
                annotationdict[foldername]={}
                annotationdict[foldername]=json.load(open(os.path.join(jsonpath,jsonname)))
    
    return annotationdict

def get_bboxes(annotationdict):
    '''
    
    anchorratios=[{'class':'car',
                   'image':'xxx.jpg',
                   'ratio':0.5}, {}, {} ]
    '''
    bboxlist=[]
    for foldername in annotationdict:
        for imagename in annotationdict[foldername]:
            if len(annotationdict[foldername][imagename])==0:
                continue
            else:
                for anno in annotationdict[foldername][imagename]['annotations']:
                    if anno['width'] and anno['height']:
                        bboxlist.append([round(anno['height']*0.625),round(anno['width']*0.46875)])
                
    return bboxlist

def getIoUMats(KmeanList, boxes, fast=True):
    """
    given the centroids lists of kmeans result and ground truth boxes, compute
    the 300x300 iou matrix
    
    """
    
    ioumat_dict={}
    
    for model_name in KmeanList:
        centroids = savedKmeanList[model_name]
        ioumat=np.zeros([300,300])
        meaniou=[]
        for [w,h] in boxes:
            # regulation for rounded indices
            if w==300:
                w=299
            if h==300:
                h=299
            
            # fast and slow methods
            if fast:
                if ioumat[w,h]==0:# skip if already calculated (!=0)
                    bbx1={'x':0, 'y':0, 'width':w, 'height':h}
                    ioulist=[]
                    for [c_w,c_h] in centroids:
                        bbx2={'x':0,'y':0,'width':c_w,'height':c_h}
                        ioulist.append(chp.getIoU(bbx1,bbx2))
                    ioumat[w,h]=max(ioulist)
            else:
                bbx1={'x':0, 'y':0, 'width':w, 'height':h}
                ioulist=[]
                for [c_w,c_h] in centroids:
                    bbx2={'x':0,'y':0,'width':c_w,'height':c_h}
                    ioulist.append(chp.getIoU(bbx1,bbx2))
                iou=max(ioulist)
                meaniou.append(iou)
                ioumat[w,h]=iou
                
        # for each model, save iou matrix and print meaniou        
        ioumat_dict[model_name]=ioumat
        if not fast:
            print('model: {}'.format(model_name))
            print('mean IoU: {}'.format(np.mean(np.array(meaniou))))
        
    return ioumat_dict
                
                
def generateAlltheBoxes():
    """
    generate all the possible boxes with a certain MAP_SIZE
    
    """
    data=[[i,j] for i in range(0,300) for j in range(0,300)]
    return data

def plotHeatMap(ioumat_dict, colormap,savepath):
    """
    plot heat map with iou matrixes and self-defined colormap
    
    """
    
    col=len(ioumat_dict)
    fig, axs = plt.subplots(1, col, figsize=(4.5*col, 4))
    
    if col>1:
        
        for [ax, modelname] in zip(axs, ioumat_dict):
            psm = ax.pcolormesh(ioumat_dict[modelname], cmap=colormap, rasterized=True, vmin=0, vmax=1)
            fig.colorbar(psm, ax=ax)
            ax.set_title(modelname)

    plt.savefig(os.path.join(savepath,'IoU heat map.png'),dpi=60)


PATH_DICT={'bdd':{'path':'D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k',
                  'label':'bdd100k_labels_images_train_VIVA_format_crop_gt22.json'},
           'cal':{'path':'D:/Private Manager/Personal File/uOttawa/Lab works/2019 summer/caltech_dataset',
                  'label':'caltech_annotation_VIVA_train.json'}
           }

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
                        default=PATH_DICT['bdd']['path'], 
                        help="File path of input data")
    #D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k
    #D:/Private Manager/Personal File/uOttawa/Lab works/2019 summer/caltech_dataset
    
    parser.add_argument('--json_label', type=str, 
                        default=PATH_DICT['bdd']['label'], 
                        help="label to specify json file")
    #bdd100k_labels_images_val_VIVA_format_crop.json
    #caltech_annotation_VIVA_test.json
    
    args = parser.parse_args()
    
    filepath = args.file_path
    jsonlabel = args.json_label
    savepath = os.path.join(filepath,'ssd_cluster_result')
    
    # set colormap
    colormap = cm.get_cmap('viridis', 32)
    #print(viridis.colors)

    # get annotations of whole dataset
    annotationdict = load_json_annotations(filepath, jsonlabel)
    
    # get bboxlist from annotation
    bboxlist = get_bboxes(annotationdict) # enable this for actual data
    #bboxlist = generateAlltheBoxes() # enable this for all possible data
    
    
    
    # compute iou matrix between bboxes and the centroid lists
    ioumat_dict_1={}
    
    # compute iou matrix between 300x300 possible bboxes and the centroid lists
    KmeanList=['ssd-origin','ssd-GPDF-gt22','ssd-regular-kmeans-gt22','ssd-strip-gt22']
    ioumat_dict_2 = getIoUMats(KmeanList,bboxlist,fast=False)
    
    
    
    
    
    # plot heat map over all the centroid lists
    plotHeatMap(ioumat_dict_2, colormap, savepath)
    