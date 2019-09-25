# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:11:27 2019

load acc radar data (in vsf format) and prediction result, compare the distance
estimation error

@author: Wen Wen
"""

import numpy as np
import os
import json
from matplotlib import pyplot as plt

def parseString2ArrayExtra(filename, string):
    '''
    the xlist begins with start date, start time, log version
    for each row, it begins with '\nX', where X is the row number, e.g. '\n4' 
    is the 4th row. at the end of the xlist there is a '\n' to show the end of 
    the file
    
    '''
    xlist=string.split('\t')
    checkend=xlist.pop()
    if checkend!='\n':
        raise IOError('the file loaded is broken and doesn\'t have an end mark')
    xlist.reverse()
    
    
    # build the header
    # header format [filename, start date, start time, log version]
    header=[filename]
    for i in range(3):
        header.append(xlist.pop())
    
    # build the table
    # table format, for each row we have 8 columns of:
    # [frame number, vehicle stability data, lon accel, lat accel, 
    #   adaptive cruise data, vehicle detected, vehicle ahead distance,
    #   vehicle ahead speed]
    # for the viewnyx project we need the last 3 columns
    table_2d = []
    if len(xlist)%8!=0:
        raise ValueError('the file loaded is broken and have imcomplete rows')
    print('{} has {} frames'.format(filename, int(len(xlist)/8)))
    curlist=[]
    while len(xlist)!=0:
        elem=xlist.pop()
        if '\n' in elem:
            elem=elem.split('\n')[1]
        curlist.append(elem)
        if len(xlist)%8==0:
            table_2d.append(curlist)
            curlist=[]
    
    return header, table_2d

def loadAccData(accpath):
    acclist=os.listdir(accpath)
    accdict={}
    for accname in acclist:
        if 'extra' in accname:
            
            print('loading '+accname)
            with open(os.path.join(accpath,accname), 'r') as fopen:
                string=fopen.read()
                fopen.close()
            accexp={}
            accexp['header'], accexp['table'] = parseString2ArrayExtra(accname, string)
            accdict[accname.split('.')[0]]=accexp
    return accdict  

def loadJsonResults(filepath, jsonlabel=''):
    """
    load prediction and tracking results in json format
    
    """
    detectdict={}
    trackdict={}
    filelist=os.listdir(filepath)
    print('loading prediction & tracking files')
    for filename in filelist:
        if jsonlabel in filename and 'distance' in filename:
            if 'detection' in filename:
                # loading detection results
                detectdict[filename]=json.load(open(os.path.join(filepath,filename)))
            elif 'tracking' in filename:
                # loading tracking results
                trackdict[filename]=json.load(open(os.path.join(filepath,filename)))
    print('loading completed')
    return detectdict, trackdict

def calculateError(acc_table, detect_table, track_table, jsonlabel='',
                   error_type='percent',round_flag=True):
    """
    load the ACC rader data as ground trugh, then calculate errors for frame by
    frame prediction and tracking results.
    
    table format, for each row we have 8 columns of:
    [frame number, vehicle stability data, lon accel, lat accel, 
     adaptive cruise data, vehicle detected, vehicle ahead distance,
     vehicle ahead speed]
    for the viewnyx project we need the last 3 columns
    
    calculate the error when distance of the forward leading car is smaller 
    than 55 meters. Also, the folder VYX_1002_1003649 is a perfect example, in
    which the acc lose detection occationally but the camera always catch the
    leading car.
    
    for each error dict, the errors are grouped as [0,5], (5,10], (10,20], 
    (20,30], (30,40], (40,50], (50,inf), and the union set of [0, inf)
    
    the ACC distance is always an integer, unit is meters. both the estimated 
    distance from detection or tracking have float precision, unit is milimeters
    
    """
    detect_error_dict = {'0':{},'5':{},'10':{},'20':{},'30':{},'40':{},'50':{},'all':{}}
    track_error_dict = {'0':{},'5':{},'10':{},'20':{},'30':{},'40':{},'50':{},'all':{}}
    for foldername in acc_table:
        for frameinfo in acc_table[foldername]['table']:
            frame_index = int(frameinfo[0])-1 # convert str to int
            acc_dist = float(frameinfo[6]) # 0-255 meters
            # load estimated distances
            detect_name='distance_{}_detection{}.json'.format(foldername,jsonlabel)
            if frame_index>=len(detect_table[detect_name]):
                continue # sometimes the ACC data has 1 more frame than video frames, skip them
            imagename=foldername+'_'+str(frame_index).zfill(5)+'.png'
            detect_dist = detect_table[detect_name][imagename]
            
                
            """
            load tracking results, to be done
            """
            
            # calculate error in percentage and save
            if acc_dist != -1 and detect_dist!=999999: 
                if error_type=='percent':
                    # valid distance in ACC and estimation data
                    if round_flag:
                        detect_dist=round(detect_dist/1000.0)
                        error = (detect_dist-acc_dist)/acc_dist
                    else:
                        detect_dist=detect_dist/1000.0
                        error = (detect_dist-acc_dist)/acc_dist
                elif error_type=='abs':
                    detect_dist=round(detect_dist/1000.0)
                    error = abs(detect_dist-acc_dist)
                # union set
                if acc_dist>0:
                    detect_error_dict['all'][imagename]=[error, detect_dist,acc_dist]
                
                # grouped distances
                if acc_dist>50:
                    detect_error_dict['50'][imagename]=[error, detect_dist,acc_dist]
                elif acc_dist>40:
                    detect_error_dict['40'][imagename]=[error, detect_dist,acc_dist]
                elif acc_dist>30:
                    detect_error_dict['30'][imagename]=[error, detect_dist,acc_dist]
                elif acc_dist>20:
                    detect_error_dict['20'][imagename]=[error, detect_dist,acc_dist]
                elif acc_dist>10:
                    detect_error_dict['10'][imagename]=[error, detect_dist,acc_dist]
                elif acc_dist>5:
                    detect_error_dict['5'][imagename]=[error, detect_dist,acc_dist]
                elif acc_dist>0:
                    detect_error_dict['0'][imagename]=[error, detect_dist,acc_dist]
    
    return detect_error_dict, track_error_dict

def errorStatistics(detect_error):
    """
    for each error dict, the errors are grouped as [0,5], (5,10], (10,20], 
    (20,30], (30,40], (40,50], (50,inf), and the union set of [0, inf)
    
    calculate mean, max, min, MSE of errors
    
    """
    stat={}
    for groupname in detect_error:
        stat[groupname]={}
        error_list=[]
        # form a list or all the errors
        for filename in detect_error[groupname]:
            error_list.append(detect_error[groupname][filename][0])
        error_list=np.array(error_list)
        
        #calculate statistics
        stat[groupname]['mean']=np.average(error_list)
        stat[groupname]['max']=np.max(error_list)
        stat[groupname]['min']=np.min(error_list)
        stat[groupname]['mse']=np.square(error_list).mean()
    
    return stat

def plotErrors(statdict,jsonlabellist,savepath,titleattach=''):
    """
    plot a single error figure for 
    
    """
    # get x_label from jsonlabellist
    x_label=[int(i.split('_')[1]) for i in jsonlabellist]
    
    plot_label_dist = ['all','0','5','10','20','30','40','50']
    plot_label_stat = ['mean','max','min','mse']
    legendlist={'all':  [1,         0,          0      ],
                '0':    [0.14509804, 0.24705882, 0.81176471],
                '5':    [0.14509804, 0.63921569, 0.81176471],
                '10':   [0.81176471, 0.14509804, 0.68235294],
                '20':   [0.02352941, 0.83529412, 0.73333333],
                '30':   [0.31372549, 0.96470588, 0.05098039],
                '40':   [0.88627451, 0.85490196, 0.03137255],
                '50':   [0.96470588, 0.7372549 , 0.13333333]
            }
    legendloc={'mean':1,
               'max':1,
               'min':1,
               'mse':1
               }
    
    # plot mean error for all distances
    for status in plot_label_stat:
        plt.figure(figsize=(8,6),dpi=100)
        for distance in plot_label_dist:
            errorlist=[]
            for label in x_label:
                x_label_str='_'+str(label)
                errorlist.append(statdict[x_label_str][distance][status])
            if distance=='all':
                plt.plot(x_label,errorlist,marker='*',markersize=15,color=legendlist[distance],label=distance)
            else:
                plt.plot(x_label,errorlist,marker='o',color=legendlist[distance],label=distance)
        plt.title('Distance Estimation Error -- {}, {}'.format(status.upper(),titleattach))
        plt.xlabel('baseline vehicle width')
        plt.ylabel('error')
        plt.legend(loc=legendloc[status])
        if savepath is not None:
            plt.savefig(os.path.join(savepath,'{}_error_{}.png'.format(status,titleattach)))
        plt.show()
    return 0 


if __name__=='__main__':
    accpath='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/Part4_ACC_Videos'
    predpath='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/Part4_ACC'
    jsonlabellist=['_140','_150','_160','_170',
                   '_180','_190','_200','_210','_220']
    error_type=['abs','percent']
    
    statdict = {}
    # load acc data
    acc_table = loadAccData(accpath)
    
    # evaluate miss rate of detection
    
    # evaluate miss rate of 
    
    # evaluations
    for etype in error_type:
        # evaluate estimation results of all the baseline widths
        for jsonlabel in jsonlabellist:
            # load prediction results
            detect_table, track_table = loadJsonResults(predpath, jsonlabel)
            
            # calculate detection/tracking error with ACC radar as ground truth
            detect_error, track_error = calculateError(acc_table, detect_table, 
                                                       track_table, jsonlabel,
                                                       error_type=etype,
                                                       round_flag=True)
            # calculate mean, max, min, MSE of errors
            statdict[jsonlabel] = errorStatistics(detect_error)
        
        # plot figures
        plotErrors(statdict,jsonlabellist,savepath=predpath,titleattach=etype)
    
    
    

""" End of File """