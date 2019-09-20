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

def loadJsonResults(filepath, jsonlabel='distance'):
    """
    load prediction and tracking results in json format
    
    """
    detectdict={}
    trackdict={}
    filelist=os.listdir(filepath)
    print('loading prediction & tracking files')
    for filename in filelist:
        if jsonlabel in filename:
            if 'detection' in filename:
                # loading detection results
                detectdict[filename]=json.load(open(os.path.join(filepath,filename)))
            elif 'tracking' in filename:
                # loading tracking results
                trackdict[filename]=json.load(open(os.path.join(filepath,filename)))
    print('loading completed')
    return detectdict, trackdict

def calculateError(acc_table, detect_table, track_table, jsonlabel='distance',
                   round_flag=True):
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
    
    for each error dict, the errors are grouped as [0,10], (10,20], (20,30],
    (30,40], (40,50], (50,inf)
    
    the ACC distance is always an integer, unit is meters. both the estimated 
    distance from detection or tracking have float precision, unit is milimeters
    
    """
    detect_error_dict = {'0':{},'10':{},'20':{},'30':{},'40':{},'50':{}}
    track_error_dict = {'0':{},'10':{},'20':{},'30':{},'40':{},'50':{}}
    for foldername in acc_table:
        for frameinfo in acc_table[foldername]['table']:
            frame_index = int(frameinfo[0])-1 # convert str to int
            acc_dist = float(frameinfo[6]) # 0-255 meters
            # load estimated distances
            detect_name='distance_'+foldername+'_detection.json'
            if frame_index>=len(detect_table[detect_name]):
                continue # sometimes the ACC data has 1 more frame than video frames, skip them
            imagename=foldername+'_'+str(frame_index).zfill(5)+'.png'
            detect_dist = detect_table[detect_name][imagename]
            
                
            """
            load tracking results, to be done
            """
            
            # calculate error in percentage and save
            if acc_dist != -1 and detect_dist!=999999: 
                # valid distance in ACC and estimation data
                if round_flag:
                    detect_dist=round(detect_dist/1000.0)
                    error = (detect_dist-acc_dist)/acc_dist
                else:
                    detect_dist=detect_dist/1000.0
                    error = (detect_dist-acc_dist)/acc_dist
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
                elif acc_dist>0:
                    detect_error_dict['0'][imagename]=[error, detect_dist,acc_dist]
    
    return detect_error_dict, track_error_dict

def errorStatistics(detect_error):
    """
    for each error dict, the errors are grouped as [0,10], (10,20], (20,30],
    (30,40], (40,50], (50,inf)
    
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

if __name__=='__main__':
    accpath='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/Part4_ACC_Videos'
    predpath='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/Part4_ACC'
    jsonlabel='distance'
    
    # load acc data
    acc_table = loadAccData(accpath)

    # load prediction results
    detect_table, track_table = loadJsonResults(predpath, jsonlabel)
    
    # calculate detection/tracking error with ACC radar as ground truth
    detect_error, track_error = calculateError(acc_table, detect_table, 
                                               track_table, jsonlabel,
                                               round_flag=True)
    # calculate mean, max, min, MSE of errors
    stat = errorStatistics(detect_error)
    
    

""" End of File """