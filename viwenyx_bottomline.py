# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 13:46:23 2018

get the bottom line of the cars from annotations in viewnyx dataset
just to know how to use the json file of bounding boxes, and then print them out
don't need to save the bottom lines into extra json files

@author: Wen Wen
"""

import json
import os
import matplotlib.pyplot as plt
import scipy.misc as scimisc

def DrawLineNSave(filepath,filename,pos_y):
    img=scimisc.imread(filepath+filename)
    for i in range(0,640):
       img[pos_y][i][0]=0
       img[pos_y][i][1]=255
       img[pos_y][i][2]=0
    scimisc.imsave(filepath+filename.split('.')[0]+'_bottom.jpg',img)

def GetJPGDict(filepath):
    jpglist=os.listdir(filepath) 
    jpgdict={}
    count=0
    for i in jpglist:
        if 'jpg' in i and 'o' not in i: # exclude the horizon and bottom img
            jpgdict[count]=i
            count+=1
    return jpgdict

if __name__=='__main__':
    
    filepath='Frame Images/video14/'
    filename='video14-3.json'
    
    jpgdict=GetJPGDict(filepath)
    annotationdict={}
    
    with open(filepath+filename, 'r') as readfile:
        annotationdict=json.load(readfile)
    
    for imgindex in jpgdict:
        img=scimisc.imread(filepath+jpgdict[imgindex])
        for bbxdict in annotationdict[jpgdict[imgindex]]:
            x=int(bbxdict['x'])
            y=int(bbxdict['y'])
            width=int(bbxdict['width'])
            height=int(bbxdict['height'])
            try:
                for i in range(x,x+width):
                    img[y+height][i][0]=0
                    img[y+height][i][1]=255
                    img[y+height][i][2]=0
            except IndexError:
                if x+width>639:
                    max_x=639
                else:
                    max_x=x+width
                for i in range(x,max_x):
                    img[y+height][i][0]=0
                    img[y+height][i][1]=255
                    img[y+height][i][2]=0
        scimisc.imsave(filepath+jpgdict[imgindex].split('.')[0]+'_bottom.jpg',img)


''' End of File '''
