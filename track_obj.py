# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:27:15 2019

object tracking

for detecting app, we can apply some tracking frames after a single detection
frame.

further more, it's possible the apply multi-threads to boost the app, e.g. one
tracking thread, one detection thread, then update the tracking feature from 
the last detection result.

@author: Wen Wen
"""
import cv2
import os
import time

tracker_types = ['MIL',
                 'KCF', 
                 'TLD', 
                 'MEDIANFLOW', 
                 'MOSSE', 
                 'CSRT']
#cwd=os.getcwd()
medianflowparams = 'trackermedianflow.json'


class ObjTracker():
    """
    object tracker
    
    """
    def __init__(self):
        self.trackertype=None
        self.tracker=None
    
    def buildTracker(self,tracker_type='MEDIANFLOW'):
        if tracker_type not in tracker_types:
            raise ValueError('invalid tracker type')
        else:
            self.trackertype=tracker_type
            if self.trackertype == 'MIL':
                self.tracker = cv2.TrackerMIL_create() # not good, doesnt adjust bbx
            if self.trackertype == 'KCF':
                self.tracker = cv2.TrackerKCF_create() # lost from 5th frame, not adjust bbx
            if self.trackertype == 'TLD':
                self.tracker = cv2.TrackerTLD_create() # FP when object is gone, some issue when adjust bbx
            if self.trackertype == 'MEDIANFLOW':
                self.tracker = cv2.TrackerMedianFlow_create() # best one
                if medianflowparams:
                    fs = cv2.FileStorage(medianflowparams,cv2.FILE_STORAGE_READ)
                    fn = fs.getFirstTopLevelNode()
                    self.tracker.read(fn)
                    self.tracker.save('loaded_tracker_params.json')
            if self.trackertype == 'MOSSE':
                self.tracker = cv2.TrackerMOSSE_create() # when the size doesnt change, it's the fastest
            if self.trackertype == "CSRT":
                self.tracker = cv2.TrackerCSRT_create() # not good, doesnt adjust bbox
    
    def refreshTracker(self):
        if self.trackertype==None:
            raise ValueError('invalid tracker type')
        self.buildTracker(tracker_type=self.trackertype)
    
    def updateTrack(self,img,init=False,bbox=None):
        """
        update the tracking, if tracker not provided, use 'medianflow' as default 
        tracker, note that this function will only track a single frame
        
        input:
            img: the image to be detected
            tracker: tracker for the tracking, default is medianflow tracker
            init: if true, the tracker will be initialized
            bbox: needed if the tracker is to be initialized
        
        output:
            flag: if object is successfully tracked, return True, else return False
            bbox: the tracking result, return (xmin,ymin,width,height) if success 
                tracking, else return None
            tracktime: time used for tracking
        
        """
        if self.tracker==None:
            raise ValueError('tracker not defined')
        if init:
            self.tracker.init(img,bbox)
            flag=True
            tracktime=0
        else:
            start=time.time()
            flag, bbox = self.tracker.update(img)
            tracktime=(time.time()-start)
        
        return flag, bbox, tracktime

if __name__=='__main__':
    imgpath='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/Part3_videoframes/VNX_10009'
    bench=(304,192,331-304,218-192)
    bbox=(304,192,331-304,218-192) # [xmin,ymin,width,height]
    tracker_type = tracker_types[3]
    start=0    
    
    # build tracker
    objtracker=ObjTracker()
    objtracker.buildTracker()
    
    if not os.path.exists(os.path.join(imgpath,'track')):
        os.mkdir(os.path.join(imgpath,'track'))        
    totaltime=0
    imglist=sorted(os.listdir(imgpath))
    for i in range(start,len(imglist)):
        if '.png' not in imglist[i]:
            continue
        img=cv2.imread(os.path.join(imgpath,imglist[i]))
        if i%20==0:
            # suppose after every 20 frames there is a detection frame
            objtracker.refreshTracker()
            objtracker.updateTrack(img,init=True,bbox=bench)
        else:
            flag, bbox, tracktime = objtracker.updateTrack(img)
            totaltime += tracktime
            if not flag:
                print('frame {} tracklost'.format(i))
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2, 1)
        cv2.imwrite(os.path.join(imgpath,'track',imglist[i].replace('.png','_track.png')), img)
    print('avg processing time= {} s'.format(totaltime/len(imglist)))   