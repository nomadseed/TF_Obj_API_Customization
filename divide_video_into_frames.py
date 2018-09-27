# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:47:43 2018

divide_video_into_frames

@author: Wen Wen
"""
import cv2
import os
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
                        default='Part3_videos', 
                        help="File path of input data") 
    args=parser.parse_args()
    
    filepath=args.file_path
    videodict=os.listdir(filepath)
    
    for videoname in videodict:
        if 'mkv' in videoname:
            print('processing: '+videoname)
            foldername=os.path.join(filepath,videoname.split('.')[0])
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            vidcap = cv2.VideoCapture(os.path.join(filepath,videoname))
            count = 0
            success, image = vidcap.read()
            while (vidcap.isOpened() and success):
                imagename=os.path.join(foldername,
                    videoname.replace('.mkv','_'+str(count).zfill(5)+'.png'))
                cv2.imwrite(imagename, image)     # save frame as JPEG file
                
                count += 1
                success, image = vidcap.read()
            print('for {}, {} image saved'.format(videoname,count))
            vidcap.release()

""" End of File """