# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:40:44 2018

test the performance of the trained model with checkpoint and saved model
# this api is for tensorflow version 1.10.0 or later
args:
    (please check the help info in the main function, which is at the bottom of
    this script)
    
usage example:
        
    python3 model_test.py --testimg_path= /YOUR/IMG/PATH 
    --ckpt_path=/PATH/tf-object-detection-api/research/viewnyx/ckpt_ssd_opt_vnx_finetune/export/frozen_inference_graph.pb 
    --label_path=/PATH/tf-object-detection-api/research/viewnyx/data/class_labels.pbtxt 
    --cam_calibration_path=/PATH/cam_mapping_viewnyx.txt 
    --class_number 1 --output_thresh 0.05
    --use_tracking True
    

@author: Wen Wen
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
import json
import time
import track_obj
import myGreedyNMS

from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops

# Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def calculate_trainable_variables():
    total_parameters = 0
    print('================= Trainable Variables =======================')
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        #print(variable_parameters)
        total_parameters += variable_parameters
    print('total trainable parameters: {}'.format(total_parameters))

def getClass(class_id,category_index,max_class):
    """
    given class id, get the class name
    
    """
    accepted_classes={'car','van','truck','bus'}
    if class_id<=0:
        raise ValueError('class id cannot be 0 or float numbers')
    elif class_id>max_class:
        return 'null'
    else:
        if  category_index[class_id]['name'] is not None:
            if category_index[class_id]['name'] in accepted_classes:
                return 'car'
            else:
                return 'null'
        else:
            raise ValueError('class doesn\'t have a valid display name or binary name')

def carClassifier(x,y,width,height,threshold=0.2, strip_x1=305,strip_x2=335,y_roof=0):
    """
    input the information of bounding box (topleft, bottomright), get the possible
    category of the detected object. the method used here could be some meta-algorithm
    like SVM or Decision Tree.
    
    categories using: 'leading','sideways'
    
    threshold is the overlapping area of bbx and vertical strip, divided by overlapping 
    area between vertical strip and horizontal extension of bbx
    
    if the overlapping percentage is above threshold, return 'leading', else 
    return 'sideways'
    
    """
    
    x1=x
    x2=x+width
    category=''
    if threshold<=0 or threshold>1:
        threshold=0.5
    if y+height<y_roof:
        category='sideways'
        return category
    if x1>strip_x1 and x2<strip_x2:
        category='leading'
    elif x2-strip_x1>(strip_x2-strip_x1)*threshold and x1-strip_x2<-(strip_x2-strip_x1)*threshold:
        category='leading'
    else:
        category='sideways'
    
    return category

def returnbottomy(bbx):
    return bbx['y']+bbx['height']

def loadImageInNpArray(image):
    # convert input image into h*w*3 in uint8 format, BGR color space
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def detectSingleImage(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def updateAnnotationDict_Raw(output_dict,annotationdict,
                         imagename,im_width,im_height,
                         max_class,category_index,
                         outputthresh=0.05,IOUthresh=0.5):
    """
    update annotation dictionary with raw box lacations and
    scores using my NMS
    
    """
    
    # loading raw result from model
    boxes=output_dict['Postprocessor/raw_box_locations'][0]
    scores=output_dict['Postprocessor/raw_box_scores'][0]
    boxlist, _ = myGreedyNMS.sortByScore(scores,boxes)
    
    # Format of NMSed_list:
    # [y_min, x_min, y_max, x_max, highest_score, predicted_class]
    # the first 4 values are of range [0.0 1.0], highest_score [0.0 1.0]
    # predicted_class is class code, which should be an integer. If class code
    # is 0, means it's background
    NMSed_list = myGreedyNMS.greedyNonMaximumSupression(boxlist,
                            clipthresh=outputthresh,
                            IOUthresh=IOUthresh)
    
    ########### save detection result (output_dict) ############
    ###### into jsondict in the format of VIVA Annotation ######
    annotationdict[imagename]={}
    annotationdict[imagename]['name']=imagename
    annotationdict[imagename]['width']=im_width
    annotationdict[imagename]['height']=im_height
    annotationdict[imagename]['annotations']=[]
    
    for i in range(len(NMSed_list)):
        annodict={}
        annodict['id']=i
        annodict['shape']=['Box',1]
        annodict['label']=getClass(NMSed_list[i][5],category_index,max_class)
        if annodict['label']=='null':
            continue
        ymin,xmin,ymax,xmax=NMSed_list[i][0:4]
        annodict['x']=int(xmin*im_width)
        annodict['y']=int(ymin*im_height)
        annodict['width']=int((xmax-xmin)*im_width)
        annodict['height']=int((ymax-ymin)*im_height)
        annodict['category']=carClassifier(annodict['x'],annodict['y'],annodict['width'],annodict['height'])
        annodict['score']=float(NMSed_list[i][4])
        
        annotationdict[imagename]['annotations'].append(annodict)    
        
    return annotationdict, boxes, scores

def updateAnnotationDict_Track(annotationdict,imagename,bbox):
    """
    update annotation dictionary using the result of object
    tracker. only save one 'leading' bbox into annotation.
    
    bbox format:[xmin,ymin,width,height]
    
    this is not for detection!
    
    """
    im_width=640
    im_height=480
    annotationdict[imagename]={}
    annotationdict[imagename]['name']=imagename
    annotationdict[imagename]['width']=im_width
    annotationdict[imagename]['height']=im_height
    annotationdict[imagename]['annotations']=[]
    
    annodict={'id':0,
             'label':'car',
             'category':'leading',
             'x':round(bbox[0]),
             'y':round(bbox[1]),
             'width':round(bbox[2]),
             'height':round(bbox[3]),
             'shape':['box',1]
            }
    annotationdict[imagename]['annotations'].append(annodict)
    
    
    return annotationdict

def updateAnnotationDict(output_dict,annotationdict,imagename,im_width,im_height,
                         max_class):
    """
    update the annotation dictionary with standard built-in 
    NMS in the object detecton API
    
    """
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    ########### save detection result (output_dict) ############
    ###### into jsondict in the format of VIVA Annotation ######
    annotationdict[imagename]={}
    annotationdict[imagename]['name']=imagename
    annotationdict[imagename]['width']=im_width
    annotationdict[imagename]['height']=im_height
    annotationdict[imagename]['annotations']=[]
    
    for i in range(output_dict['num_detections']):
        if output_dict['detection_scores'][i] < outputthresh:
            continue
        else:
            annodict={}
            annodict['id']=i
            annodict['shape']=['Box',1]
            annodict['label']=getClass(output_dict['detection_classes'][i],category_index,max_class)
            if annodict['label']=='null':
                continue
            ymin,xmin,ymax,xmax=output_dict['detection_boxes'][i]
            annodict['x']=int(xmin*im_width)
            annodict['y']=int(ymin*im_height)
            annodict['width']=int((xmax-xmin)*im_width)
            annodict['height']=int((ymax-ymin)*im_height)
            annodict['category']=carClassifier(annodict['x'],annodict['y'],annodict['width'],annodict['height'])
            annodict['score']=float(output_dict['detection_scores'][i])
            
            annotationdict[imagename]['annotations'].append(annodict)    
        
    return annotationdict

def keepOnlyOneLeading(annotationdict,imagename):
    """
    for all the vehicles considered as vehicle, keep only the nearest one,
    and remark all the others as sideways.
    note this is an top level strategy which is after the detection, and thus
    doesn't interfere the detection result
    
    input:
        annotationdict: dict to store all annotation
        imagename: current image name
    output:
        annotationdict:
        not leadingflag: return true if leading car is detected
    """    
    annotationdict[imagename]['annotations'].sort(key=returnbottomy,reverse=True)
    leadingflag = True
    bbox=(0,0,0,0)
    for i in range(len(annotationdict[imagename]['annotations'])):
        if leadingflag and annotationdict[imagename]['annotations'][i]['category']=='leading':
            leadingflag=False
            bbox=(annotationdict[imagename]['annotations'][i]['x'],
                  annotationdict[imagename]['annotations'][i]['y'],
                  annotationdict[imagename]['annotations'][i]['width'],
                  annotationdict[imagename]['annotations'][i]['height']
                    )
        else:
            # caution!!! this step will change the annotation result!!!
            annotationdict[imagename]['annotations'][i]['category']='sideways'
    
    return annotationdict, not leadingflag, bbox

def drawBBoxNSave(image_np,imagename,savepath,annotationdict,drawside=False,
                  dist_estimator=None, show_leading=False,show_dist=True):
    img=cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    font=cv2.FONT_HERSHEY_SIMPLEX
    linetype=cv2.LINE_AA
    linewidth=2
    distance=999999 # a default value if no leading car appears in the image
    for i in range(len(annotationdict[imagename]['annotations'])):
        tl=(annotationdict[imagename]['annotations'][i]['x'],annotationdict[imagename]['annotations'][i]['y'])
        br=(annotationdict[imagename]['annotations'][i]['x']+annotationdict[imagename]['annotations'][i]['width'],annotationdict[imagename]['annotations'][i]['y']+annotationdict[imagename]['annotations'][i]['height'])
        if annotationdict[imagename]['annotations'][i]['category']=='leading':
            # draw leading car in red
            if show_leading:
                img=cv2.rectangle(img,tl,br,(0,0,255),linewidth) # red
            else:
                img=cv2.rectangle(img,tl,br,(0,255,0),linewidth) # green
            #cv2.putText(img, 'leading', tl, font, 1, (0,0,255), 1, lineType=linetype)
            if dist_estimator is not None:
                bl=(annotationdict[imagename]['annotations'][i]['x'],annotationdict[imagename]['annotations'][i]['y']+annotationdict[imagename]['annotations'][i]['height']-4)
                distance=dist_estimator.estimateDistance(width=annotationdict[imagename]['annotations'][i]['width'])
                if show_dist:
                    #cv2.putText(img, 'd={:.2f}'.format(distance/1000), bl, font, 0.5, (255,255,255), 1, lineType=linetype)
                    cv2.putText(img, 'w={} pels'.format(annotationdict[imagename]['annotations'][i]['width']), bl, font, 0.5, (255,255,255), 1, lineType=linetype)
        elif drawside:
            # draw sideway cars in green
            img=cv2.rectangle(img,tl,br,(0,255,0),linewidth) # green

    cv2.imwrite(os.path.join(savepath,imagename.split('.')[0]+'_leadingdetect.jpg'),img) # don't save it in png!!!
    return distance

def drawBBoxNSave_Track(image_np,imagename,savepath,bbox,
                        last_dist,last_time,detect_time,dist_estimator=None,
                        saveimg_flag=False):
    """
    bbox=(x,y,width,height)
    """
    img=cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    font=cv2.FONT_HERSHEY_SIMPLEX
    linetype=cv2.LINE_AA
    tl=(int(bbox[0]),int(bbox[1]))
    br=(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))
    distance=999999
    if bbox[2]!=0:
        img=cv2.rectangle(img,tl,br,(0,0,255),2) # red
        if dist_estimator is not None:
            #bl=(int(bbox[0]),int(bbox[1]+bbox[3]-4))
            distance=dist_estimator.estimateDistance(width=int(bbox[2]))

            # use real detection time for real camera-captured video, for our
            # demo, however, the video are all of 10 fps sample frequency
            _,img=raiseAlert(distance,0.1,last_dist,0.1,
                              img,abs_dist_only=False)
# =============================================================================
#             _,img=raiseAlert(distance,detect_time,last_dist,last_time,
#                               img,abs_dist_only=True)
# =============================================================================
            
            cv2.putText(img, 'Distance: {:.1f}m'.format(distance/1000), (4,456), font, 0.5, (255,255,255), 1, lineType=linetype)
    if saveimg_flag:
        cv2.imwrite(os.path.join(savepath,imagename.split('.')[0]+'_leadingdetect.jpg'),img) # don't save it in png!!!
    return distance
    
def raiseAlert(dist,t,last_dist,last_t,img,abs_dist_only=True,timeahead=0.6):
    """
    raise alert according to absolute distance and high acceleration
    the distance unit is milimeters, time unit is seconds. always show the 
    higher danger level
    
    input:
        dist/last_dist: distance for current frame/last frame
        t/last_t: detection time for current frame/last frame
        img: current frame
        abs_dist_only: if true, use absolute distance only
    
    output:
        enum[lvl]: danger level in string
        img: put text on img
    
    note that the t/last is inference time when debugging with PC, when using 
    phone app, they should be interval between shooting two input frames
    
    """
    enum=('Low','Medium','High')
    # danger lvl from absolute dist
    if dist<4000:
        lvl=2
    elif dist<8000:
        lvl=1
    else:
        lvl=0
    
    # acceleration
    if not abs_dist_only:
        predict_dist=(dist-last_dist)/t*timeahead+dist
        if predict_dist<0:
            lvl=max(2,lvl)
        elif predict_dist>8000:
            lvl=max(0,lvl)
        else:
            lvl=max(1,lvl)
    
    cv2.putText(img, 'Danger:{}'.format(enum[lvl]), 
                (4,472), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255,255,255), 1, 
                lineType=cv2.LINE_AA)
    return enum[lvl],img

class DistEstimator():
    """
    estimate distance, all units are of milimeters and pixels
    
    """
    def __init__(self, width_reference=1600):
        # width_reference unit is milimeters
        self.width_reference=width_reference
        self.mapping=None
        
    def setWidthReference(self, width_reference):
        self.width_reference=width_reference
        
    def loadMappingFunc(self, filepath=None):
        """
        load file from txt and save as array, which should return a Nx2 mat,
        where the 1st column is img distance in pixels, the 2nd colume is 3D
        distance in milimeters
        
        """
        if filepath==None:
            raise ValueError('invalid filepath for pixel-dist mapping function')
        self.mapping=np.loadtxt(filepath,delimiter=';')
        return self.mapping
    
    def estimateDistance(self, width):
        if width==None or width=='null':
            raise ValueError('invalid width value to calculate distance')
        # calculate distance from mapping func
        mapping=self.mapping
        total=len(mapping[:,0])
        i=int(total/2)
        step=int(total/4)
        last_i=0
        while(i!=last_i):
            #print(i)
            last_i=i
            if width>mapping[i,0]:
                i-=step
            else:
                i+=step
            step=int(step/2)
        #print('width={},mapped_pel={}'.format(width,mapping[i,0]))
        if i==total-1:
            #width too small
            low=int(i-1)
            high=i
        elif i==0:
            #width too large
            low=i
            high=int(i+1)
        else:
            #regular case
            if width>mapping[i,0]:
                low=int(i-1)
                high=i                    
            else:
                low=i
                high=int(i+1)
        # solve linear regression between (x1,y1) and (x2,y2)
        x1=mapping[low,0]
        y1=mapping[low,1]
        x2=mapping[high,0]
        y2=mapping[high,1]
        return (y2-y1)/(x2-x1)*(width-x2)+y2
   
def detectMultipleImages(detection_graph, category_index, testimgpath, 
                         foldernumber, outputthresh=0.5, saveimg_flag=True,
                         max_class=8, dist_estimator=None, use_tracking=False,
                         folder_only='', show_leading=False, customNMS=True,
                         save_raw=False, calibration_code=''):
    '''
    load the frozen graph (model) and run detection among all the images
    
    args:
        testimgpath: the top level of your image folders, note that subfolders 
            are considered as default
        foldernumber: how many subfolders do you want to test
        detection_graph: the loaded frozen graph
        category_index: index to convert category labels into numbers
        outputthresh
        saveimg_flag: if ture, save detection results under subfolder 'leadingdetect'
        max_class: how many class to be detected
        
    output:
        output_dict: raw detection result of tensor graph
        annotationdict: save all the detections into a json file in VIVA annotator format
        sumtime_detect/filecount: average detection time
        sumtime_track/filecount: average tracking time, zero if use_tracking=False
    '''
    # constants
    foldercount=0
    filecount=-5
    sumtime=0
    timelist=[]
    last_dist=20000
    last_time=2
    chunksize=1000
    
    if use_tracking:
        objtracker=track_obj.ObjTracker()
        objtracker.buildTracker()
        maxtrack=10 # switch to detection when reach max track frame
        print('detection-tracking scheme is used')
    print('chunk size is {} images'.format(chunksize))
    
    # initialize the graph once for all
    with detection_graph.as_default():
        with tf.Session() as sess:            
                    
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            
            tensor_dict = {}
            for key in [#'num_detections', 'detection_boxes', 
                        #'detection_scores', 'detection_classes',
                        'Postprocessor/raw_box_encodings',
                        'Postprocessor/raw_box_locations',
                        'Postprocessor/raw_box_scores']:
                tensor_name = key + ':0'
                
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
           
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            
            folderdict=os.listdir(testimgpath)
            for folder in folderdict:
                # skip the files, choose folders only
                if '.' in folder:
                    continue
                
                # run model for val set only
                if folder_only!='' and folder_only not in folder:
                    continue
                
                # for debug, set the number of folders to be processed
                if foldercount>=foldernumber:
                    break
                else:
                    foldercount+=1
                
                # show folder name and create save path
                imagepath=os.path.join(testimgpath,folder)
                print('processing folder:',imagepath)
                
                savepath=os.path.join(testimgpath,folder,'leadingdetect')
                if saveimg_flag:
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                
                filedict=os.listdir(imagepath)
                annotationdict={} # save all detection result into json file
                distlist={} # save all the distances estimated from prediction
                trackcount=0 # record how many frames used for tracking
                solidtrack=False
                
                for imagename in filedict:
                    if 'jpg' in imagename or 'png' in imagename:
                        image_cv = cv2.imread(os.path.join(imagepath,imagename))
                        if image_cv is None:
                            continue
                        image = Image.open(os.path.join(imagepath,imagename))
                        (im_width, im_height) = image.size
                        
                        filecount+=1
                        # the array based representation of the image will be used 
                        # later in order to prepare the
                        # result image with boxes and labels on it.
                        image_np = loadImageInNpArray(image)
                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        # image_np_expanded = np.expand_dims(image_np, axis=0)
                        
                        ##################### Actual detection ######################
                        if not use_tracking:
                            # Run detection inference
                            starttime=time.time()
                            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image_np, 0)})
                            detect_time=time.time()-starttime
                            if filecount>0: # the first 5 images won't be counted for detection time
                                sumtime+=detect_time
                                print('processing time: {} s'.format(detect_time))
                                if filecount==chunksize:
                                    timelist.append(sumtime/chunksize)
                                    print('average time of current chunk: {}'.format(sumtime/filecount))
                                    filecount=0
                                    sumtime=0
                                #print('average detection time is {} s'.format(sumtime/filecount))
                            
                            if not customNMS:
                                # this is using first 100 detection results
                                annotationdict = updateAnnotationDict(output_dict,
                                                annotationdict,imagename,
                                                im_width,im_height,max_class)
                            else:
                                # use raw detection results with highest score
                                # NMS list will clipped by score threshold
                                annotationdict, rawboxes, rawscores = updateAnnotationDict_Raw(output_dict,annotationdict,
                                                              imagename,im_width,im_height,
                                                              max_class,category_index,
                                                              outputthresh=outputthresh,IOUthresh=0.5)
                                if save_raw:
                                    np.savez(os.path.join(savepath,imagename.split('.')[0]), 
                                             rawboxes, rawscores)
                                    #print('npz saved')
                            
                            annotationdict, _ , _ = keepOnlyOneLeading(annotationdict,imagename)
                            if saveimg_flag:
                                distlist[imagename] = drawBBoxNSave(image_np,imagename,
                                        savepath,annotationdict,
                                        drawside=True,dist_estimator=dist_estimator,
                                        show_leading=show_leading,
                                        show_dist=True)
                                
                        else:
                            # Run detection-tracking inference
                            if solidtrack and trackcount<maxtrack:
                                # refresh tracker and do tracking
                                # return solidtrack mark
                                solidtrack, bbox, detect_time = objtracker.updateTrack(image_cv)
                                #print('track frame {}, time {}'.format(filecount+5,tracktime))
                                annotationdict = updateAnnotationDict_Track(annotationdict,imagename,bbox)
                                trackcount+=1
                                sumtime+=detect_time
                            if solidtrack==False or trackcount==maxtrack:
                                # detection
                                # get bbox of leading car
                                # if has leading car:
                                    #reture solidtrack mark
                                    #trackcount=0
                                
                                starttime=time.time()
                                output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image_np, 0)})
                                detect_time=time.time()-starttime
                                if filecount>0: # the first 5 images won't be counted for detection time
                                    sumtime+=detect_time
                                    print('processing time: {} s'.format(sumtime/filecount))
                                    if filecount==chunksize:
                                        timelist.append(sumtime/chunksize)
                                        print('average time of current chunk: {}'.format(sumtime/filecount))
                                        filecount=0
                                        sumtime=0
                                if not customNMS:
                                    annotationdict = updateAnnotationDict(output_dict,
                                                    annotationdict,imagename,
                                                    im_width,im_height,max_class)
                                else:
                                    annotationdict, _ ,_ = updateAnnotationDict_Raw(output_dict,annotationdict,
                                                              imagename,im_width,im_height,
                                                              max_class,category_index,
                                                              outputthresh=outputthresh,IOUthresh=0.5)
                                # let solidtrack=True if has leading vehicle
                                annotationdict, solidtrack, bbox = keepOnlyOneLeading(annotationdict,imagename)
                                #print('detect frame {}, time {}'.format(filecount+5,detect_time))
                                # update tracker
                                if solidtrack:
                                    objtracker.refreshTracker()
                                    objtracker.updateTrack(image_cv,init=True,bbox=bbox)
                                    trackcount=0
                            # draw bbox and text and save img
                            last_dist=drawBBoxNSave_Track(image_np,imagename,savepath,bbox,
                                                last_dist,last_time,detect_time,
                                                dist_estimator = dist_estimator,
                                                saveimg_flag = saveimg_flag)
                            distlist[imagename]=last_dist
                            last_time=detect_time
                            
                timelist.append(sumtime/filecount)
                # after done save all the annotation into json file, save the file
                if not use_tracking:
                    # save annotation if not using tracking
                    with open(os.path.join(testimgpath,'annotation_{}_detection.json'.format(folder)),'w') as savefile:
                        savefile.write(json.dumps(annotationdict, sort_keys = True, indent = 4))
                else:
                    # save annotation if using tracking
                    with open(os.path.join(testimgpath,'annotation_{}_tracking.json'.format(folder)),'w') as savefile:
                        savefile.write(json.dumps(annotationdict, sort_keys = True, indent = 4))
                
                # save distance estimated from predictions or tracking
                if not use_tracking:
                    with open(os.path.join(testimgpath,'distance_{}_detection{}.json'.format(folder,calibration_code)),'w') as savefile:
                        savefile.write(json.dumps(distlist, sort_keys = True, indent = 4))
                else:
                    with open(os.path.join(testimgpath,'distance_{}_tracking{}.json'.format(folder,calibration_code)),'w') as savefile:
                        savefile.write(json.dumps(distlist, sort_keys = True, indent = 4))
    return output_dict, annotationdict, timelist, distlist


            
if __name__=='__main__':
    # pass the parameters
    parser=argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/viewnyx/ckpt_ssd_opt_vnx_finetune/export/frozen_inference_graph.pb', 
                        help="select the file path for ckpt folder")
    # D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/viewnyx/ckpt_ssd_opt_300/export/frozen_inference_graph.pb
    # D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/viewnyx/ckpt_ssd_opt_vnx_finetune/export/frozen_inference_graph.pb
    parser.add_argument('--label_path', type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/viewnyx/data/class_labels.pbtxt', 
                        help="select the file path for class labels")
    parser.add_argument('--testimg_path',type=str,
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/Part4_ACC',
                        help='path to the images to be tested')
    # D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/debug/bdd100k/images/100k
    # D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/Part4_ACC
    # D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/Part3_videoframes
    parser.add_argument('--folder_only', type=str, default='3652',
                        help="run model on a specific set only if not null, \
                        this will require that you have a folder with 'A String'\
                        in its name. set this as ''\
                         when debuging with some random folder names")
    parser.add_argument('--class_number', type=int, default=1,
                        help="set number of classes (default as 1)")
    parser.add_argument('--folder_number',type=int, default=10,
                        help='set how many folders will be processed')
    parser.add_argument('--saveimg_flag', type=bool, default=True,
                        help="flag for saving detection result or not, default as True")
    parser.add_argument('--output_thresh', type=float, default=0.2,
                        help='threshold of score for output the detected bbxs (default=0.05)')
    parser.add_argument('--cam_calibration_path',type=str,
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2019 winter/CameraCalibration/viewnyx_160.txt',
                        help='filepath of pixel-distance mapping, use None is not needed')
    # cam_mapping_viewnyx.txt # this is for demos
    # viewnyx_140.txt 
    # viewnyx_150.txt
    # viewnyx_160.txt # this is for calculating detection/tracking mIOU
    # viewnyx_170.txt
    # viewnyx_180.txt
    # viewnyx_190.txt
    # viewnyx_200.txt
    # viewnyx_210.txt
    # viewnyx_220.txt
    parser.add_argument('--use_tracking',type=bool, default=False,
                        help='use tracking to boost processing speed or not, default is false')
    parser.add_argument('--show_leading',type=bool,default=True,
                        help='show leading vehicle in red bbox if true, in green if false.')
    parser.add_argument('--save_raw_output',type=bool,default=False,
                        help='if true, save raw output in .npz format.')
    args = parser.parse_args()
    
    ckptpath = args.ckpt_path
    labelpath = args.label_path
    camcalpath = args.cam_calibration_path
    classnumber = args.class_number
    testimgpath = args.testimg_path
    folderonly=args.folder_only
    foldernumber=args.folder_number
    foldercount=0
    saveflag=args.saveimg_flag
    outputthresh=args.output_thresh
    usetracking=args.use_tracking
    showleading=args.show_leading
    saveraw=args.save_raw_output
    
        
    IMAGE_SIZE = (12, 8)# Size, in inches, of the output images.
    
    if camcalpath!=None:
        dist_estimator=DistEstimator()
        dist_estimator.setWidthReference(1600)
        mapping=dist_estimator.loadMappingFunc(filepath=camcalpath)
        #result=dist_estimator.estimateDistance(120)
        calibrationcode='_'+camcalpath.split('_')[1].split('.')[0]
    
    # reset for debugging
    tf.reset_default_graph()
    
    # Load a (frozen) Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckptpath, 'rb') as fid:
            serialized_graph = fid.read()
            # the od_graph_def is the config file for the network
            # when display this variable, please wait, cause it read the file in binary
            # but display with utf8
            
            # there is a NonMaximumSupressionV3 in the graph, but only V1 & V2 given
            # in the api if tf version<1.8.0
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
 
    # Loading label map
    label_map = label_map_util.load_labelmap(labelpath)
    categories = label_map_util.convert_label_map_to_categories(label_map, 
            max_num_classes=classnumber, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    
    # detection by function
    starttime=time.time()
    output_dict, jsondict, average_detection_time, distance_list =detectMultipleImages(
                             detection_graph, 
                             category_index=category_index, 
                             testimgpath=testimgpath, 
                             foldernumber=foldernumber,  
                             outputthresh=outputthresh, 
                             saveimg_flag=saveflag,
                             max_class=classnumber,
                             dist_estimator=dist_estimator,
                             use_tracking=usetracking,
                             folder_only=folderonly,
                             show_leading=showleading,
                             customNMS=True,
                             save_raw=saveraw,
                             calibration_code=calibrationcode)
    endtime=time.time()
    if usetracking:
        print('leading vehicle detection with tracking')
    else:
        print('frame by frame leading vehicle detection')
    print('\n processing finished, total time:{} s'.format(endtime-starttime))
    print('for each chunk, average processing time per frame is:')
    print('average detection time per chunk:')
    for i in average_detection_time:
        print(i)
    

''' End of File '''