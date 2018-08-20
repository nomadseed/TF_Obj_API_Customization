# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:40:44 2018

test the performance of the trained model with checkpoint and saved model
# this api is for tensorflow version 1.4.0 or later
args:
    ckpt_path: select the file path for ckpt folder
    label_path: select the file path for class labels
    testimg_path: path to the images to be tested

@author: Wen Wen
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
import json

from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops

# Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



def GetClass(class_id):
    """
    given class id, get the class name
    
    """
    if class_id==1:
        return 'car'
    elif class_id<=0:
        raise ValueError('class id cannot be 0 or float numbers')
    else:
        return None

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

def detectMultipleImages(detection_graph, category_index, testimgpath, 
                              foldernumber, outputthresh, 
                              saveimgpath=''):
    '''
    load the frozen graph (model) and run detection among all the images
    
    args:
        testimgpath: the top level of your image folders, note that subfolders 
            are considered as default
        foldernumber: how many subfolders do you want to test
        detection_graph: the loaded frozen graph
        saveimgpath: path to save detection images, not save any image if not
            provided
        category_index: index to convert category labels into numbers
        outputthresh
    
    output:
        jsondict: save all the detections into a json file in VIVA annotator format
    '''
    # constants and paths
    jsondict={}
    foldercount=0
    
    if saveimgpath!='':
        if not os.path.exists(saveimgpath):
            os.makedirs(saveimgpath)
    # initialize the graph once for all
    with detection_graph.as_default():
        with tf.Session() as sess:
    
    
            folderdict=os.listdir(testimgpath)
            for folder in folderdict:
                # skip the files, choose folders only
                if '.' in folder:
                    continue 
                
                # for debug, set the number of folders to be processed
                if foldercount>=foldernumber:
                    break
                else:
                    foldercount+=1
                    
                imagepath=os.path.join(testimgpath,folder)
                filedict=os.listdir(imagepath)
                    
                for imgname in filedict:
                    if 'jpg' in imgname or 'png' in imgname:
                        image = Image.open(os.path.join(imagepath,imgname))
                        # the array based representation of the image will be used 
                        # later in order to prepare the
                        # result image with boxes and labels on it.
                        image_np = loadImageInNpArray(image)
                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        # image_np_expanded = np.expand_dims(image_np, axis=0)
                        
                        ##################### Actual detection ######################
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
                            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
                            detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                            # Follow the convention by adding back the batch dimension
                            tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                        # Run inference
                        output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image_np, 0)})
                        # all outputs are float32 numpy arrays, so convert types as appropriate
                        output_dict['num_detections'] = int(output_dict['num_detections'][0])
                        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                        output_dict['detection_scores'] = output_dict['detection_scores'][0]
                        if 'detection_masks' in output_dict:
                            output_dict['detection_masks'] = output_dict['detection_masks'][0]
                        
                        # save detection result (output_dict) into jsondict in 
                        # the format of VIVA Annotator
                        
                        
                        # Visualization of the results of a detection.
                        if saveimgpath!='':
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                    image_np,
                                    output_dict['detection_boxes'],
                                    output_dict['detection_classes'],
                                    output_dict['detection_scores'],
                                    category_index,
                                    instance_masks=output_dict.get('detection_masks'),
                                    use_normalized_coordinates=True,
                                    min_score_thresh=outputthresh,
                                    line_thickness=5)
                            cv2.imwrite(os.path.join(saveimgpath,imgname.replace('.','_detection.')),cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return output_dict, jsondict

if __name__=='__main__':
    # pass the parameters
    parser=argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, 
                        default='viewnyx/ckpt_ssd_v2/export/frozen_inference_graph.pb', 
                        help="select the file path for ckpt folder")
    parser.add_argument('--label_path', type=str, 
                        default='viewnyx/data/class_labels.pbtxt', 
                        help="select the file path for class labels")
    parser.add_argument('--testimg_path',type=str,
                        default='testframes',
                        help='path to the images to be tested')
    parser.add_argument('--class_number', type=int, default=1,
                        help="set number of classes (default as 1)")
    parser.add_argument('--folder_number',type=int, default=1,
                        help='set how many folders will be processed')
    parser.add_argument('--saveimg_path', type=str, 
                        default='viewnyx/savedimg',
                        help='set how many images will be tested and saved as jpg')
    parser.add_argument('--output_thresh', type=float, default=0.5,
                        help='threshold of score for output the detected bbxs (default=0.5)')
    args = parser.parse_args()
    
    ckptpath = args.ckpt_path
    labelpath = args.label_path
    classnumber = args.class_number
    testimgpath = args.testimg_path
    foldernumber=args.folder_number
    foldercount=0
    saveimgpath=args.saveimg_path
    outputthresh=args.output_thresh
        
    IMAGE_SIZE = (12, 8)# Size, in inches, of the output images.
    
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
    output_dict, jsondict=detectMultipleImages(detection_graph, 
                              category_index, 
                              testimgpath, 
                              foldernumber,  
                              outputthresh, 
                              saveimgpath)

        
''' End of File '''