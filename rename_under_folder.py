# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 20:54:05 2018

For all the images we have are using similar names, like '0000000000.jpg', we 
need to rename all of those images and make them distinctive. here we just put 
the folder name in front of the image names.

@author: Wen Wen
"""

import argparse
import os
import cv2

from lxml import etree
import xml.etree.ElementTree as ET

if __name__ == '__main__':
    # pass the parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
                        default='./testframes/', 
                        help="File path of input data (default './testframes/')")
    
    args = parser.parse_args()
    
    # set the work directory 
    filepath=args.file_path
    
    # go to one folder
    folderlist=os.listdir(filepath)
    for i in folderlist:
        if '.' in i:
            continue # it's not a folder but a file
        imagepath=filepath+i+'/'
        imagelist=os.listdir(imagepath)
        totalfile=len(imagelist)
        cnt=0
        for j in imagelist:
            if 'flip' in j:
                continue
            elif 'jpg' in j and i not in j:
                if 'o' not in j and 'flip' not in j: # not bottom nor horizon image
                    # rename the image
                    os.rename(imagepath+j,imagepath+i+'_'+j)
                    # create a flipped copy
                    img=cv2.imread(imagepath+i+'_'+j)
                    img=cv2.flip(img, 1)
                    cv2.imwrite(imagepath+i+'_'+j.replace('.jpg','_flipped.jpg'),img)                    
                
            elif 'xml' in j and i not in j:
                # rename the xml
                os.rename(imagepath+j,imagepath+i+'_'+j)
                
            elif 'xml' in j and i in j:
                # create a flipped copy
                tree = ET.parse(imagepath+j)
                root = tree.getroot()
                if i not in root.find('filename').text:
                    root.find('filename').text=i+'_'+root.find('filename').text
                    # save xml file with flipped annotation
                    xmlstr=ET.tostring(root) # return a binary string if encoding is default
                    tree=etree.fromstring(xmlstr)
                    xmlstr=etree.tostring(tree,pretty_print=True) # reform the xml string with pretty printing
                    with open(imagepath+j,'wb') as savefile:
                        savefile.write(xmlstr)
                    
                if 'flip' not in root.find('filename').text:
                    root.find('filename').text=root.find('filename').text.replace('.jpg','_flipped.jpg')
                    width=int(float(root.find('size').find('width').text))
                    height=int(float(root.find('size').find('height').text))
                    for obj in root.iter('object'):
                        xmin_new=width-int(float(obj.find('bndbox').find('xmax').text))
                        xmax_new=width-int(float(obj.find('bndbox').find('xmin').text))
                        obj.find('bndbox').find('xmin').text=str(xmin_new)
                        obj.find('bndbox').find('xmax').text=str(xmax_new)
                    
                    # save xml file with flipped annotation
                    xmlstr=ET.tostring(root) # return a binary string if encoding is default
                    tree=etree.fromstring(xmlstr)
                    xmlstr=etree.tostring(tree,pretty_print=True) # reform the xml string with pretty printing
                    with open(imagepath+j.replace('.xml','_flipped.xml'),'wb') as savefile:
                        savefile.write(xmlstr)
                        
            if ((2*cnt)%totalfile) <= 1:
                print(100*cnt/totalfile, '%')
            cnt+=1  
                
                
""" End of File """