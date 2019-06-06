import numpy as np
import rawpy
import os
import sys
import cv2
from PIL import Image
import xml.dom.minidom as minidom

def read_raw(path):
    raw_data=rawpy.imread(path)
    raw_data=raw_data.raw_image_visible
    return raw_data

#raw_data the input raw image
#size (height,width)
def resize_raw(raw_data,size):
    image=np.zeros((raw_data.shape[0]//2,raw_data.shape[1]//2,4)).astype(np.uint16)
    image[:,:,0]=raw_data[0:raw_data.shape[0]:2,0:raw_data.shape[1]:2]
    image[:,:,1]=raw_data[0:raw_data.shape[0]:2,1:raw_data.shape[1]:2]
    image[:,:,2]=raw_data[1:raw_data.shape[0]:2,0:raw_data.shape[1]:2]
    image[:,:,3]=raw_data[1:raw_data.shape[0]:2,1:raw_data.shape[1]:2]
    image=cv2.resize(image,dsize=(size[1]//2,size[0]//2),interpolation = cv2.INTER_CUBIC)
    new_raw=np.zeros((size[0],size[1])).astype(np.uint16)
    new_raw[0:raw_data.shape[0]:2,0:raw_data.shape[1]:2]=image[:,:,0]
    new_raw[0:raw_data.shape[0]:2,1:raw_data.shape[1]:2]=image[:,:,1]
    new_raw[1:raw_data.shape[0]:2,0:raw_data.shape[1]:2]=image[:,:,2]
    new_raw[1:raw_data.shape[0]:2,1:raw_data.shape[1]:2]=image[:,:,3]
    return new_raw

def read_xml_and_get_info(path):
    dom=minidom.parse(path)
    root = dom.documentElement
    elements=root.getElementsByTagName('object')
    classes=[]
    bounding_boxs=[]
    for element in elements:
        classes.append(element.getElementsByTagName("name")[0].firstChild.data)
        location=element.getElementsByTagName("bndbox")[0]
        box=[]
        box.append(int(location.getElementsByTagName("xmin")[0].firstChild.data))
        box.append(int(location.getElementsByTagName("ymin")[0].firstChild.data))
        box.append(int(location.getElementsByTagName("xmax")[0].firstChild.data))
        box.append(int(location.getElementsByTagName("ymax")[0].firstChild.data))
        bounding_boxs.append(box)
    return classes,bounding_boxs



if __name__=='__main__':
    path="/home/huangxiaoyu/data/THUNIGHT/detection/raw/00005.CR2"
    path2="/home/huangxiaoyu/data/THUNIGHT/detection/annotations/00005.xml"
    data=read_raw(path)
    new_raw=resize_raw(data,(400,600))
    print(new_raw)
    print(new_raw.dtype)
    print(new_raw.shape)
    rgb=cv2.cvtColor((new_raw/16).astype(np.uint8),cv2.COLOR_BayerBG2BGR)
    cls,box=read_xml_and_get_info(path2)
    print(cls)
    print(box)
    # cv2.imwrite("image.jpg",rgb)
