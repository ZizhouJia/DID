import numpy as np
import os
# import utils
from . import utils
import cv2
import rawpy

class data_reader:
    def __init__(self,root,image_shape=(512,512),label_set=["__background__","bicycle","car","person"]):
        self.image_shape=image_shape
        # self.image_type=image_type
        self.root=root
        self.image_names=[]
        self.label_set=label_set
        path=os.path.join(self.root,"annotations")
        for name in os.listdir(path):
            if(".xml" in name):
                self.image_names.append(name[:-4])

    def get_length(self):
        return len(self.image_names)

    def get_label(self,index):
        if(index>=self.get_length()):
            return None
        path=os.path.join(self.root,"annotations")
        path=os.path.join(path,self.image_names[index]+".xml")
        clss,bounding_boxs=utils.read_xml_and_get_info(path)
        for i in range(0,len(clss)):
            clss[i]=self.label_set.index(clss[i])
        for i in range(0,len(bounding_boxs)):
            if(self.image_shape is not None):
                box=bounding_boxs[i]
                box[0]=int(float(box[0])*self.image_shape[1]/5496)
                box[2]=int(float(box[2])*self.image_shape[1]/5496)
                box[1]=int(float(box[1])*self.image_shape[0]/3670)
                box[3]=int(float(box[3])*self.image_shape[0]/3670)
                bounding_boxs[i]=box
        return clss,bounding_boxs,self.image_names[index]


    def get_rgb(self,index):
        if(index>=self.get_length()):
            return None
        path=os.path.join(self.root,"rgb")
        path=os.path.join(path,self.image_names[index]+".JPG")
        image=cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if(self.image_shape is not None):
            image=cv2.resize(image,(self.image_shape[1],self.image_shape[0]))
        return image

    def get_raw(self,index):
        if(index>=self.get_length()):
            return None
        path=os.path.join(self.root,"raw")
        path=os.path.join(path,self.image_names[index]+".CR2")
        raw_data=utils.read_raw(path)
        if(self.image_shape is not None):
            raw_data=utils.resize_raw(raw_data,self.image_shape)
        return np.expand_dims(raw_data, axis=2)

    def get_raw_postpress(self,index):
        if(index>=self.get_length()):
            return None
        path=os.path.join(self.root,"raw")
        path=os.path.join(path,self.image_names[index]+".CR2")
        raw_data=rawpy.imread(path)
        image=raw_data.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False, output_bps=16)
        if(self.image_shape is not None):
            image=cv2.resize(image,(self.image_shape[1],self.image_shape[0]))
        return image

    def get_reader_idx(self,img_name):
        idx=self.image_names.index(img_name)
        return idx

    def get_datas(self,filename):
        idx=self.get_reader_idx(filename)
        raw=self.get_raw(idx)
        rgb=self.get_rgb(idx)
        label=self.get_label(idx)
        postprocessraw=self.get_raw_postpress(idx)
        return raw,rgb,label,postprocessraw




if __name__ == '__main__':
    reader=data_reader(root="/home/hda/nfs_disk/data/detection/")
    print(reader.get_length())
    print(reader.get_label(100))
    print(reader.get_rgb(100).shape)
    print(reader.get_raw(100).shape)
    print(reader.get_raw_postpress(100).shape)
    # reader=data_reader(root="/home/huangxiaoyu/data/THUNIGHT/detection/")
    # print(reader.get_length())
    # print(reader.get_label(100))
    # print(reader.get_data(100).shape)
