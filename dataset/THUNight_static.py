import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import torch.utils.data as Data
import rawpy
try:
    import cPickle as pickle
except ImportError:
    import pickle
import cv2
import glob

transform_img = transforms.Compose([
                    transforms.ToTensor()
                    ])

class OurDataset(Data.Dataset):
    def __init__(self,dir = '/mnt/nfs_disk/data/dark2light/',ps=512,type='train',imgtype='raw',return_ratio=False):
        if(type=='train'):
            self.type='0'
        elif(type=='test'):
            self.type='1'
        else:
            self.type='2'

        if(imgtype=='raw'):
            self.fntype='CR2'
        else:
            self.fntype='JPG'


        self.return_ratio=return_ratio

        self.input_dir=os.path.join(dir,imgtype,'images')
        self.gt_dir=os.path.join(dir,imgtype,'groundtruth')

        self.ps=ps
        self.init_ids()

        self.gt_images=[None]*2000
        self.input_images=[[None]*10 for i in range(2000)]


    def init_ids(self):
        train_fns=glob.glob(self.gt_dir+'/'+self.type+'*.CR2')
        train_ids=[]
        for i in range(len(train_fns)):
            _,train_fn=os.path.split(train_fns[i])
            train_ids.append(int(train_fn[0:5]))

        self.train_ids=train_ids


    def pack_raw(self,raw):
        #pack Bayer image to 4 channels
        # im = raw.raw_image_visible.astype(np.float32)
        im = raw.raw_image_visible.astype(np.uint16)
        # print('im',im)
        # print('max',np.max(im))
        # print('min',np.min(im))
        im = np.maximum(im - 2048,0)#/ (16383 - 2048) #subtract the black level
        # print('after',im)

        im = np.expand_dims(im,axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.concatenate((im[0:H:2,0:W:2,:],
                           im[0:H:2,1:W:2,:],
                           im[1:H:2,1:W:2,:],
                           im[1:H:2,0:W:2,:]), axis=2)
        return out


    def __getitem__(self,ind):
        # get the path from image id
        train_id = self.train_ids[ind]
        in_files = glob.glob(self.input_dir + '/%05d_0*.CR2'%train_id)
        ridx=np.random.random_integers(0,len(in_files)-1)
        in_path = in_files[ridx]
        _, in_fn = os.path.split(in_path)

        gt_files = glob.glob(self.gt_dir + '/%05d_00*.CR2'%train_id)
        gt_path = gt_files[0]
        _, gt_fn = os.path.split(gt_path)
        in_exposure =  float(in_fn[9:-5])
        gt_exposure =  float(gt_fn[9:-5])
        # print('in_exposure',in_exposure)
        # print('gt_exposure',gt_exposure)
        ratio = min(gt_exposure/(in_exposure+0.0001),300)

        if self.input_images[ind][ridx] is None:
            raw = rawpy.imread(in_path)
            self.input_images[ind][ridx] = np.expand_dims(self.pack_raw(raw),axis=0)# *ratio

        if self.gt_images[ind] is None:
            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # self.gt_images[ind] = np.expand_dims(np.float32(im/65535.0),axis = 0)
            self.gt_images[ind] = np.expand_dims(im,axis = 0)

        #crop
        H = self.input_images[ind][ridx].shape[1]
        W = self.input_images[ind][ridx].shape[2]


        xx = np.random.randint(0,W-self.ps)
        yy = np.random.randint(0,H-self.ps)

        in_img=self.input_images[ind][ridx][0,yy:yy+self.ps,xx:xx+self.ps,:]

        input_patch = np.float32(in_img)/ (16383 - 2048)
        if(self.return_ratio==False):
            input_patch=input_patch*ratio
        gt_img=self.gt_images[ind][0,yy*2:yy*2+self.ps*2,xx*2:xx*2+self.ps*2,:]
        gt_patch = np.float32(gt_img/65535.0)


        if np.random.randint(2,size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2,size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)


        input_patch = np.minimum(input_patch,1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        in_img = torch.from_numpy(input_patch).permute(2,0,1)
        gt_img = torch.from_numpy(gt_patch).permute(2,0,1)

        if(self.return_ratio):
            return in_img,gt_img,ratio
        else:
            return in_img,gt_img

    def __len__(self):
        return len(self.train_ids)
        #return 2
