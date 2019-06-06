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

class DifLightDataset(Data.Dataset):
    # return_ratio=True,get item return input_img(imgsize or imgsize/2),gt_img(imgsize),ratio_mat,id
    # return_ratio=False,get item return lighten input_img(imgsize or imgsize/2),gt_img(imgsize),id
    def __init__(self,dir = '/home/hda/nfs_disk/data/dark2light/',imgsize=512,type='train',imgtype='raw',usenet='unet',usepack=True,return_ratio=True):
        self.usenet=usenet
        self.img_size=imgsize
        self.usepack=usepack
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

        self.img_type=imgtype


        self.return_ratio=return_ratio

        self.input_dir=os.path.join(dir,imgtype,'images')
        self.gt_dir=os.path.join(dir,imgtype,'groundtruth')

        # self.ps=ps
        self.init_ids()

        self.gt_images=[None]*2000
        self.input_images=[[None]*10 for i in range(2000)]


    def init_ids(self):
        train_fns=glob.glob(self.gt_dir+'/'+self.type+'*.'+self.fntype)
        train_ids=[]
        for i in range(len(train_fns)):
            _,train_fn=os.path.split(train_fns[i])
            train_ids.append(int(train_fn[0:5]))
        self.train_ids=train_ids


    def pack_raw(self,raw,black_level):
        #pack Bayer image to 4 channels
        im = raw.raw_image_visible.astype(np.uint16)
        im = np.maximum(im - black_level,0) #subtract the black level
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
        in_files = glob.glob((self.input_dir + '/%05d_0*.'+self.fntype)%train_id)
        ridx=np.random.random_integers(0,len(in_files)-1)
        in_path = in_files[ridx]
        _, in_fn = os.path.split(in_path)

        gt_files = glob.glob((self.gt_dir + '/%05d_00*.'+self.fntype)%train_id)
        gt_path = gt_files[0]
        _, gt_fn = os.path.split(gt_path)
        in_exposure =  float(in_fn[9:-5])
        gt_exposure =  float(gt_fn[9:-5])
        ratio = min(gt_exposure/(in_exposure+0.0001),300)
        newsize=self.img_size
        max_pixel=65535.0
        gt_max_pixel=65535.0
        black_level=2048.0

        if self.gt_images[ind] is None:
            if(self.img_type=='raw'):
                gt_raw = rawpy.imread(gt_path)
                im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                gt_max_pixel=65535.0
            else:
                gt_rgb=cv2.imread(gt_path)
                im = cv2.cvtColor(gt_rgb, cv2.COLOR_BGR2RGB)
                gt_max_pixel=255.0
            gt_img=cv2.resize(im, (newsize,newsize) , interpolation=cv2.INTER_AREA)
            self.gt_images[ind] =gt_img

        if self.input_images[ind][ridx] is None:
            if(self.img_type=='raw'):
                raw = rawpy.imread(in_path)
                if(self.usepack):
                    # H/2*W/2*4
                    im=self.pack_raw(raw,black_level)
                    if(self.usenet=='unet'):
                        newsize=self.img_size//2
                    max_pixel= 16383.0 - black_level
                else:
                    # H*W*3
                    im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                    max_pixel=65535.0
            else:
                rgb=cv2.imread(in_path)
                im = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                max_pixel=255.0
            newim=cv2.resize(im, (newsize,newsize), interpolation=cv2.INTER_AREA)
            self.input_images[ind][ridx]=newim


        in_img=self.input_images[ind][ridx]
        input_patch = np.float32(in_img)/ max_pixel
        gt_img=self.gt_images[ind]
        gt_patch = np.float32(gt_img)/gt_max_pixel

        if(self.return_ratio==False):
            # no return ratio, multiple ratio
            input_patch=input_patch*ratio

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
            # ratio_mat=torch.ones(in_img.size(1), in_img.size(2),1)*ratio
            return in_img,gt_img,ratio,train_id
        else:
            return in_img,gt_img,train_id

    def __len__(self):
        return len(self.train_ids)
