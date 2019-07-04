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
class dark2light_dataset(Data.Dataset):
    def __init__(self,dir='/mnt/nfs_disk/data/dark2light/',img_size=512,img_type='raw',dataset='ours'):
        self.root=dir
        self.img_size=img_size
        self.mode="train"
        self.dataset=dataset
        if(self.dataset=='ours'):
            self.input_dir=os.path.join(dir,img_type,'images')
            self.gt_dir=os.path.join(dir,img_type,'groundtruth')
            self.fntype='.CR2'
        else:
            self.input_dir=os.path.join(dir,'short')
            self.gt_dir=os.path.join(dir,'long')
            self.fntype='.ARW'

        self.init_ids()

        self.gt_images=[None]*3000
        self.input_images=[[None]*10 for i in range(3000)]

    def init_ids(self):
        train_fns=glob.glob(self.gt_dir+'/0*'+self.fntype)
        train_ids=[]
        for i in range(len(train_fns)):
            _,train_fn=os.path.split(train_fns[i])
            train_ids.append(int(train_fn[0:5]))
        self.train_ids=train_ids

        test_fns=glob.glob(self.gt_dir+'/1*'+self.fntype)
        test_ids=[]
        for i in range(len(test_fns)):
            _,test_fn=os.path.split(test_fns[i])
            test_ids.append(int(test_fn[0:5]))
        self.test_ids=test_ids

    def pack_raw(self,raw,black_level):
        #pack Bayer image to 4 channels
        # im = raw.raw_image_visible.astype(np.uint16)
        im = raw
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

    def set_mode(self,mode):
        self.mode=mode

    def set_return_array(self,return_array=["id","ratio","in_raw","gt_raw","in_rgb","gt_rgb","in_raw_ratio_rgb"]):
        self.return_array=return_array

    def raw_mul_ratio_get_rgb(self,raw,ratio):
        # raw4(batch,4,h,w)->rgb3(batch,h*2,w*2,3)
        gt_raw = rawpy.imread(self.gt_dir+'/00001_00_0.5s.CR2')
        # gt_raw = rawpy.imread(self.gt_dir+'/00001_00_10s.ARW')

        raw=raw.double()*ratio.view(-1,1,1,1)*(16383.0-2048.0)+2048.0
        H=raw.size(2)*2
        W=raw.size(3)*2

        new=torch.zeros((raw.size(0),H,W),dtype=torch.int16)
        new[:,0:H:2,0:W:2]=raw[:,0,:,:]
        new[:,0:H:2,1:W:2]=raw[:,1,:,:]
        new[:,1:H:2,1:W:2]=raw[:,2,:,:]
        new[:,1:H:2,0:W:2]=raw[:,3,:,:]


        # postprocess
        new_np=np.zeros((raw.size(0),H,W,3),dtype=np.float32)
        for i in range(raw.size(0)):
            gt_raw.raw_image_visible[:]=new[i,:,:].numpy()
            tmp=gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            new_np[i,:,:,:]=np.float32(tmp)/65535.0

        return new_np

    def __getitem__(self,ind):
        # id,ratio,in_raw,gt_raw,in_rgb,gt_rgb,in_raw_ratio_rgb
        return_list=[]
        # get the path from image id
        if(self.mode=='train'):
            id = self.train_ids[ind]
        else:
            id = self.test_ids[ind]
        in_files = glob.glob((self.input_dir + '/%05d_0*'+self.fntype)%id)
        ridx=np.random.random_integers(0,len(in_files)-1)
        in_path = in_files[ridx]
        _, in_fn = os.path.split(in_path)

        gt_files = glob.glob((self.gt_dir + '/%05d_00*'+self.fntype)%id)
        gt_path = gt_files[0]
        _, gt_fn = os.path.split(gt_path)
        in_exposure =  float(in_fn[9:-5])
        gt_exposure =  float(gt_fn[9:-5])
        ratio = min(gt_exposure/(in_exposure+0.0001),300)
        newsize=self.img_size
        max_pixel=16383.0
        gt_max_pixel=65535.0
        black_level=2048.0

        gt_raw = rawpy.imread(gt_path)
        in_raw = rawpy.imread(in_path)
        
        if self.gt_images[ind] is None:
            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.raw_image_visible.astype(np.uint16)
            # im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            self.gt_images[ind] = im

        if self.input_images[ind][ridx] is None:
            in_raw = rawpy.imread(in_path)
            im = in_raw.raw_image_visible.astype(np.uint16)
            self.input_images[ind][ridx]=im




        in_raw_one=self.input_images[ind][ridx]#[yy:yy+self.img_size,xx:xx+self.img_size]
        gt_raw_one=self.gt_images[ind]#[yy:yy+self.img_size,xx:xx+self.img_size]
        in_raw_four=self.pack_raw(in_raw_one,black_level)
        gt_raw_four=self.pack_raw(gt_raw_one,black_level)


        in_raw.raw_image_visible[:]=in_raw_one
        gt_raw.raw_image_visible[:]=gt_raw_one

        in_rgb=in_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_rgb=gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)


        in_raw_ratio=(in_raw_one-black_level)*ratio+black_level
        in_raw.raw_image_visible[:]=in_raw_ratio
        in_raw_ratio_rgb=in_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)


        in_raw_four=np.float32(in_raw_four)/ (max_pixel-black_level)
        gt_raw_four = np.float32(gt_raw_four)/(max_pixel-black_level)
        in_rgb=np.float32(in_rgb)/ gt_max_pixel
        gt_rgb = np.float32(gt_rgb)/gt_max_pixel
        in_raw_ratio_rgb=np.float32(in_raw_ratio_rgb)/gt_max_pixel

        if np.random.randint(2,size=1)[0] == 1:  # random flip
            in_raw_four = np.flip(in_raw_four, axis=1)
            gt_raw_four = np.flip(gt_raw_four, axis=1)
            in_rgb = np.flip(in_rgb, axis=1)
            gt_rgb = np.flip(gt_rgb, axis=1)
            in_raw_ratio_rgb = np.flip(in_raw_ratio_rgb, axis=1)
        if np.random.randint(2,size=1)[0] == 1:
            in_raw_four = np.flip(in_raw_four, axis=0)
            gt_raw_four = np.flip(gt_raw_four, axis=0)
            in_rgb = np.flip(in_rgb, axis=0)
            gt_rgb = np.flip(gt_rgb, axis=0)
            in_raw_ratio_rgb = np.flip(in_raw_ratio_rgb, axis=0)

        if(self.mode=='train'):
            H=self.input_images[ind][ridx].shape[0]//2
            W=self.input_images[ind][ridx].shape[1]//2

            xx = np.random.randint(0, W - self.img_size)
            yy = np.random.randint(0, H - self.img_size)

            in_raw_four_tmp = in_raw_four[yy:yy+self.img_size,xx:xx+self.img_size,:]
            gt_raw_four_tmp = gt_raw_four[yy:yy+self.img_size,xx:xx+self.img_size,:]
            in_rgb_tmp = in_rgb[yy*2:yy*2+self.img_size*2,xx*2:xx*2+self.img_size*2,:]
            gt_rgb_tmp = gt_rgb[yy*2:yy*2+self.img_size*2,xx*2:xx*2+self.img_size*2,:]
            in_raw_ratio_rgb_tmp = in_raw_ratio_rgb[yy*2:yy*2+self.img_size*2,xx*2:xx*2+self.img_size*2,:]

        else:
            in_raw_four_tmp = in_raw_four
            gt_raw_four_tmp = gt_raw_four
            in_rgb_tmp = in_rgb
            gt_rgb_tmp = gt_rgb
            in_raw_ratio_rgb_tmp = in_raw_ratio_rgb



        in_raw_four_tmp = np.maximum(np.minimum(in_raw_four_tmp,1.0),0.0)
        gt_raw_four_tmp = np.maximum(np.minimum(gt_raw_four_tmp,1.0),0.0)
        in_rgb_tmp = np.maximum(np.minimum(in_rgb_tmp,1.0),0.0)
        gt_rgb_tmp = np.maximum(np.minimum(gt_rgb_tmp,1.0),0.0)
        in_raw_ratio_rgb_tmp = np.maximum(np.minimum(in_raw_ratio_rgb_tmp,1.0),0.0)

        in_raw_four_torch = torch.from_numpy(in_raw_four_tmp).permute(2,0,1)
        gt_raw_four_torch = torch.from_numpy(gt_raw_four_tmp).permute(2,0,1)
        in_rgb_torch = torch.from_numpy(in_rgb_tmp).permute(2,0,1)
        gt_rgb_torch = torch.from_numpy(gt_rgb_tmp).permute(2,0,1)
        in_raw_ratio_rgb_torch = torch.from_numpy(in_raw_ratio_rgb_tmp).permute(2,0,1)

        if('id' in self.return_array):
            return_list.append(id)

        if('ratio' in self.return_array):
            return_list.append(ratio)

        if('in_raw' in self.return_array):
            return_list.append(in_raw_four_torch)
        if('gt_raw' in self.return_array):
            return_list.append(gt_raw_four_torch)

        if('in_rgb' in self.return_array):
            return_list.append(in_rgb_torch)
        if('gt_rgb' in self.return_array):
            return_list.append(gt_rgb_torch)
        if('in_raw_ratio_rgb' in self.return_array):
            return_list.append(in_raw_ratio_rgb_torch)
        return tuple(return_list)

    def __len__(self):
        if(self.mode=='train'):
            return 2
            # return len(self.train_ids)
        else:
            return 2
            # return len(self.test_ids)

if __name__=='__main__':
    dataset=dark2light_dataset()
    dataset.set_mode('test')
    dataset.set_return_array(["id","ratio","in_raw","gt_raw","in_rgb","gt_rgb","in_raw_ratio_rgb"])
    dataloader=Data.DataLoader(dataset,batch_size=2,shuffle=True,num_workers=2)
    for step,data in enumerate(dataloader):
        id,ratio,in_raw,gt_raw,in_rgb,gt_rgb,in_raw_ratio_rgb=data
        new=dataset.raw_mul_ratio_get_rgb(in_raw,ratio)
        print(new.shape)
        # print(in_raw.size())
        # print(gt_raw.size())
        # print(in_rgb.size())
        # print(gt_rgb.size())
        # print(in_raw_ratio_rgb.size())
        break
