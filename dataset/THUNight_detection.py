import torch
import torch.utils.data
import numpy as np
from . import dataset_reader
from torch.utils.data import DataLoader
import random
import pickle
from tqdm import trange
import os

import torchvision.transforms as transforms
import sys
import cv2

sys.path.append('..')
from detection_utils.data import voc_eval
# import voc_eval

VOC_CLASSES = ('__background__', 'bicycle','car','person')

class detection_dataset(torch.utils.data.Dataset):
    # img_shape,images input detection network
    def __init__(self,root="/mnt/nfs_disk/data/detection/",image_type="raw",image_shape=(512,512),split='train',dataset_name='THUNIGHT',lighten=False,lightentype='hsv',lighten_net="unet",eval_path='eval/'):
        img_shape=image_shape
        # use unet fitting,before unet is 256,after unet is 512
        # use hdrnet,before is 512,after is 512.for raw,input is 1024*1024*3->512*512*4
        if(lighten_net!="unet" and image_type=='raw'):
            img_shape=(image_shape[0]*2,image_shape[1]*2)
        self.lighten_type=lightentype
        self.lighten=lighten
        self.eval_path=eval_path
        self.root=root
        self.reader=dataset_reader.data_reader(root=root,image_shape=img_shape)
        self.img_shape=img_shape
        self.image_type=image_type
        self.name = dataset_name
        self.split=split
        self._year='2019'
        self.ids=[]
        txt_path=os.path.join(self.root,self.split+'_ids.txt')
        f=open(txt_path,'r')
        for lines in f.readlines():
            line=lines.replace("\n","")
            self.ids.append(line)
        f.close()

        self.images= [None]*len(self.ids)
        self.targets=[None]*len(self.ids)


    def load_img_xml(self,index):
        filename=self.ids[index]
        raw,rgb,label,raw_postprocess=self.reader.get_datas(filename)
        if(self.image_type=='raw'):
            if(self.lighten):
                self.images[index]=raw_postprocess.astype(np.uint16)
            else:
                self.images[index]=raw.astype(np.uint16)
        else:
            self.images[index]=rgb.astype(np.uint8)
        clss,bndbox,img_name=label
        box=np.array(bndbox).astype(np.float32)
        label=np.array(clss).astype(np.float32)
        box=box/float(self.img_shape[0])
        box=torch.from_numpy(box)
        label=torch.from_numpy(label)
        self.targets[index]=torch.cat((box,label.view(label.size(0),1)),1)



    def __getitem__(self,index):
        idx=index
        if(self.targets[idx] is None):
            self.load_img_xml(idx)

        target=self.targets[idx]

        for i in range(0,len(target)):
            bbox=target[i,:]
            if((bbox[2]-bbox[0])*(bbox[3]-bbox[1])>0.0004):
                if 'target_list' in locals().keys():
                    target_list=torch.cat((target_list,bbox.view(1,bbox.size(0))),0)
                else:
                    target_list=bbox.view(1,bbox.size(0))
            if(i==len(target)-1):
                target=target_list


        if(self.image_type=='raw'):
            raw_image=self.images[idx]
            if(self.lighten):
                # raw postprocess lighten,->H*W*3
                out=self.images[idx]
                raw_image=np.float32(out)/65535.0
                raw_image=torch.from_numpy(raw_image)
                raw_image=raw_image.permute(2,0,1)
                # return raw_image,target.numpy()
            else:
                # H*W*1->H/2*W/2*4
                raw_image=np.maximum(raw_image.astype(np.float32)-2048,0)/float(16383-2048)
                raw_image=torch.from_numpy(raw_image)
                raw_image=raw_image.permute(2,0,1)

                out=torch.cat([raw_image[:,0:self.img_shape[0]:2,0:self.img_shape[1]:2],
                                raw_image[:,0:self.img_shape[0]:2,1:self.img_shape[1]:2],
                                raw_image[:,1:self.img_shape[0]:2,1:self.img_shape[1]:2],
                                raw_image[:,1:self.img_shape[0]:2,0:self.img_shape[1]:2]],0)
                raw_image=out
            return raw_image,target.numpy()
        else:
            rgb_image=self.images[idx]
            if(self.lighten):
                # opencv lighten
                if(self.lighten_type=='hsv'):
                    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
                    hsv[:,:,2] += 50
                    hsv[:,:,2] =np.maximum(hsv[:,:,2] ,255)
                    tmp = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                rgb_image=tmp

            rgb_image=rgb_image.astype(np.float32)/255.0
            rgb_image=torch.from_numpy(rgb_image)
            rgb_image=rgb_image.permute(2,0,1)
            return rgb_image,target.numpy()


    def __len__(self):
        return len(self.ids)



    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        if(self.image_type=='rgb'):
            return self.pull_rgb_image(index)

        idx=index
        if(self.targets[idx] is None):
            self.load_img_xml(idx)
        raw_image=self.images[idx]

        if(self.lighten):
            # postprocess,H*W*3
            out=self.images[idx]
            raw_image=np.float32(out)/65535.0
            raw_image=torch.from_numpy(raw_image)
            raw_image=raw_image.permute(2,0,1)
        else:
            # H*W*1->H/2*W/2*4
            raw_image=np.maximum(raw_image.astype(np.float32)-2048,0)/float(16383-2048)
            raw_image=torch.from_numpy(raw_image)
            raw_image=raw_image.permute(2,0,1)

            out=torch.cat([raw_image[:,0:self.img_shape[0]:2,0:self.img_shape[1]:2],
                            raw_image[:,0:self.img_shape[0]:2,1:self.img_shape[1]:2],
                            raw_image[:,1:self.img_shape[0]:2,1:self.img_shape[1]:2],
                            raw_image[:,1:self.img_shape[0]:2,0:self.img_shape[1]:2]],0)
            raw_image=out

        return raw_image


    def pull_rgb_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        idx=index
        if(self.targets[idx] is None):
            self.load_img_xml(idx)
        image=self.images[idx]
        if(self.lighten):
            if(self.lighten_type=='hsv'):
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                hsv[:,:,2] += 50
                hsv[:,:,2] =np.maximum(hsv[:,:,2] ,255)
                tmp = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            rgb_image=tmp

        rgb_image=rgb_image.astype(np.float32)/255.0
        image=torch.from_numpy(rgb_image)
        # image=self.rgb_norm(image)
        image=image.permute(2,0,1)
        return image

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''

        idx=index
        if(self.targets[idx] is None):
            self.load_img_xml(idx)
        img_id=self.ids[idx]
        target=self.targets[idx]

        gt=target
        for i in range(0,len(target)):
            bbox=target[i,:]
            if((bbox[2]-bbox[0])*(bbox[3]-bbox[1])>0.0004):
                if 'target_list' in locals().keys():
                    target_list=torch.cat((target_list,bbox.view(1,bbox.size(0))),0)
                else:
                    target_list=bbox.view(1,bbox.size(0))
            if(i==len(target)-1):
                gt=target_list
        return img_id, gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        image=self.pull_image(index=index)
        return image.unsqueeze_(0)

    def evaluate_detections(self, all_boxes, output_dir=None,predict_path=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self.predict_path=predict_path
        self._write_voc_results_file(all_boxes)
        aps, map = self._do_python_eval(output_dir)
        return aps, map

    def _get_voc_results_file_template(self):
        filename = 'comp4_det_test' + '_{:s}.txt'
        filedir = os.path.join(
            self.eval_path, 'results', 'THUNIGHT' + self._year, 'Main',self.predict_path)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(VOC_CLASSES):
            cls_ind = cls_ind
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.ids):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        rootpath = os.path.join(self.root)
        name = 'test_ids'
        annopath = os.path.join(
            rootpath,
            'annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            rootpath,
            name + '.txt')
        cachedir = os.path.join(self.eval_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(VOC_CLASSES):
            if cls == '__background__':
                continue

            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval.voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')
        return aps, np.mean(aps)

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    rgb_imgs=[]
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                # imgs.append(tup)
                # print('tup',tup.size())
                if(tup.size(0)==4):
                    imgs.append(tup)
                else:
                    rgb_imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                # print('target',tup)
                annos = torch.from_numpy(tup).float()
                targets.append(annos)
    if (rgb_imgs and imgs):
        return (torch.stack(imgs, 0),torch.stack(rgb_imgs, 0), targets)
    elif (rgb_imgs):
        return (torch.stack(rgb_imgs, 0), targets)
    else:
        return (torch.stack(imgs, 0), targets)
