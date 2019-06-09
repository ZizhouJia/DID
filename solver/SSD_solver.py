import model_utils.solver as solver
import model_utils.utils as utils
import pysnooper
import torch
import numpy as np
from detection_utils.data import VOC_512
from detection_utils.layers.functions import Detect, PriorBox
from detection_utils.layers.modules import MultiBoxLoss
from detection_utils.utils.nms_wrapper import nms
# from detection_utils.utils.timer import Timer
import torch.backends.cudnn as cudnn
import os
import pickle


class SSD_solver(solver.common_solver):
    def __init__(self,saveimg_path,testresult_path,lighten_mode,pretrainedvgg):
        super(SSD_solver,self).__init__()
        self.images=[]
        self.counts=[]
        self.loc_loss=[]
        self.conf_loss=[]
        self.total_loss=[]
        self.best_value=100.0

        self.saveimg_path=saveimg_path

        self.basenet='detection_utils/weights/vgg16_reducedfc.pth'
        self.cfg=VOC_512
        self.lighten_mode=lighten_mode
        self.testset_num=50
        self.num_classes=4
        self.pretrainedvgg=pretrainedvgg
        self.all_boxes= [[[] for _ in range(self.testset_num)]
                 for _ in range(self.num_classes)]

        # _t = {'im_detect': Timer(), 'misc': Timer()}
        self.testresult_path=testresult_path
        if not os.path.exists(self.testresult_path):
            os.makedirs(self.testresult_path)
        self.test_save_dir = os.path.join(self.testresult_path, 'ss_predict')
        if not os.path.exists(self.test_save_dir):
            os.makedirs(self.test_save_dir)
        self.det_file = os.path.join(self.testresult_path, 'detections.pkl')

    def get_default_config():
        config=solver.common_solver.get_default_config()
        config["mode"]="SSD"
        config["learning_rate_decay_epochs"]=[200]
        config["max_per_image"]=200
        config["thresh"]=0
        config["lighten_model_path"]=["checkpoints/HDRNet_raw4_batchsize_14_lr_0.01_task/201906030915/best/model-0.pkl",
                                        "checkpoints/FCN_Fitting_raw_batchsize_2_lr_0.0001_task/201906031250/best/"]
        return config

    def empty_data(self):
        self.images=[]
        self.counts=[]
        self.loc_loss=[]
        self.conf_loss=[]
        self.total_loss=[]
        self.all_boxes= [[[] for _ in range(self.testset_num)]
                 for _ in range(self.num_classes)]

    def load_config(self):
        super(SSD_solver,self).load_config()
        self.mode=self.config["mode"]
        self.learning_rate_decay_epochs=self.config["learning_rate_decay_epochs"]
        self.max_per_image=self.config["max_per_image"]
        self.thresh=self.config["thresh"]
        self.models[0].module.init_model(self.basenet,self.pretrainedvgg)
        if(self.lighten_mode!="no"):
            if(len(self.models)==2):
                # hdrnet
                self.lighten_model_path=self.config["lighten_model_path"][0]
                self.models[1].load_state_dict(torch.load(self.lighten_model_path,map_location={'cuda:2': 'cuda:0'}))
            else:
                # unet+fitting
                self.lighten_model_path=self.config["lighten_model_path"][1]
                self.models[1].load_state_dict(torch.load(self.lighten_model_path+'model-0.pkl',map_location={'cuda:2': 'cuda:0'}))
                self.models[2].load_state_dict(torch.load(self.lighten_model_path+'model-1.pkl',map_location={'cuda:2': 'cuda:0'}))
        self.detector = Detect(self.num_classes, 0, self.cfg)
        self.criterion = MultiBoxLoss(self.num_classes, 0.5, True, 0, True, 3, 0.5, False)
        self.priorbox = PriorBox(self.cfg)
        self.testset=self.test_loader.dataset
        with torch.no_grad():
            self.priors = self.priorbox.forward()


    def norm_vgg(self,images):
        tensor_std=torch.ones(3).cuda()
        tensor_mean=torch.FloatTensor([0.408,0.459,0.482]).cuda()
        new=(images-tensor_mean.view(-1,3,1,1))/tensor_std.view(-1,3,1,1)
        return new

    def forward(self,data):
        imgs,targets=data
        imgs=imgs.cuda()
        with torch.no_grad():
            targets = [anno.cuda() for anno in targets]
        lighten_rgb=imgs
        if(self.lighten_mode!="no"):
            if(len(self.models)==2):
                # HDRNet
                lighten_rgb=self.models[1](imgs)
            else:
                # unet+fitting
                fit_ratio=self.models[2](imgs)
                mulratio_imgs=imgs*fit_ratio.view(-1,1,1,1)
                lighten_rgb=self.models[1](mulratio_imgs)
        if(lighten_rgb.size(1)==3):
            rgb_gen=self.norm_vgg(lighten_rgb)
        else:
            rgb_gen=lighten_rgb
        out = self.models[0](rgb_gen)
        loss_l, loss_c = self.criterion(out, self.priors, targets)
        return rgb_gen,out,loss_l,loss_c

    def train(self):
        write_dict={}
        rgb_gen,_,loss_l,loss_c=self.forward(self.request.data)
        loss = loss_l + loss_c
        loss.backward()
        self.optimize_all()
        self.zero_grad_for_all()
        write_dict["train_total_loss"]=loss.detach().cpu().item()
        write_dict["train_loss_l"]=loss_l.detach().cpu().item()
        write_dict["train_loss_c"]=loss_c.detach().cpu().item()
        if(self.request.step%20==0):
            self.write_log(write_dict,self.request.step)
            self.print_log(write_dict,self.request.epoch,self.request.iteration)

    def after_train(self):
        if((self.request.epoch+1)%10==0):
            self.if_validate=True
        else:
            self.if_validate=False
        if(self.request.epoch in self.learning_rate_decay_epochs):
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/10


    def validate(self):
        rgb_gen,out,loss_l,loss_c=self.forward(self.request.data)
        self.counts.append(rgb_gen.size(0))
        rgb_gen=rgb_gen.detach().permute(0,2,3,1).cpu().numpy()
        rgb_gen[rgb_gen>1.0]=1.0
        rgb_gen[rgb_gen<0.0]=0.0
        self.conf_loss.append(loss_c.item())
        self.loc_loss.append(loss_l.item())
        self.total_loss.append(loss_l.item()+loss_c.item())
        rgb_gen=rgb_gen*255
        rgb_gen=rgb_gen.astype(np.uint8)
        for i in range(0,len(rgb_gen)):
            self.images.append(rgb_gen[i])

        boxes, scores = self.detector.forward(out, self.priors)
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        for bi in range(len(boxes)):
            for bj in range(4):
                boxes[bi,bj]=max(min(boxes[bi,bj],1),0)
        scale = torch.Tensor([5496,3670,
                              5496,3670]).cpu().numpy()
        boxes *= scale
        i=self.request.step
        for j in range(1, self.num_classes):
            inds = np.where(scores[:, j] > self.thresh)[0]
            if len(inds) == 0:
                self.all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            # print('c_bboxes',c_bboxes)
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            cpu = False

            keep = nms(c_dets, 0.45, force_cpu=cpu,device_id=self.config["device_use"][0])
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            self.all_boxes[j][i] = c_dets
        if self.max_per_image > 0:
            image_scores = np.hstack([self.all_boxes[j][i][:, -1] for j in range(1, self.num_classes)])
            if len(image_scores) > self.max_per_image:
                image_thresh = np.sort(image_scores)[-self.max_per_image]
                for j in range(1, self.num_classes):
                    keep = np.where(self.all_boxes[j][i][:, -1] >= image_thresh)[0]
                    self.all_boxes[j][i] = self.all_boxes[j][i][keep, :]

    def after_validate(self):
        with open(self.det_file, 'wb') as f:
            pickle.dump(self.all_boxes, f, pickle.HIGHEST_PROTOCOL)
        print('Evaluating detections')
        APs, mAP = self.testset.evaluate_detections(self.all_boxes, self.testresult_path,self.config["task_name"])
        write_dict={}
        write_dict["test_total_loss"]=np.mean(np.array(self.total_loss))
        write_dict["test_loc_loss"]=np.mean(np.array(self.loc_loss))
        write_dict["test_conf_loss"]=np.mean(np.array(self.conf_loss))
        write_dict["test_bicycle_AP"]=APs[0]
        write_dict["test_car_AP"]=APs[1]
        write_dict["test_person_AP"]=APs[2]
        write_dict["test_mAP"]=mAP
        utils.write_images(self.images,self.request.epoch+1,self.saveimg_path)
        self.write_log(write_dict,self.request.epoch+1)
        self.print_log(write_dict,self.request.epoch+1,0)
        if(write_dict["test_total_loss"]<self.best_value):
            self.best_value=write_dict["test_total_loss"]
            self.save_params("best")
        self.empty_data()

    def test(self):
        self.validate()

    def after_test(self):
        self.after_validate()
