import model_utils.solver as solver
import model_utils.utils as utils
import pysnooper
import torch
import numpy as np

class FCN_Fitting_solver(solver.common_solver):
    def __init__(self,fitting_model_path,saveimg_path):
        super(FCN_Fitting_solver,self).__init__()
        self.images=[]
        self.counts=[]
        self.total_loss=[]
        self.fitting_loss=[]
        self.img_loss=[]
        self.psnrs=[]
        self.ssims=[]
        self.best_value=100.0
        self.fitting_model_path=fitting_model_path
        self.saveimg_path=saveimg_path

    def get_default_config():
        config=solver.common_solver.get_default_config()
        config["mode"]="FCN_Fitting"
        config["learning_rate_decay_epochs"]=[2000]
        return config

    def empty_data(self):
        self.images=[]
        self.counts=[]
        self.total_loss=[]
        self.fitting_loss=[]
        self.img_loss=[]
        self.psnrs=[]
        self.ssims=[]

    def load_config(self):
        super(FCN_Fitting_solver,self).load_config()
        self.mode=self.config["mode"]
        self.learning_rate_decay_epochs=self.config["learning_rate_decay_epochs"]
        self.models[1].load_state_dict(torch.load(self.fitting_model_path))


    def forward(self,data):
        x,y,ratio,id=data
        x=x.cuda()
        y=y.cuda()
        ratio=ratio.cuda()
        fit_ratio=self.models[1](x)
        fit_loss=torch.abs(ratio.double()-fit_ratio.double()).mean()
        mulratio_imgs=x*fit_ratio.view(-1,1,1,1)
        output=self.models[0](mulratio_imgs)
        img_loss=torch.abs(output.double()-y.double()).mean()
        return output,y,ratio,fit_ratio,fit_loss,img_loss

    def train(self):
        write_dict={}
        output,y,_,_,fit_loss,img_loss=self.forward(self.request.data)
        loss=fit_loss*0.1+img_loss
        loss.backward()
        self.optimize_all()
        self.zero_grad_for_all()
        write_dict["train_total_loss"]=loss.detach().cpu().item()
        write_dict["train_fit_loss"]=fit_loss.detach().cpu().item()
        write_dict["train_img_loss"]=img_loss.detach().cpu().item()
        if(self.request.step%20==0):
            self.write_log(write_dict,self.request.step)
            self.print_log(write_dict,self.request.epoch,self.request.iteration)

    def after_train(self):
        if(self.request.epoch%100==0):
            self.if_validate=True
        else:
            self.if_validate=False
        if(self.request.epoch in self.learning_rate_decay_epochs):
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/10


    def validate(self):
        if(self.request.epoch%100!=0):
            return
        output,y,ratio,fit_ratio,_,_=self.forward(self.request.data)
        self.counts.append(y.size(0))
        y=y.detach().permute(0,2,3,1).cpu().numpy()
        output=output.detach().permute(0,2,3,1).cpu().numpy()
        output[output>=1.0]=1.0
        output[output<=0.0]=0.0
        y[y>1.0]=1.0
        y[y<0.0]=0.0
        ratio=ratio.detach().cpu().numpy()
        fit_ratio=fit_ratio.detach().cpu().numpy()
        for i in range(0,len(y)):
            img_loss=np.abs(y[i]-output[i]).mean()
            fit_loss=np.abs(fit_ratio[i]-ratio[i]).mean()
            self.img_loss.append(img_loss)
            self.fitting_loss.append(fit_loss)
            self.total_loss.append(img_loss+fit_loss*0.1)
            psnr=utils.PSNR(output[i],y[i])
            ssim=utils.SSIM(output[i],y[i])
            self.psnrs.append(psnr)
            self.ssims.append(ssim)
        y=y*255
        output=output*255
        output=output.astype(np.uint8)
        y=y.astype(np.uint8)
        y=np.concatenate((y,output),axis=2)
        for i in range(0,len(y)):
            self.images.append(y[i])

    def after_validate(self):
        if(self.request.epoch%100!=0):
            return
        write_dict={}
        write_dict["test_psnr"]=np.mean(np.array(self.psnrs))
        write_dict["test_ssim"]=np.mean(np.array(self.ssims))
        write_dict["test_total_loss"]=np.mean(np.array(self.total_loss))
        write_dict["test_img_loss"]=np.mean(np.array(self.img_loss))
        write_dict["test_fitting_loss"]=np.mean(np.array(self.fitting_loss))
        utils.write_images(self.images,self.request.epoch,self.saveimg_path)
        self.write_log(write_dict,self.request.epoch)
        self.print_log(write_dict,self.request.epoch,0)
        if(write_dict["test_total_loss"]<self.best_value):
            self.best_value=write_dict["test_total_loss"]
            self.save_params("best")
        self.empty_data()

    def test(self):
        self.validate()

    def after_test(self):
        self.after_validate()
