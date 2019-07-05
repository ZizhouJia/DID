import model_utils.solver as solver
import model_utils.utils as utils
import torch
import numpy as np



class FCN_solver(solver.common_solver):
    def __init__(self):
        super(FCN_solver,self).__init__()
        self.images=[]
        self.counts=[]
        self.loss=[]
        self.psnrs=[]
        self.ssims=[]
        self.best_value=0.0


    @staticmethod
    def get_default_config():
        config=solver.common_solver.get_defualt_config()
        config.mode="SID"
        config.learning_rate_decay_epochs=[2000]
        return config

    def empty_data(self):
        self.images=[]
        self.counts=[]
        self.loss=[]
        self.psnrs=[]
        self.ssims=[]

    def load_config(self):
        super(FCN_solver,self).load_config()
        self.mode=self.config.mode
        self.learning_rate_decay_epochs=self.config.learning_rate_decay_epochs

    def inference(self,images,ratio):
        if(len(images.size())==3):
            images=images.unsqueeze(0)
        images=images*ratio
        # images=padding_images(images)
        output_images=self.models[0](images)
        # output_images=cut_images(output_images.detach())
        return output_images.detach().permute(0,2,3,1).cpu().numpy()

    def forward(self,data):
        id,ratio,x,y=data
        x=x.cuda()*ratio.cuda().view(-1,1,1,1).float()
        y=y.cuda()
        output=self.models[0](x)
        loss=torch.abs(y-output).mean()
        return output,y,loss

    def before_train(self):
        self.train_loader.dataset.set_mode("train")
        self.train_loader.dataset.set_return_array(["id","ratio","in_raw","gt_rgb"])

    def train(self):
        write_dict={}
        output,y,loss=self.forward(self.request.data)
        loss.backward()
        self.optimize_all()
        self.zero_grad_for_all()
        write_dict["train_loss"]=loss.detach().cpu().item()
        if(self.request.step%20==0):
            self.writer.write_board_line(write_dict,self.request.step)
            self.writer.write_log(write_dict,self.request.epoch,self.request.iteration)

    def after_train(self):
        if(self.request.epoch%100==0):
            self.if_validate=True
            self.train_loader.dataset.set_mode("test")
            self.train_loader.dataset.set_return_array(["id","ratio","in_raw","gt_raw","in_rgb","gt_rgb","in_raw_ratio_rgb"])
        else:
            self.if_validate=False
        if(self.request.epoch in self.learning_rate_decay_epochs):
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/10


    def validate(self):
        id,ratio,in_raw,gt_raw,in_rgb,gt_rgb,in_raw_ratio_rgb=self.request.data
        in_raw=in_raw.cuda()
        self.counts.append(gt_rgb.size(0))
        outputs=[]
        for i in range(0,in_raw.size(0)):
            output=self.inference(in_raw[0][:,:1600,:2400],ratio[0])
            outputs.append(output)
        outputs=np.concatenate(outputs,axis=0)
        outputs[outputs>=1.0]=1.0
        outputs[outputs<=0.0]=0.0
        gt_rgb=gt_rgb.permute(0,2,3,1).numpy()[:,:3200,:4800,:]
        in_rgb=in_rgb.permute(0,2,3,1).numpy()[:,:3200,:4800,:]
        in_raw_ratio_rgb=in_raw_ratio_rgb.permute(0,2,3,1).numpy()[:,:3200,:4800,:]
        for i in range(0,len(gt_rgb)):
            self.loss.append(np.abs(gt_rgb[i]-outputs[i]).mean())
            psnr=utils.PSNR(outputs[i],gt_rgb[i])
            ssim=utils.SSIM(outputs[i],gt_rgb[i])
            self.psnrs.append(psnr)
            self.ssims.append(ssim)

        target_image=np.concatenate((in_rgb,in_raw_ratio_rgb,gt_rgb,outputs),axis=1)
        target_image=target_image*255
        target_image=target_image.astype(np.uint8)
        for i in range(0,len(target_image)):
            self.images.append(target_image[i])

    def after_validate(self):
        write_dict={}
        write_dict["test_psnr"]=np.mean(np.array(self.psnrs))
        write_dict["test_ssim"]=np.mean(np.array(self.ssims))
        write_dict["test_loss"]=np.mean(np.array(self.loss))
        self.writer.write_file_image(self.images,self.request.epoch)
        self.writer.write_board_line(write_dict,self.request.epoch)
        self.writer.write_log(write_dict,self.request.epoch,0)
        if(write_dict["test_psnr"]>self.best_value):
            self.best_value=write_dict["test_psnr"]
            self.saver.save_params(self.models[0],self.get_task_identifier()+"_best.pkl")
        self.empty_data()

    def test(self):
        self.validate()

    def after_test(self):
        self.after_validate()
