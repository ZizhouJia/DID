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

    def forward(self,data):
        x,y,id=data
        x=x.cuda()
        y=y.cuda()
        output=self.models[0](x)
        loss=torch.abs(y-output).mean()
        return output,y,loss

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
        if(self.request.epoch%10==0):
            self.if_validate=True
        else:
            self.if_validate=False
        if(self.request.epoch in self.learning_rate_decay_epochs):
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/10


    def validate(self):
        if(self.request.epoch%10!=0):
            return
        output,y,loss=self.forward(self.request.data)
        self.counts.append(y.size(0))
        y=y.detach().permute(0,2,3,1).cpu().numpy()
        output=output.detach().permute(0,2,3,1).cpu().numpy()
        output[output>=1.0]=1.0
        output[output<=0.0]=0.0
        y[y>1.0]=1.0
        y[y<0.0]=0.0
        for i in range(0,len(y)):
            self.loss.append(np.abs(y[i]-output[i]).mean())
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
        if(self.request.epoch%10!=0):
            return
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
