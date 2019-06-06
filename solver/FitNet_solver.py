import model_utils.solver as solver
import model_utils.utils as utils
import pysnooper
import torch
import numpy as np

class FitNet_solver(solver.common_solver):
    def __init__(self):
        super(FitNet_solver,self).__init__()
        self.images=[]
        self.counts=[]
        self.loss=[]
        self.best_value=100.0

    def get_default_config():
        config=solver.common_solver.get_default_config()
        config["mode"]="FitNet"
        config["learning_rate_decay_epochs"]=[30,60,90]
        return config

    def empty_data(self):
        self.images=[]
        self.counts=[]
        self.loss=[]


    def load_config(self):
        super(FitNet_solver,self).load_config()
        self.mode=self.config["mode"]
        self.learning_rate_decay_epochs=self.config["learning_rate_decay_epochs"]

    def forward(self,data):
        x,y,ratio,id=data
        x=x.cuda()
        y=y.cuda()
        ratio=ratio.cuda()
        output=self.models[0](x)
        loss=torch.abs(ratio.double()-output.double()).mean()
        return output,ratio,loss

    def train(self):
        write_dict={}
        output,y,loss=self.forward(self.request.data)
        loss.backward()
        self.optimize_all()
        self.zero_grad_for_all()
        write_dict["train_loss"]=loss.detach().cpu().item()
        if(self.request.step%20==0):
            self.write_log(write_dict,self.request.step)
            self.print_log(write_dict,self.request.epoch,self.request.iteration)

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
        output,ratio,loss=self.forward(self.request.data)
        self.counts.append(output.size(0))
        ratio=ratio.detach().cpu().numpy()
        output=output.detach().cpu().numpy()
        for i in range(0,len(ratio)):
            self.loss.append(np.abs(ratio[i]-output[i]).mean())

    def after_validate(self):
        if(self.request.epoch%10!=0):
            return
        write_dict={}
        write_dict["test_loss"]=np.mean(np.array(self.loss))
        self.write_log(write_dict,self.request.epoch)
        self.print_log(write_dict,self.request.epoch,0)
        if(write_dict["test_loss"]<self.best_value):
            self.best_value=write_dict["test_loss"]
            self.save_params("best")
        self.empty_data()

    def test(self):
        self.validate()

    def after_test(self):
        self.after_validate()
