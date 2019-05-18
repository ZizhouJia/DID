import model_utils.solver as solver
import model_utils.utils as utils


class FCN_solver(solver.common_solver):
    def __init__(self):
        super(FCN_solver,self).__init__()
        self.images=[]
        self.counts=[]
        self.loss=[]
        self.psnrs=[]
        self.ssims=[]

    def get_defualt_config():
        config=solver.common_solver.get_defualt_config()
        config["mode"]="SID"
        config["learning_rate_decay_epochs"]=[2000]

    def empty_data(self):
        self.images=[]
        self.counts=[]
        self.loss=[]
        self.psnrs=[]
        self.ssims=[]

    def load_config(self):
        super(FCN_solver,self).load_config()
        self.mode=self.config["mode"]
        self.learning_rate_decay_epochs=self.config["learning_rate_decay_epochs"]

    def forward(self,data):
        x,y,xishu=data
        x=x.cuda()
        y=y.cuda()
        output=self.models[0](x)
        loss=torch.abs(y-output).mean()
        return output,y,loss

    def train(self):
        y,loss=self.forward(self.request.data)
        loss.backward()
        self.optimize_all()
        self.zero_grad_for_all()
        self.write_dict["train_loss"]=loss.detach().cpu().item()
        if(self.request.step%20==0):
            self.write_log(self.write_dict,self.request.step)
            self.print_log(self.write_dict,self.epoch,self.iteration)

    def after_train(self):
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
            loss.append(np.abs(y-output).mean())
            psnr=utils.PSNR(output[i],y[i])
            ssim=utils.SSIM(output[i],y[i])
            self.psnrs.append(psnr)
            self.ssim.append(ssim)
        y=y*255
        y=y.astype(np.uint8)
        for i in range(0,len(y)):
            self.images.append(y[i])

    def after_validate(self):
        if(self.request.epoch%10!=0):
            return
        write_dict={}
        write_dict["test_psnr"]=np.mean(np.array(self.psnrs))
        write_dict["test_ssim"]=np.mean(np.array(self.ssims))
        write_dict["test_loss"]=np.mean(np.array(self.loss))
        utils.write_images(self.images,self.request.epoch)
        self.write_log(write_dict,self.request.epoch)
        self.print_log(write_dict,self.request,0)
        self.empty_data()

    def test(self):
        self.validate()

    def after_test(self):
        self.after_validate()
