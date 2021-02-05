import solver.FCN_Fitting_solver as FCN_Fitting_solver
import torch
import torch.nn as nn
import torch.utils.data as Data
import dataset
import model
import model_utils.runner as R

def generate_dataset(batch_size,imgtype='raw',usenet='unet',usepack=True):
    thu_night=dataset.THUNight_static.DifLightDataset(imgtype=imgtype,usenet=usenet,usepack=usepack,return_ratio=True)
    loader=Data.DataLoader(thu_night,batch_size=batch_size,shuffle=True)
    return loader,loader,loader

def generate_optimizer(models,learning_rate,weight_decay=0):
    optimizer = torch.optim.Adam(models[0].parameters(
    ), lr=learning_rate,weight_decay=weight_decay)
    optimizer_fit = torch.optim.Adam(models[1].parameters(
    ), lr=learning_rate,weight_decay=weight_decay)
    return [optimizer,optimizer_fit]

batch_size=2
type="raw"
usepack=True
lr=0.0001
task_name="FCN_Fitting_"+type+"_batchsize_"+str(batch_size)+"_lr_"+str(lr)+"_task"

config=FCN_Fitting_solver.FCN_Fitting_solver.get_default_config()
config["task_name"]=task_name
config["epochs"]=4000
config["dataset_function"]=generate_dataset
config["dataset_function_params"]={"batch_size":batch_size,"imgtype":type,"usenet":'unet',"usepack":usepack}
config["model_class"]=[model.U_net.U_net,model.Net1.Net1]
config["model_params"]=[{},{}]
config["optimizer_function"]=generate_optimizer
config["optimizer_params"]={"learning_rate":lr}
config["mem_use"]=[10000,10000]


FCN_Fitting_task={
"solver":{"class":FCN_Fitting_solver.FCN_Fitting_solver,
          "params":
          {"fitting_model_path":"checkpoints/FitNet_raw_batchsize_49_lr_0.01_task/201906031212/best/model-0.pkl","saveimg_path":"result_images/"+task_name}},
"config":config
}

tasks=[FCN_Fitting_task]

runner=R.runner()
runner.generate_tasks(tasks)
runner.main_loop()
