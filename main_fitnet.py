import solver.FitNet_solver as FitNet_solver
import torch
import torch.nn as nn
import torch.utils.data as Data
import dataset
import model
import model_utils.runner as R

def generate_dataset(batch_size,imgtype='raw',usenet='unet',usepack=True):
    thu_night=dataset.THUNight_static.DifLightDataset(imgtype=imgtype,usenet=usenet,usepack=usepack,return_ratio=True)
    loader=Data.DataLoader(thu_night,batch_size=batch_size,shuffle=True)
    return loader,loader,loader,thu_night

def generate_optimizer(models,learning_rate,weight_decay=0):
    optimizer = torch.optim.Adam(models[0].parameters(
    ), lr=learning_rate,weight_decay=weight_decay)
    return [optimizer]

batch_size=98
type="raw"
usepack=True
lr=0.01
task_name="FitNet_"+type+"_batchsize_"+str(batch_size)+"_lr_"+str(lr)+"_task"

config=FitNet_solver.FitNet_solver.get_default_config()
config["task_name"]=task_name
config["epochs"]=150
config["dataset_function"]=generate_dataset
config["dataset_function_params"]={"batch_size":batch_size,"imgtype":type,"usenet":'hdrnet',"usepack":usepack}
config["model_class"]=[model.Net1.Net1]
config["model_params"]=[{}]
config["optimizer_function"]=generate_optimizer
config["optimizer_params"]={"learning_rate":lr}
config["mem_use"]=[10000,10000]

FitNet_task={
"solver":{"class":FitNet_solver.FitNet_solver,"params":{}},
"config":config
}

tasks=[FitNet_task]

runner=R.runner()
runner.generate_tasks(tasks)
runner.main_loop()
