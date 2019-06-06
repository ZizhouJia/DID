import solver.HDRNet_solver as HDRNet_solver
import torch
import torch.nn as nn
import torch.utils.data as Data
import dataset
import model
import model_utils.runner as R

def generate_dataset(batch_size,imgtype='raw',usenet='hdrnet',usepack=True):
    thu_night=dataset.THUNight_static.DifLightDataset(imgtype=imgtype,usenet=usenet,usepack=usepack,return_ratio=True)
    loader=Data.DataLoader(thu_night,batch_size=batch_size,shuffle=True)
    return loader,loader,loader

def generate_optimizer(models,learning_rate,weight_decay=0):
    optimizer = torch.optim.Adam(models[0].parameters(
    ), lr=learning_rate,weight_decay=weight_decay)
    return [optimizer]

batch_size=14
type="raw"
usepack=True
inc=4
lr=0.01
if(inc==3 and type=="raw"):
    # postprocess
    usepack=False
task_name="HDRNet_"+type+str(inc)+"_batchsize_"+str(batch_size)+"_lr_"+str(lr)+"_task"

config=HDRNet_solver.HDRNet_solver.get_default_config()
config["task_name"]=task_name
config["epochs"]=200
config["dataset_function"]=generate_dataset
config["dataset_function_params"]={"batch_size":batch_size,"imgtype":type,"usenet":'hdrnet',"usepack":usepack}
config["model_class"]=[model.HDRNet.HDRNet]
config["model_params"]=[{"inc":inc}]
config["optimizer_function"]=generate_optimizer
config["optimizer_params"]={"learning_rate":lr}
config["mem_use"]=[10000,10000]

HDR_task={
"solver":{"class":HDRNet_solver.HDRNet_solver,"params":{"saveimg_path":"result_images/"+task_name}},
"config":config
}

tasks=[HDR_task]

runner=R.runner()
runner.generate_tasks(tasks)
runner.main_loop()
