import FCN_solver
import torch
import torch.nn as nn
import torch.utils.data as Data
import dataset
import model
import model_utils.runner as R

def generate_dataset(batch_size):
    thu_nigth=dataset.THUNight_static.OurDataset(return_ratio=True)
    loader=Data.DataLoader(thu_ngith,batch_size=batch_size,shuffle=True)
    return loader,loader,loader

def generate_optimizer(models,learning_rate,weight_decay=0.0005):
    optimizer = torch.optim.Adam(models[0].parameters(
    ), lr=learning_rate, weight_decay=weight_decay)
    return [optimizer]


config=FCN_solver.FCN_solver.get_defualt_config()
config["task_name"]="SID_task"
config["epochs"]=4000
config["dataset_function"]=generate_dataset
config["dataset_function_params"]={"batch_size":1}
config["model_class"]=[model.U_net.U_net]
config["model_params"]=[{}]
config["optimizer_function"]=generate_optimizer
config["optimizer_params"]={"learning_rate":0.0001}
config["mem_use"]=[10000]

SID_task={
"solver":{"class":FCN_solver.FCN_solver,"params":{}},
"config":config
}

tasks=[SID_task]

runner=R.runner()
runner.generate_tasks(tasks)
runner.main_loop()
