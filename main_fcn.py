<<<<<<< HEAD
from solver import FCN_solver
=======
import solver.FCN_solver as FCN_solver
>>>>>>> 0ae49290142df43a050f5a3528f64cd6c8d0c1d7
import torch
import torch.nn as nn
import torch.utils.data as Data
import dataset
import model
import model_utils.runner as R

def generate_dataset(batch_size):
    thu_night=dataset.THUNight_static.DifLightDataset(return_ratio=False)
    loader=Data.DataLoader(thu_night,batch_size=batch_size,shuffle=True,num_workers=32)
    return loader,loader,loader

def generate_optimizer(models,learning_rate,weight_decay=0):
    optimizer = torch.optim.Adam(models[0].parameters(
    ), lr=learning_rate)
    return [optimizer]

batch_size=2
type="SID"
usepack=True
lr=0.0001
task_name="SID_"+type+"_batchsize_"+str(batch_size)+"_lr_"+str(lr)+"_task"


config=FCN_solver.FCN_solver.get_default_config()
config.task_name=task_name
config.epochs=4000
config.dataset_function=generate_dataset
config.dataset_function_params={"batch_size":2}
config.model_classes=[model.U_net.U_net]
config.model_params=[{}]
config.optimizer_function=generate_optimizer
config.optimizer_params={"learning_rate":lr}
config.memory_use=[10000,10000]

SID_task={
"solver":{"class":FCN_solver.FCN_solver,"params":{}},
"config":config
}

tasks=[SID_task]

runner=R.runner()
runner.add_tasks(tasks)
runner.main_loop()
