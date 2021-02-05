import solver.SSD_solver as SSD_solver
import torch
import torch.nn as nn
import torch.utils.data as Data
import dataset
import model
import model_utils.runner as R

def generate_dataset(batch_size,imgtype='raw',lighten_net="unet"):
    root="/home/huangxiaoyu/data/THUNIGHT/detection/"
    thu_night_train=dataset.THUNight_detection.detection_dataset(root=root,image_type=imgtype,split='train',lighten_net=lighten_net)
    train_loader=Data.DataLoader(thu_night_train,batch_size=batch_size,shuffle=True,collate_fn=dataset.THUNight_detection.detection_collate)
    thu_night_test=dataset.THUNight_detection.detection_dataset(root=root,image_type=imgtype,split='test',lighten_net=lighten_net)
    test_loader=Data.DataLoader(thu_night_test,batch_size=1,shuffle=False,collate_fn=dataset.THUNight_detection.detection_collate)
    return train_loader,test_loader,test_loader


def generate_optimizer(models,learning_rate,weight_decay=1e-4,momentum=0.9):
    optimizer = torch.optim.SGD(models[0].parameters(), lr=learning_rate[0],
                      momentum=momentum, weight_decay=weight_decay)
    if(len(models)==1):
        return [optimizer]
    else:
        optimizer_unet = torch.optim.Adam(models[1].parameters(
        ), lr=learning_rate[1])
        if(len(models)==2):
            return [optimizer,optimizer_unet]
        else:
            optimizer_fit = torch.optim.Adam(models[2].parameters(
            ), lr=learning_rate[2])
            return [optimizer,optimizer_unet,optimizer_fit]

batch_size=16
type="rgb"
lighten_net="unet" #hdrnet,unet,no
inc=4
if(type=="rgb"):
    inc=3
pretrainedvgg=True
lr=[0.001,0,0]
task_name="SSD_"+type+"_"+lighten_net+"_batchsize_"+str(batch_size)+"_lr_"+str(lr)+"_pretrainedvgg_"+str(pretrainedvgg)+"_task"

config=SSD_solver.SSD_solver.get_default_config()
config["task_name"]=task_name
config["epochs"]=400
config["dataset_function"]=generate_dataset
config["dataset_function_params"]={"batch_size":batch_size,"imgtype":type,"lighten_net":lighten_net}
if lighten_net=='hdrnet':
    config["model_class"]=[model.RFB_Net_vgg.RFBNet,model.HDRNet.HDRNet]
    config["model_params"]=[{"size":512,"num_classes":4},{"inc":inc}]
elif lighten_net=='unet':
    config["model_class"]=[model.RFB_Net_vgg.RFBNet,model.U_net.U_net,model.Net1.Net1]
    config["model_params"]=[{"size":512,"num_classes":4},{"input_depth":inc},{"in_channels":inc}]
else:
    config["model_class"]=[model.RFB_Net_vgg.RFBNet,model.U_net.U_net]
    config["model_params"]=[{"size":512,"num_classes":4,"inc":inc},{}]
config["optimizer_function"]=generate_optimizer
config["optimizer_params"]={"learning_rate":lr}
config["mem_use"]=[11150,11150]


SSD_task={
"solver":{"class":SSD_solver.SSD_solver,
        "params":{"saveimg_path":"result_images/"+task_name,
                  "testresult_path":"test_results/"+task_name,
                  "lighten_mode":lighten_net,#no,unet,hdrnet
                  "pretrainedvgg":pretrainedvgg}
                  },
"config":config
}



tasks=[SSD_task]
if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    runner=R.runner()
    runner.generate_tasks(tasks)
    runner.main_loop()
