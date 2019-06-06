import torch
import torch.nn as nn
import torch.nn.functional as F

class Fit_net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Fit_net,self).__init__()
        channels=[n_feature,16,16,32,32,64,64]#64,128,256,512,1024]
        layers=[]
        for i in range(2):
            layers.append(nn.Conv2d(in_channels=channels[i],out_channels=channels[i+1],kernel_size=5,stride=1,padding=2,bias=False))
            # layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_channels=channels[i+1],out_channels=channels[i+1],kernel_size=3,stride=1,padding=1,bias=False))
            # layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.AvgPool2d(kernel_size=2))

        for i in range(2,6):
            layers.append(nn.Conv2d(in_channels=channels[i],out_channels=channels[i+1],kernel_size=3,stride=1,padding=1,bias=False))
            # layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.AvgPool2d(kernel_size=2))

        self.layers=nn.Sequential(*layers)

        self.hidden_layer=torch.nn.Linear(16*16*channels[6],n_hidden)
        self.hidden_layer2=torch.nn.Linear(n_hidden,n_hidden//2)
        self.pred_layer=torch.nn.Linear(n_hidden//2,n_output)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        out=self.layers(x)
        fc=out.view(out.size(0),-1)
        # print('fc',fc.size())
        hi=self.hidden_layer(fc)
        # print('hi',hi.size())
        ri=F.relu(hi)
        hii=self.hidden_layer2(ri)
        # print('hii',hii.size())
        rii=F.relu(hii)
        pred=self.pred_layer(rii)
        self.sigmoid(pred)
        return pred
