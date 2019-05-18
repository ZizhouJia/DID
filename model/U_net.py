import torch
import torch.nn as nn
import math
import numpy as np
import time

class U_net(nn.Module):
    def __init__(self,input_depth=4,output_depth=12):
        super(U_net,self).__init__()
        self.activate=nn.LeakyReLU(0.2)
        kernels_list=[input_depth,32,64,128,256,512]
        self.down=nn.ModuleList()
        self.randint=np.random.randint(0,100000)
        for i in range(0,4):
            conv1=nn.Conv2d(kernels_list[i],kernels_list[i+1],3,1,1)
            conv2=nn.Conv2d(kernels_list[i+1],kernels_list[i+1],3,1,1)
            max_pool=nn.MaxPool2d(2)
            self.down.append(conv1)
            self.down.append(conv2)
            self.down.append(max_pool)

        self.conv_middle1=nn.Conv2d(kernels_list[4],kernels_list[5],3,1,1)
        self.conv_middle2=nn.Conv2d(kernels_list[5],kernels_list[5],3,1,1)

        self.pow=math.sqrt(output_depth/3)
        self.pow=int(self.pow)

        self.up=nn.ModuleList()
        for i in range(0,4):
            deconv1=nn.ConvTranspose2d(kernels_list[5-i],kernels_list[4-i],2,2)#,padding=1,output_padding=1)
            conv1=nn.Conv2d(kernels_list[4-i]*2,kernels_list[4-i],3,1,1)
            conv2=nn.Conv2d(kernels_list[4-i],kernels_list[4-i],3,1,1)

            self.up.append(deconv1)
            self.up.append(conv1)
            self.up.append(conv2)

        self.conv_out=nn.Conv2d(kernels_list[1],output_depth,1,1)

    def forward(self,x):
        conv_out=[]
        out=x

        for i in range(0,4):
            out=self.down[i*3+0](out)
            out=self.activate(out)
            out=self.down[i*3+1](out)
            out=self.activate(out)
            conv_out.append(out)
            out=self.down[i*3+2](out)
            out=self.activate(out)

        out=self.conv_middle1(out)
        out=self.activate(out)
        out=self.conv_middle2(out)
        out=self.activate(out)

        for i in range(0,4):
            out=self.up[3*i+0](out)
            out=self.activate(out)
            out=torch.cat((out,conv_out[3-i]),1)
            out=self.up[3*i+1](out)
            out=self.activate(out)
            out=self.up[3*i+2](out)
            out=self.activate(out)

        out=self.conv_out(out)
        C=out.size(1)/self.pow/self.pow
        H=out.size(2)
        W=out.size(3)

        param1=out.size(1)/self.pow/self.pow
        param2=out.size(2)*self.pow
        param3=out.size(3)*self.pow

        output = nn.functional.pixel_shuffle(out, 2)


        return output
