import numpy as np
import torch
import torchvision.transforms as transforms
import os
import torch.utils.data as Data
class Net1(torch.nn.Module):
    def __init__(self,in_channels=4):
        super(Net1, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.AvgPool2d(10),
            torch.nn.Conv2d(in_channels, 8, 1),
            #torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3, dilation=3),
            torch.nn.BatchNorm2d(16),
            #torch.nn.ReLU(),
            #torch.nn.Conv2d(8, 16, 3, dilation=9),
            #torch.nn.BatchNorm2d(16),
            torch.nn.AdaptiveMaxPool2d(1)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        conv_out = self.conv(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        #out = torch.div(torch.Tensor([inverse_ratio]).to(device), out)
        return out
