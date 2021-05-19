import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self,shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

class Net(nn.Module):
    def __init__(self, dims,num_classes=1024):
        super(Net, self).__init__()
        nchannels = dims[0]

        self.func = F.relu

        self.avgpool1 = nn.AvgPool3d(2,stride=2)
        self.add_module('avgpool_0', self.avgpool1)
        self.conv1 = nn.Conv3d(nchannels,out_channels=32,padding=1,kernel_size=3,stride=1) 
        self.add_module('unit1_conv',self.conv1)
        self.conv2 = nn.Conv3d(32,out_channels=32,padding=0,kernel_size=1,stride=1) 
        self.add_module('unit2_conv',self.conv2)
        self.avgpool2 = nn.AvgPool3d(2,stride=2)
        self.add_module('avgpool_1', self.avgpool2)
        self.conv3 = nn.Conv3d(32,out_channels=64,padding=1,kernel_size=3,stride=1) 
        self.add_module('unit3_conv',self.conv3)
        self.conv4 = nn.Conv3d(64,out_channels=64,padding=0,kernel_size=1,stride=1) 
        self.add_module('unit4_conv',self.conv4)
        self.avgpool3 = nn.AvgPool3d(2,stride=2)
        self.add_module('avgpool_2', self.avgpool3)
        self.conv5 = nn.Conv3d(64,out_channels=128,padding=1,kernel_size=3,stride=1) 
        self.add_module('unit5_conv',self.conv5)
        div = 2*2*2
        last_size = int(dims[1]//div * dims[2]//div * dims[3]//div * 128)
        self.flattener = View((-1,last_size))
        self.add_module('flatten',self.flattener)
        self.fc = nn.Linear(last_size,num_classes)

    def forward(self, x): 
        x = self.func(self.conv1(self.avgpool1(x)))
        x = self.func(self.conv2(x))
        x = self.func(self.conv3(self.avgpool2(x)))
        x = self.func(self.conv4(x))
        x = self.func(self.conv5(self.avgpool3(x)))
        x = self.flattener(x)
        return self.fc(x)
        # return x
