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
        self.modules = []
        nchannels = dims[0]

        self.func = F.relu

        avgpool1 = nn.AvgPool3d(2,stride=2)
        self.add_module('avgpool_0', avgpool1)
        self.modules.append(avgpool1)
        conv1 = nn.Conv3d(nchannels,out_channels=32,padding=1,kernel_size=3,stride=1) 
        self.add_module('conv1',conv1)
        self.modules.append(conv1)
        conv2 = nn.Conv3d(32,out_channels=32,padding=0,kernel_size=1,stride=1) 
        self.add_module('conv2',conv2)
        self.modules.append(conv2)
        avgpool2 = nn.AvgPool3d(2,stride=2)
        self.add_module('avgpool_1', avgpool2)
        self.modules.append(avgpool2)
        conv3 = nn.Conv3d(32,out_channels=64,padding=1,kernel_size=3,stride=1) 
        self.add_module('conv3',conv3)
        self.modules.append(conv3)
        conv4 = nn.Conv3d(64,out_channels=64,padding=0,kernel_size=1,stride=1) 
        self.add_module('conv4',conv4)
        self.modules.append(conv4)
        avgpool3 = nn.AvgPool3d(2,stride=2)
        self.add_module('avgpool_2', avgpool3)
        self.modules.append(avgpool3)
        conv5 = nn.Conv3d(64,out_channels=128,padding=1,kernel_size=3,stride=1) 
        self.add_module('conv5',conv5)
        self.modules.append(conv5)
        div = 2*2*2
        last_size = int(dims[1]//div * dims[2]//div * dims[3]//div * 128)
        flattener = View((-1,last_size))
        self.add_module('flatten',flattener)
        self.modules.append(flattener)
        fc = nn.Linear(last_size,num_classes)
        self.add_module('fc',fc)
        self.modules.append(fc)
        self.ll = nn.Linear(num_classes, 1)
        self.add_module('ll',self.ll)
        

    def forward(self, x): 
        for layer in self.modules:
            x = layer(x)
            if isinstance(layer, nn.Conv3d) or isinstance(layer, nn.Linear):
                x = self.func(x)
        return self.ll(x)
