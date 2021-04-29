import torch.nn as nn

class SiameseNet(nn.Module):
    def __init__(self, arms, rep_size):
        super(SiameseNet,self).__init__()
        self.arm = arms
        self.dg = nn.Linear(rep_size,1)
        self.ddg = nn.Linear(rep_size,1)

    def forward(self, x1,x2):
        lig1 = self.arm(x1)
        lig2 = self.arm(x2)
        lig1_aff = self.dg(lig1)
        lig2_aff = self.dg(lig2)
        diff = (lig1 - lig2)
        return self.ddg(diff), lig1_aff, lig2_aff
