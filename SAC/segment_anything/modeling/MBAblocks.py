import torch
from torch import nn
from torch.nn import functional as F
from .MBA import CBM
class MBAblocks(nn.Module):
    def __init__(self, in_dim,out_dim, hidden_dim, width, norm, act):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.out_dim=out_dim
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias = False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride= 1,padding = 1)
        self.cbm1 = CBM(128)
        self.cbm2 = CBM(256)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        self.ap = nn.AdaptiveAvgPool2d(16)


        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim , out_dim, 1, bias = False),
            norm(out_dim),
            act()
        )
    
    def forward(self, x):
        mid = self.in_conv(x)

        out = mid
        #print(out.shape)
        outlist=[]
        for i in range(self.width - 1):
            mid = self.pool(mid)
            out = self.cbm1.forward(mid)
            outlist.append(out)

        out = (outlist[1]-outlist[0])+(outlist[1]-outlist[2])
        out = self.ap(out)
        out = self.cbm1.forward(out)
        out = self.out_conv(out)

        return out



