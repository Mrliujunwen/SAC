import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type
from .FFCM import Fused_Fourier_Conv_Mixer

class LMFC(nn.Module):#b,c,256,256
    def __init__(
            self,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3,64,3,1,1)
        self.conv1 = nn.Conv2d(64,128,3,2,1)
        self.conv2 = nn.Conv2d(128,256,3,2,1)
        self.conv3 = nn.Conv2d(256,512,3,2,1)
        self.conv4 = nn.Conv2d(512,768,3,2,1)

        self.relu = nn.ReLU()
        self.conv_3x11 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1), stride=2, padding=(1,0))
        self.conv_1x31 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=2, padding=(0, 1))

        self.conv_3x12 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 1), stride=2, padding=(1, 0))
        self.conv_1x32 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=2, padding=(0, 1))

        self.conv_3x13 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 1), stride=2, padding=(1, 0))
        self.conv_1x33 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 3), stride=2, padding=(0, 1))

        self.conv_3x14 = nn.Conv2d(in_channels=512, out_channels=768, kernel_size=(3, 1), stride=2, padding=(1, 0))
        self.conv_1x34 = nn.Conv2d(in_channels=512, out_channels=768, kernel_size=(1, 3), stride=2, padding=(0, 1))

        self.conv_5x11 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 1), stride=2, padding=(2, 0))
        self.conv_1x51 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=2, padding=(0, 2))

        self.conv_5x12 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 1), stride=2, padding=(2, 0))
        self.conv_1x52 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 5), stride=2, padding=(0, 2))

        self.conv_5x13 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 1), stride=2, padding=(2, 0))
        self.conv_1x53 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 5), stride=2, padding=(0, 2))

        self.conv_5x14 = nn.Conv2d(in_channels=512, out_channels=768, kernel_size=(5, 1), stride=2, padding=(2, 0))
        self.conv_1x54 = nn.Conv2d(in_channels=512, out_channels=768, kernel_size=(1, 5), stride=2, padding=(0, 2))

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(768)
        self.Fused_Fourier_Conv_Mixer1=Fused_Fourier_Conv_Mixer(128)
        self.Fused_Fourier_Conv_Mixer2=Fused_Fourier_Conv_Mixer(256)
        self.Fused_Fourier_Conv_Mixer3=Fused_Fourier_Conv_Mixer(512)

        self.Fused_Fourier_Conv_Mixer4=Fused_Fourier_Conv_Mixer(768)
        self.convl1 = nn.Conv2d(384,768,8,8,0)
        self.convl2 = nn.Conv2d(768,768,4,4,0)
        self.convl3 = nn.Conv2d(1536,768,2,2,0)
        self.convl4 = nn.Conv2d(2304,768,1,1,0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        x13 = self.conv_1x31(x)
        x31 = self.conv_3x11(x)
        x131 = x13 * x31

        x15 = self.conv_1x51(x)
        x51 = self.conv_5x11(x)
        x151 = x15 * x51

        x = self.conv1(x)
        x = self.Fused_Fourier_Conv_Mixer1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x131 = self.bn2(x131)
        x131 = self.relu(x131)
        x151 = self.bn2(x151)
        x151 = self.relu(x151)
        x1=torch.cat([x,x131,x151],dim=1)
##########################################
        x13 = self.conv_1x32(x131)
        x31 = self.conv_3x12(x131)
        x132= x13 * x31

        x15 = self.conv_1x52(x151)
        x51 = self.conv_5x12(x151)
        x152 = x15 * x51

        x132 = self.bn3(x132)
        x132 = self.relu(x132)
        x152 = self.bn3(x152)
        x152 = self.relu(x152)
        x = self.conv2(x)
        x = self.Fused_Fourier_Conv_Mixer2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x2=torch.cat([x,x132,x152],dim=1)

##########################################
        x13 = self.conv_1x33(x132)
        x31 = self.conv_3x13(x132)
        x133= x13 * x31

        x15 = self.conv_1x53(x152)
        x51 = self.conv_5x13(x152)
        x153 = x15 * x51

        x133 = self.bn4(x133)
        x133 = self.relu(x133)
        x153 = self.bn4(x153)
        x153 = self.relu(x153)
        x = self.conv3(x)
        x = self.Fused_Fourier_Conv_Mixer3(x)
        x = self.bn4(x)
        x = self.relu(x)
        x3=torch.cat([x,x133,x153],dim=1)

##########################################
        x13 = self.conv_1x34(x133)
        x31 = self.conv_3x14(x133)
        x134 = x13 * x31

        x15 = self.conv_1x54(x153)
        x51 = self.conv_5x14(x153)
        x154 = x15 * x51
        x = self.conv4(x)
        x = self.Fused_Fourier_Conv_Mixer4(x)
        x = self.bn5(x)
        x = self.relu(x)
        x134 = self.bn5(x134)
        x134 = self.relu(x134)
        x154 = self.bn5(x154)
        x154 = self.relu(x154)
        x4=torch.cat([x,x134,x154],dim=1)
        x1= self.convl1(x1)
        x1=self.bn5(x1)
        x1=self.relu(x1)

        x2= self.convl2(x2)
        x2=self.bn5(x2)
        x2=self.relu(x2)

        x3= self.convl3(x3)
        x3=self.bn5(x3)
        x3=self.relu(x3)

        x4= self.convl4(x4)
        x4=self.bn5(x4)
        x4=self.relu(x4)
        return [x1,x2,x3,x4]
if __name__ == '__main__':
    x=torch.rand(2,3,256,256)#b,768,16,16
    L=LMFC()
    out = L.forward(x)
    # print(out)
    # print(out.shape)