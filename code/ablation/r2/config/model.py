import torch
from torch import nn
import torch.nn.functional as F
import settings
from itertools import combinations,product
import math

class Residual_Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(Residual_Block, self).__init__()
        self.channel_num = settings.channel
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.convert = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1),
            nn.LeakyReLU(0.2)
        )
        self.res = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x):
        convert = self.convert(x)
        out = convert + self.res(convert)
        return out

class SCConv(nn.Module):
    def __init__(self, planes, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(planes, planes, 3, 1, 1),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(planes, planes, 3, 1, 1),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(planes, planes, 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

class SCBottleneck(nn.Module):
    #expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, in_planes, planes):
        super(SCBottleneck, self).__init__()
        planes = int(planes / 2)

        self.conv1_a = nn.Conv2d(in_planes, planes, 1, 1)
        self.k1 = nn.Sequential(
                    nn.Conv2d(planes, planes, 3, 1, 1), 
                    nn.LeakyReLU(0.2),
                    )

        self.conv1_b = nn.Conv2d(in_planes, planes, 1, 1)
        
        self.scconv = SCConv(planes, self.pooling_r)

        self.conv3 = nn.Conv2d(planes * 2, planes * 2, 1, 1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_a = self.relu(out_a)

        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.relu(out_b)

        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))

        out += residual
        out = self.relu(out)

        return out

class Scale_attention(nn.Module):   
    def __init__(self):
        super(Scale_attention, self).__init__()
        self.scale_attention = nn.ModuleList()
        self.res_list = nn.ModuleList()
        self.channel = settings.channel
        if settings.scale_attention is True:
            for i in range(settings.num_scale_attention):
                self.scale_attention.append(
                    nn.Sequential(
                        nn.MaxPool2d(2 ** (i + 1), 2 ** (i + 1)),
                        nn.Conv2d(self.channel, self.channel, 1, 1),
                        nn.Sigmoid()
                    )
                )
        for i in range(settings.num_scale_attention):
            self.res_list.append(
                Residual_Block(self.channel, self.channel, 2)
            )

        self.conv11 = nn.Sequential(
            nn.Conv2d((settings.num_scale_attention + 1) * self.channel, self.channel, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.scn = SCBottleneck(self.channel, self.channel)

    def forward(self, x):
        b, c, h, w = x.size()
        temp = x
        out = []
        out.append(temp)
        if settings.scale_attention is True:
            for i in range(settings.num_scale_attention):
                temp = self.res_list[i](temp)
                b0,c0,h0,w0 = temp.size()
                temp = temp * F.upsample(self.scale_attention[i](x), [h0, w0])
                up = temp
                out.append(F.upsample(up, [h, w]))
            fusion = self.conv11(torch.cat(out, dim=1))

        else:
            for i in range(settings.num_scale_attention):
                temp = self.res_list[i](temp)
                up = temp
                out.append(F.upsample(up, [h, w]))
            fusion = self.conv11(torch.cat(out, dim=1))
        out = self.scn(fusion + x)
        return out

class DenseConnection(nn.Module):
    def __init__(self, unit, unit_num):
        super(DenseConnection, self).__init__()
        self.unit_num = unit_num
        self.channel = settings.channel
        self.units = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()
        for i in range(self.unit_num):
            self.units.append(unit())
            self.conv1x1.append(nn.Sequential(nn.Conv2d((i+2)*self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2)))
    
    def forward(self, x):
        cat = []
        cat.append(x)
        out = x
        for i in range(self.unit_num):
            tmp = self.units[i](out)
            cat.append(tmp)
            out = self.conv1x1[i](torch.cat(cat,dim=1))
        return out
    

class ODE_DerainNet(nn.Module):
    def __init__(self):
        super(ODE_DerainNet, self).__init__()
        self.channel = settings.channel
        self.unit_num = settings.unit_num
        self.enterBlock = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.derain_net = DenseConnection(Scale_attention, self.unit_num)
        self.exitBlock = nn.Sequential(nn.Conv2d(self.channel, 3, 3, 1, 1), nn.LeakyReLU(0.2))


    def forward(self, x):  
        image_feature = self.enterBlock(x)
        rain_feature = self.derain_net(image_feature)
        rain = self.exitBlock(rain_feature)
        derain = x - rain
        return derain
