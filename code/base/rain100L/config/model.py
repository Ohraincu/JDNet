import torch
from torch import nn
import torch.nn.functional as F
import settings
from itertools import combinations,product
import math
from function.modules import Subtraction, Subtraction2, Aggregation

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

def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc

class SAM(nn.Module):
    def __init__(self, in_planes, rel_planes, out_planes, share_planes, sa_type=0, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        if sa_type == 0:
            self.conv_w = nn.Sequential(nn.LeakyReLU(0.2),
                                        nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                        nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1))
            self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
            self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.softmax = nn.Softmax(dim=-2)
        else:
            self.conv_w = nn.Sequential(nn.LeakyReLU(0.2),
                                        nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
                                        nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
            self.unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
            self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)

    def forward(self, x):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        if self.sa_type == 0:  # pairwise
            p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
            w = self.softmax(self.conv_w(torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(x.shape[0], -1, 1, x.shape[2]*x.shape[3])
            x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, 1, x1.shape[-1])
            w = self.conv_w(torch.cat([x1, x2], 1)).view(x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
        x = self.aggregation(x3, w)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_planes, rel_planes, mid_planes, out_planes, sa_type=0, share_planes=8, kernel_size=7, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.sam = SAM(in_planes, rel_planes, mid_planes, share_planes, sa_type, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.LeakyReLU(0.2)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out += identity
        return out

class Scale_attention(nn.Module):   
    def __init__(self):
        super(Scale_attention, self).__init__()
        self.scale_attention = nn.ModuleList()
        self.res_list = nn.ModuleList()
        self.channel = settings.channel
        self.san = Bottleneck(self.channel, self.channel // 16, self.channel // 4, self.channel)
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
        x = self.san(x)
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
