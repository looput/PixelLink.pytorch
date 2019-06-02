import torch
from torchvision import models
import torch.nn as nn

__all__=['vgg16']

def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv1x1_sigmoid(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.Sigmoid(),
    )


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class VGG(nn.Module):
    def __init__(self,pretrained):
        super(VGG, self).__init__()
        self.backbone = models.vgg16_bn(pretrained=pretrained).features
        self.s1 = self.backbone[0:7]
        self.s2 = self.backbone[7:14]
        self.s3 = self.backbone[14:24]
        self.s4 = self.backbone[24:34]
        self.s5 = self.backbone[34:]
        self.s6 = nn.Sequential(
            conv3x3_bn_relu(512, 512),
            conv3x3_bn_relu(512, 512),
            conv3x3_bn_relu(512, 512))

    def forward(self, imgs):
        s1 = self.s1(imgs)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)
        s5 = self.s5(s4)
        s6 = self.s6(s5)
        return s1, s2, s3, s4, s5, s6


class vgg_pixel(nn.Module):
    '''
    Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace)
        (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (9): ReLU(inplace)
        (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (12): ReLU(inplace)
        (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (16): ReLU(inplace)
        (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (19): ReLU(inplace)
        (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (22): ReLU(inplace)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (26): ReLU(inplace)
        (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (29): ReLU(inplace)
        (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (32): ReLU(inplace)
        (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (36): ReLU(inplace)
        (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (39): ReLU(inplace)
        (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (42): ReLU(inplace)
        (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
    '''
    def __init__(self,pretrained):
        super(vgg_pixel, self).__init__()
        self.backbone = models.vgg16_bn(pretrained=pretrained).features
        self.c1 = self.backbone[0:6]
        self.c2 = self.backbone[6:13]
        self.c3 = self.backbone[13:23]
        self.c4 = self.backbone[23:33]
        self.c5 = self.backbone[33:43]
        self.fc = nn.Sequential(
            conv3x3_bn_relu(512, 512),
            conv3x3_bn_relu(512, 512))

    def forward(self, imgs):
        c1 = self.c1(imgs)
        c2 = self.c2(c1)    # /2
        c3 = self.c3(c2)    # /2
        c4 = self.c4(c3)    # /2
        c5 = self.c5(c4)    # /2
        fc = self.fc(c5)    # /1
        return c1,c2,c3,c4,c5,fc

class vgg16(nn.Module):
    def __init__(self,pretrained=False,num_classes=None):
        super(vgg16, self).__init__()
        self.backbone = vgg_pixel(pretrained)

        self.cls_conv_6=conv1x1(512,2)
        self.cls_conv_5=conv1x1(512,2)
        self.cls_conv_4=conv1x1(512,2)
        self.cls_conv_3=conv1x1(256,2)
        self.cls_conv_2=conv1x1(128,2)

        self.link_conv_6=conv1x1(512,16)
        self.link_conv_5=conv1x1(512,16)
        self.link_conv_4=conv1x1(512,16)
        self.link_conv_3=conv1x1(256,16)
        self.link_conv_2=conv1x1(128,16)

        self.upsample=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, imgs):
        _,s2, s3, s4, s5, s6 = self.backbone(imgs)

        score_5=self.cls_conv_6(s6)+self.cls_conv_5(s5)
        score_4=self.cls_conv_4(s4)+self.upsample(score_5)
        score_3=self.cls_conv_3(s3)+self.upsample(score_4)
        score_2=self.cls_conv_2(s2)+self.upsample(score_3)

        link_5=self.link_conv_6(s6)+self.link_conv_5(s5)
        link_4=self.link_conv_4(s4)+self.upsample(link_5)
        link_3=self.link_conv_3(s3)+self.upsample(link_4)
        link_2=self.link_conv_2(s2)+self.upsample(link_3)

        return score_2,link_2
