#！/usr/bin/env python
#!-*coding:utf-8 -*-
#!@Author :lzy
#!@File :net_head.py
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import backbone
from torchsummary import summary


def upsample_function(src,tar):
    """upsample function"""   
    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear',align_corners=True)
    return src

class SeparationModule(nn.Module):
    """Separation Module"""
    def __init__(self,in_channels,out_channels,kernel_size):
        super(SeparationModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = kernel_size // 2
        self.reduce_conv = torch.nn.Sequential(
            nn.Conv2d(self.in_channels,self.out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True)
            )
        self.t1 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=(kernel_size, 1),padding=(self.padding, 0),groups=self.out_channels)
        self.t2 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=(1, kernel_size),padding=(0, self.padding),groups=self.out_channels)
        self.p1 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=(1, kernel_size),padding=(0, self.padding),groups=self.out_channels)
        self.p2 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=(kernel_size, 1),padding=(self.padding, 0),groups=self.out_channels)
        self.norm=nn.BatchNorm2d(self.out_channels)
        self.relu=nn.ReLU() 

    def forward(self, x):
        """Forward function."""
        x = self.reduce_conv(x)
        x1 = self.t1(x)
        x1 = self.t2(x1)
        x2 = self.p1(x)
        x2 = self.p2(x2)
        out = self.relu(self.norm(x1 + x2))
        return out



class CALNet_Resnet(nn.Module):
   """CLN-SOD_Resnet"""
   def __init__(self,input_channels,inter_channels,prior_size,am_kernel_size,groups=1,):
        super(CALNet_Resnet, self).__init__()
        self.in_channels=input_channels
        self.out_channels=1
        self.backbone=ResNet()
        self.inter_channels = inter_channels
        self.prior_double_size = prior_size*prior_size
        self.am_kernel_size = am_kernel_size
        self.separation = SeparationModule(self.in_channels, self.inter_channels,self.am_kernel_size)
        self.prior_conv = nn.Sequential(
            nn.Conv2d(self.inter_channels,self.prior_double_size,1,padding=0,stride=1,groups=groups),
            nn.BatchNorm2d(self.prior_double_size))
        self.intra_conv =nn.Sequential(
        nn.Conv2d(self.inter_channels,self.inter_channels,1,padding=0,stride=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True))
        self.inter_conv = nn.Sequential(
            nn.Conv2d(self.inter_channels,self.inter_channels,1,padding=0,stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(True))

        self.bottleneck_1 =nn.Sequential(
        nn.Conv2d(self.in_channels + self.inter_channels * 2,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.out_channels,3,padding=1),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(True),
        )
        self.bottleneck_2 =nn.Sequential(
        nn.Conv2d(self.in_channels + self.inter_channels * 3,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.out_channels,3,padding=1),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(True)
        )
        self.bottleneck_3 =nn.Sequential(
        nn.Conv2d(self.in_channels + self.inter_channels * 3+128,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.out_channels,3,padding=1),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(True)
        )
        self.outconv=nn.Sequential(
            nn.Conv2d(self.out_channels*3,self.out_channels,1),
            nn.Sigmoid()
            )

   def forward(self, inputs):
        """Forward function."""
        H,W=inputs.shape[2],inputs.shape[3]
        features_1,features_2,features_3 = self.backbone(inputs)
        batch_size, channels, height, width = features_3.size()
        assert self.prior_double_size == height * width
        value = self.separation(features_3)
        context_aware_map = self.prior_conv(value)
        context_aware_map = context_aware_map.view(batch_size,self.prior_double_size,-1)
        context_aware_map = context_aware_map.permute(0, 2, 1)
        context_aware_map = torch.sigmoid(context_aware_map)
        inter_context_aware_map = 1 - context_aware_map
        value = value.view(batch_size, self.inter_channels, -1)
        value = value.permute(0, 2, 1)
        #intra_context_precise
        intra_context = torch.bmm(context_aware_map, value)
        intra_context = intra_context.div(self.prior_double_size)
        intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(batch_size, self.inter_channels,height,width)
        intra_context = self.intra_conv(intra_context)
        #inter_context_recall 
        inter_context = torch.bmm(inter_context_aware_map, value)
        inter_context = inter_context.div(self.prior_double_size)
        inter_context = inter_context.permute(0, 2, 1).contiguous()
        inter_context = inter_context.view(batch_size, self.inter_channels,height,width)
        inter_context = self.inter_conv(inter_context)
        cal_out1 = torch.cat([features_3, intra_context, inter_context], dim=1) 
        cal_up1 = upsample_function(cal_out1,features2)
        cal_out2 = torch.cat([features_2, cal_up1], dim=1)  
        cal_up2 = upsample_function(cal_out2,features1)     
        cal_out3 = torch.cat([features_1, cal_up2], dim=1)  
        output1 = self.bottleneck1(cal_out1)
        output1 = upsample_function(output1,features1)
        output2 = self.bottleneck2(cal_out2)
        output2 = upsample_function(output2,features1)
        output3 = self.bottleneck3(cal_out3)
        output=self.outconv(torch.cat([output1,output2,output3],dim=1))
        if self.training:
            return output, context_aware_map
        else:
            return output


class CALNet_Vgg(nn.Module):
   """CLN-SOD_Vgg"""
   def __init__(self,input_channels,inter_channels,prior_size,am_kernel_size,groups=1,):
        super(CALNet_Vgg, self).__init__()
        self.in_channels=input_channels
        self.out_channels=1
        self.backbone=VGG()
        self.inter_channels = inter_channels
        self.prior_double_size = prior_size*prior_size
        self.am_kernel_size = am_kernel_size
        self.separation = SeparationModule(self.in_channels, self.inter_channels,self.am_kernel_size)
        self.prior_conv = nn.Sequential(
            nn.Conv2d(self.inter_channels,self.prior_double_size,1,padding=0,stride=1,groups=groups),
            nn.BatchNorm2d(self.prior_double_size))
        self.intra_conv =nn.Sequential(
        nn.Conv2d(self.inter_channels,self.inter_channels,1,padding=0,stride=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True))
        self.inter_conv = nn.Sequential(
            nn.Conv2d(self.inter_channels,self.inter_channels,1,padding=0,stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(True))
        self.bottleneck =nn.Sequential(
        nn.Conv2d(self.in_channels + self.inter_channels * 2,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.out_channels,3,padding=1),
        nn.Sigmoid()
        )

   def forward(self, inputs):
        """Forward function."""
        H,W=inputs.shape[2],inputs.shape[3]
        features = self.backbone(inputs)
        batch_size, channels, height, width = features.size()
        assert self.prior_double_size == height * width
        value = self.separation(features)
        context_aware_map = self.prior_conv(value)
        context_aware_map = context_aware_map.view(batch_size,self.prior_double_size,-1)
        context_aware_map = context_aware_map.permute(0, 2, 1)
        context_aware_map = torch.sigmoid(context_aware_map)
        inter_context_aware_map = 1 - context_aware_map
        value = value.view(batch_size, self.inter_channels, -1)
        value = value.permute(0, 2, 1)
        #intra_context_precise
        intra_context = torch.bmm(context_aware_map, value)
        intra_context = intra_context.div(self.prior_double_size)
        intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(batch_size, self.inter_channels,height,width)
        intra_context = self.intra_conv(intra_context)
        #inter_context_recall 
        inter_context = torch.bmm(inter_context_aware_map, value)
        inter_context = inter_context.div(self.prior_double_size)
        inter_context = inter_context.permute(0, 2, 1).contiguous()
        inter_context = inter_context.view(batch_size, self.inter_channels,height,width)
        inter_context = self.inter_conv(inter_context)
        cp_outs = torch.cat([features, intra_context, inter_context], dim=1)
        output = self.bottleneck(cp_outs)
        output=F.interpolate(output,(H,W),mode='bilinear',align_corners=True)
        if self.training:
            return output, context_aware_map
        else:
            return output


class CALNet_U2net(nn.Module):
   """CALNet_U2net"""
   def __init__(self,input_channels,inter_channels,prior_size,am_kernel_size,groups=1,):
        
        super(CALNet_U2net, self).__init__()
        self.in_channels=input_channels
        self.out_channels=1
        self.backbone=U2NETP()
        self.inter_channels = inter_channels
        self.prior_double_size = prior_size*prior_size
        self.am_kernel_size = am_kernel_size
        self.separation = SeparationModule(self.in_channels, self.inter_channels,self.am_kernel_size)
        self.prior_conv = nn.Sequential(
            nn.Conv2d(self.inter_channels,self.prior_double_size,1,padding=0,stride=1,groups=groups),
            nn.BatchNorm2d(self.prior_double_size))
        self.intra_conv =nn.Sequential(
        nn.Conv2d(self.inter_channels,self.inter_channels,1,padding=0,stride=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True))
        self.inter_conv = nn.Sequential(
            nn.Conv2d(self.inter_channels,self.inter_channels,1,padding=0,stride=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(True))
        self.bottleneck_1 =nn.Sequential(
        nn.Conv2d(self.in_channels + self.inter_channels * 2,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.out_channels,3,padding=1),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(True),
        )
        self.bottleneck_2 =nn.Sequential(
        nn.Conv2d(self.in_channels*2 + self.inter_channels * 2,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.out_channels,3,padding=1),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(True)
        )
        self.bottleneck_3 =nn.Sequential(
        nn.Conv2d(self.in_channels*3 + self.inter_channels * 2,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.out_channels,3,padding=1),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(True)
        )
        self.bottleneck_4 =nn.Sequential(
        nn.Conv2d(self.in_channels*4 + self.inter_channels * 2,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.inter_channels,3,padding=1),
        nn.BatchNorm2d(self.inter_channels),
        nn.ReLU(True),
        nn.Conv2d(self.inter_channels,self.out_channels,3,padding=1),
        nn.BatchNorm2d(self.out_channels),
        nn.ReLU(True),
        )
        self.outconv=nn.Sequential(
            nn.Conv2d(self.out_channels*4,self.out_channels,1),
            nn.Sigmoid()
            )

   def forward(self, inputs):
        """Forward function."""
        H,W=inputs.shape[2],inputs.shape[3]
        features_0,features_1,features_2,features_3 = self.backbone(inputs)
        batch_size, channels, height, width = features_3.size()
        assert self.prior_double_size == height * width
        value = self.separation(features_3)
        context_aware_map = self.prior_conv(value)
        context_aware_map = context_aware_map.view(batch_size,self.prior_double_size,-1)
        context_aware_map = context_aware_map.permute(0, 2, 1)
        context_aware_map = torch.sigmoid(context_aware_map)
        inter_context_aware_map = 1 - context_aware_map
        value = value.view(batch_size, self.inter_channels, -1)
        value = value.permute(0, 2, 1)
        #intra_context_precise
        intra_context = torch.bmm(context_aware_map, value)
        intra_context = intra_context.div(self.prior_double_size)
        intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(batch_size, self.inter_channels,height,width)
        intra_context = self.intra_conv(intra_context)
        #intra_context_recall
        inter_context = torch.bmm(inter_context_aware_map, value)
        inter_context = inter_context.div(self.prior_double_size)
        inter_context = inter_context.permute(0, 2, 1).contiguous()
        inter_context = inter_context.view(batch_size, self.inter_channels,height,width)
        inter_context = self.inter_conv(inter_context)
        cal_out1 = torch.cat([features_3, intra_context, inter_context], dim=1) 
        cal_up1 = upsample_function(cal_out1,features2)
        cal_out2 = torch.cat([features_2, cal_up1], dim=1) 
        cal_up2 = upsample_function(cal_out2,features1)     
        cal_out3 = torch.cat([features_1, cal_up2], dim=1)  
        cal_up3 =upsample_function(cal_out3,features0)
        cal_out4= torch.cat([features_0,cal_up3],dim=1)  
        output1 = self.bottleneck1(cal_out1)
        output1 = upsample_function(output1,features0)
        output2 = self.bottleneck2(cal_out2)
        output2 = upsample_function(output2,features0)
        output3 = self.bottleneck3(cal_out3)
        output3 = upsample_function(output3,features0)
        output4 = self.bottleneck4(cal_out4)
        output=self.outconv(torch.cat([output1,output2,output3,output4],dim=1))
        
        if self.training:
            return output, context_aware_map
        else:
            return output

   

if __name__=='__main__':

    #backbonepath = None
    model =  CALNet_U2net(input_channels=64,inter_channels=2,prior_size=64,am_kernel_size=5).cuda()  #U2NETP
    #model = CALNet_Resnet(input_channels=1024,inter_channels=256,prior_size=64,am_kernel_size=5).cuda()  #ResNet
    #model = CALNet_Vgg(input_channels=512,inter_channels=256,prior_size=64,am_kernel_size=5).cuda()  #VGG
    #print(model.training)
    #model.train(True)
    output,c_p_m=model(torch.rand(1,3,512,512).cuda())
    print(output.shape)
    print(c_p_m.shape)
    #model.eval()
    #print(model.training)
    #with torch.no_grad():
        #output=model(torch.rand(2,3,256,256).cuda())
        #print(output.shape)
    #info=summary(model,input_size=[(3,512,512)],batch_size=-1,device='cpu')
    #print(info)







    
