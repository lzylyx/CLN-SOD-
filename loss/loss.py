#！/usr/bin/env python
#!-*coding:utf-8 -*-
#!@Author :lzy
#!@File :loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CrossCorrelationLoss(nn.Module):
    """Cross Correlation Loss"""
    def __init__(self, patch_size,reduction='mean',loss_weight=1.0):
        super(CrossCorrelationLoss, self).__init__()
        self.loss_weight = loss_weight
        self.num_classes = 1
        self.patch_size=patch_size
    
    # Cross Correlation map
    def _construct_cross_correlation_matrix(self, label, label_size):
        """Cross Correlation map"""
        if len(label.shape)==3:
            label = torch.unsqueeze(label, dim=1)
        scaled_labels = F.interpolate(label, size=label_size, mode="nearest")
        scaled_labels = scaled_labels.squeeze_(dim=1).long()
        scaled_labels[scaled_labels == 255] = self.num_classes
        one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1)
        one_hot_labels = one_hot_labels.view((one_hot_labels.shape[0],-1, self.num_classes + 1)).float()
        cross_correlation_matrix = torch.bmm(one_hot_labels,one_hot_labels.permute((0, 2, 1)))
        return cross_correlation_matrix
      

    def forward(self, cls_score, label):
        """forward function"""
        cross_correlation_matrix = self._construct_cross_correlation_matrix(label, label_size=self.patch_size)
        unary_term = F.binary_cross_entropy(cls_score, cross_correlation_matrix)
        diagonal_matrix = (1 - torch.eye(cross_correlation_matrix.shape[1])).to("cuda")
        vtarget = diagonal_matrix*cross_correlation_matrix
        recall_part = torch.sum(cls_score * vtarget, dim=2)
        denominator = torch.sum(vtarget, dim=2)
        denominator = denominator.masked_fill_(~(denominator>0),1)
        recall_part = recall_part.div_(denominator)
        recall_label = torch.ones_like(recall_part)
        recall_loss = F.binary_cross_entropy(recall_part, recall_label)
        spec_part = torch.sum((1 - cls_score)* (1 - cross_correlation_matrix), dim=2)
        denominator = torch.sum(1 - cross_correlation_matrix, dim=2)
        denominator = denominator.masked_fill_(~(denominator>0),1)
        spec_part = spec_part.div_(denominator)
        spec_label = torch.ones_like(spec_part)
        spec_loss = F.binary_cross_entropy(spec_part, spec_label)

        precision_part = torch.sum(cls_score*vtarget,dim=2)
        denominator = torch.sum(cls_score, dim=2)
        denominator = denominator.masked_fill_(~(denominator>0),1)
        precision_part = precision_part.div_(denominator)
        precision_label = torch.ones_like(precision_part)
        precision_loss = F.binary_cross_entropy(precision_part, precision_label)
        global_term = (recall_loss + spec_loss + precision_loss) / 64
        loss_cls = self.loss_weight*(unary_term + global_term)
        return loss_cls

class PerceptionLoss(nn.Module):
    """Perception Loss"""
    def __init__(self):
        super(PerceptionLoss, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True).cuda()
        self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False

    def forward(self, output, gt):
        """forward function"""
        output=output.expand(-1,3,-1,-1)
        gt=gt.expand(-1,3,-1,-1)
        vgg_output = self.vgg16_conv_4_3(output)
        with torch.no_grad():
            vgg_gt = self.vgg16_conv_4_3(gt.detach())
        loss = F.mse_loss(vgg_output, vgg_gt)
        return loss

class BinaryCrossEntropyLoss(nn.Module):
    """Binary Cross Entropy Loss"""
    def __init__(self):
        super(BinaryCrossEntropyLoss,self).__init__()
        self.bce_loss = nn.BCELoss(reduction='mean')
    def forward(self,prevs,labels):
        """forward function"""
        loss=self.bce_loss(prevs,labels)
        return loss
        
#
#binaryloss=BinaryCrossEntropyLoss()
#gt = torch.rand([2, 1,512, 512]).cuda()
#output = torch.rand([2, 1,512, 512]).cuda()
#loss=binaryloss(output,gt)
#print(loss)
#perceptionloss=PerceptionLoss()
#gt = torch.rand([2, 1,512, 512]).cuda()
#output = torch.rand([2, 1,512, 512]).cuda()
#loss=perceptionloss(output,gt)
#print(loss)
#crosscorrelationloss = CrossCorrelationLoss(patch_size=[64,64])
#pred = torch.rand([2, 4096, 4096]).cuda()
#la = torch.randint(0, 2, [2,1, 512, 512]).to("cuda").float()
#out = crosscorrelationloss(pred, la)
#print(out)
