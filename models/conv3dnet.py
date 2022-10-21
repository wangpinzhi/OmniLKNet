from random import triangular
import torch.nn as nn
import math
from models.submodule import hourglass, convbn_3d
from torch.autograd import Variable
import torch
import torch.nn.functional as F


class Conv3DNet(nn.Module):
    def __init__(self, origin_img_h, origin_img_w, maxdisp, training=False):
        super(Conv3DNet, self).__init__()
        print("Conv3DNet matching network!")
        self.maxdisp = maxdisp
        self.training = training
        self.sample_height = origin_img_h # origin image height, so we can recover the resolution
        self.sample_width = origin_img_w

        self.dres0 = nn.Sequential(convbn_3d(2048, 32, 3, 1, 1), nn.ReLU(
            inplace=True), convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(
            inplace=True), convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(
            inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(
            inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(
            inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * \
                    m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left_features, right_features):
        refimg_fea = left_features
        targetimg_fea = right_features

        # matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[
                        1] * 2, self.maxdisp // 32, refimg_fea.size()[2], refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp // 32):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.size()[1]:, i, :,
                     i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.interpolate(cost1, [self.maxdisp, self.sample_height, self.sample_width], mode='trilinear', align_corners=True)
            cost2 = F.interpolate(cost2, [self.maxdisp, self.sample_height, self.sample_width], mode='trilinear', align_corners=True)

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)

        cost3 = F.interpolate(cost3, [self.maxdisp, self.sample_height, self.sample_width], mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3


def create_Conv3DNet(input_size,maxdisp,training):
    return Conv3DNet(input_size[1],input_size[2],maxdisp,training)