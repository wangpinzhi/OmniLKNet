import torch.nn as nn
import torch.nn.functional as F

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

  return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False), nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

  return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False), nn.BatchNorm3d(out_planes))

class hourglass(nn.Module):
  def __init__(self, inplanes):
    super(hourglass, self).__init__()

    self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1), nn.ReLU(inplace=True))

    self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

    self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1), nn.ReLU(inplace=True))

    self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1), nn.ReLU(inplace=True))

    self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(inplanes * 2))  #+conv2

    self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(inplanes))  #+x

  def forward(self, x, presqu, postsqu):

    out = self.conv1(x)  #in:1/4 out:1/8
    pre = self.conv2(out)  #in:1/8 out:1/8
    if postsqu is not None:
      pre = F.relu(pre + postsqu, inplace=True)
    else:
      pre = F.relu(pre, inplace=True)

    out = self.conv3(pre)  #in:1/8 out:1/16
    out = self.conv4(out)  #in:1/16 out:1/16

    if presqu is not None:
      post = F.relu(self.conv5(out) + presqu, inplace=True)  #in:1/16 out:1/8
    else:
      post = F.relu(self.conv5(out) + pre, inplace=True)

    out = self.conv6(post)  #in:1/8 out:1/4

    return out, pre, post