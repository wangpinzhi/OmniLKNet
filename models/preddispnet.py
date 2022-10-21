import torch.nn as nn
import torch
import numpy as np

class disparityregression(nn.Module):
  def __init__(self, maxdisp):
    super(disparityregression, self).__init__()
    self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda()

  def forward(self, x):
    out = torch.sum(x * self.disp.data, 1, keepdim=True)
    return out

class PredDispNet(nn.Module):
    def __init__(self, maxdisp, strategy='softmax'):
        super(PredDispNet, self).__init__()
        self.strategy = strategy
        self.maxdisp = maxdisp

    def forward(self, cost_volume1, cost_volume2=None, cost_volume3=None): # Cost Volume Size: BxDxHxW
      pred1 = disparityregression(self.maxdisp)(cost_volume1)
      pred2 = None
      pred3 = None
      if cost_volume2 is not None:
        pred2 = disparityregression(self.maxdisp)(cost_volume2)
      if cost_volume3 is not None:
        pred3 = disparityregression(self.maxdisp)(cost_volume3)
      return pred1, pred2, pred3

def create_PredDispNet(maxdisp):
  return PredDispNet(maxdisp)
      
