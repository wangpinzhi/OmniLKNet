import torch.nn as nn
from models.replknet import create_RepLKNet31B
from models.conv3dnet import create_Conv3DNet
from models.preddispnet import create_PredDispNet



class OmniLKNet(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.first_step = create_RepLKNet31B() # to extract image features
        self.second_step = create_Conv3DNet(args.input_size, args.maxdisp, args.train) # to aggregate cost volume
        self.final_step = create_PredDispNet(args.maxdisp) # to compute disparity

    def forward(self, leftImg, rightImg):

        features_left = self.first_step(leftImg)
        features_right = self.first_step(rightImg)

        if self.training:
            fixed_cost_volume1, fixed_cost_volume2, fixed_cost_volume3 = self.second_step(features_left,features_right)
        else:
            fixed_cost_volume1 = self.second_step(features_left,features_right)

        pred1, pred2, pred3 = self.final_step(fixed_cost_volume1,fixed_cost_volume2,fixed_cost_volume3)

        return pred1, pred2, pred3


