import torch
from torch import nn
from torchvision import models
import copy

feature_extractor = models.efficientnet_b0(pretrained=True)

class TwoInToolsOut(nn.Module):
    '''
    TwoInToolsOut gets top and side frames and predicts
    the tools of the left and right hand.
    '''
    def __init__(self, num_classes, dropout_p=0.3):
        super().__init__()
        feature_extractor.classifier[-1] = nn.Identity()
        features_dim = 1280
        self.feature_extractor_top = copy.deepcopy(feature_extractor)
        self.feature_extractor_side = copy.deepcopy(feature_extractor)
        self.dropout = nn.Dropout(dropout_p)
        self.output_layer_left = nn.Linear(features_dim * 2, num_classes)
        self.output_layer_right = nn.Linear(features_dim * 2, num_classes)

    def forward(self, top_x, side_x):
        top_x = self.feature_extractor_top(top_x)
        side_x = self.feature_extractor_side(side_x)
        x = torch.cat((top_x, side_x), dim=1)
        x = self.dropout(x)
        out_right = self.output_layer_right(x)
        out_left = self.output_layer_left(x)
        return out_right, out_left, x