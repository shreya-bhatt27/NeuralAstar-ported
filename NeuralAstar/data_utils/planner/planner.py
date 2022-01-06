from data_utils.astar.differentiable_astar import DifferentiableAstar
import torch
import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp   #install package

class Unet(nn.Module):

    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    def __init__(self, input_dim, encoder_backbone, encoder_depth):
        super().__init__()
        decoder_channels = self.DECODER_CHANNELS[:encoder_depth]
        self.model = smp.Unet(
            encoder_name=encoder_backbone,
            encoder_weights=None,
            classes=1,
            in_channels=input_dim,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )

    def forward(self, x, map_designs):
        y = torch.sigmoid(self.model(x))
        if map_designs is not None:
            y = y * map_designs + torch.ones_like(y) * (1 - map_designs)
        return y

class NeuralAstar(nn.Module):
    def __init__(
        self,
        mechanism,
    ):
        super().__init__()
        self.mechanism = mechanism
        self.encoder_input = 'm+'
        self.encoder_arch = 'Unet'
        self.encoder_backbone='vgg16_bn'
        self.encoder_depth = 5
        self.ignore_obstacles = True
        self.learn_obstacles = False
        self.g_ratio = 0.5
        self.Tmax = 0.25
        self.detach_g = True
        self.astar = DifferentiableAstar(
            mechanism=self.mechanism,
            g_ratio=self.g_ratio,
            Tmax=self.Tmax,
            detach_g=self.detach_g,
        )
        self.encoder = Unet(len(self.encoder_input), self.encoder_backbone, self.encoder_depth)

    def forward(self, map_designs, start_maps, goal_maps):
        inputs = map_designs
        if "+" in self.encoder_input:
            inputs = torch.cat((inputs, start_maps + goal_maps), dim=1)
        pred_cost_maps = self.encoder(
            inputs, map_designs if not self.ignore_obstacles else None)
        obstacles_maps = map_designs if not self.learn_obstacles else torch.ones_like(
            map_designs)

        histories, paths = self.astar(pred_cost_maps, start_maps, goal_maps,
                                      obstacles_maps)

        return histories, paths, pred_cost_maps
