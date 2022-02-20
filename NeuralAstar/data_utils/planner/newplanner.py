import torch
import torch.nn as nn
import segmentation_models_pytorch as smp   #install package
from data_utils.astar.differentiable_astar import DifferentiableAstar
from data_utils.utils._il_utils import sample_onehot
from data_utils.utils._il_utils import get_hard_medium_easy_masks
from data_utils.utils._il_utils import _sample_onehot


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
            aux_params= {"classes" : 1, "dropout" : 0.5}
        )

    def forward(self, x, map_designs):
        print("x before is = ", x)
        x  = torch.stack(tuple(x), dim= 0)
        a = self.model(x)
        y = torch.sigmoid(a[0])
        if map_designs is not None:
            y = y * map_designs + torch.ones_like(y) * (1 - map_designs)
        return y
    
class NeuralAstar(nn.Module):
    def __init__(
        self,
        mechanism,
        g_ratio, 
        encoder_backbone, 
        dilate_gt, 
        encoder_input,
    ):
        super().__init__()
        self.mechanism = mechanism
        self.encoder_input = encoder_input
        self.encoder_arch = 'Unet'
        self.encoder_backbone= encoder_backbone
        self.encoder_depth = 4
        self.ignore_obstacles = True
        self.learn_obstacles = False
        self.g_ratio = g_ratio
        self.Tmax = 0.25
        self.detach_g = True
        self.astar = DifferentiableAstar(
            mechanism=self.mechanism,
            g_ratio=self.g_ratio,
            Tmax=self.Tmax,
            detach_g=self.detach_g,
        )
        self.encoder = Unet(len(self.encoder_input), self.encoder_backbone, self.encoder_depth)

    def forward(self, map_designs, start_maps, goal_maps, device):
        inputs = map_designs
        if "+" in self.encoder_input:
            inputs = torch.cat((inputs.float(), start_maps.float() + goal_maps.float()), dim=1)
        pred_cost_maps = self.encoder(
            inputs, map_designs.float() if not self.ignore_obstacles else None)
        obstacles_maps = map_designs.float() if not self.learn_obstacles else torch.ones_like(
            map_designs)

        histories, paths = self.astar(pred_cost_maps, start_maps, goal_maps,
                                      obstacles_maps, device)

        return histories, paths, pred_cost_maps

class combine_planner():
    def __init__(self,mechanism, g_ratio, encoder_backbone, dilate_gt, encoder_input):
        self.dilate_gt = dilate_gt
        self.g_ratio = g_ratio
        self.encoder_backbone = encoder_backbone
        self.encoder_input = encoder_input
        self.model = NeuralAstar(mechanism, self.g_ratio, self.encoder_backbone, self.dilate_gt, self.encoder_input)
        self.mechanism = mechanism
        self.astar_ref = DifferentiableAstar(self.mechanism, g_ratio=0.5, Tmax=1)
    
    def forward(self, map_designs, start_maps, goal_maps, device):
        outputs = self.model.forward(map_designs, start_maps, goal_maps, device)
        return outputs

    def get_opt_trajs(self, start_maps, goal_maps, opt_policies, mechanism, device):

        opt_trajs = torch.zeros_like(start_maps)
        opt_policies = opt_policies.permute(0, 2, 3, 4, 1)

        for i in range(len(opt_trajs)):
            current_loc = torch.tensor(torch.nonzero(start_maps[i].squeeze()))
            goal_loc = torch.tensor(torch.nonzero(goal_maps[i].squeeze()))
            goal_loc = goal_loc.squeeze()

            while not (torch.equal(current_loc, goal_loc)):
                current_loc = current_loc.squeeze()
                opt_trajs[i][0][current_loc[0]][current_loc[1]] = 1.0
                next_loc = mechanism.next_loc(current_loc,
                                            opt_policies[i][0][current_loc[0]][current_loc[1]], device)
                next_loc = next_loc.squeeze()

                assert (
                    opt_trajs[i][0][next_loc[0]][next_loc[1]].float() == 0.0
                ), "Revisiting the same position while following the optimal policy"
                current_loc = next_loc

            opt_trajs[i][0][current_loc[0]][current_loc[1]] = 1.0
        return opt_trajs

    def create_start_maps(self, opt_dists, device):
        masks = get_hard_medium_easy_masks(opt_dists, device, reduce_dim=True)
        (masks, indices) = torch.cat(masks, axis=1).max(axis=1, keepdim=True)
        start_maps = _sample_onehot(masks, device)
        return start_maps




