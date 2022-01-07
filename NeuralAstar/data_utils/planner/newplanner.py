from data_utils.planner.planner import NeuralAstar
import torch
import torch.nn as nn
import numpy as np
from data_utils.utils.data import get_hard_medium_easy_masks
from data_utils.utils.data import _sample_onehot
from data_utils.astar.differentiable_astar import DifferentiableAstar

class combine_planner():
    def __init__(self,mechanism):
        self.model = NeuralAstar(mechanism)
        self.mechanism = mechanism
        self.dilate_gt = 0.5
        self.astar_ref = DifferentiableAstar(self.mechanism, g_ratio=0.5, Tmax=1)
    
    def forward(self, map_designs, start_maps, goal_maps):
        outputs = self.model.forward(map_designs, start_maps, goal_maps)
        return outputs

    def get_opt_trajs(self, start_maps, goal_maps, opt_policies, mechanism):

        opt_trajs = torch.zeros_like(start_maps)
        opt_policies = opt_policies.permute(0, 2, 3, 4, 1)
        opt_policies = opt_policies.cpu().numpy()

        for i in range(len(opt_trajs)):
            current_loc = tuple(np.array(np.nonzero(start_maps[i].cpu().numpy())).squeeze())
            goal_loc = tuple(np.array(np.nonzero(goal_maps[i].cpu().numpy())).squeeze())

            while goal_loc != current_loc:
                opt_trajs[i][current_loc] = 1.0
                next_loc = mechanism.next_loc(current_loc,
                                            opt_policies[i][current_loc])
                assert (
                    opt_trajs[i][next_loc] == 0.0
                ), "Revisiting the same position while following the optimal policy"
                current_loc = next_loc

            opt_trajs[i][current_loc] = 1.0
        #opt_trajs = torch.from_numpy(opt_trajs)
        return opt_trajs

    def create_start_maps(self, opt_dists):
        masks = get_hard_medium_easy_masks(opt_dists.cpu().numpy(), reduce_dim=True)
        masks = np.concatenate(masks, axis=1).max(axis=1, keepdims=True)
        start_maps = _sample_onehot(masks)
        start_maps = torch.from_numpy(start_maps)
        return start_maps.cuda()