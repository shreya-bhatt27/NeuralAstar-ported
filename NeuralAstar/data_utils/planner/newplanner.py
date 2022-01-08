from data_utils.planner.planner import NeuralAstar
import torch
import torch.nn as nn
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

        for i in range(len(opt_trajs)):
            current_loc = torch.tensor(torch.nonzero(start_maps[i].squeeze()))
            goal_loc = torch.tensor(torch.nonzero(goal_maps[i].squeeze()))

            while torch.equal(current_loc, goal_loc):
                opt_trajs[i][current_loc] = 1.0
                next_loc = mechanism.next_loc(current_loc,
                                            opt_policies[i][current_loc])
                assert (
                    opt_trajs[i][next_loc] == 0.0
                ), "Revisiting the same position while following the optimal policy"
                current_loc = next_loc

            opt_trajs[i][current_loc] = 1.0
        return opt_trajs

    def create_start_maps(self, opt_dists):
        masks = get_hard_medium_easy_masks(opt_dists, reduce_dim=True)
        (masks, indices) = torch.concat(masks, axis=1).max(axis=1, keepdims=True)
        start_maps = _sample_onehot(masks)
        return start_maps