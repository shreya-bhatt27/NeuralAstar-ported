from planner import NeuralAstar
import torch
import torch.nn as nn
import numpy as np
from data import get_hard_medium_easy_masks
from data import _sample_onehot

class combine_planner():
    def __init__(self,mechanism):
        self.model = NeuralAstar(mechanism)
        self.mechanism = mechanism
        self.dilate_gt = 0.5
    
    def forward(self, map_designs, start_maps, goal_maps):
        outputs = self.model.forward(map_designs, start_maps, goal_maps)
        return outputs

    def get_opt_trajs(start_maps, goal_maps, opt_policies, mechanism):

        opt_trajs = np.zeros_like(start_maps)
        opt_policies = opt_policies.transpose((0, 2, 3, 4, 1))

        for i in range(len(opt_trajs)):
            current_loc = tuple(np.array(np.nonzero(start_maps[i])).squeeze())
            goal_loc = tuple(np.array(np.nonzero(goal_maps[i])).squeeze())

            while goal_loc != current_loc:
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
        masks = np.concatenate(masks, axis=1).max(axis=1, keepdims=True)
        start_maps = _sample_onehot(masks)
        return start_maps