import torch
import numpy as np
import torch.nn as nn
from data_utils.utils.mechanism import Mechanism
from data_utils.utils._il_utils import backtrack, _st_softmax_noexp, expand
import math

class DifferentiableAstar(nn.Module):
    """
    Implementation based on https://rosettacode.org/wiki/A*_search_algorithm
    """
    def __init__(self, 
                 mechanism,
                 g_ratio,
                 Tmax,
                 detach_g=True,
                 verbose=False):

        super().__init__()

        neighbor_filter = mechanism.get_neighbor_filter()
        self.neighbor_filter = nn.Parameter(neighbor_filter,
                                            requires_grad=False)
        self.get_heuristic = mechanism.get_heuristic

        self.g_ratio = g_ratio
        assert (Tmax > 0) & (Tmax <= 1), "Tmax must be within (0, 1]"
        self.Tmax = Tmax
        self.detach_g = detach_g
        self.verbose = verbose

    def forward(self, cost_maps, start_maps, goal_maps, obstacles_maps):
        assert cost_maps.ndim == 4
        assert start_maps.ndim == 4
        assert goal_maps.ndim == 4
        assert obstacles_maps.ndim == 4

        cost_maps = cost_maps[:, 0]
        start_maps = start_maps[:, 0]
        goal_maps = goal_maps[:, 0]
        obstacles_maps = obstacles_maps[:, 0]

        num_samples = start_maps.shape[0]
        neighbor_filter = self.neighbor_filter
        neighbor_filter = torch.repeat_interleave(neighbor_filter, num_samples,
                                                  0)
        size = start_maps.shape[-1]

        open_maps = start_maps
        histories = torch.zeros_like(start_maps)

        h = self.get_heuristic(goal_maps)
        h = h + cost_maps
        g = torch.zeros_like(start_maps)

        parents = (
            torch.ones_like(start_maps).reshape(num_samples, -1) *
            goal_maps.reshape(num_samples, -1).max(-1, keepdim=True)[-1])

        size = cost_maps.shape[-1]
        Tmax = self.Tmax if self.training else 1.
        Tmax = int(Tmax * size * size)
        for t in range(Tmax):

            # select the node that minimizes cost
            f = self.g_ratio * g + (1 - self.g_ratio) * h
            f_exp = torch.exp(-1 * f / math.sqrt(cost_maps.shape[-1]))
            f_exp = f_exp * open_maps
            selected_node_maps = _st_softmax_noexp(f_exp)

            # break if arriving at the goal
            dist_to_goal = (selected_node_maps * goal_maps).sum((1, 2),
                                                                keepdim=True)
            is_unsolved = (dist_to_goal < 1e-8).float()
            if torch.all(is_unsolved == 0):
                if self.verbose:
                    print("All problems solved at", t)
                break

            histories = histories + selected_node_maps
            histories = torch.clamp(histories, 0, 1)
            open_maps = open_maps.float() - is_unsolved * selected_node_maps.float()
            open_maps = torch.clamp(open_maps, 0, 1)

            # open neighboring nodes, add them to the openlist if they satisfy certain requirements
            neighbor_nodes = expand(selected_node_maps, neighbor_filter)
            neighbor_nodes = neighbor_nodes * obstacles_maps

            # update g if one of the following conditions is met
            # 1) neighbor is not in the close list (1 - histories) nor in the open list (1 - open_maps)
            # 2) neighbor is in the open list but g < g2
            g2 = expand((g + cost_maps) * selected_node_maps, neighbor_filter)
            idx = (1 - open_maps) * (1 - histories) + open_maps * (g > g2)
            idx = idx * neighbor_nodes
            idx = idx.detach()
            g = g2 * idx + g * (1 - idx)
            if self.detach_g:
                g = g.detach()

            # update open maps
            open_maps = torch.clamp(open_maps + idx, 0, 1)
            open_maps = open_maps.detach()

            # for backtracking
            idx = idx.reshape(num_samples, -1)
            snm = selected_node_maps.reshape(num_samples, -1)
            new_parents = snm.max(-1, keepdim=True)[1]
            parents = new_parents * idx + parents * (1 - idx)

        # backtracking
        path_maps = backtrack(start_maps, goal_maps, parents, t)
        return histories.unsqueeze(1), path_maps.unsqueeze(1)
