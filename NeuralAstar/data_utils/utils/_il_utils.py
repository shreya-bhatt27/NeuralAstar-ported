from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
TEST_RANDOM_SEED = 2020
NUM_POINTS_PER_MAP = 5

def _st_softmax_noexp(val):
    val_ = val.reshape(val.shape[0], -1)
    y = val_ / (val_.sum(dim=-1, keepdim=True))
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y)
    y_hard[range(len(y_hard)), ind] = 1
    y_hard = y_hard.reshape_as(val)
    y = y.reshape_as(val)
    return (y_hard - y).detach() + y

def expand(x, neighbor_filter, padding=1):
    x = x.unsqueeze(0)
    num_samples = x.shape[1]
    y = F.conv2d(x, neighbor_filter, padding=padding,
                 groups=num_samples).squeeze()
    y = y.squeeze(0)
    return y

def backtrack(start_maps, goal_maps, parents, current_t):
    num_samples = start_maps.shape[0]
    parents = parents.type(torch.long)
    goal_maps = goal_maps.type(torch.long)
    start_maps = start_maps.type(torch.long)
    path_maps = goal_maps.type(torch.long)
    num_samples = len(parents)
    loc = (parents * goal_maps.view(num_samples, -1)).sum(-1)
    for t in range(current_t):
        path_maps.view(num_samples, -1)[range(num_samples), loc] = 1
        loc = parents[range(num_samples), loc]
    return path_maps

def get_min(val):
    y = val.reshape(val.shape[0], -1)
    min_val, ind = y.min(dim=-1)
    y_hard = torch.zeros_like(y)
    y_hard[range(len(y_hard)), ind] = 1
    y_hard = y_hard.reshape_as(val)
    y = y.reshape_as(val)
    return min_val, y_hard

def sample_onehot(binmaps):
    n_samples = len(binmaps)
    binmaps_n = binmaps * torch.rand_like(binmaps)
    binmaps_vct = binmaps_n.reshape(n_samples, -1)
    ind = binmaps_vct.max(dim=-1)[1]
    onehots = torch.zeros_like(binmaps_vct)
    onehots[range(n_samples), ind] = 1
    onehots = onehots.reshape_as(binmaps_n)

    return onehots

def dilate_opt_trajs(opt_trajs, map_designs, mechanism):
    neighbor_filter = mechanism.get_neighbor_filter()
    neighbor_filter = neighbor_filter.type_as(opt_trajs)
    num_samples = len(opt_trajs)
    neighbor_filter = torch.repeat_interleave(neighbor_filter, num_samples, 0)
    ot_conv = expand(opt_trajs.squeeze(), neighbor_filter)
    ot_conv = torch.clamp(ot_conv.reshape_as(opt_trajs), 0, 1)
    ot_conv = ot_conv * map_designs
    return ot_conv

def get_hard_medium_easy_masks(opt_dists,
                               device,
                               reduce_dim: bool = True,
                               num_points_per_map: int = 5):
    wall_dist = torch.min(opt_dists)
    n_samples = opt_dists.shape[0]
    od_vct = opt_dists.reshape(n_samples, -1)
    od_nan = od_vct.clone()
    od_nan[od_nan == wall_dist] = float('nan')
    od_nan = torch.nan_to_num(od_nan)
    (od_min, indices) = torch.min(od_nan, axis=1, keepdims=True)
    thes = od_min.matmul(torch.tensor([[1.0, 0.85, 0.70, 0.55]], device=device))
    thes = thes.int()
    thes = torch.transpose(thes, 0, 1)
    thes = thes.reshape(4, n_samples, 1, 1, 1)
    masks_list = []

    for i in range(3):
        binmaps = ((thes[i] <= opt_dists) &
                   (opt_dists < thes[i + 1])) * 1.0
        binmaps = torch.repeat_interleave(binmaps, num_points_per_map, 0)
        masks = _sample_onehot(binmaps, device)
        masks = masks.reshape(n_samples, num_points_per_map,
                              *opt_dists.shape[1:])
        if reduce_dim:
            (masks, indices) = masks.max(axis=1)
        masks_list.append(masks.bool())
    return masks_list

def _sample_onehot(binmaps, device):
    n_samples = len(binmaps)
    binmaps_n = binmaps * torch.rand(binmaps.shape, device= device)
    
    binmaps_vct = binmaps_n.reshape(n_samples, -1)
    ind = binmaps_vct.argmax(axis=-1)
    onehots = torch.zeros_like(binmaps_vct)
    onehots[range(n_samples), ind] = 1
    onehots = onehots.reshape(binmaps_n.shape)
    onehots = onehots.bool()
    return onehots
