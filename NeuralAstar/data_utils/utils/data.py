from __future__ import print_function
import torch
import torch.utils.data as data
from .mechanism import Mechanism

TEST_RANDOM_SEED = 2020
NUM_POINTS_PER_MAP = 5

def get_opt_trajs(start_maps, goal_maps,
                  opt_policies, mechanism: Mechanism):

    opt_trajs = torch.zeros_like(start_maps)
    opt_policies = opt_policies.transpose((0, 2, 3, 4, 1))

    for i in range(len(opt_trajs)):
        current_loc = tuple(torch.tensor(torch.nonzero(start_maps[i])).squeeze())
        goal_loc = tuple(torch.array(torch.nonzero(goal_maps[i])).squeeze())

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


def get_hard_medium_easy_masks(opt_dists,
                               device,
                               reduce_dim: bool = True,
                               num_points_per_map: int = 5):
    # make sure the selected nodes are random but fixed
    torch.random.seed()
    # impossible distance
    wall_dist = torch.min(opt_dists)
    n_samples = opt_dists.shape[0]
    od_vct = opt_dists.reshape(n_samples, -1)
    od_nan = od_vct.clone()
    od_nan[od_nan == wall_dist] = torch.nan
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
