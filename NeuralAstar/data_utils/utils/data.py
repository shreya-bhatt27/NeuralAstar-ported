from __future__ import print_function
import numpy as np
import torch
import torch.utils.data as data
from .mechanism import Mechanism

TEST_RANDOM_SEED = 2020
NUM_POINTS_PER_MAP = 5

def get_opt_trajs(start_maps: np.ndarray, goal_maps: np.ndarray,
                  opt_policies: np.ndarray, mechanism: Mechanism):

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


def get_hard_medium_easy_masks(opt_dists_CPU: np.ndarray,
                               reduce_dim: bool = True,
                               num_points_per_map: int = 5):
    # make sure the selected nodes are random but fixed
    np.random.seed(TEST_RANDOM_SEED)
    # impossible distance
    wall_dist = np.min(opt_dists_CPU)

    n_samples = opt_dists_CPU.shape[0]
    od_vct = opt_dists_CPU.reshape(n_samples, -1)
    od_nan = od_vct.copy()
    od_nan[od_nan == wall_dist] = np.nan
    od_min = np.nanmin(od_nan, axis=1, keepdims=True)
    thes = od_min.dot(np.array([[1.0, 0.85, 0.70, 0.55]])).astype("int").T
    thes = thes.reshape(4, n_samples, 1, 1, 1)

    masks_list = []
    for i in range(3):
        binmaps = ((thes[i] <= opt_dists_CPU) &
                   (opt_dists_CPU < thes[i + 1])) * 1.0
        binmaps = np.repeat(binmaps, num_points_per_map, 0)
        masks = _sample_onehot(binmaps)
        masks = masks.reshape(n_samples, num_points_per_map,
                              *opt_dists_CPU.shape[1:])
        if reduce_dim:
            masks = masks.max(axis=1)
        masks_list.append(masks.astype(bool))
    return masks_list


def _sample_onehot(binmaps):
    n_samples = len(binmaps)
    binmaps_n = binmaps * np.random.rand(*binmaps.shape)

    binmaps_vct = binmaps_n.reshape(n_samples, -1)
    ind = binmaps_vct.argmax(axis=-1)
    onehots = np.zeros_like(binmaps_vct)
    onehots[range(n_samples), ind] = 1
    onehots = onehots.reshape(binmaps_n.shape).astype("bool")

    return onehots
