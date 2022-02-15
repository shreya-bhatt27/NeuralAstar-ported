"""
Functions for evaluation for metrics (Opt, Exp, Hmean and their confidence intervals
and their condfidence intervals)
Adapted in PyTorch from https://github.com/omron-sinicx/neural-astar
"""

import torch

def compute_opt_suc_exp(pred_dists, rel_exps, opt_dists, masks):

    wall_dist = torch.min(opt_dists)  # impossible distance
    diff_dists = pred_dists - opt_dists
    opt = (diff_dists == 0)[masks]
    suc = ~(pred_dists == wall_dist)[masks]
    exp = (torch.ones_like(pred_dists)[masks] if rel_exps is None else
           rel_exps[(pred_dists != wall_dist) & masks])
    if len(exp) == 0:
        exp = torch.ones_like(pred_dists)[masks]  ## workaround

    return opt, suc, exp


def compute_mean_metrics(pred_dists, rel_exps, opt_dists, masks):

    opt, suc, exp = compute_opt_suc_exp(pred_dists, rel_exps, opt_dists, masks)

    return opt.float().mean(), suc.float().mean(), exp.float().mean()


def compute_opt_exp(pred_dists, rel_exps, opt_dists, masks):

    wall_dist = torch.min(opt_dists)  # impossible distance
    diff_dists = pred_dists - opt_dists
    opt1 = (diff_dists == 0)[(pred_dists != wall_dist) & masks]
    exp = (torch.ones_like(pred_dists)[masks] if rel_exps is None else
           rel_exps[(pred_dists != wall_dist) & masks])

    return opt1, exp
