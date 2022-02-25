import torch

def compute_opt_suc_exp(pred_dists, rel_exps, pred_exps, exps, opt_dists, masks):

    wall_dist = torch.min(opt_dists)  # impossible distance
    diff_dists = pred_dists - opt_dists
    opt = (diff_dists == 0)[masks]
    suc = ~(pred_dists == wall_dist)[masks]
    exp = (torch.ones_like(pred_dists)[masks] if rel_exps is None else
           rel_exps[(pred_dists != wall_dist) & masks])
    num = (torch.ones_like(pred_dists)[masks] if  pred_exps is None else
           pred_exps[(pred_dists != wall_dist) & masks])
    denom = (torch.ones_like(pred_dists)[masks] if exps is None else
           exps[(pred_dists != wall_dist) & masks])
    if len(exp) == 0:
        exp = torch.ones_like(pred_dists)[masks]  ## workaround
    return opt, suc, exp, num, denom


def compute_mean_metrics(pred_dists, rel_exps, pred_exps, exps, opt_dists, masks):

    opt, suc, exp, num, denom = compute_opt_suc_exp(pred_dists, rel_exps, pred_exps, exps, opt_dists, masks)
#     exp_length = len(exp)
#     print("Opt Length =", len(opt))
    print("Exp Length =", len(exp))
    print("Num Length =", len(num))
    print("Denom Length =", len(denom))
    return opt.float().mean(), suc.float().mean(), exp.float().mean(), num.float().mean(), denom.float().mean()


def compute_opt_exp(pred_dists, rel_exps, opt_dists, masks):

    wall_dist = torch.min(opt_dists)  # impossible distance
    diff_dists = pred_dists - opt_dists
    opt1 = (diff_dists == 0)[(pred_dists != wall_dist) & masks]
    exp = (torch.ones_like(pred_dists)[masks] if rel_exps is None else
           rel_exps[(pred_dists != wall_dist) & masks])

    return opt1, exp
