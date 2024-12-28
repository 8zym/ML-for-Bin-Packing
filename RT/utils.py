import numpy as np
import torch
from Params import args
import random
import math

#seed=523
def seed_torch(seed=523):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def _calc_log_likelihood_3d(actions, s_log, r_log):

    # actions (batch, 4)
    # log (batch, action_num)
    
    # (batch, 1)
    action_r = actions['rotate']
    action_x = actions['x']
    action_y = actions['y']

    #(batch)
    action_s = actions['index']
    s_log_p = s_log.gather(1, action_s).squeeze(-1)
    assert (s_log_p > -1000).data.all(), "log probability should not -inf, check sampling"

    r_log_p = r_log.gather(1, action_r).squeeze(-1)
    assert (r_log_p > -1000).data.all(), "log probability should not -inf, check sampling"
    
    log_likelihood = s_log_p + r_log_p

    return log_likelihood

def _calc_entropy_3d(s_log, r_log):
    # log (batch, action_num)
    
    # S=-/sum_i (p_i \ln p_i)
    s_entropy = -1 * (s_log.exp() *s_log).sum(dim=-1)

    r_entropy = -1 * (r_log.exp() * r_log).sum(dim=-1)
    
    # (batch)
    entropys = torch.stack([s_entropy, r_entropy], dim=-1)
    assert not torch.isnan(entropys).any()

    return entropys

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary