import os, math, numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint as odeint2
from torchmetrics.functional import mean_squared_error
import ot
    
class EnergySolver(nn.Module):
    def __init__(self, flow_net, growth_net, state_cost, data_manifold_metric=None, samples=None, timestep_idx=0):
        super(EnergySolver, self).__init__()
        self.flow_net = flow_net
        self.growth_net = growth_net
        self.state_cost = state_cost
        
        self.data_manifold_metric = data_manifold_metric
        self.samples = samples
        self.timestep_idx = timestep_idx

    def forward(self, t, state):
        xt, wt, mt = state
        
        xt.requires_grad_(True)
        wt.requires_grad_(True)
        mt.requires_grad_(True)
        
        t.requires_grad_(True)

        ut = self.flow_net(t, xt)
        gt = self.growth_net(t, xt)

        time=t.expand(xt.shape[0], 1)
        time.requires_grad_(True) 
        
        dx_dt = ut
        dw_dt = gt
                        
        if self.data_manifold_metric is not None:
            vel, _, _ = self.data_manifold_metric.calculate_velocity(
                xt, ut, self.samples, self.timestep_idx
            )
                
            dm_dt = ((vel ** 2).sum(dim =-1) + (gt ** 2)) * wt
        else:
            dm_dt = ((ut**2).sum(dim =-1) + self.state_cost(xt) + (0.1 * (gt ** 2))) * wt
        
        assert xt.shape == dx_dt.shape, f"dx mismatch: expected {xt.shape}, got {dx_dt.shape}"
        assert wt.shape == dw_dt.shape, f"dw mismatch: expected {wt.shape}, got {dw_dt.shape}"
        assert mt.shape == dm_dt.shape, f"dm mismatch: expected {mt.shape}, got {dm_dt.shape}"
        return dx_dt, dw_dt, dm_dt
    
class ReconsLoss(nn.Module):
    def __init__(self, hinge_value=0.01):
        super(ReconsLoss, self).__init__()
        self.hinge_value = hinge_value

    def __call__(self, source, target, groups = None, to_ignore = None, top_k = 5):
        if groups is not None:
            # for global loss
            c_dist = torch.stack([
                torch.cdist(source[i], target[i]) 

                for i in range(1,len(groups))
                if groups[i] != to_ignore
            ])
        else:
            # for local loss
             c_dist = torch.stack([
                torch.cdist(source, target)                 
            ])
        values, _ = torch.topk(c_dist, top_k, dim=2, largest=False, sorted=False)
        values -= self.hinge_value
        values[values<0] = 0
        loss = torch.mean(values)
        return loss