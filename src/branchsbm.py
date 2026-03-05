import sys
import torch
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, pad_t_like_x
import torch.nn as nn

class BranchSBM(ConditionalFlowMatcher):
    def __init__(
        self, geopath_nets: nn.ModuleList = None, alpha: float = 1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.geopath_nets = geopath_nets
        if self.alpha != 0:
            assert (
                geopath_nets is not None
            ), "GeoPath model must be provided if alpha != 0"
            
        self.branches = len(geopath_nets)

    def gamma(self, t, t_min, t_max):
        return (
            1.0
            - ((t - t_min) / (t_max - t_min)) ** 2
            - ((t_max - t) / (t_max - t_min)) ** 2
        )

    def d_gamma(self, t, t_min, t_max):
        return 2 * (-2 * t + t_max + t_min) / (t_max - t_min) ** 2

    def compute_mu_t(self, x0, x1, t, t_min, t_max, branch_idx):
        assert branch_idx < self.branches, "Index out of bounds"

        with torch.enable_grad():
            t = pad_t_like_x(t, x0)
            if self.alpha == 0:
                return (t_max - t) / (t_max - t_min) * x0 + (t - t_min) / (
                    t_max - t_min
                ) * x1
                
            # compute value for specific branch
            self.geopath_net_output = self.geopath_nets[branch_idx](x0, x1, t)
            if self.geopath_nets[branch_idx].time_geopath:
                self.doutput_dt = torch.autograd.grad(
                    self.geopath_net_output,
                    t,
                    grad_outputs=torch.ones_like(self.geopath_net_output),
                    create_graph=False, 
                    retain_graph=True,
                )[0]
        return (
            (t_max - t) / (t_max - t_min) * x0
            + (t - t_min) / (t_max - t_min) * x1
            + self.gamma(t, t_min, t_max) * self.geopath_net_output
        )

    def sample_xt(self, x0, x1, t, epsilon, t_min, t_max, branch_idx):
        assert branch_idx < self.branches, "Index out of bounds"
        mu_t = self.compute_mu_t(x0, x1, t, t_min, t_max, branch_idx)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def sample_location_and_conditional_flow(
        self,
        x0,
        x1,
        t_min,
        t_max,
        branch_idx,
        training_geopath_net=False,
        midpoint_only=False,
        t=None,
    ):

        self.training_geopath_net = training_geopath_net
        with torch.enable_grad():
            if t is None:
                t = torch.rand(x0.shape[0], requires_grad=True)
            t = t.type_as(x0)
            t = t * (t_max - t_min) + t_min
            if midpoint_only:
                t = (t_max + t_min) / 2 * torch.ones_like(t).type_as(x0)
        
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        
        # compute xt and ut for branch_idx
        xt = self.sample_xt(x0, x1, t, eps, t_min, t_max, branch_idx)
        ut = self.compute_conditional_flow(x0, x1, t, xt, t_min, t_max, branch_idx)

        return t, xt, ut

    def compute_conditional_flow(self, x0, x1, t, xt, t_min, t_max, branch_idx):
        del xt
        t = pad_t_like_x(t, x0)
        if self.alpha == 0:
            return (x1 - x0) / (t_max - t_min)
        
        return (
            (x1 - x0) / (t_max - t_min)
            + self.d_gamma(t, t_min, t_max) * self.geopath_net_output
            + (
                self.gamma(t, t_min, t_max) * self.doutput_dt
                if self.geopath_nets[branch_idx].time_geopath
                else 0
            )
        )