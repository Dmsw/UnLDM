import torch 
import torch.nn as nn
from typing import Optional
from torch.cuda.amp import autocast


class ManifoldAssemble(nn.Module):
    @autocast(enabled=False)
    def __init__(self, P, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.P = P.to(dtype)
        up, sp, vp = torch.linalg.svd(P, full_matrices=True)
        vp = vp.mH
        c = P.shape[0]
        self.up = up # [c, c]  
        self.sp = sp # [c]
        self.vp1 = vp[..., :c] # [b, c]
        self.vp0 = vp[..., c:] # [b, b-c]
        
        
    def calculate_manifold(self, samples, dimension:Optional[int]=1, threshold:Optional[float]=0.99):
        """Calculate the manifold of the samples

        Args:
            samples (List[Tensor]): shape [N, B, H, W] or [B, H, W]
            dimension (int): the dimension of the manifold
            threshold (float): the threshold of the energy

        Returns:
            u: shape [P, B, d]
            s: shape [P, d]
            where P is the number of pixels, d is the dimension, S is the number of samples
        """
        B, H, W = samples[0].shape[-3:]
        X = torch.stack(samples, dim=0).view(-1, B, H * W)
        # X = torch.einsum('n c p -> p c n', X)
        X = X.permute(2, 1, 0)
        u, s, _ = torch.linalg.svd(X, full_matrices=False)
        energy = torch.cumsum(s ** 2, dim=1)
        energy_rate = energy / energy[..., -1:]
        if dimension is not None:
            d = dimension
            s = s[..., :d]
            u = u[..., :d]
        else:
            raise NotImplementedError
            assert threshold is not None
            c = torch.sum(energy_rate < threshold, dim=1).max().item()
        # self.norm_x = torch.norm(X, dim=[1, 2], keepdim=True)
        # X = X / self.norm_x
        # assert torch.allclose(X, u, atol=1e-6), f"{(X - u).abs().max()}"
        return u, energy_rate[..., d-1]
    
    def cal_x(self, x1, x0):
        """estimate x from x1 and x0

        Args:
            x1 (Tensor): shape [P, 3]
            x0 (Tensor): shape [P, B-3]
        """
        # x = torch.einsum('p c, b c -> p b', x1, self.vp1) + \
        #     torch.einsum('p c, b c -> p b', x0, self.vp0)
        x = torch.matmul(x1, self.vp1.T) + torch.matmul(x0, self.vp0.T)
        return x
    
    def estimate_observable(self, y):
        """estimate the observable part of x from y

        Args:
            y (Tensor): shape [P, C]
        """
        P_pinv = torch.diag(1/self.sp) @ self.up.T
        # x1 = torch.einsum('p c, b c -> p b', y, P_pinv)
        x1 = torch.matmul(y, P_pinv.T)
        return x1
    
    def project_unobservable(self, x):
        """project x to the unobservable space

        Args:
            x (Tensor): shape [P, C]
        """
        # x0 = torch.einsum('p b, d b -> p d', x, self.vp0.T)
        x0 = torch.matmul(x, self.vp0)
        return x0

    def estimate_unobservable_(self, u, y):
        # Pl = torch.einsum('p b d, c b -> p c d', u, self.P) # [P, 3, d]
        Pl = torch.matmul(self.P, u)
        assert torch.allclose(Pl[..., 0], y / self.norm_x[..., 0], atol=1e-3), f"{(Pl[..., 0] - y / self.norm_x[..., 0]).abs().max()}"
        # Pl_inv = torch.linalg.pinv(Pl) # [P, 3, d]
        pltpl = torch.einsum('p d c, p d e -> p c e', Pl, Pl)
        assert torch.allclose(pltpl[..., 0, 0], (torch.norm(y, dim=1)**2)/(self.norm_x[..., 0, 0] ** 2), atol=1e-3), f"{(pltpl[..., 0, 0] - (torch.norm(y, dim=1)**2)/(self.norm_x[..., 0, 0] ** 2)).abs().max()}"
        Pl_inv = torch.einsum('p e d, p c d -> p e c', 1/pltpl, Pl) # TODO: check the correctness of the calculation
        x1 = torch.einsum('p c d, p c -> p d', Pl_inv, y)
        # x1 = self.norm_x[..., 0]
        assert torch.allclose(self.norm_x[..., 0], x1.squeeze(), atol=1e-3), f"{(self.norm_x[..., 0] - x1).abs().max()}"
        x = torch.einsum('p d, p b d -> p b', x1, u)
        
        # x_ = torch.einsum('p b, p b d -> p d', x, u)
        
        x0 = self.project_unobservable(x)
        return x0
    
    def estimate_unobservable(self, u, y):
        Pl = torch.matmul(self.P, u)
        x1 = torch.linalg.lstsq(Pl, y).solution
        # assert torch.allclose(self.norm_x[..., 0], x1, atol=1e-2), f"{(self.norm_x[..., 0] - x1).abs().max()}"
        x = torch.einsum('p d, p b d -> p b', x1, u)                
        x0 = self.project_unobservable(x)
        return x0
    
    def forward_per_rgb(self, samples, rgb):
        c, H, W = rgb.shape[-3:]
        B = self.P.shape[1]
        assert c == 3
        y = rgb.reshape([c, H*W]).T
        u, s = self.calculate_manifold(samples, dimension=1)
        x0 = self.estimate_unobservable(u, y)
        x1 = self.estimate_observable(y)
        x = self.cal_x(x1, x0).T
        hsi = x.reshape([B, H, W])
        return hsi, s
    
    @autocast(enabled=False)
    def forward(self, samples, rgb):
        rgb = torch.squeeze(rgb)
        dtype = rgb.dtype
        rgb = rgb.to(self.dtype)
        samples = [s.to(self.dtype) for s in samples]
        assert len(rgb.shape) == 3
        hsi, s =  self.forward_per_rgb(samples=samples, rgb=rgb)
        return hsi.to(dtype)[None], s.to(dtype)[None]
           