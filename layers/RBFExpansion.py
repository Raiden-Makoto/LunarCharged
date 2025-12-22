import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore

class RBFExpansion(nn.Module):
    """
    Expands distances into a wave-like representation (Gaussian Basis).
    This helps the model learn chemical bond lengths (e.g., 1.5A vs 2.0A).
    """
    def __init__(self, vmin=0, vmax=10, bins=40):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.centers = torch.linspace(vmin, vmax, bins)
        self.gamma = (vmax - vmin) / bins

    def forward(self, distance):
        # distance: (Batch, N, N)
        # centers: (bins)
        return torch.exp(-(distance.unsqueeze(-1) - self.centers.to(distance.device))**2 / self.gamma**2)