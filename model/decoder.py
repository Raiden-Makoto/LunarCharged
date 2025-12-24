import torch #type: ignore
import torch.nn as nn #type: ignore
from layers.SineEmbed import SinusoidalTimeEmbeddings
from layers.RBFExpansion import RBFExpansion
from layers.CGNNLayer import CGNNLayer

class DenoisingDecoder(nn.Module):
    def __init__(self, hidden_dim=128, latent_dim=64, num_layers=3, use_checkpoint=False, rbf_bins=32, rbf_vmin=0, rbf_vmax=8.0):
        super().__init__()
        
        # Embeddings
        self.atom_embedding = nn.Embedding(100, hidden_dim, padding_idx=0)
        self.time_embedding = SinusoidalTimeEmbeddings(hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim) # Project z to match hidden size
        self.rbf = RBFExpansion(vmin=rbf_vmin, vmax=rbf_vmax, bins=rbf_bins)
        
        # Layers
        self.layers = nn.ModuleList([CGNNLayer(hidden_dim, rbf_bins=rbf_bins) for _ in range(num_layers)])
        
        # Output Heads
        # 1. Predict Coordinate Shift (Noise)
        self.coord_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3) # Output: (dx, dy, dz)
        )
        
        # 2. Predict Atom Types (Optional for now, but good to have)
        self.type_predictor = nn.Linear(hidden_dim, 100) 

    def forward(self, atom_types, frac_coords, lattice, mask, t, z):
        """
        Input:
            atom_types: (B, N)
            frac_coords: (B, N, 3) -- These are NOISY coordinates
            t: (B) -- Time step integers
            z: (B, Latent) -- Latent vector
        """
        # 1. Prepare Conditioning
        t_emb = self.time_embedding(t)    # (B, H)
        z_emb = self.latent_proj(z)       # (B, H)
        cond = torch.cat([t_emb, z_emb], dim=-1) # (B, 2*H)
        
        # 2. Geometry Setup (Same as Encoder)
        cart_coords = torch.bmm(frac_coords, lattice)
        diff = cart_coords.unsqueeze(2) - cart_coords.unsqueeze(1)
        dist_sq = torch.sum(diff**2, dim=-1)
        dist = torch.sqrt(dist_sq + 1e-6)
        rbf_edges = self.rbf(dist)
        
        # 3. Initial Features
        h = self.atom_embedding(atom_types)
        
        # 4. Run Conditioned GNN
        for layer in self.layers:
            h = layer(h, rbf_edges, mask, cond)
            
        # 5. Predictions
        pred_noise = self.coord_predictor(h) # (B, N, 3)
        pred_types = self.type_predictor(h)  # (B, N, 100)
        
        return pred_noise, pred_types