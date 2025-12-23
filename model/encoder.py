import torch #type: ignore
import torch.nn as nn #type: ignore
from layers.RBFExpansion import RBFExpansion
from layers.DGNNLayer import DGNNLayer

class CrystalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, latent_dim=64, num_layers=2, use_checkpoint=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Embeddings
        self.atom_embedding = nn.Embedding(100, hidden_dim, padding_idx=0)
        self.rbf = RBFExpansion(bins=40)
        
        # 2. GNN Layers
        self.layers = nn.ModuleList([DGNNLayer(hidden_dim) for _ in range(num_layers)])
        
        # 3. Output Heads (Mean and Variance for VAE)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, atom_types, frac_coords, lattice, mask):
        """
        Input:
            atom_types: (B, N)
            frac_coords: (B, N, 3)
            lattice: (B, 3, 3)
            mask: (B, N) boolean
        """
        # 1. Get Cartesian Coords (fractional * lattice)
        # (B, N, 3)
        cart_coords = torch.bmm(frac_coords, lattice)
        
        # 2. Compute Pairwise Distances (B, N, N)
        # (x-y)^2 = x^2 + y^2 - 2xy
        diff = cart_coords.unsqueeze(2) - cart_coords.unsqueeze(1)
        dist_sq = torch.sum(diff**2, dim=-1)
        dist = torch.sqrt(dist_sq + 1e-6) # Add epsilon to avoid NaN at 0
        
        # --- DEBUG INSERTION ---
        if torch.rand(1).item() < 0.01: # Print only 1% of the time to avoid spam
            print(f"\n--- DEBUG DIAGNOSTICS ---")
            print(f"Distances (Min/Mean/Max): {dist.min().item():.2f} / {dist.mean().item():.2f} / {dist.max().item():.2f}")
            
            # Check RBF activation
            rbf_vals = self.rbf(dist)
            print(f"RBF Activations (Mean): {rbf_vals.mean().item():.4f}")
            if rbf_vals.mean().item() < 1e-4:
                print("⚠️ CRITICAL WARNING: RBF Layer is dead (zeros). Atoms are disconnected.")
        # -----------------------
        
        # 3. Expand Distances (RBF)
        rbf_edges = self.rbf(dist) # (B, N, N, 40)
        
        # 4. Initial Node Embeddings
        h = self.atom_embedding(atom_types) # (B, N, H)
        
        # 5. Run GNN Layers
        for layer in self.layers:
            h = layer(h, rbf_edges, mask)
        
        # 6. Global Pooling (Average over atoms)
        # We must ignore padded atoms in the average
        mask_float = mask.unsqueeze(-1).float() # (B, N, 1)
        sum_h = torch.sum(h * mask_float, dim=1) # (B, H)
        num_valid = torch.sum(mask_float, dim=1) # (B, 1)
        
        global_feat = sum_h / (num_valid + 1e-6)
        
        # 7. Predict Latent Distribution
        mu = self.fc_mu(global_feat)
        log_var = self.fc_var(global_feat)
        
        return mu, log_var