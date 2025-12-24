import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore

class DGNNLayer(nn.Module):
    """
    A single layer of Message Passing for dense graphs.
    """
    def __init__(self, hidden_dim, rbf_bins=32):
        super().__init__()
        # Message function: Process neighbor info + edge info
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + rbf_bins, hidden_dim), # *2 for node+neighbor, +rbf_bins for RBF
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Update function: Combine old node info with new message
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj_dist, mask):
        # x: (Batch, N, Hidden) - Atom features
        # adj_dist: (Batch, N, N, RBF_Bins) - Distance embeddings
        # mask: (Batch, N) - Valid atoms
        
        B, N, H = x.shape
        
        # 1. Create Source and Target features for all pairs
        # x_i: (B, N, 1, H) -> Target atoms (repeated N times)
        # x_j: (B, 1, N, H) -> Source atoms (repeated N times)
        x_i = x.unsqueeze(2).expand(B, N, N, H)
        x_j = x.unsqueeze(1).expand(B, N, N, H)
        
        # 2. Concatenate Node Features + Edge Features
        # input: (B, N, N, 2*H + RBF)
        combined = torch.cat([x_i, x_j, adj_dist], dim=-1)
        
        # 3. Compute Messages
        messages = self.message_mlp(combined) # (B, N, N, H)
        
        # 4. Mask out messages from "Padding" atoms
        # mask_j: (B, 1, N, 1)
        mask_j = mask.unsqueeze(1).unsqueeze(-1).float()
        messages = messages * mask_j 
        
        # 5. Aggregate Messages (Sum over neighbors j)
        aggr_messages = torch.sum(messages, dim=2) # (B, N, H)
        
        # 6. Update Node Features
        out = torch.cat([x, aggr_messages], dim=-1)
        out = self.update_mlp(out)
        
        # Residual connection + Norm
        out = self.norm(x + out)
        return out