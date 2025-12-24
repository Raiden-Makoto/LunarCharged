import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore

class CGNNLayer(nn.Module):
    """
    GNN Layer that accepts Latent Code (z) and Time (t) conditioning.
    """
    def __init__(self, hidden_dim, rbf_bins=32):
        super().__init__()
        # Input size is larger because we concat (Node + Neighbor + Edge + z + t)
        input_dim = (hidden_dim * 2) + rbf_bins + hidden_dim + hidden_dim 
        
        self.message_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj_dist, mask, cond_vec):
        # x: (Batch, N, Hidden)
        # cond_vec: (Batch, Hidden + Hidden) -> Combined z and t
        
        B, N, H = x.shape
        
        # 1. Expand Conditioning to all nodes
        # cond_expanded: (B, N, N, Cond_Dim)
        cond_expanded = cond_vec.unsqueeze(1).unsqueeze(1).expand(B, N, N, -1)
        
        # 2. Pairwise Features
        x_i = x.unsqueeze(2).expand(B, N, N, H)
        x_j = x.unsqueeze(1).expand(B, N, N, H)
        
        # 3. Combine Everything: Source + Target + Distance + Conditioning
        combined = torch.cat([x_i, x_j, adj_dist, cond_expanded], dim=-1)
        
        # 4. Message Passing
        messages = self.message_mlp(combined)
        mask_j = mask.unsqueeze(1).unsqueeze(-1).float()
        messages = messages * mask_j
        aggr_messages = torch.sum(messages, dim=2)
        
        # 5. Update
        out = torch.cat([x, aggr_messages], dim=-1)
        out = self.update_mlp(out)
        out = self.norm(x + out)
        
        return out