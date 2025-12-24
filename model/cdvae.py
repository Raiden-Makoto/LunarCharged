import torch #type: ignore
import torch.nn as nn #type: ignore
import numpy as np #type: ignore
from torch.utils.checkpoint import checkpoint #type: ignore
from model.encoder import CrystalEncoder
from model.decoder import DenoisingDecoder

class CrystalDiffusionVAE(nn.Module):
    def __init__(self, hidden_dim=128, latent_dim=64, num_layers=3, num_timesteps=1000, use_checkpoint=True, rbf_bins=32, rbf_vmin=0, rbf_vmax=8.0):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.encoder = CrystalEncoder(hidden_dim, latent_dim, num_layers, use_checkpoint=False, rbf_bins=rbf_bins, rbf_vmin=rbf_vmin, rbf_vmax=rbf_vmax)
        self.decoder = DenoisingDecoder(hidden_dim, latent_dim, num_layers, use_checkpoint=False, rbf_bins=rbf_bins, rbf_vmin=rbf_vmin, rbf_vmax=rbf_vmax)
        
        # --- Diffusion Noise Schedule (Linear) ---
        # Reduced beta_end to increase signal-to-noise ratio (easier task for small model)
        self.num_timesteps = num_timesteps
        beta_start = 0.0001
        beta_end = 0.01  # Reduced from 0.02 to increase signal-to-noise ratio
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # Pre-calculate diffusion terms to save speed
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register as buffers (so they save with the model but aren't trainable)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def reparameterize(self, mu, log_var):
        """
        The 'Magic' VAE step: 
        Takes Mean and Variance -> Returns a sampled Vector z
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, atom_types, frac_coords, lattice, mask):
        """
        Runs the entire training step in one go.
        Returns:
            pred_noise: What the model predicted
            target_noise: The actual noise we added (Ground Truth)
            mu, log_var: For KL Loss
        """
        device = frac_coords.device
        batch_size = frac_coords.size(0)

        # 1. ENCODE
        # Get the "Blueprint" (z) of the crystal
        if self.use_checkpoint and self.training:
            mu, log_var = checkpoint(
                self.encoder,
                atom_types, frac_coords, lattice, mask,
                use_reentrant=False
            )
        else:
            mu, log_var = self.encoder(atom_types, frac_coords, lattice, mask)
        z = self.reparameterize(mu, log_var)

        # 2. DIFFUSION (Add Noise)
        # Pick a random time 't' for every crystal in the batch
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        # Generate random noise (epsilon)
        target_noise = torch.randn_like(frac_coords)
        
        # Create the Noisy Coords (x_t)
        # x_t = signal * x_0 + noise_factor * noise
        # We use the buffers we defined in __init__
        noise_level = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        signal_level = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        
        noisy_coords = signal_level * frac_coords + noise_level * target_noise
        
        # Apply Mask (Don't noise the padding!)
        mask_3d = mask.unsqueeze(-1).float()
        noisy_coords = noisy_coords * mask_3d
        target_noise = target_noise * mask_3d

        # 3. DECODE (Predict Noise)
        # Ask the model: "Given this mess (noisy_coords) and the blueprint (z), find the noise."
        if self.use_checkpoint and self.training:
            pred_noise, pred_types = checkpoint(
                self.decoder,
                atom_types, noisy_coords, lattice, mask, t, z,
                use_reentrant=False
            )
        else:
            pred_noise, pred_types = self.decoder(atom_types, noisy_coords, lattice, mask, t, z)
        
        # Mask output again to be safe
        pred_noise = pred_noise * mask_3d

        return pred_noise, target_noise, mu, log_var