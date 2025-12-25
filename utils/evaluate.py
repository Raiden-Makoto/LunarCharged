import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- IMPORTS ---
from model.cdvae import CrystalDiffusionVAE

# --- CONFIGURATION ---
# MUST match your training settings exactly
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "generated_batteries"

# Model Hyperparameters (Must match train.py)
HIDDEN_DIM = 128
LATENT_DIM = 64
NUM_LAYERS = 3
NUM_TIMESTEPS = 1000

# --- HELPER: POST-PROCESSING ---
def tensor_to_structure(atom_types, frac_coords, lattice_matrix):
    """
    Converts model outputs back to a Pymatgen Structure.
    """
    # 1. Map atom types (0, 1, 2...) back to elements
    # You need the same mapping from your dataset!
    # For batteries, common elements: Li, O, P, S, La, Zr...
    # If you lost the mapping, we can guess or use a dummy map.
    # ideally load 'atom_mapping.pkl' if you saved it.
    
    # For now, let's assume a generic mapping or use placeholders if unknown
    # This is just for visualization
    known_elements = [
        "Li", "O", "P", "S", "La", "Zr", "Ti", "Al", "Ge", "Si", 
        "Co", "Ni", "Mn", "Fe", "Cu", "Zn"
    ]
    
    species = []
    for t in atom_types:
        idx = int(t.item())
        if idx < len(known_elements):
            species.append(known_elements[idx])
        else:
            species.append("X") # Unknown placeholder

    # 2. Create Lattice
    # lattice_matrix is (3, 3)
    l = Lattice(lattice_matrix.cpu().numpy())
    
    # 3. Create Structure
    coords = frac_coords.cpu().numpy()
    return Structure(l, species, coords)

# --- GENERATION LOOP ---
@torch.no_grad()
def generate(checkpoint_epoch=None):
    """
    Generate crystal structures using the trained model.
    
    Args:
        checkpoint_epoch: Which epoch checkpoint to load (e.g., 10, 20, etc.)
                          If None, loads the latest checkpoint.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Model
    if checkpoint_epoch is None:
        # Find latest checkpoint
        checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("cdvae_epoch_") and f.endswith(".pt")]
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
        # Extract epoch numbers and find max
        epochs = [int(f.split("_")[2].split(".")[0]) for f in checkpoint_files]
        checkpoint_epoch = max(epochs)
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"cdvae_epoch_{checkpoint_epoch}.pt")
    print(f"Loading model from {checkpoint_path}...")
    
    model = CrystalDiffusionVAE(
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        num_timesteps=NUM_TIMESTEPS,
        use_checkpoint=False
    ).to(DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 2. Setup Generation
    num_samples = 5
    print(f"Generating {num_samples} crystal candidates...")
    
    # We need a template for the lattice and atom counts.
    # In a real conditional generation, you'd sample these.
    # For now, we manually define a "Battery-like" box.
    
    # Example: Li7La3Zr2O12 (LLZO) is ~12-13 Angstroms
    # Let's try a smaller cubic box for stability: 10x10x10 Angstroms
    # With ~20 atoms
    dummy_lattice = torch.eye(3).unsqueeze(0).to(DEVICE) * 10.0  # 10 Angstrom Cube
    dummy_lattice = dummy_lattice.repeat(num_samples, 1, 1)  # Batch of lattices
    
    # Random atom counts (e.g. 20 atoms)
    num_atoms = 20
    atom_types = torch.randint(1, 10, (num_samples, num_atoms), device=DEVICE)  # Random types (1-9, avoid 0)
    mask = torch.ones((num_samples, num_atoms), dtype=torch.bool, device=DEVICE)
    
    # 3. Sample Latent Space (z) - random latent vectors
    z = torch.randn(num_samples, LATENT_DIM, device=DEVICE)
    
    # 4. Start Reverse Diffusion (Denoising)
    # Start with Pure Gaussian Noise for coordinates
    # Note: Model expects fractional coordinates in [-0.5, 0.5] range (centered at zero)
    curr_frac_coords = torch.rand(num_samples, num_atoms, 3, device=DEVICE) - 0.5
    
    print("Denoising...")
    for t in tqdm(reversed(range(NUM_TIMESTEPS)), desc="Denoising"):
        t_batch = torch.full((num_samples,), t, device=DEVICE, dtype=torch.long)
        
        # Generate time embedding
        t_emb = model.time_embedding(t_batch.float())
        
        # Predict Noise (Epsilon) in Cartesian (Angstroms)
        pred_noise_cart, _ = model.decoder(
            atom_types, 
            curr_frac_coords, 
            dummy_lattice, 
            mask, 
            t_emb, 
            z
        )
        
        # --- DDPM SAMPLER MATH ---
        # Get diffusion schedule values from model buffers
        alpha_bar_t = model.alphas_cumprod[t]  # Tensor scalar
        alpha_bar_t_prev = model.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=DEVICE)
        
        # Calculate alpha_t (not stored directly, derive from alpha_bar)
        alpha_t = alpha_bar_t / alpha_bar_t_prev if t > 0 else alpha_bar_t
        beta_t = 1.0 - alpha_t
        
        # Convert predicted noise from Cartesian to Fractional for the update step
        # (Inverse of: cart = frac @ lattice) -> frac = cart @ inv_lattice
        inv_lattice = torch.linalg.inv(dummy_lattice)  # (B, 3, 3)
        pred_noise_frac = torch.bmm(pred_noise_cart, inv_lattice)  # (B, N, 3)
        
        # Apply mask
        mask_3d = mask.unsqueeze(-1).float()
        pred_noise_frac = pred_noise_frac * mask_3d
        
        # Update x_{t-1} using DDPM sampling formula
        # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        coeff = beta_t / sqrt_one_minus_alpha_bar_t
        mean = (1.0 / sqrt_alpha_t) * (curr_frac_coords - coeff * pred_noise_frac)
        
        # Add noise (except for last step t=0)
        if t > 0:
            # Posterior variance: beta_tilde = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
            posterior_variance = beta_t * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
            sigma = torch.sqrt(posterior_variance)
            noise = torch.randn_like(curr_frac_coords)
            curr_frac_coords = mean + sigma * noise
        else:
            curr_frac_coords = mean
        
        # Wrap to [-0.5, 0.5] range (Periodic Boundary)
        curr_frac_coords = curr_frac_coords - torch.round(curr_frac_coords)
        # Apply mask
        curr_frac_coords = curr_frac_coords * mask_3d

    # 5. Save Results
    print(f"Saving to {OUTPUT_DIR}...")
    for i in range(num_samples):
        try:
            struct = tensor_to_structure(atom_types[i], curr_frac_coords[i], dummy_lattice[i])
            filepath = os.path.join(OUTPUT_DIR, f"battery_{i}.cif")
            CifWriter(struct).write_file(filepath)
            print(f"  Saved {filepath}")
        except Exception as e:
            print(f"  Failed to save sample {i}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate crystal structures")
    parser.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch to load (default: latest)")
    args = parser.parse_args()
    generate(checkpoint_epoch=args.epoch)