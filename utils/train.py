import torch #type: ignore
import torch.nn.functional as F #type: ignore
import torch.optim as optim #type: ignore
from tqdm import tqdm #type: ignore
import os #type: ignore
import sys

# --- CUSTOM MODULES ---
# Add parent directory to path so we can import from utils and model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.dataloader import BatteryDataset, collate_mols
from torch.utils.data import DataLoader, random_split #type: ignore
from model.cdvae import CrystalDiffusionVAE

# --- HYPERPARAMETERS ---
# Hardware / System
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"

# Training Config
BATCH_SIZE = 16        # Safe for 10GB Unified Memory
GRAD_ACCUM_STEPS = 8   # Accumulate gradients over 8 batches (effective batch size = 16 * 8 = 128)
LR = 1e-3              # Learning Rate
EPOCHS = 150           # Total passes through data
KL_WEIGHT = 1.0       # Weight for KL Divergence (keeps latent space neat)
GRAD_CLIP = 1.0        # Prevents exploding gradients

# Model Architecture
HIDDEN_DIM = 128
LATENT_DIM = 64
NUM_LAYERS = 3
TIMESTEPS = 1000

def train():
    # 1. Setup Environment
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    print(f"🚀 Starting MOF-Diffusion Training")
    print(f"   Device:      {DEVICE}")
    print(f"   Batch Size:  {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS} with gradient accumulation)")
    print(f"   Latent Dim:  {LATENT_DIM}")
    print("=" * 60)

    # 2. Load Data and Split (80% train, 20% test - but we only use train)
    # 'processed_graphs' is the folder created by process_mofs_atomic
    full_dataset = BatteryDataset("data/battery_materials.pkl")
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    # Split dataset
    train_dataset, _ = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create dataloader only for training data
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_mols,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"📚 Data Loaded: {total_size} total crystals")
    print(f"   Training: {train_size} crystals (80%)")
    print(f"   Test: {test_size} crystals (20% - not used for training)")

    # 3. Initialize Model
    model = CrystalDiffusionVAE(
        hidden_dim=HIDDEN_DIM, 
        latent_dim=LATENT_DIM, 
        num_layers=NUM_LAYERS,
        num_timesteps=TIMESTEPS
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 4. Training Loop
    model.train()
    
    for epoch in range(EPOCHS):
        total_recon_loss = 0
        total_kl_loss = 0
        batch_count = 0
        
        # Progress Bar for this Epoch
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            # --- MEMORY FIX 1: Use set_to_none=True to delete grads instead of zeroing ---
            optimizer.zero_grad(set_to_none=True)  # SAVES MEMORY: deletes grads instead of zeroing them
            # Move Data to Device
            atom_types = batch['atom_types'].to(DEVICE)
            frac_coords = batch['frac_coords'].to(DEVICE)
            lattice = batch['lattice'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            # --- FORWARD PASS ---
            # Model handles encoding, sampling z, adding noise, and decoding
            pred_noise, target_noise, mu, log_var = model(
                atom_types, frac_coords, lattice, mask
            )
            
            # --- LOSS CALCULATION ---
            # 1. Reconstruction Loss (MSE between Predicted Noise and Real Noise)
            # We must normalize by the number of valid atoms (sum of mask)
            # Note: The output is already masked inside the model, but we double-check mask sum
            num_valid_atoms = torch.sum(mask)
            recon_loss = F.mse_loss(pred_noise, target_noise, reduction='sum') / (num_valid_atoms + 1e-6)
            
            # 2. KL Divergence (Regularization)
            # Analytical KL for Normal Distributions
            # sum(1 + log(var) - mu^2 - var)
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            # Normalize by batch size and apply weight
            kl_loss = (kld / atom_types.size(0)) * KL_WEIGHT
            
            # Total Loss - scale by accumulation steps to average over accumulated batches
            loss = (recon_loss + kl_loss) / GRAD_ACCUM_STEPS
            
            # --- BACKPROPAGATION (Accumulate Gradients) ---
            loss.backward()
            
            # --- LOGGING ---
            # Store loss values before deletion for logging
            recon_loss_val = recon_loss.item()
            kl_loss_val = kl_loss.item()
            total_recon_loss += recon_loss_val
            total_kl_loss += kl_loss_val
            batch_count += 1
            
            # Update every GRAD_ACCUM_STEPS batches or at the end of epoch
            is_accumulation_step = (batch_idx + 1) % GRAD_ACCUM_STEPS == 0
            is_last_batch = (batch_idx + 1) == len(dataloader)
            
            if is_accumulation_step or is_last_batch:
                # Clip Gradients (Crucial for GNN stability)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                
                # Take optimizer step (this uses accumulated gradients)
                optimizer.step()
            
            # --- MEMORY FIX 2: DELETE TENSORS ---
            # Explicitly delete heavy variables to free graph references
            del pred_noise, target_noise, mu, log_var, loss, recon_loss, kl_loss, kld
            
            # --- MEMORY FIX 3: Force Apple Silicon to clear the cache ---
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Update Progress Bar (show unscaled losses)
            pbar.set_postfix({
                'Recon': f"{recon_loss_val:.4f}", 
                'KL': f"{kl_loss_val:.4f}",
                'Accum': f"{(batch_idx + 1) % GRAD_ACCUM_STEPS}/{GRAD_ACCUM_STEPS}"
            })
            
        # End of Epoch Summary
        avg_recon = total_recon_loss / batch_count
        avg_kl = total_kl_loss / batch_count
        avg_total_loss = avg_recon + avg_kl
        print(f"   Done. Avg Recon: {avg_recon:.4f} | Avg KL: {avg_kl:.4f} | Avg Total Loss: {avg_total_loss:.4f}")
        
        # --- CHECKPOINTING ---
        # Save model every 10 epochs (and the first one)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"cdvae_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss,
            }, save_path)
            print(f"   💾 Saved checkpoint to {save_path}")

    print("\n✅ Training Complete.")

if __name__ == "__main__":
    train()