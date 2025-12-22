import os
import pickle
import torch #type: ignore
import warnings
from pymatgen.core import Structure #type: ignore
from tqdm import tqdm #type: ignore

# --- CONFIG ---
INPUT_DIR = "biomedical_mofs"
OUTPUT_DIR = "processed_graphs"
# We still filter massive ones to save GPU memory later
MAX_ATOMS = 150

warnings.filterwarnings("ignore")

def process_mofs_atomic():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    cif_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".cif")]
    print(f"⚡️ Speed-Processing {len(cif_files)} MOFs (Atom-Level)...")

    saved_count = 0
    skipped_count = 0

    # List to store summary stats for normalization later
    all_num_atoms = []

    for cif_file in tqdm(cif_files):
        cif_path = os.path.join(INPUT_DIR, cif_file)
        mof_id = cif_file.replace(".cif", "")
        
        try:
            # 1. Load Structure
            struct = Structure.from_file(cif_path)
            
            # 2. Filter by Size (Critical for VAE memory usage)
            if len(struct) > MAX_ATOMS:
                skipped_count += 1
                continue
            
            # 3. Extract Raw Tensors (The Input for CDVAE)
            # atom_types: Tensor of atomic numbers (e.g., [6, 1, 8, 30...])
            atom_types = torch.tensor([site.specie.number for site in struct], dtype=torch.long)
            
            # frac_coords: Fractional coordinates (0 to 1) relative to box
            frac_coords = torch.tensor(struct.frac_coords, dtype=torch.float32)
            
            # lattice: 3x3 matrix defining the unit cell box
            lattice = torch.tensor(struct.lattice.matrix, dtype=torch.float32)
            
            # 4. Save
            data_packet = {
                "mp_id": mof_id,
                "atom_types": atom_types,
                "frac_coords": frac_coords,
                "lattice": lattice,
                "num_atoms": len(struct)
            }
            
            # Save as individual pickle files (standard for lazy loading)
            with open(os.path.join(OUTPUT_DIR, f"{mof_id}.pkl"), "wb") as f:
                pickle.dump(data_packet, f)
            
            saved_count += 1
            all_num_atoms.append(len(struct))
            
        except Exception as e:
            print(f"Error reading {mof_id}: {e}")

    print("\n" + "="*30)
    print(f"✅ Ready for VAE: {saved_count} structures")
    print(f"⏭  Skipped (> {MAX_ATOMS} atoms): {skipped_count}")
    print(f"Data saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    process_mofs_atomic()