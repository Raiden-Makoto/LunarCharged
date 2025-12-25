import os
import warnings
from chgnet.model import StructOptimizer # <-- The correct class for moving atoms
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Suppress warnings
warnings.filterwarnings("ignore")

# CONFIG
INPUT_DIR = "generated_batteries"
OUTPUT_DIR = "relaxed_batteries"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def relax_crystals():
    print("🔋 Loading CHGNet Optimizer...")
    # This loads the physics engine specifically designed for relaxation
    relaxer = StructOptimizer()
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".cif")]
    print(f"Found {len(files)} candidates to polish.")

    for i, filename in enumerate(files):
        try:
            # 1. Load the "Messy" VAE Crystal
            file_path = os.path.join(INPUT_DIR, filename)
            struct = Structure.from_file(file_path)
            
            # 2. Relax Geometry
            print(f"[{i+1}/{len(files)}] Relaxing {filename}...")
            
            # The .relax() method handles the loop to minimize forces
            result = relaxer.relax(
                struct, 
                fmax=0.05,  # Stop when forces are < 0.05 eV/A
                steps=500,  # Max steps
                verbose=False
            )
            
            final_struct = result['final_structure']
            # trajectory = result['trajectory'] # Optional: See how it moved
            final_energy = result['trajectory'].energies[-1] / len(final_struct)
            
            # 3. Check for Symmetry (Optional Polish)
            try:
                analyzer = SpacegroupAnalyzer(final_struct, symprec=0.1)
                symmetrized_struct = analyzer.get_symmetrized_structure()
                sg_symbol = analyzer.get_space_group_symbol()
            except:
                sg_symbol = "P1" # No symmetry found
                
            # 4. Save
            out_name = f"relaxed_{filename.replace('.cif', '')}_{sg_symbol}.cif"
            CifWriter(final_struct).write_file(os.path.join(OUTPUT_DIR, out_name))
            
            print(f"  --> Done! Energy: {final_energy:.3f} eV/atom | Symmetry: {sg_symbol}")

        except Exception as e:
            print(f"  --> Failed on {filename}: {e}")

if __name__ == "__main__":
    relax_crystals()