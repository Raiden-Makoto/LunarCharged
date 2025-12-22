import os
import shutil
import warnings
from pymatgen.core import Structure #type: ignore

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SOURCE_DIR = "MOF_ASR"  # The folder you unzipped
TARGET_DIR = "biomedical_mofs"     # Where safe MOFs will go
SAFE_METALS = {"Fe", "Zn", "Zr", "Mg", "Ca", "Ti"}
# We also allow organic elements (C, H, O, N) and common halogens/non-metals (F, Cl, P, S, Br, I)
ALLOWED_NON_METALS = {"C", "H", "O", "N", "F", "Cl", "P", "S", "Br", "I"}

# Combine them for the master allow-list
ALLOWED_ELEMENTS = SAFE_METALS.union(ALLOWED_NON_METALS)

def is_biocompatible(structure):
    """
    Returns True if the structure ONLY contains allowed elements.
    """
    elements = set([e.symbol for e in structure.composition.elements])
    
    # Check if all elements in the MOF are in our allowed list
    if elements.issubset(ALLOWED_ELEMENTS):
        # DOUBLE CHECK: Ensure at least one metal is present (it must be a MOF)
        if not elements.isdisjoint(SAFE_METALS):
            return True
    return False

def preprocess():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    cif_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".cif")]
    print(f"Scanning {len(cif_files)} structures in {SOURCE_DIR}...")

    count = 0
    errors = 0

    for cif in cif_files:
        path = os.path.join(SOURCE_DIR, cif)
        try:
            # Load structure with Pymatgen
            struct = Structure.from_file(path)
            
            if is_biocompatible(struct):
                # Copy to target directory
                shutil.copy(path, os.path.join(TARGET_DIR, cif))
                count += 1
                if count % 100 == 0:
                    print(f"Found {count} biocompatible MOFs so far...")
                    
        except Exception as e:
            # Some CIFs in CoRE MOF are malformed; skip them
            errors += 1
            continue

    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Total Biocompatible MOFs Saved: {count}")
    print(f"Files failed to load: {errors}")
    print(f"Data ready in: {TARGET_DIR}")

if __name__ == "__main__":
    preprocess()