import os
import numpy as np
import networkx as nx
from pymatgen.core import Structure

# CONFIG
INPUT_DIR = "relaxed_batteries"  # Use your P1 relaxed structures
JUMP_CUTOFF = 4.0  # Max distance Li can jump (Angstroms). 
                   # 3.0-4.0 is typical for superionic conductors.

def check_percolation():
    print(f"⚡ Testing Lithium Percolation (Jump Limit: {JUMP_CUTOFF} Å)...")
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".cif")]
    
    for filename in files:
        try:
            struct = Structure.from_file(os.path.join(INPUT_DIR, filename))
            
            # 1. Get all Lithium indices
            li_indices = [i for i, site in enumerate(struct) if site.specie.symbol == "Li"]
            
            if len(li_indices) == 0:
                print(f"  {filename}: No Lithium found! (Not a battery)")
                continue

            # 2. Build the Connectivity Graph
            # We use a Supercell (3x3x3) to check if ions can move infinitely
            # otherwise a small box might look connected just because it's small.
            struct_super = struct * (3, 3, 3) 
            li_super_indices = [i for i, site in enumerate(struct_super) if site.specie.symbol == "Li"]
            
            # Get neighbors for every Li atom
            # "Is there another Li within jumping distance?"
            all_neighbors = struct_super.get_all_neighbors(JUMP_CUTOFF, include_index=True)
            
            G = nx.Graph()
            G.add_nodes_from(li_super_indices)
            
            for i, neighbors in enumerate(all_neighbors):
                if i not in li_super_indices: continue # Skip non-Li
                
                for neighbor in neighbors:
                    n_dist = neighbor.nn_distance
                    n_index = neighbor.index
                    
                    if n_index in li_super_indices:
                        G.add_edge(i, n_index, weight=n_dist)

            # 3. Analyze the Largest Cluster
            # If the largest connected cluster of Li atoms spans the whole 3x3x3 box,
            # it means we have a percolating network.
            
            clusters = list(nx.connected_components(G))
            largest_cluster = max(clusters, key=len)
            fraction_connected = len(largest_cluster) / len(li_super_indices)
            
            status = "❌ Insulator"
            if fraction_connected > 0.8: # If 80% of Li are connected
                status = "✅ SUPERIONIC CONDUCTOR"
            elif fraction_connected > 0.5:
                status = "⚠️ Poor Conductor"
                
            print(f"  {filename}: {status} (Connected Li: {fraction_connected*100:.1f}%)")
            
        except Exception as e:
            print(f"  {filename}: Error {e}")

if __name__ == "__main__":
    check_percolation()