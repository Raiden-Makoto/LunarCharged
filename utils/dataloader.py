import torch #type: ignore
import os
import pickle
from torch.utils.data import Dataset, DataLoader #type: ignore
from torch.nn.utils.rnn import pad_sequence #type: ignore

class MOFDataset(Dataset):
    def __init__(self, data_dir):
        """
        Reads the .pkl files from your processed directory.
        """
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]
        print(f"Dataset loaded: {len(self.files)} crystals found.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            data = pickle.load(f)
        return data

def collate_mols(batch):
    """
    Custom function to stack crystals of different sizes into one batch.
    We 'pad' smaller crystals with zeros so they match the largest crystal in the batch.
    """
    # 1. Extract lists of features
    atom_types_list = [b['atom_types'] for b in batch]  # List of 1D tensors
    frac_coords_list = [b['frac_coords'] for b in batch] # List of 2D tensors (N x 3)
    lattices = torch.stack([b['lattice'] for b in batch]) # 3x3 matrices (always same size)
    mp_ids = [b['mp_id'] for b in batch]
    
    # 2. Pad them to the max size in this specific batch
    # padding_value=0 is safe because atomic number 0 doesn't exist (H=1)
    atom_types_padded = pad_sequence(atom_types_list, batch_first=True, padding_value=0)
    frac_coords_padded = pad_sequence(frac_coords_list, batch_first=True, padding_value=0)
    
    # 3. Create a "Mask" (Critical for the Model)
    # The model needs to know which atoms are real (1) and which are padding (0)
    lengths = torch.tensor([len(x) for x in atom_types_list])
    max_len = max(lengths)
    # Create a boolean mask: True where atoms exist, False where padded
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    
    return {
        'atom_types': atom_types_padded, # Shape: (Batch, Max_Atoms)
        'frac_coords': frac_coords_padded, # Shape: (Batch, Max_Atoms, 3)
        'lattice': lattices,             # Shape: (Batch, 3, 3)
        'mask': mask,                    # Shape: (Batch, Max_Atoms)
        'lengths': lengths,              # Shape: (Batch)
        'mp_ids': mp_ids
    }

def get_dataloader(data_dir, batch_size=32, shuffle=True):
    dataset = MOFDataset(data_dir)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_mols,
        num_workers=0, # Keep 0 on Mac to avoid multiprocessing fork errors
        pin_memory=True
    )