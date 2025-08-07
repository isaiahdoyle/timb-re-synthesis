import os
import torch
import pandas as pd
import numpy as np
from freevc import FreeVC

from torch.utils.data import Dataset

class TimbreDataset(Dataset):
    """Perceptual timbre dataset."""

    def __init__(
            self,
            path: str,
            root_dir: str,
            timbre_model: FreeVC,
            train: bool = True):
        """Dataset initialization.
        
        Args:
            path: Path to the labelled .xlsx file.
            root_dir: Directory containing the audio files.
            timbre_model: The FreeVC model to use for encoding timbre.
            train: Whether to use a training or testing portion of the dataset.
        """
        if train:
            self.dataframe = pd.read_excel(path).iloc[:500]
        else:
            self.dataframe = pd.read_excel(path).iloc[500:]

        self.root_dir = root_dir
        self.model = timbre_model

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        timbre = self.model.get_timbre(path, output=False).squeeze().to(dtype=torch.float)
        labels = self.dataframe.iloc[idx, 13:].to_numpy(dtype=np.float32)
        sample = {'timbre': timbre, 'labels': labels}

        return sample
