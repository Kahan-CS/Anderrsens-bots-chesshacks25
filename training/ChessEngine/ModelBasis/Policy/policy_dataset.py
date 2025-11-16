# policy_dataset_stream.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PolicyDatasetStream(Dataset):
    """
    Streams all chunk files as a single large dataset.
    """

    def __init__(self, chunk_dir):
        self.chunk_paths = sorted(
            os.path.join(chunk_dir, f)
            for f in os.listdir(chunk_dir)
            if f.endswith(".npy")
        )

        self.index_map = []
        offset = 0

        for path in self.chunk_paths:
            data = np.load(path, mmap_mode="r")
            length = len(data)
            self.index_map.extend((path, i) for i in range(length))

        self.length = len(self.index_map)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path, i = self.index_map[idx]
        data = np.load(path, allow_pickle=True)[i]

        x, policy = data
        x = torch.tensor(x, dtype=torch.float32)
        policy = torch.tensor(policy, dtype=torch.float32)

        return x, policy
