import os
from typing import List

import numpy as np
import torch
from sierraecg import read_file
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """Dataset for processing ECG files from HuggingFace dataset."""

    def __init__(self, data_dir: str, ecg_paths: List[str], mode: str = "12lead"):
        """
        Initialize ECG dataset.

        Args:
            data_dir: Directory containing ECG data
            ecg_paths: List of ecg paths
            mode: "1lead" or "12lead"
        """
        self.data_dir = data_dir
        self.ecg_paths = ecg_paths
        self.mode = mode
        self.input_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.new_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

        print(f"Created dataset with {len(self.ecg_paths)} ECG files")

    def __len__(self):
        return len(self.ecg_paths)

    def z_score_normalization(self, signal):
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # data = [wfdb.rdsamp(os.path.join(self.data_dir, self.ecg_paths[idx]))]
        # data = np.array([signal for signal, meta in data])
        # data = np.nan_to_num(data, nan=0)
        # result = self.check_nan_in_array(data)
        # if result != 0:
        #     print(self.ecg_paths[idx])
        # data = data.squeeze(0)
        # data = np.transpose(data, (1, 0))
        # data = data[self.lead_indices, :]
        # signal = self.z_score_normalization(data)
        # signal = torch.FloatTensor(signal)
        signal = []
        data = read_file(os.path.join(self.data_dir, self.ecg_paths[idx]))
        for lead in data.leads:
            signal.append(lead.samples)
        signal = np.array(signal)
        signal = self.z_score_normalization(signal)
        signal = torch.FloatTensor(signal)

        # Convert to torch tensors
        return {"ecg_data": signal, "ecg_path": self.ecg_paths[idx]}
