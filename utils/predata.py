from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
import torch

def _n2t(arr, device="cuda:0"):
    return torch.tensor(arr).to(device=device)

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}


def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] - 1
    return X

class MyDataset(Dataset):
    def __init__(self, data_path, phase, max_seq_len, max_smi_len, seq_size, ):
        self.max_seq_len = max_seq_len
        self.max_smi_len = max_smi_len
        self.seq_size = seq_size

        data_path = Path(data_path)
        
        affinity = {}
        affinity_df = pd.read_csv(data_path / 'affinity_data.csv')
        for _, row in affinity_df.iterrows():
            affinity[row.iloc[0]] = row.iloc[1]
        self.affinity = affinity

        ligands_df = pd.read_csv(data_path / f"{phase}_smi.csv")
        ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        self.smi = ligands

        seq_path = data_path / phase / 'global'
        self.seq_path = sorted(list(seq_path.glob('*')))
        assert len(self.seq_path) == len(self.smi)
        self.length = len(self.smi)

    def __getitem__(self, idx):
        seq = self.seq_path[idx]

        _seq_tensor = pd.read_csv(seq, index_col=0).drop(['idx'], axis=1).values[:self.max_seq_len]
        seq_tensor = np.zeros((self.max_seq_len, self.seq_size))
        seq_tensor[:len(_seq_tensor)] = _seq_tensor

        return (
            label_smiles(self.smi[seq.name.split('.')[0]], self.max_smi_len),
            seq_tensor.astype(np.float32),
            np.array(self.affinity[seq.name.split('.')[0]], dtype=np.float32)
            )

    def __len__(self):
        return self.length

# ========================================== #
# --- إضافة كلاس الـ DDI الجديد هنا ---
# ========================================== #
class DDIDataset(Dataset):
    """
    Dataset class for Drug-Drug Interaction (DDI) prediction.
    يفترض أن ملف الـ CSV يحتوي على الأعمدة: SMILES1, SMILES2, Label
    """
    def __init__(self, ddi_csv_path, max_smi_len):
        self.max_smi_len = max_smi_len
        self.ddi_df = pd.read_csv(ddi_csv_path)
        self.length = len(self.ddi_df)

    def __getitem__(self, idx):
        row = self.ddi_df.iloc[idx]
        
        smi1_str = row['SMILES1']
        smi2_str = row['SMILES2']
        label = row['Label'] 

        # استخدام دالة التشفير الموجودة لتوحيد التمثيل
        smi1_tensor = label_smiles(smi1_str, self.max_smi_len)
        smi2_tensor = label_smiles(smi2_str, self.max_smi_len)

        return (
            smi1_tensor.astype(np.int64), 
            smi2_tensor.astype(np.int64),
            np.array(label, dtype=np.float32)
        )

    def __len__(self):
        return self.length


def process_addition_data():
    dataset_path = "C:\haq_project\bio_information\23AIBox-CSCo-DTA-main\data"
class DDIDataset(Dataset):
    def __init__(self, ddi_csv_path, max_smi_len):
        self.max_smi_len = max_smi_len
        self.ddi_df = pd.read_csv(ddi_csv_path)
        self.length = len(self.ddi_df)

    def __getitem__(self, idx):
        row = self.ddi_df.iloc[idx]
        smi1_str = row['SMILES1']
        smi2_str = row['SMILES2']
        label = row['Label'] 
        
        smi1_tensor = label_smiles(smi1_str, self.max_smi_len)
        smi2_tensor = label_smiles(smi2_str, self.max_smi_len)

        return (
            smi1_tensor.astype(np.int64), 
            smi2_tensor.astype(np.int64),
            np.array(label, dtype=np.float32)
        )

    def __len__(self):
        return self.length