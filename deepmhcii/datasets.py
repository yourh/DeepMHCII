#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/11/23
@author yrh

"""

import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from deepmhcii.data_utils import ACIDS

__all__ = ['MHCIIDataset']


class MHCIIDataset(Dataset):
    """

    """
    def __init__(self, data_list, peptide_len=20, peptide_pad=3, mhc_len=34, padding_idx=0):
        self.mhc_names, self.peptide_x, self.mhc_x, self.targets = [], [], [], []
        for mhc_name, peptide_seq, mhc_seq, score in tqdm(data_list, leave=False):
            self.mhc_names.append(mhc_name)
            peptide_x = [ACIDS.index(x if x in ACIDS else '-') for x in peptide_seq][:peptide_len]
            self.peptide_x.append([padding_idx] * peptide_pad +
                                  peptide_x + [padding_idx] * (peptide_len - len(peptide_x)) +
                                  [padding_idx] * peptide_pad)
            assert len(self.peptide_x[-1]) == peptide_len + peptide_pad * 2
            self.mhc_x.append([ACIDS.index(x if x in ACIDS else '-') for x in mhc_seq])
            assert len(self.mhc_x[-1]) == mhc_len
            self.targets.append(score)
        self.peptide_x, self.mhc_x = np.asarray(self.peptide_x), np.asarray(self.mhc_x)
        self.targets = np.asarray(self.targets, dtype=np.float32)

    def __getitem__(self, item):
        return (self.peptide_x[item], self.mhc_x[item]), self.targets[item]

    def __len__(self):
        return len(self.mhc_names)
