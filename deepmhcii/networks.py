#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/11/23
@author yrh

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepmhcii.data_utils import ACIDS
from deepmhcii.modules import *
from deepmhcii.init import truncated_normal_

__all__ = ['DeepMHCII']


class Network(nn.Module):
    """

    """
    def __init__(self, *, emb_size, vocab_size=len(ACIDS), padding_idx=0, peptide_pad=3, mhc_len=34, **kwargs):
        super(Network, self).__init__()
        self.peptide_emb = nn.Embedding(vocab_size, emb_size)
        self.mhc_emb = nn.Embedding(vocab_size, emb_size)
        self.peptide_pad, self.padding_idx, self.mhc_len = peptide_pad, padding_idx, mhc_len

    def forward(self, peptide_x, mhc_x, *args, **kwargs):
        masks = peptide_x[:, self.peptide_pad: peptide_x.shape[1] - self.peptide_pad] != self.padding_idx
        return self.peptide_emb(peptide_x), self.mhc_emb(mhc_x), masks

    def reset_parameters(self):
        nn.init.uniform_(self.peptide_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.mhc_emb.weight, -0.1, 0.1)


class DeepMHCII(Network):
    """

    """
    def __init__(self, *, conv_num, conv_size, conv_off, linear_size, dropout=0.5, pooling=True, **kwargs):
        super(DeepMHCII, self).__init__(**kwargs)
        self.conv = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.conv_bn = nn.ModuleList(nn.BatchNorm1d(cn) for cn in conv_num)
        self.conv_off = conv_off
        self.dropout = nn.Dropout(dropout)
        linear_size = [sum(conv_num)] + linear_size
        self.linear = nn.ModuleList([nn.Conv1d(in_s, out_s, 1)
                                     for in_s, out_s in zip(linear_size[:-1], linear_size[1:])])
        self.linear_bn = nn.ModuleList([nn.BatchNorm1d(out_s) for out_s in linear_size[1:]])
        self.output = nn.Conv1d(linear_size[-1], 1, 1)
        self.pooling = pooling
        self.reset_parameters()

    def forward(self, peptide_x, mhc_x, pooling=None, **kwargs):
        peptide_x, mhc_x, masks = super(DeepMHCII, self).forward(peptide_x, mhc_x)
        conv_out = torch.cat([conv_bn(F.relu(conv(peptide_x[:, off: peptide_x.shape[1] - off], mhc_x)))
                              for conv, conv_bn, off in zip(self.conv, self.conv_bn, self.conv_off)], dim=1)
        conv_out = self.dropout(conv_out)
        for linear, linear_bn in zip(self.linear, self.linear_bn):
            conv_out = linear_bn(F.relu(linear(conv_out)))
        conv_out = self.dropout(conv_out)
        masks = masks[:, None, -conv_out.shape[2]:]
        if pooling or self.pooling:
            pool_out, _ = conv_out.masked_fill(~masks, -np.inf).max(dim=2, keepdim=True)
            return torch.sigmoid(self.output(pool_out).flatten())
        else:
            return torch.sigmoid(self.output(conv_out)).masked_fill(~masks, -np.inf).squeeze(1)

    def reset_parameters(self):
        super(DeepMHCII, self).reset_parameters()
        for conv, conv_bn in zip(self.conv, self.conv_bn):
            conv.reset_parameters()
            conv_bn.reset_parameters()
            nn.init.normal_(conv_bn.weight.data, mean=1.0, std=0.002)
        for linear, linear_bn in zip(self.linear, self.linear_bn):
            truncated_normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)
            linear_bn.reset_parameters()
            nn.init.normal_(linear_bn.weight.data, mean=1.0, std=0.002)
        truncated_normal_(self.output.weight, std=0.1)
        nn.init.zeros_(self.output.bias)
