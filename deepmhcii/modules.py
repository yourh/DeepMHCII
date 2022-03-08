#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/11/23
@author yrh

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepmhcii.init import truncated_normal_

__all__ = ['IConv', 'LinearAndOut']


class IConv(nn.Module):
    """

    """
    def __init__(self, out_channels, kernel_size, mhc_len=34, stride=1, **kwargs):
        super(IConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, kernel_size, mhc_len))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.stride, self.kernel_size = stride, kernel_size
        self.reset_parameters()

    def forward(self, peptide_x, mhc_x, **kwargs):
        bs = peptide_x.shape[0]
        kernel = F.relu(torch.einsum('nld,okl->nodk', mhc_x, self.weight))
        outputs = F.conv1d(peptide_x.transpose(1, 2).reshape(1, -1, peptide_x.shape[1]),
                           kernel.contiguous().view(-1, *kernel.shape[-2:]), stride=self.stride, groups=bs)
        return outputs.view(bs, -1, outputs.shape[-1]) + self.bias[:, None]

    def reset_parameters(self):
        truncated_normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)


class LinearAndOut(nn.Module):
    """

    """
    def __init__(self, linear_size):
        super(LinearAndOut, self).__init__()
        self.linear = nn.ModuleList(nn.Linear(in_s, out_s)
                                    for in_s, out_s in zip(linear_size[:-1], linear_size[1:]))
        self.output = nn.Linear(linear_size[-1], 1)
        self.reset_parameters()

    def forward(self, inputs):
        linear_out = inputs
        for linear in self.linear:
            linear_out = F.relu(linear(linear_out))
        return torch.sigmoid(torch.squeeze(self.output(linear_out), -1))

    def reset_parameters(self):
        for linear in self.linear:
            linear.reset_parameters()
            truncated_normal_(linear.weight, std=0.1)
        self.output.reset_parameters()
        truncated_normal_(self.output.weight, std=0.1)
