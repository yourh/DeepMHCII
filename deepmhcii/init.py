#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/11/26
@author yrh

"""

import torch

__all__ = ['truncated_normal_']


@torch.no_grad()
def truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
