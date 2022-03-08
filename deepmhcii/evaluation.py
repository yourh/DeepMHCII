#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/11/23
@author yrh

"""

import math
import csv
import numpy as np
from collections import namedtuple
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from logzero import logger

__all__ = ['CUTOFF', 'get_auc', 'get_pcc', 'get_srcc', 'get_group_metrics', 'output_res']

CUTOFF = 1.0 - math.log(500, 50000)
Metrics = namedtuple('Metrics', ['auc', 'pcc', 'srcc'])


def get_auc(targets, scores):
    return roc_auc_score(targets >= CUTOFF, scores)


def get_pcc(targets, scores):
    return np.corrcoef(targets, scores)[0, 1]


def get_srcc(targets, scores):
    return spearmanr(targets, scores)[0]


def get_group_metrics(mhc_names, targets, scores, reduce=True):
    mhc_names, targets, scores = np.asarray(mhc_names), np.asarray(targets), np.asarray(scores)
    mhc_groups, metrics = [], Metrics([], [], [])
    for mhc_name_ in sorted(set(mhc_names)):
        t_, s_ = targets[mhc_names == mhc_name_], scores[mhc_names == mhc_name_]
        if len(t_) > 30 and len(t_[t_ >= CUTOFF]) >= 3:
            mhc_groups.append(mhc_name_)
            metrics.auc.append(get_auc(t_, s_))
            metrics.pcc.append(get_pcc(t_, s_))
            metrics.srcc.append(get_srcc(t_, s_))
    return (np.mean(x) for x in metrics) if reduce else (mhc_groups,) + metrics


def output_res(mhc_names, targets, scores, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, scores)
    eval_out_path = output_path.with_suffix('.csv')
    mhc_names, targets, scores, metrics = np.asarray(mhc_names), np.asarray(targets), np.asarray(scores), []
    with open(eval_out_path, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['allele', 'total', 'positive', 'AUC', 'PCC', 'SRCC'])
        mhc_groups, auc, pcc, srcc = get_group_metrics(mhc_names, targets, scores, reduce=False)
        for mhc_name_, auc_, pcc_, srcc_ in zip(mhc_groups, auc, pcc, srcc):
            t_ = targets[mhc_names == mhc_name_]
            writer.writerow([mhc_name_, len(t_), len(t_[t_ >= CUTOFF]), auc_, pcc_, srcc_])
            metrics.append((auc_, pcc_, srcc_))
        metrics = np.mean(metrics, axis=0)
        writer.writerow([''] * 3 + metrics.tolist())
    logger.info(f'AUC: {metrics[0]:3f} PCC: {metrics[1]:3f} SRCC: {metrics[2]:3f}')
