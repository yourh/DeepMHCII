#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/11/23
@author yrh

"""

__all__ = ['ACIDS', 'get_mhc_name_seq', 'get_data', 'get_binding_data', 'get_seq2logo_data']

ACIDS = '0-ACDEFGHIKLMNPQRSTVWY'


def get_mhc_name_seq(mhc_name_seq_file):
    mhc_name_seq = {}
    with open(mhc_name_seq_file) as fp:
        for line in fp:
            mhc_name, mhc_seq = line.split()
            mhc_name_seq[mhc_name] = mhc_seq
    return mhc_name_seq


def get_data(data_file, mhc_name_seq):
    data_list = []
    with open(data_file) as fp:
        for line in fp:
            peptide_seq, score, mhc_name = line.split()
            if len(peptide_seq) >= 9:
                data_list.append((mhc_name, peptide_seq, mhc_name_seq[mhc_name], float(score)))
    return data_list


def get_binding_data(data_file, mhc_name_seq, peptide_pad=3, core_len=9):
    data_list = []
    with open(data_file) as fp:
        for line in fp:
            pdb, mhc_name, mhc_seq, peptide_seq, core = line.split()
            assert len(core) == core_len
            data_list.append(((pdb, mhc_name, core), peptide_seq, mhc_name_seq[mhc_name], 0.0))
    return data_list


def get_seq2logo_data(data_file, mhc_name, mhc_seq):
    with open(data_file) as fp:
        return [(mhc_name, line.strip(), mhc_seq, 0.0) for line in fp]
