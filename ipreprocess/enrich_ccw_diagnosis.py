import os
import shutil
import zipfile

import urllib.parse
import urllib.request

import torch
import torch.utils.data
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import Counter, defaultdict
import pandas as pd
import json


# def load_icd_to_ccw(path='mapping/CCW_to_use.json'):
#     """ e.g. there exisit E93.50 v.s. E935.0.
#             thus 5 more keys in the dot-ICD than nodot-ICD keys
#             [('E9350', 2),
#              ('E9351', 2),
#              ('E8500', 2),
#              ('E8502', 2),
#              ('E8501', 2),
#              ('31222', 1),
#              ('31200', 1),
#              ('3124', 1),
#              ('F919', 1),
#              ('31281', 1)]
#     :param path:
#     :return:
#     """
#     with open(path) as f:
#         data = json.load(f)
#         name_id = {x: str(i) for i, x in enumerate(data.keys())}
#         n_dx = 0
#         icd_ccwid = {}
#         icddot_ccwid = {}
#         for name, dx in data.items():
#             n_dx += len(dx)
#             for icd in dx:
#                 icd_ccwid[icd.strip().replace('.', '')] = name_id.get(name)
#                 icddot_ccwid[icd] = name_id.get(name)
#
#         return icd_ccwid, name_id, icddot_ccwid, data


def dump_ad_risk_codes_to_json():
    dfs = pd.read_excel(r'mapping/AD Prediction ICD Codes_error_corrected.xlsx', sheet_name=None, dtype=str)
    print(len(dfs))

    # test
    # icd_to_ccw, ccwname_id, icddot_to_ccw, data = load_icd_to_ccw('mapping/CCW_to_use.json')
    # ccw_ob = set([x.strip().replace('.', '') for x in data['Obesity']])
    # bian_ob = set(dfs['Obesity']['ICD-10 Code']) | set(dfs['Obesity']['ICD-9 Code'])

    dfs_json = {}
    comorbidity_names = list(dfs.keys())
    for i, name in enumerate(comorbidity_names):
        icd9 = list(dfs[name]['ICD-9 Code'].loc[dfs[name]['ICD-9 Code'].notnull()])
        icd10 = list(dfs[name]['ICD-10 Code'].loc[dfs[name]['ICD-10 Code'].notnull()])
        codes = list(set(icd9 + icd10))
        print(i, name, len(codes), len(icd9), len(icd10))
        dfs_json[name] = codes

    with open("mapping/ad_prediction_icd_codes.json", "w") as outfile:
        json.dump(dfs_json, outfile, indent=4)

    return dfs_json


def enrich_ccw_codes_and_exclude_ad():
    enriched_ccw = {}
    with open('mapping/ad_prediction_icd_codes.json') as f:
        ad_predict = json.load(f)
    enriched_names = {'Sleep':'Sleep disorders',
                      'Periodontitis':'Periodontitis',
                      'Menopause':'Menopause'}
    exclude_names = ["Alzheimer's Disease", "Alzheimer's Disease and Related Disorders or Senile Dementia"]
    print('Enrich: ', enriched_names)
    print('Exclude AD:', exclude_names)
    with open('mapping/CCW_to_use.json') as f:
        ccw = json.load(f)
        print('len(ccw): ', len(ccw))
    for name, dx in ccw.items():
        codes = set([])
        for icd in dx:
            codes.add(icd.strip().replace('.', ''))
        if name not in exclude_names:
            enriched_ccw[name] = sorted(list(codes))

    for key, val in enriched_names.items():
        enriched_ccw[val] = ad_predict[key]

    print('len(enriched_ccw): ', len(enriched_ccw))
    with open("mapping/CCW_to_use_enriched.json", "w") as outfile:
        json.dump(enriched_ccw, outfile, indent=4)

    return enriched_ccw


def ccw_ad_comorbidity_codes():
    with open('mapping/CCW_to_use_enriched.json') as f:
        ccw = json.load(f)
        print('len(ccw): ', len(ccw))
    comorbidity_names = [
        "Obesity", "Diabetes",
        "Hyperlipidemia", "Hypertension",
        "Heart Failure", "Ischemic Heart Disease",
        "Stroke / Transient Ischemic Attack", "Depression",
        "Anxiety Disorders", "Traumatic Brain Injury and Nonpsychotic Mental Disorders due to Brain Damage",
        'Sleep disorders', 'Periodontitis',
        'Menopause', "Tobacco Use",
        "Alcohol Use Disorders"]
    print('len(comorbidity_names):', len(comorbidity_names))
    selected_ccw = {}
    for key, val in ccw.items():
        if key in comorbidity_names:
            selected_ccw[key] = val

    print('len(selected_ccw): ', len(selected_ccw))
    with open("mapping/CCW_AD_comorbidity.json", "w") as outfile:
        json.dump(selected_ccw, outfile, indent=4)

    return selected_ccw


if __name__ == '__main__':
    dfs_json = dump_ad_risk_codes_to_json()
    enriched_ccw = enrich_ccw_codes_and_exclude_ad()
    selected_ccw = ccw_ad_comorbidity_codes()
