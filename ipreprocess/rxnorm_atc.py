import os
import shutil
import zipfile

import urllib.parse
import urllib.request
import tqdm
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import Counter, defaultdict
import pandas as pd
import json
import requests

# def load_latest_rxnorm_info():
#     print('********load_latest_rxnorm_info*********')
#     df = pd.read_csv(r'../ipreprocess/mapping/RXNORM.csv', dtype=str)  # (40157, 4)
#
#     rxnorm_name = {}  # len: 26978
#     for index, row in df.iterrows():
#
#         rxnorm = row[r'Class ID'].strip().split('/')[-1]
#         name = row[r'Preferred Label']
#         rxnorm_name[rxnorm] = name
#     print('df.shape:', df.shape, 'len(rxnorm_name):', len(rxnorm_name))
#     with open(r'pickles/rxnorm_label_mapping.pkl', 'wb') as f:
#         pickle.dump(rxnorm_name, f)
#     return rxnorm_name, df
#
#
# # RXNORM_to_ATC.text, NDC_to_ATC.text  from Jie
# def load_rxnorm_to_atc():
#     print('1-Build from ../ipreprocess/mapping/RXNORM_to_ATC.text')
#     rx_atc = pd.read_csv('../ipreprocess/mapping/RXNORM_to_ATC.text', sep='|', header=None, dtype=str)
#     atc2_rx = {}
#     rx_atc2 = {}
#     for index, row in tqdm(rx_atc.iterrows(), total=len(rx_atc)):
#         rx = row[3]
#         rx_name = row[9]
#         rx_ing = row[11]
#         atc4 = row[15]
#         atc_name = row[17]
#         if pd.notna(atc4):
#             atc2 = atc4[:3]
#             if atc2 in atc2_rx:
#                 # atc2_rx[atc2].add(rx_ing)
#                 atc2_rx[atc2].update([rx_ing,rx])
#
#             else:
#                 atc2_rx[atc2] = set([rx_ing, rx])
#
#             if rx_ing in rx_atc2:
#                 # assert atc2 == rx_atc2[rx_ing]
#                 rx_atc2[rx_ing].add(atc2)
#             else:
#                 rx_atc2[rx_ing] = set([atc2])
#
#             if rx in rx_atc2:
#                 # assert atc2 == rx_atc2[rx_ing]
#                 rx_atc2[rx].add(atc2)
#             else:
#                 rx_atc2[rx] = set([atc2])
#
#     print('len(rx_atc2)', len(rx_atc2))
#     print('len(atc2_rx)', len(atc2_rx))
#
#     print('2-Build from pickles/atc_rxnorm_pair.pkl')
#     with open(r'pickles/atc_rxnorm_pair.pkl', 'rb') as f:
#         atc_rxnorm_pair = pickle.load(f)
#     for atc, rx in tqdm(atc_rxnorm_pair):
#         atc2 = atc[:3]
#         if atc2 in atc2_rx:
#             atc2_rx[atc2].add(rx)
#         else:
#             atc2_rx[atc2] = set([rx])
#
#         if rx in rx_atc2:
#             # assert atc2 == rx_atc2[rx_ing]
#             rx_atc2[rx].add(atc2)
#         else:
#             rx_atc2[rx] = set([atc2])
#
#     print('len(rx_atc2)', len(rx_atc2))
#     print('len(atc2_rx)', len(atc2_rx))
#
#     print('3-../ipreprocess/mapping/NDC_to_ATC.text')
#     ndc_atc = pd.read_csv('../ipreprocess/mapping/NDC_to_ATC.text', sep='|', header=None, dtype=str)
#     for index, row in tqdm(ndc_atc.iterrows(), total=len(ndc_atc)):
#         rx_ing = row[15]
#         atc4 = row[19]
#         atc_name = row[21]
#         if pd.notna(atc4):
#             atc2 = atc4[:3]
#             if atc2 in atc2_rx:
#                 atc2_rx[atc2].add(rx_ing)
#                 # atc2_rx[atc2].update([rx_ing,rx])
#
#             else:
#                 atc2_rx[atc2] = set([rx_ing])
#
#             if rx_ing in rx_atc2:
#                 # assert atc2 == rx_atc2[rx_ing]
#                 rx_atc2[rx_ing].add(atc2)
#             else:
#                 rx_atc2[rx_ing] = set([atc2])
#
#     print('len(rx_atc2)', len(rx_atc2))
#     print('len(atc2_rx)', len(atc2_rx))
#
#     with open(r'pickles/rx_atc2.pkl', 'wb') as f:
#         pickle.dump(rx_atc2, f)
#     with open(r'pickles/atc2_rx.pkl', 'wb') as f:
#         pickle.dump(atc2_rx, f)


# Data source 1: https://bioportal.bioontology.org/ontologies/ATC, https://bioportal.bioontology.org/ontologies/RXNORM
def rxnorm_atc_from_bioportal():
    df_rx = pd.read_csv(r'mapping/RXNORM.csv', dtype=str)  # (40157, 4)
    df_atc = pd.read_csv(r'mapping/ATC.csv', dtype=str)  # 6567, CUI: 6440

    d = df_atc[pd.notnull(df_atc.CUI)].merge(df_rx[pd.notnull(df_rx.CUI)], on='CUI', how='inner') #, lsuffix='_atc', rsuffix='_rx')
    d.to_csv('debug/ATC_inner_RXNORM.csv')
    ra = set()
    for index, row in d.iterrows():
        atc = row[r'Class ID_x'].strip().split('/')[-1]
        rx = row[r'Class ID_y'].strip().split('/')[-1]
        name = row[r'Preferred Label_x'].strip()
        ra.add((rx, atc, name))

    df = pd.DataFrame(ra, columns=['rxnorm', 'atc', 'name']).sort_values(by='rxnorm', key=lambda x: x.astype(int))
    df.to_csv('debug/rxnorm_atc_from_bioportal.csv')
    return ra, df


# Data source 2: https://www.nlm.nih.gov/research/umls/rxnorm/index.html,
# https://www.nlm.nih.gov/research/umls/rxnorm/docs/techdoc.html#conso
# https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm=6883
def rxnorm_atc_from_NIH_UMLS():
    rx = pd.read_csv('mapping/RXNCONSO.RRF', sep='|',
                     header=None, dtype=str)
    atc = rx.loc[rx[11] == 'ATC']
    ra = set()
    for index, row in atc.iterrows():
        rx = row[0].strip()
        atc = row[13].strip()
        name = row[14]
        ra.add((rx, atc, name))

    df = pd.DataFrame(ra, columns=['rxnorm', 'atc', 'name']).sort_values(by='rxnorm', key=lambda x: x.astype(int))
    df.to_csv('debug/rxnorm_atc_from_NIH_UMLS.csv')
    return ra, df


# Data source 3: RXNORM_to_ATC.text from Jie
def rxnorm_atc_from_RXNORM_to_ATC_text():
    rx = pd.read_csv('mapping/RXNORM_to_ATC.text', sep='|',
                     header=None, dtype=str)
    ra = set()
    for index, row in rx.iterrows():
        if pd.isnull(row[15]):
            continue
        # try:
        rx = row[3].strip()
        name = row[9].strip()
        rx_ing = row[11].strip()
        atc = row[15].strip()
            # atc_name = row[17].strip()
        # except:
        #     print(index)
        #     print(row)
        ra.add((rx, atc, name))
        ra.add((rx_ing, atc, name))

    df = pd.DataFrame(ra, columns=['rxnorm', 'atc', 'name']).sort_values(by='rxnorm', key=lambda x: x.astype(int))
    df.to_csv('debug/rxnorm_atc_from_RXNORM_to_ATC_text.csv')
    return ra, df


# Data source 4: NDC_to_ATC.text from Jie
def rxnorm_atc_from_NDC_to_ATC_text():
    rx = pd.read_csv('mapping/NDC_to_ATC.text', sep='|',
                     header=None, dtype=str)
    ra = set()
    for index, row in rx.iterrows():
        if pd.isnull(row[19]):
            continue
        rx = row[15].strip()
        atc = row[19].strip()
        name = row[13].strip()

        ra.add((rx, atc, name))

    df = pd.DataFrame(ra, columns=['rxnorm', 'atc', 'name']).sort_values(by='rxnorm', key=lambda x: x.astype(int))
    df.to_csv('debug/rxnorm_atc_from_NDC_to_ATC_text.csv')
    return ra, df


# Data source 5: nih rxclass api https://rxnav.nlm.nih.gov/api-RxClass.getClassByRxNormDrugId.html
def _parse_from_nih_rxnorm_api(rxcui, name):
    r = requests.get('https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui={}&relaSource=ATC'.format(rxcui))
    data = r.json()
    ra = set()
    if ('rxclassDrugInfoList' in data) and ('rxclassDrugInfo' in data['rxclassDrugInfoList']):
        for x in data['rxclassDrugInfoList']['rxclassDrugInfo']:
            rx = x['minConcept']['rxcui']
            rx_name = x['minConcept']['name']
            atc = x['rxclassMinConceptItem']['classId']
            ra.add((rxcui, atc, name))
            ra.add((rx, atc, rx_name))
    return ra

# https://mor.nlm.nih.gov/RxNav/
def rxnorm_atc_from_nih_rxnorm_api():
    drug_in_data = pd.read_csv('debug/drug_prevalence_from_dispensing_plus_prescribing.csv', dtype=str)
    print('drug_in_data.shape: ', drug_in_data.shape)
    ra = set()
    n_total = 0
    n_find = 0
    for index, row in drug_in_data.iterrows():
        n_total += 1
        rx = row['Ingredient_RXNORM_CUI']
        label = row['label']
        n_patient_take = row['n_patient_take']
        n_patient_take_times = row['n_patient_take_times']
        # if rx not in rx_atc:
        ra_list = _parse_from_nih_rxnorm_api(rx, label)
        ra.update(ra_list)
        if ra_list:
            n_find += 1
        print('Find', rx, label, ra_list)

    print('Find {}/{} rxnom_cui have atc mapping'.format(n_find, n_total))
    with open(r'pickles/rxnorm_atc_from_nih_rxnorm_api.pkl', 'wb') as f:
        pickle.dump(ra, f)

    df = pd.DataFrame(ra, columns=['rxnorm', 'atc', 'name']).sort_values(by='rxnorm', key=lambda x: x.astype(int))
    df.to_csv('debug/rxnorm_atc_from_nih_rxnorm_api.csv')

    return ra, df


# https://www.whocc.no/atc/structure_and_principles/
def _add_one_record(atc_rx, rx_atc, record, atc_level=2):
    level_digit = {1:1, 2:3, 3:4, 4:5, 5:7}
    rx = record[0]
    atc = record[1][:level_digit.get(atc_level)]

    if atc in atc_rx:
        atc_rx[atc].add(rx)
    else:
        atc_rx[atc] = {rx}

    if rx in rx_atc:
        rx_atc[rx].add(atc)
    else:
        rx_atc[rx] = {atc}


def combine_rxnorm_atc_mapping(data_list, atc_level=2):
    print('len(data_list): ', len(data_list), 'Choose atc level:', atc_level)
    drug_in_data = pd.read_csv('debug/drug_prevalence_from_dispensing_plus_prescribing.csv', dtype=str)
    rx_in_data = set(drug_in_data['Ingredient_RXNORM_CUI'])
    print('drug_prevalence_from_dispensing_plus_prescribing.csv: len(rx_in_data):', len(rx_in_data))

    atc_rx = {}
    rx_atc = {}
    for index, data in enumerate(data_list):
        print('...', index, 'len(data):', len(data))
        for record in data:
            _add_one_record(atc_rx, rx_atc, record, atc_level=atc_level)
        print('...len(rx_atc)', len(rx_atc))
        print('...len(atc_rx)', len(atc_rx))
        print('... among ', len(rx_in_data), len(set(rx_in_data - rx_atc.keys())), 'is still missing')

    with open(r'pickles/rx_atcL{}.pkl'.format(atc_level), 'wb') as f:
        pickle.dump(rx_atc, f)
    with open(r'pickles/atcL{}_rx.pkl'.format(atc_level), 'wb') as f:
        pickle.dump(atc_rx, f)
    print('Done!')

    drug_atc_bool_info = []
    for index, row in drug_in_data.iterrows():
        rx = row['Ingredient_RXNORM_CUI']
        label = row['label']
        n_patient_take = row['n_patient_take']
        n_patient_take_times = row['n_patient_take_times']
        drug_atc_bool_info.append([rx, label, rx in rx_atc, n_patient_take, n_patient_take_times])

    df = pd.DataFrame(drug_atc_bool_info, columns=["Ingredient_RXNORM_CUI",
                                              "label",
                                              "hasATC",
                                              "n_patient_take",
                                              "n_patient_take_times"])
    df.to_csv('debug/rxnorm_hasorno_atc.csv')
    return atc_rx, rx_atc


if __name__ == '__main__':
    # Run first
    # d0, df0 = rxnorm_atc_from_nih_rxnorm_api()
    #
    d1, df1 = rxnorm_atc_from_bioportal()
    d2, df2 = rxnorm_atc_from_NIH_UMLS()
    d3, df3 = rxnorm_atc_from_RXNORM_to_ATC_text()
    d4, df4 = rxnorm_atc_from_NDC_to_ATC_text()
    with open(r'pickles/rxnorm_atc_from_nih_rxnorm_api.pkl', 'rb') as f:
        d0 = pickle.load(f)
    data_list = [d1, d2, d3, d4, d0]
    atc_rx, rx_atc = combine_rxnorm_atc_mapping(data_list, atc_level=2)

    print('Done!')
