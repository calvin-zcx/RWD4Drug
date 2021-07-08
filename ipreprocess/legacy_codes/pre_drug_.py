import pandas as pd
import time
from collections import defaultdict
import re
import pickle
import argparse
import csv
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import math
import itertools
import os
import scipy
import numpy as np
from datetime import datetime
import copy
from utils import str_to_datetime


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--input_dispense', default=r'DELETE-ADD-LATER\DISPENSING.csv',
                        help='input data MCI_DISPENSING.csv directory')
    parser.add_argument('--input_prescribe', default=r'DELETE-ADD-LATER\PRESCRIBING.csv',
                        help='input data MCI_PRESCRIBING.csv directory')
    args = parser.parse_args()
    return args


# %% Build NDC / RXNORM to ingredients mapping
def combine_two_dict(d1, d2):
    """
    Combine two dicts with same semantics of key and value,
    No nan as keys or values of any dicts
    Check consistency
    """
    print('***combine_two_dict:')
    k_v = {}
    for d in [d1, d2]:
        print('len(d): ', len(d))
        for key, value in d.items():
            # build ndc_ing
            assert not pd.isnull(value)
            assert not pd.isnull(key)

            if key in k_v:
                # check consistency
                if value != k_v[key]:
                    print('inconsistency! Key: ', key, 'value != k_v[key]', value, k_v[key])
            else:
                k_v[key] = value
    print('len(d1): ', len(d1), '+', 'len(d2): ', len(d2), '--> len(k_v): ', len(k_v))
    return k_v


def clean_str_list(a):
    # a = a.replace('\'', '')
    # a = a.replace('"', '')
    # a = a.replace('[', '')
    # a = a.replace(']', '')
    # a = re.sub(r"\s+", "", a, flags=re.UNICODE)

    a = re.sub(r"['\"\[\]\s+]", "", a, flags=re.UNICODE)
    a = a.split(',')
    a = [x for x in a if len(x) > 0]
    return a


def load_ndc_to_ingredient():
    # read from 2 files and build ndc_to_ingredient mappings
    # no nan as values of any dictionary
    print('********load_ndc_to_ingredient*********')
    df_map1 = pd.read_csv(r'mapping/NDC_RXNorm_mapping.csv', dtype=str)  # (40157, 4)
    df_map2 = pd.read_csv(r'mapping/RXNORM_Ingredient_mapping.csv', dtype=str)  # (19171, 4)
    df_map2['NDC'] = df_map2['NDC'].apply(clean_str_list)

    ndc_ing_1 = {}  # len: 26978
    n_null_ing_1 = 0
    for index, row in df_map1.iterrows():
        # NDC_RXNorm_mapping.csv:
        #       NDC	ingredient_code	rxnrom
        # 0	68462041438
        # 1	11523716001	28889	206805
        # 2	65862042001	10180	198335
        ndc = row['NDC']
        rxcui = row['rxnrom']
        ing = row['ingredient_code']

        # No nan value record.
        if pd.isnull(ing):
            n_null_ing_1 += 1
            continue

        if ndc in ndc_ing_1:
            # check inconsistency:
            # seems no duplicated ingredients
            if ing != ndc_ing_1[ndc]:
                print('inconsistency ing != ndc_rx1[ndc]:', ing, ndc_ing_1[ndc])
        else:
            ndc_ing_1[ndc] = ing

    ndc_ing_2 = {}  # len:
    n_null_ing_2 = 0
    for index, row in df_map2.iterrows():
        ndc = row['NDC']
        rxcui = row['RXNORM_CUI']
        ing = row['ingredient_code']

        # No nan value record.
        if pd.isnull(ing):
            n_null_ing_2 += 1
            continue

        for x in ndc:
            if x in ndc_ing_2:
                # check inconsistency:
                # seems no duplicated ingredients
                if ing != ndc_ing_2[x]:
                    print('inconsistency ing != ndc_rx1[ndc]:', ing, ndc_ing_1[x])
            else:
                ndc_ing_2[x] = ing
    print("NDC_RXNorm_mapping.csv:\n",
          'len(df_map1): ', len(df_map1),
          'n_null_ing_1: ', n_null_ing_1,
          'len(ndc_ing_1): ', len(ndc_ing_1))

    print("RXNORM_Ingredient_mapping.csv:\n",
          'len(df_map2): ', len(df_map2),
          'n_null_ing_2: ', n_null_ing_2,
          'len(ndc_ing_2): ', len(ndc_ing_2))
    return ndc_ing_1, ndc_ing_2


def load_rxnorm_to_ingredient():
    """
    Read from 2 files and build rxnorm_to_ingredient mappings
    No nan as keys or values of any dictionary
    :return: two dicts: rxnorm_ing_1, rxnorm_ing_2
    """
    print('********load_rxnorm_to_ingredient*********')
    df_map1 = pd.read_csv(r'mapping/NDC_RXNorm_mapping.csv', dtype=str)  # (40157, 4)
    df_map2 = pd.read_csv(r'mapping/RXNORM_Ingredient_mapping.csv', dtype=str)  # (19171, 4)
    df_map2['NDC'] = df_map2['NDC'].apply(clean_str_list)

    rxnorm_ing_1 = {}  # len: 26978
    n_null_rxOring_1 = 0
    for index, row in df_map1.iterrows():
        # NDC_RXNorm_mapping.csv:
        #       NDC	ingredient_code	rxnrom
        # 0	68462041438
        # 1	11523716001	28889	206805
        # 2	65862042001	10180	198335
        ndc = row['NDC']
        rxnorm = row['rxnrom']
        ing = row['ingredient_code']

        # No nan value record.
        if pd.isnull(rxnorm) or pd.isnull(ing):
            n_null_rxOring_1 += 1
            continue

        if rxnorm in rxnorm_ing_1:
            # check inconsistency:
            # seems no duplicated ingredients, but many dumplicated rxnorm, because different NDCs may have same rxnorm
            if ing != rxnorm_ing_1[rxnorm]:
                print('inconsistency ing != rxnorm_ing_1[rxnrom]:', ing, rxnorm_ing_1[rxnorm])
        else:
            rxnorm_ing_1[rxnorm] = ing

    rxnorm_ing_2 = {}  # len:
    n_null_ing_2 = 0
    for index, row in df_map2.iterrows():
        # RXNORM_Ingredient_mapping.csv
        # 	RXNORM_CUI	ingredient_code	NDC
        # 0	1092360	69036	['62856058446']
        # 1	197407	1514	['00168004015', '00168004046', '00472037015', ...]
        # 2	1741423	828529	['67467062303', '68982062303']
        ndc = row['NDC']
        rxnorm = row['RXNORM_CUI']
        ing = row['ingredient_code']

        # No nan value record.
        if pd.isnull(ing):
            n_null_ing_2 += 1
            continue

        if rxnorm in rxnorm_ing_2:
            # check inconsistency:
            # seems no duplicated ingredients
            if ing != rxnorm_ing_2[rxnorm]:
                print('inconsistency ing != rxnorm_ing_2[rxnrom]:', ing, rxnorm_ing_2[rxnorm])
        else:
            rxnorm_ing_2[rxnorm] = ing

    print("NDC_RXNorm_mapping.csv:\n",
          'len(df_map1): ', len(df_map1),
          'n_null_rxOring_1: ', n_null_rxOring_1,
          'len(rxnorm_ing_1): ', len(rxnorm_ing_1))

    print("RXNORM_Ingredient_mapping.csv:\n",
          'len(df_map2): ', len(df_map2),
          'n_null_ing_2: ', n_null_ing_2,
          'len(rxnorm_ing_2): ', len(rxnorm_ing_2))
    return rxnorm_ing_1, rxnorm_ing_2


def generate_and_dump_drug_mappings_to_ingredients():
    # 1. combine drugs from NDC_RXNorm_mapping.csv and RXNORM_Ingredient_mapping.csv
    # 2. translate both NDC and RXNORM to their active ingredients
    ndc_ing_1, ndc_ing_2 = load_ndc_to_ingredient()
    ndc_to_ing = combine_two_dict(ndc_ing_2, ndc_ing_1)

    rxnorm_ing_1, rxnorm_ing_2 = load_rxnorm_to_ingredient()
    rxnorm_to_ing = combine_two_dict(rxnorm_ing_2, rxnorm_ing_1)

    with open(r'pickles/ndc_to_ingredient.pickle', 'wb') as f:
        pickle.dump(ndc_to_ing, f, pickle.HIGHEST_PROTOCOL)
    print(r'dump pickles/ndc_to_ingredient.pickle done!')

    with open(r'pickles/rxnorm_to_ingredient.pickle', 'wb') as f:
        pickle.dump(rxnorm_to_ing, f, pickle.HIGHEST_PROTOCOL)
    print(r'dump pickles/rxnorm_to_ingredient.pickle done!')


def load_drug_mappings_to_ingredients():
    with open(r'pickles/ndc_to_ingredient.pickle', 'rb') as f:
        ndc_to_ing = pickle.load(f)
    print(r'Load pickles/ndc_to_ingredient.pickle done：')
    print('***len(ndc_to_ing): ', len(ndc_to_ing))
    print('***unique ingredients: len(set(ndc_to_ing.values())): ', len(set(ndc_to_ing.values())))

    with open(r'pickles/rxnorm_to_ingredient.pickle', 'rb') as f:
        rxnorm_to_ing = pickle.load(f)
    print(r'Load pickles/rxnorm_to_ingredient.pickle done!')
    print('***len(rxnorm_to_ing): ', len(rxnorm_to_ing))
    print('***unique ingredients: len(set(rxnorm_to_ing.values())): ', len(set(rxnorm_to_ing.values())))

    print('unique ingredients of union set(ndc_to_ing.values()) | set(rxnorm_to_ing.values()):  ', len(
        set(ndc_to_ing.values()) | set(rxnorm_to_ing.values())
    ))
    return ndc_to_ing, rxnorm_to_ing


# %% Addtional test, deprecated
def _test_load_drug_code_mappinng():
    # combine two mappings into  two NDC -> [(rxnorm, ingredient)]
    # and then check consistency
    df_map1 = pd.read_csv(r'mapping/NDC_RXNorm_mapping.csv', dtype=str)
    df_map2 = pd.read_csv(r'mapping/RXNORM_Ingredient_mapping.csv', dtype=str)
    df_map2['NDC'] = df_map2['NDC'].apply(clean_str_list)

    # hypothesis: df_map2 has more NDC than df_map1
    # hypothesis: they are consistent w.r.t the rxnorm and ingredients

    ndc_rxing = {}  # len: 169029
    ndc_list = []  # len: 169029
    for index, row in df_map2.iterrows():
        ndc = row['NDC']
        ndc_list.extend(ndc)
        rxcui = row['RXNORM_CUI']
        ing = row['ingredient_code']

        for x in ndc:
            if x in ndc_rxing:
                ndc_rxing[x].append((rxcui, ing))
            else:
                ndc_rxing[x] = [(rxcui, ing)]

    ndc_rxing_less = {}  # len: 40157
    ndc_list_less = []  # len: 40157
    for index, row in df_map1.iterrows():
        ndc = row['NDC']
        ndc_list_less.append(ndc)
        rxcui = row['rxnrom']
        ing = row['ingredient_code']

        if ndc in ndc_rxing_less:
            ndc_rxing_less[ndc].append((rxcui, ing))
        else:
            ndc_rxing_less[ndc] = [(rxcui, ing)]

    return ndc_rxing, ndc_rxing_less, ndc_list, ndc_list_less


def _check_code_consistency():
    ndc_rxing, ndc_rxing_less, ndc_list, ndc_list_less = _test_load_drug_code_mappinng()
    # 1. length: consistency checked
    print('len(ndc_rxing): ', len(ndc_rxing))
    print('len(ndc_rxing_less): ', len(ndc_rxing_less))
    print('len(ndc_list):', len(ndc_list))
    print('len(ndc_list_less):', len(ndc_list_less))
    assert len(ndc_rxing) == len(ndc_list)
    assert len(ndc_rxing_less) == len(ndc_list_less)

    # 2. check multiple rxnorm and ing: no multiple ingredients consistency checked
    for key, value in ndc_rxing_less.items():
        if len(value) > 1:
            print(key, value, 'more than 1 in ndc_rxing_less')
    for key, value in ndc_rxing.items():
        if len(value) > 1:
            print(key, value, 'more than 1 in ndc_rxing')

    # 3. check consistency of rxnorm and ing:
    # no inconsistency checked
    # but there are missing values, e.g.:
    #   68462041438 [(nan, nan)] is not in ndc_rxing
    #   65162099308 [('2110780', '4083')] is not in ndc_rxing
    #   46122006765[('637121', '6750')] is not in ndc_rxing
    for key, value in ndc_rxing_less.items():
        if key not in ndc_rxing:
            print(key, value, 'is not in ndc_rxing')
        else:
            rx1, ing1 = value[0]
            rx2, ing2 = ndc_rxing[key][0]
            if (pd.isnull(rx1) and not pd.isnull(rx2)) or (not pd.isnull(rx1) and pd.isnull(rx2)) or rx1 != rx2:
                print(rx1, ing1, rx2, ing2, 'not consistency')

            if (pd.isnull(ing1) and not pd.isnull(ing2)) or (not pd.isnull(ing1) and pd.isnull(ing2)) or ing1 != ing2:
                print(rx1, ing1, rx2, ing2, 'not consistency')

    print('end test')


# %% Build patients drug table
def pre_drug_table_from_dispensing(args):
    """
    PATID	DISPENSE_DATE	NDC	DISPENSE_SUP
    :param args:
    :return: mci_drug_taken_by_patient
    """
    print('*******pre_drug_table_from_dispensing********:')
    ndc_to_ing, rxnorm_to_ing = load_drug_mappings_to_ingredients()
    mci_drug_taken_by_patient = defaultdict(dict)
    dispensing_file = args.input_dispense
    prescribing_file = args.input_prescribe

    # Load from Dispensing table
    print('Open file: ', dispensing_file, flush=True)
    n_records = 0
    n_row_not_in_map = 0
    n_no_day_supply = 0

    with open(dispensing_file, 'r') as f:
        # col_name = next(f)
        # for row in f:
        #     row = row.strip('\n')
        #     row = row.split(',')
        col_name = next(csv.reader(f))  # read from first non-name row, above code from 2nd, wrong
        for row in csv.reader(f):
            n_records += 1
            patid, date, ndc, day = row[1], row[3], row[4], row[6]
            if patid.startswith('--'):
                print('Skip {}th row: {}'.format(n_records, row))
                continue

            if date and day and (date != 'NULL') and (day != 'NULL'):
                day = int(float(day))
                if ndc in ndc_to_ing:
                    ing = ndc_to_ing.get(ndc)
                    if ing not in mci_drug_taken_by_patient:
                        mci_drug_taken_by_patient[ing][patid] = set([(date, day)])
                    else:
                        if patid not in mci_drug_taken_by_patient.get(ing):
                            mci_drug_taken_by_patient[ing][patid] = set([(date, day)])
                        else:
                            mci_drug_taken_by_patient[ing][patid].add((date, day))
                else:
                    n_row_not_in_map += 1
            else:
                n_no_day_supply += 1

    print('n_records: ', n_records)
    print('n_no_day_supply: ', n_no_day_supply)
    print('n_row_not_in_map: ', n_row_not_in_map)
    print('finish dump', flush=True)
    print('Scan n_records: ', n_records)
    print('# of Drugs: {}\t'.format(len(mci_drug_taken_by_patient)))

    try:
        print('dumping...', flush=True)
        pickle.dump(mci_drug_taken_by_patient,
                    open('pickles/mci_drug_taken_by_patient_from_dispensing.pkl', 'wb'))
    except Exception as e:
        print(e)

    print('dump pickles/mci_drug_taken_by_patient_from_dispensing.pkl done!')
    return mci_drug_taken_by_patient


def pre_drug_table_from_prescribing(args):
    """
    Caution: can have "," cases, can not simply use row-wise .split(',') strategy:
    better load at once by read_csv or preprocessing raw data

    e.g. : from old MCI data
    889803      morphine sulfate injection,-
    889815           meperidine injection, -
    889848         vancomycin  IV piggyback,
    889898           lidocaine 2% injection,
    889935     EPINEPHrine 1:1,000 injection
                           ...
    4386834           multivitamin, prenatal
    4392070              emollients, topical
    4396984              emollients, topical
    4397433              emollients, topical
    4397434              emollients, topical

    for big file:
    import re
    cStr = '"aaaa","bbbb","ccc,ddd"'
    newStr = re.split(r',(?=")', cStr)

    for small one, just read_csv
    PATID	RX_ORDER_DATE RX_DAYS_SUPPLY RXNORM_CUI RAW_RX_MED_NAME RAW_RXNORM_CUI

    e.g. : from New MCI data
    	    index	        1
        0	PRESCRIBINGID	11eab4b479393f72b4e70050569ea8fb
        1	PATID	11e827a2d4330c5691410050569ea8fb
        2	ENCOUNTERID	cfNPcr8ET1Kgrw27
        3	RX_PROVIDERID	cfNP
        4	RX_ORDER_DATE	2020-03-07
        5	RX_ORDER_TIME	00:00
        6	RX_START_DATE	2020-03-07
        7	RX_END_DATE	NaN
        8	RX_DOSE_ORDERED	NaN
        9	RX_DOSE_ORDERED_UNIT	NaN
        10	RX_QUANTITY	NaN
        11	RX_DOSE_FORM	NaN
        12	RX_REFILLS	NaN
        13	RX_DAYS_SUPPLY	5.0
        14	RX_FREQUENCY	NI
        15	RX_PRN_FLAG	N
        16	RX_ROUTE	OT
        17	RX_BASIS	NaN
        18	RXNORM_CUI	798230
        19	RX_SOURCE	OD
        20	RX_DISPENSE_AS_WRITTEN	NaN
        21	RAW_RX_MED_NAME	PNEUMOCOCCAL 13-VAL CONJ VACC IM SUSP
        22	RAW_RX_FREQUENCY	PRIOR TO DISCHARGE
        23	RAW_RXNORM_CUI	798230
        24	RAW_RX_QUANTITY	NaN
        25	RAW_RX_NDC	NaN
        26	RAW_RX_DOSE_ORDERED	NaN
        27	RAW_RX_DOSE_ORDERED_UNIT	NaN
        28	RAW_RX_ROUTE	NaN
        29	RAW_RX_REFILLS	NaN
        30	UPDATED	Jun 22 2020 2:16PM
        31	SOURCE	UMI
        32	RAW_RX_QUANTITY_UNIT	NaN
    :param args: file path
    :return: mci_drug_taken_by_patient
    """
    print('*******pre_drug_table_from_prescribing********:')
    ndc_to_ing, rxnorm_to_ing = load_drug_mappings_to_ingredients()
    mci_drug_taken_by_patient = defaultdict(dict)
    dispensing_file = args.input_dispense
    prescribing_file = args.input_prescribe

    # Load from prescribing table
    print('Open file: ', prescribing_file, flush=True)
    n_no_day_supply = 0
    n_records = 0
    n_row_not_in_map = 0
    n_day_supply_exist = 0
    n_day_supply_impute = 0
    with open(prescribing_file, 'r') as f:
        # col_name = next(f)
        col_name = next(csv.reader(f))  # may have , in quote, name column
        for row in csv.reader(f):
            n_records += 1
            patid, order_date, rx, rx_raw, day, name = row[1], row[4], row[18], row[23], row[13], row[21]
            start_date, end_date = row[6], row[7]
            if patid.startswith('--'):
                print('Skip {}th row: {}'.format(n_records, row))
                continue
            # if n_records == 889803 + 1:
            #     name: morphine sulfate injection,-
            #     print(row)

            if (start_date != '') and (start_date != 'NULL'):
                date = start_date
            else:
                date = order_date

            # day may be ''
            if (day == '') or (day == 'NULL'):
                # a lot of drugs, e.g. sodium, are not informative
                # discard this part?
                # keep no day supply as -1, and can discard later
                n_no_day_supply += 1
                # impute code here
                if (end_date != '') and (end_date != 'NULL'):
                    sup_day = (str_to_datetime(end_date) - str_to_datetime(date)).days
                    if sup_day >= 0:
                        day = sup_day  # str(sup_day)
                        n_day_supply_impute += 1
                    else:
                        day = -1  #'-1'
                else:
                    day = -1   # '-1'
            else:
                day = int(float(day))
                n_day_supply_exist += 1

            # first use RXNORM_CUI and raw
            if rx in rxnorm_to_ing:
                ing = rxnorm_to_ing.get(rx)
            elif rx_raw in rxnorm_to_ing:
                ing = rxnorm_to_ing.get(rx_raw)
            else:
                n_row_not_in_map += 1
                ing = ''

            if ing:
                if ing not in mci_drug_taken_by_patient:
                    mci_drug_taken_by_patient[ing][patid] = set([(date, day)])
                else:
                    if patid not in mci_drug_taken_by_patient.get(ing):
                        mci_drug_taken_by_patient[ing][patid] = set([(date, day)])
                    else:
                        mci_drug_taken_by_patient[ing][patid].add((date, day))

    print('# of Drugs: {}\t'.format(len(mci_drug_taken_by_patient)))
    print('Scan # n_records: ', n_records)
    print('n_no_day_supply: ', n_no_day_supply)
    print('n_day_supply_exist: ', n_day_supply_exist)
    print('n_day_supply_impute: ', n_day_supply_impute)
    print('n_row_not_in_map: ', n_row_not_in_map)

    try:
        print('dumping...', flush=True)
        pickle.dump(mci_drug_taken_by_patient,
                    open('pickles/mci_drug_taken_by_patient_from_prescribing.pkl', 'wb'))
    except Exception as e:
        print(e)
    print('finish dump', flush=True)

    return mci_drug_taken_by_patient


def load_latest_rxnorm_info():
    print('********load_latest_rxnorm_info*********')
    df = pd.read_csv(r'mapping/RXNORM.csv', dtype=str)  # (40157, 4)

    rxnorm_name = {}  # len: 26978
    for index, row in df.iterrows():

        rxnorm = row[r'Class ID'].strip().split('/')[-1]
        name = row[r'Preferred Label']
        rxnorm_name[rxnorm] = name
    print('df.shape:', df.shape, 'len(rxnorm_name):', len(rxnorm_name))
    return rxnorm_name, df


def _check_prescribing_ing(args):
    """
    Conclusion:
    1.  Majority of prescription records have NO day supply
    2.  There are many RAW_RXNORM_CUI, RXNORM_CUI have no ingredient mapping
        to discard them? or to clean them???
        save results in 'debug/rxnorm_from_prescribing_not_in_ingredient_mapping.csv'
    # n_records:        4400766
    # n_no_day_supply:  3629723
    # n_row_not_in_map:  273780
    # len(not_in_map) : 6094 for not_in_map.add((rx_raw, rx)), 12038 for not_in_map.add((rx_raw, rx, name))
    :param args:
    :return:
    """
    print('*******check_prescribing_ing********:')
    rxnorm_name, _ = load_latest_rxnorm_info()
    ndc_to_ing, rxnorm_to_ing = load_drug_mappings_to_ingredients()
    mci_drug_taken_by_patient = defaultdict(dict)
    dispensing_file = args.input_dispense
    prescribing_file = args.input_prescribe

    # Load from prescribing table
    print('Open file: ', prescribing_file, flush=True)
    n_no_day_supply = 0
    n_records = 0
    n_row_not_in_map = 0
    not_in_map = set()
    single_code_count = {}
    code_count = {}
    consist = []
    inconsist = []
    n_both_have_ing = 0
    n_same_ing_value = 0
    with open(prescribing_file, 'r') as f:
        # col_name = next(f)
        col_name = next(csv.reader(f))  # may have , in quote, name column
        for row in csv.reader(f):
            n_records += 1
            patid, date, rx, rx_raw, day, name = row[1], row[4], row[18], row[23], row[13], row[21]
            if patid.startswith('--'):
                print('Skip {}th row: {}'.format(n_records, row))
                continue
            # if n_records == 889803 + 1:
            #     name: morphine sulfate injection,-
            #     print(row)

            # day may be ''
            if not day or day == 'NULL':
                n_no_day_supply += 1
                day = -1

            # Check consistency:
            if rx_raw in rxnorm_to_ing and rx in rxnorm_to_ing:
                ing1 = rxnorm_to_ing.get(rx_raw)
                ing2 = rxnorm_to_ing.get(rx)
                n_both_have_ing += 1
                consist.append((patid, date, day, name,
                                rx, rxnorm_name[rx],
                                ing2, rxnorm_name[ing2],
                                rx_raw, rxnorm_name[rx_raw],
                                ing1, rxnorm_name[ing1]))
                if ing1 == ing2:
                    n_same_ing_value += 1
                else:
                    inconsist.append((patid, date, day, name,
                                rx, rxnorm_name[rx],
                                ing2, rxnorm_name[ing2],
                                rx_raw, rxnorm_name[rx_raw],
                                ing1, rxnorm_name[ing1]))

            # first use RAW_RXNORM_CUI in old code.
            # should first use RXNORM_CUI code
            # if rx_raw in rxnorm_to_ing:
            #     ing = rxnorm_to_ing.get(rx_raw)
            # elif rx in rxnorm_to_ing:
            #     ing = rxnorm_to_ing.get(rx)
            if rx in rxnorm_to_ing:
                ing = rxnorm_to_ing.get(rx)
            elif rx_raw in rxnorm_to_ing:
                ing = rxnorm_to_ing.get(rx_raw)
            else:  # should I discard these rx_raw/rx codes?
                n_row_not_in_map += 1
                not_in_map.add((rx_raw, rx, name))
                if (rx_raw, rx, name) in code_count:
                    code_count[(rx_raw, rx, name)] += 1
                else:
                    code_count[(rx_raw, rx, name)] = 1

                if rx_raw in single_code_count:
                    single_code_count[rx_raw] += 1
                else:
                    single_code_count[rx_raw] = 1

                if rx in single_code_count:
                    single_code_count[rx] += 1
                else:
                    single_code_count[rx] = 1

                ing = ''

        pd_consist = pd.DataFrame(consist, columns=['PATID', 'RX_ORDER_DATE', 'RX_DAYS_SUPPLY', 'RAW_RX_MED_NAME',
                                                    'RXNORM_CUI', 'label_RXNORM_CUI',
                                                    'ing_RXNORM_CUI', 'label_ing_RXNORM_CUI',
                                                    'RAW_RXNORM_CUI', 'label_RAW_RXNORM_CUI',
                                                    'ing_RAW_RXNORM_CUI', 'label_ing_RAW_RXNORM_CUI'
                                                    ])
        print('n_both_have_ing: ', n_both_have_ing, 'n_same_ing_value:', n_same_ing_value)
        pd_consist.to_csv('debug/prescribing_rx_and_rawrx_both_have_ings.csv')

        pd_inconsist = pd.DataFrame(inconsist, columns=['PATID', 'RX_ORDER_DATE', 'RX_DAYS_SUPPLY', 'RAW_RX_MED_NAME',
                                                    'RXNORM_CUI', 'label_RXNORM_CUI',
                                                    'ing_RXNORM_CUI', 'label_ing_RXNORM_CUI',
                                                    'RAW_RXNORM_CUI', 'label_RAW_RXNORM_CUI',
                                                    'ing_RAW_RXNORM_CUI', 'label_ing_RAW_RXNORM_CUI'
                                                    ])
        pd_inconsist.to_csv('debug/prescribing_rx_and_rawrx_both_have_ings_InconsistPart.csv')

        print('n_records: ', n_records)
        print('n_no_day_supply: ', n_no_day_supply)
        print('n_row_not_in_map: ', n_row_not_in_map)
        print('len(not_in_map) :', len(not_in_map))
        # print(not_in_map)
        not_in_map_enriched = [(x[0], x[1], x[2], code_count[x], single_code_count[x[0]], single_code_count[x[1]])
                               for x in not_in_map]
        pd_not_in = pd.DataFrame(not_in_map_enriched, columns=['RAW_RXNORM_CUI', 'RXNORM_CUI', 'RAW_RX_MED_NAME',
                                                               '#triple', '#RAW_RXNORM_CUI', '#RXNORM_CUI'])
        pd_not_in.sort_values(by=['#triple'], inplace=True, ascending=False)
        pd_not_in.to_csv('debug/rxnorm_from_prescribing_not_in_ingredient_mapping.csv')
        print(pd_not_in)
        # n_both_have_ing:  2294642 n_same_ing_value: 2175490
        # n_records:  5966370
        # n_no_day_supply:  3952977
        # n_row_not_in_map:  1144113
        # len(not_in_map) : 22404


def generate_and_dump_drug_patient_records():
    # 1. generate mci_drug_taken_by_patient from dispensing and prescribing
    # 2. combine drugs_patients from dispensing and prescribing

    start_time = time.time()
    mci_drug_taken_by_patient_dis = pre_drug_table_from_dispensing(args=parse_args())
    mci_drug_taken_by_patient_pre = pre_drug_table_from_prescribing(args=parse_args())

    mci_drug_taken_by_patient = copy.deepcopy(mci_drug_taken_by_patient_dis)
    for ing, taken_by_patient in mci_drug_taken_by_patient_pre.items():
        for patid, take_times_list in taken_by_patient.items():
            for take_time in take_times_list:
                date, day = take_time
                if ing not in mci_drug_taken_by_patient:
                    mci_drug_taken_by_patient[ing][patid] = set([(date, day)])
                else:
                    if patid not in mci_drug_taken_by_patient.get(ing):
                        mci_drug_taken_by_patient[ing][patid] = set([(date, day)])
                    else:
                        mci_drug_taken_by_patient[ing][patid].add((date, day))

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    with open(r'pickles/mci_drug_taken_by_patient_from_dispensing_plus_prescribing.pkl', 'wb') as f:
        pickle.dump(mci_drug_taken_by_patient, f)  # , pickle.HIGHEST_PROTOCOL)
    print(r'dump pickles/mci_drug_taken_by_patient_from_dispensing_plus_prescribing.pkl done!')


def load_drug_patient_records():
    with open(r'pickles/mci_drug_taken_by_patient_from_dispensing.pkl', 'rb') as f:
        mci_drug_taken_by_patient_dis = pickle.load(f)
    print(r'Load pickles/mci_drug_taken_by_patient_from_dispensing.pkl done：')
    print('***len(mci_drug_taken_by_patient_dis): ', len(mci_drug_taken_by_patient_dis))

    with open(r'pickles/mci_drug_taken_by_patient_from_prescribing.pkl', 'rb') as f:
        mci_drug_taken_by_patient_pre = pickle.load(f)
    print(r'Load pickles/mci_drug_taken_by_patient_from_prescribing.pkl done!')
    print('***len(mci_drug_taken_by_patient_pre): ', len(mci_drug_taken_by_patient_pre))

    with open(r'pickles/mci_drug_taken_by_patient_from_dispensing_plus_prescribing.pkl', 'rb') as f:
        mci_drug_taken_by_patient_dis_plus_pre = pickle.load(f)
    print(r'Load pickles/mci_drug_taken_by_patient_from_dispensing_plus_prescribing.pkl done!')
    print('***len(mci_drug_taken_by_patient_dis_plus_pre): ', len(mci_drug_taken_by_patient_dis_plus_pre))

    return mci_drug_taken_by_patient_dis, mci_drug_taken_by_patient_pre, mci_drug_taken_by_patient_dis_plus_pre


def count_drug_frequency():
    rxnorm_name, _ = load_latest_rxnorm_info()
    ci_drug_taken_by_patient_dis, mci_drug_taken_by_patient_pre, mci_drug_taken_by_patient_dis_plus_pre = \
        load_drug_patient_records()
    k_v = [ci_drug_taken_by_patient_dis, mci_drug_taken_by_patient_pre, mci_drug_taken_by_patient_dis_plus_pre]
    fname = ['debug/drug_prevalence_from_dispensing.csv',
             'debug/drug_prevalence_from_prescribing.csv',
             'debug/drug_prevalence_from_dispensing_plus_prescribing.csv']
    sheet_name = ['dispensing', 'prescribing', 'dispensing_plus_prescribing']
    writer = pd.ExcelWriter('debug/drug_prevalence_all_MCI_cohort.xlsx', engine='xlsxwriter')
    i = 0
    for mci_drug_taken_by_patient in k_v:
        drug_patient = []
        for ing, taken_by_patient in mci_drug_taken_by_patient.items():
            n_drug_time = 0
            for patid, take_times_list in taken_by_patient.items():
                n_drug_time += len(take_times_list)
            drug_patient.append((ing, rxnorm_name[ing], len(taken_by_patient), n_drug_time))

        pd_drug = pd.DataFrame(drug_patient, columns=['Ingredient_RXNORM_CUI', 'label',
                                                      'n_patient_take', 'n_patient_take_times'])
        pd_drug.sort_values(by=['n_patient_take'], inplace=True, ascending=False)
        pd_drug.to_csv(fname[i])
        pd_drug.to_excel(writer, sheet_name=sheet_name[i])
        i += 1
    writer.save()


def add_drug_name_to_cohort():
    df = pd.read_csv(r'debug/cohort_all_name_size_positive.csv')  # (40157, 4)
    df['ratio'] = df['n_positive'] / df['n_patients']
    rxnorm_name, _ = load_latest_rxnorm_info()
    df['drug_name'] = df['cohort_name'].apply(lambda x : rxnorm_name[x.split('.')[0]])
    df.to_csv('debug/cohort_all_name_size_positive_with_name.csv')


if __name__ == '__main__':
    print(parse_args())
    # add_drug_name_to_cohort()

    # start_time = time.time()
    # generate_and_dump_drug_mappings_to_ingredients()
    # print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    # start_time = time.time()
    # ndc_to_ing, rxnorm_to_ing = load_drug_mappings_to_ingredients()
    # print('Load Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # start_time = time.time()
    # generate_and_dump_drug_patient_records()
    # mci_drug_taken_by_patient_dis, mci_drug_taken_by_patient_pre, mci_drug_taken_by_patient_dis_plus_pre = \
    #     load_drug_patient_records()
    # print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    ## debug part
    # start_time = time.time()
    # _check_prescribing_ing(args=parse_args())
    # pre_drug_table_from_prescribing(args=parse_args())
    # pre_drug_table_from_dispensing(args=parse_args())
    count_drug_frequency()
    # print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
