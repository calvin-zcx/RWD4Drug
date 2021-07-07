from collections import defaultdict
from datetime import datetime
import os
import pandas as pd
from tqdm import tqdm
import time
import pickle
import argparse
import csv
from utils import *
import numpy as np


# def _csv_row_read_select_write(infile, outfile, columns_index):
#     n_read = 0
#     n_dump = 0
#     with open(infile, 'r') as f:
#         col_name = next(csv.reader(f))  # read from first non-name row, above code from 2nd, wrong
#         check_and_mkdir(outfile)
#         with open(outfile, "w", newline='', encoding='utf-8') as fout:
#             fcsv = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#             fcsv.writerow([col_name[i] for i in columns_index])
#             for row in csv.reader(f):
#                 n_read += 1
#                 if row[0].startswith('--'):
#                     print('Skip {}th row: {}'.format(n_read, row))
#                     continue
#                 fcsv.writerow([row[i] for i in columns_index])
#                 n_dump += 1
#     return n_read, n_dump, [col_name[i] for i in columns_index]


def data_normalize_demographics(infile, outfile):
    """Demographics table:
    Output:
        0. patient_id
        1. sex
        2. birth_date (only year in marketscan)
        3. zip code (Marketscan: region)
        4. race (not in Marketscan, not use)
    Input: Florida data:
    PATID	BIRTH_DATE	BIRTH_TIME	SEX	SEXUAL_ORIENTATION	GENDER_IDENTITY	HISPANIC	RACE	BIOBANK_FLAG	PAT_PREF_LANGUAGE_SPOKEN	RAW_SEX	RAW_SEXUAL_ORIENTATION	RAW_GENDER_IDENTITY	RAW_HISPANIC	RAW_RACE	RAW_PAT_PREF_LANGUAGE_SPOKEN	UPDATED	SOURCE	ZIP_CODE	LAST_ENCOUNTERID
    """
    start_time = time.time()
    columns_index = [0, 3, 1, 18, 7]  # 0-PATID, 3-SEX, 1-BIRTH_DATE, 18-ZIP_CODE, 7-RACE
    n_read = 0
    n_dump = 0
    with open(infile, 'r') as f:
        col_name = next(csv.reader(f))  # read from first non-name row, above code from 2nd, wrong
        check_and_mkdir(outfile)
        with open(outfile, "w", newline='', encoding='utf-8') as fout:
            fcsv = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            name = [col_name[i] for i in columns_index]
            fcsv.writerow(name)
            for row in csv.reader(f):
                n_read += 1
                if row[0].startswith('--'):
                    print('Skip {}th row: {}'.format(n_read, row))
                    continue
                fcsv.writerow([row[i] for i in columns_index])
                n_dump += 1

    print('read {}, dump {}, column {}'.format(n_read, n_dump, name))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def data_normalize_diagnosis(infile, outfile):
    """Diagnosis table:
        Output:
            0. patient_id
            1. date (administration/diagnosis date)
            2. diagnosis_code  (e.g. ICD 9, 10, etc.)
            3. diagnosis_code_type (e.g. Indicators for ICD 9, 10, etc.)
        Input: Florida data:
        DIAGNOSISID	PATID	ENCOUNTERID	ENC_TYPE	ADMIT_DATE	PROVIDERID	DX	DX_TYPE	DX_DATE	DX_SOURCE	DX_ORIGIN	PDX	DX_POA	RAW_DX	RAW_DX_TYPE	RAW_DX_SOURCE	RAW_PDX	RAW_DX_POA	UPDATED	SOURCE	RAW_DX_ORIGIN
        Notice: using column 8-DX_DATE to impute column 4-ADMIT_DATE, if ADMIT_DATE is NULL
    """
    start_time = time.time()
    columns_index = [1, 4, 6, 7]  # 1-PATID, 4-ADMIT_DATE, 6-DX, 7-DX_TYPE, 8-DX_DATE
    n_read = 0
    n_dump = 0
    n_impute = 0
    with open(infile, 'r') as f:
        col_name = next(csv.reader(f))  # read from first non-name row, above code from 2nd, wrong
        check_and_mkdir(outfile)
        with open(outfile, "w", newline='', encoding='utf-8') as fout:
            fcsv = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            name = [col_name[i] for i in columns_index]
            fcsv.writerow(name)
            for row in csv.reader(f):
                n_read += 1
                if row[0].startswith('--'):
                    print('Skip {}th row: {}'.format(n_read, row))
                    continue
                # using column 8-DX_DATE to impute column 4-ADMIT_DATE, if ADMIT_DATE is NULL
                if (row[4] == '') or (row[4] == 'NULL'):
                    print(n_read, row[4], row[8])
                    row[4] = row[8]
                    n_impute += 1

                fcsv.writerow([row[i] for i in columns_index])
                fout.flush()
                n_dump += 1
    print('read {}, dump {}, column {}, impute {}'.format(n_read, n_dump, name, n_impute))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def data_normalize_medication_dispensing(infile, outfile):
    """Diagnosis table:
        Output:
            0. patient_id
            1. date (order/dispensing date)
            2. drug_code  (e.g. NDC, rxnorm, GPI, , etc.)
            3. supply_days (e.g. 30, 90 etc days )
            4. drug_name (if any)
        Input: Florida data:
        DISPENSINGID	PATID	PRESCRIBINGID	DISPENSE_DATE	NDC	DISPENSE_SOURCE	DISPENSE_SUP	DISPENSE_AMT	DISPENSE_DOSE_DISP	DISPENSE_DOSE_DISP_UNIT	DISPENSE_ROUTE	RAW_NDC	RAW_DISPENSE_DOSE_DISP	RAW_DISPENSE_DOSE_DISP_UNIT	RAW_DISPENSE_ROUTE	UPDATED	SOURCE
        Notice:
    """
    start_time = time.time()
    columns_index = [1, 3, 4, 6]  # 1-PATID, 3-DISPENSE_DATE, 4-NDC, 6-DISPENSE_SUP
    n_read = 0
    n_dump = 0
    n_impute = 0
    with open(infile, 'r') as f:
        col_name = next(csv.reader(f))  # read from first non-name row, above code from 2nd, wrong
        check_and_mkdir(outfile)
        with open(outfile, "w", newline='', encoding='utf-8') as fout:
            fcsv = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            name = [col_name[i] for i in columns_index]
            fcsv.writerow(name)
            for row in csv.reader(f):
                n_read += 1
                if row[0].startswith('--'):
                    print('Skip {}th row: {}'.format(n_read, row))
                    continue
                # using column 7-DISPENSE_AMT to impute column 6-DISPENSE_SUP, if DISPENSE_SUP is NULL
                if (row[6] == '') or (row[6] == 'NULL'):
                    print(n_read, row[6], row[7])
                    row[6] = row[7]
                    n_impute += 1

                fcsv.writerow([row[i] for i in columns_index])
                fout.flush()
                n_dump += 1
    print('read {}, dump {}, column {}, impute {}'.format(n_read, n_dump, name, n_impute))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def data_normalize_medication_prescribing(infile, outfile):
    """Diagnosis table:
        Output:
            0. patient_id
            1. date (order/dispensing date)
            2. drug_code  (e.g. NDC, rxnorm, GPI, , etc.)
            3. supply_days (e.g. 30, 90 etc days )
            4. drug_name (if any)
        Input: Florida data:
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
        Notice:
    """
    start_time = time.time()
    columns_index = [1, 6, 18, 13, 21]  # 1-PATID, 6-RX_START_DATE,  18-RXNORM_CUI, 13-RX_DAYS_SUPPLY, 21-RAW_RX_MED_NAME
    n_read = 0
    n_dump = 0
    n_impute_start_day = 0
    n_impute_supply_day = 0
    n_empty_supply_day = 0
    with open(infile, 'r') as f:
        col_name = next(csv.reader(f))  # read from first non-name row, above code from 2nd, wrong
        check_and_mkdir(outfile)
        with open(outfile, "w", newline='', encoding='utf-8') as fout:
            fcsv = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            name = [col_name[i] for i in columns_index]
            fcsv.writerow(name)
            for row in csv.reader(f):
                n_read += 1
                if row[0].startswith('--'):
                    print('Skip {}th row: {}'.format(n_read, row))
                    continue
                # using column 4-RX_ORDER_DATE to impute column 6-RX_START_DATE, if RX_START_DATE is NULL
                if (row[6] == '') or (row[6] == 'NULL'):
                    print(n_read, row[6], row[4])
                    row[6] = row[4]
                    n_impute_start_day += 1
                # imputing 13-RX_DAYS_SUPPLY by 6-RX_START_DATE, and 7-RX_END_DATE
                if (row[13] == '') or (row[13] == 'NULL'):
                    n_empty_supply_day += 1
                    if (row[7] != '') and (row[7] != 'NULL') and (row[6] != '') and (row[6] != 'NULL'):
                        imputed_day = str((str_to_datetime(row[7]) - str_to_datetime(row[6])).days)
                        print(n_read, row[13], row[6], row[7], imputed_day)
                        row[13] = imputed_day
                        n_impute_supply_day += 1
                fcsv.writerow([row[i] for i in columns_index])
                fout.flush()
                n_dump += 1
    print('read {}, dump {}, column {}, n_impute_start_day {}, n_impute_supply_day {}, n_empty_supply_day {}'.
          format(n_read, n_dump, name, n_impute_start_day, n_impute_supply_day, n_empty_supply_day))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


if __name__ == '__main__':
    start_time = time.time()
    data_normalize_demographics(r'C:\Users\zangc\Documents\Boston\data_large\MCI_data_20210421\DEMOGRAPHIC.csv',
                                r'../data/florida/demographic.csv')
    data_normalize_diagnosis(r'C:\Users\zangc\Documents\Boston\data_large\MCI_data_20210421\DIAGNOSIS.csv',
                             r'../data/florida/diagnosis.csv')
    data_normalize_medication_dispensing(r'C:\Users\zangc\Documents\Boston\data_large\MCI_data_20210421\DISPENSING.csv',
                             r'../data/florida/dispensing.csv')
    data_normalize_medication_prescribing(r'C:\Users\zangc\Documents\Boston\data_large\MCI_data_20210421\PRESCRIBING.csv',
                                         r'../data/florida/prescribing.csv')
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    print('Done')
