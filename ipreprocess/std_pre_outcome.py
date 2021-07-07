from collections import defaultdict
from datetime import datetime
import pickle
from tqdm import tqdm
import os
import pandas as pd
import utils
import csv

# def is_valid_outcome_range(dx, code_range):
#     for code in code_range:
#         if dx.startswith(code):
#             return True
#     return False


def pre_user_cohort_outcome(indir, patient_list, criterion):
    print('pre_user_cohort_outcome...')
    cohort_outcome = {key: defaultdict(list) for key in criterion.keys()}

    # data format: {patient_id: [date_of_outcome, date_of_outcome,...,date_of_outcome]}
    n_records = 0
    n_not_in_patient_list_or_nan = 0
    n_no_date = 0
    with open(indir, 'r') as f:
        col_name = next(csv.reader(f))  # read from first non-name row, above code from 2nd, wrong
        print('read from ', indir, col_name)
        for row in tqdm(csv.reader(f)):  # , total=34554738):
            n_records += 1
            patid, date, dx = row[0], row[1], row[2]

            if (date == '') or (date == 'NULL'):
                n_no_date += 1
                continue
            else:
                date = utils.str_to_datetime(date)

            if patid not in patient_list:
                # or pd.isna(date) or pd.isna(dx):
                n_not_in_patient_list_or_nan += 1
            else:
                for key, fc in criterion.items():
                    if fc(dx):
                        cohort_outcome[key][patid].append(date)

    return cohort_outcome
