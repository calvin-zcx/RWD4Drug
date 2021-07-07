from collections import defaultdict
from datetime import datetime
import os
import pandas as pd
from tqdm import tqdm
import csv
import utils


def get_user_dx(indir, patient_set, icd9_to_ccs, icd10_to_ccs, icd_to_ccw, coding):

    user_dx = defaultdict(dict)

    DXVER_dict = {9: icd9_to_ccs, 10: icd10_to_ccs, 'ccw': icd_to_ccw}
    n_records = 0
    n_not_in_patient_list_or_nan = 0
    n_not_in_ccs = 0
    n_no_date = 0
    with open(indir, 'r') as f:
        col_name = next(csv.reader(f))  # read from std dx format
        print('read from ', indir, col_name)
        for row in tqdm(csv.reader(f)):  # , total=34554738):
            n_records += 1
            patid, date, dx = row[0], row[1], row[2]

            if (date == '') or (date == 'NULL'):
                n_no_date += 1
                continue
            else:
                date = utils.str_to_datetime(date)

            if patid not in patient_set:
                # or pd.isna(date) or pd.isna(dx):
                n_not_in_patient_list_or_nan += 1
                continue

            try:
                # florida:
                    # 10                22540060
                    # 9                11195478
                    # 10                483263
                    # 09                326864
                    # OT                8131
                    # SM                942
                # marketscan: no diagnosis type
                dx_type = int(row[3])
            except:
                dx_type = -1

            if (dx_type == 9) or (dx_type == 10):
                ccs = get_ccs_code_for_1_icd(dx, DXVER_dict[dx_type])
            else:  # no diagnosis type specified, try both ICD-9 and ICD-10, and selected non-null string
                ccs1 = get_ccs_code_for_1_icd(dx, DXVER_dict[9])
                ccs2 = get_ccs_code_for_1_icd(dx, DXVER_dict[10])
                ccs = max(ccs1, ccs2)  # '' v.s. string code

            ccw = get_ccs_code_for_1_icd(dx, DXVER_dict['ccw'])

            if coding == 'ccs':
                dxs = ccs
            elif coding == 'ccw':
                dxs = ccw
            else:
                raise ValueError

            if dxs:
                if patid not in user_dx:
                    user_dx[patid][date] = [(dx, ccs, ccw), ]  # [dxs, ]
                else:
                    if date not in user_dx[patid]:
                        user_dx[patid][date] = [(dx, ccs, ccw), ]  # [dxs, ]
                    else:
                        user_dx[patid][date].append((dx, ccs, ccw))  # (dxs)

    return user_dx


def get_ccs_code_for_1_icd(icd_code: str, icd_to_css):
    assert isinstance(icd_code,  str)
    css_code = ''
    icd_code = icd_code.strip().replace('.', '')
    for i in range(len(icd_code), -1, -1):
        if icd_code[:i] in icd_to_css:
            return icd_to_css.get(icd_code[:i])
    return css_code


# def get_css_code_for_icd(icd_codes, icd_to_css):
#     css_codes = []
#     for icd_code in icd_codes:
#         if not pd.isnull(icd_code):
#             for i in range(len(icd_code), -1, -1):
#                 if icd_code[:i] in icd_to_css:
#                     css_codes.append(icd_to_css.get(icd_code[:i]))
#                     break
#
#     return css_codes


def pre_user_cohort_dx(user_dx, prescription_taken_by_patient, min_patients, coding):
    user_cohort_dx = AutoVivification()
    for drug, taken_by_patient in tqdm(prescription_taken_by_patient.items()):
        if len(taken_by_patient.keys()) >= min_patients:
            for patient, taken_times in taken_by_patient.items():
                index_date = taken_times[0]
                date_codes = user_dx.get(patient)
                for date, codes in date_codes.items():
                    # date = utils.str_to_datetime(date)  # datetime.strptime(date, '%m/%d/%Y')
                    icd_codes, ccs_codes, ccw_codes = zip(*codes)  #

                    if coding == 'ccs':
                        code_list = ccs_codes
                    elif coding == 'ccw':
                        code_list = ccw_codes
                    else:
                        raise ValueError

                    # the major ccs/ccw code is not '', and keep all (dx, ccs, ccw) codes???
                    # code_list = codes

                    if date <= index_date:  # use <=, ranther than < in 2021-07-06
                        if drug not in user_cohort_dx:
                            user_cohort_dx[drug][patient][date] = set(code_list)
                        else:
                            if patient not in user_cohort_dx[drug]:
                                user_cohort_dx[drug][patient][date] = set(code_list)
                            else:
                                if date not in user_cohort_dx[drug][patient]:
                                    user_cohort_dx[drug][patient][date] = set(code_list)
                                else:
                                    user_cohort_dx[drug][patient][date].union(code_list)

    return user_cohort_dx


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def get_user_cohort_dx(indir, prescription_taken_by_patient, icd9_to_css, icd10_to_css, icd_to_ccw, coding,
                       min_patient, patient_set):
    print('get_user_cohort_dx...')
    print('Diagnosis coding type: ', coding)
    user_dx = get_user_dx(indir, patient_set, icd9_to_css, icd10_to_css, icd_to_ccw, coding)
    return pre_user_cohort_dx(user_dx, prescription_taken_by_patient, min_patient, coding), user_dx

