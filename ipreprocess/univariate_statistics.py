import argparse
import os
import time
import sys
# for linux env.
sys.path.insert(0,'..')
import pickle
import utils
import pickle
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import Counter, defaultdict
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
from scipy import stats
from ipreprocess.utils import load_icd_to_ccw
import functools
print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--dx_file', default=r'../data/florida/diagnosis.csv',
                        help='input diagnosis file with std format')
    parser.add_argument('--drug_coding', choices=['rxnorm', 'gpi'], default='rxnorm')
    args = parser.parse_args()
    return args


def get_ccw_code_for_1_icd(icd_code: str, icd_to_ccw):
    assert isinstance(icd_code, str)
    css_code = ''
    icd_code = icd_code.strip().replace('.', '')
    for i in range(len(icd_code), -1, -1):
        if icd_code[:i] in icd_to_ccw:
            return icd_to_ccw.get(icd_code[:i])
    return css_code


def ASTD_IQR(s):
    return '{:.2f} ({:.2f}), {:.2f} ([{:.2f}, {:.2f}])'.format(s.mean(), s.std(),
                                                               s.quantile(.5), s.quantile(.25), s.quantile(.75))


def t_test(a, b):
    # pvalue (statistics)
    try:
        r = stats.ttest_ind(a, b, equal_var=False)
        return '{:.4f} ({:.4f})'.format(r[1], r[0])
    except Exception as e:
        # print('t_test: ', str(e))
        return str(e)


def chi2(ad, other, cat=(0, 1)):
    contingency_table = np.array([
        [(ad == c).sum() for c in cat],
        [(other == c).sum() for c in cat]
    ])
    try:
        r = stats.chi2_contingency(contingency_table)
        return '{:.4f} ({:.4f})'.format(r[1], r[0])
    except Exception as e:
        # print('chi2: ', str(e))
        return str(e)


def build_patient_characteristics(id_demo, drug_taken_by_patient, patient_dates,
                                  icd_2_comorbidity, comorbidityname_2_id, dx_file, atc2rxnorm,
                                  out_file, drug_coding='rxnorm'):
    print('build_patient_characteristics...')
    # 1. demo features
    df = pd.DataFrame([(key, val[0], val[1], val[2]) for key, val in id_demo.items()],
                      columns=['patid', 'bdate', 'sex', 'race'])
    df.set_index("patid", inplace=True)

    # 2. age, key dates and outcome
    df['MCI_date'] = np.nan
    df['age_of_MCI'] = np.nan
    df['AD_date'] = np.nan
    df['AD'] = 0

    for pid, val in tqdm(patient_dates.items()):
        #  patient_dates: {0: 'birth_date',  1: '1st_diagnosis_date', 2: '1st_mci_date',
        #  3: '1st_AD_date', 4: '1st_dementia_date',  5: '1st_ADRD_date',  6: 'last_diagnosis_date'}
        if (not pd.isna(val[2])) and (not pd.isna(val[0])):
            df.at[pid, 'MCI_date'] = val[2]
            df.at[pid, 'age_of_MCI'] = val[2].year - val[0].year  # (val[2] - val[0]).days / 365.

        if not pd.isna(val[3]):
            df.at[pid, 'AD_date'] = val[3]
            df.at[pid, 'AD'] = 1

    # 3. Drug,
    #    On antidiabetic drug A10    gpi:27 00 00 00 Antidiabetics
    #    On antihypertensives C02    gpi:36 00 00 00 Antihypertensives

    if drug_coding.lower() == 'gpi':
        is_antidiabetic = lambda x: (x[:2] == '27')
        is_antihypertensives = lambda x: (x[:2] == '36')
    else:
        is_antidiabetic = lambda x: x in atc2rxnorm['A10']
        is_antihypertensives = lambda x: x in atc2rxnorm['C02']

    df['antidiabetic'] = 0
    df['antihypertensives'] = 0
    for drug, taken_by_patient in tqdm(drug_taken_by_patient.items()):
        for patient, take_times in taken_by_patient.items():
            if patient in df.index:
                if is_antidiabetic(drug):
                    df.at[patient, 'antidiabetic'] += len(take_times)
                if is_antihypertensives(drug):
                    df.at[patient, 'antihypertensives'] += len(take_times)

    # 4. comorbidity
    for col in comorbidityname_2_id.keys():
        df[col] = 0

    n_records = 0
    n_no_date = 0
    with open(dx_file, 'r') as f:
        col_name = next(csv.reader(f))  # read from std dx format
        print('read from ', dx_file, col_name)
        for row in tqdm(csv.reader(f)):
            n_records += 1
            patid, date, dx = row[0], row[1], row[2]

            if (date == '') or (date == 'NULL'):
                n_no_date += 1
                continue
            else:
                date = utils.str_to_datetime(date)

            ccwname = get_ccw_code_for_1_icd(dx, icd_2_comorbidity)
            if ccwname and (patid in df.index):
                df.at[patid, ccwname] += 1

    utils.check_and_mkdir(out_file)
    pickle.dump(df, open(out_file, 'wb'))
    df.to_csv(out_file.replace('.pkl', '') + '.csv')
    print('Dump {} {} done!'.format(out_file + '.csv/pkl', df.shape))
    return df


def statistics_from_df(df, out_file):
    """
        1.florida data: e.g:
        df_demo['SEX'].value_counts():
            F --> 0     33498
            M --> 1     26692
            UN --> 1        2
        df_demo['RACE'].value_counts():
            05    29526 05=White
            OT    10726 OT=Other
            UN     9403 UN=Unknown
            03     9312 03=Black or African American
            02      477 02=Asian
            06      395 06=Multiple race
            NI      152 NI=No information
            01       99   01=American Indian or Alaska Native
            07       74 07=Refuse to answer
            04       28 04=Native Hawaiian or Other Pacific Islander
        2. marketscan data: no race, all 0
        """
    print(df.columns)
    df1 = df.loc[df['AD'] >= 1, :]
    df0 = df.loc[df['AD'] <= 0, :]  # may have -1 as censoring

    # Caution: not comparing race?
    index_cnt = ['No.', 'age_of_MCI']
    index_sex = ['sex', 'sex-0-female', 'sex-1-male']
    index_race = ['race', 'race-5-white', 'race-0-other', 'race-3-black', 'race-2-asian']
    index_risk = ['antidiabetic', 'antihypertensives', 'Alcohol Use Disorders',
                  'Anxiety Disorders', 'Depression', 'Diabetes', 'Heart Failure',
                  'Hyperlipidemia', 'Hypertension', 'Ischemic Heart Disease', 'Obesity',
                  'Stroke / Transient Ischemic Attack', 'Tobacco Use',
                  'Traumatic Brain Injury and Nonpsychotic Mental Disorders due to Brain Damage',
                  'Sleep disorders', 'Periodontitis', 'Menopause']
    columns = ['Overall_MCI', 'AD', 'Others', 'p-value']
    rdf = pd.DataFrame(columns=columns, index=index_cnt + index_sex + index_race + index_risk)

    rdf.at['No.', 'Overall_MCI'] = str(len(df))
    rdf.at['No.', 'AD'] = '{} ({:.2f}%)'.format(len(df1), len(df1) * 100.0 / len(df))
    rdf.at['No.', 'Others'] = '{} ({:.2f}%)'.format(len(df0), len(df0) * 100.0 / len(df))

    rdf.at['age_of_MCI', 'Overall_MCI'] = ASTD_IQR(df['age_of_MCI'])
    rdf.at['age_of_MCI', 'AD'] = ASTD_IQR(df1['age_of_MCI'])
    rdf.at['age_of_MCI', 'Others'] = ASTD_IQR(df0['age_of_MCI'])
    rdf.at['age_of_MCI', 'p-value'] = t_test(df1['age_of_MCI'], df0['age_of_MCI'])

    # sex, Chi-square test of independence
    rdf.at['sex', 'p-value'] = chi2(df1['sex'], df0['sex'])
    for c, name in {0: 'female', 1: 'male'}.items():
        rdf.at['sex-{}-{}'.format(c, name), 'Overall_MCI'] = '{} ({:.2f}%)'.format((df['sex'] == c).sum(),
                                                                                   (df['sex'] == c).sum() * 100.0 / len(
                                                                                       df))
        rdf.at['sex-{}-{}'.format(c, name), 'AD'] = '{} ({:.2f}%)'.format((df1['sex'] == c).sum(),
                                                                          (df1['sex'] == c).sum() * 100.0 / len(df1))
        rdf.at['sex-{}-{}'.format(c, name), 'Others'] = '{} ({:.2f}%)'.format((df0['sex'] == c).sum(),
                                                                              (df0['sex'] == c).sum() * 100.0 / len(
                                                                                  df0))

    # race if any, Chi-square test of independence: ['race','race-5-white','race-0-other','race-3-black','race-2-asian']
    rdf.at['race', 'p-value'] = chi2(df1['race'], df0['race'], (5, 0, 3, 2))
    for c, name in {5: 'white', 0: 'other', 3: 'black', 2: 'asian'}.items():
        rdf.at['race-{}-{}'.format(c, name), 'Overall_MCI'] = '{} ({:.2f}%)'.format((df['race'] == c).sum(),
                                                                                    (df[
                                                                                         'race'] == c).sum() * 100.0 / len(
                                                                                        df))
        rdf.at['race-{}-{}'.format(c, name), 'AD'] = '{} ({:.2f}%)'.format((df1['race'] == c).sum(),
                                                                           (df1['race'] == c).sum() * 100.0 / len(df1))
        rdf.at['race-{}-{}'.format(c, name), 'Others'] = '{} ({:.2f}%)'.format((df0['race'] == c).sum(),
                                                                               (df0['race'] == c).sum() * 100.0 / len(
                                                                                   df0))
    for risk in index_risk:
        s1 = df1[risk]
        s0 = df0[risk]
        rdf.at[risk, 'p-value'] = chi2(s1.mask(s1 >= 1, 1), s0.mask(s0 >= 1, 1))
        rdf.at[risk, 'Overall_MCI'] = '{} ({:.2f}%)'.format((df[risk] >= 1).sum(),
                                                            (df[risk] >= 1).sum() * 100.0 / len(df))
        rdf.at[risk, 'AD'] = '{} ({:.2f}%)'.format((df1[risk] >= 1).sum(),
                                                   (df1[risk] >= 1).sum() * 100.0 / len(df1))
        rdf.at[risk, 'Others'] = '{} ({:.2f}%)'.format((df0[risk] >= 1).sum(),
                                                       (df0[risk] >= 1).sum() * 100.0 / len(df0))

    if '.csv' in out_file:
        utils.check_and_mkdir(out_file)
        rdf.to_csv(out_file)
        print('Dump {} {} done!'.format(out_file, df.shape))
    else:
        utils.check_and_mkdir(out_file)
        pickle.dump(rdf, open(out_file, 'wb'))
        rdf.to_csv(out_file.replace('.pkl', '') + '.csv')
        print('Dump {} {} done!'.format(out_file + '.csv/pkl', df.shape))

    return rdf


def build_patient_characteristics_from_triples(trips, comorbidityid_2_name, drug_criterion):
    # demo_feature_vector: [age, sex, race, days_since_mci]
    # triple = (patient,
    #           [rx_codes, dx_codes, demo_feature_vector[0], demo_feature_vector[1], demo_feature_vector[3]],
    #           (outcome, outcome_t2e))
    # if drug_coding.lower() == 'gpi':
    #     is_antidiabetic = lambda x: (x[:2] == '27')
    #     is_antihypertensives = lambda x: (x[:2] == '36')
    # else:
    #     is_antidiabetic = lambda x: x in atc2rxnorm['A10']
    #     is_antihypertensives = lambda x: x in atc2rxnorm['C02']
    # drug_criterion = {'antidiabetic':is_antidiabetic, 'antihypertensives':is_antihypertensives}
    # start_time = time.time()
    patid_list = [x[0] for x in trips]
    drug_patient = pd.DataFrame(0,
                                columns=['age', 'sex', 'days_since_mci', 'AD', 't2e',
                                         'antidiabetic', 'antihypertensives',
                                         'Alcohol Use Disorders',
                                         'Anxiety Disorders', 'Depression', 'Diabetes', 'Heart Failure',
                                         'Hyperlipidemia', 'Hypertension', 'Ischemic Heart Disease', 'Obesity',
                                         'Stroke / Transient Ischemic Attack', 'Tobacco Use',
                                         'Traumatic Brain Injury and Nonpsychotic Mental Disorders due to Brain Damage',
                                         'Sleep disorders', 'Periodontitis', 'Menopause'],
                                index=patid_list)
    drug_patient.index.name = 'patid'

    for record in tqdm(trips):
        patient = record[0]
        rx_codes = record[1][0]
        dx_codes = record[1][1]

        # build patient characteristics of this drug trial
        drug_patient.at[patient, 'age'] = record[1][2]
        drug_patient.at[patient, 'sex'] = record[1][3]
        drug_patient.at[patient, 'days_since_mci'] = np.round(record[1][4] / 30.)

        drug_patient.at[patient, 'AD'] = record[2][0]
        drug_patient.at[patient, 't2e'] = np.round(record[2][1] / 30.)

        crx = Counter([x for l in rx_codes for x in l])
        for rx, v in crx.items():
            for key, fc in drug_criterion.items():
                if fc(rx):
                    drug_patient.at[patient, key] += v

        # for rx_list in rx_codes:
        #     for rx in rx_list:
        #         for key, fc in drug_criterion.items():
        #             if fc(rx):
        #                 drug_patient.at[patient, key] += 1

        cdx = Counter([x for l in dx_codes for x in l])
        for k, v in cdx.items():
            if k in comorbidityid_2_name:
                ccwname = comorbidityid_2_name[k]
                drug_patient.at[patient, ccwname] += v

        # for dx_list in dx_codes:
        #     for dx in dx_list:
        #         if dx in comorbidityid_2_name:
        #             ccwname = comorbidityid_2_name[dx]
        #             drug_patient.at[patient, ccwname] += 1

    return drug_patient


def statistics_for_treated_control(treat, control, out_file, add_rows=None):
    print(treat.columns)

    # Caution: not comparing race?
    index_cnt = ['No.', 'age']
    index_sex = ['sex', 'sex-0-female', 'sex-1-male']
    index_days = ["days_since_mci"]
    index_ad = ['AD', 'Non-AD', 'Censoring']
    index_t2e = ['t2e', 't2e-AD', 't2e-Non-AD', 't2e-Censoring']
    index_risk = ['antidiabetic', 'antihypertensives', 'Alcohol Use Disorders',
                  'Anxiety Disorders', 'Depression', 'Diabetes', 'Heart Failure',
                  'Hyperlipidemia', 'Hypertension', 'Ischemic Heart Disease', 'Obesity',
                  'Stroke / Transient Ischemic Attack', 'Tobacco Use',
                  'Traumatic Brain Injury and Nonpsychotic Mental Disorders due to Brain Damage',
                  'Sleep disorders', 'Periodontitis', 'Menopause']
    columns = ['treat', 'control', 'p-value']
    rdf = pd.DataFrame(columns=columns, index=index_cnt + index_sex + index_days + index_ad + index_t2e + index_risk)

    rdf.at['No.', 'treat'] = '{} ({:.2f}%)'.format(len(treat), len(treat) * 100.0 / (len(treat) + len(control)))
    rdf.at['No.', 'control'] = '{} ({:.2f}%)'.format(len(control), len(control) * 100.0 / (len(treat) + len(control)))

    rdf.at['age', 'treat'] = ASTD_IQR(treat['age'])
    rdf.at['age', 'control'] = ASTD_IQR(control['age'])
    rdf.at['age', 'p-value'] = t_test(treat['age'], control['age'])

    rdf.at['days_since_mci', 'treat'] = ASTD_IQR(treat['days_since_mci'])
    rdf.at['days_since_mci', 'control'] = ASTD_IQR(control['days_since_mci'])
    rdf.at['days_since_mci', 'p-value'] = t_test(treat['days_since_mci'], control['days_since_mci'])

    # sex, Chi-square test of independence
    rdf.at['sex', 'p-value'] = chi2(treat['sex'], control['sex'])
    for c, name in {0: 'female', 1: 'male'}.items():
        rdf.at['sex-{}-{}'.format(c, name), 'treat'] = '{} ({:.2f}%)'.format((treat['sex'] == c).sum(),
                                                                             (treat['sex'] == c).sum() * 100.0 / len(
                                                                                 treat))
        rdf.at['sex-{}-{}'.format(c, name), 'control'] = '{} ({:.2f}%)'.format((control['sex'] == c).sum(),
                                                                               (control[
                                                                                    'sex'] == c).sum() * 100.0 / len(
                                                                                   control))

    # Chi-square test of independence index_ad = ['AD', 'Non-AD', 'Censoring']
    rdf.at['AD', 'p-value'] = chi2(treat['AD'], control['AD'], (1, 0, -1))
    for c, name in {1: 'AD', 0: 'Non-AD', -1: 'Censoring'}.items():
        rdf.at[name, 'treat'] = '{} ({:.2f}%)'.format((treat['AD'] == c).sum(),
                                                      (treat['AD'] == c).sum() * 100.0 / len(treat))
        rdf.at[name, 'control'] = '{} ({:.2f}%)'.format((control['AD'] == c).sum(),
                                                        (control['AD'] == c).sum() * 100.0 / len(
                                                            control))
    # index_t2e = ['t2e', 't2e-AD', 't2e-Non-AD', 't2e-Censoring']
    rdf.at['t2e', 'treat'] = ASTD_IQR(treat['t2e'])
    rdf.at['t2e', 'control'] = ASTD_IQR(control['t2e'])
    rdf.at['t2e', 'p-value'] = t_test(treat['t2e'], control['t2e'])
    for t, c in {'t2e-AD': 1, 't2e-Non-AD': 0, 't2e-Censoring': -1}.items():
        idx1 = treat['AD'] == c
        idx0 = control['AD'] == c
        rdf.at[t, 'treat'] = ASTD_IQR(treat.loc[idx1, 't2e'])
        rdf.at[t, 'control'] = ASTD_IQR(control.loc[idx0, 't2e'])
        rdf.at[t, 'p-value'] = t_test(treat.loc[idx1, 't2e'], control.loc[idx0, 't2e'])

    for risk in index_risk:
        s1 = treat[risk]
        s0 = control[risk]
        rdf.at[risk, 'p-value'] = chi2(s1.mask(s1 >= 1, 1), s0.mask(s0 >= 1, 1))

        rdf.at[risk, 'treat'] = '{} ({:.2f}%)'.format((treat[risk] >= 1).sum(),
                                                      (treat[risk] >= 1).sum() * 100.0 / len(treat))
        rdf.at[risk, 'control'] = '{} ({:.2f}%)'.format((control[risk] >= 1).sum(),
                                                        (control[risk] >= 1).sum() * 100.0 / len(control))

    rdf = rdf.append(add_rows)
    if '.csv' in out_file:
        utils.check_and_mkdir(out_file)
        rdf.to_csv(out_file)
        print('Dump {} {} done!'.format(out_file, rdf.shape))
    else:
        utils.check_and_mkdir(out_file)
        pickle.dump(rdf, open(out_file, 'wb'))
        rdf.to_csv(out_file.replace('.pkl', '') + '.csv')
        print('Dump {} {} done!'.format(out_file + '.csv/pkl', rdf.shape))

    return rdf


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    print(args)

    # Load saved std data
    mci_prescription_taken_by_patient = pickle.load(open(os.path.join('output', 'mci_drug_taken_by_patient.pkl'), 'rb'))
    patient_dates = pickle.load(open(os.path.join('output', 'patient_dates.pkl'), 'rb'))
    patient_demo = pickle.load(open(os.path.join('output', 'patient_demo.pkl'), 'rb'))
    # data_info = (name_id, id_name, data)
    icd_to_ADcomorbidity, icd_to_ADcomorbidityName, ADcomorbidity_info = load_icd_to_ccw(
        'mapping/CCW_AD_comorbidity.json')
    atc2rxnorm = pickle.load(open(os.path.join('pickles', 'atcL2_rx.pkl'), 'rb'))  # ATC2DRUG.pkl

    print("**********build_patient_characteristics from all data**********")
    df = build_patient_characteristics(patient_demo, mci_prescription_taken_by_patient,
                                       patient_dates, icd_to_ADcomorbidityName,
                                       ADcomorbidity_info[0], args.dx_file,
                                       atc2rxnorm,
                                       out_file='output/patient_characteristics.pkl',
                                       drug_coding=args.drug_coding)

    # df = pickle.load(open('output/patient_characteristics.pkl', 'rb'))
    rdf = statistics_from_df(df, 'output/patient_characteristics_stats.pkl')
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
