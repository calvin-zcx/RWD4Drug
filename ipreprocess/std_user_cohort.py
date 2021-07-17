from datetime import datetime
import pickle

import numpy as np
from tqdm import tqdm
import os
import utils
import pandas as pd
from univariate_statistics import statistics_from_df


def pre_user_cohort_triplet(prescription_taken_by_patient, user_cohort_rx, user_cohort_dx,
                            save_cohort_outcome, user_cohort_demo, out_file_root,
                            patient_dates, followup, drug_name,
                            comorbidityid_2_name, drug_criterion):
    print('pre_user_cohort_triplet...')
    cohorts_size = dict()
    cohorts_size_pos = []

    for drug, taken_by_patient in tqdm(user_cohort_rx.items()):
        # file_x = '{}/{}.pkl'.format(out_file_root, drug)
        file_x = os.path.join(out_file_root, '{}.pkl'.format(drug))
        triples = []
        # initialize patient characteristics for this drug trial
        drug_patient = pd.DataFrame(0,
                                    columns=['bdate', 'sex', 'race', 'MCI_date', 'age_of_MCI', 'AD_date', 'AD',
                                             'antidiabetic', 'antihypertensives', 'Alcohol Use Disorders',
                                             'Anxiety Disorders', 'Depression', 'Diabetes', 'Heart Failure',
                                             'Hyperlipidemia', 'Hypertension', 'Ischemic Heart Disease', 'Obesity',
                                             'Stroke / Transient Ischemic Attack', 'Tobacco Use',
                                             'Traumatic Brain Injury and Nonpsychotic Mental Disorders due to Brain Damage',
                                             'Sleep disorders', 'Periodontitis', 'Menopause'],
                                    index=taken_by_patient.keys())
        drug_patient.index.name = 'patid'
        drug_patient['MCI_date'] = np.nan
        drug_patient['age_of_MCI'] = np.nan
        drug_patient['AD_date'] = np.nan
        n_no_drug = 0
        n_no_diagnosis = 0
        for patient, taken_times in taken_by_patient.items():
            drug_dates = prescription_taken_by_patient.get(drug).get(patient)
            index_date = drug_dates[0]
            lastdx_date = patient_dates[patient][6]  # new added
            mci_date = patient_dates[patient][2]  # new added
            bdate = patient_dates[patient][0]  # new added

            dx = user_cohort_dx.get(drug).get(patient)  # timestamps --> diagnosis set

            demo = user_cohort_demo.get(patient)  # a tuple
            # demo_feature_vector = get_demo_feature_vector(demo, index_date)  # a list
            demo_feature_vector = get_demo_feature_vector(demo, index_date, mci_date)  # demo_feature_vector: [age, sex, race, days_since_mci]

            outcome_feature_vector = []   # a two dimensional vec for AD and demential outcome
            outcome_time_vector = []  # new added
            for outcome_name, outcome_map in save_cohort_outcome.items():
                outcome_dates = outcome_map.get(patient, [])
                # dates =  [utils.str_to_datetime(date.strip('\n')) for date in outcome_dates if date]  # datetime.strptime(date.strip('\n'), '%m/%d/%Y')
                # dates = sorted(dates)
                outcome_dates_sorted = sorted(outcome_dates)
                flg, t2event = get_outcome_feature_vector(outcome_dates_sorted, index_date, lastdx_date, followup, drug_dates)
                outcome_feature_vector.append(flg)
                outcome_time_vector.append(t2event)

            outcome = max(outcome_feature_vector)  # get AD/demential, or censoring, one digit bool
            outcome_t2e = min(outcome_time_vector)  # return both event time or censor time, last diagnosis in the system

            # a list of lists of drugs or diagnosis (before index date), sorted by their dates.
            # but not using their date timestamps for modeling
            # may use later. what impact? temporal outcome instead of 0/1?
            rx_codes, dx_codes = [], []
            if taken_times:  # LATER: add time for modeling later
                rx_codes = [rx_code for date, rx_code in sorted(taken_times.items(), key=lambda x:x[0])]  # sort by key (date)
            else:  # newly added for debugging
                n_no_drug += 1
                print('No drug baseline covariates for', patient, 'in building triples of ', drug)

            if dx:
                dx_codes = [list(dx_code) for date, dx_code in sorted(dx.items(), key=lambda x:x[0])]
            else:  # newly added for debugging
                n_no_diagnosis += 1
                print('No diagnosis baseline covariates for', patient, 'in building triples of ', drug)

            # 2021-07-08 I will keep patients who have no CCW codes in their baseline. No ccw codes are also informative
            # AD or AD/RD ccw codes are not used, because they should not appear in baseline

            # keep this format, consistency with PSModels and dateset. may change later
            # Change demo, may add race later
            # triple = (patient, [rx_codes, dx_codes, demo_feature_vector[0], demo_feature_vector[1]], outcome)  # only use 0-1 right now
            # demo_feature_vector: [age, sex, race, days_since_mci]
            # 20210608 add one more demo feature: days since mci
            # 20210615 add time 2 event outcome, make it and flag as a tuple.
            triple = (patient,
                      [rx_codes, dx_codes, demo_feature_vector[0], demo_feature_vector[1], demo_feature_vector[3]],
                      (outcome, outcome_t2e))

            # triple = (patient,
            #           [rx_codes, dx_codes, demo_feature_vector],
            #           (outcome, outcome_t2e))
            triples.append(triple)

            # build patient characteristics of this drug trial
            drug_patient.at[patient, 'bdate'] = demo[0]
            drug_patient.at[patient, 'sex'] = demo[1]
            drug_patient.at[patient, 'race'] = demo[2]
            if (not pd.isna(mci_date)) and (not pd.isna(bdate)):
                drug_patient.at[patient, 'MCI_date'] = mci_date
                drug_patient.at[patient, 'age_of_MCI'] = mci_date.year - bdate.year  # (mci_date - bdate).days / 365.

            drug_patient.at[patient, 'AD_date'] = outcome_t2e
            drug_patient.at[patient, 'AD'] = outcome

            for rx_list in rx_codes:
                for rx in rx_list:
                    for key, fc in drug_criterion.items():
                        if fc(rx):
                            drug_patient.at[patient, key] += 1

            for dx_list in dx_codes:
                for dx in dx_list:
                    if dx in comorbidityid_2_name:
                        ccwname = comorbidityid_2_name[dx]
                        drug_patient.at[patient, ccwname] += 1

        dirname = os.path.dirname(file_x)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        pickle.dump(triples, open(file_x, 'wb'))

        # pickle.dump(drug_patient, open(file_x.replace('.pkl', '') + '_characteristics.pkl', 'wb'))
        drug_patient.to_csv(file_x.replace('.pkl', '') + '_characteristics.csv')
        drug_patient_stats = statistics_from_df(drug_patient, file_x.replace('.pkl', '') + '_characteristics_stats.csv')

        cohorts_size['{}.pkl'.format(drug)] = len(triples)
        cohorts_size_pos.append(
            ('{}.pkl'.format(drug), len(triples),
             sum([x[-1][0]==1 for x in triples]),
             sum([x[-1][0]==-1 for x in triples]),
             sum([x[-1][0]==0 for x in triples]),
             np.mean([x[-1][1] for x in triples if x[-1][0] == 1]),
             np.mean([x[-1][1] for x in triples if x[-1][0] == -1]),
             np.mean([x[-1][1] for x in triples if x[-1][0] == 0]),
             n_no_drug,
             n_no_diagnosis
             )
        )
        # cohorts_size_pos.append(('{}.pkl'.format(drug), len(triples), sum([x[-1][0] for x in triples])))

    pickle.dump(cohorts_size, open(os.path.join(out_file_root, 'cohorts_size.pkl'), 'wb'))  # '{}/cohorts_size.pkl'.format(out_file_root)
    df = pd.DataFrame(cohorts_size_pos, columns=['cohort_name', 'n_patients', 'n_pos', 'n_censor', 'n_neg',
                                                 't2e_pos_avg', 't2e_censor_avg', 't2e_neg_avg',
                                                 "n_no_drug", "n_no_diagnosis"])

    # add more info
    df['pos_ratio'] = df['n_pos'] / df['n_patients']
    df['drug_name'] = df['cohort_name'].apply(lambda x: drug_name[x.split('.')[0]])
    df.sort_values(by=['n_patients'], inplace=True, ascending=False)
    df.to_csv(os.path.join(out_file_root, 'cohort_all_name_size_positive.csv'))  # '{}/cohort_all_name_size_positive.csv'.format(out_file_root)

    print('...pre_user_cohort_triplet done!')


def get_outcome_feature_vector(outcome_dates, index_date, lastdx_date, followup, drug_dates):
    # may delete <= 730 criterion??? CAD is happen is defined within 2 years. AD is chronic, should delet
    # need to make sure the first diagnosis after index date?
    #  or such people are excluded in the previous part
    if outcome_dates:
        assert outcome_dates[-1] <= lastdx_date
    # # 1. event happened during followup period
    # for date in dates:
    #     if date > index_date and (date - index_date).days <= followup:  # 730:
    #         return 1, (date - index_date).days  # positive events, event time
    # # else case of above: no events, or events after followup
    # # 2. no events in followup period,  and lastdx within followup period
    # if (lastdx_date - index_date).days < followup:
    #     return -1, (lastdx_date - index_date).days  # censored events, censored time
    # else:  # 3. lastdx after followup, and no events in followup period or events after followup
    #     return 0, followup   #(lastdx_date - index_date).days  # negative events,
    # for date in dates:
    #     if date > index_date:  # 730:
    #         return 1, (date - index_date).days  # positive event, event time
    # return 0, (lastdx_date - index_date).days  # censored event, censored time

    # simple binary case, no censoring because in user_cohort_.py
    # (dates[-1] - dates[0]).days >= interval == followup
    # lastdrug_date - index_date >= followup, but
    # lastdx_date - index_date may not
    for date in outcome_dates:
        if (date > index_date) and ((date - index_date).days <= followup):  # 730:
            return 1, (date - index_date).days  # positive events, event time
    # else case of above: no events, or events after followup
    # return 0, followup    # if drugdate[-1] - [0] >= interval which == followup, (lastdx_date - index_date).days  >= followup
    last_date = max(lastdx_date, drug_dates[-1])
    if (last_date - index_date).days > followup:
        return 0, followup  # negative events, event time == followup time
    else:
        return -1, (last_date - index_date).days  # censoring events, censoring time


def get_rx_feature_vector(taken_times, RX2id, size):
    feature_vector = [0] * size
    for rx in taken_times:
        if rx in RX2id:
            id = RX2id.get(rx)
            feature_vector[id] = 1

    return feature_vector


def get_dx_feature_vector(dx, CCS2id, size):

    feature_vector = [0] * size
    not_find = set()
    for code in dx:
        for c in code:
            if c in CCS2id:
                id = CCS2id.get(c)
                feature_vector[id] = 1

    return feature_vector, not_find


def get_demo_feature_vector(demo, index_date, mci_date):
    if not demo:
        return [0, 0, 0, 0]
    # db, sex = demo
    bdate, sex, race, zipcode = demo
    age = index_date.year - bdate.year
    days_since_mci = (index_date - mci_date).days
    # not using zipcode here, may add later
    return [age, sex, race, days_since_mci]


