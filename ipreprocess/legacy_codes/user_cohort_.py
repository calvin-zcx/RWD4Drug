from datetime import datetime
import pickle

import numpy as np
from tqdm import tqdm
import os
import utils
import pandas as pd


def pre_user_cohort_triplet(prescription_taken_by_patient, user_cohort_rx, user_cohort_dx,
                            save_cohort_outcome, user_cohort_demo, out_file_root,
                            patient_dates, followup):
    print('pre_user_cohort_triplet...')
    cohorts_size = dict()
    cohorts_size_pos = []
    for drug, taken_by_patient in tqdm(user_cohort_rx.items()):
        file_x = '{}/{}.pkl'.format(out_file_root, drug)
        triples = []
        for patient, taken_times in taken_by_patient.items():
            drug_dates = prescription_taken_by_patient.get(drug).get(patient)
            index_date = drug_dates[0]
            lastdx_date = patient_dates[patient][6]  # new added
            mci_date = patient_dates[patient][2]  # new added

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
            if dx:
                dx_codes = [list(dx_code) for date, dx_code in sorted(dx.items(), key=lambda x:x[0])]
            # keep this format, consistency with model and dateset. may change later
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

        dirname = os.path.dirname(file_x)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        pickle.dump(triples, open(file_x, 'wb'))

        cohorts_size['{}.pkl'.format(drug)] = len(triples)
        cohorts_size_pos.append(
            ('{}.pkl'.format(drug), len(triples),
             sum([x[-1][0]==1 for x in triples]),
             sum([x[-1][0]==-1 for x in triples]),
             sum([x[-1][0]==0 for x in triples]),
             np.mean([x[-1][1] for x in triples if x[-1][0] == 1]),
             np.mean([x[-1][1] for x in triples if x[-1][0] == -1]),
             np.mean([x[-1][1] for x in triples if x[-1][0] == 0]),
             )
        )
        # cohorts_size_pos.append(('{}.pkl'.format(drug), len(triples), sum([x[-1][0] for x in triples])))

    pickle.dump(cohorts_size, open('{}/cohorts_size.pkl'.format(out_file_root), 'wb'))
    df = pd.DataFrame(cohorts_size_pos, columns=['cohort_name', 'n_patients', 'n_pos', 'n_censor', 'n_neg',
                                                 't2e_pos_avg', 't2e_censor_avg', 't2e_neg_avg'])

    # add more info
    df['pos_ratio'] = df['n_pos'] / df['n_patients']
    from pre_drug_ import load_latest_rxnorm_info
    rxnorm_name, _ = load_latest_rxnorm_info()
    df['drug_name'] = df['cohort_name'].apply(lambda x: rxnorm_name[x.split('.')[0]])
    df.to_csv('{}/cohort_all_name_size_positive.csv'.format(out_file_root))

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


