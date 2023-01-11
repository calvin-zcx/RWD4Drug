import sys

# for linux env.
sys.path.insert(0, '..')
import argparse
import os
import time
from std_eligibility_screen import exclude
from std_pre_cohort_rx import pre_user_cohort_rx_v2
from std_pre_cohort_dx import get_user_cohort_dx
from std_user_cohort import pre_user_cohort_triplet
import pickle
import utils
import json
import pandas as pd
from tqdm import tqdm
import functools

print = functools.partial(print, flush=True)
from collections import defaultdict
from datetime import datetime
import pickle
from utils import str_to_datetime


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')

    parser.add_argument('--followup', default=1825, type=int,
                        help='number of days of followup period, to define outcome')
    parser.add_argument('--save_cohort_all', default='output_marketscan/save_cohort_all_loose_f5yrs/')

    args = parser.parse_args()
    return args


def get_outcome_feature_vector(ad1st_date, index_date, lastdx_date, followup, drug_dates):
    if pd.notna(ad1st_date):
        if (ad1st_date > index_date) and ((ad1st_date - index_date).days <= followup):  # 730:
            return 1, (ad1st_date - index_date).days  # positive events, event time
    # else case of above: no events, or events after followup
    # return 0, followup    # if drugdate[-1] - [0] >= interval which == followup, (lastdx_date - index_date).days  >= followup
    last_date = max(lastdx_date, drug_dates[-1])
    if (last_date - index_date).days > followup:
        return 0, followup  # negative events, event time == followup time
    else:
        return -1, (last_date - index_date).days  # censoring events, censoring time


if __name__ == '__main__':
    start_time = time.time()
    # main(args=parse_args())
    args = parse_args()
    print(args)

    dirname = os.path.dirname(args.save_cohort_all)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    print('**********Loading prescription data**********')
    # mci_drug_taken_by_patient_from_dispensing.pkl
    # mci_prescription_taken_by_patient = pickle.load(
    #     open(os.path.join('output_marketscan', 'mci_drug_taken_by_patient.pkl'), 'rb'))

    # mci_prescription_taken_by_patient_sort = defaultdict(dict)
    # for drug, taken_by_patient in tqdm(mci_prescription_taken_by_patient.items()):
    #     for patient, take_times in taken_by_patient.items():
    #         # no need for both if date and days, days are not '', they have int value, confused with value 0
    #         # actually no need for date, date is not '' according to the empirical data, but add this is OK, more rigid
    #         dates = [str_to_datetime(date) for (date, days) in take_times if date != '']  # and days datetime.strptime(date, '%m/%d/%Y')
    #         dates = sorted(dates)
    #         dates_days = {str_to_datetime(date): int(days) for (date, days) in take_times if
    #                       date != ''}  #  and days datetime.strptime(date, '%m/%d/%Y')
    #         mci_prescription_taken_by_patient_sort[drug][patient] = dates  # drug cohorts, should use exclude
    #
    # pickle.dump(mci_prescription_taken_by_patient_sort,
    #             open(os.path.join('output_marketscan', '_mci_drug_taken_by_patient_sorted_.pkl'), 'wb'))

    # pat_drugdates = defaultdict(list)
    # for drug, taken_by_patient in tqdm(mci_prescription_taken_by_patient.items()):
    #     for patient, take_times in taken_by_patient.items():
    #         # no need for both if date and days, days are not '', they have int value, confused with value 0
    #         # actually no need for date, date is not '' according to the empirical data, but add this is OK, more rigid
    #         dates = [str_to_datetime(date) for (date, days) in take_times if date != '']  # and days datetime.strptime(date, '%m/%d/%Y')
    #         pat_drugdates[patient].extend(dates)
    #
    # for patid, dates in pat_drugdates.items():
    #     # add a set operation to reduce duplicates
    #     # sorted returns a sorted list
    #     dates_sorted = sorted(set(dates))
    #     pat_drugdates[patid] = dates_sorted
    # print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    # pickle.dump(pat_drugdates,
    #             open(os.path.join('output_marketscan', '_patient_drugdates_sorted_.pkl'), 'wb'))

    mci_prescription_taken_by_patient_sort = pickle.load(
        open(os.path.join('output_marketscan', '_mci_drug_taken_by_patient_sorted_.pkl'), 'rb'))

    # pat_drugdates = pickle.dump(
    #     open(os.path.join('output_marketscan', '_patient_drugdates_sorted_.pkl'), 'rb'))
    #
    print('**********Loading patient demo dates**********')
    patient_dates = pickle.load(open(os.path.join('output_marketscan', 'patient_dates.pkl'), 'rb'))
    patient_demo = pickle.load(open(os.path.join('output_marketscan', 'patient_demo.pkl'), 'rb'))

    source_dir = r'output_marketscan/save_cohort_all_loose/'
    cohort_size = pickle.load(open(r'output_marketscan/save_cohort_all_loose/cohorts_size.pkl', 'rb'))
    name_cnt = sorted(cohort_size.items(), key=lambda x: x[1], reverse=True)
    drug_list_all = [drug.split('.')[0] for drug, cnt in name_cnt]

    for drug in tqdm(drug_list_all):
        print('drug:', drug)
        triples = pickle.load(open(source_dir + drug + '.pkl', 'rb'))
        triples_new = []
        for triple in triples:
            patient, covs, outcome_ = triple
            outcome, outcome_t2e = outcome_
            # triple = (patient,
            #           [rx_codes, dx_codes, demo_feature_vector[0], demo_feature_vector[1], demo_feature_vector[3]],
            #           (outcome, outcome_t2e))
            lastdx_date = patient_dates[patient][6]  # new added
            mci_date = patient_dates[patient][2]  # new added
            bdate = patient_dates[patient][0]  # new added
            ad1st_date = patient_dates[patient][3]

            drug_dates = mci_prescription_taken_by_patient_sort.get(drug).get(patient)
            index_date = drug_dates[0]
            flg, t2event = get_outcome_feature_vector(ad1st_date, index_date, lastdx_date, args.followup, drug_dates)

            # if outcome == 1:
            #     print(outcome_t2e, ad1st_date, index_date, drug_dates[-1])
            #
            # if (outcome == 0) and (flg == 1):
            #     print(outcome_t2e, ad1st_date, index_date, drug_dates[-1])
            #     print('-->', t2event)

            triple_new = (patient, covs, (flg, t2event))
            triples_new.append(triple_new)

        if not os.path.exists(args.save_cohort_all):
            os.makedirs(args.save_cohort_all)
        pickle.dump(triples, open(args.save_cohort_all + drug + '.pkl', 'wb'))

    print('Done')