from collections import defaultdict
from datetime import  datetime
import pickle
from tqdm import tqdm

#
# def pre_user_cohort_rx(prescription_taken_by_patient, patient_take_prescription, min_patients):
#     user_cohort_rx = defaultdict(dict)
#
#     for drug, taken_by_patient in tqdm(prescription_taken_by_patient.items()):
#         if len(taken_by_patient.keys()) >= min_patients:
#             for patient, take_dates in taken_by_patient.items():
#                 index_date = take_dates[0]
#                 patient_prescription_list = patient_take_prescription.get(patient)
#                 for prescription, dates_days in patient_prescription_list.items():
#                     dates = [datetime.strptime(date, '%m/%d/%Y') for date, days in dates_days]
#                     dates = sorted(dates)
#                     if drug_is_taken_in_baseline(index_date, dates):
#                         if drug not in user_cohort_rx:
#                             user_cohort_rx[drug][patient] = [prescription]
#                         else:
#                             if patient not in user_cohort_rx[drug]:
#                                 user_cohort_rx[drug][patient] = [prescription]
#                             else:
#                                 user_cohort_rx[drug][patient].append(prescription)
#
#     return user_cohort_rx
#
#
# def get_prescription_taken_times(index_date, dates, dates_2_days):
#     cnt = 0
#     for date in dates:
#         if (index_date - date).days - dates_2_days[date] > 0:
#             cnt += 1
#         else:
#             return cnt
#     return cnt
#
#
# # v1
# def drug_is_taken_in_baseline(index_date, dates):
#     for date in dates:
#         if (index_date - date).days > 0:
#             return True
#     return False


# v2
def pre_user_cohort_rx_v2(prescription_taken_by_patient, patient_take_prescription, min_patients):
    print('pre_user_cohort_rx_v2..., min_patients:', min_patients)
    user_cohort_rx = AutoVivification()

    for drug, taken_by_patient in tqdm(prescription_taken_by_patient.items()):
        if len(taken_by_patient.keys()) >= min_patients:
            for patient, take_dates in taken_by_patient.items():
                index_date = take_dates[0]
                patient_prescription_list = patient_take_prescription.get(patient)
                for prescription, dates_days in patient_prescription_list.items():
                    # dates = [datetime.strptime(date, '%m/%d/%Y') for date, days in dates_days]
                    dates = sorted(dates_days)
                    dates = drug_is_taken_in_baseline_v2(index_date, dates)  # 1. add = to include other drugs in the index date
                    # if dates:
                    if dates and (prescription != drug):  # not include the trial drug as baseline. other wise will include 1 record of trial drug at index date
                        for date in dates:
                            if drug not in user_cohort_rx:
                                user_cohort_rx[drug][patient][date] = [prescription]
                            else:
                                if patient not in user_cohort_rx[drug]:
                                    user_cohort_rx[drug][patient][date] = [prescription]
                                else:
                                    if date not in user_cohort_rx[drug][patient]:
                                        user_cohort_rx[drug][patient][date] = [prescription]
                                    else:
                                        user_cohort_rx[drug][patient][date].append(prescription)

    return user_cohort_rx


# v2 for LSTM - save timestamp
def drug_is_taken_in_baseline_v2(index_date, dates):
    res = []
    for date in dates:
        if (index_date - date).days >= 0:  #> 0: use >= 0 2021-07-6
            res.append(date)
    if len(res)>0:
        return res
    return False


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

