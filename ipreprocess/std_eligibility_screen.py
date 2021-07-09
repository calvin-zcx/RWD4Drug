from collections import defaultdict
from datetime import datetime
import pickle
from utils import str_to_datetime
import pandas as pd


def exclude(prescription_taken_by_patient, patient_dates, eligibility_criteria):
    # interval, followup, baseline, min_prescription):  # patient_1stDX_date, patient_start_date
    # Input: drug --> patient --> [(date, supply day),]     patient_dates: patient --> [birth_date, other dates]
    # Output: save_prescription: drug --> patients --> [date1, date2, ...] sorted
    #         save_patient: patient --> drugs --> [date1, date2, ...] sorted
    # print('exclude... interval:{}, followup:{}, baseline:{}, min_prescription:{}'.format(
    #     interval, followup, baseline, min_prescription))
    print('eligibility_criteria:\n', eligibility_criteria)
    prescription_taken_by_patient_exclude = defaultdict(dict)
    # patient_take_prescription_exclude = defaultdict(dict)
    patient_take_prescription = defaultdict(dict)

    for drug, taken_by_patient in prescription_taken_by_patient.items():
        for patient, take_times in taken_by_patient.items():
            # no need for both if date and days, days are not '', they have int value, confused with value 0
            # actually no need for date, date is not '' according to the empirical data, but add this is OK, more rigid
            dates = [str_to_datetime(date) for (date, days) in take_times if date != '']  # and days datetime.strptime(date, '%m/%d/%Y')
            dates = sorted(dates)
            dates_days = {str_to_datetime(date): int(days) for (date, days) in take_times if
                          date != ''}  #  and days datetime.strptime(date, '%m/%d/%Y')

            # patient_dates: {0: 'birth_date',  1: '1st_diagnosis_date', 2: '1st_mci_date',
            #  3: '1st_AD_date', 4: '1st_dementia_date',  5: '1st_ADRD_date',  6: 'last_diagnosis_date'}
            birth_date = patient_dates[patient][0]
            if pd.isna(patient_dates[patient][5]):
                first_adrd_date = datetime.max
            else:
                first_adrd_date = patient_dates[patient][5]
            # DX = patient_1stDX_date.get(patient, datetime.max)
            initial_date = patient_dates[patient][2]
            index_date = dates[0]
            # start_date = patient_start_date.get(patient, datetime.max)
            start_date = patient_dates[patient][1]
            # criteria_0_is_valid(index_date, birth_date)
            # 2021-07-06 use patient_take_prescription rather than patient_take_prescription_exclude for
            # later baseline covariates construction.
            patient_take_prescription[patient][drug] = dates
            if criteria_0_is_valid(initial_date, birth_date, eligibility_criteria) and \
                    criteria_1_is_valid(index_date, initial_date, eligibility_criteria) and \
                    criteria_2_is_valid(dates, dates_days, eligibility_criteria) and \
                    criteria_3_is_valid(index_date, start_date, eligibility_criteria) and \
                    criteria_4_is_valid(index_date, first_adrd_date, eligibility_criteria):
                prescription_taken_by_patient_exclude[drug][patient] = dates  # drug cohorts, should use exclude
                # patient_take_prescription_exclude[patient][drug] = dates
                # #patient_take_prescription_exclude maybe wrong. used for building baseline. should use all drug info, instead of only exclude information

    # return prescription_taken_by_patient_exclude, patient_take_prescription_exclude
    return prescription_taken_by_patient_exclude, patient_take_prescription


def criteria_0_is_valid(initial_date, birth_date, eligibility_criteria):
    # (index or initial age) >= 50
    # use initial date 2021-06-15
    age = eligibility_criteria['min_age']
    # return (initial_date - birth_date).days >= age*365
    return initial_date.year - birth_date.year >= age


def criteria_1_is_valid(index_date, initial_date, eligibility_criteria):
    # The first prescription is after the MCI initiation date, and
    # MCI happens close to Index date, say within one year
    # return (index_date - DX).days > -90
    left = eligibility_criteria['index_minus_init_min']
    right = eligibility_criteria['index_minus_init_max']
    return left <= (index_date - initial_date).days <= right


def criteria_2_is_valid(dates, dates_days, eligibility_criteria):
    # Persistently prescribed the treated drug
    # >= (e.g. 2 years (730 days)) follow-up period
    # Why followup - 89?  some delay effects of drug prescription
    # Should lift this criteria for survival analysis?! or at least 1 year?
    # Caution: This criteria explicitly indicates at least 2 prescriptions. If deleted, add additional #>= 2 criteria

    ## NMI paper
    # if (dates[-1] - dates[0]).days <= (followup - 89):  # followup:  # (followup - 89):
    #     return False
    # for i in range(1, len(dates)):
    #     sup_day = dates_days.get(dates[i - 1])  # How to deal with -1?
    #     if (dates[i] - dates[i - 1]).days - sup_day > interval:
    #         return False

    ## IBM paper?
    # follow IBM paper: at least two prescriptions, >= 30? (using 180 days here) days apart
    # define interval as the days apart between first and last prescriptions
    # only using followup to define the outcome, not using supply_days here
    # set interval == followup for binary outcome, no censoring
    # for time2event outcome, maybe set interval to 6 months? and change outcome definition in the uder_cohort_.py file
    min_prescription = eligibility_criteria['min_prescription']
    interval = eligibility_criteria['exposure_interval']
    if len(dates) < min_prescription or (dates[-1] - dates[0]).days < interval:  # set to the same with followup time, with a 3 month elastic time
        return False

    return True


def criteria_3_is_valid(index_date, start_date, eligibility_criteria):
    # >= (e.g. 1 year) baseline period
    # should I add criteria that MCI initialization must occur in the baseline period?
    baseline = eligibility_criteria['baseline']
    return (index_date - start_date).days >= baseline


def criteria_4_is_valid(index_date, first_adrd_date, eligibility_criteria):
    # Exclude patient who had diagnoses of AD/dementia before the index date and 3 months after the index date
    adrd_minus_index_min = eligibility_criteria['adrd_minus_index_min']
    return (first_adrd_date - index_date).days >= adrd_minus_index_min
