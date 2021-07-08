import argparse
import os
import time
import sys
# from utils import get_patient_init_date
from pre_cohort_ import exclude
from pre_cohort_rx_ import pre_user_cohort_rx_v2
from pre_cohort_dx_ import get_user_cohort_dx
from pre_demo_ import read_demo
from pre_outcome_ import pre_user_cohort_outcome
from user_cohort_ import pre_user_cohort_triplet
import pickle
import utils
import json


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # use default value
    parser.add_argument('--min_patients', default=20, type=int, help='minimum number of patients for each cohort. [Default 20]')  #500
    parser.add_argument('--min_age', default=50, type=int, help='minimum age at initiation date [Default 50 years old]')
    # Selection criterion: Value to play with
    parser.add_argument('--min_prescription', default=4, type=int, help='minimum times of prescriptions of each patient in each drug trial.')
    parser.add_argument('--exposure_interval', default=730, type=int,
                        help='Drug exposure period, minimum time interval for the first and last prescriptions')
    parser.add_argument('--followup', default=730, type=int, help='number of days of followup period, to define outcome')
    parser.add_argument('--baseline', default=365, type=int, help='number of days of baseline period, to collect covariates')
    parser.add_argument('--index_minus_init_min', default=0, type=int,
                        help='min <= (index_date - initiation_date).days <= max')
    parser.add_argument('--index_minus_init_max', default=99999999999999, type=int,
                        help='min <= (index_date - initiation_date).days <= max')
    parser.add_argument('--adrd_minus_index_min', default=0, type=int,
                        help='min bound <= (first_adrd_date - index_date).days')
    # others: folder and encodes
    parser.add_argument('--input_data', default='../data/CAD')
    parser.add_argument('--pickles', default='pickles')
    # parser.add_argument('--outcome_icd9', default='outcome_icd9.txt', help='outcome definition with ICD-9 codes')
    # parser.add_argument('--outcome_icd10', default='outcome_icd10.txt', help='outcome definition with ICD-10 codes')
    parser.add_argument('--save_cohort_all', default='save_cohort_all/')
    parser.add_argument('--dx_coding', choices=['ccs', 'ccw'], default='ccw')
    args = parser.parse_args()
    return args


def get_patient_list(min_patient, prescription_taken_by_patient):
    # logics: maybe should return drug list?
    print('get_patient_list...min_patient:', min_patient)
    patients_list = set()
    for drug, patients in prescription_taken_by_patient.items():
        if len(patients) >= min_patient:
            for patient in patients:
                patients_list.add(patient)
    return patients_list


def load_icd_to_ccw(path='mapping/CCW_to_use.json'):
    """ e.g. there exisit E93.50 v.s. E935.0.
            thus 5 more keys in the dot-ICD than nodot-ICD keys
            [('E9350', 2),
             ('E9351', 2),
             ('E8500', 2),
             ('E8502', 2),
             ('E8501', 2),
             ('31222', 1),
             ('31200', 1),
             ('3124', 1),
             ('F919', 1),
             ('31281', 1)]
    :param path:
    :return:
    """
    with open(path) as f:
        data = json.load(f)
        name_id = {x: str(i) for i, x in enumerate(data.keys())}
        n_dx = 0
        icd_ccwid = {}
        icddot_ccwid = {}
        for name, dx in data.items():
            n_dx += len(dx)
            for icd in dx:
                icd_ccwid[icd.strip().replace('.', '')] = name_id.get(name)
                icddot_ccwid[icd] = name_id.get(name)

        return icd_ccwid, name_id, icddot_ccwid


if __name__ == '__main__':
    start_time = time.time()
    # main(args=parse_args())
    args = parse_args()
    eligibility_criteria = {'min_age': args.min_age,
                            'min_prescription': args.min_prescription,
                            'exposure_interval': args.exposure_interval,
                            'followup': args.followup,
                            'baseline': args.baseline,
                            'index_minus_init_min': args.index_minus_init_min,
                            'index_minus_init_max': args.index_minus_init_max,
                            'adrd_minus_index_min': args.adrd_minus_index_min,
                            }
    dirname = os.path.dirname(args.save_cohort_all)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(args.save_cohort_all+r'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print(json.dumps(args.__dict__, sort_keys=False, indent=2))

    # with open(args.save_cohort_all+r'/commandline_args.txt', 'r') as f:
    #     args.__dict__ = json.load(f)

    print('**********Loading prescription data**********')
    # mci_drug_taken_by_patient_from_dispensing.pkl
    mci_prescription_taken_by_patient = pickle.load(
        open(os.path.join(args.pickles, 'mci_drug_taken_by_patient_from_dispensing_plus_prescribing.pkl'), 'rb'))

    print('**********Loading patient dates**********')
    patient_dates = pickle.load(
        open(os.path.join(args.pickles, 'patient_dates.pkl'), 'rb'))

    # patient_1stDX_date, patient_start_date = get_patient_init_date(args.input_data, args.pickles)
    print('**********Loading icd to ccs code**********')
    icd9_to_ccs = pickle.load(open(os.path.join(args.pickles, 'icd9_to_ccs.pkl'), 'rb'))
    icd10_to_ccs = pickle.load(open(os.path.join(args.pickles, 'icd10_to_ccs.pkl'), 'rb'))
    icd_to_ccw, _ccwname_id, _icddot_to_ccw = load_icd_to_ccw('mapping/CCW_to_use.json')

    print('**********Preprocessing patient data**********')
    # Input: drug --> patient --> [(date, supply day),]     patient_dates: patient --> [birth_date, other dates]
    # Output: save_prescription: drug --> patients --> [date1, date2, ...] sorted
    #         save_patient: patient --> drugs --> [date1, date2, ...] sorted
    # Notes: 20210608: not using followup here, define time_interval as time between first and last prescriptions
    save_prescription, save_patient = exclude(mci_prescription_taken_by_patient, patient_dates, eligibility_criteria)
    # args.time_interval, args.followup, args.baseline, args.min_prescription)

    # Patient set after exclusion (patients who take drugs which have >= min_patients patients)
    patient_list = get_patient_list(args.min_patients, save_prescription)

    # Should I only calculate the (rx, dx) covariates in the baseline period within only baseline-time windows?
    # drug --> patient --> dates --> prescription list in the Baseline
    save_cohort_rx = pre_user_cohort_rx_v2(save_prescription, save_patient, args.min_patients)
    # drug --> patient --> dates --> diagnosis list in the Baseline
    # _user_dx: patient-->dates-->diagnosis list for patients in patient_list, for debug, not used
    save_cohort_dx, _user_dx = get_user_cohort_dx(args.input_data, save_prescription, icd9_to_ccs, icd10_to_ccs,
                                                  icd_to_ccw, args.dx_coding, args.min_patients, patient_list)
    # patient --> demo feature tuple
    save_cohort_demo = read_demo(args.input_data, patient_list)  # get_user_cohort_demo(args.input_data, patient_list)
    # event type --> patient --> event date list
    # 2021-06-14 Only using AD, not using dementia: #{'AD': utils.is_AD, 'dementia': utils.is_dementia}
    save_cohort_outcome = pre_user_cohort_outcome(args.input_data, patient_list, {'AD': utils.is_AD})

    print('**********Generating patient cohort**********')
    # for each drug, dump (patients, [rx_codes, dx_codes, demo], outcome)
    # rx_codes, dx_codes: ordered list of varying-length list of codes (drug ingredient, ccs codes), not dates info kept
    pre_user_cohort_triplet(save_prescription, save_cohort_rx, save_cohort_dx,
                            save_cohort_outcome, save_cohort_demo, args.save_cohort_all,
                            patient_dates, args.followup)  # last 2 are new added
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

