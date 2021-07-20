import sys
# for linux env.
sys.path.insert(0,'..')
from datetime import datetime
import os
import pandas as pd
from tqdm import tqdm
import time
import pickle
import argparse
import csv
import utils
import numpy as np
import functools
print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--demo_file', default=r'../data/florida/demographic.csv',
                        help='input demographics file with std format')
    parser.add_argument('--dx_file', default=r'../data/florida/diagnosis.csv',
                        help='input diagnosis file with std format')
    # parser.add_argument('-out_file', default=r'output/patient_dates.pkl')
    args = parser.parse_args()
    return args


def read_demo(data_file, out_file=''):
    """
    :param data_file: input demographics file with std format
    :param out_file: output id_demo[patid] = (sex, bdate, race, zipcode) pickle
    # :param patient_list: a patient ID set for whose demographic features returned. None for returning all patients
    :return: id_demo[patid] = (sex, bdate, race, zipcode)
    :Notice:
        1.florida data: e.g:
        PATID,SEX,BIRTH_DATE,ZIP_CODE,RACE
        11e75060991379ccbf840050569ea8fb,F,1965-01-21,333161883,03
        11e75060992c22ec88ed0050569ea8fb,F,1940-06-02,331695504,03
        df_demo['SEX'].value_counts():
            F     33498
            M     26692
            UN        2
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
        2. marketscan data: e.g.
        ENROLID,SEX,DOBYR,REGION
        3636091801,2,1926,4
        2217960801,1,1931,3
           SEX  1 means male and 2 means female)
           DOBYR shows the year in which the patient has been born
           REGION shows the geographical region (1: northeast, 2: north central, 3: south, 4: west, 5: unknown).
    """
    start_time = time.time()
    n_read = 0
    n_invalid = 0
    id_demo = {}
    with open(data_file, 'r') as f:
        col_name = next(csv.reader(f))  # read from first non-name row, above code from 2nd, wrong
        print("read from ", data_file, col_name)
        for row in csv.reader(f):
            n_read += 1
            patid = row[0]
            sex = 0 if (row[1] == 'F' or row[1] == '2') else 1  # Female 0, Male, and UN 1
            try:
                bdate = utils.str_to_datetime(row[2])
            except:
                print('invalid birth date in ', n_read, row)
                n_invalid += 1
                continue

            zipcode = row[3]
            try:
                race = int(row[4])
            except:
                race = 0  # denote all OT, UN, NI as 0

            # if (patient_set is None) or (patid in patient_set):
            id_demo[patid] = (bdate, sex, race, zipcode)
    print('read {} rows, len(id_demo) {}'.format(n_read, len(id_demo)))
    print('n_invalid rows: ', n_invalid)
    print('read_demo done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    if out_file:
        utils.check_and_mkdir(out_file)
        pickle.dump(id_demo, open(out_file, 'wb'))

        df = pd.DataFrame(id_demo).T
        df = df.rename(columns={0: 'birth_date',
                                1: 'sex',
                                2: 'race',
                                3: 'zipcode'})
        df.to_csv(out_file.replace('.pkl', '') + '.csv')
        print('dump done to {}'.format(out_file))

    return id_demo


def build_patient_dates(demo_file, dx_file, out_file):
    """
    # Start date: the first date in the EHR database
    # Initiation date: The first date when patients were diagnosed with MCI.
    # Index date:  the date of the first prescription of the assigned drug  (can not determine here. need
    #              mci_drug_taken_by_patient_from_dispensing_plus_prescribing later)
    # last date: last date of drug, or diagnosis? Use drug time in the cohort building and selection code part

    Caution:
    # First use DX_DATE, then ADMIT_DATE, then discard record
    # Only add validity check datetime(1990,1,1)<date<datetime(2030,1,1) for updating 1st and last diagnosis date

    # Input
    :param DATA_DIR:
    :return:
    """
    print("******build_patient_dates*******")
    id_demo = read_demo(demo_file)
    # 0-birth date
    # 1-start diagnosis date, 2-initial mci diagnosis date,
    # 3-first AD diagnosis date, 4-first dementia diagnosis date
    # 5-first ADRD diagnosis, 6-last diagnosis

    patient_dates = {pid: [val[0], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                     for pid, val in id_demo.items()}  # (pid, bdate) in zip(df_demo['PATID'], df_demo['BIRTH_DATE'])
    n_records = 0
    n_no_date = 0
    n_no_pid = 0
    min_date = datetime.max
    max_date = datetime.min
    with open(dx_file, 'r') as f:
        col_name = next(csv.reader(f))
        print('read from ', dx_file, col_name)
        for row in tqdm(csv.reader(f)):
            n_records += 1
            patid, date, dx = row[0], row[1], row[2]
            # dx_type = row[3]

            # First use ADMIT_DATE , then DX_DATE, then discard record
            if (date == '') or (date == 'NULL'):
                n_no_date += 1
                continue
            else:
                date = utils.str_to_datetime(date)

            if date > max_date:
                max_date = date
            if date < min_date:
                min_date = date

            if patid not in patient_dates:
                n_no_pid += 1
                continue

            # 1-start diagnosis date
            if pd.isna(patient_dates[patid][1]) or date < patient_dates[patid][1]:
                if datetime(1990, 1, 1) < date < datetime(2030, 1, 1):
                    patient_dates[patid][1] = date

            #  2-initial mci diagnosis date
            if utils.is_mci(dx):
                if pd.isna(patient_dates[patid][2]) or date < patient_dates[patid][2]:
                    patient_dates[patid][2] = date

            # 3-firs AD diagnosis date
            if utils.is_AD(dx):
                if pd.isna(patient_dates[patid][3]) or date < patient_dates[patid][3]:
                    patient_dates[patid][3] = date

            # 4-first dementia diagnosis date
            if utils.is_dementia(dx):
                if pd.isna(patient_dates[patid][4]) or date < patient_dates[patid][4]:
                    patient_dates[patid][4] = date

            # 5-first AD or dementia diagnosis date
            if utils.is_AD(dx) or utils.is_dementia(dx):
                if pd.isna(patient_dates[patid][5]) or date < patient_dates[patid][5]:
                    patient_dates[patid][5] = date

            # 6-last diagnosis
            if pd.isna(patient_dates[patid][6]) or date > patient_dates[patid][6]:
                if datetime(1990, 1, 1) < date < datetime(2030, 1, 1):
                    patient_dates[patid][6] = date

    print('len(patient_dates)', len(patient_dates))
    print('n_records', n_records)
    print('n_no_date', n_no_date)
    print('n_no_pid', n_no_pid)

    utils.check_and_mkdir(out_file)
    pickle.dump(patient_dates, open(out_file, 'wb'))

    df = pd.DataFrame(patient_dates).T
    df = df.rename(columns={0: 'birth_date',
                            1: '1st_diagnosis_date',
                            2: '1st_mci_date',
                            3: '1st_AD_date',
                            4: '1st_dementia_date',
                            5: '1st_ADRD_date',
                            6: 'last_diagnosis_date'})

    idx_ADRD = pd.notna(df['1st_ADRD_date'])
    df.loc[idx_ADRD, 'MCI<ADRD'] = df.loc[idx_ADRD, '1st_mci_date'] < df.loc[idx_ADRD, '1st_ADRD_date']

    idx_AD = pd.notna(df['1st_AD_date'])
    df.loc[idx_AD, 'MCI<AD'] = df.loc[idx_AD, '1st_mci_date'] < df.loc[idx_AD, '1st_AD_date']

    idx_RD = pd.notna(df['1st_dementia_date'])
    df.loc[idx_RD, 'Dementia<AD'] = df.loc[idx_RD, '1st_mci_date'] < df.loc[idx_RD, '1st_dementia_date']

    df.to_csv(out_file.replace('.pkl', '') + '.csv')
    print('dump done')
    return patient_dates


def plot_MCI_to_ADRD():
    import matplotlib.pyplot as plt
    import pandas as pd
    pdates = pd.read_csv(os.path.join('debug', 'patient_dates.csv'))
    idx = pd.notna(pdates['1st_ADRD_date'])
    pt = pd.to_datetime(pdates.loc[idx, '1st_ADRD_date']) - pd.to_datetime(pdates.loc[idx, '1st_mci_date'])
    pt.apply(lambda x: x.days/365).hist()  # bins=25
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    print(args)
    id_demo = read_demo(args.demo_file, r'output/patient_demo.pkl')
    patient_dates = build_patient_dates(args.demo_file, args.dx_file, r'output/patient_dates.pkl')
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
