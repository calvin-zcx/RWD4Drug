from collections import defaultdict
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


def read_demo(DATA_DIR, patient_list=None):
    """
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
    data top lines:
    PATID,BIRTH_DATE,BIRTH_TIME,SEX,SEXUAL_ORIENTATION,GENDER_IDENTITY,HISPANIC,RACE,BIOBANK_FLAG,PAT_PREF_LANGUAGE_SPOKEN,RAW_SEX,RAW_SEXUAL_ORIENTATION,RAW_GENDER_IDENTITY,RAW_HISPANIC,RAW_RACE,RAW_PAT_PREF_LANGUAGE_SPOKEN,UPDATED,SOURCE,ZIP_CODE,LAST_ENCOUNTERID
    11e75060a017514ea3fa0050569ea8fb,1928-09-14,NULL,F,NULL,NULL,N,03,N,NULL,F,NULL,NULL,B,B,NULL,2020-06-01 14:15:29,FLM,338534901,YvJLcr8CS1WhqgO/f/UW	2017-02-12
    11e75060a49a95e6bb6a0050569ea8fb,1936-09-11,NULL,F,NULL,NULL,N,03,N,NULL,F,NULL,NULL,B,B,NULL,2020-06-01 14:15:29,FLM,349472302,YvJLcr0CS1SqrgS7f/gZ	2015-08-06
    11e75060a5eb4d8cbc3a0050569ea8fb,1932-04-25,04:00,F,NI,NI,N,05,N,ENG,30000000|362,NULL,NULL,30000000|312507,30000000|309316,30000000|151,2019-08-29 15:47:23,AVH,327017752,NULL
    11e75060a8f9918c929b0050569ea8fb,1942-06-12,NULL,F,NULL,NULL,N,03,N,NULL,F,NULL,NULL,B,B,NULL,2020-06-01 14:15:29,FLM,34220,YvJLcrECS1etrQC7e/Yd	2019-12-13
    11e75060ab1a545688ed0050569ea8fb,1952-09-14,00:00,F,NULL,NULL,N,05,N,ENG,Female,NULL,NULL,Not Hispanic or Latino,White,ENGLISH,2020-06-02 19:46:50,LNK,32207-5787,"cfhOcL0HT1aprQI=	2020-02-19
    "
    11e75060acc6e09ea7850050569ea8fb,1977-05-06,00:00,F,NULL,NULL,N,05,N,ENG,Female,NULL,NULL,Not Hispanic or Latino,White,ENGLISH,2020-06-02 19:47:02,LNK,32246-3767,"cfhOcL0LQl+gogE=	2020-04-30
    "
    11e75060b17a2f6aa7850050569ea8fb,1952-04-10,00:00,F,NI,NI,N,03,N,ENG,F,NULL,NULL,B,B,ENGLISH,2020-06-01 14:15:29,LNK,322443700,"YvJLcr0CS1Ksowe8f/Ae	2015-06-01
"
    :param DATA_DIR:
    :param file_name:
    :return: id_demo[patid] = (sex, bdate, race, zipcode)
    """
    print('read_demo...')
    path = os.path.join(DATA_DIR, 'DEMOGRAPHIC.csv')
    df_demo = pd.read_csv(path, skiprows=[1])  # new added skiprows=[1], new data 1st -------
    id_demo = {}
    if patient_list is not None:
        patient_list = set(patient_list)

    for index, row in tqdm(df_demo.iterrows(), total=len(df_demo)):
        patid = row['PATID']
        sex = 0 if row['SEX'] == 'F' else 1  # Female 0, Male, and UN 1
        bdate = utils.str_to_datetime(row['BIRTH_DATE'])  # datetime(*list(map(int, row['BIRTH_DATE'].split('-'))))  # year-month-day
        # birth_year = bdate.year
        try:
            race = int(row['RACE'])
        except ValueError:
            race = 0  # denote all OT, UN, NI as 0
        zipcode = row['ZIP_CODE']

        if (patient_list is None) or (patid in patient_list):
            id_demo[patid] = (bdate, sex, race, zipcode)

    return id_demo


def build_patient_dates(DATA_DIR):
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
    df_demo = pd.read_csv(os.path.join(DATA_DIR, 'DEMOGRAPHIC.csv'), skiprows=[1])
    # 0-birth date
    # 1-start diagnosis date, 2-initial mci diagnosis date,
    # 3-first AD diagnosis date, 4-first dementia diagnosis date
    # 5-first ADRD diagnosis, 6-last diagnosis

    patient_dates = {pid: [utils.str_to_datetime(bdate), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                     for (pid, bdate) in zip(df_demo['PATID'], df_demo['BIRTH_DATE'])}
    n_records = 0
    n_no_date = 0
    min_date = datetime.max
    max_date = datetime.min
    with open(os.path.join(DATA_DIR, 'DIAGNOSIS.csv'), 'r') as f:  # 'MCI_DIANGOSIS.csv'
        col_name = next(csv.reader(f))  # read from first non-name row, above code from 2nd, wrong
        for row in tqdm(csv.reader(f)):  # , total=34554738):
            n_records += 1
            patid, ad_date, dx, dx_type = row[1], row[4], row[6], row[7]
            dx_date = row[8]

            if patid.startswith('--'):
                print('Skip {}th row: {}'.format(n_records, row))
                continue
            # # datetime(*list(map(int, date.split('-'))))

            # First use ADMIT_DATE , then DX_DATE, then discard record
            if (ad_date != '') and (ad_date != 'NULL'):
                date = utils.str_to_datetime(ad_date)
            elif (dx_date != '') and (dx_date != 'NULL'):
                date = utils.str_to_datetime(dx_date)
            else:
                n_no_date += 1
                continue

            if date > max_date:
                max_date = date
            if date < min_date:
                min_date = date

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

    pickle.dump(patient_dates,
                open('pickles/patient_dates.pkl', 'wb'))

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

    df.to_csv('debug/patient_dates.csv')
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


# future add patient in list function
if __name__ == '__main__':
    DATA_DIR = 'DELETE-ADD-LATER'
    start_time = time.time()
    # id_demo = read_demo(DATA_DIR)
    patient_dates = build_patient_dates(DATA_DIR=DATA_DIR)

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))