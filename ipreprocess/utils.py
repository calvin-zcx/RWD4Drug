from datetime import datetime
import os
# import pandas as pd
# import time
# import pickle
# import csv
# import numpy as np
import re
import json


def check_and_mkdir(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print('make dir:', dirname)
    else:
        print(dirname, 'exists, no change made')


def str_to_datetime(s):
    # input: e.g. '2016-10-14'
    # output: datetime.datetime(2016, 10, 14, 0, 0)
    # Other choices:
    #       pd.to_datetime('2016-10-14')  # very very slow
    #       datetime.strptime('2016-10-14', '%Y-%m-%d')   #  slow
    # ymd = list(map(int, s.split('-')))
    ymd = list(map(int, re.split(r'[-\/:.]', s)))
    assert (len(ymd) == 3) or (len(ymd) == 1)
    if len(ymd) == 3:
        assert 1 <= ymd[1] <= 12
        assert 1 <= ymd[2] <= 31
    elif len(ymd) == 1:
        ymd = ymd + [1, 1]  # If only year, set to Year-Jan.-1st
    return datetime(*ymd)


def is_mci(code):
    """
        ICD9
            331.83 Mild cognitive impairment, so stated
            294.9	 Unspecified persistent mental disorders due to conditions classified elsewhere
        ICD10
            G31.84 Mild cognitive impairment, so stated
            F09	Unspecified mental disorder due to known physiological condition

     :param code: code string to test
     :return:  true or false
     """
    assert isinstance(code, str)
    code_set = ('331.83', '294.9', 'G31.84', 'F09', '33183', '2949', 'G3184')
    return code.startswith(code_set)


def is_AD(code):
    """
    ICD9
        331.0		Alzheimer's disease
    ICD10
        G30     Alzheimer's disease
        G30.0	Alzheimer's disease with early onset
        G30.1 	Alzheimer's disease with late onset
        G30.8	Other Alzheimer's disease
        G30.9	Alzheimer's disease, unspecified

    :param code: code string to test
    :return:  true or false
    """
    assert isinstance(code, str)
    code_set = ('331.0', '3310', 'G30')
    return code.startswith(code_set)


def is_dementia(code):
    """
    ICD9
        294.10, 294.11 , 294.20, 294.21
        290.-   (all codes and variants in 290.- tree)
    ICD10
        F01.- All codes and their variants in this trees
        F02.- All codes and their variants in this trees
        F03.- All codes and their variants in this trees
    :param code: code string to test
    :return:  true or false
    """
    assert isinstance(code, str)
    code_set = ('294.10', '294.11', '294.20', '294.21', '2941', '29411', '2942', '29421')
    code_set += ('290',)
    code_set += ('F01', 'F02', 'F03')
    return code.startswith(code_set)


def load_icd_to_ccw(path):
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
    :param path: e.g. 'mapping/CCW_to_use_enriched.json'
    :return:
    """
    with open(path) as f:
        data = json.load(f)
        print('len(ccw_codes):', len(data))
        name_id = {x: str(i) for i, x in enumerate(data.keys())}
        id_name = {v:k for k, v in name_id.items()}
        n_dx = 0
        icd_ccwid = {}
        icd_ccwname = {}
        icddot_ccwid = {}
        for name, dx in data.items():
            n_dx += len(dx)
            for icd in dx:
                icd_no_dot = icd.strip().replace('.', '')
                if icd_no_dot:  # != ''
                    icd_ccwid[icd_no_dot] = name_id.get(name)
                    icd_ccwname[icd_no_dot] = name
                    icddot_ccwid[icd] = name_id.get(name)
        data_info = (name_id, id_name, data)
        return icd_ccwid, icd_ccwname, data_info
