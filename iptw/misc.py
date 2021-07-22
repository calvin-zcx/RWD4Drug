import os
import shutil
import zipfile

import urllib.parse
import urllib.request

import torch
import torch.utils.data
from dataset import *
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import Counter, defaultdict
import pandas as pd


def shell_for_ml(cohort_dir_name='save_cohort_all_loose', model='LR'):
    cohort_size = pickle.load(open(r'../ipreprocess/output/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    fo = open('shell_{}_{}.sh'.format(model, cohort_dir_name), 'w')  # 'a'

    name_cnt = sorted(cohort_size.items(), key=lambda x: x[1], reverse=True)
    fo.write('mkdir -p output/{}/{}/log\n'.format(cohort_dir_name, model))
    n = 0
    for x in name_cnt:
        k, v = x
        if v >= 500:
            drug = k.split('.')[0]
            for ctrl_type in ['random', 'atc']:
                for seed in range(0, 20):
                    cmd = "python main.py --data_dir ../ipreprocess/output/{}/ --treated_drug {} " \
                          "--controlled_drug {} --run_model LR --output_dir output/{}/{}/ --random_seed {} " \
                          "--drug_coding rxnorm --med_code_topk 200 --stats  " \
                          "2>&1 | tee output/{}/{}/log/{}_S{}D200C{}_{}.log\n".format(
                        cohort_dir_name, drug,
                        ctrl_type, cohort_dir_name, model, seed,
                        cohort_dir_name, model, drug, seed, ctrl_type, model)
                    fo.write(cmd)
                    n += 1

    fo.close()
    print('In total ', n, ' commands')


if __name__ == '__main__':
    shell_for_ml(cohort_dir_name='save_cohort_all_loose', model='LR')
    print('Done')
