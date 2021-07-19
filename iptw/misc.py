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


def prepare_bash():
    cohort_size = pickle.load(open(r'../ipreprocess/save_cohort_all/cohorts_size.pkl', 'rb'))
    fo = open('run_bash.cmd', 'w')  # 'a'
    for k, v in cohort_size.items():
        if v >= 200:
            name = k.split('.')[0]
            cmd = "python main.py --data_dir ../ipreprocess/save_cohort_all/  --controlled_drug random " \
                  "--pickles_dir  pickles --treated_drug_file {} " \
                  "--save_model_filename  save/save_model_test/{}.pt --random_seed 99 " \
                  "--outputs_lstm log/log_LSTM_test/{}.csv " \
                  "--outputs_lr  log/log_LR_test/{}.csv " \
                  "--save_db  save/save_db_test/{}.db \n".format(name,name,name,name,name) # *((name,)*5)
            fo.write(cmd)

    fo.close()


def prepare_bash_MLP():
    cohort_size = pickle.load(open(r'../ipreprocess/output/save_cohort_all_loose/cohorts_size.pkl', 'rb'))
    fo = open('run_bash_MLP.cmd', 'w')  # 'a'

    name_cnt = sorted(cohort_size.items(), key=lambda x: x[1], reverse=True)
    for x in name_cnt:
        k, v = x
    # for k, v in cohort_size.items():
        if v >= 500:
            name = k.split('.')[0]
            cmd = "python main.py --data_dir ../ipreprocess/output/save_cohort_all_loose/  --controlled_drug random " \
                  "--run_model MLP --treated_drug_file {} " \
                  "--save_model_filename  save/save_model_test/{}.pt " \
                  "--random_seed 1 " \
                  "--drug_coding rxnorm  " \
                  "--med_code_topk 150\n".format(name, name)  # *((name,)*5)
            fo.write(cmd)

    fo.close()


def prepare_bash_matter_arising():
    # --data_dir.. / ipreprocess / save_cohort_all / --controlled_drug random - -pickles_dir pickles  --treated_drug_file
    # 83367 - -save_model_filename
    # save / save_model_test / 83367.
    # pt  --random_seed
    # 99 - -outputs_lstm
    # log / log_LSTM_test / 83367.
    # csv - -outputs_lr
    # log / log_LR_test / 83367.
    # csv - -save_db
    # save / save_db_test / 83367.
    # db
    #
    n = 50
    fo = open('run_bash_83367.cmd', 'w')  # 'a'
    for i in range(n):
        cmd = "python main.py --data_dir ../ipreprocess/save_cohort_all/  --controlled_drug random " \
                  "--pickles_dir  pickles --treated_drug_file 83367 " \
                  "--save_model_filename  save/save_model_test/83367_{}.pt --random_seed {} " \
                  "--outputs_lstm log/log_LSTM_test/83367_{}.csv " \
                  "--outputs_lr  log/log_LR_test/83367_{}.csv " \
                  "--save_db  save/save_db_test/83367_{}.db \n".format(i, i, i, i, i)  # *((name,)*5)
        fo.write(cmd)

    fo.close()



def analyse_cohorts(patients_threshold=200):
    cohort_size = pickle.load(open(r'../ipreprocess/save_cohort_all/cohorts_size.pkl', 'rb'))
    cohorts = []
    for k, v in cohort_size.items():
        if v >= patients_threshold:
            name = k.split('.')[0]
            c = pickle.load(open(r'../ipreprocess/save_cohort_all/' + k, 'rb'))
            cohorts.append(c)

    return cohorts


def combine_results_old():
    df = pd.read_csv(r'../ipreprocess/save_cohort_all/cohort_all_name_size_positive_le200_results.csv')
    df['ATE_original'] = np.nan
    df['ATE_LR'] = np.nan
    df['LR_uf_ratio'] = np.nan
    df['ATE_LSTM'] = np.nan
    df['ATE_LSTM'] = np.nan
    df['LSTM_uf_ratio'] = np.nan

    for index, row in df.iterrows():
        drug = int(row['cohort_name'].split('.')[0])
        print(drug, flush=True)
        lstm = pd.read_csv(r'log/log_LSTM_test/{}.csv'.format(drug), index_col='drug')
        ATE_original = lstm.loc[drug, 'ATE_original']
        ATE_LSTM = lstm.loc[drug, 'ATE_weighted']

        lr = pd.read_csv(r'log/log_LR_test/{}.csv'.format(drug), index_col='drug')
        ATE_LR = lr.loc[drug, 'ATE_weighted']
        df.loc[index, 'ATE_original'] = ATE_original
        df.loc[index, 'ATE_LSTM'] = ATE_LSTM
        df.loc[index, 'ATE_LR'] = ATE_LR

        df.loc[index, 'LSTM_uf_ratio'] = lstm.loc[drug, 'n_unbalanced_feature_w']/lstm.loc[drug, 'n_feature']
        df.loc[index, 'LR_uf_ratio'] = lr.loc[drug,'n_unbalanced_feature_w'] / lr.loc[drug,'n_feature']

    df.to_csv(r'../ipreprocess/save_cohort_all/cohort_all_name_size_positive_le200_results_combined_withunbalanceratio.csv')
    print('Done')


def combine_results():
    # cohort_all_name_size_positive_le200_results.csv
    df = pd.read_csv(r'../ipreprocess/save_cohort_all/cohort_all_name_size_positive.csv')
    add_col = [
        'n_user','n_nonuser','max_unbalanced_original','max_unbalanced_weighted',
        'n_unbalanced_feature', 'n_unbalanced_feature_w','n_feature', 'UncorrectedEstimator_EY1_val',
        'UncorrectedEstimator_EY0_val', 'ATE_original','IPWEstimator_EY1_val','IPWEstimator_EY0_val','ATE_weighted']
    lr_col = ['max_unbalanced_original', 'max_unbalanced_weighted', 'n_unbalanced_feature', 'n_unbalanced_feature_w',
              'n_feature', 'IPWEstimator_EY1_val', 'IPWEstimator_EY0_val', 'ATE_weighted']
    for c in add_col:
        df[c] = np.nan
    for c in lr_col:
        df['LR_'+ c] = np.nan
    df['LSTM_UnbalFeat_ratio'] = np.nan
    df['LR_UnbalFeat_ratio'] = np.nan

    for index, row in df.iterrows():
        drug = int(row['cohort_name'].split('.')[0])
        print(drug, flush=True)
        if os.path.exists(r'log/log_LSTM_test/{}.csv'.format(drug)):
            lstm = pd.read_csv(r'log/log_LSTM_test/{}.csv'.format(drug), index_col='drug')
            for c in add_col:
                df.loc[index, c] = lstm.loc[drug, c]
            lr = pd.read_csv(r'log/log_LR_test/{}.csv'.format(drug), index_col='drug')
            for c in lr_col:
                df.loc[index, 'LR_'+c] = lr.loc[drug, c]

            df.loc[index, 'LSTM_UnbalFeat_ratio'] = lstm.loc[drug, 'n_unbalanced_feature_w']/lstm.loc[drug, 'n_feature']
            df.loc[index, 'LR_UnbalFeat_ratio'] = lr.loc[drug, 'n_unbalanced_feature_w'] / lr.loc[drug, 'n_feature']

    df.to_csv(r'../ipreprocess/save_cohort_all/cohort_all_name_size_positive_with_ALL_ATE_RESUTLS.csv')
    print('Done')


def analyse_MLP_results():
    cohort_size = pickle.load(open(r'../ipreprocess/save_cohort_all_mediate/cohorts_size.pkl', 'rb'))
    name_cnt = sorted(cohort_size.items(), key=lambda x:x[1], reverse=True)
    train = val = test = all = None
    for x in name_cnt:
        k, v = x
        if v >= 100:
            name = k.split('.')[0]
            # c = pickle.load(open(r'../ipreprocess/save_cohort_all/' + k, 'rb'))
            # cohorts.append(c)
            df = pd.read_csv('save/save_model_test_mediate/{}.pt_results_MLP_all.csv'.format(name), index_col=0)

            if train is not None:
                train = train.append(df.loc['train',:])
            else:
                train = df.loc[['train'],:]

            if val is not None:
                val = val.append(df.loc[['val'],:])
            else:
                val = df.loc[['val'],:]

            if test is not None:
                test = test.append(df.loc[['test'],:])
            else:
                test = df.loc[['test'],:]

            if all is not None:
                all = all.append(df.loc[['all'],:])
            else:
                all = df.loc[['all'],:]

    with pd.ExcelWriter('save/save_model_test_mediate/MLP_all_results.xlsx') as f:
        train.to_excel(f, sheet_name='train')
        val.to_excel(f, sheet_name='val')
        test.to_excel(f, sheet_name='test')
        all.to_excel(f, sheet_name='all')
    # train.to_csv('save/save_model_test/MLP_train_results.csv')
    # val.to_csv('save/save_model_test/MLP_val_results.csv')
    # test.to_csv('save/save_model_test/MLP_test_results.csv')
    # all.to_csv('save/save_model_test/MLP_all_results.csv')


if __name__ == '__main__':
#     # prepare_bash()
#     # cohorts = analyse_cohorts(patients_threshold=200)
#     # combine_results()
#     # prepare_bash_matter_arising()
#     # prepare_bash_MLP()
#     # analyse_MLP_results()
#     # rxnorm_name, df = load_latest_rxnorm_info()
#     # with open(r'pickles/rxnorm_label_mapping.pkl', 'rb') as f:
#     #     rxnorm_name2 = pickle.load(f)
#
#     # build atc to rxnorm mapping!
#     # df, ar = map_rxnorm_and_atc()
#     load_rxnorm_to_atc()
    print()

