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


def shell_for_ml(cohort_dir_name, model):
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


def results_model_selection_for_ml(cohort_dir_name, model):
    N_DIM = 267
    cohort_size = pickle.load(open(r'../ipreprocess/output/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    name_cnt = sorted(cohort_size.items(), key=lambda x: x[1], reverse=True)
    drug_list_all = [drug.split('.')[0] for drug, cnt in name_cnt]
    dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
    drug_in_dir = set([x for x in os.listdir(dirname) if x.isdigit()])
    drug_list = [x for x in drug_list_all if x in drug_in_dir]  # in order

    for drug in drug_list:
        results = []
        for ctrl_type in ['random', 'atc']:
            for seed in range(0, 20):
                fname = dirname + drug + "/{}_S{}D200C{}_{}".format(drug, seed, ctrl_type, model)
                try:
                    df = pd.read_csv(fname + '_ALL-model-select.csv')
                except:
                    print('No file exisits: ', fname + '_ALL-model-select.csv')
                # 1. selected by AUC
                dftmp = df.sort_values(by=['val_auc', 'i'], ascending=[False, True])
                val_auc = dftmp.iloc[0, dftmp.columns.get_loc('val_auc')]
                val_auc_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]

                # 2. selected by val_max_smd_iptw
                dftmp = df.sort_values(by=['val_max_smd_iptw', 'i'], ascending=[True, True])
                val_max_smd = dftmp.iloc[0, dftmp.columns.get_loc('val_max_smd_iptw')]
                val_max_smd_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]

                # 3. selected by val_n_unbalanced_feat_iptw
                dftmp = df.sort_values(by=['val_n_unbalanced_feat_iptw', 'i'], ascending=[True, True])
                val_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('val_n_unbalanced_feat_iptw')]
                val_nsmd_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]

                # 4. selected by train_max_smd_iptw
                dftmp = df.sort_values(by=['train_max_smd_iptw', 'i'], ascending=[True, True])
                train_max_smd = dftmp.iloc[0, dftmp.columns.get_loc('train_max_smd_iptw')]
                train_max_smd_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]

                # 5. selected by train_n_unbalanced_feat_iptw
                dftmp = df.sort_values(by=['train_n_unbalanced_feat_iptw', 'i'], ascending=[True, True])
                train_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('train_n_unbalanced_feat_iptw')]
                train_nsmd_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]

                # 6. selected by trainval_max_smd_iptw
                dftmp = df.sort_values(by=['trainval_max_smd_iptw', 'i'], ascending=[True, True])
                trainval_max_smd = dftmp.iloc[0, dftmp.columns.get_loc('trainval_max_smd_iptw')]
                trainval_max_smd_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]

                # 7. selected by trainval_n_unbalanced_feat_iptw
                dftmp = df.sort_values(by=['trainval_n_unbalanced_feat_iptw', 'i'], ascending=[True, True])
                trainval_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('trainval_n_unbalanced_feat_iptw')]
                trainval_nsmd_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]

                # 8. FINAL: selected by trainval_n_unbalanced_feat_iptw + val AUC
                dftmp = df.sort_values(by=['trainval_n_unbalanced_feat_iptw', 'val_auc'], ascending=[True, False])
                trainval_final_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('trainval_n_unbalanced_feat_iptw')]
                trainval_final_valauc = dftmp.iloc[0, dftmp.columns.get_loc('val_auc')]
                trainval_final_finalnsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]

                results.append(["{}_S{}D200C{}_{}".format(drug, seed, ctrl_type, model), ctrl_type,
                                val_auc, val_auc_nsmd,
                                val_max_smd, val_max_smd_nsmd,
                                val_nsmd, val_nsmd_nsmd,
                                train_max_smd, train_max_smd_nsmd,
                                train_nsmd, train_nsmd_nsmd,
                                trainval_max_smd, trainval_max_smd_nsmd,
                                trainval_nsmd, trainval_nsmd_nsmd,
                                trainval_final_nsmd, trainval_final_valauc, trainval_final_finalnsmd
                                ])

        rdf = pd.DataFrame(results, columns=['fname', 'ctrl_type',
                                             "val_auc", "val_auc_nsmd",
                                             "val_max_smd", "val_max_smd_nsmd",
                                             "val_nsmd", "val_nsmd_nsmd",
                                             "train_max_smd", "train_max_smd_nsmd",
                                             "train_nsmd", "train_nsmd_nsmd",
                                             "trainval_max_smd", "trainval_max_smd_nsmd",
                                             "trainval_nsmd", "trainval_nsmd_nsmd",
                                             "trainval_final_nsmd", "trainval_final_valauc", "trainval_final_finalnsmd"
                                             ])
        rdf.to_csv(dirname + 'results/' + drug + '_model_selection.csv')

        for t in ['random', 'atc', 'all']:
            fig = plt.figure(figsize=(20,15))
            if t != 'all':
                idx = rdf['ctrl_type'] == t
            else:
                idx = rdf['ctrl_type'].notna()
            boxplot = rdf[idx].boxplot(column=["val_auc_nsmd", "val_max_smd_nsmd", "val_nsmd_nsmd", "train_max_smd_nsmd",
                                          "train_nsmd_nsmd", "trainval_max_smd_nsmd", "trainval_nsmd_nsmd",
                                          "trainval_final_finalnsmd"], rot=25, fontsize=15)
            boxplot.set_title("{}_S{}D200C{}_{}".format(drug, '0-19', t, model), fontsize=25)
            plt.xlabel("Model selection methods", fontsize=15)
            plt.ylabel("Distribution of #unbalanced_feat_iptw of boostrap experiments", fontsize=15)
            fig.savefig(dirname + 'results/' + drug + '_model_selection_boxplot-{}.png'.format(t))
            # plt.show()


    print()


if __name__ == '__main__':
    shell_for_ml(cohort_dir_name='save_cohort_all_loose', model='LIGHTGBM')
    # results_model_selection_for_ml(cohort_dir_name='save_cohort_all_loose', model='LR')
    print('Done')
