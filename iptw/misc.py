import os
import shutil
import zipfile

import torch
import torch.utils.data
from dataset import *
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

import numpy as np
import csv
from collections import Counter, defaultdict
import pandas as pd
from utils import check_and_mkdir
from scipy import stats
import re
import itertools
import functools
import random
import seaborn as sns

print = functools.partial(print, flush=True)

MAX_NO_UNBALANCED_FEATURE = 5

np.random.seed(0)
random.seed(0)


def IQR(s):
    return [np.quantile(s, .5), np.quantile(s, .25), np.quantile(s, .75)]


def stringlist_2_list(s):
    r = s.strip('][').replace(',', ' ').split()
    r = list(map(float, r))
    return r


def boot_matrix(z, B):
    """Bootstrap sample

    Returns all bootstrap samples in a matrix"""
    z = np.array(z).flatten()
    n = len(z)  # sample size
    idz = np.random.randint(0, n, size=(B, n))  # indices to pick for all boostrap samples
    return z[idz]


def bootstrap_mean_ci(x, B=1000, alpha=0.05):
    n = len(x)
    # Generate boostrap distribution of sample mean
    xboot = boot_matrix(x, B=B)
    sampling_distribution = xboot.mean(axis=1)
    quantile_confidence_interval = np.percentile(sampling_distribution, q=(100 * alpha / 2, 100 * (1 - alpha / 2)))
    std = sampling_distribution.std()
    # if plot:
    #     plt.hist(sampling_distribution, bins="fd")
    return quantile_confidence_interval, std


def bootstrap_mean_pvalue(x, expected_mean=0., B=1000):
    """
    Ref:
    1. https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#cite_note-:0-1
    2. https://www.tau.ac.il/~saharon/StatisticsSeminar_files/Hypothesis.pdf
    3. https://github.com/mayer79/Bootstrap-p-values/blob/master/Bootstrap%20p%20values.ipynb
    4. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html?highlight=one%20sample%20ttest
    Bootstrap p values for one-sample t test
    Returns boostrap p value, test statistics and parametric p value"""
    n = len(x)
    orig = stats.ttest_1samp(x, expected_mean)
    # Generate boostrap distribution of sample mean
    x_boots = boot_matrix(x - x.mean() + expected_mean, B=B)
    x_boots_mean = x_boots.mean(axis=1)
    t_boots = (x_boots_mean - expected_mean) / (x_boots.std(axis=1, ddof=1) / np.sqrt(n))
    p = np.mean(t_boots >= orig[0])
    p_final = 2 * min(p, 1 - p)
    # Plot bootstrap distribution
    # if plot:
    #     plt.figure()
    #     plt.hist(x_boots_mean, bins="fd")
    return p_final, orig


def bootstrap_mean_pvalue_2samples(x, y, equal_var=False, B=1000):
    """
    Bootstrap hypothesis testing for comparing the means of two independent samples
    Ref:
    1. https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#cite_note-:0-1
    2. https://www.tau.ac.il/~saharon/StatisticsSeminar_files/Hypothesis.pdf
    3. https://github.com/mayer79/Bootstrap-p-values/blob/master/Bootstrap%20p%20values.ipynb
    4. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html?highlight=one%20sample%20ttest
    Bootstrap p values for one-sample t test
    Returns boostrap p value, test statistics and parametric p value"""
    n = len(x)
    orig = stats.ttest_ind(x, y, equal_var=equal_var)
    pooled_mean = np.concatenate((x, y), axis=None).mean()

    xboot = boot_matrix(x - x.mean() + pooled_mean,
                        B=B)  # important centering step to get sampling distribution under the null
    yboot = boot_matrix(y - y.mean() + pooled_mean, B=B)
    sampling_distribution = stats.ttest_ind(xboot, yboot, axis=1, equal_var=equal_var)[0]

    if np.isnan(orig[1]):
        p_final = np.nan
    else:
        # Calculate proportion of bootstrap samples with at least as strong evidence against null
        p = np.mean(sampling_distribution >= orig[0])
        # RESULTS
        # print("p value for null hypothesis of equal population means:")
        # print("Parametric:", orig[1])
        # print("Bootstrap:", 2 * min(p, 1 - p))
        p_final = 2 * min(p, 1 - p)

    return p_final, orig


def shell_for_ml(cohort_dir_name, model, niter=50, min_patients=500, stats=True, more_para=''):
    cohort_size = pickle.load(open(r'../ipreprocess/output/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    fo = open('shell_{}_{}.sh'.format(model, cohort_dir_name), 'w')  # 'a'
    name_cnt = sorted(cohort_size.items(), key=lambda x: x[1], reverse=True)

    # load others:
    df = pd.read_excel(r'../data/repurposed_AD_under_trials_20200227.xlsx', dtype=str)
    added_drug = []
    for index, row in df.iterrows():
        rx = row['rxcui']
        gpi = row['gpi']
        if pd.notna(rx):
            rx = [x + '.pkl' for x in re.split('[,;+]', rx)]
            added_drug.extend(rx)

        if pd.notna(gpi):
            gpi = [x + '.pkl' for x in re.split('[,;+]', gpi)]
            added_drug.extend(gpi)

    print('len(added_drug): ', len(added_drug))
    print(added_drug)

    fo.write('mkdir -p output/{}/{}/log\n'.format(cohort_dir_name, model))
    n = 0
    for x in name_cnt:
        k, v = x
        if (v >= min_patients) or (k in added_drug):
            drug = k.split('.')[0]
            for ctrl_type in ['random', 'atc']:
                for seed in range(0, niter):
                    cmd = "python main.py --data_dir ../ipreprocess/output/{}/ --treated_drug {} " \
                          "--controlled_drug {} --run_model {} --output_dir output/{}/{}/ --random_seed {} " \
                          "--drug_coding rxnorm --med_code_topk 200 {} {} " \
                          "2>&1 | tee output/{}/{}/log/{}_S{}D200C{}_{}.log\n".format(
                        cohort_dir_name, drug,
                        ctrl_type, model, cohort_dir_name, model, seed, '--stats' if stats else '', more_para,
                        cohort_dir_name, model, drug, seed, ctrl_type, model)
                    fo.write(cmd)
                    n += 1

    fo.close()
    print('In total ', n, ' commands')


def split_shell_file(fname, divide=2, skip_first=1):
    f = open(fname, 'r')
    content_list = f.readlines()
    n = len(content_list)
    n_d = np.ceil((n - skip_first) / divide)
    seg = [0, ] + [int(i * n_d + skip_first) for i in range(1, divide)] + [n]
    for i in range(divide):
        fout_name = fname.split('.')
        fout_name = ''.join(fout_name[:-1]) + '-' + str(i) + '.' + fout_name[-1]
        fout = open(fout_name, 'w')
        for l in content_list[seg[i]:seg[i + 1]]:
            fout.write(l)
        fout.close()
    print('dump done')


def results_model_selection_for_ml(cohort_dir_name, model, drug_name, niter=50):
    cohort_size = pickle.load(open(r'../ipreprocess/output/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    name_cnt = sorted(cohort_size.items(), key=lambda x: x[1], reverse=True)
    drug_list_all = [drug.split('.')[0] for drug, cnt in name_cnt]
    dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
    drug_in_dir = set([x for x in os.listdir(dirname) if x.isdigit()])
    drug_list = [x for x in drug_list_all if x in drug_in_dir]  # in order
    check_and_mkdir(dirname + 'results/')

    for drug in drug_list:
        results = []
        for ctrl_type in ['random', 'atc']:
            for seed in range(0, niter):
                fname = dirname + drug + "/{}_S{}D200C{}_{}".format(drug, seed, ctrl_type, model)
                try:
                    df = pd.read_csv(fname + '_ALL-model-select.csv')
                except:
                    print('No file exisits: ', fname + '_ALL-model-select.csv')

                # 1. selected by AUC
                dftmp = df.sort_values(by=['val_auc', 'i'], ascending=[False, True])
                val_auc = dftmp.iloc[0, dftmp.columns.get_loc('val_auc')]
                val_auc_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]
                val_auc_testauc = dftmp.iloc[0, dftmp.columns.get_loc('test_auc')]

                # 2. selected by val_max_smd_iptw
                dftmp = df.sort_values(by=['val_max_smd_iptw', 'i'], ascending=[True, True])
                val_maxsmd = dftmp.iloc[0, dftmp.columns.get_loc('val_max_smd_iptw')]
                val_maxsmd_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]
                val_maxsmd_testauc = dftmp.iloc[0, dftmp.columns.get_loc('test_auc')]

                # 3. selected by val_n_unbalanced_feat_iptw
                dftmp = df.sort_values(by=['val_n_unbalanced_feat_iptw', 'i'], ascending=[True, True])
                val_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('val_n_unbalanced_feat_iptw')]
                val_nsmd_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]
                val_nsmd_testauc = dftmp.iloc[0, dftmp.columns.get_loc('test_auc')]

                # 4. selected by train_max_smd_iptw
                dftmp = df.sort_values(by=['train_max_smd_iptw', 'i'], ascending=[True, True])
                train_maxsmd = dftmp.iloc[0, dftmp.columns.get_loc('train_max_smd_iptw')]
                train_maxsmd_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]
                train_maxsmd_testauc = dftmp.iloc[0, dftmp.columns.get_loc('test_auc')]

                # 5. selected by train_n_unbalanced_feat_iptw
                dftmp = df.sort_values(by=['train_n_unbalanced_feat_iptw', 'i'], ascending=[True, True])
                train_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('train_n_unbalanced_feat_iptw')]
                train_nsmd_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]
                train_nsmd_testauc = dftmp.iloc[0, dftmp.columns.get_loc('test_auc')]

                # 6. selected by trainval_max_smd_iptw
                dftmp = df.sort_values(by=['trainval_max_smd_iptw', 'i'], ascending=[True, True])
                trainval_maxsmd = dftmp.iloc[0, dftmp.columns.get_loc('trainval_max_smd_iptw')]
                trainval_maxsmd_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]
                trainval_maxsmd_testauc = dftmp.iloc[0, dftmp.columns.get_loc('test_auc')]

                # 7. selected by trainval_n_unbalanced_feat_iptw
                dftmp = df.sort_values(by=['trainval_n_unbalanced_feat_iptw', 'i'], ascending=[True, True])
                trainval_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('trainval_n_unbalanced_feat_iptw')]
                trainval_nsmd_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]
                trainval_nsmd_testauc = dftmp.iloc[0, dftmp.columns.get_loc('test_auc')]

                # 8. FINAL: selected by trainval_n_unbalanced_feat_iptw + val AUC
                dftmp = df.sort_values(by=['trainval_n_unbalanced_feat_iptw', 'val_auc'], ascending=[True, False])
                trainval_final_nsmd = dftmp.iloc[0, dftmp.columns.get_loc('trainval_n_unbalanced_feat_iptw')]
                trainval_final_valauc = dftmp.iloc[0, dftmp.columns.get_loc('val_auc')]
                trainval_final_finalnsmd = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]
                trainval_final_testnauc = dftmp.iloc[0, dftmp.columns.get_loc('test_auc')]

                results.append(["{}_S{}D200C{}_{}".format(drug, seed, ctrl_type, model), ctrl_type,
                                val_auc, val_auc_nsmd, val_auc_testauc,
                                val_maxsmd, val_maxsmd_nsmd, val_maxsmd_testauc,
                                val_nsmd, val_nsmd_nsmd, val_nsmd_testauc,
                                train_maxsmd, train_maxsmd_nsmd, train_maxsmd_testauc,
                                train_nsmd, train_nsmd_nsmd, train_nsmd_testauc,
                                trainval_maxsmd, trainval_maxsmd_nsmd, trainval_maxsmd_testauc,
                                trainval_nsmd, trainval_nsmd_nsmd, trainval_nsmd_testauc,
                                trainval_final_nsmd, trainval_final_valauc, trainval_final_finalnsmd,
                                trainval_final_testnauc,
                                ])

        rdf = pd.DataFrame(results, columns=['fname', 'ctrl_type',
                                             "val_auc", "val_auc_nsmd", "val_auc_testauc",
                                             "val_maxsmd", "val_maxsmd_nsmd", "val_maxsmd_testauc",
                                             "val_nsmd", "val_nsmd_nsmd", "val_nsmd_testauc",
                                             "train_maxsmd", "train_maxsmd_nsmd", "train_maxsmd_testauc",
                                             "train_nsmd", "train_nsmd_nsmd", "train_nsmd_testauc",
                                             "trainval_maxsmd", "trainval_maxsmd_nsmd", "trainval_maxsmd_testauc",
                                             "trainval_nsmd", "trainval_nsmd_nsmd", "trainval_nsmd_testauc",
                                             "trainval_final_nsmd", "trainval_final_valauc", "trainval_final_finalnsmd",
                                             "trainval_final_testnauc",
                                             ])

        rdf.to_csv(dirname + 'results/' + drug + '_model_selection.csv')

        for t in ['random', 'atc', 'all']:
            # fig = plt.figure(figsize=(20, 15))
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 18))
            if t != 'all':
                idx = rdf['ctrl_type'] == t
            else:
                idx = rdf['ctrl_type'].notna()
            boxplot = rdf[idx].boxplot(column=["val_auc_nsmd", "val_maxsmd_nsmd", "val_nsmd_nsmd", "train_maxsmd_nsmd",
                                               "train_nsmd_nsmd", "trainval_maxsmd_nsmd", "trainval_nsmd_nsmd",
                                               "trainval_final_finalnsmd"], rot=25, fontsize=15, ax=ax1)
            ax1.axhline(y=5, color='r', linestyle='-')
            boxplot.set_title("{}-{}_S{}D200C{}_{}".format(drug, drug_name.get(drug), '0-19', t, model), fontsize=25)
            # plt.xlabel("Model selection methods", fontsize=15)
            ax1.set_ylabel("#unbalanced_feat_iptw of boostrap experiments", fontsize=20)
            # fig.savefig(dirname + 'results/' + drug + '_model_selection_boxplot-{}-allnsmd.png'.format(t))
            # plt.show()

            # fig = plt.figure(figsize=(20, 15))
            boxplot = rdf[idx].boxplot(column=["val_auc_testauc", "val_maxsmd_testauc", "val_nsmd_testauc",
                                               "train_maxsmd_testauc", "train_nsmd_testauc", "trainval_maxsmd_testauc",
                                               "trainval_nsmd_testauc", 'trainval_final_testnauc'], rot=25, fontsize=15,
                                       ax=ax2)
            # plt.axhline(y=0.5, color='r', linestyle='-')
            # boxplot.set_title("{}-{}_S{}D200C{}_{}".format(drug, drug_name.get(drug), '0-19', t, model), fontsize=25)
            ax2.set_xlabel("Model selection methods", fontsize=20)
            ax2.set_ylabel("test_auc of boostrap experiments", fontsize=20)
            plt.tight_layout()
            fig.savefig(dirname + 'results/' + drug + '_model_selection_boxplot-{}.png'.format(t))
            plt.clf()
    print()


def results_model_selection_for_ml_step2(cohort_dir_name, model, drug_name):
    cohort_size = pickle.load(open(r'../ipreprocess/output/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    name_cnt = sorted(cohort_size.items(), key=lambda x: x[1], reverse=True)
    drug_list_all = [drug.split('.')[0] for drug, cnt in name_cnt]
    dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
    drug_in_dir = set([x for x in os.listdir(dirname) if x.isdigit()])
    drug_list = [x for x in drug_list_all if x in drug_in_dir]  # in order
    check_and_mkdir(dirname + 'results/')

    writer = pd.ExcelWriter(dirname + 'results/summarized_model_selection_{}.xlsx'.format(model), engine='xlsxwriter')
    for t in ['random', 'atc', 'all']:
        results = []
        for drug in drug_list:
            rdf = pd.read_csv(dirname + 'results/' + drug + '_model_selection.csv')

            if t != 'all':
                idx = rdf['ctrl_type'] == t
            else:
                idx = rdf['ctrl_type'].notna()

            r = [drug, drug_name.get(drug, '')]
            col_name = ['drug', 'drug_name']
            # zip(["val_auc_nsmd", "val_maxsmd_nsmd", "val_nsmd_nsmd", "train_maxsmd_nsmd",
            #      "train_nsmd_nsmd", "trainval_maxsmd_nsmd", "trainval_nsmd_nsmd",
            #      "trainval_final_finalnsmd"],
            #     ["val_auc_testauc", "val_maxsmd_testauc", "val_nsmd_testauc",
            #      "train_maxsmd_testauc", "train_nsmd_testauc", "trainval_maxsmd_testauc",
            #      "trainval_nsmd_testauc", 'trainval_final_testnauc'])
            for c1, c2 in zip(["val_auc_nsmd", "val_maxsmd_nsmd", "trainval_final_finalnsmd"],
                              ["val_auc_testauc", "val_maxsmd_testauc", 'trainval_final_testnauc']):
                nsmd = rdf.loc[idx, c1]
                auc = rdf.loc[idx, c2]

                nsmd_med = IQR(nsmd)[0]
                nsmd_iqr = IQR(nsmd)[1:]

                nsmd_mean = nsmd.mean()
                nsmd_mean_ci, nsmd_mean_std = bootstrap_mean_ci(nsmd, alpha=0.05)

                success_rate = (nsmd <= MAX_NO_UNBALANCED_FEATURE).mean()
                success_rate_ci, success_rate_std = bootstrap_mean_ci(nsmd <= MAX_NO_UNBALANCED_FEATURE, alpha=0.05)

                auc_med = IQR(auc)[0]
                auc_iqr = IQR(auc)[1:]

                auc_mean = auc.mean()
                auc_mean_ci, auc_mean_std = bootstrap_mean_ci(auc, alpha=0.05)

                r.extend([nsmd_med, nsmd_iqr,
                          nsmd_mean, nsmd_mean_ci, nsmd_mean_std,
                          success_rate, success_rate_ci, success_rate_std,
                          auc_med, auc_iqr, auc_mean, auc_mean_ci, auc_mean_std])
                col_name.extend(
                    ["nsmd_med-" + c1, "nsmd_iqr-" + c1, "nsmd_mean-" + c1, "nsmd_mean_ci-" + c1, "nsmd_mean_std-" + c1,
                     "success_rate-" + c1, "success_rate_ci-" + c1, "success_rate_std-" + c1,
                     "auc_med-" + c2, "auc_iqr-" + c2, "auc_mean-" + c2, "auc_mean_ci-" + c2, "auc_mean_std-" + c2])

            x = np.array(rdf.loc[idx, "trainval_final_finalnsmd"] <= MAX_NO_UNBALANCED_FEATURE, dtype=np.float)
            y1 = np.array(rdf.loc[idx, "val_auc_nsmd"] <= MAX_NO_UNBALANCED_FEATURE, dtype=np.float)
            y2 = np.array(rdf.loc[idx, "val_maxsmd_nsmd"] <= MAX_NO_UNBALANCED_FEATURE, dtype=np.float)
            p1, test_orig1 = bootstrap_mean_pvalue_2samples(x, y1)
            p2, test_orig2 = bootstrap_mean_pvalue_2samples(x, y2)
            p3, test_orig3 = bootstrap_mean_pvalue_2samples(y1, y2)
            r.extend([p1, test_orig1[1], p2, test_orig2[1], p3, test_orig3[1]])
            col_name.extend(
                ['pboot-succes-final-vs-auc', 'p-succes-final-vs-auc',
                 'pboot-succes-final-vs-maxsmd', 'p-succes-final-vs-maxsmd',
                 'pboot-succes-auc-vs-maxsmd', 'p-succes-auc-vs-maxsmd'])

            results.append(r)
        df = pd.DataFrame(results, columns=col_name)
        df.to_excel(writer, sheet_name=t)
    writer.save()
    print()


def results_model_selection_for_ml_step2More(cohort_dir_name, model, drug_name):
    cohort_size = pickle.load(open(r'../ipreprocess/output/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    name_cnt = sorted(cohort_size.items(), key=lambda x: x[1], reverse=True)
    drug_list_all = [drug.split('.')[0] for drug, cnt in name_cnt]
    dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
    drug_in_dir = set([x for x in os.listdir(dirname) if x.isdigit()])
    drug_list = [x for x in drug_list_all if x in drug_in_dir]  # in order
    check_and_mkdir(dirname + 'results/')

    writer = pd.ExcelWriter(dirname + 'results/summarized_model_selection_{}-More.xlsx'.format(model), engine='xlsxwriter')
    for t in ['random', 'atc', 'all']:
        results = []
        for drug in drug_list:
            rdf = pd.read_csv(dirname + 'results/' + drug + '_model_selection.csv')

            if t != 'all':
                idx = rdf['ctrl_type'] == t
            else:
                idx = rdf['ctrl_type'].notna()

            r = [drug, drug_name.get(drug, '')]
            col_name = ['drug', 'drug_name']
            for c1, c2 in zip(
                    ["val_auc_nsmd", "val_maxsmd_nsmd", "val_nsmd_nsmd", "train_maxsmd_nsmd",
                     "train_nsmd_nsmd", "trainval_maxsmd_nsmd", "trainval_nsmd_nsmd", "trainval_final_finalnsmd"],
                    ["val_auc_testauc", "val_maxsmd_testauc", "val_nsmd_testauc", "train_maxsmd_testauc",
                     "train_nsmd_testauc", "trainval_maxsmd_testauc", "trainval_nsmd_testauc", 'trainval_final_testnauc']):
            # for c1, c2 in zip(["val_auc_nsmd", "val_maxsmd_nsmd", "val_nsmd_nsmd", "train_maxsmd_nsmd", "train_nsmd_nsmd", "trainval_final_finalnsmd"],
            #                   ["val_auc_testauc", "val_maxsmd_testauc", 'trainval_final_testnauc']):
                nsmd = rdf.loc[idx, c1]
                auc = rdf.loc[idx, c2]

                nsmd_med = IQR(nsmd)[0]
                nsmd_iqr = IQR(nsmd)[1:]

                nsmd_mean = nsmd.mean()
                nsmd_mean_ci, nsmd_mean_std = bootstrap_mean_ci(nsmd, alpha=0.05)

                success_rate = (nsmd <= MAX_NO_UNBALANCED_FEATURE).mean()
                success_rate_ci, success_rate_std = bootstrap_mean_ci(nsmd <= MAX_NO_UNBALANCED_FEATURE, alpha=0.05)

                auc_med = IQR(auc)[0]
                auc_iqr = IQR(auc)[1:]

                auc_mean = auc.mean()
                auc_mean_ci, auc_mean_std = bootstrap_mean_ci(auc, alpha=0.05)

                r.extend([nsmd_med, nsmd_iqr,
                          nsmd_mean, nsmd_mean_ci, nsmd_mean_std,
                          success_rate, success_rate_ci, success_rate_std,
                          auc_med, auc_iqr, auc_mean, auc_mean_ci, auc_mean_std])
                col_name.extend(
                    ["nsmd_med-" + c1, "nsmd_iqr-" + c1, "nsmd_mean-" + c1, "nsmd_mean_ci-" + c1, "nsmd_mean_std-" + c1,
                     "success_rate-" + c1, "success_rate_ci-" + c1, "success_rate_std-" + c1,
                     "auc_med-" + c2, "auc_iqr-" + c2, "auc_mean-" + c2, "auc_mean_ci-" + c2, "auc_mean_std-" + c2])

            x = np.array(rdf.loc[idx, "trainval_final_finalnsmd"] <= MAX_NO_UNBALANCED_FEATURE, dtype=np.float)
            y1 = np.array(rdf.loc[idx, "val_auc_nsmd"] <= MAX_NO_UNBALANCED_FEATURE, dtype=np.float)
            y2 = np.array(rdf.loc[idx, "val_maxsmd_nsmd"] <= MAX_NO_UNBALANCED_FEATURE, dtype=np.float)
            p1, test_orig1 = bootstrap_mean_pvalue_2samples(x, y1)
            p2, test_orig2 = bootstrap_mean_pvalue_2samples(x, y2)
            p3, test_orig3 = bootstrap_mean_pvalue_2samples(y1, y2)
            r.extend([p1, test_orig1[1], p2, test_orig2[1], p3, test_orig3[1]])
            col_name.extend(
                ['pboot-succes-final-vs-auc', 'p-succes-final-vs-auc',
                 'pboot-succes-final-vs-maxsmd', 'p-succes-final-vs-maxsmd',
                 'pboot-succes-auc-vs-maxsmd', 'p-succes-auc-vs-maxsmd'])

            results.append(r)
        df = pd.DataFrame(results, columns=col_name)
        df.to_excel(writer, sheet_name=t)
    writer.save()
    print()


def results_ATE_for_ml(cohort_dir_name, model, niter=50):
    cohort_size = pickle.load(open(r'../ipreprocess/output/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    name_cnt = sorted(cohort_size.items(), key=lambda x: x[1], reverse=True)
    drug_list_all = [drug.split('.')[0] for drug, cnt in name_cnt]
    dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
    drug_in_dir = set([x for x in os.listdir(dirname) if x.isdigit()])
    drug_list = [x for x in drug_list_all if x in drug_in_dir]  # in order
    check_and_mkdir(dirname + 'results/')

    for drug in drug_list:
        results = []
        for ctrl_type in ['random', 'atc']:
            for seed in range(0, niter):
                fname = dirname + drug + "/{}_S{}D200C{}_{}".format(drug, seed, ctrl_type, model)
                try:
                    df = pd.read_csv(fname + '_results.csv')
                except:
                    print('No file exisits: ', fname + '_results.csv')

                r = df.loc[3, :]
                for c in ["KM_time_points", "KM1_original", "KM0_original", "KM1-0_original",
                          "KM1_IPTW", "KM0_IPTW", "KM1-0_IPTW"]:
                    r.loc[c] = stringlist_2_list(r.loc[c])[-1]
                r = pd.Series(["{}_S{}D200C{}_{}".format(drug, seed, ctrl_type, model), ctrl_type],
                              index=['fname', 'ctrl_type']).append(r)
                results.append(r)
        rdf = pd.DataFrame(results)
        rdf.to_excel(dirname + 'results/' + drug + '_results.xlsx')
    print('Done')


def results_ATE_for_ml_step2(cohort_dir_name, model, drug_name):
    cohort_size = pickle.load(open(r'../ipreprocess/output/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    name_cnt = sorted(cohort_size.items(), key=lambda x: x[1], reverse=True)
    drug_list_all = [drug.split('.')[0] for drug, cnt in name_cnt]
    dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
    drug_in_dir = set([x for x in os.listdir(dirname) if x.isdigit()])
    drug_list = [x for x in drug_list_all if x in drug_in_dir]  # in order
    check_and_mkdir(dirname + 'results/')

    writer = pd.ExcelWriter(dirname + 'results/summarized_IPTW_ATE_{}.xlsx'.format(model), engine='xlsxwriter')
    for t in ['random', 'atc', 'all']:
        results = []
        for drug in drug_list:
            rdf = pd.read_excel(dirname + 'results/' + drug + '_results.xlsx')

            if t != 'all':
                idx_all = (rdf['ctrl_type'] == t)
            else:
                idx_all = (rdf['ctrl_type'].notna())

            # Only select balanced trial
            idx = idx_all & (rdf['n_unbalanced_feature_IPTW'] <= MAX_NO_UNBALANCED_FEATURE)

            print('drug: ', drug, t, 'support:', idx.sum())
            r = [drug, drug_name.get(drug, ''), idx_all.sum(), idx.sum()]
            col_name = ['drug', 'drug_name', 'niters', 'support']

            for c in ["n_treat", "n_ctrl", "n_feature"]:  # , 'HR_IPTW', 'HR_IPTW_CI'
                nv = rdf.loc[idx, c]
                nv_mean = nv.mean()
                r.append(nv_mean)
                col_name.append(c)

                nv = rdf.loc[idx_all, c]
                nv_mean = nv.mean()
                r.append(nv_mean)
                col_name.append(c + '-uab')

            for c in ["n_unbalanced_feature", "n_unbalanced_feature_IPTW"]:  # , 'HR_IPTW', 'HR_IPTW_CI'
                nv = rdf.loc[idx, c]
                if len(nv) > 0:
                    med = IQR(nv)[0]
                    iqr = IQR(nv)[1:]

                    mean = nv.mean()
                    mean_ci = bootstrap_mean_ci(nv, alpha=0.05)
                    r.extend([med, iqr, mean, mean_ci])
                else:
                    r.extend([np.nan, np.nan, np.nan, np.nan])
                col_name.extend(["med-" + c, "iqr-" + c, "mean-" + c, "mean_ci-" + c])

                nv = rdf.loc[idx_all, c]
                if len(nv) > 0:
                    med = IQR(nv)[0]
                    iqr = IQR(nv)[1:]

                    mean = nv.mean()
                    mean_ci = bootstrap_mean_ci(nv, alpha=0.05)
                    r.extend([med, iqr, mean, mean_ci])
                else:
                    r.extend([np.nan, np.nan, np.nan, np.nan])
                col_name.extend(
                    ["med-" + c + '-uab', "iqr-" + c + '-uab', "mean-" + c + '-uab', "mean_ci-" + c + '-uab'])

            for c in ["ATE_original", "ATE_IPTW", "KM1-0_original", "KM1-0_IPTW", 'HR_IPTW']:
                if c not in rdf.columns:
                    continue

                nv = rdf.loc[idx, c]
                if len(nv) > 0:
                    med = IQR(nv)[0]
                    iqr = IQR(nv)[1:]

                    mean = nv.mean()
                    mean_ci = bootstrap_mean_ci(nv, alpha=0.05)

                    if 'HR' in c:
                        p, _ = bootstrap_mean_pvalue(nv, expected_mean=1)
                    else:
                        p, _ = bootstrap_mean_pvalue(nv, expected_mean=0)

                    r.extend([med, iqr, mean, mean_ci, p])
                else:
                    r.extend([np.nan, np.nan, np.nan, np.nan, np.nan])
                col_name.extend(["med-" + c, "iqr-" + c, "mean-" + c, "mean_ci-" + c, 'pvalue-' + c])

            r.append(';'.join(rdf.loc[idx, 'HR_IPTW_CI']))
            col_name.append('HR_IPTW_CI')

            results.append(r)
        df = pd.DataFrame(results, columns=col_name)
        df.to_excel(writer, sheet_name=t)
    writer.save()
    print()


def check_drug_name_code():
    df = pd.read_excel(r'../data/repurposed_AD_under_trials_20200227.xlsx', dtype=str)
    rx_df = pd.read_csv(r'../ipreprocess/output/save_cohort_all_loose/cohort_all_name_size_positive_loose.csv',
                        index_col='cohort_name', dtype=str)
    gpi_df = pd.read_csv(r'../ipreprocess/output_marketscan/save_cohort_all_loose/cohort_all_name_size_positive.csv',
                         index_col='cohort_name', dtype=str)

    df['rx_drug_name'] = ''
    df['rx_n_patients'] = ''
    df['rx_n_pos'] = ''
    df['rx_pos_ratio'] = ''

    df['gpi_drug_name'] = ''
    df['gpi_n_patients'] = ''
    df['gpi_n_pos'] = ''
    df['gpi_pos_ratio'] = ''

    for index, row in df.iterrows():
        rx = row['rxcui']
        gpi = row['gpi']
        # print(index, row)
        if pd.notna(rx):
            rx = [x + '.pkl' for x in re.split('[,;+]', rx)]
        else:
            rx = []

        if pd.notna(gpi):
            gpi = [x + '.pkl' for x in re.split('[,;+]', gpi)]
        else:
            gpi = []

        for r in rx:
            if r in rx_df.index:
                df.loc[index, 'rx_drug_name'] += ('+' + rx_df.loc[r, 'drug_name'])
                df.loc[index, 'rx_n_patients'] += ('+' + rx_df.loc[r, 'n_patients'])
                df.loc[index, 'rx_n_pos'] += ('+' + rx_df.loc[r, 'n_pos'])
                df.loc[index, 'rx_pos_ratio'] += ('+' + rx_df.loc[r, 'pos_ratio'])

        for r in gpi:
            if r in gpi_df.index:
                df.loc[index, 'gpi_drug_name'] += ('+' + gpi_df.loc[r, 'drug_name'])
                df.loc[index, 'gpi_n_patients'] += ('+' + gpi_df.loc[r, 'n_patients'])
                df.loc[index, 'gpi_n_pos'] += ('+' + gpi_df.loc[r, 'n_pos'])
                df.loc[index, 'gpi_pos_ratio'] += ('+' + gpi_df.loc[r, 'pos_ratio'])

    df.to_excel(r'../data/repurposed_AD_under_trials_20200227-CHECK.xlsx')
    return df


def bar_plot_model_selection(cohort_dir_name, model, contrl_type='random', dump=True, colorful=True):
    dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
    dfall = pd.read_excel(dirname + 'results/summarized_model_selection_{}.xlsx'.format(model), sheet_name=contrl_type)
    idx = dfall['success_rate-trainval_final_finalnsmd'] >= 0.1
    idx_auc = dfall['success_rate-val_auc_nsmd'] >= 0.1
    idx_smd = dfall['success_rate-val_maxsmd_nsmd'] >= 0.1
    print('Total drug trials: ', len(idx))
    print(r"#df['success_rate-trainval_final_finalnsmd'] > 0: ", idx.sum(), '({:.2f}%)'.format(idx.mean() * 100))
    print(r"#df['success_rate-val_auc_nsmd'] > 0: ", idx_auc.sum(), '({:.2f}%)'.format(idx_auc.mean() * 100))
    print(r"#df['success_rate-val_maxsmd_nsmd'] > 0: ", idx_smd.sum(), '({:.2f}%)'.format(idx_smd.mean() * 100))

    df = dfall.loc[idx, :].sort_values(by=['success_rate-trainval_final_finalnsmd'], ascending=[False])
    # df['nsmd_mean_ci-val_auc_nsmd']

    N = len(df)
    top_1 = df.loc[:, 'success_rate-val_auc_nsmd']  # * 100
    top_1_ci = np.array(
        df.loc[:, 'success_rate_ci-val_auc_nsmd'].apply(lambda x: stringlist_2_list(x)).to_list())  # *100
    # top_1_ci = df.loc[:, 'success_rate_std-val_auc_nsmd']

    top_2 = df.loc[:, 'success_rate-val_maxsmd_nsmd']  # * 100
    top_2_ci = np.array(
        df.loc[:, 'success_rate_ci-val_maxsmd_nsmd'].apply(lambda x: stringlist_2_list(x)).to_list())  # *100
    # top_2_ci = df.loc[:, 'success_rate_std-val_maxsmd_nsmd']

    top_3 = df.loc[:, 'success_rate-trainval_final_finalnsmd']  # * 100
    top_3_ci = np.array(
        df.loc[:, 'success_rate_ci-trainval_final_finalnsmd'].apply(lambda x: stringlist_2_list(x)).to_list())  # *100
    # top_3_ci = df.loc[:, 'success_rate_std-trainval_final_finalnsmd']

    pauc = np.array(df.loc[:, "p-succes-final-vs-auc"])
    psmd = np.array(df.loc[:, "p-succes-final-vs-maxsmd"])
    paucsmd = np.array(df.loc[:, "p-succes-auc-vs-maxsmd"])

    xlabels = df.loc[:, 'drug_name']

    width = 0.45  # the width of the bars
    ind = np.arange(N) * width * 4  # the x locations for the groups

    colors = ['#FAC200', '#82A2D3', '#F65453']
    fig, ax = plt.subplots(figsize=(18, 8))
    error_kw = {'capsize': 3, 'capthick': 1, 'ecolor': 'black'}
    # plt.ylim([0, 1.05])
    rects1 = ax.bar(ind, top_1, width, yerr=[top_1 - top_1_ci[:, 0], top_1_ci[:, 1] - top_1], error_kw=error_kw,
                    color=colors[0], edgecolor="black")  # , edgecolor='b'
    rects2 = ax.bar(ind + width, top_2, width, yerr=[top_2 - top_2_ci[:, 0], top_2_ci[:, 1] - top_2], error_kw=error_kw,
                    color=colors[1], edgecolor="black")
    rects3 = ax.bar(ind + 2 * width, top_3, width, yerr=[top_3 - top_3_ci[:, 0], top_3_ci[:, 1] - top_3],
                    error_kw=error_kw, color=colors[2], edgecolor="black")  # , hatch='.')
    # rects1 = ax.bar(ind, top_1, width, yerr=[top_1_ci, top_1_ci], error_kw=error_kw,
    #                 color='#FAC200', edgecolor="black")  # , edgecolor='b'
    # rects2 = ax.bar(ind + width, top_2, width, yerr=[top_2_ci, top_2_ci], error_kw=error_kw,
    #                 color='#82A2D3', edgecolor="black")
    # rects3 = ax.bar(ind + 2 * width, top_3, width, yerr=[top_3_ci, top_3_ci],
    #                 error_kw=error_kw, color='#F65453', edgecolor="black")  # , hatch='.')

    ax.set_xticks(ind + width)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_color('#DDDDDD')
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.yaxis.grid(True, color='#EEEEEE', which='both')
    ax.xaxis.grid(False)
    ax.set_xticklabels(xlabels, fontsize=20, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel("Drug Trials", fontsize=25)
    ax.set_ylabel("Success Rate of Balancing", fontsize=25)

    def significance(val):
        if val <= 0.001:
            return '***'
        elif val <= 0.01:
            return '**'
        elif val <= 0.05:
            return '*'
        else:
            return 'ns'

    # def labelvalue(rects, val, height=None):
    #     for i, rect in enumerate(rects):
    #         if height is None:
    #             h = rect.get_height() * 1.03
    #         else:
    #             h = height[i] * 1.03
    #         ax.text(rect.get_x() + rect.get_width() / 2., h,
    #                 significance(val[i]),
    #                 ha='center', va='bottom', fontsize=11)
    #
    # labelvalue(rects1, pauc, top_1_ci[:,1])
    # labelvalue(rects2, psmd, top_2_ci[:,1])

    for i, rect in enumerate(rects3):
        d = 0.02
        y = top_3_ci[i, 1] * 1.03  # rect.get_height()
        w = rect.get_width()
        x = rect.get_x()
        x1 = x - 2 * w
        x2 = x - 1 * w

        y1 = top_1_ci[i, 1] * 1.03
        y2 = top_2_ci[i, 1] * 1.03

        # auc v.s. final
        l, r = x1, x + w
        ax.plot([l, l, (l+r) / 2], [y + 2 * d, y + 3 * d, y + 3 * d], lw=1.2, c=colors[0] if colorful else 'black')
        ax.plot([(l+r) / 2, r, r], [y + 3 * d, y + 3 * d, y + 2 * d], lw=1.2, c=colors[2] if colorful else 'black')
        # ax.plot([x1, x1, x, x], [y+2*d, y+3*d, y+3*d, y+2*d], c='#FAC200') #c="black")
        ax.text((l+r) / 2, y + 2.6 * d, significance(pauc[i]), ha='center', va='bottom', fontsize=13)

        # smd v.s. final
        l, r = x2 + 0.6*w, x + w
        ax.plot([l, l, (l + r) / 2], [y, y + d, y + d], lw=1.2, c=colors[1] if colorful else 'black')
        ax.plot([(l + r) / 2, r, r], [y + d, y + d, y], lw=1.2, c=colors[2] if colorful else 'black')
        # ax.plot([x2, x2, x, x], [y, y + d, y + d, y], c='#82A2D3') #c="black")
        ax.text((l + r) / 2, y + 0.6 * d, significance(psmd[i]), ha='center', va='bottom', fontsize=13)

        # auc v.s. smd
        l, r = x1, x2 + 0.4*w
        ax.plot([l, l, (l + r) / 2], [y, y + d, y + d], lw=1.2, c=colors[0] if colorful else 'black')
        ax.plot([(l + r) / 2, r, r], [y + d, y + d, y], lw=1.2, c=colors[1] if colorful else 'black')
        # ax.plot([x1, x1, x, x], [y+2*d, y+3*d, y+3*d, y+2*d], c='#FAC200') #c="black")
        ax.text((l + r) / 2, y + .6 * d, significance(paucsmd[i]), ha='center', va='bottom', fontsize=13)

    # ax.set_title('Success Rate of Balancing by Different PS Model Selection Methods')
    ax.legend((rects1[0], rects2[0], rects3[0]), ('Val-AUC Select', 'Val-SMD Select', 'Our Strategy'),
              fontsize=25)  # , bbox_to_anchor=(1.13, 1.01))

    # ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_xmargin(0.01)
    plt.tight_layout()
    if dump:
        fig.savefig(dirname + 'results/balance_rate_barplot-{}-{}.png'.format(model, contrl_type))
        fig.savefig(dirname + 'results/balance_rate_barplot-{}-{}.pdf'.format(model, contrl_type))
    plt.show()
    plt.clf()


def box_plot_model_selection(cohort_dir_name, model, contrl_type='random', dump=True, colorful=True):
    dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
    dfall = pd.read_excel(dirname + 'results/summarized_model_selection_{}.xlsx'.format(model),
                          sheet_name=contrl_type, converters={'drug': str})
    idx = dfall['success_rate-trainval_final_finalnsmd'] >= 0.1
    idx_auc = dfall['success_rate-val_auc_nsmd'] >= 0.1
    idx_smd = dfall['success_rate-val_maxsmd_nsmd'] >= 0.1
    print('Total drug trials: ', len(idx))
    print(r"#df['success_rate-trainval_final_finalnsmd'] > 0: ", idx.sum(), '({:.2f}%)'.format(idx.mean() * 100))
    print(r"#df['success_rate-val_auc_nsmd'] > 0: ", idx_auc.sum(), '({:.2f}%)'.format(idx_auc.mean() * 100))
    print(r"#df['success_rate-val_maxsmd_nsmd'] > 0: ", idx_smd.sum(), '({:.2f}%)'.format(idx_smd.mean() * 100))

    df = dfall.loc[idx, :].sort_values(by=['success_rate-trainval_final_finalnsmd'], ascending=[False])
    # df['nsmd_mean_ci-val_auc_nsmd']

    N = len(df)
    drug_list = df.loc[idx, 'drug']
    drug_name_list = df.loc[idx, 'drug_name']

    data_1 = []
    data_2 = []
    data_3 = []
    data_pvalue = []
    for drug in drug_list:
        rdf = pd.read_csv(dirname + 'results/' + drug + '_model_selection.csv')
        if contrl_type != 'all':
            idx = rdf['ctrl_type'] == contrl_type
        else:
            idx = rdf['ctrl_type'].notna()
        data_1.append(np.array(rdf.loc[idx, "val_auc_testauc"]))
        data_2.append(np.array(rdf.loc[idx, "val_maxsmd_testauc"]))
        data_3.append(np.array(rdf.loc[idx, "trainval_final_testnauc"]))
        p1, test_orig1 = bootstrap_mean_pvalue_2samples(data_3[-1], data_1[-1])
        p2, test_orig2 = bootstrap_mean_pvalue_2samples(data_3[-1], data_2[-1])
        p3, test_orig3 = bootstrap_mean_pvalue_2samples(data_1[-1], data_2[-1])
        # test_orig1_man = stats.mannwhitneyu(data_3[-1], data_1[-1])
        # test_orig2_man = stats.mannwhitneyu(data_3[-1], data_2[-1])
        data_pvalue.append([test_orig1[1], test_orig2[1], test_orig3[1]])

    colors = ['#FAC200', '#82A2D3', '#F65453']
    fig, ax = plt.subplots(figsize=(18, 8))
    width = 0.5  # the width of the bars
    ind = np.arange(N) * width * 4  # the x locations for the groups
    sym = 'o'
    # 'meanline':True,
    box_kw = {"sym": sym, "widths": width, "patch_artist": True, "notch": True,
              'showmeans':True, #'meanline':True,
              "meanprops": dict(linestyle='--', linewidth=1, markeredgecolor='purple', marker='^', markerfacecolor="None")}
    rects1 = plt.boxplot(data_1, positions=ind - 0.08, **box_kw)
    rects2 = plt.boxplot(data_2, positions=ind + width, **box_kw)
    rects3 = plt.boxplot(data_3, positions=ind + 2 * width + 0.08, **box_kw)

    def plot_strip(ind, data, color):
        w = width - 0.15
        swarm1 = pd.DataFrame([(ind[i], data[i][j]) for i in range(len(ind)) for j in range(len(data[i]))],
                              columns=['x', 'y'])
        strip_rx = stats.uniform(-w/ 2., w).rvs(len(swarm1))
        # sns.stripplot(x='x', y='y', data=swarm1, color=".25", alpha=0.2, ax=ax)
        plt.scatter(swarm1['x'] + strip_rx, swarm1['y'], alpha=0.2, c=color)

    # ticks = list(drug_name_list)
    for i, bplot in enumerate([rects1, rects2, rects3]):
        for patch in bplot['boxes']:
            patch.set_facecolor(colors[i])
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .7))
        # plt.setp(bplot['boxes'], color=color)
        # plt.setp(bplot['whiskers'], color=color)
        # plt.setp(bplot['caps'], color=color)
        plt.setp(bplot['medians'], color='black')

    plot_strip(ind - 0.08, data_1, colors[0])
    plot_strip(ind + width, data_2, colors[1])
    plot_strip(ind + 2 * width + 0.08, data_3, colors[2])

    # plt.ylim([0.5, 0.85])
    ax.set_xticks(ind + width)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_color('#DDDDDD')
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.yaxis.grid(True, color='#EEEEEE', which='both')
    ax.xaxis.grid(False)
    ax.set_xticklabels(drug_name_list, fontsize=20, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel("Drug Trials", fontsize=25)
    ax.set_ylabel("Test AUC", fontsize=25)

    def significance(val):
        if val <= 0.001:
            return '***'
        elif val <= 0.01:
            return '**'
        elif val <= 0.05:
            return '*'
        else:
            return 'ns'

    # def labelvalue(rects, x, y, p):
    #     for i, rect in enumerate(rects):
    #         ax.text(x[i], y[i],
    #                 significance(p[i]),
    #                 ha='center', va='bottom', fontsize=11)
    #
    # labelvalue(rects1["boxes"], ind - 0.08, np.max(data_1, axis=1)*1.01, np.array(data_pvalue)[:,0])
    # labelvalue(rects2["boxes"], ind + width, np.max(data_2, axis=1)*1.01, np.array(data_pvalue)[:,1])

    p_v = np.array(data_pvalue)
    for i in range(N):
        d = 0.008
        y = np.max([data_1[i].max(), data_2[i].max(), data_3[i].max()]) * 1.01  # rect.get_height()
        x = ind[i] + 2 * width + 0.08  # + width/2
        x1 = ind[i] - 0.08  # - width/2
        x2 = ind[i] + width  # - width/2

        # auc v.s. smd
        l, r = x - 0.5*width, x2 - 0.08
        ax.plot([x1, x1, (x2 + x1) / 2], [y, y + d, y + d], lw=1.2, c=colors[0] if colorful else 'black')
        ax.plot([(x2 + x1) / 2, x2 - 0.08, x2-0.08], [y + d, y + d, y], lw=1.2, c=colors[1] if colorful else 'black')
        ax.text((x2 + x1) / 2, y + d, significance(p_v[i, 2]), ha='center', va='bottom', fontsize=12)

        # auc v.s. final
        ax.plot([x1, x1, (x + x1) / 2], [y + 2 * d, y + 3 * d, y + 3 * d], lw=1.2, c=colors[0] if colorful else 'black')
        ax.plot([(x + x1) / 2, x, x], [y + 3 * d, y + 3 * d, y + 2 * d], lw=1.2, c=colors[2] if colorful else 'black')
        # ax.plot([x1, x1, x, x], [y+2*d, y+3*d, y+3*d, y+2*d], c="black")
        ax.text((x + x1) / 2, y + 3 * d, significance(p_v[i, 0]), ha='center', va='bottom', fontsize=12)

        # smd v.s. final
        ax.plot([x2+0.08, x2+0.08, (x + x2) / 2], [y, y + d, y + d], lw=1.2, c=colors[1] if colorful else 'black')
        ax.plot([(x + x2) / 2, x, x], [y + d, y + d, y], lw=1.2, c=colors[2] if colorful else 'black')
        # ax.plot([x2, x2, x, x], [y, y + d, y + d, y], c="black")
        ax.text((x + x2) / 2, y + 1 * d, significance(p_v[i, 1]), ha='center', va='bottom', fontsize=12)

    ax.legend((rects1["boxes"][0], rects2["boxes"][0], rects3["boxes"][0]),
              ('Val-AUC Select', 'Val-SMD Select', 'Our Strategy'),
              fontsize=20)
    ax.set_xmargin(0.01)
    plt.tight_layout()
    if dump:
        fig.savefig(dirname + 'results/test_auc_boxplot-{}-{}.png'.format(model, contrl_type))
        fig.savefig(dirname + 'results/test_auc_boxplot-{}-{}.pdf'.format(model, contrl_type))
    plt.show()
    plt.clf()


if __name__ == '__main__':
    # check_drug_name_code()
    # test bootstrap method

    # rvs = stats.norm.rvs(loc=0, scale=10, size=(100, 1))
    # # ci = bootstrap_mean_ci(rvs)
    # # p, test_orig = bootstrap_mean_pvalue(rvs, expected_mean=0., B=1000)
    #
    # rvs2 = stats.norm.rvs(loc=0, scale=10, size=(100, 1))
    # p, test_orig = bootstrap_mean_pvalue_2samples(rvs, rvs2)

    with open(r'pickles/rxnorm_label_mapping.pkl', 'rb') as f:
        drug_name = pickle.load(f)
        print('Using rxnorm_cui vocabulary, len(drug_name) :', len(drug_name))

    # shell_for_ml(cohort_dir_name='save_cohort_all_loose', model='LR', niter=50)
    # shell_for_ml(cohort_dir_name='save_cohort_all_loose', model='LIGHTGBM', niter=50, stats=False)
    # shell_for_ml(cohort_dir_name='save_cohort_all_loose', model='MLP', niter=50, stats=False)
    # split_shell_file("shell_MLP_save_cohort_all_loose.sh", divide=4, skip_first=1)
    # shell_for_ml(cohort_dir_name='save_cohort_all_loose', model='LSTM', niter=50, stats=False,
    #              more_para='--epochs 10 --batch_size 128')
    # split_shell_file("shell_LSTM_save_cohort_all_loose.sh", divide=4, skip_first=1)

    model = 'MLP'  # 'LR' #'LIGHTGBM'  #'LR'
    results_model_selection_for_ml(cohort_dir_name='save_cohort_all_loose', model=model, drug_name=drug_name, niter=50)
    results_model_selection_for_ml_step2(cohort_dir_name='save_cohort_all_loose', model=model, drug_name=drug_name)
    results_model_selection_for_ml_step2More(cohort_dir_name='save_cohort_all_loose', model=model, drug_name=drug_name)
    results_ATE_for_ml(cohort_dir_name='save_cohort_all_loose', model=model, niter=50)
    results_ATE_for_ml_step2(cohort_dir_name='save_cohort_all_loose', model=model, drug_name=drug_name)
    # # #
    bar_plot_model_selection(cohort_dir_name='save_cohort_all_loose', model=model, contrl_type='random')
    bar_plot_model_selection(cohort_dir_name='save_cohort_all_loose', model=model, contrl_type='atc')
    bar_plot_model_selection(cohort_dir_name='save_cohort_all_loose', model=model, contrl_type='all')
    #
    box_plot_model_selection(cohort_dir_name='save_cohort_all_loose', model=model, contrl_type='random')
    box_plot_model_selection(cohort_dir_name='save_cohort_all_loose', model=model, contrl_type='atc')
    box_plot_model_selection(cohort_dir_name='save_cohort_all_loose', model=model, contrl_type='all')

    # results_model_selection_for_ml(cohort_dir_name='save_cohort_all_loose', model='LIGHTGBM', drug_name=drug_name)
    print('Done')
