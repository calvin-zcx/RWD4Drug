import os
import shutil
import sys
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
import statsmodels.stats.multitest as smsmlt
import multipy.fdr as fdr
from sklearn.metrics import mean_squared_error

print = functools.partial(print, flush=True)

MAX_NO_UNBALANCED_FEATURE = 2  # 5  # 2  #5 #5  # 0 # 10 #5
# 5
print('Global MAX_NO_UNBALANCED_FEATURE: ', MAX_NO_UNBALANCED_FEATURE)

np.random.seed(0)
random.seed(0)


def IQR(s):
    return [np.quantile(s, .5), np.quantile(s, .25), np.quantile(s, .75)]


def stringlist_2_list(s):
    r = s.strip('][').replace(',', ' ').split()
    r = list(map(float, r))
    return r


def stringlist_2_str(s, percent=False, digit=-1):
    r = s.strip('][').replace(',', ' ').split()
    r = list(map(float, r))
    if percent:
        r = [x * 100 for x in r]

    if digit == 0:
        rr = ','.join(['{:.0f}'.format(x) for x in r])
    elif digit == 1:
        rr = ','.join(['{:.1f}'.format(x) for x in r])
    elif digit == 2:
        rr = ','.join(['{:.2f}'.format(x) for x in r])
    elif digit == 3:
        rr = ','.join(['{:.1f}'.format(x) for x in r])
    else:
        rr = ','.join(['{}'.format(x) for x in r])
    return rr


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


def shell_for_ml_simulation(model, niter=10, start=0, more_para=''):
    # fo = open('simulate_shell_{}-server2-part2.sh'.format(model), 'w')  # 'a'
    fo = open('simulate_v3_shell_{}.sh'.format(model), 'w')  # 'a'

    fo.write('mkdir -p output/simulate_v3/{}/log\n'.format(model))
    r = 0
    for n in [3000, 3500, 4000, 4500, 5000]:  # [2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]: #[2000, 4000, 6000]:
        for nolinear in ['no', 'moderate']:  # , 'strong'
            for covspec in ['correct',  'incorrect']:  # 'partial',
                for train_ratio in [0.8]:  # , 0.6
                    drug = '{}{}{}tr{:.1f}'.format(n, nolinear, covspec, train_ratio)
                    for seed in range(start, niter):
                        cmd = "python main_revise_v3_testset_simulate.py --nsim {} --train_ratio {} " \
                              "--run_model {} --nonlin {} --covspecs {} --output_dir output/simulate_v3/{}/ --random_seed {} {}" \
                              "2>&1 | tee output/simulate_v3/{}/log/{}_S{}D267_{}.log\n".format(
                            n, train_ratio,
                            model, nolinear, covspec, model, seed, more_para,
                            model, drug, seed, model)
                        fo.write(cmd)
                        r += 1
    fo.close()
    print('In total ', r, ' commands')


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


def _simplify_col_(x):
    # if 'mean' in x:
    #     x = x.replace(r"', 'mean')", '').replace(r"('", '')
    # elif 'std' in x:
    #     x = x.replace(r"', 'std')", '').replace(r"('", '') + '-std'
    # else:
    #     x = x.replace(r"', '')", '').replace(r"('", '')
    if ('-mean' in x) and (x != 'i-mean'):
        x = x.replace('-mean', '')
    else:
        x = x
    return x


def results_model_selection_for_ml(model, niter=10):
    dirname = r'output/simulate_v3/{}/'.format(model)
    drug_list = sorted(
        [x for x in os.listdir(dirname) if x.startswith('simun') and (('partial' not in x) and ('strong' not in x))],
        reverse=True)
    check_and_mkdir(dirname + 'results/')

    for drug in drug_list:
        results = []
        for seed in range(0, niter):
            fname = dirname + drug + "/{}_S{}D200Crandom_{}".format(drug, seed, model)
            try:
                df = pd.read_csv(fname + '_ALL-model-select-agg.csv')
                df.rename(columns=_simplify_col_, inplace=True)
            except:
                print('No file exisits: ', fname + '_ALL-model-select-agg.csv')
                continue

            selection_configs = [
                ('val_auc', 'i', False, True), ('val_loss', 'i', True, True),
                ('val_max_smd_iptw', 'i', True, True), ('val_n_unbalanced_feat_iptw', 'i', True, True),
                ('train_auc', 'i', False, True), ('train_loss', 'i', True, True),
                ('train_max_smd_iptw', 'i', True, True), ('train_n_unbalanced_feat_iptw', 'i', True, True),
                ('train_n_unbalanced_feat_iptw', 'val_auc', True, True),
                ('trainval_auc', 'i', False, True), ('trainval_loss', 'i', True, True),
                ('trainval_max_smd_iptw', 'i', True, True), ('trainval_n_unbalanced_feat_iptw', 'i', True, True),
                ('trainval_n_unbalanced_feat_iptw', 'val_auc', True, False),
                ('trainval_n_unbalanced_feat_iptw', 'val_loss', True, True)
            ]
            selection_results = []
            selection_results_colname = []
            for col1, col2, order1, order2 in selection_configs:
                # print('col1, col2, ascending order1, order2:', col1, col2, order1, order2)
                dftmp = df.sort_values(by=[col1, col2], ascending=[order1, order2])
                sr = []
                sr_colname = []
                for c in [col1, col2,
                          'train_loss', 'train_auc', 'train_max_smd', 'train_max_smd_iptw',
                          'train_n_unbalanced_feat',
                          'train_n_unbalanced_feat_iptw', 'train_HR_ori', 'train_HR_IPTW',
                          'test_loss', 'test_auc', 'test_max_smd', 'test_max_smd_iptw', 'test_n_unbalanced_feat',
                          'test_n_unbalanced_feat_iptw', 'test_HR_ori', 'test_HR_IPTW',
                          'all_loss', 'all_auc', 'all_max_smd', 'all_max_smd_iptw', 'all_n_unbalanced_feat',
                          'all_n_unbalanced_feat_iptw', 'all_HR_ori', 'all_HR_IPTW',
                          'train_reduction_n_unbalance', 'train_reduction_n_unbalance_percent',
                          'test_reduction_n_unbalance', 'test_reduction_n_unbalance_percent',
                          'all_reduction_n_unbalance', 'all_reduction_n_unbalance_percent']:
                    # print(c)
                    if c == 'all_reduction_n_unbalance':
                        sr.append(dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat')] -
                                  dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')])
                    elif c == 'test_reduction_n_unbalance':
                        sr.append(dftmp.iloc[0, dftmp.columns.get_loc('test_n_unbalanced_feat')] -
                                  dftmp.iloc[0, dftmp.columns.get_loc('test_n_unbalanced_feat_iptw')])
                    elif c == 'train_reduction_n_unbalance':
                        sr.append(dftmp.iloc[0, dftmp.columns.get_loc('train_n_unbalanced_feat')] -
                                  dftmp.iloc[0, dftmp.columns.get_loc('train_n_unbalanced_feat_iptw')])
                    elif c == 'all_reduction_n_unbalance_percent':
                        delta = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat')] - \
                                dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat_iptw')]
                        denom = dftmp.iloc[0, dftmp.columns.get_loc('all_n_unbalanced_feat')]
                        per = delta / denom if denom != 0 else 0
                        sr.append(per)
                    elif c == 'test_reduction_n_unbalance_percent':
                        delta = dftmp.iloc[0, dftmp.columns.get_loc('test_n_unbalanced_feat')] - \
                                dftmp.iloc[0, dftmp.columns.get_loc('test_n_unbalanced_feat_iptw')]
                        denom = dftmp.iloc[0, dftmp.columns.get_loc('test_n_unbalanced_feat')]
                        per = delta / denom if denom != 0 else 0
                        sr.append(per)
                    elif c == 'train_reduction_n_unbalance_percent':
                        delta = dftmp.iloc[0, dftmp.columns.get_loc('train_n_unbalanced_feat')] - \
                                dftmp.iloc[0, dftmp.columns.get_loc('train_n_unbalanced_feat_iptw')]
                        denom = dftmp.iloc[0, dftmp.columns.get_loc('train_n_unbalanced_feat')]
                        per = delta / denom if denom != 0 else 0
                        sr.append(per)
                    else:
                        sr.append(dftmp.iloc[0, dftmp.columns.get_loc(c)])
                    if (c in [col1, col2]) and (c != 'i'):
                        sr_colname.append(c)
                    else:
                        sr_colname.append('{}-{}-{}'.format(col1, col2, c))
                selection_results.extend(sr)
                selection_results_colname.extend(sr_colname)

            results.append(["{}_S{}D200Crandom_{}".format(drug, seed, model), drug] + selection_results)

        rdf = pd.DataFrame(results, columns=['fname', 'ctrl_type'] + selection_results_colname)
        rdf.to_csv(dirname + 'results/' + drug + '_model_selection.csv')

        pre_col = []
        for col1, col2, order1, order2 in selection_configs:
            pre_col.append('{}-{}-'.format(col1, col2))

        for t in ['all']:
            # fig = plt.figure(figsize=(20, 15))
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 18))
            if t != 'all':
                idx = rdf['ctrl_type'] == t
            else:
                idx = rdf['ctrl_type'].notna()
            boxplot = rdf[idx].boxplot(column=[x + 'all_n_unbalanced_feat_iptw' for x in pre_col], fontsize=15, ax=ax1,
                                       showmeans=True)  # rot=25,
            ax1.axhline(y=5, color='r', linestyle='-')
            boxplot.set_title("{}-S{}D200C{}_{}".format(drug, '0-19', t, model), fontsize=25)
            # plt.xlabel("Model selection methods", fontsize=15)
            ax1.set_ylabel("#unbalanced_feat_iptw of boostrap experiments", fontsize=20)
            print(ax1.get_xticklabels())
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
            # fig.savefig(dirname + 'results/' + drug + '_model_selection_boxplot-{}-allnsmd.png'.format(t))
            # plt.show()

            # fig = plt.figure(figsize=(20, 15))
            boxplot = rdf[idx].boxplot(column=[x + 'test_auc' for x in pre_col], fontsize=15, ax=ax2, showmeans=True)
            # plt.axhline(y=0.5, color='r', linestyle='-')
            ax2.set_xlabel("Model selection methods", fontsize=20)
            ax2.set_ylabel("test_auc of boostrap experiments", fontsize=20)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')

            boxplot = rdf[idx].boxplot(column=[x + 'all_reduction_n_unbalance_percent' for x in pre_col], fontsize=15,
                                       ax=ax3, showmeans=True)
            ax3.set_xlabel("Model selection methods", fontsize=20)
            ax3.set_ylabel("all_reduction_n_unbalance", fontsize=20)
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, horizontalalignment='right')

            boxplot = rdf[idx].boxplot(column=[x + 'test_reduction_n_unbalance_percent' for x in pre_col], fontsize=15,
                                       ax=ax4, showmeans=True)
            ax4.set_xlabel("Model selection methods", fontsize=20)
            ax4.set_ylabel("test_reduction_n_unbalance", fontsize=20)
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, horizontalalignment='right')

            plt.tight_layout()
            fig.savefig(dirname + 'results/' + drug + '_model_selection_boxplot-{}.png'.format(t))
            plt.clf()
    print()


def results_model_selection_for_ml_step2(model):
    dirname = r'output/simulate_v3/{}/'.format(model)
    drug_list = sorted(
        [x for x in os.listdir(dirname) if x.startswith('simun') and (('partial' not in x) and ('strong' not in x))],
        reverse=True)
    check_and_mkdir(dirname + 'results/')

    writer = pd.ExcelWriter(dirname + 'results/summarized_model_selection_{}.xlsx'.format(model), engine='xlsxwriter')
    for t in ['all']:
        results = []
        for drug in drug_list:
            rdf = pd.read_csv(dirname + 'results/' + drug + '_model_selection.csv')

            if t != 'all':
                idx = rdf['ctrl_type'] == t
            else:
                idx = rdf['ctrl_type'].notna()

            if idx.sum() == 0:
                print('empty trials in:', drug, t)
                continue

            r = [drug, drug]
            col_name = ['drug', 'drug_name']
            # zip(["val_auc_nsmd", "val_maxsmd_nsmd", "val_nsmd_nsmd", "train_maxsmd_nsmd",
            #      "train_nsmd_nsmd", "trainval_maxsmd_nsmd", "trainval_nsmd_nsmd",
            #      "trainval_final_finalnsmd"],
            #     ["val_auc_testauc", "val_maxsmd_testauc", "val_nsmd_testauc",
            #      "train_maxsmd_testauc", "train_nsmd_testauc", "trainval_maxsmd_testauc",
            #      "trainval_nsmd_testauc", 'trainval_final_testnauc'])
            # for c1, c2 in zip(["val_auc_nsmd", "val_maxsmd_nsmd", "trainval_final_finalnsmd"],
            #                   ["val_auc_testauc", "val_maxsmd_testauc", 'trainval_final_testnauc']):
            # for c1, c2 in zip(["val_auc-i-all_n_unbalanced_feat_iptw", "train_loss-i-all_n_unbalanced_feat_iptw",
            #                    "trainval_n_unbalanced_feat_iptw-val_auc-all_n_unbalanced_feat_iptw"],
            #                   ["val_auc-i-test_auc", "train_loss-i-test_auc",
            #                    "trainval_n_unbalanced_feat_iptw-val_auc-test_auc"]):
            for c1, c2 in zip(["val_auc-i-all_n_unbalanced_feat_iptw", "val_loss-i-all_n_unbalanced_feat_iptw",
                               "trainval_n_unbalanced_feat_iptw-val_auc-all_n_unbalanced_feat_iptw"],
                              ["val_auc-i-test_auc", "val_loss-i-test_auc",
                               "trainval_n_unbalanced_feat_iptw-val_auc-test_auc"]):
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

            x = np.array(rdf.loc[
                             idx, "trainval_n_unbalanced_feat_iptw-val_auc-all_n_unbalanced_feat_iptw"] <= MAX_NO_UNBALANCED_FEATURE,
                         dtype=np.float64)
            y1 = np.array(rdf.loc[idx, "val_auc-i-all_n_unbalanced_feat_iptw"] <= MAX_NO_UNBALANCED_FEATURE,
                          dtype=np.float64)
            y2 = np.array(rdf.loc[idx, "val_loss-i-all_n_unbalanced_feat_iptw"] <= MAX_NO_UNBALANCED_FEATURE,
                          dtype=np.float64)
            p1, test_orig1 = bootstrap_mean_pvalue_2samples(x, y1)
            p2, test_orig2 = bootstrap_mean_pvalue_2samples(x, y2)
            p3, test_orig3 = bootstrap_mean_pvalue_2samples(y1, y2)
            r.extend([p1, test_orig1[1], p2, test_orig2[1], p3, test_orig3[1]])
            col_name.extend(
                ['pboot-succes-final-vs-1', 'p-succes-final-vs-1',
                 'pboot-succes-final-vs-2', 'p-succes-final-vs-2',
                 'pboot-succes-1-vs-2', 'p-succes-1-vs-2'])

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

    writer = pd.ExcelWriter(dirname + 'results/summarized_model_selection_{}-More.xlsx'.format(model),
                            engine='xlsxwriter')
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
                     "train_nsmd_testauc", "trainval_maxsmd_testauc", "trainval_nsmd_testauc",
                     'trainval_final_testnauc']):
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
            col = ['val_auc_nsmd', 'val_maxsmd_nsmd', 'val_nsmd_nsmd',
                   'train_maxsmd_nsmd', 'train_nsmd_nsmd',
                   'trainval_maxsmd_nsmd', 'trainval_nsmd_nsmd']  # ,'success_rate-trainval_final_finalnsmd']
            for c in col:
                y = np.array(rdf.loc[idx, c] <= MAX_NO_UNBALANCED_FEATURE, dtype=np.float)
                p, test_orig = bootstrap_mean_pvalue_2samples(x, y)
                r.append(test_orig[1])
                col_name.append('p-succes-fvs-' + c)

            results.append(r)
        df = pd.DataFrame(results, columns=col_name)
        df.to_excel(writer, sheet_name=t)
    writer.save()
    print()


def bar_plot_model_selection(model, contrl_type='all', dump=True, colorful=True):
    dirname = r'output/simulate_v3/{}/'.format(model)
    dfall = pd.read_excel(dirname + 'results/summarized_model_selection_{}.xlsx'.format(model), sheet_name=contrl_type)

    c1 = 'success_rate-val_auc-i-all_n_unbalanced_feat_iptw'
    c2 = 'success_rate-val_loss-i-all_n_unbalanced_feat_iptw'
    c3 = 'success_rate-trainval_n_unbalanced_feat_iptw-val_auc-all_n_unbalanced_feat_iptw'

    idx_auc = dfall[c1] >= 0.1
    idx_smd = dfall[c2] >= 0.1
    idx = dfall[c3] >= 0.1

    idx = dfall[c3].notna()

    print('Total drug trials: ', len(idx))
    print(r"#df[{}] > 0: ".format(c1), idx_auc.sum(), '({:.2f}%)'.format(idx_auc.mean() * 100))
    print(r"#df[{}] > 0: ".format(c2), idx_smd.sum(), '({:.2f}%)'.format(idx_smd.mean() * 100))
    print(r"#df[{}] > 0: ".format(c3), idx.sum(), '({:.2f}%)'.format(idx.mean() * 100))

    df = dfall.loc[idx, :].sort_values(by=[c3], ascending=[False])
    # df['nsmd_mean_ci-val_auc_nsmd']
    df = dfall

    N = len(df)
    top_1 = df.loc[:, c1]  # * 100
    top_1_ci = np.array(
        df.loc[:, c1.replace('success_rate', 'success_rate_ci')].apply(
            lambda x: stringlist_2_list(x)).to_list())  # *100
    # top_1_ci = df.loc[:, 'success_rate_std-val_auc_nsmd']

    top_2 = df.loc[:, c2]  # * 100
    top_2_ci = np.array(
        df.loc[:, c2.replace('success_rate', 'success_rate_ci')].apply(
            lambda x: stringlist_2_list(x)).to_list())  # *100
    # top_2_ci = df.loc[:, 'success_rate_std-val_maxsmd_nsmd']

    top_3 = df.loc[:, c3]  # * 100
    top_3_ci = np.array(
        df.loc[:, c3.replace('success_rate', 'success_rate_ci')].apply(
            lambda x: stringlist_2_list(x)).to_list())  # *100
    # top_3_ci = df.loc[:, 'success_rate_std-trainval_final_finalnsmd']

    pauc = np.array(df.loc[:, "p-succes-final-vs-1"])
    psmd = np.array(df.loc[:, "p-succes-final-vs-2"])
    paucsmd = np.array(df.loc[:, "p-succes-1-vs-2"])

    xlabels = df.loc[:, 'drug_name']
    # xlabels = [x[5:] for x in xlabels] # drugname.replace('simun', '').replace('train0.8', ''), #
    xlabels = [x.replace('simun', '').replace('train0.8', '').replace('no', '-lin-').replace('moderate', '-nonLin-') for
               x in xlabels]

    width = 0.45  # the width of the bars
    ind = np.arange(N) * width * 4  # the x locations for the groups

    colors = ['#FAC200', '#82A2D3', '#F65453']
    fig, ax = plt.subplots(figsize=(18, 8))
    error_kw = {'capsize': 3, 'capthick': 1, 'ecolor': 'black'}
    # plt.ylim([0, 1.05])
    rects1 = ax.bar(ind, top_1, width, yerr=[top_1 - top_1_ci[:, 0], top_1_ci[:, 1] - top_1], error_kw=error_kw,
                    color=colors[0], edgecolor=None)  # , edgecolor='b' "black"
    rects2 = ax.bar(ind + width, top_2, width, yerr=[top_2 - top_2_ci[:, 0], top_2_ci[:, 1] - top_2], error_kw=error_kw,
                    color=colors[1], edgecolor=None)
    rects3 = ax.bar(ind + 2 * width, top_3, width, yerr=[top_3 - top_3_ci[:, 0], top_3_ci[:, 1] - top_3],
                    error_kw=error_kw, color=colors[2], edgecolor=None)  # , hatch='.')
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
    ax.set_xlabel("Simulation Settings", fontsize=25)
    # ax.set_ylabel("Prop. of success balancing", fontsize=25)  # Success Rate of Balancing
    ax.set_ylabel("Ratio of success balancing", fontsize=25)  # Success Rate of Balancing

    # ax.set_title(model, fontsize=25) #fontweight="bold")
    # plt.axhline(y=0.5, color='#888888', linestyle='-')

    def significance(val):
        if val < 0.001:
            return '***'
        elif val < 0.01:
            return '**'
        elif val < 0.05:
            return '*'
        else:
            return 'ns'

    # # def labelvalue(rects, val, height=None):
    # #     for i, rect in enumerate(rects):
    # #         if height is None:
    # #             h = rect.get_height() * 1.03
    # #         else:
    # #             h = height[i] * 1.03
    # #         ax.text(rect.get_x() + rect.get_width() / 2., h,
    # #                 significance(val[i]),
    # #                 ha='center', va='bottom', fontsize=11)
    # #
    # # labelvalue(rects1, pauc, top_1_ci[:,1])
    # # labelvalue(rects2, psmd, top_2_ci[:,1])
    #
    for i, rect in enumerate(rects3):
        d = 0.02
        y = np.max([top_3_ci[i, 1], top_2_ci[i, 1], top_1_ci[i, 1]]) * 1.03  # rect.get_height()
        w = rect.get_width()
        x = rect.get_x()
        x1 = x - 2 * w
        x2 = x - 1 * w

        y1 = top_1_ci[i, 1] * 1.03
        y2 = top_2_ci[i, 1] * 1.03

        # auc v.s. final
        l, r = x1, x + w
        ax.plot([l, l, (l + r) / 2], [y + 2 * d, y + 3 * d, y + 3 * d], lw=1.2, c=colors[0] if colorful else 'black')
        ax.plot([(l + r) / 2, r, r], [y + 3 * d, y + 3 * d, y + 2 * d], lw=1.2, c=colors[2] if colorful else 'black')
        # ax.plot([x1, x1, x, x], [y+2*d, y+3*d, y+3*d, y+2*d], c='#FAC200') #c="black")
        ax.text((l + r) / 2, y + 2.6 * d, significance(pauc[i]), ha='center', va='bottom', fontsize=13)

        # smd v.s. final
        l, r = x2 + 0.6 * w, x + w
        ax.plot([l, l, (l + r) / 2], [y, y + d, y + d], lw=1.2, c=colors[1] if colorful else 'black')
        ax.plot([(l + r) / 2, r, r], [y + d, y + d, y], lw=1.2, c=colors[2] if colorful else 'black')
        # ax.plot([x2, x2, x, x], [y, y + d, y + d, y], c='#82A2D3') #c="black")
        ax.text((l + r) / 2, y + 0.6 * d, significance(psmd[i]), ha='center', va='bottom', fontsize=13)

        # auc v.s. smd
        l, r = x1, x2 + 0.4 * w
        ax.plot([l, l, (l + r) / 2], [y, y + d, y + d], lw=1.2, c=colors[0] if colorful else 'black')
        ax.plot([(l + r) / 2, r, r], [y + d, y + d, y], lw=1.2, c=colors[1] if colorful else 'black')
        # ax.plot([x1, x1, x, x], [y+2*d, y+3*d, y+3*d, y+2*d], c='#FAC200') #c="black")
        ax.text((l + r) / 2, y + .6 * d, significance(paucsmd[i]), ha='center', va='bottom', fontsize=13)

    # ax.set_title('Success Rate of Balancing by Different PS Model Selection Methods')
    ax.legend((rects1[0], rects2[0], rects3[0]), ('Val-AUC Select', 'Val-Loss Select', 'Our Strategy'),
              fontsize=25, loc='center right')  # , bbox_to_anchor=(1.13, 1.01))

    # ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_xmargin(0.01)
    plt.tight_layout()
    if dump:
        check_and_mkdir(dirname + 'results/fig/')
        fig.savefig(dirname + 'results/fig/balance_rate_barplot-{}-{}.png'.format(model, contrl_type))
        fig.savefig(dirname + 'results/fig/balance_rate_barplot-{}-{}.pdf'.format(model, contrl_type))
    plt.show()
    plt.clf()


def arrow_plot_model_selection_unbalance_reduction(model, contrl_type='all', dump=True,
                                                   colorful=True, datapart='all', log=False):
    # dataset: train, test, all
    dirname = r'output/simulate_v3/{}/'.format(model)
    dfall = pd.read_excel(dirname + 'results/summarized_model_selection_{}.xlsx'.format(model),
                          sheet_name=contrl_type, converters={'drug': str})
    c1 = 'val_auc-i'
    c2 = 'val_loss-i'
    c30 = 'trainval_n_unbalanced_feat_iptw-val_auc'
    c3 = 'trainval_n_unbalanced_feat_iptw-val_auc'  # val_loss

    idx_auc = dfall['success_rate-' + c1 + '-all_n_unbalanced_feat_iptw'] >= 0.1
    idx_smd = dfall['success_rate-' + c2 + '-all_n_unbalanced_feat_iptw'] >= 0.1
    idx = dfall['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'] >= 0.1

    print('Total drug trials: ', len(idx))
    print(r"#df[{}] > 0: ".format('success_rate-' + c1 + '-all_n_unbalanced_feat_iptw'), idx_auc.sum(),
          '({:.2f}%)'.format(idx_auc.mean() * 100))
    print(r"#df[{}] > 0: ".format('success_rate-' + c2 + '-all_n_unbalanced_feat_iptw'), idx_smd.sum(),
          '({:.2f}%)'.format(idx_smd.mean() * 100))
    print(r"#df[{}] > 0: ".format('success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'), idx.sum(),
          '({:.2f}%)'.format(idx.mean() * 100))

    idx = dfall['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'].notna()
    df = dfall.loc[idx, :].sort_values(by=['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'], ascending=[False])
    # df = dfall.sort_values(by=['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'], ascending=[False])
    df = dfall

    # df['nsmd_mean_ci-val_auc_nsmd']

    N = len(df)
    drug_list = df.loc[idx, 'drug']
    drug_name_list = df.loc[idx, 'drug_name']

    # df.loc[idx, :].to_csv(dirname + 'results/selected_balanced_drugs_for_screen.csv')
    # data_1 = []
    # data_2 = []
    # data_3 = []
    # data_pvalue = []
    data = [[], [], []]
    for drug, drugname in zip(drug_list, drug_name_list):
        rdf = pd.read_csv(dirname + 'results/' + drug + '_model_selection.csv')
        if contrl_type != 'all':
            idx = rdf['ctrl_type'] == contrl_type
        else:
            idx = rdf['ctrl_type'].notna()

        for ith, c in enumerate([c1, c2, c3]):
            before = np.array(rdf.loc[idx, c + '-{}_n_unbalanced_feat'.format(datapart)])
            after = np.array(rdf.loc[idx, c + '-{}_n_unbalanced_feat_iptw'.format(datapart)])
            change = after - before
            before_med = IQR(before)[0]
            before_iqr = IQR(before)[1:]
            before_mean = before.mean()
            before_mean_ci, before_mean_std = bootstrap_mean_ci(before, alpha=0.05)
            after_med = IQR(after)[0]
            after_iqr = IQR(after)[1:]
            after_mean = after.mean()
            after_mean_ci, before_mean_std = bootstrap_mean_ci(after, alpha=0.05)
            change_med = IQR(change)[0]
            change_iqr = IQR(change)[1:]
            change_mean = change.mean()
            change_mean_ci, change_mean_std = bootstrap_mean_ci(change, alpha=0.05)

            data[ith].append([drugname.replace('simun', '').replace('train0.8', '').replace('no', '-lin-').replace(
                'moderate', '-nonLin-'),  # drugname[5:],
                              before_mean, after_mean, change_mean])

    data_df = []
    for d in data:
        df = pd.DataFrame(d, columns=['subject', 'before', 'after', 'change'], index=range(len(d)))
        data_df.append(df)

    fig = plt.figure(figsize=(5, 6))
    # add start points
    ax = sns.stripplot(data=data_df[1],
                       x='before',
                       y='subject',
                       orient='h',
                       order=data_df[1]['subject'],
                       size=8,
                       color='black',
                       facecolors="none",
                       jitter=0, alpha=0.7
                       )

    # plt.scatter(swarm1['x'] + strip_rx, swarm1['y'], alpha=.6, c=color)

    d_pos = [-0.15, 0, 0.15]
    colors = ['#FAC200', '#82A2D3', '#F65453']
    for igroup, data in enumerate(data_df):
        arrow_starts = data['before'].values
        arrow_lengths = data['change'].values
        arrow_ends = data['after'].values
        # add arrows to plot

        for i, subject in enumerate(data['subject']):
            prop = dict(arrowstyle="->,head_width=0.4,head_length=0.8",
                        shrinkA=0, shrinkB=0, color=colors[igroup])
            # plt.annotate("", xy=(arrow_ends[i], i+d_pos[igroup]), xytext=(arrow_starts[i], i+d_pos[igroup]), arrowprops=prop)
            # ax.arrow(row['2009'], idx, row['2013'] - row['2009'], 0, head_width=0.2, head_length=0.7, width=0.03, fc=c,
            #          ec=c)
            # plt.annotate(arrow_ends[i], xy=(arrow_ends[i], i+d_pos[igroup]), xytext=(arrow_ends[i]+d_pos[igroup], i+d_pos[igroup]), arrowprops=prop)

            ax.arrow(arrow_starts[i],  # x start point
                     i + d_pos[igroup],  # y start point
                     arrow_lengths[i],  # change in x
                     0,  # change in y
                     head_width=0.2,  # arrow head width
                     head_length=1.2,  # arrow head length
                     width=0.02,  # arrow stem width
                     fc=colors[igroup],  # arrow fill color
                     ec=colors[igroup]
                     )  # arrow edge color

    # format plot
    ax = sns.stripplot(data=data_df[1],
                       x='before',
                       y='subject',
                       orient='h',
                       order=data_df[1]['subject'],
                       size=8,
                       color='black',
                       facecolors="none", jitter=0, alpha=0.7
                       )
    ax.set_title('before vs. after re-weighting on ' + datapart)  # add title
    ax.grid(axis='y', color='0.9')  # add a light grid
    # if datapart == 'test':
    #     # ax.set_xlim(75, 150)  # set x axis limits
    #     pass
    # elif datapart == 'all':
    #     # ax.set_xlim(-1, 40)  # set x axis limits
    #     ax.axvline(x=0, color='0.9', ls='--', lw=2, zorder=0)  # add line at x=0
    #
    # else:
    #     ax.axvline(x=0, color='0.9', ls='--', lw=2, zorder=0)  # add line at x=0
    #

    if datapart == 'all':
        ax.set_xlim(right=20)  # set x axis limits
        # plt.xlim(right=35)
    elif datapart == 'train':
        ax.set_xlim(right=25)

    ax.axvline(x=0, color='0.9', ls='--', lw=2, zorder=0)  # add line at x=0

    if log:
        plt.xscale("log")
    ax.set_xlabel('No. of unbalanced features')  # label the x axis
    ax.set_ylabel('Simulation Settings')  # label the y axis
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    if dump:
        check_and_mkdir(dirname + 'results/fig/')
        fig.savefig(dirname + 'results/fig/arrow_nsmd_reduce-{}-{}-{}{}.png'.format(model, contrl_type, datapart,
                                                                                    '-log' if log else ''))
        fig.savefig(dirname + 'results/fig/arrow_nsmd_reduce-{}-{}-{}{}.pdf'.format(model, contrl_type, datapart,
                                                                                    '-log' if log else ''))
    plt.show()
    plt.clf()


def arrow_plot_model_selection_bias_reduction(model, groundtruth_dict, contrl_type='all', dump=True,
                                              colorful=True, datapart='all', log=False, ):
    # dataset: train, test, all
    dirname = r'output/simulate_v3/{}/'.format(model)
    dfall = pd.read_excel(dirname + 'results/summarized_model_selection_{}.xlsx'.format(model),
                          sheet_name=contrl_type, converters={'drug': str})
    c1 = 'val_auc-i'
    c2 = 'val_loss-i'
    c30 = 'trainval_n_unbalanced_feat_iptw-val_auc'
    c3 = 'trainval_n_unbalanced_feat_iptw-val_auc'  # val_loss

    idx_auc = dfall['success_rate-' + c1 + '-all_n_unbalanced_feat_iptw'] >= 0.1
    idx_smd = dfall['success_rate-' + c2 + '-all_n_unbalanced_feat_iptw'] >= 0.1
    idx = dfall['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'] >= 0.1

    print('Total drug trials: ', len(idx))
    print(r"#df[{}] > 0: ".format('success_rate-' + c1 + '-all_n_unbalanced_feat_iptw'), idx_auc.sum(),
          '({:.2f}%)'.format(idx_auc.mean() * 100))
    print(r"#df[{}] > 0: ".format('success_rate-' + c2 + '-all_n_unbalanced_feat_iptw'), idx_smd.sum(),
          '({:.2f}%)'.format(idx_smd.mean() * 100))
    print(r"#df[{}] > 0: ".format('success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'), idx.sum(),
          '({:.2f}%)'.format(idx.mean() * 100))

    idx = dfall['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'].notna()
    df = dfall.loc[idx, :].sort_values(by=['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'], ascending=[False])

    df = dfall

    N = len(df)
    drug_list = df.loc[idx, 'drug']
    drug_name_list = df.loc[idx, 'drug_name']

    data = [[], [], []]
    for drug, drugname in zip(drug_list, drug_name_list):
        rdf = pd.read_csv(dirname + 'results/' + drug + '_model_selection.csv')
        if contrl_type != 'all':
            idx = rdf['ctrl_type'] == contrl_type
        else:
            idx = rdf['ctrl_type'].notna()

        if 'no' in drugname:
            groundtruth = groundtruth_dict['no']
        elif 'moderate' in drugname:
            groundtruth = groundtruth_dict['moderate']
        else:
            raise ValueError

        for ith, c in enumerate([c1, c2, c3]):
            before = np.abs(groundtruth - np.array(rdf.loc[idx, c + '-{}_HR_ori'.format(datapart)]))
            after = np.abs(groundtruth - np.array(rdf.loc[idx, c + '-{}_HR_IPTW'.format(datapart)]))
            change = after - before
            before_med = IQR(before)[0]
            before_iqr = IQR(before)[1:]
            before_mean = before.mean()
            before_mean_ci, before_mean_std = bootstrap_mean_ci(before, alpha=0.05)
            after_med = IQR(after)[0]
            after_iqr = IQR(after)[1:]
            after_mean = after.mean()
            after_mean_ci, before_mean_std = bootstrap_mean_ci(after, alpha=0.05)
            change_med = IQR(change)[0]
            change_iqr = IQR(change)[1:]
            change_mean = change.mean()
            change_mean_ci, change_mean_std = bootstrap_mean_ci(change, alpha=0.05)

            data[ith].append([drugname.replace('simun', '').replace('train0.8', '').replace('no', '-lin-').replace(
                'moderate', '-nonLin-'),  # drugname[5:],
                              before_mean, after_mean, change_mean])

    data_df = []
    for d in data:
        df = pd.DataFrame(d, columns=['subject', 'before', 'after', 'change'], index=range(len(d)))
        data_df.append(df)

    fig = plt.figure(figsize=(5, 6))
    # add start points
    ax = sns.stripplot(data=data_df[1],
                       x='before',
                       y='subject',
                       orient='h',
                       order=data_df[1]['subject'],
                       size=8,
                       color='black',
                       facecolors="none",
                       jitter=0, alpha=0.7
                       )

    # plt.scatter(swarm1['x'] + strip_rx, swarm1['y'], alpha=.6, c=color)

    d_pos = [-0.15, 0, 0.15]
    colors = ['#FAC200', '#82A2D3', '#F65453']
    for igroup, data in enumerate(data_df):
        arrow_starts = data['before'].values
        arrow_lengths = data['change'].values
        arrow_ends = data['after'].values
        # add arrows to plot

        for i, subject in enumerate(data['subject']):
            prop = dict(arrowstyle="->,head_width=0.4,head_length=0.8",
                        shrinkA=0, shrinkB=0, color=colors[igroup])
            # plt.annotate("", xy=(arrow_ends[i], i+d_pos[igroup]), xytext=(arrow_starts[i], i+d_pos[igroup]), arrowprops=prop)
            # ax.arrow(row['2009'], idx, row['2013'] - row['2009'], 0, head_width=0.2, head_length=0.7, width=0.03, fc=c,
            #          ec=c)
            # plt.annotate(arrow_ends[i], xy=(arrow_ends[i], i+d_pos[igroup]), xytext=(arrow_ends[i]+d_pos[igroup], i+d_pos[igroup]), arrowprops=prop)

            ax.arrow(arrow_starts[i],  # x start point
                     i + d_pos[igroup],  # y start point
                     arrow_lengths[i],  # change in x
                     0,  # change in y
                     head_width=0.1,  # arrow head width
                     head_length=0.02,  # arrow head length
                     width=0.02,  # arrow stem width
                     fc=colors[igroup],  # arrow fill color
                     ec=colors[igroup]
                     )  # arrow edge color

    # format plot
    ax = sns.stripplot(data=data_df[1],
                       x='before',
                       y='subject',
                       orient='h',
                       order=data_df[1]['subject'],
                       size=8,
                       color='black',
                       facecolors="none", jitter=0, alpha=0.7
                       )
    ax.set_title('before vs. after re-weighting on ' + datapart)  # add title
    ax.grid(axis='y', color='0.9')  # add a light grid
    # if datapart == 'test':
    #     # ax.set_xlim(75, 150)  # set x axis limits
    #     pass
    # else:
    # ax.set_xlim(0.2, 1.1)  # set x axis limits
    # ax.axvline(x=groundtruth, color='0.9', ls='--', lw=2, zorder=0)  # add line at x=0
    ax.set_xlim(0, 0.3)
    ax.axvline(x=0, color='0.9', ls='--', lw=2, zorder=0)  # add line at x=0

    if log:
        plt.xscale("log")
    ax.set_xlabel('Bias of marginal hazard ratio')  # label the x axis
    ax.set_ylabel('Simulation Settings')  # label the y axis
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    if dump:
        check_and_mkdir(dirname + 'results/fig/')
        fig.savefig(dirname + 'results/fig/arrow_bias_reduce-{}-{}-{}{}.png'.format(model, contrl_type, datapart,
                                                                                    '-log' if log else ''))
        fig.savefig(dirname + 'results/fig/arrow_bias_reduce-{}-{}-{}{}.pdf'.format(model, contrl_type, datapart,
                                                                                    '-log' if log else ''))
    plt.show()
    plt.clf()


def arrow_plot_model_selection_mse_reduction(model, groundtruth_dict, contrl_type='all', dump=True,
                                             colorful=True, datapart='all', log=False, ):
    # dataset: train, test, all
    dirname = r'output/simulate_v3/{}/'.format(model)
    dfall = pd.read_excel(dirname + 'results/summarized_model_selection_{}.xlsx'.format(model),
                          sheet_name=contrl_type, converters={'drug': str})

    # print only
    c1 = 'val_auc-i'
    c2 = 'val_loss-i'
    c30 = 'trainval_n_unbalanced_feat_iptw-val_auc'
    c3 = 'trainval_n_unbalanced_feat_iptw-val_auc'  # val_loss

    idx_auc = dfall['success_rate-' + c1 + '-all_n_unbalanced_feat_iptw'] >= 0.1
    idx_smd = dfall['success_rate-' + c2 + '-all_n_unbalanced_feat_iptw'] >= 0.1
    idx = dfall['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'] >= 0.1

    print('Total drug trials: ', len(idx))
    print(r"#df[{}] > 0: ".format('success_rate-' + c1 + '-all_n_unbalanced_feat_iptw'), idx_auc.sum(),
          '({:.2f}%)'.format(idx_auc.mean() * 100))
    print(r"#df[{}] > 0: ".format('success_rate-' + c2 + '-all_n_unbalanced_feat_iptw'), idx_smd.sum(),
          '({:.2f}%)'.format(idx_smd.mean() * 100))
    print(r"#df[{}] > 0: ".format('success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'), idx.sum(),
          '({:.2f}%)'.format(idx.mean() * 100))

    # use all to compare
    idx = dfall['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'].notna()
    df = dfall.loc[idx, :].sort_values(by=['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'], ascending=[False])

    df = dfall

    N = len(df)
    drug_list = df.loc[idx, 'drug']
    drug_name_list = df.loc[idx, 'drug_name']

    data = [[], [], []]
    for drug, drugname in zip(drug_list, drug_name_list):
        rdf = pd.read_csv(dirname + 'results/' + drug + '_model_selection.csv')
        if contrl_type != 'all':
            idx = rdf['ctrl_type'] == contrl_type
        else:
            idx = rdf['ctrl_type'].notna()

        if 'no' in drugname:
            groundtruth = groundtruth_dict['no']
        elif 'moderate' in drugname:
            groundtruth = groundtruth_dict['moderate']
        else:
            raise ValueError

        for ith, c in enumerate([c1, c2, c3]):
            mse_ori = mean_squared_error(np.ones_like(idx) * groundtruth, np.array(
                rdf.loc[idx, c + '-{}_HR_ori'.format(datapart)]))  # mean_squared_error(y_true, y_pred)
            mse_iptw = mean_squared_error(np.ones_like(idx) * groundtruth, np.array(
                rdf.loc[idx, c + '-{}_HR_IPTW'.format(datapart)]))  # mean_squared_error(y_true, y_pred)
            change = mse_iptw - mse_ori

            data[ith].append([drugname.replace('simun', '').replace('train0.8', '').replace('no', '-lin-').replace(
                'moderate', '-nonLin-'),  # drugname[5:],
                              mse_ori, mse_iptw, change])

    data_df = []
    for d in data:
        df = pd.DataFrame(d, columns=['subject', 'before', 'after', 'change'], index=range(len(d)))
        data_df.append(df)

    fig = plt.figure(figsize=(5, 6))
    # add start points
    ax = sns.stripplot(data=data_df[1],
                       x='before',
                       y='subject',
                       orient='h',
                       order=data_df[1]['subject'],
                       size=8,
                       color='black',
                       facecolors="none",
                       jitter=0, alpha=0.7
                       )

    # plt.scatter(swarm1['x'] + strip_rx, swarm1['y'], alpha=.6, c=color)

    d_pos = [-0.15, 0, 0.15]
    colors = ['#FAC200', '#82A2D3', '#F65453']
    for igroup, data in enumerate(data_df):
        arrow_starts = data['before'].values
        arrow_lengths = data['change'].values
        arrow_ends = data['after'].values
        # add arrows to plot

        for i, subject in enumerate(data['subject']):
            prop = dict(arrowstyle="->,head_width=0.4,head_length=0.8",
                        shrinkA=0, shrinkB=0, color=colors[igroup])
            # plt.annotate("", xy=(arrow_ends[i], i+d_pos[igroup]), xytext=(arrow_starts[i], i+d_pos[igroup]), arrowprops=prop)
            # ax.arrow(row['2009'], idx, row['2013'] - row['2009'], 0, head_width=0.2, head_length=0.7, width=0.03, fc=c,
            #          ec=c)
            # plt.annotate(arrow_ends[i], xy=(arrow_ends[i], i+d_pos[igroup]), xytext=(arrow_ends[i]+d_pos[igroup], i+d_pos[igroup]), arrowprops=prop)

            ax.arrow(arrow_starts[i],  # x start point
                     i + d_pos[igroup],  # y start point
                     arrow_lengths[i],  # change in x
                     0,  # change in y
                     head_width=0.1,  # arrow head width
                     head_length=0.002,  # arrow head length
                     width=0.001,  # arrow stem width
                     fc=colors[igroup],  # arrow fill color
                     ec=colors[igroup]
                     )  # arrow edge color

    # format plot
    ax = sns.stripplot(data=data_df[1],
                       x='before',
                       y='subject',
                       orient='h',
                       order=data_df[1]['subject'],
                       size=8,
                       color='black',
                       facecolors="none", jitter=0, alpha=0.7
                       )
    ax.set_title('before vs. after re-weighting on ' + datapart)  # add title
    ax.grid(axis='y', color='0.9')  # add a light grid
    # if datapart == 'test':
    #     # ax.set_xlim(75, 150)  # set x axis limits
    #     pass
    # else:
    # ax.set_xlim(0.2, 1.1)  # set x axis limits
    # ax.axvline(x=groundtruth, color='0.9', ls='--', lw=2, zorder=0)  # add line at x=0
    ax.set_xlim(-0.01, 0.08)
    ax.axvline(x=0, color='0.9', ls='--', lw=2, zorder=0)  # add line at x=0

    if log:
        plt.xscale("log")
    ax.set_xlabel('MSE of marginal hazard ratio')  # label the x axis
    ax.set_ylabel('Simulation Settings')  # label the y axis
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    if dump:
        check_and_mkdir(dirname + 'results/fig/')
        fig.savefig(dirname + 'results/fig/arrow_MSE_reduce-{}-{}-{}{}.png'.format(model, contrl_type, datapart,
                                                                                   '-log' if log else ''))
        fig.savefig(dirname + 'results/fig/arrow_MSE_reduce-{}-{}-{}{}.pdf'.format(model, contrl_type, datapart,
                                                                                   '-log' if log else ''))
    plt.show()
    plt.clf()
    return data_df


def bar_plot_ahr_coverage(model, groundtruth_dict, contrl_type='all', dump=True, datapart='all', log=False, ):
    # dataset: train, test, all
    dirname = r'output/simulate_v3/{}/'.format(model)
    dfall = pd.read_excel(dirname + 'results/summarized_model_selection_{}.xlsx'.format(model),
                          sheet_name=contrl_type, converters={'drug': str})
    c1 = 'val_auc-i'
    c2 = 'val_loss-i'
    c30 = 'trainval_n_unbalanced_feat_iptw-val_auc'
    c3 = 'trainval_n_unbalanced_feat_iptw-val_auc'  # val_loss

    # select drug trials with at least 10% balanced trials
    # not used, just print here

    idx_auc = dfall['success_rate-' + c1 + '-all_n_unbalanced_feat_iptw'] >= 0.1
    idx_smd = dfall['success_rate-' + c2 + '-all_n_unbalanced_feat_iptw'] >= 0.1
    idx = dfall['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'] >= 0.1

    print('Total drug trials: ', len(idx))
    print(r"#df[{}] > 0.1: ".format('success_rate-' + c1 + '-all_n_unbalanced_feat_iptw'), idx_auc.sum(),
          '({:.2f}%)'.format(idx_auc.mean() * 100))
    print(r"#df[{}] > 0.1: ".format('success_rate-' + c2 + '-all_n_unbalanced_feat_iptw'), idx_smd.sum(),
          '({:.2f}%)'.format(idx_smd.mean() * 100))
    print(r"#df[{}] > 0.1: ".format('success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'), idx.sum(),
          '({:.2f}%)'.format(idx.mean() * 100))

    # use all to compare
    idx = dfall['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'].notna()
    # df = dfall.loc[idx, :].sort_values(by=['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'], ascending=[False])

    # just use name/experiment order, not the results based order
    df = dfall
    N = len(df)
    drug_list = df.loc[idx, 'drug']
    drug_name_list = df.loc[idx, 'drug_name']

    # df.loc[idx, :].to_csv(dirname + 'results/selected_balanced_drugs_for_screen.csv')
    # data_1 = []
    # data_2 = []
    # data_3 = []
    # data_pvalue = []
    data = [[], [], []]
    for drug, drugname in zip(drug_list, drug_name_list):
        rdf = pd.read_csv(dirname + 'results/' + drug + '_model_selection.csv')
        if contrl_type != 'all':
            idx = rdf['ctrl_type'] == contrl_type
        else:
            idx = rdf['ctrl_type'].notna()  # always use this

        if 'no' in drugname:
            groundtruth = groundtruth_dict['no']
        elif 'moderate' in drugname:
            groundtruth = groundtruth_dict['moderate']
        else:
            raise ValueError

        for ith, c in enumerate([c1, c2, c3]):
            before = np.abs(groundtruth - np.array(rdf.loc[idx, c + '-{}_HR_ori'.format(datapart)]))
            after = np.abs(groundtruth - np.array(rdf.loc[idx, c + '-{}_HR_IPTW'.format(datapart)]))
            change = after - before
            # before_med = IQR(before)[0]
            # before_iqr = IQR(before)[1:]
            before_mean = before.mean()
            before_std = before.std()
            # before_mean_ci, before_mean_std = bootstrap_mean_ci(before, alpha=0.05)
            # after_med = IQR(after)[0]
            # after_iqr = IQR(after)[1:]
            after_mean = after.mean()
            after_std = after.std()
            # after_mean_ci, before_mean_std = bootstrap_mean_ci(after, alpha=0.05)
            # change_med = IQR(change)[0]
            # change_iqr = IQR(change)[1:]
            change_mean = change.mean()
            change_std = change.std()
            # change_mean_ci, change_mean_std = bootstrap_mean_ci(change, alpha=0.05)

            mse_ori = mean_squared_error(np.ones_like(idx) * groundtruth, np.array(
                rdf.loc[idx, c + '-{}_HR_ori'.format(datapart)]))  # mean_squared_error(y_true, y_pred)
            mse_iptw = mean_squared_error(np.ones_like(idx) * groundtruth, np.array(
                rdf.loc[idx, c + '-{}_HR_IPTW'.format(datapart)]))  # mean_squared_error(y_true, y_pred)

            ahr_iptw = np.array(rdf.loc[idx, c + '-{}_HR_IPTW'.format(datapart)])
            ahr_ori = np.array(rdf.loc[idx, c + '-{}_HR_ori'.format(datapart)])
            ahr_mean_iptw = ahr_iptw.mean()
            ahr_var_iptw = ahr_iptw.std()
            # ahr_mean_ori = ahr_ori.mean()
            # ahr_var_ori = ahr_ori.std()

            ahr_iptw_left = ahr_iptw - 1.96 * ahr_var_iptw
            ahr_iptw_right = ahr_iptw + 1.96 * ahr_var_iptw

            coverage_n = ((ahr_iptw_left <= groundtruth) & (groundtruth <= ahr_iptw_right)).sum()
            coverage_mean = ((ahr_iptw_left <= groundtruth) & (groundtruth <= ahr_iptw_right)).mean()

            # data[ith].append([drugname[4:],
            #                   before_mean, after_mean, change_mean,
            #                   before_std, after_std, change_std,
            #                   mse_ori, mse_iptw,
            #                   ahr_mean_iptw, ahr_var_iptw, coverage_n, coverage_mean])
            data[ith].append([drugname.replace('simun', '').replace('train0.8', '').replace('no', '-lin-').replace(
                'moderate', '-nonLin-'),  # drugname[5:],
                              ahr_mean_iptw, ahr_var_iptw,
                              after_mean, mse_iptw, coverage_mean])

    data_df = []
    for d, tab in zip(data, [c1, c2, 'our']):
        df_ate = pd.DataFrame(d, columns=['Exp setup',
                                          "ahr_mean_iptw", "ahr_var_iptw",
                                          'iptw bias mean', "mse_iptw", "coverage_mean"
                                          ],
                              index=range(len(d)))
        data_df.append(df_ate)

    N = len(data_df[0])
    top_1 = data_df[0]["coverage_mean"]  # df.loc[:, c1]  # * 100
    # top_1_ci = np.array(
    #     df.loc[:, c1.replace('success_rate', 'success_rate_ci')].apply(
    #         lambda x: stringlist_2_list(x)).to_list())  # *100
    # top_1_ci = df.loc[:, 'success_rate_std-val_auc_nsmd']

    top_2 = data_df[1]["coverage_mean"]  # df.loc[:, c2]  # * 100
    # top_2_ci = np.array(
    #     df.loc[:, c2.replace('success_rate', 'success_rate_ci')].apply(
    #         lambda x: stringlist_2_list(x)).to_list())  # *100
    # top_2_ci = df.loc[:, 'success_rate_std-val_maxsmd_nsmd']

    top_3 = data_df[2]["coverage_mean"]  # df.loc[:, c3]  # * 100
    # top_3_ci = np.array(
    #     df.loc[:, c3.replace('success_rate', 'success_rate_ci')].apply(
    #         lambda x: stringlist_2_list(x)).to_list())  # *100
    # top_3_ci = df.loc[:, 'success_rate_std-trainval_final_finalnsmd']

    # pauc = np.array(df.loc[:, "p-succes-final-vs-1"])
    # psmd = np.array(df.loc[:, "p-succes-final-vs-2"])
    # paucsmd = np.array(df.loc[:, "p-succes-1-vs-2"])

    # xlabels = df.loc[:, 'drug_name']
    # xlabels = [x[5:] for x in xlabels]
    xlabels = data_df[0]['Exp setup']

    width = 0.45  # the width of the bars
    ind = np.arange(N) * width * 4  # the x locations for the groups

    colors = ['#FAC200', '#82A2D3', '#F65453']
    fig, ax = plt.subplots(figsize=(18, 8))
    error_kw = {'capsize': 3, 'capthick': 1, 'ecolor': 'black'}
    # plt.ylim([0, 1.05])
    rects1 = ax.bar(ind, top_1, width,  # yerr=[top_1 - top_1_ci[:, 0], top_1_ci[:, 1] - top_1], error_kw=error_kw,
                    color=colors[0], edgecolor=None)  # , edgecolor='b' "black"
    rects2 = ax.bar(ind + width, top_2, width,
                    # yerr=[top_2 - top_2_ci[:, 0], top_2_ci[:, 1] - top_2], error_kw=error_kw,
                    color=colors[1], edgecolor=None)
    rects3 = ax.bar(ind + 2 * width, top_3, width,
                    # yerr=[top_3 - top_3_ci[:, 0], top_3_ci[:, 1] - top_3], error_kw=error_kw,
                    color=colors[2], edgecolor=None)  # , hatch='.')

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
    ax.set_xlabel("Simulation Settings", fontsize=25)
    # ax.set_ylabel("Prop. of success balancing", fontsize=25)  # Success Rate of Balancing
    ax.set_ylabel("CI coverage of truth (%)", fontsize=25)  # Success Rate of Balancing

    # ax.set_title(model, fontsize=25) #fontweight="bold")
    # plt.axhline(y=0.5, color='#888888', linestyle='-')

    def significance(val):
        if val < 0.001:
            return '***'
        elif val < 0.01:
            return '**'
        elif val < 0.05:
            return '*'
        else:
            return 'ns'

    # ax.set_title('Success Rate of Balancing by Different PS Model Selection Methods')
    ax.legend((rects1[0], rects2[0], rects3[0]), ('Val-AUC Select', 'Val-Loss Select', 'Our Strategy'),
              fontsize=25, loc='center right')  # , bbox_to_anchor=(1.13, 1.01))

    # ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_xmargin(0.01)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if dump:
        check_and_mkdir(dirname + 'results/fig/')
        fig.savefig(dirname + 'results/fig/true_ahr_coverage_barplot-{}-{}.png'.format(model, contrl_type))
        fig.savefig(dirname + 'results/fig/true_ahr_coverage_barplot-{}-{}.pdf'.format(model, contrl_type))
    plt.show()
    plt.clf()


def aHR_evaluation_table(model, groundtruth_dict, contrl_type='all', dump=True, colorful=True, datapart='all', log=False, ):
    # dataset: train, test, all
    dirname = r'output/simulate_v3/{}/'.format(model)
    dfall = pd.read_excel(dirname + 'results/summarized_model_selection_{}.xlsx'.format(model),
                          sheet_name=contrl_type, converters={'drug': str})
    c1 = 'val_auc-i'
    c2 = 'val_loss-i'
    c30 = 'trainval_n_unbalanced_feat_iptw-val_auc'
    c3 = 'trainval_n_unbalanced_feat_iptw-val_auc'  # val_loss
    # select drug trials with at least 10% balanced trials
    idx_auc = dfall['success_rate-' + c1 + '-all_n_unbalanced_feat_iptw'] >= 0.1
    idx_smd = dfall['success_rate-' + c2 + '-all_n_unbalanced_feat_iptw'] >= 0.1
    idx = dfall['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'] >= 0.1

    print('Total drug trials: ', len(idx))
    print(r"#df[{}] > 0.1: ".format('success_rate-' + c1 + '-all_n_unbalanced_feat_iptw'), idx_auc.sum(),
          '({:.2f}%)'.format(idx_auc.mean() * 100))
    print(r"#df[{}] > 0.1: ".format('success_rate-' + c2 + '-all_n_unbalanced_feat_iptw'), idx_smd.sum(),
          '({:.2f}%)'.format(idx_smd.mean() * 100))
    print(r"#df[{}] > 0.1: ".format('success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'), idx.sum(),
          '({:.2f}%)'.format(idx.mean() * 100))

    idx = dfall['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'].notna()
    # df = dfall.loc[idx, :].sort_values(by=['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'], ascending=[False])

    # just use name/experiment order, not the results based order
    df = dfall
    N = len(df)
    drug_list = df.loc[idx, 'drug']
    drug_name_list = df.loc[idx, 'drug_name']

    # df.loc[idx, :].to_csv(dirname + 'results/selected_balanced_drugs_for_screen.csv')
    # data_1 = []
    # data_2 = []
    # data_3 = []
    # data_pvalue = []
    data = [[], [], []]
    for drug, drugname in zip(drug_list, drug_name_list):
        rdf = pd.read_csv(dirname + 'results/' + drug + '_model_selection.csv')
        if contrl_type != 'all':
            idx = rdf['ctrl_type'] == contrl_type
        else:
            idx = rdf['ctrl_type'].notna()  # always use this

        if 'no' in drugname:
            groundtruth = groundtruth_dict['no']
        elif 'moderate' in drugname:
            groundtruth = groundtruth_dict['moderate']
        else:
            raise ValueError

        for ith, c in enumerate([c1, c2, c3]):
            before = np.abs(groundtruth - np.array(rdf.loc[idx, c + '-{}_HR_ori'.format(datapart)]))
            after = np.abs(groundtruth - np.array(rdf.loc[idx, c + '-{}_HR_IPTW'.format(datapart)]))
            change = after - before
            # before_med = IQR(before)[0]
            # before_iqr = IQR(before)[1:]
            before_mean = before.mean()
            before_std = before.std()
            # before_mean_ci, before_mean_std = bootstrap_mean_ci(before, alpha=0.05)
            # after_med = IQR(after)[0]
            # after_iqr = IQR(after)[1:]
            after_mean = after.mean()
            after_std = after.std()
            # after_mean_ci, before_mean_std = bootstrap_mean_ci(after, alpha=0.05)
            # change_med = IQR(change)[0]
            # change_iqr = IQR(change)[1:]
            change_mean = change.mean()
            change_std = change.std()
            # change_mean_ci, change_mean_std = bootstrap_mean_ci(change, alpha=0.05)

            mse_ori = mean_squared_error(np.ones_like(idx) * groundtruth, np.array(
                rdf.loc[idx, c + '-{}_HR_ori'.format(datapart)]))  # mean_squared_error(y_true, y_pred)
            mse_iptw = mean_squared_error(np.ones_like(idx) * groundtruth, np.array(
                rdf.loc[idx, c + '-{}_HR_IPTW'.format(datapart)]))  # mean_squared_error(y_true, y_pred)

            ahr_iptw = np.array(rdf.loc[idx, c + '-{}_HR_IPTW'.format(datapart)])
            ahr_ori = np.array(rdf.loc[idx, c + '-{}_HR_ori'.format(datapart)])
            ahr_mean_iptw = ahr_iptw.mean()
            ahr_var_iptw = ahr_iptw.std()
            # ahr_mean_ori = ahr_ori.mean()
            # ahr_var_ori = ahr_ori.std()

            ahr_iptw_left = ahr_iptw - 1.96 * ahr_var_iptw
            ahr_iptw_right = ahr_iptw + 1.96 * ahr_var_iptw

            coverage_n = ((ahr_iptw_left <= groundtruth) & (groundtruth <= ahr_iptw_right)).sum()
            coverage_mean = ((ahr_iptw_left <= groundtruth) & (groundtruth <= ahr_iptw_right)).mean()

            # data[ith].append([drugname[4:],
            #                   before_mean, after_mean, change_mean,
            #                   before_std, after_std, change_std,
            #                   mse_ori, mse_iptw,
            #                   ahr_mean_iptw, ahr_var_iptw, coverage_n, coverage_mean])
            data[ith].append([drugname.replace('simun', '').replace('train0.8', '').replace('no', '-lin-').replace(
                'moderate', '-nonLin-'),  # drugname[4:],
                              ahr_mean_iptw, ahr_var_iptw,
                              after_mean, mse_iptw, coverage_mean])

    data_df = []
    writer = pd.ExcelWriter(dirname + 'results/aHR_evaluation_table_{}_simple.xlsx'.format(model), engine="xlsxwriter")
    # columns=['subject',
    #                                       'ori bias mean', 'iptw bias mean', 'bias reduction mean',
    #                                       'ori bias std', 'iptw bias std', 'bias reduction std',
    #                                       "mse_ori", "mse_iptw",
    #                                       "ahr_mean_iptw", "ahr_var_iptw", "coverage_n", "coverage_mean"],
    for d, tab in zip(data, [c1, c2, 'our']):
        df = pd.DataFrame(d, columns=['Exp setup',
                                      "ahr_mean_iptw", "ahr_var_iptw",
                                      'iptw bias mean', "mse_iptw", "coverage_mean"
                                      ],
                          index=range(len(d)))
        data_df.append(df)
        df.to_excel(writer, sheet_name=tab)
    writer.close()

    return data_df


if __name__ == '__main__':
    # # 2023-9-11
    # # 2023-9-19
    # shell_for_ml_simulation('LR', niter=100, start=0, more_para='')  #
    # split_shell_file("simulate_v3_shell_LR.sh", divide=8, skip_first=1)
    # sys.exit(0)
    #
    # # 2023-7-6
    # # shell_for_ml_simulation('LR', niter=50, start=0, more_para='')  #
    # # split_shell_file("simulate_v2_shell_LR.sh", divide=4, skip_first=1)
    # #
    # # # shell_for_ml_simulation('LIGHTGBM', niter=100, start=0, more_para='') #
    # # # split_shell_file("simulate_shell_LIGHTGBM-server2.sh", divide=5, skip_first=1)
    # # # sys.exit(0)
    # # zz

    # cohort_dir_name = 'save_cohort_all_loose'
    model = 'LR'  # 'LR'  # 'MLP'  # 'LR' #'LIGHTGBM'  #'LR'  #'LSTM'
    # results_model_selection_for_ml(model=model, niter=100) #100
    # results_model_selection_for_ml_step2(model=model)
    # # # results_model_selection_for_ml_step2More(cohort_dir_name=cohort_dir_name, model=model, drug_name=drug_name)
    #
    # # major plots from 3 methods
    # # bar_plot_model_selection(cohort_dir_name=cohort_dir_name, model=model, contrl_type='random')
    # # bar_plot_model_selection(cohort_dir_name=cohort_dir_name, model=model, contrl_type='atc')
    # bar_plot_model_selection(model=model, contrl_type='all')
    #
    # arrow_plot_model_selection_unbalance_reduction(model=model, datapart='all')
    # arrow_plot_model_selection_unbalance_reduction(model=model, datapart='train')
    # arrow_plot_model_selection_unbalance_reduction(model=model, datapart='test')
    #
    # # simulation sample number args.nsim:  1000000
    #
    ##groundtruth = 0.578  #4870728502016 # 0.5781950897226341
    groundtruth = {'no': 0.5780982066480141, 'moderate': 0.5780913629899157}

    # arrow_plot_model_selection_bias_reduction(model=model, groundtruth_dict=groundtruth, datapart='all')
    # arrow_plot_model_selection_bias_reduction(model=model, groundtruth_dict=groundtruth, datapart='train')
    # arrow_plot_model_selection_bias_reduction(model=model, groundtruth_dict=groundtruth, datapart='test')
    #
    # data_df = arrow_plot_model_selection_mse_reduction(model=model, groundtruth_dict=groundtruth, datapart='all')
    # arrow_plot_model_selection_mse_reduction(model=model, groundtruth_dict=groundtruth, datapart='train')
    # arrow_plot_model_selection_mse_reduction(model=model, groundtruth_dict=groundtruth, datapart='test')
    #

    # data_df = aHR_evaluation_table(model=model, groundtruth_dict=groundtruth, datapart='all')
    bar_plot_ahr_coverage(model=model, groundtruth_dict=groundtruth, datapart='all')

    print('Done')
