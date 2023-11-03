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

print = functools.partial(print, flush=True)

MAX_NO_UNBALANCED_FEATURE = 5  # 0 # 10 #5
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
    fo = open('simulate_shell_{}-server2-part2.sh'.format(model), 'w')  # 'a'
    fo.write('mkdir -p output/simulate/{}/log\n'.format(model))
    r = 0
    for n in [3000, 3500, 4000, 4500, 5000]:  # [2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]: #[2000, 4000, 6000]:
        for train_ratio in [0.8, 0.6]:
            drug = 'simun{}train{:.2f}'.format(n, train_ratio)
            for seed in range(start, niter):
                cmd = "python main_revise_testset_simulate.py --nsim {} --train_ratio {} " \
                      "--run_model {} --output_dir output/simulate/{}/ --random_seed {} {}" \
                      "2>&1 | tee output/simulate/{}/log/{}_S{}D267_{}.log\n".format(
                    n, train_ratio,
                    model, model, seed, more_para,
                    model, drug, seed, model)
                fo.write(cmd)
                r += 1
    fo.close()
    print('In total ', r, ' commands')


def shell_for_ml(cohort_dir_name, model, niter=50, min_patients=500, stats=True, more_para=''):
    cohort_size = pickle.load(open(r'../ipreprocess/output/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    fo = open('revise_shell_{}_{}.sh'.format(model, cohort_dir_name), 'w')  # 'a'
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

    fo.write('mkdir -p output/revise/{}/{}/log\n'.format(cohort_dir_name, model))
    n = 0
    for x in name_cnt:
        k, v = x
        if (v >= min_patients) or (k in added_drug):
            drug = k.split('.')[0]
            for ctrl_type in ['random', 'atc']:
                for seed in range(0, niter):
                    cmd = "python main_revise.py --data_dir ../ipreprocess/output/{}/ --treated_drug {} " \
                          "--controlled_drug {} --run_model {} --output_dir output/revise/{}/{}/ --random_seed {} " \
                          "--drug_coding rxnorm --med_code_topk 200 {} {} " \
                          "2>&1 | tee output/revise/{}/{}/log/{}_S{}D200C{}_{}.log\n".format(
                        cohort_dir_name, drug,
                        ctrl_type, model, cohort_dir_name, model, seed, '--stats' if stats else '', more_para,
                        cohort_dir_name, model, drug, seed, ctrl_type, model)
                    fo.write(cmd)
                    n += 1

    fo.close()
    print('In total ', n, ' commands')


def shell_for_ml_selected_drugs(drug_list, cohort_dir_name, model, niter=50, min_patients=500, stats=True,
                                more_para='', folder='revise_testset'):
    cohort_size = pickle.load(open(r'../ipreprocess/output/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    fo = open('revise_{}_shell_{}_{}_selected.sh'.format(folder, model, cohort_dir_name), 'w')  # 'a'
    name_cnt = sorted(cohort_size.items(), key=lambda x: x[1], reverse=True)

    # load others:
    # df = pd.read_excel(r'../data/repurposed_AD_under_trials_20200227.xlsx', dtype=str)
    added_drug = []
    for rx in drug_list:
        if pd.notna(rx):
            rx = [x + '.pkl' for x in re.split('[,;+]', rx)]
            added_drug.extend(rx)

    print('len(added_drug): ', len(added_drug))
    print(added_drug)

    fo.write('mkdir -p output/{}/{}/{}/log\n'.format(folder, cohort_dir_name, model))
    n = 0
    for x in name_cnt:
        k, v = x
        if k in added_drug:
            drug = k.split('.')[0]
            for ctrl_type in ['random', 'atc']:
                for seed in range(0, niter):
                    cmd = "python main_revise_testset.py --data_dir ../ipreprocess/output/{}/ --treated_drug {} " \
                          "--controlled_drug {} --run_model {} --output_dir output/{}/{}/{}/ --random_seed {} " \
                          "--drug_coding rxnorm --med_code_topk 200 {} {} " \
                          "2>&1 | tee output/{}/{}/{}/log/{}_S{}D200C{}_{}.log\n".format(
                        cohort_dir_name, drug,
                        ctrl_type, model, folder, cohort_dir_name, model, seed, '--stats' if stats else '', more_para,
                        folder, cohort_dir_name, model, drug, seed, ctrl_type, model)
                    fo.write(cmd)
                    n += 1

    fo.close()
    print('In total ', n, ' commands')


def shell_for_ml_marketscan(cohort_dir_name, model, niter=50, min_patients=500, stats=True, more_para=''):
    cohort_size = pickle.load(
        open(r'../ipreprocess/output_marketscan/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    fo = open('shell_{}_{}_marketscan.sh'.format(model, cohort_dir_name), 'w')  # 'a'
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

    fo.write('mkdir -p output_marketscan/{}/{}/log\n'.format(cohort_dir_name, model))
    n = 0
    for x in name_cnt:
        k, v = x
        if (v >= min_patients) or (k in added_drug):
            drug = k.split('.')[0]
            for ctrl_type in ['random', 'atc']:
                for seed in range(0, niter):
                    cmd = "python main.py --data_dir ../ipreprocess/output_marketscan/{}/ --treated_drug {} " \
                          "--controlled_drug {} --run_model {} --output_dir output_marketscan/{}/{}/ --random_seed {} " \
                          "--drug_coding gpi --med_code_topk 200 {} {} " \
                          "2>&1 | tee output_marketscan/{}/{}/log/{}_S{}D200C{}_{}.log\n".format(
                        cohort_dir_name, drug,
                        ctrl_type, model, cohort_dir_name, model, seed, '--stats' if stats else '', more_para,
                        cohort_dir_name, model, drug, seed, ctrl_type, model)
                    fo.write(cmd)
                    n += 1

    fo.close()
    print('In total ', n, ' commands')


def shell_for_ml_marketscan_stats_exist(cohort_dir_name, model, niter=10, min_patients=500):
    cohort_size = pickle.load(
        open(r'../ipreprocess/output_marketscan/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    fo = open('shell_{}_{}_marketscan_stats_exist.sh'.format(model, cohort_dir_name), 'w')  # 'a'
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

    fo.write('mkdir -p output_marketscan/{}/{}/log_stats_exit\n'.format(cohort_dir_name, model))
    n = 0
    for x in name_cnt:
        k, v = x
        if (v >= min_patients) or (k in added_drug):
            drug = k.split('.')[0]
            for ctrl_type in ['random', 'atc']:
                for seed in range(0, niter):
                    cmd = "python main.py --data_dir ../ipreprocess/output_marketscan/{}/ --treated_drug {} " \
                          "--controlled_drug {} --run_model {} --output_dir output_marketscan/{}/{}/ --random_seed {} " \
                          "--drug_coding gpi --med_code_topk 200 --stats --stats_exit " \
                          "2>&1 | tee output_marketscan/{}/{}/log_stats_exit/{}_S{}D200C{}_{}.log\n".format(
                        cohort_dir_name, drug,
                        ctrl_type, model, cohort_dir_name, model, seed,
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
                    df.rename(columns=_simplify_col_, inplace=True)
                except:
                    print('No file exisits: ', fname + '_ALL-model-select.csv')
                    continue

                selection_configs = [
                    ('val_auc', 'i', False, True), ('val_loss', 'i', True, True),
                    ('val_max_smd_iptw', 'i', True, True), ('val_n_unbalanced_feat_iptw', 'i', True, True),
                    ('train_auc', 'i', False, True), ('train_loss', 'i', True, True),
                    ('train_max_smd_iptw', 'i', True, True), ('train_n_unbalanced_feat_iptw', 'i', True, True),
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
                              'train_n_unbalanced_feat_iptw',
                              'test_loss', 'test_auc', 'test_max_smd', 'test_max_smd_iptw', 'test_n_unbalanced_feat',
                              'test_n_unbalanced_feat_iptw',
                              'all_loss', 'all_auc', 'all_max_smd', 'all_max_smd_iptw', 'all_n_unbalanced_feat',
                              'all_n_unbalanced_feat_iptw',
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

                results.append(["{}_S{}D200C{}_{}".format(drug, seed, ctrl_type, model), ctrl_type] + selection_results)

        rdf = pd.DataFrame(results, columns=['fname', 'ctrl_type'] + selection_results_colname)
        rdf.to_csv(dirname + 'results/' + drug + '_model_selection.csv')

        pre_col = []
        for col1, col2, order1, order2 in selection_configs:
            pre_col.append('{}-{}-'.format(col1, col2))

        for t in ['random', 'atc', 'all']:
            # fig = plt.figure(figsize=(20, 15))
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 18))
            if t != 'all':
                idx = rdf['ctrl_type'] == t
            else:
                idx = rdf['ctrl_type'].notna()
            boxplot = rdf[idx].boxplot(column=[x + 'all_n_unbalanced_feat_iptw' for x in pre_col], fontsize=15, ax=ax1,
                                       showmeans=True)  # rot=25,
            ax1.axhline(y=5, color='r', linestyle='-')
            boxplot.set_title("{}-{}_S{}D200C{}_{}".format(drug, drug_name.get(drug), '0-19', t, model), fontsize=25)
            # plt.xlabel("Model selection methods", fontsize=15)
            ax1.set_ylabel("#unbalanced_feat_iptw of boostrap experiments", fontsize=20)
            # print(ax1.get_xticklabels())
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
            # fig.savefig(dirname + 'results/' + drug + '_model_selection_boxplot-{}-allnsmd.png'.format(t))
            # plt.show()

            # fig = plt.figure(figsize=(20, 15))
            boxplot = rdf[idx].boxplot(column=[x + 'test_auc' for x in pre_col], fontsize=15, ax=ax2, showmeans=True)
            # plt.axhline(y=0.5, color='r', linestyle='-')
            # boxplot.set_title("{}-{}_S{}D200C{}_{}".format(drug, drug_name.get(drug), '0-19', t, model), fontsize=25)
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


def results_model_selection_for_ml_step2(cohort_dir_name, model, drug_name):
    cohort_size = pickle.load(open(r'../ipreprocess/output/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    name_cnt = sorted(cohort_size.items(), key=lambda x: x[1], reverse=True)
    drug_list_all = [drug.split('.')[0] for drug, cnt in name_cnt]
    dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
    drug_in_dir = set([x for x in os.listdir(dirname) if x.isdigit()])
    drug_list = [x for x in drug_list_all if x in drug_in_dir]  # in order
    check_and_mkdir(dirname + 'results/')

    writer = pd.ExcelWriter(dirname + 'results/summarized_model_selection_{}.xlsx'.format(model), engine='xlsxwriter')
    for t in ['all', 'random', 'atc']:
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

            r = [drug, drug_name.get(drug, '')]
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


def results_ATE_for_ml(cohort_dir_name, model, niter=50):
    cohort_size = pickle.load(open(r'../ipreprocess/output/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    name_cnt = sorted(cohort_size.items(), key=lambda x: x[1], reverse=True)
    drug_list_all = [drug.split('.')[0] for drug, cnt in name_cnt]
    dirname = r'output/revise/{}/{}/'.format(cohort_dir_name, model)
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
                    r = df.loc[df.index[df['Unnamed: 0'] == 'all'].tolist()[0], :]
                    for c in ["KM_time_points", "KM1_original", "KM0_original", "KM1-0_original", "KM1-0_original_p",
                              "KM1_IPTW", "KM0_IPTW", "KM1-0_IPTW", "KM1-0_IPTW_p"]:
                        r.loc[c] = stringlist_2_list(r.loc[c])[-1]
                    r = pd.Series(["{}_S{}D200C{}_{}".format(drug, seed, ctrl_type, model), ctrl_type],
                                  index=['fname', 'ctrl_type']).append(r)
                    results.append(r)

                except:
                    print('No file exisits: ', fname + '_results.csv')
                    continue

        rdf = pd.DataFrame(results)
        rdf.to_excel(dirname + 'results/' + drug + '_results.xlsx')
    print('Done')


def results_ATE_for_ml_step2(cohort_dir_name, model, drug_name):
    cohort_size = pickle.load(open(r'../ipreprocess/output/{}/cohorts_size.pkl'.format(cohort_dir_name), 'rb'))
    name_cnt = sorted(cohort_size.items(), key=lambda x: x[1], reverse=True)
    drug_list_all = [drug.split('.')[0] for drug, cnt in name_cnt]
    dirname = r'output/revise/{}/{}/'.format(cohort_dir_name, model)
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

            print('drug: ', drug, drug_name.get(drug, ''), t, 'support:', idx.sum())
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
                    mean_ci, _ = bootstrap_mean_ci(nv, alpha=0.05)
                    r.extend([med, iqr, mean, mean_ci])
                else:
                    r.extend([np.nan, np.nan, np.nan, np.nan])
                col_name.extend(["med-" + c, "iqr-" + c, "mean-" + c, "mean_ci-" + c])

                nv = rdf.loc[idx_all, c]
                if len(nv) > 0:
                    med = IQR(nv)[0]
                    iqr = IQR(nv)[1:]

                    mean = nv.mean()
                    mean_ci, _ = bootstrap_mean_ci(nv, alpha=0.05)
                    r.extend([med, iqr, mean, mean_ci])
                else:
                    r.extend([np.nan, np.nan, np.nan, np.nan])
                col_name.extend(
                    ["med-" + c + '-uab', "iqr-" + c + '-uab', "mean-" + c + '-uab', "mean_ci-" + c + '-uab'])

            for c in ["ATE_original", "ATE_IPTW",
                      "KM1-0_original", "KM1-0_IPTW_p", "KM1-0_IPTW", "KM1-0_IPTW_p",
                      'HR_ori', 'HR_ori_p', 'HR_IPTW', 'HR_IPTW_p']:
                if c not in rdf.columns:
                    continue

                nv = rdf.loc[idx, c]
                if len(nv) > 0:
                    med = IQR(nv)[0]
                    iqr = IQR(nv)[1:]

                    mean = nv.mean()
                    mean_ci, _ = bootstrap_mean_ci(nv, alpha=0.05)

                    # if 'HR' in c:
                    #     p, _ = bootstrap_mean_pvalue(nv, expected_mean=1)
                    # else:
                    #     p, _ = bootstrap_mean_pvalue(nv, expected_mean=0)

                    # r.extend([med, iqr, mean, mean_ci, p])
                    r.extend([med, iqr, mean, mean_ci])
                else:
                    # r.extend([np.nan, np.nan, np.nan, np.nan, np.nan])
                    r.extend([np.nan, np.nan, np.nan, np.nan])

                # col_name.extend(["med-" + c, "iqr-" + c, "mean-" + c, "mean_ci-" + c, 'pvalue-' + c])
                col_name.extend(["med-" + c, "iqr-" + c, "mean-" + c, "mean_ci-" + c, ])

            if len(nv) > 0:
                nv1 = rdf.loc[idx, 'HR_ori']
                p1, _ = bootstrap_mean_pvalue(nv1, expected_mean=1)

                nv2 = rdf.loc[idx, 'HR_IPTW']
                p2, _ = bootstrap_mean_pvalue(nv2, expected_mean=1)

                nv3 = rdf.loc[idx, 'KM1-0_original']
                p3, _ = bootstrap_mean_pvalue(nv3, expected_mean=0)

                nv4 = rdf.loc[idx, 'KM1-0_IPTW']
                p4, _ = bootstrap_mean_pvalue(nv4, expected_mean=0)

                r.extend([p1, p2, p3, p4])

            else:
                r.extend([np.nan, np.nan, np.nan, np.nan])
            col_name.extend(['pvalue-HR_ori-boostrap', 'pvalue-HR_IPTW-boostrap',
                             'pvalue-KM1-0_original-boostrap', 'pvalue-KM1-0_IPTW-boostrap'])

            if len(nv) > 0:
                nv = rdf.loc[idx, 'HR_IPTW_p']
                _rid = nv[nv == nv.quantile(interpolation='nearest')].index[0]
                HR_ori_CI = rdf.loc[_rid, 'HR_ori_CI']
                HR_IPTW_CI = rdf.loc[_rid, 'HR_IPTW_CI']
                r.extend([HR_ori_CI, HR_IPTW_CI])
            else:
                r.extend([np.nan, np.nan])

            col_name.extend(["med-HR_ori_CI", "med-HR_IPTW_CI"])

            if 'HR_ori_CI' in rdf.columns:
                r.append(';'.join(rdf.loc[idx, 'HR_ori_CI']))
                col_name.append('HR_ori_CI')
            r.append(';'.join(rdf.loc[idx, 'HR_IPTW_CI']))
            col_name.append('HR_IPTW_CI')

            results.append(r)
        df = pd.DataFrame(results, columns=col_name)
        df.to_excel(writer, sheet_name=t)
    writer.save()
    print()


def results_ATE_for_ml_step3_finalInfo(cohort_dir_name, model):
    dirname = r'output/revise/{}/{}/'.format(cohort_dir_name, model)
    df_all = pd.read_excel(dirname + 'results/summarized_IPTW_ATE_{}.xlsx'.format(model), sheet_name=None)
    writer = pd.ExcelWriter(dirname + 'results/summarized_IPTW_ATE_{}_finalInfo.xlsx'.format(model),
                            engine='xlsxwriter')
    for sheet in ['all', 'random', 'atc']:
        df = df_all[sheet]
        # Only select drugs with selection criteria trial
        # 1. minimum support set 10, may choose 20 later
        # 2. p value < 0.05
        # idx = (df['support'] >= 10) & (df['pvalue-KM1-0_IPTW'] <= 0.05)
        # idx = (df['support'] >= 10) & (df['med-HR_IPTW_p'] <= 0.05)
        # idx = (df['support'] >= 10) & (df['pvalue-HR_IPTW-boostrap'] <= 0.05)
        idx = (df['support'] >= 10) & (df['pvalue-KM1-0_IPTW-boostrap'] <= 0.05)

        # df_sort = df.loc[idx, :].sort_values(by=['mean-KM1-0_IPTW'], ascending=[False])
        df_sort = df.loc[idx, :].sort_values(by=['mean-HR_IPTW'], ascending=[True])

        df_final = df_sort[
            ['drug', 'drug_name', 'niters', 'support', 'n_treat', 'n_ctrl', 'n_feature',
             'mean-n_unbalanced_feature', 'mean_ci-n_unbalanced_feature',
             'mean-n_unbalanced_feature_IPTW', 'mean_ci-n_unbalanced_feature_IPTW',
             # 'mean-ATE_original', 'mean_ci-ATE_original', 'pvalue-ATE_original',
             # 'mean-ATE_IPTW', 'mean_ci-ATE_IPTW', 'pvalue-ATE_IPTW',
             # 'mean-KM1-0_original', 'mean_ci-KM1-0_original', 'pvalue-KM1-0_original',
             'mean-KM1-0_IPTW', 'mean_ci-KM1-0_IPTW', 'mean-KM1-0_IPTW_p',  # 'pvalue-KM1-0_IPTW',
             'mean-HR_IPTW', 'med-HR_IPTW', 'mean_ci-HR_IPTW', 'mean-HR_IPTW_p', 'med-HR_IPTW_p',  # 'pvalue-HR_IPTW'
             "med-HR_ori_CI", "med-HR_IPTW_CI",
             'pvalue-HR_ori-boostrap', 'pvalue-HR_IPTW-boostrap', 'pvalue-KM1-0_original-boostrap',
             'pvalue-KM1-0_IPTW-boostrap'
             ]]

        df_final['n_ctrl'] = df_final['n_ctrl'].apply(
            lambda x: '{:.1f}'.format(x))

        df_final['mean-n_unbalanced_feature'] = df_final['mean-n_unbalanced_feature'].apply(
            lambda x: '{:.1f}'.format(x))
        df_final['mean_ci-n_unbalanced_feature'] = df_final['mean_ci-n_unbalanced_feature'].apply(
            lambda x: stringlist_2_str(x, False, 1))

        df_final['mean-n_unbalanced_feature_IPTW'] = df_final['mean-n_unbalanced_feature_IPTW'].apply(
            lambda x: '{:.1f}'.format(x))
        df_final['mean_ci-n_unbalanced_feature_IPTW'] = df_final['mean_ci-n_unbalanced_feature_IPTW'].apply(
            lambda x: stringlist_2_str(x, False, 1))

        df_final['mean-KM1-0_IPTW'] = df_final['mean-KM1-0_IPTW'].apply(
            lambda x: '{:.1f}'.format(x * 100))
        df_final['mean_ci-KM1-0_IPTW'] = df_final['mean_ci-KM1-0_IPTW'].apply(
            lambda x: stringlist_2_str(x, True, 1))

        df_final['mean-HR_IPTW'] = df_final['mean-HR_IPTW'].apply(
            lambda x: '{:.2f}'.format(x))
        df_final['med-HR_IPTW'] = df_final['med-HR_IPTW'].apply(
            lambda x: '{:.2f}'.format(x))

        df_final['mean_ci-HR_IPTW'] = df_final['mean_ci-HR_IPTW'].apply(
            lambda x: stringlist_2_str(x, False, 2))

        df_final['med-HR_ori_CI'] = df_final['med-HR_ori_CI'].apply(lambda x: stringlist_2_str(x, False, 2))
        df_final['med-HR_IPTW_CI'] = df_final['med-HR_IPTW_CI'].apply(lambda x: stringlist_2_str(x, False, 2))

        df_final.to_excel(writer, sheet_name=sheet)

    writer.save()
    print('Done results_ATE_for_ml_step3_finalInfo')


def combine_ate_final_LR_with(cohort_dir_name, model):
    dirname = r'output/{}/LR/'.format(cohort_dir_name)
    df_lr = pd.read_excel(dirname + 'results/summarized_IPTW_ATE_{}_finalInfo-allPvalue.xlsx'.format('LR'),
                          sheet_name=None,
                          dtype=str)
    df_other = pd.read_excel(r'output/{}/{}/'.format(cohort_dir_name, model) +
                             'results/summarized_IPTW_ATE_{}_finalInfo-allPvalue.xlsx'.format(model), sheet_name=None,
                             dtype=str)
    writer = pd.ExcelWriter(dirname + 'results/summarized_IPTW_ATE_LR_finalInfo_cat_{}-allPvalue.xlsx'.format(model),
                            engine='xlsxwriter')
    writer2 = pd.ExcelWriter(
        dirname + 'results/summarized_IPTW_ATE_LR_finalInfo_outerjoin_{}-allPvalue.xlsx'.format(model),
        engine='xlsxwriter')

    col_name = ['drug', 'Drug', 'Model', 'niters', 'Support', 'Treat', 'Ctrl',
                'n_feature', ' Unbalanced', 'Unbalanced IPTW', 'KM', 'HR']

    def significance(val):
        if val < 0.001:
            return '***'
        elif val < 0.01:
            return '**'
        elif val < 0.05:
            return '*'
        else:
            return 'ns'

    def return_select_content(key, row, null_model=''):
        data = [key, ]
        col1 = ['drug_name', 'Model', 'niters', 'support', 'n_treat', 'n_ctrl',
                'n_feature', 'mean-n_unbalanced_feature', 'mean-n_unbalanced_feature_IPTW']
        if null_model:
            for c in col1:
                data.append(row[c])
            data[2] = null_model.upper()
            data[4] = 0
            data[5] = data[6] = data[7] = data[8] = data[9] = np.nan
            data.append(np.nan)
            data.append(np.nan)
        else:
            for c in col1:
                data.append(row[c])
            data.append(row['mean-KM1-0_IPTW'] + ' (' + row['mean_ci-KM1-0_IPTW'] + ')' + '$^{' + significance(
                float(row['pvalue-KM1-0_IPTW'])) + '}$')
            data.append(row['mean-HR_IPTW'] + ' (' + row['mean_ci-HR_IPTW'] + ')' + '$^{' + significance(
                float(row['pvalue-HR_IPTW'])) + '}$')
        return data

    for sheet in ['random', 'atc', 'all']:
        df1 = df_lr[sheet]
        df1['Model'] = 'lr'
        df2 = df_other[sheet]
        df2['Model'] = 'lstm'

        df_outer = df1.join(df2.set_index('drug'), lsuffix='', rsuffix='_{}'.format(model), on='drug', how='outer')
        df_outer.to_excel(writer2, sheet_name=sheet)

        df1 = df1.set_index('drug')
        df2 = df2.set_index('drug')
        data = []
        for key, row in df1.iterrows():
            data.append(return_select_content(key, row))
            if key in df2.index:
                data.append(return_select_content(key, df2.loc[key, :]))
            else:
                data.append(return_select_content(key, row, null_model=model))

        df_final = pd.DataFrame(data=data, columns=col_name)
        df_final.to_excel(writer, sheet_name=sheet)
    writer.save()
    writer2.save()
    print('Done results_ATE_for_ml_step3_finalInfo')


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

    c1 = 'success_rate-val_auc-i-all_n_unbalanced_feat_iptw'
    # c2 = 'success_rate-train_loss-i-all_n_unbalanced_feat_iptw'
    c2 = 'success_rate-val_loss-i-all_n_unbalanced_feat_iptw'
    c3 = 'success_rate-trainval_n_unbalanced_feat_iptw-val_auc-all_n_unbalanced_feat_iptw'

    idx_auc = dfall[c1] >= 0.1
    idx_smd = dfall[c2] >= 0.1
    idx = dfall[c3] >= 0.1

    print('Total drug trials: ', len(idx))
    print(r"#df[{}] > 0: ".format(c1), idx_auc.sum(), '({:.2f}%)'.format(idx_auc.mean() * 100))
    print(r"#df[{}] > 0: ".format(c2), idx_smd.sum(), '({:.2f}%)'.format(idx_smd.mean() * 100))
    print(r"#df[{}] > 0: ".format(c3), idx.sum(), '({:.2f}%)'.format(idx.mean() * 100))

    df = dfall.loc[idx, :].sort_values(by=[c3], ascending=[False])
    # df['nsmd_mean_ci-val_auc_nsmd']

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

    # add source data 2023-11-2
    df_source_data = pd.DataFrame({'drug_name': xlabels,
                                   'Success Rate by Validation AUC Select -Method 1': top_1,
                                   'Success Rate by Validation Loss Select - Method 2': top_2,
                                   'Success Rate by Our Selection Strategy': top_3,
                                   r"P-value Welch's t-test between Method 1 and Our Strategy": df.loc[:, "p-succes-final-vs-1"],
                                   r"P-value Welch's t-test between Method 2 and Our Strategy": df.loc[:, "p-succes-final-vs-2"],
                                   r"P-value Welch's t-test between Method 1 and Method 2": df.loc[:, "p-succes-1-vs-2"]
                                   })
    check_and_mkdir(dirname + 'results/fig/')
    df_source_data.to_csv(dirname + 'results/fig/balance_rate_barplot-{}-{}-SourceData.csv'.format(model, contrl_type))
    # return df_source_data
    #

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
    ax.set_xlabel("Drug Trials", fontsize=25)
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
              fontsize=25)  # , bbox_to_anchor=(1.13, 1.01))

    # ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_xmargin(0.01)
    plt.tight_layout()
    if dump:
        check_and_mkdir(dirname + 'results/fig/')
        fig.savefig(dirname + 'results/fig/balance_rate_barplot-{}-{}.png'.format(model, contrl_type))
        fig.savefig(dirname + 'results/fig/balance_rate_barplot-{}-{}.pdf'.format(model, contrl_type))
    plt.show()
    plt.clf()
    return df_source_data


def bar_plot_model_selectionV2(cohort_dir_name, model, contrl_type='random', dump=True, colorful=True):
    dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
    dfall = pd.read_excel(dirname + 'results/summarized_model_selection_{}-More.xlsx'.format(model),
                          sheet_name=contrl_type)
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
    col = ['success_rate-val_auc_nsmd', 'success_rate-val_maxsmd_nsmd', 'success_rate-val_nsmd_nsmd',
           'success_rate-train_maxsmd_nsmd', 'success_rate-train_nsmd_nsmd',
           'success_rate-trainval_maxsmd_nsmd', 'success_rate-trainval_nsmd_nsmd',
           'success_rate-trainval_final_finalnsmd']
    legs = ['_'.join(x.split('-')[1].split('_')[:-1]) for x in col]
    # col_ci = [x.replace('rate', 'rate_ci') for x in col]
    top = []
    top_ci = []
    for c in col:
        top.append(df.loc[:, c])
        top_ci.append(np.array(df.loc[:, c.replace('rate', 'rate_ci')].apply(lambda x: stringlist_2_list(x)).to_list()))

    pauc = np.array(df.loc[:, "p-succes-final-vs-auc"])
    psmd = np.array(df.loc[:, "p-succes-final-vs-maxsmd"])
    paucsmd = np.array(df.loc[:, "p-succes-auc-vs-maxsmd"])

    xlabels = df.loc[:, 'drug_name']

    width = 0.45  # the width of the bars
    ind = np.arange(N) * width * (len(col) + 1)  # the x locations for the groups

    colors = ['#FAC200', '#82A2D3', '#F65453']
    fig, ax = plt.subplots(figsize=(24, 8))
    error_kw = {'capsize': 3, 'capthick': 1, 'ecolor': 'black'}
    # plt.ylim([0, 1.05])
    rects = []
    for i in range(len(top)):
        top_1 = top[i]
        top_1_ci = top_ci[i]
        if i <= 1 or i == len(top) - 1:
            rect = ax.bar(ind + width * i, top_1, width, yerr=[top_1 - top_1_ci[:, 0], top_1_ci[:, 1] - top_1],
                          error_kw=error_kw,
                          color=colors[min(i, len(colors) - 1)], edgecolor=None)  # "black")
        else:
            rect = ax.bar(ind + width * i, top_1, width, yerr=[top_1 - top_1_ci[:, 0], top_1_ci[:, 1] - top_1],
                          error_kw=error_kw,
                          edgecolor="black")
        rects.append(rect)

    ax.set_xticks(ind + int(len(top) / 2) * width)
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
    ax.set_ylabel("Prop. of success balancing", fontsize=25)  # Success Rate of Balancing

    def significance(val):
        if val < 0.001:
            return '***'
        elif val < 0.01:
            return '**'
        elif val < 0.05:
            return '*'
        else:
            return 'ns'

    def labelvalue(rects, val, height=None):
        for i, rect in enumerate(rects):
            if height is None:
                h = rect.get_height() * 1.02
            else:
                h = height[i] * 1.02
            ax.text(rect.get_x() + rect.get_width() / 2., h,
                    significance(val[i]),
                    ha='center', va='bottom', fontsize=11)

    for i in range(len(rects) - 1):
        pv = np.array(df.loc[:, "p-succes-fvs-" + col[i].split('-')[-1]])
        labelvalue((rects[i]), pv, top_ci[i][:, 1])

    # labelvalue(rects1, pauc, top_1_ci[:,1])
    # labelvalue(rects2, psmd, top_2_ci[:,1])

    # for i, rect in enumerate(rects3):
    #     d = 0.02
    #     y = top_3_ci[i, 1] * 1.03  # rect.get_height()
    #     w = rect.get_width()
    #     x = rect.get_x()
    #     x1 = x - 2 * w
    #     x2 = x - 1 * w
    #
    #     y1 = top_1_ci[i, 1] * 1.03
    #     y2 = top_2_ci[i, 1] * 1.03
    #
    #     # auc v.s. final
    #     l, r = x1, x + w
    #     ax.plot([l, l, (l+r) / 2], [y + 2 * d, y + 3 * d, y + 3 * d], lw=1.2, c=colors[0] if colorful else 'black')
    #     ax.plot([(l+r) / 2, r, r], [y + 3 * d, y + 3 * d, y + 2 * d], lw=1.2, c=colors[2] if colorful else 'black')
    #     # ax.plot([x1, x1, x, x], [y+2*d, y+3*d, y+3*d, y+2*d], c='#FAC200') #c="black")
    #     ax.text((l+r) / 2, y + 2.6 * d, significance(pauc[i]), ha='center', va='bottom', fontsize=13)
    #
    #     # smd v.s. final
    #     l, r = x2 + 0.6*w, x + w
    #     ax.plot([l, l, (l + r) / 2], [y, y + d, y + d], lw=1.2, c=colors[1] if colorful else 'black')
    #     ax.plot([(l + r) / 2, r, r], [y + d, y + d, y], lw=1.2, c=colors[2] if colorful else 'black')
    #     # ax.plot([x2, x2, x, x], [y, y + d, y + d, y], c='#82A2D3') #c="black")
    #     ax.text((l + r) / 2, y + 0.6 * d, significance(psmd[i]), ha='center', va='bottom', fontsize=13)
    #
    #     # auc v.s. smd
    #     l, r = x1, x2 + 0.4*w
    #     ax.plot([l, l, (l + r) / 2], [y, y + d, y + d], lw=1.2, c=colors[0] if colorful else 'black')
    #     ax.plot([(l + r) / 2, r, r], [y + d, y + d, y], lw=1.2, c=colors[1] if colorful else 'black')
    #     # ax.plot([x1, x1, x, x], [y+2*d, y+3*d, y+3*d, y+2*d], c='#FAC200') #c="black")
    #     ax.text((l + r) / 2, y + .6 * d, significance(paucsmd[i]), ha='center', va='bottom', fontsize=13)

    # ax.set_title('Success Rate of Balancing by Different PS Model Selection Methods')
    ax.legend((rect[0] for rect in rects), (x for x in legs),
              fontsize=18)  # , bbox_to_anchor=(1.13, 1.01))

    # ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_xmargin(0.01)
    plt.tight_layout()
    if dump:
        check_and_mkdir(dirname + 'results/fig/')
        fig.savefig(dirname + 'results/fig/balance_rate_barplot-{}-{}-all.png'.format(model, contrl_type))
        fig.savefig(dirname + 'results/fig/balance_rate_barplot-{}-{}-all.pdf'.format(model, contrl_type))
    plt.show()
    plt.clf()


def bar_plot_model_selectionV2_test(cohort_dir_name, model, contrl_type='random', dump=True, colorful=True):
    dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
    dfall = pd.read_excel(dirname + 'results/summarized_model_selection_{}-More.xlsx'.format(model),
                          sheet_name=contrl_type)
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
    col = ['success_rate-val_auc_nsmd']  # , 'success_rate-val_maxsmd_nsmd']
    # , 'success_rate-val_nsmd_nsmd',
    #    'success_rate-train_maxsmd_nsmd', 'success_rate-train_nsmd_nsmd',
    #    'success_rate-trainval_maxsmd_nsmd', 'success_rate-trainval_nsmd_nsmd',
    #    'success_rate-trainval_final_finalnsmd']
    legs = ['_'.join(x.split('-')[1].split('_')[:-1]) for x in col]
    # col_ci = [x.replace('rate', 'rate_ci') for x in col]
    top = []
    top_ci = []
    for c in col:
        top.append(df.loc[:, c])
        top_ci.append(np.array(df.loc[:, c.replace('rate', 'rate_ci')].apply(lambda x: stringlist_2_list(x)).to_list()))

    pauc = np.array(df.loc[:, "p-succes-final-vs-auc"])
    psmd = np.array(df.loc[:, "p-succes-final-vs-maxsmd"])
    paucsmd = np.array(df.loc[:, "p-succes-auc-vs-maxsmd"])

    xlabels = df.loc[:, 'drug_name']

    width = 0.45  # the width of the bars
    ind = np.arange(N) * width * (len(col) + 1)  # the x locations for the groups

    colors = ['#FAC200', '#82A2D3', '#F65453']
    fig, ax = plt.subplots(figsize=(24, 8))
    error_kw = {'capsize': 3, 'capthick': 1, 'ecolor': 'black'}
    plt.ylim([0, 1.1])
    rects = []
    for i in range(len(top)):
        top_1 = top[i]
        top_1_ci = top_ci[i]
        if i <= 1 or i == len(top) - 1:
            rect = ax.bar(ind + width * i, top_1, width, yerr=[top_1 - top_1_ci[:, 0], top_1_ci[:, 1] - top_1],
                          error_kw=error_kw,
                          color=colors[min(i, len(colors) - 1)], edgecolor=None)  # "black")
        else:
            rect = ax.bar(ind + width * i, top_1, width, yerr=[top_1 - top_1_ci[:, 0], top_1_ci[:, 1] - top_1],
                          error_kw=error_kw,
                          edgecolor="black")
        rects.append(rect)

    ax.set_xticks(ind + int(len(top) / 2) * width)
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
    ax.set_ylabel("Prop. of success balancing", fontsize=25)  # Success Rate of Balancing

    def significance(val):
        if val < 0.001:
            return '***'
        elif val < 0.01:
            return '**'
        elif val < 0.05:
            return '*'
        else:
            return 'ns'

    def labelvalue(rects, val, height=None):
        for i, rect in enumerate(rects):
            if height is None:
                h = rect.get_height() * 1.02
            else:
                h = height[i] * 1.02
            ax.text(rect.get_x() + rect.get_width() / 2., h,
                    significance(val[i]),
                    ha='center', va='bottom', fontsize=11)

    for i in range(len(rects) - 1):
        pv = np.array(df.loc[:, "p-succes-fvs-" + col[i].split('-')[-1]])
        labelvalue((rects[i]), pv, top_ci[i][:, 1])

    # labelvalue(rects1, pauc, top_1_ci[:,1])
    # labelvalue(rects2, psmd, top_2_ci[:,1])

    # for i, rect in enumerate(rects3):
    #     d = 0.02
    #     y = top_3_ci[i, 1] * 1.03  # rect.get_height()
    #     w = rect.get_width()
    #     x = rect.get_x()
    #     x1 = x - 2 * w
    #     x2 = x - 1 * w
    #
    #     y1 = top_1_ci[i, 1] * 1.03
    #     y2 = top_2_ci[i, 1] * 1.03
    #
    #     # auc v.s. final
    #     l, r = x1, x + w
    #     ax.plot([l, l, (l+r) / 2], [y + 2 * d, y + 3 * d, y + 3 * d], lw=1.2, c=colors[0] if colorful else 'black')
    #     ax.plot([(l+r) / 2, r, r], [y + 3 * d, y + 3 * d, y + 2 * d], lw=1.2, c=colors[2] if colorful else 'black')
    #     # ax.plot([x1, x1, x, x], [y+2*d, y+3*d, y+3*d, y+2*d], c='#FAC200') #c="black")
    #     ax.text((l+r) / 2, y + 2.6 * d, significance(pauc[i]), ha='center', va='bottom', fontsize=13)
    #
    #     # smd v.s. final
    #     l, r = x2 + 0.6*w, x + w
    #     ax.plot([l, l, (l + r) / 2], [y, y + d, y + d], lw=1.2, c=colors[1] if colorful else 'black')
    #     ax.plot([(l + r) / 2, r, r], [y + d, y + d, y], lw=1.2, c=colors[2] if colorful else 'black')
    #     # ax.plot([x2, x2, x, x], [y, y + d, y + d, y], c='#82A2D3') #c="black")
    #     ax.text((l + r) / 2, y + 0.6 * d, significance(psmd[i]), ha='center', va='bottom', fontsize=13)
    #
    #     # auc v.s. smd
    #     l, r = x1, x2 + 0.4*w
    #     ax.plot([l, l, (l + r) / 2], [y, y + d, y + d], lw=1.2, c=colors[0] if colorful else 'black')
    #     ax.plot([(l + r) / 2, r, r], [y + d, y + d, y], lw=1.2, c=colors[1] if colorful else 'black')
    #     # ax.plot([x1, x1, x, x], [y+2*d, y+3*d, y+3*d, y+2*d], c='#FAC200') #c="black")
    #     ax.text((l + r) / 2, y + .6 * d, significance(paucsmd[i]), ha='center', va='bottom', fontsize=13)

    # ax.set_title('Success Rate of Balancing by Different PS Model Selection Methods')
    ax.legend((rect[0] for rect in rects), (x for x in legs),
              fontsize=18)  # , bbox_to_anchor=(1.13, 1.01))

    # ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_xmargin(0.01)
    plt.tight_layout()
    if dump:
        check_and_mkdir(dirname + 'results/fig/')
        fig.savefig(dirname + 'results/fig/balance_rate_barplot-{}-{}-auc.png'.format(model, contrl_type))
        fig.savefig(dirname + 'results/fig/balance_rate_barplot-{}-{}-auc.pdf'.format(model, contrl_type))
    plt.show()
    plt.clf()


def arrow_plot_model_selection_unbalance_reduction(cohort_dir_name, model, contrl_type='random', dump=True,
                                                   colorful=True, datapart='all', log=False):
    # dataset: train, test, all
    dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
    dfall = pd.read_excel(dirname + 'results/summarized_model_selection_{}.xlsx'.format(model),
                          sheet_name=contrl_type, converters={'drug': str})
    c1 = 'val_auc-i'
    # c2 = 'train_loss-i'
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

    df = dfall.loc[idx, :].sort_values(by=['success_rate-' + c30 + '-all_n_unbalanced_feat_iptw'], ascending=[False])
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

            data[ith].append([drugname, before_mean, after_mean, change_mean])


    data_df = []
    appendix = [c1, c2, c3]
    # for d in data:
    for igroup, d in enumerate(data):
        df = pd.DataFrame(d, columns=['subject', 'before', 'after', 'change'], index=range(len(d)))
        data_df.append(df)
        check_and_mkdir(dirname + 'results/fig/')
        df.to_csv(dirname + 'results/fig/arrow_nsmd_reduce-{}-{}-{}{}-{}.csv'.format(model, contrl_type, datapart,
                                                                                     '-log' if log else '',
                                                                                     appendix[igroup]))

    # return data_df

    fig = plt.figure(figsize=(5, 8))
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
    # else:
    #     ax.axvline(x=0, color='0.9', ls='--', lw=2, zorder=0)  # add line at x=0
    ax.axvline(x=0, color='0.9', ls='--', lw=2, zorder=0)  # add line at x=0
    if log:
        plt.xscale("log")
    ax.set_xlabel('No. of unbalanced features')  # label the x axis
    ax.set_ylabel('Drug Trials')  # label the y axis
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
    return data_df

def box_plot_model_selection(cohort_dir_name, model, contrl_type='random', dump=True, colorful=True):
    dirname = r'output/revise_testset/{}/{}/'.format(cohort_dir_name, model)
    dfall = pd.read_excel(dirname + 'results/summarized_model_selection_{}.xlsx'.format(model),
                          sheet_name=contrl_type, converters={'drug': str})
    c1 = 'success_rate-val_auc-i-all_n_unbalanced_feat_iptw'
    c2 = 'success_rate-train_loss-i-all_n_unbalanced_feat_iptw'
    c3 = 'success_rate-trainval_n_unbalanced_feat_iptw-val_auc-all_n_unbalanced_feat_iptw'

    idx_auc = dfall[c1] >= 0.1
    idx_smd = dfall[c2] >= 0.1
    idx = dfall[c3] >= 0.1

    print('Total drug trials: ', len(idx))
    print(r"#df[{}] > 0: ".format(c1), idx_auc.sum(), '({:.2f}%)'.format(idx_auc.mean() * 100))
    print(r"#df[{}] > 0: ".format(c2), idx_smd.sum(), '({:.2f}%)'.format(idx_smd.mean() * 100))
    print(r"#df[{}] > 0: ".format(c3), idx.sum(), '({:.2f}%)'.format(idx.mean() * 100))

    df = dfall.loc[idx, :].sort_values(by=[c3], ascending=[False])
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
        data_1.append(
            np.array(rdf.loc[idx, c1.replace("success_rate-", '').replace("all_n_unbalanced_feat_iptw", "test_auc")]))
        data_2.append(
            np.array(rdf.loc[idx, c2.replace("success_rate-", '').replace("all_n_unbalanced_feat_iptw", "test_auc")]))
        data_3.append(
            np.array(rdf.loc[idx, c3.replace("success_rate-", '').replace("all_n_unbalanced_feat_iptw", "test_auc")]))
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
              'showmeans': True,  # 'meanline':True,
              "meanprops": dict(linestyle='--', linewidth=1, markeredgecolor='purple', marker='^',
                                markerfacecolor="None")}
    rects1 = plt.boxplot(data_1, positions=ind - 0.08, **box_kw)
    rects2 = plt.boxplot(data_2, positions=ind + width, **box_kw)
    rects3 = plt.boxplot(data_3, positions=ind + 2 * width + 0.08, **box_kw)

    def plot_strip(ind, data, color):
        w = width - 0.15
        swarm1 = pd.DataFrame([(ind[i], data[i][j]) for i in range(len(ind)) for j in range(len(data[i]))],
                              columns=['x', 'y'])
        strip_rx = stats.uniform(-w / 2., w).rvs(len(swarm1))
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
        if val < 0.001:
            return '***'
        elif val < 0.01:
            return '**'
        elif val < 0.05:
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
        l, r = x - 0.5 * width, x2 - 0.08
        ax.plot([x1, x1, (x2 + x1) / 2], [y, y + d, y + d], lw=1.2, c=colors[0] if colorful else 'black')
        ax.plot([(x2 + x1) / 2, x2 - 0.08, x2 - 0.08], [y + d, y + d, y], lw=1.2, c=colors[1] if colorful else 'black')
        ax.text((x2 + x1) / 2, y + d, significance(p_v[i, 2]), ha='center', va='bottom', fontsize=12)

        # auc v.s. final
        ax.plot([x1, x1, (x + x1) / 2], [y + 2 * d, y + 3 * d, y + 3 * d], lw=1.2, c=colors[0] if colorful else 'black')
        ax.plot([(x + x1) / 2, x, x], [y + 3 * d, y + 3 * d, y + 2 * d], lw=1.2, c=colors[2] if colorful else 'black')
        # ax.plot([x1, x1, x, x], [y+2*d, y+3*d, y+3*d, y+2*d], c="black")
        ax.text((x + x1) / 2, y + 3 * d, significance(p_v[i, 0]), ha='center', va='bottom', fontsize=12)

        # smd v.s. final
        ax.plot([x2 + 0.08, x2 + 0.08, (x + x2) / 2], [y, y + d, y + d], lw=1.2, c=colors[1] if colorful else 'black')
        ax.plot([(x + x2) / 2, x, x], [y + d, y + d, y], lw=1.2, c=colors[2] if colorful else 'black')
        # ax.plot([x2, x2, x, x], [y, y + d, y + d, y], c="black")
        ax.text((x + x2) / 2, y + 1 * d, significance(p_v[i, 1]), ha='center', va='bottom', fontsize=12)

    ax.legend((rects1["boxes"][0], rects2["boxes"][0], rects3["boxes"][0]),
              ('Val-AUC Select', 'Train-Loss Select', 'Our Strategy'),
              fontsize=20, loc='lower left')
    ax.set_xmargin(0.01)
    plt.tight_layout()
    if dump:
        check_and_mkdir(dirname + 'results/fig/')
        fig.savefig(dirname + 'results/fig/test_auc_boxplot-{}-{}.png'.format(model, contrl_type))
        fig.savefig(dirname + 'results/fig/test_auc_boxplot-{}-{}.pdf'.format(model, contrl_type))
    plt.show()
    plt.clf()


def box_plot_model_selectionV2(cohort_dir_name, model, contrl_type='random', dump=True, colorful=True):
    dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
    dfall = pd.read_excel(dirname + 'results/summarized_model_selection_{}-More.xlsx'.format(model),
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
    col = ['val_auc_testauc', 'val_maxsmd_testauc', 'val_nsmd_testauc',
           'train_maxsmd_testauc', 'train_nsmd_testauc',
           'trainval_maxsmd_testauc', 'trainval_nsmd_testauc',
           'trainval_final_testnauc']
    legs = ['_'.join(x.split('_')[:-1]) for x in col]
    data_list = [[] for i in range(len(col))]
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
        for i in range(len(col)):
            data_list[i].append(np.array(rdf.loc[idx, col[i]]))

        p_v = []
        for i in range(1, len(col)):
            a = data_list[0][-1]
            b = data_list[i][-1]
            p, test_orig = bootstrap_mean_pvalue_2samples(a, b)
            p_v.append(test_orig[1])
        data_pvalue.append(p_v)

    colors = ['#FAC200', '#82A2D3', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#F65453']
    # color_others = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig, ax = plt.subplots(figsize=(24, 8))
    width = 0.5  # the width of the bars
    ind = np.arange(N) * width * (len(col) + 1)  # the x locations for the groups
    sym = 'o'
    # 'meanline':True,
    box_kw = {"sym": sym, "widths": width - 0.08, "patch_artist": True, "notch": True,
              'showmeans': True,  # 'meanline':True,
              "meanprops": dict(linestyle='--', linewidth=1, markeredgecolor='purple', marker='^',
                                markerfacecolor="None")}
    rects = []
    for i, data in enumerate(data_list):
        rect = plt.boxplot(data, positions=ind + i * width, **box_kw)
        rects.append(rect)

    def plot_strip(ind, data, color=None):
        w = width - 0.15
        swarm1 = pd.DataFrame([(ind[i], data[i][j]) for i in range(len(ind)) for j in range(len(data[i]))],
                              columns=['x', 'y'])
        strip_rx = stats.uniform(-w / 2., w).rvs(len(swarm1))
        # sns.stripplot(x='x', y='y', data=swarm1, color=".25", alpha=0.2, ax=ax)
        if color is None:
            plt.scatter(swarm1['x'] + strip_rx, swarm1['y'], alpha=0.2)
        else:
            plt.scatter(swarm1['x'] + strip_rx, swarm1['y'], alpha=0.2, c=color)

    # ticks = list(drug_name_list)
    for i, bplot in enumerate(rects):
        for patch in bplot['boxes']:
            patch.set_facecolor(colors[i])
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.7))
        # plt.setp(bplot['boxes'], color=color)
        # plt.setp(bplot['whiskers'], color=color)
        # plt.setp(bplot['caps'], color=color)
        plt.setp(bplot['medians'], color='black')

    for i, data in enumerate(data_list):
        plot_strip(ind + i * width, data, colors[i])

    # plt.ylim([0.5, 0.85])
    ax.set_xticks(ind + int(len(col) / 2) * width)
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
        if val < 0.001:
            return '***'
        elif val < 0.01:
            return '**'
        elif val < 0.05:
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
        for j in range(1, len(col)):
            y = data_list[j][i].max() * 1.01
            x = ind[i] + j * width
            ax.text(x, y + d, significance(p_v[i, j - 1]), ha='center', va='bottom', fontsize=9)
        # y = np.max([data_1[i].max(), data_2[i].max(), data_3[i].max()]) * 1.01  # rect.get_height()
        # x = ind[i] + 2 * width + 0.08  # + width/2
        # x1 = ind[i] - 0.08  # - width/2
        # x2 = ind[i] + width  # - width/2
        #
        # # auc v.s. smd
        # l, r = x - 0.5*width, x2 - 0.08
        # ax.plot([x1, x1, (x2 + x1) / 2], [y, y + d, y + d], lw=1.2, c=colors[0] if colorful else 'black')
        # ax.plot([(x2 + x1) / 2, x2 - 0.08, x2-0.08], [y + d, y + d, y], lw=1.2, c=colors[1] if colorful else 'black')
        # ax.text((x2 + x1) / 2, y + d, significance(p_v[i, 2]), ha='center', va='bottom', fontsize=12)
        #
        # # auc v.s. final
        # ax.plot([x1, x1, (x + x1) / 2], [y + 2 * d, y + 3 * d, y + 3 * d], lw=1.2, c=colors[0] if colorful else 'black')
        # ax.plot([(x + x1) / 2, x, x], [y + 3 * d, y + 3 * d, y + 2 * d], lw=1.2, c=colors[2] if colorful else 'black')
        # # ax.plot([x1, x1, x, x], [y+2*d, y+3*d, y+3*d, y+2*d], c="black")
        # ax.text((x + x1) / 2, y + 3 * d, significance(p_v[i, 0]), ha='center', va='bottom', fontsize=12)
        #
        # # smd v.s. final
        # ax.plot([x2+0.08, x2+0.08, (x + x2) / 2], [y, y + d, y + d], lw=1.2, c=colors[1] if colorful else 'black')
        # ax.plot([(x + x2) / 2, x, x], [y + d, y + d, y], lw=1.2, c=colors[2] if colorful else 'black')
        # # ax.plot([x2, x2, x, x], [y, y + d, y + d, y], c="black")
        # ax.text((x + x2) / 2, y + 1 * d, significance(p_v[i, 1]), ha='center', va='bottom', fontsize=12)

    ax.legend((rect["boxes"][0] for rect in rects),
              (x for x in legs),
              fontsize=15)
    ax.set_xmargin(0.01)
    plt.tight_layout()
    if dump:
        check_and_mkdir(dirname + 'results/fig/')
        fig.savefig(dirname + 'results/fig/test_auc_boxplot-{}-{}-all.png'.format(model, contrl_type))
        fig.savefig(dirname + 'results/fig/test_auc_boxplot-{}-{}-all.pdf'.format(model, contrl_type))
    plt.show()
    plt.clf()


def box_plot_ate(cohort_dir_name, model, model2='LSTM', contrl_type='random', dump=True, colorful=True):
    dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
    dirname2 = r'output/{}/{}/'.format(cohort_dir_name, model2)
    df_all = pd.read_excel(dirname + 'results/summarized_IPTW_ATE_{}.xlsx'.format(model),
                           dtype={'drug': str},
                           sheet_name=None)

    df_all2 = pd.read_excel(dirname2 + 'results/summarized_IPTW_ATE_{}.xlsx'.format(model2),
                            dtype={'drug': str},
                            sheet_name=None)

    df = df_all[contrl_type]
    # Only select drugs with selection criteria trial
    # 1. minimum support set 10, may choose 20 later
    # 2. p value < 0.05
    idx = (df['support'] >= 50) & (df['pvalue-KM1-0_IPTW'] <= 0.05)
    df_sort = df.loc[idx, :].sort_values(by=['mean-KM1-0_IPTW'], ascending=[False])

    df2 = df_all2[contrl_type]
    idx2 = (df2['support'] >= 50) & (df2['pvalue-KM1-0_IPTW'] <= 0.05)
    df_sort2 = df2.loc[idx2, :].sort_values(by=['mean-KM1-0_IPTW'], ascending=[False])

    data_1 = []
    data_2 = []
    data_pvalue = []
    drug_list = df_sort['drug'].tolist()
    drug_name_list = df_sort['drug_name'].tolist()
    drug_list2 = df_sort2['drug'].tolist()
    print('len(drug_list):', len(drug_list), 'len(drug_list2)', len(drug_list2))
    N = len(drug_list)
    n_box = 2
    for drug in drug_list:
        rdf = pd.read_excel(dirname + 'results/' + drug + '_results.xlsx')

        if contrl_type != 'all':
            idx_all = (rdf['ctrl_type'] == contrl_type)
        else:
            idx_all = (rdf['ctrl_type'].notna())

        # Only select balanced trial
        idx = idx_all & (rdf['n_unbalanced_feature_IPTW'] <= MAX_NO_UNBALANCED_FEATURE)
        c = "KM1-0_IPTW"  # "ATE_original", "ATE_IPTW", "KM1-0_original", "KM1-0_IPTW", 'HR_ori', 'HR_IPTW'
        nv = rdf.loc[idx, c]
        data_1.append(np.array(rdf.loc[idx, c]) * 100)

        rdf2 = pd.read_excel(dirname2 + 'results/' + drug + '_results.xlsx')
        if contrl_type != 'all':
            idx_all2 = (rdf['ctrl_type'] == contrl_type)
        else:
            idx_all2 = (rdf['ctrl_type'].notna())
        # Only select balanced trial
        idx2 = idx_all2 & (rdf2['n_unbalanced_feature_IPTW'] <= MAX_NO_UNBALANCED_FEATURE)
        nv2 = rdf.loc[idx2, c]
        if drug in drug_list2:
            data_2.append(np.array(rdf2.loc[idx2, c]) * 100)
        else:
            data_2.append(np.array([]))

        # if len(nv) > 0:
        #     med = IQR(nv)[0]
        #     iqr = IQR(nv)[1:]
        #
        #     mean = nv.mean()
        #     mean_ci, _ = bootstrap_mean_ci(nv, alpha=0.05)
        #
        #     if 'HR' in c:
        #         p, _ = bootstrap_mean_pvalue(nv, expected_mean=1)
        #     else:
        #         p, _ = bootstrap_mean_pvalue(nv, expected_mean=0)
        #
        #     r.extend([med, iqr, mean, mean_ci, p])
        # else:
        #     r.extend([np.nan, np.nan, np.nan, np.nan, np.nan])

    colors = ['#F65453', '#82A2D3', '#FAC200']
    fig, ax = plt.subplots(figsize=(12, 8))  # 18
    width = 0.35  # 0.5 #the width of the bars
    ind = np.arange(N) * width * (n_box + 1)  # the x locations for the groups
    sym = 'o'
    # 'meanline':True,
    box_kw = {"sym": sym, "widths": width - 0.04, "patch_artist": True, "notch": True,
              'showmeans': True,  # 'meanline':True,
              "meanprops": dict(linestyle='--', linewidth=1, markeredgecolor='black', marker='^',
                                markerfacecolor="None")}
    rects1 = plt.boxplot(data_1, positions=ind, **box_kw)
    rects2 = plt.boxplot(data_2, positions=ind + width, **box_kw)

    # rects3 = plt.boxplot(data_3, positions=ind + 2 * width + 0.08, **box_kw)

    def plot_strip(ind, data, color):
        w = width - 0.15
        swarm1 = pd.DataFrame([(ind[i], data[i][j]) for i in range(len(ind)) for j in range(len(data[i]))],
                              columns=['x', 'y'])
        strip_rx = stats.uniform(-w / 2., w).rvs(len(swarm1))
        # sns.stripplot(x='x', y='y', data=swarm1, color=".25", alpha=0.2, ax=ax)
        plt.scatter(swarm1['x'] + strip_rx, swarm1['y'], alpha=.6, c=color)
        # plt.scatter(swarm1['x'] + strip_rx, swarm1['y'], alpha=1, facecolors='none', edgecolors=color)  # c=color) #,

    # ticks = list(drug_name_list)
    for i, bplot in enumerate([rects1, rects2]):
        for patch in bplot['boxes']:
            patch.set_facecolor(colors[i])
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .6))
        # plt.setp(bplot['boxes'], color=color)
        # plt.setp(bplot['whiskers'], color=color)
        # plt.setp(bplot['caps'], color=color)
        plt.setp(bplot['medians'], color='black')

    plot_strip(ind, data_1, colors[0])
    plot_strip(ind + width, data_2, colors[1])
    # plot_strip(ind + 2 * width + 0.08, data_3, colors[2])

    # plt.ylim([0.5, 0.85])
    ax.set_xticks(ind + width / 2)
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
    ax.set_ylabel("Adjusted survival difference %", fontsize=25)
    plt.axhline(y=0., color='black', linestyle='--')

    def significance(val):
        if val < 0.001:
            return '***'
        elif val < 0.01:
            return '**'
        elif val < 0.05:
            return '*'
        else:
            return 'ns'

    for i in range(N):
        n1 = len(data_1[i])
        n2 = len(data_2[i])
        y1 = max(data_1[i]) * 1.02
        try:
            y2 = max(data_2[i]) * 1.02
        except:
            y2 = y1
        x1 = ind[i]
        x2 = ind[i] + width
        ax.text(x1, y1, str(n1), ha='center', va='bottom', fontsize=14)
        ax.text(x2, y2, str(n2), ha='center', va='bottom', fontsize=14)

    # def labelvalue(rects, x, y, p):
    #     for i, rect in enumerate(rects):
    #         ax.text(x[i], y[i],
    #                 significance(p[i]),
    #                 ha='center', va='bottom', fontsize=11)
    #
    # labelvalue(rects1["boxes"], ind - 0.08, np.max(data_1, axis=1)*1.01, np.array(data_pvalue)[:,0])
    # labelvalue(rects2["boxes"], ind + width, np.max(data_2, axis=1)*1.01, np.array(data_pvalue)[:,1])

    # p_v = np.array(data_pvalue)
    # for i in range(N):
    #     d = 0.008
    #     y = np.max([data_1[i].max(), data_2[i].max(), data_3[i].max()]) * 1.01  # rect.get_height()
    #     x = ind[i] + 2 * width + 0.08  # + width/2
    #     x1 = ind[i] - 0.08  # - width/2
    #     x2 = ind[i] + width  # - width/2
    #
    #     # auc v.s. smd
    #     l, r = x - 0.5 * width, x2 - 0.08
    #     ax.plot([x1, x1, (x2 + x1) / 2], [y, y + d, y + d], lw=1.2, c=colors[0] if colorful else 'black')
    #     ax.plot([(x2 + x1) / 2, x2 - 0.08, x2 - 0.08], [y + d, y + d, y], lw=1.2, c=colors[1] if colorful else 'black')
    #     ax.text((x2 + x1) / 2, y + d, significance(p_v[i, 2]), ha='center', va='bottom', fontsize=12)
    #
    #     # auc v.s. final
    #     ax.plot([x1, x1, (x + x1) / 2], [y + 2 * d, y + 3 * d, y + 3 * d], lw=1.2, c=colors[0] if colorful else 'black')
    #     ax.plot([(x + x1) / 2, x, x], [y + 3 * d, y + 3 * d, y + 2 * d], lw=1.2, c=colors[2] if colorful else 'black')
    #     # ax.plot([x1, x1, x, x], [y+2*d, y+3*d, y+3*d, y+2*d], c="black")
    #     ax.text((x + x1) / 2, y + 3 * d, significance(p_v[i, 0]), ha='center', va='bottom', fontsize=12)
    #
    #     # smd v.s. final
    #     ax.plot([x2 + 0.08, x2 + 0.08, (x + x2) / 2], [y, y + d, y + d], lw=1.2, c=colors[1] if colorful else 'black')
    #     ax.plot([(x + x2) / 2, x, x], [y + d, y + d, y], lw=1.2, c=colors[2] if colorful else 'black')
    #     # ax.plot([x2, x2, x, x], [y, y + d, y + d, y], c="black")
    #     ax.text((x + x2) / 2, y + 1 * d, significance(p_v[i, 1]), ha='center', va='bottom', fontsize=12)

    ax.legend((rects1["boxes"][0], rects2["boxes"][0]),
              (model, model2), fontsize=20)
    ax.set_xmargin(0.01)
    plt.tight_layout()
    if dump:
        check_and_mkdir(dirname + 'results/fig/')
        fig.savefig(
            dirname + 'results/fig/adjusted_survival_diff_boxplot-{}-{}-{}.png'.format(model, model2, contrl_type))
        fig.savefig(
            dirname + 'results/fig/adjusted_survival_diff_boxplot-{}-{}-{}.pdf'.format(model, model2, contrl_type))
    plt.show()
    plt.clf()


def box_plot_ate_V2(cohort_dir_name, models=['LR', 'LSTM', 'MLP', 'LIGHTGBM'], contrl_type='random', dump=True):
    dirname_list = []
    df_all_list = []
    data_list = []
    for model in models:
        dirname = r'output/{}/{}/'.format(cohort_dir_name, model)
        df_all = pd.read_excel(dirname + 'results/summarized_IPTW_ATE_{}.xlsx'.format(model),
                               dtype={'drug': str},
                               sheet_name=None)
        dirname_list.append(dirname)
        df_all_list.append(df_all)

        df = df_all[contrl_type]
        # Only select drugs with selection criteria trial
        # 1. minimum support set 10, may choose 20 later
        # 2. p value < 0.05
        idx = (df['support'] >= 50) & (df['pvalue-KM1-0_IPTW'] <= 0.05)
        df_sort = df.loc[idx, :].sort_values(by=['mean-KM1-0_IPTW'], ascending=[False])

        data = []
        data_pvalue = []
        if model == 'LR':
            drug_list = df_sort['drug'].tolist()
            drug_name_list = df_sort['drug_name'].tolist()
            print('len(drug_list):', len(drug_list))
            N = len(drug_list)
            n_box = len(models)

        drug_list2 = df_sort['drug'].tolist()
        for drug in drug_list:
            rdf = pd.read_excel(dirname + 'results/' + drug + '_results.xlsx')

            if contrl_type != 'all':
                idx_all = (rdf['ctrl_type'] == contrl_type)
            else:
                idx_all = (rdf['ctrl_type'].notna())

            # Only select balanced trial
            idx = idx_all & (rdf['n_unbalanced_feature_IPTW'] <= MAX_NO_UNBALANCED_FEATURE)
            c = "KM1-0_IPTW"  # "ATE_original", "ATE_IPTW", "KM1-0_original", "KM1-0_IPTW", 'HR_ori', 'HR_IPTW'
            nv = rdf.loc[idx, c]
            if drug in drug_list2:
                data.append(np.array(rdf.loc[idx, c]) * 100)
            else:
                data.append(np.array([]))

        data_list.append(data)

    colors = ['#F65453', '#82A2D3', '#FAC200', 'purple']
    fig, ax = plt.subplots(figsize=(18, 8))
    width = 0.5  # the width of the bars
    ind = np.arange(N) * width * (n_box + 1)  # the x locations for the groups
    sym = 'o'
    # 'meanline':True,
    box_kw = {"sym": sym, "widths": width - 0.04, "patch_artist": True, "notch": True,
              'showmeans': True,  # 'meanline':True,
              "meanprops": dict(linestyle='--', linewidth=1, markeredgecolor='black', marker='^',
                                markerfacecolor="None")}

    rects_list = []
    for i in range(len(data_list)):
        rects = plt.boxplot(data_list[i], positions=ind + i * width, **box_kw)
        rects_list.append(rects)

    def plot_strip(ind, data, color):
        w = width - 0.15
        swarm1 = pd.DataFrame([(ind[i], data[i][j]) for i in range(len(ind)) for j in range(len(data[i]))],
                              columns=['x', 'y'])
        strip_rx = stats.uniform(-w / 2., w).rvs(len(swarm1))
        # sns.stripplot(x='x', y='y', data=swarm1, color=".25", alpha=0.2, ax=ax)
        plt.scatter(swarm1['x'] + strip_rx, swarm1['y'], alpha=.6, c=color)
        # plt.scatter(swarm1['x'] + strip_rx, swarm1['y'], alpha=1, facecolors='none', edgecolors=color)  # c=color) #,

    # ticks = list(drug_name_list)
    for i, bplot in enumerate(rects_list):
        for patch in bplot['boxes']:
            patch.set_facecolor(colors[i])
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .6))
        # plt.setp(bplot['boxes'], color=color)
        # plt.setp(bplot['whiskers'], color=color)
        # plt.setp(bplot['caps'], color=color)
        plt.setp(bplot['medians'], color='black')

    for i in range(len(data_list)):
        plot_strip(ind + i * width, data_list[i], colors[i])

    # plt.ylim([0.5, 0.85])
    ax.set_xticks(ind + width / 2)
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
    ax.set_ylabel("Adjusted survival difference %", fontsize=25)
    plt.axhline(y=0., color='black', linestyle='--')

    def significance(val):
        if val < 0.001:
            return '***'
        elif val < 0.01:
            return '**'
        elif val < 0.05:
            return '*'
        else:
            return 'ns'

    # for d in range(len(data_list)):
    #     data = data_list[d]
    #     for i in range(N):
    #         n1 = len(data[i])
    #         n2 = len(data_2[i])
    #         y1 = max(data_1[i]) * 1.02
    #         try:
    #             y2 = max(data_2[i]) * 1.02
    #         except:
    #             y2 = y1
    #         x1 = ind[i]
    #         x2 = ind[i] + width
    #         ax.text(x1, y1, str(n1), ha='center', va='bottom', fontsize=14)
    #         ax.text(x2, y2, str(n2), ha='center', va='bottom', fontsize=14)

    ax.legend((rects["boxes"][0] for rects in rects_list),
              (m if m != 'LIGHTGBM' else 'GBM' for m in models), fontsize=20)
    ax.set_xmargin(0.01)
    plt.tight_layout()
    if dump:
        dirname = dirname_list[0]
        check_and_mkdir(dirname + 'results/fig/')
        fig.savefig(
            dirname + 'results/fig/adjusted_survival_diff_boxplot-modelall-{}.png'.format(contrl_type))
        fig.savefig(
            dirname + 'results/fig/adjusted_survival_diff_boxplot-modelall-{}.pdf'.format(contrl_type))
    plt.show()
    plt.clf()


if __name__ == '__main__':
    with open(r'pickles/rxnorm_label_mapping.pkl', 'rb') as f:
        drug_name = pickle.load(f)
        print('Using rxnorm_cui vocabulary, len(drug_name) :', len(drug_name))

    cohort_dir_name = 'save_cohort_all_loose'
    model = 'LSTM'  # 'LSTM'  # 'MLP'  #'LIGHTGBM'  # 'MLP'  #'LR'  #'LSTM' #
    # results_model_selection_for_ml(cohort_dir_name=cohort_dir_name, model=model, drug_name=drug_name, niter=50)
    # results_model_selection_for_ml_step2(cohort_dir_name=cohort_dir_name, model=model, drug_name=drug_name)

    # results_model_selection_for_ml_step2More(cohort_dir_name=cohort_dir_name, model=model, drug_name=drug_name)

    # results_ATE_for_ml(cohort_dir_name=cohort_dir_name, model=model, niter=50)
    # results_ATE_for_ml_step2(cohort_dir_name=cohort_dir_name, model=model, drug_name=drug_name)
    # results_ATE_for_ml_step3_finalInfo(cohort_dir_name, model)

    #
    # combine_ate_final_LR_with(cohort_dir_name, 'LSTM') # needs to compute lstm case first
    #
    # major plots from 3 methods
    # bar_plot_model_selection(cohort_dir_name=cohort_dir_name, model=model, contrl_type='random')
    # bar_plot_model_selection(cohort_dir_name=cohort_dir_name, model=model, contrl_type='atc')
    bar_plot_model_selection(cohort_dir_name=cohort_dir_name, model=model, contrl_type='all')

    # zz
    # sys.exit(0)

    # arrow_plot_model_selection_unbalance_reduction(cohort_dir_name=cohort_dir_name, model=model, contrl_type='random',
    #                                                datapart='all')
    # arrow_plot_model_selection_unbalance_reduction(cohort_dir_name=cohort_dir_name, model=model, contrl_type='random',
    #                                                datapart='train')
    # arrow_plot_model_selection_unbalance_reduction(cohort_dir_name=cohort_dir_name, model=model, contrl_type='random',
    #                                                datapart='test')
    #
    # arrow_plot_model_selection_unbalance_reduction(cohort_dir_name=cohort_dir_name, model=model, contrl_type='atc',
    #                                                datapart='all')
    # arrow_plot_model_selection_unbalance_reduction(cohort_dir_name=cohort_dir_name, model=model, contrl_type='atc',
    #                                                datapart='train')
    # arrow_plot_model_selection_unbalance_reduction(cohort_dir_name=cohort_dir_name, model=model, contrl_type='atc',
    #                                                datapart='test')

    arrow_plot_model_selection_unbalance_reduction(cohort_dir_name=cohort_dir_name, model=model, contrl_type='all',
                                                   datapart='all')
    arrow_plot_model_selection_unbalance_reduction(cohort_dir_name=cohort_dir_name, model=model, contrl_type='all',
                                                   datapart='train')
    arrow_plot_model_selection_unbalance_reduction(cohort_dir_name=cohort_dir_name, model=model, contrl_type='all',
                                                   datapart='test')

    zzz

    sys.exit(0)
    #
    box_plot_model_selection(cohort_dir_name=cohort_dir_name, model=model, contrl_type='random')
    box_plot_model_selection(cohort_dir_name=cohort_dir_name, model=model, contrl_type='atc')
    box_plot_model_selection(cohort_dir_name=cohort_dir_name, model=model, contrl_type='all')
    sys.exit(0)

    # # # ## all methods plots in appendix
    bar_plot_model_selectionV2(cohort_dir_name=cohort_dir_name, model=model, contrl_type='random')
    bar_plot_model_selectionV2(cohort_dir_name=cohort_dir_name, model=model, contrl_type='atc')
    bar_plot_model_selectionV2(cohort_dir_name=cohort_dir_name, model=model, contrl_type='all')

    # bar_plot_model_selectionV2_test(cohort_dir_name=cohort_dir_name, model=model, contrl_type='random')
    # bar_plot_model_selectionV2_test(cohort_dir_name=cohort_dir_name, model=model, contrl_type='atc')
    # bar_plot_model_selectionV2_test(cohort_dir_name=cohort_dir_name, model=model, contrl_type='all')
    # # #
    # box_plot_model_selectionV2(cohort_dir_name=cohort_dir_name, model=model, contrl_type='random')
    # box_plot_model_selectionV2(cohort_dir_name=cohort_dir_name, model=model, contrl_type='atc')
    # box_plot_model_selectionV2(cohort_dir_name=cohort_dir_name, model=model, contrl_type='all')

    # box_plot_ate(cohort_dir_name, model=model, model2='MLP', contrl_type='random')
    # box_plot_ate(cohort_dir_name, model=model, model2='MLP', contrl_type='atc')
    # box_plot_ate(cohort_dir_name, model=model, model2='MLP', contrl_type='all')

    # box_plot_ate(cohort_dir_name, model=model, model2='MLP', contrl_type='all')
    # box_plot_ate(cohort_dir_name, model=model, model2='LIGHTGBM', contrl_type='all')
    # box_plot_ate(cohort_dir_name, model=model, model2='LSTM', contrl_type='all')
    #
    # box_plot_ate(cohort_dir_name, model=model, model2='MLP', contrl_type='atc')
    # box_plot_ate(cohort_dir_name, model=model, model2='LIGHTGBM', contrl_type='atc')
    # box_plot_ate(cohort_dir_name, model=model, model2='LSTM', contrl_type='atc')
    #
    # box_plot_ate(cohort_dir_name, model=model, model2='MLP', contrl_type='random')
    # box_plot_ate(cohort_dir_name, model=model, model2='LIGHTGBM', contrl_type='random')
    # box_plot_ate(cohort_dir_name, model=model, model2='LSTM', contrl_type='random')

    # box_plot_ate_V2(cohort_dir_name, models=['LR', 'LSTM', 'MLP', 'LIGHTGBM'], contrl_type='random')
    # box_plot_ate_V2(cohort_dir_name, models=['LR', 'LSTM', 'MLP', 'LIGHTGBM'], contrl_type='atc')
    # box_plot_ate_V2(cohort_dir_name, models=['LR', 'LSTM', 'MLP', 'LIGHTGBM'], contrl_type='all')

    print('Done')
