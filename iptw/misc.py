import os
import shutil
import zipfile

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
from utils import check_and_mkdir
from scipy import stats
import re
import itertools
import functools


print = functools.partial(print, flush=True)


MAX_NO_UNBALANCED_FEATURE = 5


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
    # if plot:
    #     plt.hist(sampling_distribution, bins="fd")
    return quantile_confidence_interval


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


def shell_for_ml(cohort_dir_name, model, iter=50, min_patients=500, stats=True):
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
                for seed in range(0, iter):
                    cmd = "python main.py --data_dir ../ipreprocess/output/{}/ --treated_drug {} " \
                          "--controlled_drug {} --run_model {} --output_dir output/{}/{}/ --random_seed {} " \
                          "--drug_coding rxnorm --med_code_topk 200 {} " \
                          "2>&1 | tee output/{}/{}/log/{}_S{}D200C{}_{}.log\n".format(
                        cohort_dir_name, drug,
                        ctrl_type, model, cohort_dir_name, model, seed, '--stats' if stats else '',
                        cohort_dir_name, model, drug, seed, ctrl_type, model)
                    fo.write(cmd)
                    n += 1

    fo.close()
    print('In total ', n, ' commands')


def results_model_selection_for_ml(cohort_dir_name, model, drug_name):
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


def IQR(s):
    return [np.quantile(s, .5), np.quantile(s, .25), np.quantile(s, .75)]


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
                nsmd_mean_ci = bootstrap_mean_ci(nsmd, alpha=0.05)

                success_rate = (nsmd <= MAX_NO_UNBALANCED_FEATURE).mean()
                success_rate_ci = bootstrap_mean_ci(nsmd <= MAX_NO_UNBALANCED_FEATURE, alpha=0.05)

                auc_med = IQR(auc)[0]
                auc_iqr = IQR(auc)[1:]

                auc_mean = auc.mean()
                auc_mean_ci = bootstrap_mean_ci(auc, alpha=0.05)

                r.extend([nsmd_med, nsmd_iqr, nsmd_mean, nsmd_mean_ci, success_rate, success_rate_ci,
                          auc_med, auc_iqr, auc_mean, auc_mean_ci])
                col_name.extend(
                    ["nsmd_med-" + c1, "nsmd_iqr-" + c1, "nsmd_mean-" + c1, "nsmd_mean_ci-" + c1,
                     "success_rate-" + c1, "success_rate_ci-" + c1,
                     "auc_med-" + c2, "auc_iqr-" + c2, "auc_mean-" + c2, "auc_mean_ci-" + c2])

            results.append(r)
        df = pd.DataFrame(results, columns=col_name)
        df.to_excel(writer, sheet_name=t)
    writer.save()
    print()


def results_ATE_for_ml(cohort_dir_name, model, dug_name):
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
            for seed in range(0, 20):
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
                idx = (rdf['ctrl_type'] == t)
            else:
                idx = (rdf['ctrl_type'].notna())

            # Only select balanced trial
            idx = idx & (rdf['n_unbalanced_feature_IPTW'] <= MAX_NO_UNBALANCED_FEATURE)
            print('drug: ', drug, t, 'support:', idx.sum())
            r = [drug, drug_name.get(drug, ''), cohort_size.get(drug+'.pkl'), idx.sum()]
            col_name = ['drug', 'drug_name', 'n_treat_size', 'support']

            for c in ["n_treat", "n_ctrl", "n_feature"]:  # , 'HR_IPTW', 'HR_IPTW_CI'
                nv = rdf.loc[idx, c]
                nv_mean = nv.mean()
                r.append(nv_mean)
                col_name.append(c)

            for c in ["n_unbalanced_feature", "n_unbalanced_feature_IPTW",
                      "ATE_original", "ATE_IPTW",
                      "KM1-0_original", "KM1-0_IPTW"]:  # , 'HR_IPTW', 'HR_IPTW_CI'
                nv = rdf.loc[idx, c]
                if len(nv) > 0:
                    med = IQR(nv)[0]
                    iqr = IQR(nv)[1:]

                    mean = nv.mean()
                    mean_ci = bootstrap_mean_ci(nv, alpha=0.05)

                    p, _ = bootstrap_mean_pvalue(nv, 0)

                    r.extend([med, iqr, mean, mean_ci, p])
                else:
                    r.extend([np.nan, np.nan, np.nan, np.nan, np.nan])
                col_name.extend(["med-"+c, "iqr-"+c, "mean-"+c, "mean_ci-"+c, 'pvalue-'+c])

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


if __name__ == '__main__':
    # check_drug_name_code()
    # test bootstrap method
    # rvs = stats.norm.rvs(loc=0, scale=10, size=(500, 1))
    # ci = bootstrap_mean_ci(rvs)
    # p, test_orig = bootstrap_mean_pvalue(rvs, expected_mean=0., B=1000)

    with open(r'pickles/rxnorm_label_mapping.pkl', 'rb') as f:
        drug_name = pickle.load(f)
        print('Using rxnorm_cui vocabulary, len(drug_name) :', len(drug_name))

    shell_for_ml(cohort_dir_name='save_cohort_all_loose', model='LR')
    # results_model_selection_for_ml(cohort_dir_name='save_cohort_all_loose', model='LR', drug_name=drug_name)
    # results_model_selection_for_ml_step2(cohort_dir_name='save_cohort_all_loose', model='LR', drug_name=drug_name)
    # results_ATE_for_ml(cohort_dir_name='save_cohort_all_loose', model='LR', drug_name=drug_name)
    # results_ATE_for_ml_step2(cohort_dir_name='save_cohort_all_loose', model='LR', drug_name=drug_name)

    shell_for_ml(cohort_dir_name='save_cohort_all_loose', model='LIGHTGBM', stats=False)
    # results_model_selection_for_ml(cohort_dir_name='save_cohort_all_loose', model='LIGHTGBM', drug_name=drug_name)

    print('Done')
