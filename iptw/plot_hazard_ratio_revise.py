import os
import shutil
import zipfile

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
from misc import stringlist_2_str, stringlist_2_list
import zepid
from zepid.graphics import EffectMeasurePlot

np.random.seed(0)
random.seed(0)


def results_extract(output_dir, cohort_dir_name, model, drug_list=[]):
    dirname = r'{}/{}/{}/'.format(output_dir, cohort_dir_name, model)
    df_all = pd.read_excel(dirname + 'results/summarized_IPTW_ATE_{}.xlsx'.format(model), sheet_name=None,
                           dtype={'drug': str})
    writer = pd.ExcelWriter(dirname + 'results/summarized_IPTW_ATE_{}_selectedByIDs.xlsx'.format(model),
                            engine='xlsxwriter')
    for sheet in ['random', 'atc', 'all']:
        df = df_all[sheet]
        if drug_list:
            idx = df['drug'].isin(drug_list)
            df_sort = df.loc[idx, :]
            df_sort['drug_list_order'] = np.nan
            for i, drug in enumerate(drug_list):
                df_sort.loc[df_sort['drug'] == drug, 'drug_list_order'] = i
            df_sort = df_sort.sort_values(by=['drug_list_order'], ascending=[True])
        else:
            # Only select drugs with selection criteria trial
            # 1. minimum support set 10, may choose 20 later
            # 2. p value < 0.05
            idx = (df['support'] >= 10) & (df['pvalue-KM1-0_IPTW'] <= 0.05)
            df_sort = df.loc[idx, :].sort_values(by=['mean-KM1-0_IPTW'], ascending=[False])

        df_final = df_sort[
            ['drug', 'drug_name', 'niters', 'support', 'n_treat', 'n_ctrl', 'n_feature',
             'mean-n_unbalanced_feature', 'mean_ci-n_unbalanced_feature',
             'mean-n_unbalanced_feature_IPTW', 'mean_ci-n_unbalanced_feature_IPTW',
             # 'mean-ATE_original', 'mean_ci-ATE_original', 'pvalue-ATE_original',
             # 'mean-ATE_IPTW', 'mean_ci-ATE_IPTW', 'pvalue-ATE_IPTW',
             # 'mean-KM1-0_original', 'mean_ci-KM1-0_original', 'pvalue-KM1-0_original',
             'mean-KM1-0_IPTW', 'mean_ci-KM1-0_IPTW', 'pvalue-KM1-0_IPTW',
             'mean-HR_IPTW', 'mean_ci-HR_IPTW', 'pvalue-HR_IPTW']]

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
        df_final['mean_ci-HR_IPTW'] = df_final['mean_ci-HR_IPTW'].apply(
            lambda x: stringlist_2_str(x, False, 2))

        df_final.to_excel(writer, sheet_name=sheet)

    writer.save()
    print('Done results_extract')


def plot_forest_for_drug(drug_id, drug_id_gpi, drug_name):
    output_dir = r'plots/hazard_ratio_revise/'
    df_fl = pd.read_excel('output/revise/save_cohort_all_loose/LR/results/summarized_IPTW_ATE_LR_finalInfo_fl.xlsx',
                          sheet_name=None,
                          dtype={'drug': str})

    df_ms = pd.read_excel(
        # 'output_marketscan/revise/save_cohort_all_loose/LR/results/summarized_IPTW_ATE_LR_finalInfo_ms.xlsx',
        'output_marketscan/revise/save_cohort_all_loose/LR/results/summarized_IPTW_ATE_LR_finalInfo_ms_threshold1.xlsx',
        sheet_name=None,
        dtype={'drug': str})

    labs = ['FL-All', 'FL-Rand', 'FL-ATC', 'MS-All', 'MS-Rand', 'MS-ATC']
    measure = []
    lower = []
    upper = []

    npos = []
    nneg = []

    for sheet in ['all', 'random', 'atc']:
        df = df_fl[sheet].set_index('drug')
        v = df.loc[drug_id, 'mean-HR_IPTW']
        ci = stringlist_2_list(df.loc[drug_id, 'mean_ci-HR_IPTW'])
        measure.append(v)
        lower.append(ci[0])
        upper.append(ci[1])

        npos.append(df.loc[drug_id, 'n_treat'])
        nneg.append(df.loc[drug_id, 'n_ctrl'])

    for sheet in ['all', 'random', 'atc']:
        df = df_ms[sheet].set_index('drug')
        v = df.loc[drug_id_gpi, 'mean-HR_IPTW']
        ci = stringlist_2_list(df.loc[drug_id_gpi, 'mean_ci-HR_IPTW'])
        measure.append(v)
        lower.append(ci[0])
        upper.append(ci[1])

        npos.append(df.loc[drug_id_gpi, 'n_treat'])
        nneg.append(df.loc[drug_id_gpi, 'n_ctrl'])


    # dump sourc data 2023-11-1
    df_source = pd.DataFrame({'label': labs, 'aHR': measure, 'CI lower': lower, 'CI upper': upper,
                              'n treat': npos, 'n control': nneg})
    df_source.to_csv(output_dir + '{}_{}_forest-source-data.csv'.format(drug_id, drug_name))
    # return df_source
    ##

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
                          ncumIncidence=npos, nabs=nneg)

    p.labels(effectmeasure='aHR', add_label1='#Treated', add_label2='#Contrl')
    # '#F65453', '#82A2D3'
    c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA', '#86CEFA']
    p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)
    # ax = p.plot(figsize=(7.5, 3), t_adjuster=0.09, max_value=1.2, min_value=0.5, decimal=2)

    ax = p.plot_with_incidence(figsize=(8.5, 3), t_adjuster=0.09, max_value=1.2, min_value=0.5, decimal=2)
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    plt.title(drug_name, loc="center", x=-0.5, y=1.045)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)
    ax.set_yticklabels(labs, fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    # plt.tight_layout()

    check_and_mkdir(output_dir)
    plt.savefig(output_dir + '{}_{}_forest.png'.format(drug_id, drug_name), bbox_inches='tight')
    plt.savefig(output_dir + '{}_{}_forest.pdf'.format(drug_id, drug_name), bbox_inches='tight', transparent=True)
    plt.show()
    # plt.clf()
    plt.close()
    return df_source

def plot_forest_for_drug_selectcov(drug_id, drug_id_gpi, drug_name):
    output_dir = r'plots/hazard_ratio_revise_selectcov/'
    df_fl = pd.read_excel(
        'output/revise_selectcov/save_cohort_all_loose/LR/results/summarized_IPTW_ATE_LR_finalInfo_fl.xlsx',
        sheet_name=None,
        dtype={'drug': str})

    df_ms = pd.read_excel(
        'output_marketscan/revise_selectcov/save_cohort_all_loose/LR/results/summarized_IPTW_ATE_LR_finalInfo_ms.xlsx',
        sheet_name=None,
        dtype={'drug': str})

    labs = ['FL-All', 'FL-Rand', 'FL-ATC', 'MS-All', 'MS-Rand', 'MS-ATC']
    measure = []
    lower = []
    upper = []
    npos = []
    nneg = []

    for sheet in ['all', 'random', 'atc']:
        df = df_fl[sheet].set_index('drug')
        v = df.loc[drug_id, 'mean-HR_IPTW']
        ci = stringlist_2_list(df.loc[drug_id, 'mean_ci-HR_IPTW'])
        measure.append(v)
        lower.append(ci[0])
        upper.append(ci[1])
        npos.append(df.loc[drug_id, 'n_treat'])
        nneg.append(df.loc[drug_id, 'n_ctrl'])

    for sheet in ['all', 'random', 'atc']:
        df = df_ms[sheet].set_index('drug')
        v = df.loc[drug_id_gpi, 'mean-HR_IPTW']
        ci = stringlist_2_list(df.loc[drug_id_gpi, 'mean_ci-HR_IPTW'])
        measure.append(v)
        lower.append(ci[0])
        upper.append(ci[1])
        npos.append(df.loc[drug_id_gpi, 'n_treat'])
        nneg.append(df.loc[drug_id_gpi, 'n_ctrl'])

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
                          ncumIncidence=npos, nabs=nneg)
    p.labels(effectmeasure='aHR', add_label1='#Treated', add_label2='#Contrl')
    # p.labels(scale='log')

    # '#F65453', '#82A2D3'
    c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA', '#86CEFA']
    p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)
    # ax = p.plot(figsize=(7.5, 3), t_adjuster=0.09, max_value=1.1, min_value=0.45, decimal=2)
    ax = p.plot_with_incidence(figsize=(8.5, 3), t_adjuster=0.09, max_value=1.1, min_value=0.45, decimal=2)

    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    plt.title(drug_name, loc="center", x=-0.5, y=1.045)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)
    ax.set_yticklabels(labs, fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    # plt.tight_layout()

    check_and_mkdir(output_dir)
    plt.savefig(output_dir + '{}_{}_forest.png'.format(drug_id, drug_name), bbox_inches='tight')
    plt.savefig(output_dir + '{}_{}_forest.pdf'.format(drug_id, drug_name), bbox_inches='tight', transparent=True)
    plt.show()
    # plt.clf()
    plt.close()


def plot_forest_for_drug_f5yrs(drug_id, drug_id_gpi, drug_name):
    output_dir = r'plots/hazard_ratio_revise_f5yrs/'
    df_fl = pd.read_excel(
        'output/revise_f5yrs/save_cohort_all_loose_f5yrs/LR/results/summarized_IPTW_ATE_LR_finalInfo_fl.xlsx',
        sheet_name=None,
        dtype={'drug': str})

    df_ms = pd.read_excel(
        # 'output_marketscan/revise/save_cohort_all_loose/LR/results/summarized_IPTW_ATE_LR_finalInfo_ms.xlsx',
        'output_marketscan/revise_f5yrs/save_cohort_all_loose_f5yrs/LR/results/summarized_IPTW_ATE_LR_finalInfo_ms.xlsx',
        sheet_name=None,
        dtype={'drug': str})

    labs = ['FL-All', 'FL-Rand', 'FL-ATC', 'MS-All', 'MS-Rand', 'MS-ATC']
    measure = []
    lower = []
    upper = []
    npos = []
    nneg = []

    for sheet in ['all', 'random', 'atc']:
        df = df_fl[sheet].set_index('drug')
        v = df.loc[drug_id, 'mean-HR_IPTW']
        ci = stringlist_2_list(df.loc[drug_id, 'mean_ci-HR_IPTW'])
        measure.append(v)
        lower.append(ci[0])
        upper.append(ci[1])
        npos.append(df.loc[drug_id, 'n_treat'])
        nneg.append(df.loc[drug_id, 'n_ctrl'])

    for sheet in ['all', 'random', 'atc']:
        df = df_ms[sheet].set_index('drug')
        v = df.loc[drug_id_gpi, 'mean-HR_IPTW']
        ci = stringlist_2_list(df.loc[drug_id_gpi, 'mean_ci-HR_IPTW'])
        measure.append(v)
        lower.append(ci[0])
        upper.append(ci[1])
        npos.append(df.loc[drug_id_gpi, 'n_treat'])
        nneg.append(df.loc[drug_id_gpi, 'n_ctrl'])

    # dump sourc data 2023-11-1
    df_source = pd.DataFrame({'label': labs, 'aHR': measure, 'CI lower': lower, 'CI upper':upper,
                              'n treat': npos, 'n control': nneg})
    df_source.to_csv(output_dir + '{}_{}_forest-source-data.csv'.format(drug_id, drug_name))
    # return df_source
    ##

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
                          ncumIncidence=npos, nabs=nneg)
    p.labels(effectmeasure='aHR', add_label1='#Treated', add_label2='#Contrl')
    # '#F65453', '#82A2D3'
    c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA', '#86CEFA']
    p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)
    # ax = p.plot(figsize=(7.5, 3), t_adjuster=0.09, max_value=1.2, min_value=0.5, decimal=2)
    ax = p.plot_with_incidence(figsize=(8.5, 3), t_adjuster=0.09, max_value=1.2, min_value=0.5, decimal=2)

    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    plt.title(drug_name, loc="center", x=-0.5, y=1.045)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)
    ax.set_yticklabels(labs, fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    # plt.tight_layout()

    check_and_mkdir(output_dir)
    plt.savefig(output_dir + '{}_{}_forest.png'.format(drug_id, drug_name), bbox_inches='tight')
    plt.savefig(output_dir + '{}_{}_forest.pdf'.format(drug_id, drug_name), bbox_inches='tight', transparent=True)
    plt.show()
    # plt.clf()
    plt.close()
    return df_source


if __name__ == '__main__':
    cohort_dir_name = 'save_cohort_all_loose'
    model = 'LR'  # 'MLP'  # 'LR' #'LIGHTGBM'  #'LR'  #'LSTM'

    drug_label = ['pantoprazole', 'gabapentin', 'atorvastatin', 'albuterol', 'fluticasone', 'omeprazole']
    drug_id = ['40790', '25480', '83367', '435', '41126', '7646']
    drug_id_gpi = ['49270070', '72600030', '39400010', '44201010', '42200032', '49270060']
    #
    # drug_id = ['40790', '25480', '161', '83367']
    # drug_label = ['pantoprazole', 'gabapentin', 'acetaminophen', 'atorvastatin']
    # drug_id_gpi = ['49270070', '72600030', '65991702', '39400010']

    # results_extract('output', cohort_dir_name, model, drug_id)
    # results_extract('output_marketscan', cohort_dir_name, model, drug_id_gpi)

    # for idx in range(len(drug_id)):
    #     plot_forest_for_drug(drug_id[idx], drug_id_gpi[idx], drug_label[idx])

    # for idx in range(len(drug_id)):
    #     plot_forest_for_drug_selectcov(drug_id[idx], drug_id_gpi[idx], drug_label[idx])

    # drug_label = ['prednisone', 'amoxicillin']
    # drug_id = ['8640', '723']
    # drug_id_gpi = ['22100045', '01200010']

    df_source_dict = {}
    for idx in range(len(drug_id)):
        # df_source = plot_forest_for_drug(drug_id[idx], drug_id_gpi[idx], drug_label[idx])
        df_source = plot_forest_for_drug_f5yrs(drug_id[idx], drug_id_gpi[idx], drug_label[idx])
        df_source_dict[drug_label[idx]] = df_source


    print('Done')
