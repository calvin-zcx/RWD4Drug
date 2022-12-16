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

np.random.seed(0)
random.seed(0)


def plot_balance_scatter(input_dir, cohort_dir_name, model):
    output_dir = r'plots/balance/'
    dirname = r'{}/{}/{}/'.format(input_dir, cohort_dir_name, model)
    df_all = pd.read_excel(dirname + 'results/summarized_IPTW_ATE_{}.xlsx'.format(model), sheet_name=None)

    if input_dir == 'output_marketscan':
        data = 'marketscan'
    else:
        data = 'florida'

    for sheet in ['random', 'atc', 'all']:
        df = df_all[sheet]
        # Only select drugs with selection criteria trial
        # 1. minimum support set 10, may choose 20 later
        # 2. p value < 0.05

        color = (df['support'] / df['niters'])
        size = 10 * (df['support'] + 2) #+ 0.01
        x = df['n_treat-uab'] + df['n_ctrl-uab']
        y = df['mean-n_unbalanced_feature-uab']  # df['mean-n_unbalanced_feature_IPTW-uab']  #
        c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA', '#86CEFA']
        colors = ['#FAC200', '#82A2D3', '#F65453']
        fig, ax = plt.subplots() #figsize=(10, 10)) #figsize=(24, 8)
        im = ax.scatter(x, y, c=color, s=size, alpha=0.4, edgecolors='none') #, cmap='viridis')
        # ax.set_xscale('log')
        ax.set_xlabel("No. of patients", fontsize=20)
        ax.set_ylabel("No. of unbalanced features IPTW", fontsize=20)  # Success Rate of Balancing
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('balance ratio', fontsize=15)
        fig.tight_layout()

        check_and_mkdir(output_dir)
        plt.savefig(output_dir + 'scatter_{}_{}_{}.png'.format(data, model, sheet), bbox_inches='tight')
        plt.savefig(output_dir + 'scatter_{}_{}_{}.pdf'.format(data, model, sheet), bbox_inches='tight', transparent=True)
        plt.show()
        # plt.clf()
        plt.close()

        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(True)
        # ax.spines['left'].set_visible(False)
        # idx = (df['support'] >= 10) & (df['pvalue-KM1-0_IPTW'] <= 0.05)
        # df_sort = df.loc[idx, :].sort_values(by=['mean-KM1-0_IPTW'], ascending=[False])
        #
        # df_final = df_sort[
        #     ['drug', 'drug_name', 'niters', 'support', 'n_treat', 'n_ctrl', 'n_feature',
        #      'mean-n_unbalanced_feature', 'mean_ci-n_unbalanced_feature',
        #      'mean-n_unbalanced_feature_IPTW', 'mean_ci-n_unbalanced_feature_IPTW',
        #      # 'mean-ATE_original', 'mean_ci-ATE_original', 'pvalue-ATE_original',
        #      # 'mean-ATE_IPTW', 'mean_ci-ATE_IPTW', 'pvalue-ATE_IPTW',
        #      # 'mean-KM1-0_original', 'mean_ci-KM1-0_original', 'pvalue-KM1-0_original',
        #      'mean-KM1-0_IPTW', 'mean_ci-KM1-0_IPTW', 'pvalue-KM1-0_IPTW',
        #      'mean-HR_IPTW', 'mean_ci-HR_IPTW', 'pvalue-HR_IPTW']]

    print('Done results_ATE_for_ml_step3_finalInfo')


if __name__ == '__main__':
    cohort_dir_name = 'save_cohort_all_loose'
    model = 'LR'  # 'MLP'  # 'LR' #'LIGHTGBM'  #'LR'  #'LSTM'
    plot_balance_scatter('output', cohort_dir_name, model)
    plot_balance_scatter('output_marketscan', cohort_dir_name, model)

    print('Done!')


