import sys

from numpy import ndarray

# for linux env.
sys.path.insert(0, '..')
import os

if os.name == 'posix':
    try:
        from sklearnex import patch_sklearn

        patch_sklearn(["LogisticRegression"])
        print('using sklearnex')
    except ModuleNotFoundError as err:
        # Error handling
        print(err)

import time
# from dataset import *
import pickle
import argparse
# from torch.utils.data.sampler import SubsetRandomSampler
from evaluation import *
# import torch.nn.functional as F
from utils import save_model, load_model, check_and_mkdir
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
from ipreprocess.utils import load_icd_to_ccw
# from PSModels import mlp, lstm
from PSModels import ml

import itertools
from tqdm import tqdm
from sklearn.model_selection import KFold
import seaborn as sns

import functools
from scipy.stats import bernoulli

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--data_dir', type=str, default='../ipreprocess/output/save_cohort_all_loose/')
    parser.add_argument('--treated_drug', type=str, default='simu')
    parser.add_argument('--controlled_drug', choices=['atc', 'random'], default='random')
    parser.add_argument('--controlled_drug_ratio', type=int, default=3)  # 2 seems not good. keep unchanged
    parser.add_argument("--random_seed", type=int, default=0)

    parser.add_argument('--train_ratio', type=float, default=0.8)  # )0001)
    parser.add_argument("--nsim", type=int, default=1000000)
    parser.add_argument('--nonlin', choices=['no', 'moderate', 'strong'], default='strong')
    # Output
    parser.add_argument('--output_dir', type=str, default='output/simulate/')

    args = parser.parse_args()

    # Modifying args
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.random_seed >= 0:
        rseed = args.random_seed
    else:
        from datetime import datetime
        rseed = datetime.now()

    args.treated_drug = args.treated_drug + 'n{}train{:.1f}{}'.format(args.nsim, args.train_ratio, args.nonlin)
    args.random_seed = rseed

    return args


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('SMD_THRESHOLD: ', SMD_THRESHOLD)
    print('random_seed: ', args.random_seed)
    print('Drug {} cohort: '.format(args.treated_drug))
    # print('save_model_filename', args.save_model_filename)

    print('device: ', args.device)
    print('torch.cuda.device_count():', torch.cuda.device_count())
    print('torch.cuda.current_device():', torch.cuda.current_device())
    # %% 1. Generate Data
    # from scipy.stats import bernoulli
    n = args.nsim
    print('simulation sample number args.nsim: ', args.nsim)
    x1 = np.random.binomial(1, p=0.5, size=(n,))
    x2 = np.random.binomial(1, p=(0.3 + x1 * 0.1))
    x3 = np.random.binomial(1, p=0.5, size=(n,))
    x4 = np.random.normal(0, 1, (n,))
    x6 = np.random.normal(0, 1, (n,))
    x5 = 0.3 + x6 * 0.1 + np.random.normal(0, 1, (n,))

    h = 261
    h1 = 5
    x_h1 = np.random.binomial(1, p=0.4, size=(n, h1))  # difficult to balance
    x_h2 = np.random.binomial(1, p=0.2, size=(n, h - h1))

    if args.nonlin == 'no':
        print('Only linear effect!')
        logits_wo_bias = (
                np.log(2) * x2 + np.log(3) * x3 + np.log(2) * x5 + np.log(2) * x6 + np.log(1.5) * x_h1.sum(axis=1)
                + np.log(1.1) * x_h2.sum(axis=1))
        # bias = -6.863981632404233   # trial seed determined by bias_ = target_logits_mean - logits_wo_bias.mean()
        bias = -6.84

    elif args.nonlin == 'moderate':
        print('Non-linear effect!')
        logits_wo_bias = (x1 * (np.log(2) * x2 ** 2 + np.log(3) * x3 * x2 + np.log(2) * x5 + np.log(2) * x6 + np.log(
            1.5) * x_h1.sum(axis=1)) + np.log(1.1) * x_h2.sum(axis=1))
        # bias = -5.797399810222922 # trial seed bias_ = target_logits_mean - logits_wo_bias.mean()
        bias = -5.72

    elif args.nonlin == 'strong':
        print('Strong nolinear effect! (discarded!)')
        logits_wo_bias = x1 * (np.log(2) * x2 ** 2 + np.log(3) * x3 * x2 + np.log(2) * x5 + np.log(2) * x6 +
                               np.log(1.5) * x_h1.sum(axis=1) + np.log(1.1) * x_h2.sum(axis=1))
        # bias = -3.3022269580355963
        # bias = -3.3530774116786355
        bias = -3.5

    else:
        raise ValueError

    print('bias:', bias)
    # to help set right bias term, towards target aHR
    # target_logits_mean = -0.15
    # bias_ = target_logits_mean - logits_wo_bias.mean()
    # print(bias_)
    # zz

    logits = bias + logits_wo_bias

    z = np.random.binomial(1, p=(1 / (1 + np.exp(-logits))))

    r = 1.8
    a = 2
    cof = np.log(r) * x1 + np.log(r) * x2 + np.log(r) * x4 + np.log(2.3) * x5 ** 2 + np.log(1.5) * x_h1.sum(
        axis=1) + np.log(1.1) * x_h2.sum(axis=1) - 5.67 - 1. * z
    T = (-np.log(np.random.uniform(0, 1, n)) / np.exp(cof)) ** (1 / a) * 100

    cof_counter = np.log(r) * x1 + np.log(r) * x2 + np.log(r) * x4 + np.log(2.3) * x5 ** 2 + np.log(1.5) * x_h1.sum(
        axis=1) + np.log(1.1) * x_h2.sum(axis=1) - 5.67 - 1. * (1 - z)
    T_counter = (-np.log(np.random.uniform(0, 1, n)) / np.exp(cof_counter)) ** (1 / a) * 100

    Tend = 200
    T_both = np.concatenate((z, 1 - z))
    Y_both = np.concatenate((np.stack((T <= Tend, T), axis=1), np.stack((T_counter <= Tend, T_counter), axis=1)))

    HR_ori, CI_ori, cph_ori = cox_no_weight(T_both, Y_both)
    print('HR {} ({}) p:{}'.format(HR_ori, CI_ori, cph_ori.summary.p.treatment))
    print('np.exp(-1)', np.exp(-1))

    """
    Only linear effect!
    bias: -6.84
    HR 0.5780982066480141 ([0.57641897 0.57978233]) p:0.0
    np.exp(-1) 0.36787944117144233
    
    Non-linear effect!
    bias: -5.72
    HR 0.5780913629899157 ([0.57641216 0.57977545]) p:0.0
    np.exp(-1) 0.36787944117144233
    
    Strong nolinear effect! (discarded!)
    bias: -3.5
    HR 0.5788432844157858 ([0.57716217 0.5805293 ]) p:0.0
    np.exp(-1) 0.36787944117144233
    """

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
