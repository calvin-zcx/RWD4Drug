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
    parser.add_argument('--nonlin', choices=['no', 'moderate', 'strong'], default='no')

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
    # args.save_model_filename = os.path.join(args.output_dir, args.treated_drug,
    #                                         args.treated_drug + '_S{}D{}C{}_{}'.format(args.random_seed,
    #                                                                                    args.med_code_topk,
    #                                                                                    args.controlled_drug,
    #                                                                                    args.run_model))
    # check_and_mkdir(args.save_model_filename)

    # args.hidden_size = [int(x.strip()) for x in args.hidden_size.split(',')
    #                     if (x.strip() not in ('', '0'))]
    # if args.med_code_topk < 1:
    #     args.med_code_topk = None

    return args


# flaten series into static
# train_x, train_t, train_y = flatten_data(my_dataset, train_indices)  # (1764,713), (1764,), (1764,)
def flatten_data(mdata, data_indices, verbose=1):
    print('flatten_data...')
    x, t, y = [], [], []
    for idx in tqdm(data_indices):
        confounder, treatment, outcome = mdata[idx][0], mdata[idx][1], mdata[idx][2]
        dx, rx, sex, age, days = confounder[0], confounder[1], confounder[2], confounder[3], confounder[4]
        dx, rx = np.sum(dx, axis=0), np.sum(rx, axis=0)
        dx = np.where(dx > 0, 1, 0)
        rx = np.where(rx > 0, 1, 0)

        x.append(np.concatenate((dx, rx, [sex], [age], [days])))
        t.append(treatment)
        y.append(outcome)

    x, t, y = np.asarray(x), np.asarray(t), np.asarray(y)
    if verbose:
        d1 = len(dx)
        d2 = len(rx)
        print('...dx:', x[:, :d1].shape, 'non-zero ratio:', (x[:, :d1] != 0).mean(), 'all-zero:',
              (x[:, :d1].mean(0) == 0).sum())
        print('...rx:', x[:, d1:d1 + d2].shape, 'non-zero ratio:', (x[:, d1:d1 + d2] != 0).mean(), 'all-zero:',
              (x[:, d1:d1 + d2].mean(0) == 0).sum())
        print('...all:', x.shape, 'non-zero ratio:', (x != 0).mean(), 'all-zero:', (x.mean(0) == 0).sum())
    return x, t, y


def flatten_data_sim(x, treat, y, t2e, data_indices, verbose=1):
    print('flatten_data_sim...')
    y = np.stack((y, t2e), axis=1)

    x_sample = x[data_indices]
    treat_sample = treat[data_indices]
    y_sample = y[data_indices]

    if verbose:
        print('treatment:', treat_sample.mean(), treat_sample.sum())
        print('y_sample:', treat_sample.mean(axis=0), treat_sample.sum(axis=0))
        print('...all:', x_sample.shape, 'non-zero ratio:', (x_sample!= 0).mean(), 'all-zero:', (x_sample.mean(0) == 0).sum())
    return x_sample, treat_sample, y_sample


def _evaluation_helper(X, T, PS_logits, loss):
    y_pred_prob = logits_to_probability(PS_logits, normalized=False)
    auc = roc_auc_score(T, y_pred_prob)
    max_smd, smd, max_smd_weighted, smd_w = cal_deviation(X, T, PS_logits, normalized=False, verbose=False)
    n_unbalanced_feature = len(np.where(smd > SMD_THRESHOLD)[0])
    n_unbalanced_feature_weighted = len(np.where(smd_w > SMD_THRESHOLD)[0])
    result = (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
    return result


def _loss_helper(v_loss, v_weights):
    return np.dot(v_loss, v_weights) / np.sum(v_weights)


# def main(args):
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
    x2 = np.random.binomial(1, p=(0.3+x1*0.1))
    x3 = np.random.binomial(1, p=0.5, size=(n,))
    x4 = np.random.normal(0, 1, (n,))
    x6 = np.random.normal(0, 1, (n,))
    x5 = 0.3 + x6 * 0.1 + np.random.normal(0, 1, (n,))

    h = 261
    h1 = 5
    x_h1 = np.random.binomial(1, p=0.4, size=(n, h1)) # difficult to balance
    x_h2 = np.random.binomial(1, p=0.2, size=(n, h-h1))

    logits = -3.1 + np.log(3) * x2**2 + np.log(1.5) * x3 + np.log(1.5) * x5 + np.log(2) * x6 + np.log(3) * x_h1.sum(axis=1) + np.log(1.1) * x_h1.sum(axis=1)
    # logits = -1.2 + np.log(3) * x2**2 + np.log(1.5) * x3 + np.log(1.5) * x5 + np.log(2) * x6
    logits = -2.85 + np.log(1.5) * x2**2 + np.log(3) * x3 * x2 + np.log(1.5) * x5 + np.log(2) * x6 + np.log(3) * x_h1.sum(axis=1) + np.log(1.1) * x_h1.sum(axis=1)

    if args.nonlin == 'no':
        print('No nolinear effect!')
        logits_wo_bias = (
                np.log(2) * x2 + np.log(3) * x3 + np.log(2) * x5 + np.log(2) * x6 + np.log(1.5) * x_h1.sum(axis=1)
                + np.log(1.1) * x_h2.sum(axis=1))
        bias = -6.863981632404233

    elif args.nonlin == 'moderate':
        print('Moderate nolinear effect!')
        logits_wo_bias = (x1 * (np.log(2) * x2 ** 2 + np.log(3) * x3 * x2 + np.log(2) * x5 + np.log(2) * x6 + np.log(
            1.5) * x_h1.sum(axis=1)) + np.log(1.1) * x_h2.sum(axis=1))
        bias = -5.797399810222922
    elif args.nonlin == 'strong':
        print('Strong nolinear effect!')
        logits_wo_bias = x1 * (np.log(2) * x2 ** 2 + np.log(3) * x3 * x2 + np.log(2) * x5 + np.log(2) * x6 +
                               np.log(1.5) * x_h1.sum(axis=1) + np.log(1.1) * x_h2.sum(axis=1))
        bias = -3.3022269580355963
    else:
        raise ValueError
    logits = bias + logits_wo_bias

    z = np.random.binomial(1, p=(1/(1+np.exp(-logits))))
    X = np.stack((x1, x2, x3, x4, x5, x6), axis=1)
    X = np.concatenate((X, x_h1, x_h2), axis=1)
    n_feature = X.shape[1]
    r = 1.8
    a = 2
    cof = np.log(r)*x1 + np.log(r)*x2 + np.log(r)*x4 + np.log(2.3)*x5**2 + np.log(1.5)* x_h1.sum(axis=1) + np.log(1.1) * x_h2.sum(axis=1) - 5.673 - 1.*z
    T = (-np.log(np.random.uniform(0, 1, n)) / np.exp(cof)) **(1/a) * 100

    cof_counter = np.log(r)*x1 + np.log(r)*x2 + np.log(r)*x4 + np.log(2.3)*x5**2 + np.log(1.5)* x_h1.sum(axis=1) + np.log(1.1) * x_h2.sum(axis=1) - 5.673 - 1.*(1-z)
    T_counter = (-np.log(np.random.uniform(0, 1, n)) / np.exp(cof_counter)) **(1/a) * 100

    Tend = 200
    T_both = np.concatenate((z, 1-z))
    Y_both = np.concatenate((np.stack((T <= Tend, T), axis=1), np.stack((T_counter <= Tend, T_counter), axis=1)))

    HR_ori, CI_ori, cph_ori = cox_no_weight(T_both, Y_both)
    print('HR {} ({}) p:{}'.format(HR_ori, CI_ori, cph_ori.summary.p.treatment))
    print('np.exp(-1)', np.exp(-1))

    """
    Strong nolinear effect!
HR 0.5788475575804051 ([0.5771662  0.58053381]) p:0.0
np.exp(-1) 0.36787944117144233

Moderate nolinear effect!
HR 0.5787255780477224 ([0.5770445  0.58041155]) p:0.0
np.exp(-1) 0.36787944117144233
Done! Total Time used: 00:02:48

no nolinear effect!
HR 0.5781315363886127 ([0.57645202 0.57981595]) p:0.0
np.exp(-1) 0.36787944117144233

    """


    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

