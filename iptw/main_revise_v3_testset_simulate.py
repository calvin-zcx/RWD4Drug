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
    parser.add_argument("--nsim", type=int, default=4000)

    parser.add_argument('--covspecs', choices=['correct', 'partial', 'incorrect'], default='correct')
    parser.add_argument('--nonlin', choices=['no', 'moderate', 'strong'], default='strong')

    parser.add_argument('--run_model', choices=['LSTM', 'LR', 'MLP', 'XGBOOST', 'LIGHTGBM'], default='LR')
    parser.add_argument('--med_code_topk', type=int, default=200)
    parser.add_argument('--drug_coding', choices=['rxnorm', 'gpi'], default='rxnorm')
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--stats_exit', action='store_true')
    # Deep PSModels
    parser.add_argument('--batch_size', type=int, default=256)  # 768)  # 64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # 0.001
    parser.add_argument('--weight_decay', type=float, default=1e-6)  # )0001)
    parser.add_argument('--epochs', type=int, default=3)  # 15 #30
    # LSTM
    parser.add_argument('--diag_emb_size', type=int, default=128)
    parser.add_argument('--med_emb_size', type=int, default=128)
    parser.add_argument('--med_hidden_size', type=int, default=64)
    parser.add_argument('--diag_hidden_size', type=int, default=64)
    parser.add_argument('--lstm_hidden_size', type=int, default=100)
    # MLP
    parser.add_argument('--hidden_size', type=str, default='', help=', delimited integers')
    # Output
    parser.add_argument('--output_dir', type=str, default='output/simulate_v3/')

    # discarded
    # parser.add_argument('--save_db', type=str)
    # parser.add_argument('--outcome', choices=['bool', 'time'], default='bool')
    # parser.add_argument('--pickles_dir', type=str, default='pickles/')
    # parser.add_argument('--hidden_size', type=int, default=100)
    # parser.add_argument('--save_model_filename', type=str, default='tmp/1346823.pt')
    args = parser.parse_args()

    # Modifying args
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.random_seed >= 0:
        rseed = args.random_seed
    else:
        from datetime import datetime
        rseed = datetime.now()

    args.treated_drug = args.treated_drug + 'n{}train{:.1f}{}{}'.format(args.nsim, args.train_ratio, args.nonlin, args.covspecs)
    args.random_seed = rseed
    args.save_model_filename = os.path.join(args.output_dir, args.treated_drug,
                                            args.treated_drug + '_S{}D{}C{}_{}'.format(args.random_seed,
                                                                                       args.med_code_topk,
                                                                                       args.controlled_drug,
                                                                                       args.run_model))
    check_and_mkdir(args.save_model_filename)

    args.hidden_size = [int(x.strip()) for x in args.hidden_size.split(',')
                        if (x.strip() not in ('', '0'))]
    if args.med_code_topk < 1:
        args.med_code_topk = None

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
        print('...all:', x_sample.shape, 'non-zero ratio:', (x_sample != 0).mean(), 'all-zero:',
              (x_sample.mean(0) == 0).sum())
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
    print('save_model_filename', args.save_model_filename)

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
        print('No nolinear effect!')
        logits_wo_bias = (
                np.log(2) * x2 + np.log(3) * x3 + np.log(2) * x5 + np.log(2) * x6 + np.log(1.5) * x_h1.sum(axis=1)
                + np.log(1.1) * x_h2.sum(axis=1))
        if args.covspecs == 'incorrect':
            X = np.stack((x1 ** 2, x2 ** 2, x3 ** 2, x4, x5, x6), axis=1)
            X = np.concatenate((X, x_h1, x_h2), axis=1)
        elif args.covspecs == 'partial':
            X = np.stack((x1, x2, x3, x4, x5, x6), axis=1)
            X = np.concatenate((X, x_h1, x_h2), axis=1)
        elif args.covspecs == 'correct':
            X = np.stack((x2, x3, x5, x6), axis=1)
            X = np.concatenate((X, x_h1, x_h2), axis=1)
        else:
            raise ValueError

    elif args.nonlin == 'moderate':
        print('Moderate nolinear effect!')
        logits_wo_bias = (x1 * (np.log(2) * x2 ** 2 + np.log(3) * x3 * x2 + np.log(2) * x5 + np.log(2) * x6 + np.log(
            1.5) * x_h1.sum(axis=1)) + np.log(1.1) * x_h2.sum(axis=1))
        if args.covspecs == 'incorrect':
            X = np.stack((x1, x2, x3, x4, x5, x6), axis=1)
            X = np.concatenate((X, x_h1, x_h2), axis=1)
        elif args.covspecs == 'partial':
            X = np.stack((x1 * x2, x1 * x3, x1 * x5, x1 * x6), axis=1)
            X = np.concatenate((X, x_h1, x_h2), axis=1)
        elif args.covspecs == 'correct':
            X = np.stack((x1 * (x2 ** 2), x1 * x3 * x2, x1 * x5, x1 * x6), axis=1)
            X = np.concatenate((X, np.expand_dims(x1, axis=1) * x_h1, x_h2), axis=1)
        else:
            raise ValueError
    elif args.nonlin == 'strong':
        print('Strong nolinear effect!')
        logits_wo_bias = x1 * (np.log(2) * x2 ** 2 + np.log(3) * x3 * x2 + np.log(2) * x5 + np.log(2) * x6 +
                               np.log(1.5) * x_h1.sum(axis=1) + np.log(1.1) * x_h2.sum(axis=1))
        if args.covspecs == 'incorrect':
            X = np.stack((x1, x2, x3, x4, x5, x6), axis=1)
            X = np.concatenate((X, x_h1, x_h2), axis=1)
        elif args.covspecs == 'partial':
            X = np.stack((x1 * x2, x1 * x3, x1 * x5, x1 * x6), axis=1)
            X = np.concatenate((X, x_h1, x_h2), axis=1)
        elif args.covspecs == 'correct':
            X = np.stack((x1 * (x2 ** 2), x1 * x3 * x2, x1 * x5, x1 * x6), axis=1)
            X = np.concatenate((X, np.expand_dims(x1, axis=1) * x_h1, np.expand_dims(x1, axis=1) * x_h2), axis=1)
        else:
            raise ValueError
    else:
        raise ValueError

    target_logits_mean = -0.15  # -0.013648586697455571
    bias = target_logits_mean - logits_wo_bias.mean()
    logits = bias + logits_wo_bias

    print('target_logits_mean:', target_logits_mean, 'logits.mean()', logits.mean(),
          '\nbias:', bias, 'logits_wo_bias.mean():', logits_wo_bias.mean(), )

    z = np.random.binomial(1, p=(1 / (1 + np.exp(-logits))))

    print('covspecs:', args.covspecs)

    n_feature = X.shape[1]
    r = 1.8
    a = 2
    cof = np.log(r) * x1 + np.log(r) * x2 + np.log(r) * x4 + np.log(2.3) * x5 ** 2 + np.log(1.5) * x_h1.sum(
        axis=1) + np.log(1.1) * x_h2.sum(axis=1) - 5.673 - 1. * z
    T = (-np.log(np.random.uniform(0, 1, n)) / np.exp(cof)) ** (1 / a) * 100

    Tend = 200  # np.inf #200
    print('Tend:', Tend)

    Y = T <= Tend
    T_censor = np.copy(T)
    T_censor[T > Tend] = Tend

    X1 = X[z == 1]
    X0 = X[z == 0]
    smd = smd_func(X1, np.ones((len(X1), 1)), X0, np.ones((len(X0), 1)), abs=True)
    # print('smd', smd)
    # plot survival time
    data_debug = pd.DataFrame(
        data={'treatment': z, 't2e': T})
    if os.name != 'posix':
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sns.histplot(ax=axes[0], data=data_debug, x='t2e', hue='treatment', kde=True, stat="probability",
                     common_norm=False)
        axes[0].set_ylabel('T2E Probability')  # label the y axis
        axes[0].set_xlabel('Time-2-Event')

        sns.ecdfplot(ax=axes[1], data=data_debug, x='t2e', hue='treatment', complementary=True)
        axes[1].set_ylabel('Survival Probability')  # label the y axis
        axes[1].set_xlabel('Time-2-Event')

        sns.ecdfplot(ax=axes[2], data=data_debug, x='t2e', hue='treatment', complementary=False)
        axes[2].set_ylabel('Cumulative Incidence')  # label the y axis
        axes[2].set_xlabel('Time-2-Event')

        for ax in axes:
            ax.set_xlim(-1, Tend + 20)
        plt.tight_layout()
        check_and_mkdir(args.output_dir + 'results/fig/')
        fig.savefig(args.output_dir + 'results/fig/simulate_data_describe-{}.png'.format(args.nonlin))
        fig.savefig(args.output_dir + 'results/fig/simulate_data_describe-{}.pdf'.format(args.nonlin))
        print('done')
        plt.show()

    # zz
    # sys.exit(0)
    # train_ratio = 0.8  # 0.5
    train_ratio = args.train_ratio  # default 0.8
    print('train_ratio: ', train_ratio,
          'test_ratio: ', 1 - train_ratio)

    my_dataset = X
    dataset_size = len(my_dataset)
    indices = list(range(dataset_size))
    train_index = int(np.floor(train_ratio * dataset_size))
    # val_index = int(np.floor(val_ratio * dataset_size))
    np.random.shuffle(indices)

    train_indices, test_indices = indices[:train_index], indices[train_index:]

    # %% Logistic regression PS PSModels
    if args.run_model in ['LR', 'XGBOOST', 'LIGHTGBM']:
        print("**************************************************")
        print(args.run_model, ' PS model learning:')

        print('Train data:')
        train_x, train_t, train_y = flatten_data_sim(X, z, Y, T_censor, train_indices, verbose=1)
        # train_x, train_t, train_y = flatten_data(my_dataset, train_indices)
        # print('Validation data:')
        # val_x, val_t, val_y = flatten_data(my_dataset, val_indices)
        print('Test data:')
        test_x, test_t, test_y = flatten_data_sim(X, z, Y, T_censor, test_indices, verbose=1)
        # test_x, test_t, test_y = flatten_data(my_dataset, test_indices)
        print('All data:')
        x, t, y = flatten_data_sim(X, z, Y, T_censor, indices,
                                   verbose=1)  # flatten_data(my_dataset, indices)  # all the data

        # put fixed parameters also into a list e.g. 'objective' : ['binary',]
        if args.run_model == 'LR':
            paras_grid = {
                'penalty': ['l1', 'l2'],
                'C': 10 ** np.arange(-3, 3, 0.5),
                # 0.2),  # 'C': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20],
                'max_iter': [200],  # [100, 200, 500],
                'random_state': [args.random_seed],
            }
            #
            # paras_grid = {
            #     'penalty': ['l1',],
            #     'C': [1,],
            #     # 0.2),  # 'C': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20],
            #     'max_iter': [200],  # [100, 200, 500],
            #     'random_state': [args.random_seed],
            # }
        elif args.run_model == 'XGBOOST':
            paras_grid = {
                'max_depth': [3, 4],
                'min_child_weight': np.linspace(0, 1, 5),
                'learning_rate': np.arange(0.01, 1, 0.1),
                'colsample_bytree': np.linspace(0.05, 1, 5),
                'random_state': [args.random_seed],
            }
        elif args.run_model == 'LIGHTGBM':
            paras_grid = {
                'max_depth': [3, 4, 5],
                'learning_rate': np.arange(0.01, 1, 0.25),
                'num_leaves': np.arange(10, 120, 30),
                'min_child_samples': [200, 250, 300],
                'random_state': [args.random_seed],
            }
        else:
            paras_grid = {}

        # ----2. Learning IPW using PropensityEstimator
        # model = ml.PropensityEstimator(args.run_model, paras_grid).fit(train_x, train_t, val_x, val_t)
        # model = ml.PropensityEstimator(args.run_model, paras_grid).fit_and_test(train_x, train_t, val_x, val_t, test_x,
        #                                                                         test_t)

        model = ml.PropensityEstimator(
            args.run_model, paras_grid, random_seed=args.random_seed).cross_validation_fit_withtestset_witheffect(
            train_x, train_t, train_y, test_x, test_t, test_y, verbose=1)

        # with open(args.save_model_filename, 'wb') as f:
        #     pickle.dump(model, f)

        model.results.to_csv(args.save_model_filename + '_ALL-model-select.csv')
        model.results_agg.to_csv(args.save_model_filename + '_ALL-model-select-agg.csv')
        # ----3. Evaluation learned PropensityEstimator
        # results_all_list, results_all_df = final_eval_ml(model, args, train_x, train_t, train_y, val_x, val_t, val_y,
        #                                                  test_x, test_t, test_y, x, t, y,
        #                                                  drug_name, feature_name, n_feature, dump_ori=False)
        results_all_list, results_all_df = final_eval_ml_CV_revise_traintest(
            model, args, train_x, train_t, train_y, test_x, test_t, test_y, x, t, y,
            {'simu': 'simu', args.treated_drug: args.treated_drug}, np.array([str(i + 1) for i in range(n_feature)]),
            n_feature, dump_ori=False)

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

# if __name__ == "__main__":
#     start_time = time.time()
#     main(args=parse_args())
#     print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
# from line_profiler import LineProfiler
#
# lprofiler = LineProfiler()
# lprofiler.add_function(build_patient_characteristics_from_triples)
# lprofiler.add_function(statistics_for_treated_control)
# lp_wrapper = lprofiler(main_func)
#
# lp_wrapper()
# lprofiler.print_stats()
