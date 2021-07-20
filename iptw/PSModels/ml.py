import sys

# for linux env.
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm, tree
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import itertools
from sklearn.metrics import roc_auc_score
import time
from sklearn.metrics import log_loss
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
from iptw.evaluation import cal_deviation, SMD_THRESHOLD


class PropensityEstimator:
    def __init__(self, learner, paras_grid=None):
        self.learner = learner
        assert self.learner in ('LR', 'XGBOOST', 'LIGHTGBM')

        if (paras_grid is None) or (not paras_grid) or (not isinstance(paras_grid, dict)):
            self.paras_grid = {}
        else:
            self.paras_grid = {k: v for k, v in paras_grid.items()}
            for k, v in self.paras_grid.items():
                if isinstance(v, str):
                    print(k, v, 'is a fixed parameter')
                    self.paras_grid[k] = [v, ]

        if self.paras_grid:
            paras_names, paras_v = zip(*self.paras_grid.items())
            paras_list = list(itertools.product(*paras_v))
            self.paras_names = paras_names
            self.paras_list = [{self.paras_names[i]: para[i] for i in range(len(para))} for para in paras_list]
        else:
            self.paras_names = []
            self.paras_list = [{}]

        self.best_hyper_paras = None
        self.best_model = None

        self.best_val = float('-inf')
        self.best_balance = float('inf')

        self.global_best_val = float('-inf')
        self.global_best_balance = float('inf')

        self.results = []

    def fit(self, X_train, T_train, X_val, T_val, verbose=1):
        start_time = time.time()
        if verbose:
            print('Model {} Searching Space N={}: '.format(self.learner, len(self.paras_list)), self.paras_grid)

        for para_d in tqdm(self.paras_list):
            if self.learner == 'LR':
                if para_d.get('penalty', '') == 'l1':
                    para_d['solver'] = 'liblinear'
                else:
                    para_d['solver'] = 'lbfgs'
                model = LogisticRegression(**para_d).fit(X_train, T_train)
            elif self.learner == 'XGBOOST':
                model = xgb.XGBClassifier(**para_d).fit(X_train, T_train)
            elif self.learner == 'LIGHTGBM':
                model = lgb.LGBMClassifier(**para_d).fit(X_train, T_train)
            else:
                raise ValueError

            T_val_predict = model.predict_proba(X_val)[:, 1]
            auc_val = roc_auc_score(T_val, T_val_predict)
            max_smd, smd, max_smd_weighted, smd_w = cal_deviation(X_val, T_val, T_val_predict,
                                                                  normalized=True, verbose=False)
            n_unbalanced_feature = len(np.where(smd > SMD_THRESHOLD)[0])
            n_unbalanced_feature_w = len(np.where(smd_w > SMD_THRESHOLD)[0])

            T_train_predict = model.predict_proba(X_train)[:, 1]
            auc_train = roc_auc_score(T_train, T_train_predict)
            loss_train = log_loss(T_train, T_train_predict)

            self.results.append((para_d, loss_train, auc_train, auc_val,
                                 max_smd, n_unbalanced_feature, max_smd_weighted,
                                 n_unbalanced_feature_w))  # model,  not saving model for less disk

            if (max_smd_weighted <= self.best_balance): # and (auc_val >= self.best_val):  #
                self.best_model = model
                self.best_hyper_paras = para_d
                self.best_val = auc_val
                self.best_balance = max_smd_weighted

            if auc_val > self.global_best_val:
                self.global_best_val = auc_val

            if max_smd_weighted <= self.global_best_balance:
                self.global_best_balance = max_smd_weighted

        self.results = pd.DataFrame(self.results, columns=['paras', 'train_loss', 'train_auc', 'validation_auc',
                                                           "max_smd", "n_unbalanced_feature", "max_smd_weighted",
                                                           "n_unbalanced_feature_w"])
        if verbose:
            self.report_stats()

        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        return self

    def fit_godview(self, X_train, T_train, X_val, T_val, X_all, T_all, verbose=1):
        start_time = time.time()
        if verbose:
            print('Model {} Searching Space N={}: '.format(self.learner, len(self.paras_list)), self.paras_grid)

        for para_d in tqdm(self.paras_list):
            if self.learner == 'LR':
                if para_d.get('penalty', '') == 'l1':
                    para_d['solver'] = 'liblinear'
                else:
                    para_d['solver'] = 'lbfgs'
                model = LogisticRegression(**para_d).fit(X_train, T_train)
            elif self.learner == 'XGBOOST':
                model = xgb.XGBClassifier(**para_d).fit(X_train, T_train)
            elif self.learner == 'LIGHTGBM':
                model = lgb.LGBMClassifier(**para_d).fit(X_train, T_train)
            else:
                raise ValueError

            # validation data
            T_val_predict = model.predict_proba(X_val)[:, 1]
            auc_val = roc_auc_score(T_val, T_val_predict)
            max_smd, smd, max_smd_weighted, smd_w = cal_deviation(X_val, T_val, T_val_predict, normalized=True, verbose=False)
            n_unbalanced_feature = len(np.where(smd > SMD_THRESHOLD)[0])
            n_unbalanced_feature_w = len(np.where(smd_w > SMD_THRESHOLD)[0])

            # train data
            T_train_predict = model.predict_proba(X_train)[:, 1]
            auc_train = roc_auc_score(T_train, T_train_predict)
            loss_train = log_loss(T_train, T_train_predict)
            max_smd_train, smd_train, max_smd_weighted_train, smd_w_train = cal_deviation(X_train, T_train, T_train_predict, normalized=True, verbose=False)
            n_unbalanced_feature_train = len(np.where(smd_train > SMD_THRESHOLD)[0])
            n_unbalanced_feature_w_train = len(np.where(smd_w_train > SMD_THRESHOLD)[0])

            # all data
            T_all_predict = model.predict_proba(X_all)[:, 1]
            auc_all = roc_auc_score(T_all, T_all_predict)
            max_smd_all, smd_all, max_smd_weighted_all, smd_w_all = cal_deviation(X_all, T_all, T_all_predict, normalized=True, verbose=False)
            n_unbalanced_feature_all = len(np.where(smd_all > SMD_THRESHOLD)[0])
            n_unbalanced_feature_w_all = len(np.where(smd_w_all > SMD_THRESHOLD)[0])

            self.results.append((para_d, loss_train,
                                 auc_train, max_smd_train, n_unbalanced_feature_train, max_smd_weighted_train, n_unbalanced_feature_w_train,
                                 auc_val, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_w,
                                 auc_all, max_smd_all, n_unbalanced_feature_all, max_smd_weighted_all,  n_unbalanced_feature_w_all))  # model,  not saving model for less disk

            if (max_smd_weighted <= self.best_balance):  # and (auc_val >= self.best_val):  #
                self.best_model = model
                self.best_hyper_paras = para_d
                self.best_val = auc_val
                self.best_balance = max_smd_weighted

            if auc_val > self.global_best_val:
                self.global_best_val = auc_val

            if max_smd_weighted <= self.global_best_balance:
                self.global_best_balance = max_smd_weighted

        self.results = pd.DataFrame(self.results, columns=['paras', 'train_loss',
                                                           'train_auc',
                                                           "max_smd_train", "n_unbalanced_feature_train",
                                                           "max_smd_weighted_train", "n_unbalanced_feature_w_train",
                                                           'validation_auc',
                                                           "max_smd_val", "n_unbalanced_feature_val",
                                                           "max_smd_weighted_val", "n_unbalanced_feature_w_val",
                                                           'auc_all',
                                                           "max_smd_all", "n_unbalanced_feature_all",
                                                           "max_smd_weighted_all", "n_unbalanced_feature_w_all"])
        if verbose:
            self.report_stats()

        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        return self

    def report_stats(self):
        print('Model {} Searching Space N={}: '.format(self.learner, len(self.paras_list)), self.paras_grid)
        print('Best model: ', self.best_model)
        print('Best configuration: ', self.best_hyper_paras)
        print('Best fit value ', self.best_val, ' Global Best fit balue: ', self.global_best_val)
        print('Best balance: ', self.best_balance, ' Global Best balance: ', self.global_best_balance)
        pd.set_option('display.max_columns', None)
        describe = self.results.describe()
        print('AUC stats:\n', describe)
        return describe

    def predict_ps(self, X):
        pred_ps = self.best_model.predict_proba(X)[:, 1]
        # pred_clip_propensity = np.clip(pred_propensity, a_min=np.quantile(pred_propensity, 0.1), a_max=np.quantile(pred_propensity, 0.9))
        return pred_ps

    def predict_loss(self, X, T):
        T_pre = self.predict_ps(X)
        return log_loss(T, T_pre)

# class OutcomeEstimator:
#     def __init__(self, learner, x_input, outcome, sample_weights=None):
#         if learner == 'Logistic-regression':
#             self.learner = LogisticRegression(solver='liblinear', penalty='l2', C=1).fit(x_input, outcome, sample_weight=sample_weights)
#         elif learner == 'SGD':
#             self.learner = SGDClassifier(loss='log').fit(x_input, outcome)
#         elif learner == 'AdaBoost':
#             self.learner = AdaBoostClassifier().fit(x_input, outcome)
#
#     def predict_outcome(self, x_input):
#         pred_outcome = self.learner.predict_proba(x_input)[:, 1]
#         return pred_outcome


# class PropensityEstimator:
#     def __init__(self, learner, confounder, treatment):
#         if learner == 'Logistic-regression':
#             self.learner = LogisticRegression(solver='liblinear', penalty='l2', C=1).fit(confounder, treatment)
#         elif learner == 'SVM':
#             self.learner = svm.SVC().fit(confounder, treatment)
#         elif learner == 'CART':
#             self.learner = tree.DecisionTreeClassifier(max_depth=6).fit(confounder, treatment)
#
#     def compute_weights(self, confounder):
#         pred_propensity = self.learner.predict_proba(confounder)[:, 1]
#         # pred_clip_propensity = np.clip(pred_propensity, a_min=np.quantile(pred_propensity, 0.1), a_max=np.quantile(pred_propensity, 0.9))
#         # inverse_propensity = 1. / pred_propensity
#         return pred_propensity
