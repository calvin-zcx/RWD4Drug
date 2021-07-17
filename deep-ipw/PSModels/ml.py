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

            T_train_predict = model.predict_proba(X_train)[:, 1]
            auc_train = roc_auc_score(T_train, T_train_predict)
            loss_train = log_loss(T_train, T_train_predict)

            self.results.append((para_d, loss_train, auc_train, auc_val))  # model,  not saving model for less disk
            if auc_val > self.best_val:
                self.best_model = model
                self.best_hyper_paras = para_d
                self.best_val = auc_val

        if verbose:
            self.report_stats()

        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        return self

    def report_stats(self):
        print('Model {} Searching Space N={}: '.format(self.learner, len(self.paras_list)), self.paras_grid)
        print('Best model: ', self.best_model)
        print('Best configuration: ', self.best_hyper_paras)
        print('Best value: ', self.best_val)
        print('AUC stats:\n', pd.DataFrame(self.results,
                                           columns=['paras', 'train_loss', 'train_auc', 'validation_auc']).describe())

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
