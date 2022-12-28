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
from sklearn.model_selection import KFold
import time
from sklearn.metrics import log_loss
from tqdm import tqdm
# import xgboost as xgb
import lightgbm as lgb
from iptw.evaluation import cal_deviation, SMD_THRESHOLD, cal_weights


class PropensityEstimator:
    def __init__(self, learner, paras_grid=None, random_seed=0):
        self.learner = learner
        self.random_seed = random_seed
        assert self.learner in ('LR', 'XGBOOST', 'LIGHTGBM')

        if (paras_grid is None) or (not paras_grid) or (not isinstance(paras_grid, dict)):
            self.paras_grid = {}
        else:
            self.paras_grid = {k: v for k, v in paras_grid.items()}
            for k, v in self.paras_grid.items():
                if isinstance(v, str) or not isinstance(v, (list, set, np.ndarray, pd.Series)):
                    print(k, v, 'is a fixed parameter')
                    self.paras_grid[k] = [v, ]

        if self.paras_grid:
            paras_names, paras_v = zip(*self.paras_grid.items())
            paras_list = list(itertools.product(*paras_v))
            self.paras_names = paras_names
            self.paras_list = [{self.paras_names[i]: para[i] for i in range(len(para))} for para in paras_list]
            if self.learner == 'LR':
                no_penalty_case = {'penalty': 'none', 'max_iter': 200, 'random_state': random_seed}
                if (no_penalty_case not in self.paras_list) and (len(self.paras_list) > 1):
                    self.paras_list.append(no_penalty_case)
                    # self.paras_list = [no_penalty_case,] + self.paras_list # debug
                    print('Add no penalty case to logistic regression model:', no_penalty_case)
        else:
            self.paras_names = []
            self.paras_list = [{}]

        self.best_hyper_paras = None
        self.best_model = None

        self.best_val = float('-inf')
        self.best_balance = float('inf')

        self.global_best_val = float('-inf')
        self.global_best_balance = float('inf')

        self.best_balance_k_folds_detail = []  # k #(SMD>threshold)
        self.best_val_k_folds_detail = []  # k AUC

        self.results = []
        self.results_retrain = []
        self.results_agg = []

    @staticmethod
    def _evaluation_helper(X, T, T_pre):
        loss = log_loss(T, T_pre)
        auc = roc_auc_score(T, T_pre)
        max_smd, smd, max_smd_weighted, smd_w = cal_deviation(X, T, T_pre, normalized=True, verbose=False)
        n_unbalanced_feature = len(np.where(smd > SMD_THRESHOLD)[0])
        n_unbalanced_feature_weighted = len(np.where(smd_w > SMD_THRESHOLD)[0])
        result = (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
        return result

    def fit(self, X_train, T_train, X_val, T_val, verbose=1):
        start_time = time.time()
        if verbose:
            print('Model {} Searching Space N={}: '.format(self.learner, len(self.paras_list)), self.paras_grid)
        i = -1
        for para_d in tqdm(self.paras_list):
            i += 1
            if self.learner == 'LR':
                if para_d.get('penalty', '') == 'l1':
                    para_d['solver'] = 'liblinear'
                else:
                    para_d['solver'] = 'lbfgs'
                model = LogisticRegression(**para_d).fit(X_train, T_train)
            # elif self.learner == 'XGBOOST':
            #     model = xgb.XGBClassifier(**para_d).fit(X_train, T_train)
            elif self.learner == 'LIGHTGBM':
                model = lgb.LGBMClassifier(**para_d).fit(X_train, T_train)
            else:
                raise ValueError

            T_train_pre = model.predict_proba(X_train)[:, 1]
            T_val_pre = model.predict_proba(X_val)[:, 1]

            result_train = self._evaluation_helper(X_train, T_train, T_train_pre)
            result_val = self._evaluation_helper(X_val, T_val, T_val_pre)

            result_trainval = self._evaluation_helper(
                np.concatenate((X_train, X_val)),
                np.concatenate((T_train, T_val)),
                np.concatenate((T_train_pre, T_val_pre))
            )

            self.results.append((i, para_d) + result_train + result_val + result_trainval)

            if (result_trainval[5] < self.best_balance) or \
                    ((result_trainval[5] == self.best_balance) and (result_val[1] > self.best_val)):
                self.best_model = model
                self.best_hyper_paras = para_d
                self.best_val = result_val[1]
                self.best_balance = result_trainval[5]

            if result_val[1] > self.global_best_val:
                self.global_best_val = result_val[1]

            if result_trainval[5] <= self.global_best_balance:
                self.global_best_balance = result_trainval[5]

        name = ['loss', 'auc', 'max_smd', 'n_unbalanced_feat', 'max_smd_iptw', 'n_unbalanced_feat_iptw']
        col_name = ['i', 'paras'] + [pre + x for pre in ['train_', 'val_', 'trainval_'] for x in name]
        self.results = pd.DataFrame(self.results, columns=col_name)

        if verbose:
            self.report_stats()
        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        return self

    def fit_and_test(self, X_train, T_train, X_val, T_val, X_test, T_test, verbose=1):
        start_time = time.time()
        if verbose:
            print('Model {} Searching Space N={}: '.format(self.learner, len(self.paras_list)), self.paras_grid)
        i = -1
        for para_d in tqdm(self.paras_list):
            i += 1
            if self.learner == 'LR':
                if para_d.get('penalty', '') == 'l1':
                    para_d['solver'] = 'liblinear'
                else:
                    para_d['solver'] = 'lbfgs'
                model = LogisticRegression(**para_d).fit(X_train, T_train)
            # elif self.learner == 'XGBOOST':
            #     model = xgb.XGBClassifier(**para_d).fit(X_train, T_train)
            elif self.learner == 'LIGHTGBM':
                model = lgb.LGBMClassifier(**para_d).fit(X_train, T_train)
            else:
                raise ValueError

            T_train_pre = model.predict_proba(X_train)[:, 1]
            T_val_pre = model.predict_proba(X_val)[:, 1]
            T_test_pre = model.predict_proba(X_test)[:, 1]

            result_train = self._evaluation_helper(X_train, T_train, T_train_pre)
            result_val = self._evaluation_helper(X_val, T_val, T_val_pre)
            result_test = self._evaluation_helper(X_test, T_test, T_test_pre)

            result_trainval = self._evaluation_helper(
                np.concatenate((X_train, X_val)),
                np.concatenate((T_train, T_val)),
                np.concatenate((T_train_pre, T_val_pre))
            )

            result_all = self._evaluation_helper(
                np.concatenate((X_train, X_val, X_test)),
                np.concatenate((T_train, T_val, T_test)),
                np.concatenate((T_train_pre, T_val_pre, T_test_pre))
            )

            self.results.append((i, para_d) + result_train + result_val + result_test + result_trainval + result_all)

            if (result_trainval[5] < self.best_balance) or \
                    ((result_trainval[5] == self.best_balance) and (result_val[1] > self.best_val)):
                self.best_model = model
                self.best_hyper_paras = para_d
                self.best_val = result_val[1]
                self.best_balance = result_trainval[5]

            if result_val[1] > self.global_best_val:
                self.global_best_val = result_val[1]

            if result_trainval[5] <= self.global_best_balance:
                self.global_best_balance = result_trainval[5]

        name = ['loss', 'auc', 'max_smd', 'n_unbalanced_feat', 'max_smd_iptw', 'n_unbalanced_feat_iptw']
        col_name = ['i', 'paras'] + [pre + x for pre in ['train_', 'val_', 'test_', 'trainval_', 'all_'] for x in name]
        self.results = pd.DataFrame(self.results, columns=col_name)

        if verbose:
            self.report_stats()
        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        return self

    def _model_estimation(self, para_d, X_train, T_train):
        # model estimation on training data
        if self.learner == 'LR':
            if para_d.get('penalty', '') == 'l1':
                para_d['solver'] = 'liblinear'
            else:
                para_d['solver'] = 'lbfgs'
            model = LogisticRegression(**para_d).fit(X_train, T_train)
        elif self.learner == 'LIGHTGBM':
            model = lgb.LGBMClassifier(**para_d).fit(X_train, T_train)
        # elif learner == 'SVM':
        #   model = svm.SVC().fit(confounder, treatment)
        # elif learner == 'CART':
        #   model = tree.DecisionTreeClassifier(max_depth=6).fit(confounder, treatment)
        else:
            raise ValueError

        return model

    def cross_validation_fit(self, X, T, kfold=10, verbose=1, shuffle=True):
        start_time = time.time()
        kf = KFold(n_splits=kfold, random_state=self.random_seed, shuffle=shuffle)
        if verbose:
            print('Model {} Searching Space N={} by '
                  '{}-k-fold cross validation: '.format(self.learner,
                                                        len(self.paras_list),
                                                        kf.get_n_splits()), self.paras_grid)
        # For each model in model space, do cross-validation training and testing,
        # performance of a model is average (std) over K cross-validated datasets
        # select best model with the best average K-cross-validated performance
        X = np.asarray(X)
        T = np.asarray(T)
        for i, para_d in tqdm(enumerate(self.paras_list, 1), total=len(self.paras_list)):
            i_model_balance_over_kfold = []
            i_model_fit_over_kfold = []
            for k, (train_index, test_index) in enumerate(kf.split(X), 1):
                print('Training {}th (/{}) model {} over the {}th-fold data'.format(i, len(self.paras_list), para_d, k))
                # training and testing datasets:
                X_train = X[train_index, :]
                T_train = T[train_index]
                X_test = X[test_index, :]
                T_test = T[test_index]

                # model estimation on training data
                model = self._model_estimation(para_d, X_train, T_train)

                # propensity scores on training and testing datasets
                T_train_pre = model.predict_proba(X_train)[:, 1]
                T_test_pre = model.predict_proba(X_test)[:, 1]

                # evaluating goodness-of-balance and goodness-of-fit
                result_train = self._evaluation_helper(X_train, T_train, T_train_pre)
                result_test = self._evaluation_helper(X_test, T_test, T_test_pre)
                result_all = self._evaluation_helper(
                    np.concatenate((X_train, X_test)),
                    np.concatenate((T_train, T_test)),
                    np.concatenate((T_train_pre, T_test_pre))
                )# (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
                i_model_balance_over_kfold.append(result_all[5])
                i_model_fit_over_kfold.append(result_test[1])

                self.results.append((i, k, para_d) + result_train + result_test + result_all)
                # end of one fold

            i_model_balance = [np.mean(i_model_balance_over_kfold), np.std(i_model_balance_over_kfold)]
            i_model_fit = [np.mean(i_model_fit_over_kfold), np.std(i_model_fit_over_kfold)]

            if (i_model_balance[0] < self.best_balance) or \
                    ((i_model_balance[0] == self.best_balance) and (i_model_fit[0] > self.best_val)):
                # model with current best configuration re-trained on the whole dataset.
                # self.best_model = self._model_estimation(para_d, X, T)
                self.best_hyper_paras = para_d
                self.best_balance = i_model_balance[0]
                self.best_val = i_model_fit[0]
                self.best_balance_k_folds_detail = i_model_balance_over_kfold
                self.best_val_k_folds_detail = i_model_fit_over_kfold

            if i_model_fit[0] > self.global_best_val:
                self.global_best_val = i_model_fit[0]

            if i_model_balance[0] < self.global_best_balance:
                self.global_best_balance = i_model_balance[0]

            # save re-trained results on the whole data, for model selection exp only. Not necessary for later use
            model_retrain = self._model_estimation(para_d, X, T)
            T_pre = model_retrain.predict_proba(X)[:, 1]
            result_retrain = self._evaluation_helper(X, T, T_pre)
            self.results_retrain.append((i, 'retrain', para_d) + result_retrain)

            if verbose:
                self.report_stats()

        # end of training
        print('best model parameter:', self.best_hyper_paras)
        print('re-training best model on all the data using best model parameter...')
        self.best_model = self._model_estimation(self.best_hyper_paras, X, T)
        name = ['loss', 'auc', 'max_smd', 'n_unbalanced_feat', 'max_smd_iptw', 'n_unbalanced_feat_iptw']
        col_name = ['i', 'fold-k', 'paras'] + [pre + x for pre in ['train_', 'val_', 'beforRetrain all_'] for x in name]
        self.results = pd.DataFrame(self.results, columns=col_name)
        self.results['paras_str'] = self.results['paras'].apply(lambda x: str(x))

        col_name_retrain = ['i', 'fold-k', 'paras'] + [pre + x for pre in ['all_'] for x in name]
        self.results_retrain = pd.DataFrame(self.results_retrain, columns=col_name_retrain)
        self.results_retrain['paras_str'] = self.results_retrain['paras'].apply(lambda x: str(x))

        results_agg = self.results.groupby('paras_str').agg(['mean', 'std']).reset_index().sort_values(by=[('i', 'mean')])
        results_agg.columns = results_agg.columns.to_flat_index()
        results_agg.columns = results_agg.columns.map('-'.join)
        self.results_agg = pd.merge(results_agg, self.results_retrain, left_on='paras_str-', right_on='paras_str', how='left')

        if verbose:
            self.report_stats()
        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        return self

    def cross_validation_fit_withtestset(self, X, T, X_test, T_test, kfold=10, verbose=1, shuffle=True):
        """
        # CV model selection and training on X, T
        # out-of-sample test on the Xtest and Ttest
        :return:
        """

        start_time = time.time()
        kf = KFold(n_splits=kfold, random_state=self.random_seed, shuffle=shuffle)
        if verbose:
            print('Model {} Searching Space N={} by '
                  '{}-k-fold cross validation: '.format(self.learner,
                                                        len(self.paras_list),
                                                        kf.get_n_splits()), self.paras_grid)
        # For each model in model space, do cross-validation training and testing,
        # performance of a model is average (std) over K cross-validated datasets
        # select best model with the best average K-cross-validated performance
        X = np.asarray(X)  # as training set for cross-valiadtion into train and val
        T = np.asarray(T)  # as training set for cross-valiadtion into train and val
        # for out-of-sample test
        X_test = np.asarray(X_test)
        T_test = np.asarray(T_test)
        X_all = np.concatenate((X, X_test))
        T_all = np.concatenate((T, T_test))

        for i, para_d in tqdm(enumerate(self.paras_list, 1), total=len(self.paras_list)):
            i_model_balance_over_kfold = []
            i_model_fit_over_kfold = []
            for k, (train_index, val_index) in enumerate(kf.split(X), 1):
                print('Training {}th (/{}) model {} over the {}th-fold data'.format(i, len(self.paras_list), para_d, k))
                # training and testing datasets:
                X_train = X[train_index, :]
                T_train = T[train_index]
                X_val = X[val_index, :]
                T_val = T[val_index]

                # model estimation on training data
                model = self._model_estimation(para_d, X_train, T_train)

                # propensity scores on training and testing datasets
                T_train_pre = model.predict_proba(X_train)[:, 1]
                T_val_pre = model.predict_proba(X_val)[:, 1]

                # evaluating goodness-of-balance and goodness-of-fit
                result_train = self._evaluation_helper(X_train, T_train, T_train_pre)
                result_val = self._evaluation_helper(X_val, T_val, T_val_pre)
                result_trainval = self._evaluation_helper(
                    np.concatenate((X_train, X_val)),
                    np.concatenate((T_train, T_val)),
                    np.concatenate((T_train_pre, T_val_pre))
                )# (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
                i_model_balance_over_kfold.append(result_trainval[5])
                i_model_fit_over_kfold.append(result_val[1])

                self.results.append((i, k, para_d) + result_train + result_val + result_trainval)
                # end of one fold

            i_model_balance = [np.mean(i_model_balance_over_kfold), np.std(i_model_balance_over_kfold)]
            i_model_fit = [np.mean(i_model_fit_over_kfold), np.std(i_model_fit_over_kfold)]

            if (i_model_balance[0] < self.best_balance) or \
                    ((i_model_balance[0] == self.best_balance) and (i_model_fit[0] > self.best_val)):
                # model with current best configuration re-trained on the whole dataset.
                # self.best_model = self._model_estimation(para_d, X, T)
                self.best_hyper_paras = para_d
                self.best_balance = i_model_balance[0]
                self.best_val = i_model_fit[0]
                self.best_balance_k_folds_detail = i_model_balance_over_kfold
                self.best_val_k_folds_detail = i_model_fit_over_kfold

            if i_model_fit[0] > self.global_best_val:
                self.global_best_val = i_model_fit[0]

            if i_model_balance[0] < self.global_best_balance:
                self.global_best_balance = i_model_balance[0]

            # save re-trained results on the whole (training+val) data, for model selection exp only. Not necessary for later use
            model_retrain = self._model_estimation(para_d, X, T)
            T_pre = model_retrain.predict_proba(X)[:, 1]
            result_retrain = self._evaluation_helper(X, T, T_pre)

            # testing model on the test data, for model selection exp only. Not necessary for later use
            T_test_pre = model_retrain.predict_proba(X_test)[:, 1]
            result_test = self._evaluation_helper(X_test, T_test, T_test_pre)
            T_all_pre = model_retrain.predict_proba(X_all)[:, 1]
            result_all = self._evaluation_helper(X_all, T_all, T_all_pre)

            # cross-validation part build train and val results
            # this part build retrain on train+val, test, and all results.
            self.results_retrain.append((i, 'retrain', para_d) + result_retrain + result_test + result_all)

            if verbose:
                print('Finish training {}th (/{}) model {} over the {}th-fold data'.format(i, len(self.paras_list), para_d, kfold))
                print('CV Balance mean, std:', i_model_balance,  'k_folds:', i_model_balance_over_kfold)
                print('CV Fit mean, std:', i_model_fit,  'k_folds:', i_model_fit_over_kfold)
                self.report_stats()

        # end of training
        print('best model parameter:', self.best_hyper_paras)
        print('re-training best model on all the data using best model parameter...')
        # best model is used in predicting ps
        # retrained here
        self.best_model = self._model_estimation(self.best_hyper_paras, X, T)
        name = ['loss', 'auc', 'max_smd', 'n_unbalanced_feat', 'max_smd_iptw', 'n_unbalanced_feat_iptw']
        col_name = ['i', 'fold-k', 'paras'] + [pre + x for pre in ['train_', 'val_', 'beforRetrain trainval_'] for x in name]
        self.results = pd.DataFrame(self.results, columns=col_name)
        self.results['paras_str'] = self.results['paras'].apply(lambda x: str(x))

        col_name_retrain = ['i', 'fold-k', 'paras'] + [pre + x for pre in ['trainval_', 'test_', 'all_'] for x in name]
        self.results_retrain = pd.DataFrame(self.results_retrain, columns=col_name_retrain)
        self.results_retrain['paras_str'] = self.results_retrain['paras'].apply(lambda x: str(x))

        results_agg = self.results.groupby('paras_str').agg(['mean', 'std']).reset_index().sort_values(by=[('i', 'mean')])
        results_agg.columns = results_agg.columns.to_flat_index()
        results_agg.columns = results_agg.columns.map('-'.join)
        self.results_agg = pd.merge(results_agg, self.results_retrain, left_on='paras_str-', right_on='paras_str', how='left')

        if verbose:
            self.report_stats()
        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        return self

    def report_stats(self):
        print('Model {} Searching Space N={}: '.format(self.learner, len(self.paras_list)), self.paras_grid)
        print('Best model: ', self.best_model)
        print('Best configuration: ', self.best_hyper_paras)
        print('Best balance: ', self.best_balance, ' Global Best balance: ', self.global_best_balance)
        print('Best fit value ', self.best_val, ' Global Best fit balue: ', self.global_best_val)
        try:
            pd.set_option('display.max_columns', None)
            describe = self.results.describe()
            print('AUC stats:\n', describe)
            return describe
        except:
            print('')


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
