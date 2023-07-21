import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import softmax
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import survival_difference_at_fixed_point_in_time_test
import pandas as pd
from utils import save_model, load_model, check_and_mkdir
import pickle
import os

# Define unbalanced threshold, where SMD > SMD_THRESHOLD are defined as unbalanced features
SMD_THRESHOLD = 0.1


def transfer_data_lstm(model, dataloader, cuda=True):
    with torch.no_grad():
        model.eval()
        loss_treatment = []
        logits_treatment = []
        labels_treatment = []
        labels_outcome = []
        original_val = []
        for confounder, treatment, outcome in dataloader:
            if cuda:
                for i in range(len(confounder)):
                    confounder[i] = confounder[i].to('cuda')
                treatment = treatment.to('cuda')
                # outcome = outcome.to('cuda')

            treatment_logits, original = model(confounder)
            loss_t = F.binary_cross_entropy_with_logits(treatment_logits, treatment.float())

            if cuda:
                logits_t = treatment_logits.to('cpu').detach().data.numpy()
                labels_t = treatment.to('cpu').detach().data.numpy()
                original = original.to('cpu').detach().data.numpy()
                # labels_o = outcome.to('cpu').detach().data.numpy()
            else:
                logits_t = treatment_logits.detach().data.numpy()
                labels_t = treatment.detach().data.numpy()
                original = original.detach().data.numpy()

            labels_o = outcome.detach().data.numpy()

            logits_treatment.append(logits_t)
            labels_treatment.append(labels_t)
            labels_outcome.append(labels_o)
            loss_treatment.append(loss_t.item())
            original_val.extend(original)

        loss_treatment = np.mean(loss_treatment)

        golds_treatment = np.concatenate(labels_treatment)
        golds_outcome = np.concatenate(labels_outcome)
        logits_treatment = np.concatenate(logits_treatment)

        return loss_treatment, golds_treatment, logits_treatment, golds_outcome, original_val


def transfer_data(model, dataloader, cuda=True):
    with torch.no_grad():
        model.eval()
        loss_treatment = []
        logits_treatment = []
        labels_treatment = []
        labels_outcome = []
        original_val = []
        for confounder, treatment, outcome in dataloader:
            dx, rx, sex, age, days = confounder[0], confounder[1], confounder[2], confounder[3], confounder[4]
            dx, rx = torch.sum(dx, 1), torch.sum(rx, 1)
            dx = torch.where(dx > 0, 1., 0.)
            rx = torch.where(rx > 0, 1., 0.)
            X = torch.cat((dx, rx, sex.unsqueeze(1), age.unsqueeze(1), days.unsqueeze(1)), 1)
            if cuda:
                X = X.float().to('cuda')
                treatment = treatment.to('cuda')
                outcome = outcome.to('cuda')

            treatment_logits = model(X)
            loss_t = F.cross_entropy(treatment_logits, treatment)

            if cuda:
                logits_t = treatment_logits.to('cpu').detach().data.numpy()
                labels_t = treatment.to('cpu').detach().data.numpy()
                original = X.to('cpu').detach().data.numpy()
                labels_o = outcome.to('cpu').detach().data.numpy()
            else:
                logits_t = treatment_logits.detach().data.numpy()
                labels_t = treatment.detach().data.numpy()
                original = X.detach().data.numpy()
                labels_o = outcome.detach().data.numpy()

            logits_treatment.append(logits_t)  # [:,1])
            labels_treatment.append(labels_t)
            labels_outcome.append(labels_o)
            loss_treatment.append(loss_t.item())
            original_val.extend(original)

        loss_treatment = np.mean(loss_treatment)
        golds_treatment = np.concatenate(labels_treatment)
        golds_outcome = np.concatenate(labels_outcome)
        logits_treatment = np.concatenate(logits_treatment)

        return loss_treatment, golds_treatment, logits_treatment, golds_outcome, original_val


def cox_no_weight(golds_treatment, golds_outcome):
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    if len(golds_outcome.shape) == 2:
        treated_outcome, controlled_outcome = golds_outcome[ones_idx, 0], golds_outcome[zeros_idx, 0]
        treated_outcome[treated_outcome == -1] = 0
        controlled_outcome[controlled_outcome == -1] = 0
    else:
        raise ValueError

    T = golds_outcome[:, 1]
    treated_t2e, controlled_t2e = T[ones_idx], T[zeros_idx]
    # cox for hazard ratio
    cph = CoxPHFitter()
    event = golds_outcome[:, 0]
    event[event == -1] = 0
    cox_data = pd.DataFrame({'T': T, 'event': event, 'treatment': golds_treatment, })
    try:
        cph_ori = CoxPHFitter()
        cox_data_ori = pd.DataFrame({'T': T, 'event': event, 'treatment': golds_treatment})
        cph_ori.fit(cox_data_ori, 'T', 'event')
        HR_ori = cph_ori.hazard_ratios_['treatment']
        CI_ori = np.exp(cph_ori.confidence_intervals_.values.reshape(-1))
    except:
        cph_ori = HR_ori = CI_ori = None

    return (HR_ori, CI_ori, cph_ori)


def cal_survival_HR_simple(golds_treatment, logits_treatment, golds_outcome, normalized):
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    treated_w, controlled_w = cal_weights(golds_treatment, logits_treatment, normalized)
    if len(golds_outcome.shape) == 2:
        treated_outcome, controlled_outcome = golds_outcome[ones_idx, 0], golds_outcome[zeros_idx, 0]
        treated_outcome[treated_outcome == -1] = 0
        controlled_outcome[controlled_outcome == -1] = 0
    else:
        raise ValueError

    # kmf = KaplanMeierFitter()
    T = golds_outcome[:, 1]
    treated_t2e, controlled_t2e = T[ones_idx], T[zeros_idx]
    # cox for hazard ratio
    cph = CoxPHFitter()
    event = golds_outcome[:, 0]
    event[event == -1] = 0
    weight = np.zeros(len(golds_treatment))
    weight[ones_idx] = treated_w.squeeze()
    weight[zeros_idx] = controlled_w.squeeze()
    cox_data = pd.DataFrame({'T': T, 'event': event, 'treatment': golds_treatment, 'weights': weight})
    try:
        cph.fit(cox_data, 'T', 'event', weights_col='weights', robust=True)
        HR = cph.hazard_ratios_['treatment']
        CI = np.exp(cph.confidence_intervals_.values.reshape(-1))

        cph_ori = CoxPHFitter()
        cox_data_ori = pd.DataFrame({'T': T, 'event': event, 'treatment': golds_treatment})
        cph_ori.fit(cox_data_ori, 'T', 'event')
        HR_ori = cph_ori.hazard_ratios_['treatment']
        CI_ori = np.exp(cph_ori.confidence_intervals_.values.reshape(-1))
    except:
        cph = HR = CI = None
        cph_ori = HR_ori = CI_ori = None

    return (HR_ori, CI_ori, cph_ori), (HR, CI, cph)


def model_eval_common_simple(X, T, Y, PS_logits, loss=None, normalized=False, verbose=1, figsave='', report=5):
    y_pred_prob = logits_to_probability(PS_logits, normalized)
    # 1. IPTW sample weights
    treated_w, controlled_w = cal_weights(T, PS_logits, normalized=normalized, stabilized=True)
    treated_PS, control_PS = y_pred_prob[T == 1], y_pred_prob[T == 0]
    n_treat, n_control = (T == 1).sum(), (T == 0).sum()

    cox_HR_ori, cox_HR = cal_survival_HR_simple(T, PS_logits, Y, normalized)
    KM_ALL = (np.nan, np.nan, cox_HR_ori, cox_HR)

    if verbose:
        print('loss: {}'.format(loss))
        print('treated_weights:',
              pd.Series(treated_w.flatten()).describe().to_string().replace('\n', ';'))  # stats.describe(treated_w))
        print('controlled_weights:', pd.Series(controlled_w.flatten()).describe().to_string().replace('\n',
                                                                                                      ';'))  # stats.describe(controlled_w))
        print('treated_PS:',
              pd.Series(treated_PS.flatten()).describe().to_string().replace('\n', ';'))  # stats.describe(treated_PS))
        print('controlled_PS:',
              pd.Series(control_PS.flatten()).describe().to_string().replace('\n', ';'))  # stats.describe(control_PS))
        print('Cox Hazard ratio ori {} (CI: {})'.format(cox_HR_ori[0], cox_HR_ori[1]))
        print('Cox Hazard ratio iptw {} (CI: {})'.format(cox_HR[0], cox_HR[1]))

    return KM_ALL


def model_eval_common(X, T, Y, PS_logits, loss=None, normalized=False, verbose=1, figsave='', report=5):
    y_pred_prob = logits_to_probability(PS_logits, normalized)
    # 1. IPTW sample weights
    treated_w, controlled_w = cal_weights(T, PS_logits, normalized=normalized, stabilized=True)
    treated_PS, control_PS = y_pred_prob[T == 1], y_pred_prob[T == 0]
    n_treat, n_control = (T == 1).sum(), (T == 0).sum()

    if verbose:
        print('loss: {}'.format(loss))
        print('treated_weights:',
              pd.Series(treated_w.flatten()).describe().to_string().replace('\n', ';'))  # stats.describe(treated_w))
        print('controlled_weights:', pd.Series(controlled_w.flatten()).describe().to_string().replace('\n',
                                                                                                      ';'))  # stats.describe(controlled_w))
        print('treated_PS:',
              pd.Series(treated_PS.flatten()).describe().to_string().replace('\n', ';'))  # stats.describe(treated_PS))
        print('controlled_PS:',
              pd.Series(control_PS.flatten()).describe().to_string().replace('\n', ';'))  # stats.describe(control_PS))

    IPTW_ALL = (loss, treated_w, controlled_w,
                pd.Series(treated_w.flatten()).describe().to_string().replace('\n', ';'),
                pd.Series(controlled_w.flatten()).describe().to_string().replace('\n', ';'),
                # stats.describe(treated_w), stats.describe(controlled_w),
                pd.Series(treated_PS.flatten()).describe().to_string().replace('\n', ';'),
                pd.Series(control_PS.flatten()).describe().to_string().replace('\n', ';'),
                # stats.describe(treated_PS), stats.describe(control_PS),
                n_treat, n_control)

    # 2. AUC score
    AUC = roc_auc_score(T, y_pred_prob)
    AUC_weighted = roc_auc_IPTW(T, PS_logits, normalized)  # newly added, not sure
    AUC_expected = roc_auc_expected(PS_logits, normalized)  # newly added, not sure
    AUC_diff = np.absolute(AUC_expected - AUC)
    AUC_ALL = (AUC, AUC_weighted, AUC_expected, AUC_diff)
    if verbose:
        print('AUC:{}\tAUC_weighted:{}\tAUC_expected:{}\tAUC_diff:{}'.
              format(AUC, AUC_weighted, AUC_expected, AUC_diff))

    # 3. SMD score
    if report >= 3:
        max_smd, smd, max_smd_weighted, smd_w = cal_deviation(X, T, PS_logits, normalized)
        n_unbalanced_feature = len(np.where(smd > SMD_THRESHOLD)[0])
        n_unbalanced_feature_w = len(np.where(smd_w > SMD_THRESHOLD)[0])
        SMD_ALL = (max_smd, smd, n_unbalanced_feature, max_smd_weighted, smd_w, n_unbalanced_feature_w)
        if verbose:
            print('max_smd_original:{}\tmax_smd_IPTW:{}\tn_unbalanced_feature_origin:{}\tn_unbalanced_feature_IPTW:{}'.
                  format(max_smd, max_smd_weighted, n_unbalanced_feature, n_unbalanced_feature_w))
    else:
        SMD_ALL = []

    # 4. ATE score
    if report >= 4:
        ATE, ATE_w = cal_ATE(T, PS_logits, Y, normalized)
        UncorrectedEstimator_EY1, UncorrectedEstimator_EY0, ATE_original = ATE
        IPWEstimator_EY1, IPWEstimator_EY0, ATE_weighted = ATE_w
        ATE_ALL = (UncorrectedEstimator_EY1, UncorrectedEstimator_EY0, ATE_original,
                   IPWEstimator_EY1, IPWEstimator_EY0, ATE_weighted)
        if verbose:
            print('E[Y1]:{}\tE[Y0]:{}\tATE:{}\tIPTW-E[Y1]:{}\tIPTW-E[Y0]:{}\tIPTW-ATE:{}'.
                  format(UncorrectedEstimator_EY1, UncorrectedEstimator_EY0, ATE_original,
                         IPWEstimator_EY1, IPWEstimator_EY0, ATE_weighted))
    else:
        ATE_ALL = []

    # 5. Survival
    if report >= 5:
        survival, survival_w, cox_HR_ori, cox_HR = cal_survival_KM(T, PS_logits, Y, normalized)
        kmf1, kmf0, ate, survival_1, survival_0, results = survival
        kmf1_w, kmf0_w, ate_w, survival_1_w, survival_0_w, results_w = survival_w
        KM_ALL = (survival, survival_w, cox_HR_ori, cox_HR)
        if verbose:
            print('KM_treated at [180, 365, 540, 730]:', survival_1,
                  'KM_control at [180, 365, 540, 730]:', survival_0)
            # results.print_summary()
            print('KM_treated_IPTW at [180, 365, 540, 730]:', survival_1_w,
                  'KM_control_IPTW at [180, 365, 540, 730]:', survival_0_w)
            print('KM_treated - KM_control:', ate)
            print('KM_treated_IPTW - KM_control_IPTW:', ate_w)
            print('Cox Hazard ratio ori {} (CI: {})'.format(cox_HR_ori[0], cox_HR_ori[1]))
            print('Cox Hazard ratio iptw {} (CI: {})'.format(cox_HR[0], cox_HR[1]))
            # results_w.print_summary()
            ax = plt.subplot(111)
            ax.set_title(os.path.basename(figsave))
            kmf1.plot_survival_function(ax=ax)
            kmf0.plot_survival_function(ax=ax)
            kmf1_w.plot_survival_function(ax=ax)
            kmf0_w.plot_survival_function(ax=ax)
            if figsave:
                plt.savefig(figsave + '_km.png')
            plt.clf()
            # plt.show()
    else:
        KM_ALL = []

    return IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL


def model_eval_deep(model, dataloader, verbose=1, normalized=False, cuda=True, figsave='', report=5, lstm=False):
    """  PSModels evaluation for deep PS PSModels """
    # loss_treatment, golds_treatment, logits_treatment, golds_outcome, original_val
    if lstm:
        loss, T, PS_logits, Y, X = transfer_data_lstm(model, dataloader, cuda=cuda)
    else:
        loss, T, PS_logits, Y, X = transfer_data(model, dataloader, cuda=cuda)
    return model_eval_common(X, T, Y, PS_logits, loss=loss, normalized=normalized,
                             verbose=verbose, figsave=figsave, report=report)


def final_eval_deep(model_class, args, train_loader, val_loader, test_loader, data_loader,
                    drug_name, feature_name, n_feature, dump_ori=True, lstm=False):
    mymodel = load_model(model_class, args.save_model_filename)
    mymodel.to(args.device)

    # ----. Model Evaluation & Final ATE results
    print("*****" * 5, 'Evaluation on Train data:')
    train_IPTW_ALL, train_AUC_ALL, train_SMD_ALL, train_ATE_ALL, train_KM_ALL = model_eval_deep(
        mymodel, train_loader, verbose=1, normalized=False, cuda=args.cuda,
        figsave=args.save_model_filename + '_train', lstm=lstm)
    print("*****" * 5, 'Evaluation on Validation data:')
    val_IPTW_ALL, val_AUC_ALL, val_SMD_ALL, val_ATE_ALL, val_KM_ALL = model_eval_deep(
        mymodel, val_loader, verbose=1, normalized=False, cuda=args.cuda,
        figsave=args.save_model_filename + '_val', lstm=lstm)
    print("*****" * 5, 'Evaluation on Test data:')
    test_IPTW_ALL, test_AUC_ALL, test_SMD_ALL, test_ATE_ALL, test_KM_ALL = model_eval_deep(
        mymodel, test_loader, verbose=1, normalized=False, cuda=args.cuda,
        figsave=args.save_model_filename + '_test', lstm=lstm)
    print("*****" * 5, 'Evaluation on ALL data:')
    IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL = model_eval_deep(
        mymodel, data_loader, verbose=1, normalized=False, cuda=args.cuda,
        figsave=args.save_model_filename + '_all', lstm=lstm)

    results_train_val_test_all = []
    results_all_list = [
        (train_IPTW_ALL, train_AUC_ALL, train_SMD_ALL, train_ATE_ALL, train_KM_ALL),
        (val_IPTW_ALL, val_AUC_ALL, val_SMD_ALL, val_ATE_ALL, val_KM_ALL),
        (test_IPTW_ALL, test_AUC_ALL, test_SMD_ALL, test_ATE_ALL, test_KM_ALL),
        (IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL)
    ]
    for tw, tauc, tsmd, tate, tkm in results_all_list:
        results_train_val_test_all.append(
            [args.treated_drug, drug_name[args.treated_drug], tw[7], tw[8],
             tw[0], tauc[0], tauc[1], tauc[2], tauc[3],
             tsmd[0], tsmd[3], tsmd[2], tsmd[5],
             n_feature, (tsmd[1] >= 0).sum(),
             *tate,
             [180, 365, 540, 730],
             tkm[0][3], tkm[0][4], tkm[0][2],
             tkm[1][3], tkm[1][4], tkm[1][2],
             tkm[2][0], tkm[2][1], tkm[3][0], tkm[3][1],
             tw[1].sum(), tw[2].sum(),
             tw[3], tw[4], tw[5], tw[6],
             ';'.join(feature_name[np.where(tsmd[1] > SMD_THRESHOLD)[0]]),
             ';'.join(feature_name[np.where(tsmd[4] > SMD_THRESHOLD)[0]])
             ])
    results_train_val_test_all = pd.DataFrame(results_train_val_test_all,
                                              columns=['drug', 'name', 'n_treat', 'n_ctrl',
                                                       'loss', 'AUC', 'AUC_IPTW', 'AUC_expected', 'AUC_diff',
                                                       'max_unbalanced_original', 'max_unbalanced_weighted',
                                                       'n_unbalanced_feature', 'n_unbalanced_feature_IPTW',
                                                       'n_feature', 'n_feature_nonzero',
                                                       'EY1_original', 'EY0_original', 'ATE_original',
                                                       'EY1_IPTW', 'EY0_IPTW', 'ATE_IPTW',
                                                       'KM_time_points',
                                                       'KM1_original', 'KM0_original', 'KM1-0_original',
                                                       'KM1_IPTW', 'KM0_IPTW', 'KM1-0_IPTW', 'HR_ori', 'HR_ori_CI',
                                                       'HR_IPTW', 'HR_IPTW_CI',
                                                       'n_treat_IPTW', 'n_ctrl_IPTW',
                                                       'treat_IPTW_stats', 'ctrl_IPTW_stats',
                                                       'treat_PS_stats', 'ctrl_PS_stats',
                                                       'unbalance_feature', 'unbalance_feature_IPTW'],
                                              index=['train', 'val', 'test', 'all'])

    # SMD_ALL = (max_smd, smd, n_unbalanced_feature, max_smd_weighted, smd_w, n_unbalanced_feature_w)
    print('Unbalanced features SMD:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          np.where(SMD_ALL[1] > SMD_THRESHOLD)[0])
    print('Unbalanced features SMD value:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          SMD_ALL[1][SMD_ALL[1] > SMD_THRESHOLD])
    print('Unbalanced features SMD names:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          feature_name[np.where(SMD_ALL[1] > SMD_THRESHOLD)[0]])
    print('Unbalanced features IPTW-SMD:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          np.where(SMD_ALL[4] > SMD_THRESHOLD)[0])
    print('Unbalanced features IPTW-SMD value:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          SMD_ALL[4][SMD_ALL[4] > SMD_THRESHOLD])
    print('Unbalanced features IPTW-SMD names:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          feature_name[np.where(SMD_ALL[4] > SMD_THRESHOLD)[0]])

    results_train_val_test_all.to_csv(args.save_model_filename + '_results.csv')
    print('Dump to ', args.save_model_filename + '_results.csv')
    if dump_ori:
        with open(args.save_model_filename + '_results.pt', 'wb') as f:
            pickle.dump(results_all_list + [feature_name, ], f)  #
            print('Dump to ', args.save_model_filename + '_results.pt')

    return results_all_list, results_train_val_test_all


def final_eval_deep_cv_revise(model_class, args, train_loader, val_loader, test_loader, data_loader,
                              drug_name, feature_name, n_feature, dump_ori=True, lstm=False):
    mymodel = load_model(model_class, args.save_model_filename)
    mymodel.to(args.device)

    print("*****" * 5, 'Evaluation on ALL data:')
    IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL = model_eval_deep(
        mymodel, data_loader, verbose=1, normalized=False, cuda=args.cuda,
        figsave=args.save_model_filename + '_all', lstm=lstm)

    results_train_val_test_all = []
    results_all_list = [
        # (train_IPTW_ALL, train_AUC_ALL, train_SMD_ALL, train_ATE_ALL, train_KM_ALL),
        # (val_IPTW_ALL, val_AUC_ALL, val_SMD_ALL, val_ATE_ALL, val_KM_ALL),
        # (test_IPTW_ALL, test_AUC_ALL, test_SMD_ALL, test_ATE_ALL, test_KM_ALL),
        (IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL)
    ]
    for tw, tauc, tsmd, tate, tkm in results_all_list:
        results_train_val_test_all.append(
            [args.treated_drug, drug_name[args.treated_drug], tw[7], tw[8],
             tw[0], tauc[0], tauc[1], tauc[2], tauc[3],
             tsmd[0], tsmd[3], tsmd[2], tsmd[5],
             n_feature, (tsmd[1] >= 0).sum(),
             *tate,
             [180, 365, 540, 730],
             tkm[0][3], tkm[0][4], tkm[0][2], tkm[0][5].p_value,
             tkm[1][3], tkm[1][4], tkm[1][2], tkm[1][5].p_value,
             tkm[2][0], tkm[2][1], tkm[2][2].summary.p.treatment if pd.notna(tkm[2][2]) else np.nan,
             tkm[3][0], tkm[3][1], tkm[3][2].summary.p.treatment if pd.notna(tkm[3][2]) else np.nan,
             tw[1].sum(), tw[2].sum(),
             tw[3], tw[4], tw[5], tw[6],
             ';'.join(feature_name[np.where(tsmd[1] > SMD_THRESHOLD)[0]]),
             ';'.join(feature_name[np.where(tsmd[4] > SMD_THRESHOLD)[0]])
             ])
    results_train_val_test_all = pd.DataFrame(results_train_val_test_all,
                                              columns=['drug', 'name', 'n_treat', 'n_ctrl',
                                                       'loss', 'AUC', 'AUC_IPTW', 'AUC_expected', 'AUC_diff',
                                                       'max_unbalanced_original', 'max_unbalanced_weighted',
                                                       'n_unbalanced_feature', 'n_unbalanced_feature_IPTW',
                                                       'n_feature', 'n_feature_nonzero',
                                                       'EY1_original', 'EY0_original', 'ATE_original',
                                                       'EY1_IPTW', 'EY0_IPTW', 'ATE_IPTW',
                                                       'KM_time_points',
                                                       'KM1_original', 'KM0_original', 'KM1-0_original',
                                                       'KM1-0_original_p',
                                                       'KM1_IPTW', 'KM0_IPTW', 'KM1-0_IPTW', 'KM1-0_IPTW_p',
                                                       'HR_ori', 'HR_ori_CI', 'HR_ori_p',
                                                       'HR_IPTW', 'HR_IPTW_CI', 'HR_IPTW_p',
                                                       'n_treat_IPTW', 'n_ctrl_IPTW',
                                                       'treat_IPTW_stats', 'ctrl_IPTW_stats',
                                                       'treat_PS_stats', 'ctrl_PS_stats',
                                                       'unbalance_feature', 'unbalance_feature_IPTW'],
                                              index=['all'])  # 'train', 'val', 'test',

    # SMD_ALL = (max_smd, smd, n_unbalanced_feature, max_smd_weighted, smd_w, n_unbalanced_feature_w)
    print('Unbalanced features SMD:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          np.where(SMD_ALL[1] > SMD_THRESHOLD)[0])
    print('Unbalanced features SMD value:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          SMD_ALL[1][SMD_ALL[1] > SMD_THRESHOLD])
    print('Unbalanced features SMD names:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          feature_name[np.where(SMD_ALL[1] > SMD_THRESHOLD)[0]])
    print('Unbalanced features IPTW-SMD:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          np.where(SMD_ALL[4] > SMD_THRESHOLD)[0])
    print('Unbalanced features IPTW-SMD value:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          SMD_ALL[4][SMD_ALL[4] > SMD_THRESHOLD])
    print('Unbalanced features IPTW-SMD names:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          feature_name[np.where(SMD_ALL[4] > SMD_THRESHOLD)[0]])

    results_train_val_test_all.to_csv(args.save_model_filename + '_results.csv')
    print('Dump to ', args.save_model_filename + '_results.csv')
    if dump_ori:
        with open(args.save_model_filename + '_results.pt', 'wb') as f:
            pickle.dump(results_all_list + [feature_name, ], f)  #
            print('Dump to ', args.save_model_filename + '_results.pt')

    return results_all_list, results_train_val_test_all


def final_eval_deep_cv_revise_traintest(model_class, args,
                                        train_loader,
                                        test_loader,
                                        data_loader, drug_name, feature_name, n_feature, dump_ori=True, lstm=False):
    mymodel = load_model(model_class, args.save_model_filename)
    mymodel.to(args.device)
    # ----. Model Evaluation & Final ATE results
    print("*****" * 5, 'Evaluation on Train data:')
    train_IPTW_ALL, train_AUC_ALL, train_SMD_ALL, train_ATE_ALL, train_KM_ALL = model_eval_deep(
        mymodel, train_loader, verbose=1, normalized=False, cuda=args.cuda,
        figsave=args.save_model_filename + '_train', lstm=lstm)
    print("*****" * 5, 'Evaluation on Test data:')
    test_IPTW_ALL, test_AUC_ALL, test_SMD_ALL, test_ATE_ALL, test_KM_ALL = model_eval_deep(
        mymodel, test_loader, verbose=1, normalized=False, cuda=args.cuda,
        figsave=args.save_model_filename + '_test', lstm=lstm)
    print("*****" * 5, 'Evaluation on ALL data:')
    IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL = model_eval_deep(
        mymodel, data_loader, verbose=1, normalized=False, cuda=args.cuda,
        figsave=args.save_model_filename + '_all', lstm=lstm)

    results_train_val_test_all = []
    results_all_list = [
        (train_IPTW_ALL, train_AUC_ALL, train_SMD_ALL, train_ATE_ALL, train_KM_ALL),
        # (val_IPTW_ALL, val_AUC_ALL, val_SMD_ALL, val_ATE_ALL, val_KM_ALL),
        (test_IPTW_ALL, test_AUC_ALL, test_SMD_ALL, test_ATE_ALL, test_KM_ALL),
        (IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL)
    ]
    for tw, tauc, tsmd, tate, tkm in results_all_list:
        results_train_val_test_all.append(
            [args.treated_drug, drug_name[args.treated_drug], tw[7], tw[8],
             tw[0], tauc[0], tauc[1], tauc[2], tauc[3],
             tsmd[0], tsmd[3], tsmd[2], tsmd[5],
             n_feature, (tsmd[1] >= 0).sum(),
             *tate,
             [180, 365, 540, 730],
             tkm[0][3], tkm[0][4], tkm[0][2], tkm[0][5].p_value,
             tkm[1][3], tkm[1][4], tkm[1][2], tkm[1][5].p_value,
             tkm[2][0], tkm[2][1], tkm[2][2].summary.p.treatment if pd.notna(tkm[2][2]) else np.nan,
             tkm[3][0], tkm[3][1], tkm[3][2].summary.p.treatment if pd.notna(tkm[3][2]) else np.nan,
             tw[1].sum(), tw[2].sum(),
             tw[3], tw[4], tw[5], tw[6],
             ';'.join(feature_name[np.where(tsmd[1] > SMD_THRESHOLD)[0]]),
             ';'.join(feature_name[np.where(tsmd[4] > SMD_THRESHOLD)[0]])
             ])
    results_train_val_test_all = pd.DataFrame(results_train_val_test_all,
                                              columns=['drug', 'name', 'n_treat', 'n_ctrl',
                                                       'loss', 'AUC', 'AUC_IPTW', 'AUC_expected', 'AUC_diff',
                                                       'max_unbalanced_original', 'max_unbalanced_weighted',
                                                       'n_unbalanced_feature', 'n_unbalanced_feature_IPTW',
                                                       'n_feature', 'n_feature_nonzero',
                                                       'EY1_original', 'EY0_original', 'ATE_original',
                                                       'EY1_IPTW', 'EY0_IPTW', 'ATE_IPTW',
                                                       'KM_time_points',
                                                       'KM1_original', 'KM0_original', 'KM1-0_original',
                                                       'KM1-0_original_p',
                                                       'KM1_IPTW', 'KM0_IPTW', 'KM1-0_IPTW', 'KM1-0_IPTW_p',
                                                       'HR_ori', 'HR_ori_CI', 'HR_ori_p',
                                                       'HR_IPTW', 'HR_IPTW_CI', 'HR_IPTW_p',
                                                       'n_treat_IPTW', 'n_ctrl_IPTW',
                                                       'treat_IPTW_stats', 'ctrl_IPTW_stats',
                                                       'treat_PS_stats', 'ctrl_PS_stats',
                                                       'unbalance_feature', 'unbalance_feature_IPTW'],
                                              index=['train', 'test', 'all'])  # 'train', 'val', 'test',

    # SMD_ALL = (max_smd, smd, n_unbalanced_feature, max_smd_weighted, smd_w, n_unbalanced_feature_w)
    print('Unbalanced features SMD:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          np.where(SMD_ALL[1] > SMD_THRESHOLD)[0])
    print('Unbalanced features SMD value:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          SMD_ALL[1][SMD_ALL[1] > SMD_THRESHOLD])
    print('Unbalanced features SMD names:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          feature_name[np.where(SMD_ALL[1] > SMD_THRESHOLD)[0]])
    print('Unbalanced features IPTW-SMD:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          np.where(SMD_ALL[4] > SMD_THRESHOLD)[0])
    print('Unbalanced features IPTW-SMD value:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          SMD_ALL[4][SMD_ALL[4] > SMD_THRESHOLD])
    print('Unbalanced features IPTW-SMD names:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          feature_name[np.where(SMD_ALL[4] > SMD_THRESHOLD)[0]])

    results_train_val_test_all.to_csv(args.save_model_filename + '_results.csv')
    print('Dump to ', args.save_model_filename + '_results.csv')
    if dump_ori:
        with open(args.save_model_filename + '_results.pt', 'wb') as f:
            pickle.dump(results_all_list + [feature_name, ], f)  #
            print('Dump to ', args.save_model_filename + '_results.pt')

    return results_all_list, results_train_val_test_all


def final_eval_ml(model, args, train_x, train_t, train_y, val_x, val_t, val_y, test_x, test_t, test_y,
                  x, t, y, drug_name, feature_name, n_feature, dump_ori=True):
    # ----. Model Evaluation & Final ATE results
    # model_eval_common(X, T, Y, PS_logits, loss=None, normalized=False, verbose=1, figsave='', report=5)
    print("*****" * 5, 'Evaluation on Train data:')
    train_IPTW_ALL, train_AUC_ALL, train_SMD_ALL, train_ATE_ALL, train_KM_ALL = model_eval_common(
        train_x, train_t, train_y, model.predict_ps(train_x), loss=model.predict_loss(train_x, train_t),
        normalized=True, figsave=args.save_model_filename + '_train')

    print("*****" * 5, 'Evaluation on Validation data:')
    val_IPTW_ALL, val_AUC_ALL, val_SMD_ALL, val_ATE_ALL, val_KM_ALL = model_eval_common(
        val_x, val_t, val_y, model.predict_ps(val_x), loss=model.predict_loss(val_x, val_t),
        normalized=True, figsave=args.save_model_filename + '_val')

    print("*****" * 5, 'Evaluation on Test data:')
    test_IPTW_ALL, test_AUC_ALL, test_SMD_ALL, test_ATE_ALL, test_KM_ALL = model_eval_common(
        test_x, test_t, test_y, model.predict_ps(test_x), loss=model.predict_loss(test_x, test_t),
        normalized=True, figsave=args.save_model_filename + '_test')

    print("*****" * 5, 'Evaluation on ALL data:')
    IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL = model_eval_common(
        x, t, y, model.predict_ps(x), loss=model.predict_loss(x, t),
        normalized=True, figsave=args.save_model_filename + '_all')

    results_train_val_test_all = []
    results_all_list = [
        (train_IPTW_ALL, train_AUC_ALL, train_SMD_ALL, train_ATE_ALL, train_KM_ALL),
        (val_IPTW_ALL, val_AUC_ALL, val_SMD_ALL, val_ATE_ALL, val_KM_ALL),
        (test_IPTW_ALL, test_AUC_ALL, test_SMD_ALL, test_ATE_ALL, test_KM_ALL),
        (IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL)
    ]
    for tw, tauc, tsmd, tate, tkm in results_all_list:
        results_train_val_test_all.append(
            [args.treated_drug, drug_name[args.treated_drug], tw[7], tw[8],
             tw[0], tauc[0], tauc[1], tauc[2], tauc[3],
             tsmd[0], tsmd[3], tsmd[2], tsmd[5],
             n_feature, (tsmd[1] >= 0).sum(),
             *tate,
             [180, 365, 540, 730],
             tkm[0][3], tkm[0][4], tkm[0][2],
             tkm[1][3], tkm[1][4], tkm[1][2],
             tkm[2][0], tkm[2][1], tkm[3][0], tkm[3][1],
             tw[1].sum(), tw[2].sum(),
             tw[3], tw[4], tw[5], tw[6],
             ';'.join(feature_name[np.where(tsmd[1] > SMD_THRESHOLD)[0]]),
             ';'.join(feature_name[np.where(tsmd[4] > SMD_THRESHOLD)[0]])
             ])
    results_train_val_test_all = pd.DataFrame(results_train_val_test_all,
                                              columns=['drug', 'name', 'n_treat', 'n_ctrl',
                                                       'loss', 'AUC', 'AUC_IPTW', 'AUC_expected', 'AUC_diff',
                                                       'max_unbalanced_original', 'max_unbalanced_weighted',
                                                       'n_unbalanced_feature', 'n_unbalanced_feature_IPTW',
                                                       'n_feature', 'n_feature_nonzero',
                                                       'EY1_original', 'EY0_original', 'ATE_original',
                                                       'EY1_IPTW', 'EY0_IPTW', 'ATE_IPTW',
                                                       'KM_time_points',
                                                       'KM1_original', 'KM0_original', 'KM1-0_original',
                                                       'KM1_IPTW', 'KM0_IPTW', 'KM1-0_IPTW', 'HR_ori', 'HR_ori_CI',
                                                       'HR_IPTW', 'HR_IPTW_CI',
                                                       'n_treat_IPTW', 'n_ctrl_IPTW',
                                                       'treat_IPTW_stats', 'ctrl_IPTW_stats',
                                                       'treat_PS_stats', 'ctrl_PS_stats',
                                                       'unbalance_feature', 'unbalance_feature_IPTW'],
                                              index=['train', 'val', 'test', 'all'])

    # SMD_ALL = (max_smd, smd, n_unbalanced_feature, max_smd_weighted, smd_w, n_unbalanced_feature_w)
    print('Unbalanced features SMD:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          np.where(SMD_ALL[1] > SMD_THRESHOLD)[0])
    print('Unbalanced features SMD value:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          SMD_ALL[1][SMD_ALL[1] > SMD_THRESHOLD])
    print('Unbalanced features SMD names:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          feature_name[np.where(SMD_ALL[1] > SMD_THRESHOLD)[0]])
    print('Unbalanced features IPTW-SMD:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          np.where(SMD_ALL[4] > SMD_THRESHOLD)[0])
    print('Unbalanced features IPTW-SMD value:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          SMD_ALL[4][SMD_ALL[4] > SMD_THRESHOLD])
    print('Unbalanced features IPTW-SMD names:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          feature_name[np.where(SMD_ALL[4] > SMD_THRESHOLD)[0]])

    results_train_val_test_all.to_csv(args.save_model_filename + '_results.csv')
    print('Dump to ', args.save_model_filename + '_results.csv')
    if dump_ori:
        with open(args.save_model_filename + '_results.pt', 'wb') as f:
            pickle.dump(results_all_list + [feature_name, ], f)  #
            print('Dump to ', args.save_model_filename + '_results.pt')

    return results_all_list, results_train_val_test_all


def final_eval_ml_CV_revise(model, args, x, t, y, drug_name, feature_name, n_feature, dump_ori=True):
    # ----. Model Evaluation & Final ATE results
    # model_eval_common(X, T, Y, PS_logits, loss=None, normalized=False, verbose=1, figsave='', report=5)

    print("*****" * 5, 'Evaluation on ALL data:')
    IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL = model_eval_common(
        x, t, y, model.predict_ps(x), loss=model.predict_loss(x, t),
        normalized=True, figsave=args.save_model_filename + '_all')

    results_train_val_test_all = []
    results_all_list = [
        # (train_IPTW_ALL, train_AUC_ALL, train_SMD_ALL, train_ATE_ALL, train_KM_ALL),
        # (val_IPTW_ALL, val_AUC_ALL, val_SMD_ALL, val_ATE_ALL, val_KM_ALL),
        # (test_IPTW_ALL, test_AUC_ALL, test_SMD_ALL, test_ATE_ALL, test_KM_ALL),
        (IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL)
    ]
    for tw, tauc, tsmd, tate, tkm in results_all_list:
        results_train_val_test_all.append(
            [args.treated_drug, drug_name[args.treated_drug], tw[7], tw[8],
             tw[0], tauc[0], tauc[1], tauc[2], tauc[3],
             tsmd[0], tsmd[3], tsmd[2], tsmd[5],
             n_feature, (tsmd[1] >= 0).sum(),
             *tate,
             [180, 365, 540, 730],
             tkm[0][3], tkm[0][4], tkm[0][2], tkm[0][5].p_value,
             tkm[1][3], tkm[1][4], tkm[1][2], tkm[1][5].p_value,
             tkm[2][0], tkm[2][1], tkm[2][2].summary.p.treatment if pd.notna(tkm[2][2]) else np.nan,
             tkm[3][0], tkm[3][1], tkm[3][2].summary.p.treatment if pd.notna(tkm[3][2]) else np.nan,
             tw[1].sum(), tw[2].sum(),
             tw[3], tw[4], tw[5], tw[6],
             ';'.join(feature_name[np.where(tsmd[1] > SMD_THRESHOLD)[0]]),
             ';'.join(feature_name[np.where(tsmd[4] > SMD_THRESHOLD)[0]])
             ])
    results_train_val_test_all = pd.DataFrame(results_train_val_test_all,
                                              columns=['drug', 'name', 'n_treat', 'n_ctrl',
                                                       'loss', 'AUC', 'AUC_IPTW', 'AUC_expected', 'AUC_diff',
                                                       'max_unbalanced_original', 'max_unbalanced_weighted',
                                                       'n_unbalanced_feature', 'n_unbalanced_feature_IPTW',
                                                       'n_feature', 'n_feature_nonzero',
                                                       'EY1_original', 'EY0_original', 'ATE_original',
                                                       'EY1_IPTW', 'EY0_IPTW', 'ATE_IPTW',
                                                       'KM_time_points',
                                                       'KM1_original', 'KM0_original', 'KM1-0_original',
                                                       'KM1-0_original_p',
                                                       'KM1_IPTW', 'KM0_IPTW', 'KM1-0_IPTW', 'KM1-0_IPTW_p',
                                                       'HR_ori', 'HR_ori_CI', 'HR_ori_p',
                                                       'HR_IPTW', 'HR_IPTW_CI', 'HR_IPTW_p',
                                                       'n_treat_IPTW', 'n_ctrl_IPTW',
                                                       'treat_IPTW_stats', 'ctrl_IPTW_stats',
                                                       'treat_PS_stats', 'ctrl_PS_stats',
                                                       'unbalance_feature', 'unbalance_feature_IPTW'],
                                              index=['all'])  # 'train', 'val', 'test',

    # SMD_ALL = (max_smd, smd, n_unbalanced_feature, max_smd_weighted, smd_w, n_unbalanced_feature_w)
    print('Unbalanced features SMD:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          np.where(SMD_ALL[1] > SMD_THRESHOLD)[0])
    print('Unbalanced features SMD value:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          SMD_ALL[1][SMD_ALL[1] > SMD_THRESHOLD])
    print('Unbalanced features SMD names:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          feature_name[np.where(SMD_ALL[1] > SMD_THRESHOLD)[0]])
    print('Unbalanced features IPTW-SMD:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          np.where(SMD_ALL[4] > SMD_THRESHOLD)[0])
    print('Unbalanced features IPTW-SMD value:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          SMD_ALL[4][SMD_ALL[4] > SMD_THRESHOLD])
    print('Unbalanced features IPTW-SMD names:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          feature_name[np.where(SMD_ALL[4] > SMD_THRESHOLD)[0]])

    results_train_val_test_all.to_csv(args.save_model_filename + '_results.csv')
    print('Dump to ', args.save_model_filename + '_results.csv')
    if dump_ori:
        with open(args.save_model_filename + '_results.pt', 'wb') as f:
            pickle.dump(results_all_list + [feature_name, ], f)  #
            print('Dump to ', args.save_model_filename + '_results.pt')

    return results_all_list, results_train_val_test_all

def final_eval_ml_nestedCV_revise(model, kout, args, x, t, y, drug_name, feature_name, n_feature, dump_ori=True):
    # ----. Model Evaluation & Final ATE results
    # model_eval_common(X, T, Y, PS_logits, loss=None, normalized=False, verbose=1, figsave='', report=5)

    print("*****" * 5, 'Evaluation on ALL data:')
    IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL = model_eval_common(
        x, t, y, model.predict_ps_nestedCV(x, kout), loss=model.predict_loss_nestedCV(x, t, kout),
        normalized=True, figsave=args.save_model_filename + '_all-kout{}'.format(kout))

    results_train_val_test_all = []
    results_all_list = [
        # (train_IPTW_ALL, train_AUC_ALL, train_SMD_ALL, train_ATE_ALL, train_KM_ALL),
        # (val_IPTW_ALL, val_AUC_ALL, val_SMD_ALL, val_ATE_ALL, val_KM_ALL),
        # (test_IPTW_ALL, test_AUC_ALL, test_SMD_ALL, test_ATE_ALL, test_KM_ALL),
        (IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL)
    ]
    for tw, tauc, tsmd, tate, tkm in results_all_list:
        results_train_val_test_all.append(
            [args.treated_drug, drug_name[args.treated_drug], tw[7], tw[8],
             tw[0], tauc[0], tauc[1], tauc[2], tauc[3],
             tsmd[0], tsmd[3], tsmd[2], tsmd[5],
             n_feature, (tsmd[1] >= 0).sum(),
             *tate,
             [180, 365, 540, 730],
             tkm[0][3], tkm[0][4], tkm[0][2], tkm[0][5].p_value,
             tkm[1][3], tkm[1][4], tkm[1][2], tkm[1][5].p_value,
             tkm[2][0], tkm[2][1], tkm[2][2].summary.p.treatment if pd.notna(tkm[2][2]) else np.nan,
             tkm[3][0], tkm[3][1], tkm[3][2].summary.p.treatment if pd.notna(tkm[3][2]) else np.nan,
             tw[1].sum(), tw[2].sum(),
             tw[3], tw[4], tw[5], tw[6],
             ';'.join(feature_name[np.where(tsmd[1] > SMD_THRESHOLD)[0]]),
             ';'.join(feature_name[np.where(tsmd[4] > SMD_THRESHOLD)[0]])
             ])
    results_train_val_test_all = pd.DataFrame(results_train_val_test_all,
                                              columns=['drug', 'name', 'n_treat', 'n_ctrl',
                                                       'loss', 'AUC', 'AUC_IPTW', 'AUC_expected', 'AUC_diff',
                                                       'max_unbalanced_original', 'max_unbalanced_weighted',
                                                       'n_unbalanced_feature', 'n_unbalanced_feature_IPTW',
                                                       'n_feature', 'n_feature_nonzero',
                                                       'EY1_original', 'EY0_original', 'ATE_original',
                                                       'EY1_IPTW', 'EY0_IPTW', 'ATE_IPTW',
                                                       'KM_time_points',
                                                       'KM1_original', 'KM0_original', 'KM1-0_original',
                                                       'KM1-0_original_p',
                                                       'KM1_IPTW', 'KM0_IPTW', 'KM1-0_IPTW', 'KM1-0_IPTW_p',
                                                       'HR_ori', 'HR_ori_CI', 'HR_ori_p',
                                                       'HR_IPTW', 'HR_IPTW_CI', 'HR_IPTW_p',
                                                       'n_treat_IPTW', 'n_ctrl_IPTW',
                                                       'treat_IPTW_stats', 'ctrl_IPTW_stats',
                                                       'treat_PS_stats', 'ctrl_PS_stats',
                                                       'unbalance_feature', 'unbalance_feature_IPTW'],
                                              index=['all'])  # 'train', 'val', 'test',

    # SMD_ALL = (max_smd, smd, n_unbalanced_feature, max_smd_weighted, smd_w, n_unbalanced_feature_w)
    print('Unbalanced features SMD:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          np.where(SMD_ALL[1] > SMD_THRESHOLD)[0])
    print('Unbalanced features SMD value:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          SMD_ALL[1][SMD_ALL[1] > SMD_THRESHOLD])
    print('Unbalanced features SMD names:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          feature_name[np.where(SMD_ALL[1] > SMD_THRESHOLD)[0]])
    print('Unbalanced features IPTW-SMD:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          np.where(SMD_ALL[4] > SMD_THRESHOLD)[0])
    print('Unbalanced features IPTW-SMD value:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          SMD_ALL[4][SMD_ALL[4] > SMD_THRESHOLD])
    print('Unbalanced features IPTW-SMD names:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          feature_name[np.where(SMD_ALL[4] > SMD_THRESHOLD)[0]])

    # 2023-6-22, too many to dump, only dump aggregated df
    # results_train_val_test_all.to_csv(args.save_model_filename + '_results-kout{}.csv'.format(kout))
    # print('Dump to ', args.save_model_filename + '_results-kout{}.csv'.format(kout))
    if dump_ori:
        results_train_val_test_all.to_csv(args.save_model_filename + '_results-kout{}.csv'.format(kout))
        print('Dump to ', args.save_model_filename + '_results-kout{}.csv'.format(kout))
        with open(args.save_model_filename + '_results-kout{}.pt'.format(kout), 'wb') as f:
            pickle.dump(results_all_list + [feature_name, ], f)  #
            print('Dump to ', args.save_model_filename + '_results-kout{}.pt'.format(kout))

    return results_all_list, results_train_val_test_all


def final_eval_ml_CV_revise_traintest(model, args,
                                      train_x, train_t, train_y,
                                      test_x, test_t, test_y,
                                      x, t, y, drug_name, feature_name, n_feature, dump_ori=True):
    # ----. Model Evaluation & Final ATE results
    # model_eval_common(X, T, Y, PS_logits, loss=None, normalized=False, verbose=1, figsave='', report=5)
    print("*****" * 5, 'Evaluation on Train data:')
    train_IPTW_ALL, train_AUC_ALL, train_SMD_ALL, train_ATE_ALL, train_KM_ALL = model_eval_common(
        train_x, train_t, train_y, model.predict_ps(train_x), loss=model.predict_loss(train_x, train_t),
        normalized=True, figsave=args.save_model_filename + '_train')

    print("*****" * 5, 'Evaluation on Test data:')
    test_IPTW_ALL, test_AUC_ALL, test_SMD_ALL, test_ATE_ALL, test_KM_ALL = model_eval_common(
        test_x, test_t, test_y, model.predict_ps(test_x), loss=model.predict_loss(test_x, test_t),
        normalized=True, figsave=args.save_model_filename + '_test')

    print("*****" * 5, 'Evaluation on ALL data:')
    IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL = model_eval_common(
        x, t, y, model.predict_ps(x), loss=model.predict_loss(x, t),
        normalized=True, figsave=args.save_model_filename + '_all')

    results_train_val_test_all = []
    results_all_list = [
        (train_IPTW_ALL, train_AUC_ALL, train_SMD_ALL, train_ATE_ALL, train_KM_ALL),
        # (val_IPTW_ALL, val_AUC_ALL, val_SMD_ALL, val_ATE_ALL, val_KM_ALL),
        (test_IPTW_ALL, test_AUC_ALL, test_SMD_ALL, test_ATE_ALL, test_KM_ALL),
        (IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL)
    ]
    for tw, tauc, tsmd, tate, tkm in results_all_list:
        results_train_val_test_all.append(
            [args.treated_drug, drug_name[args.treated_drug], tw[7], tw[8],
             tw[0], tauc[0], tauc[1], tauc[2], tauc[3],
             tsmd[0], tsmd[3], tsmd[2], tsmd[5],
             n_feature, (tsmd[1] >= 0).sum(),
             *tate,
             [180, 365, 540, 730],
             tkm[0][3], tkm[0][4], tkm[0][2], tkm[0][5].p_value,
             tkm[1][3], tkm[1][4], tkm[1][2], tkm[1][5].p_value,
             tkm[2][0], tkm[2][1], tkm[2][2].summary.p.treatment if pd.notna(tkm[2][2]) else np.nan,
             tkm[3][0], tkm[3][1], tkm[3][2].summary.p.treatment if pd.notna(tkm[3][2]) else np.nan,
             tw[1].sum(), tw[2].sum(),
             tw[3], tw[4], tw[5], tw[6],
             ';'.join(feature_name[np.where(tsmd[1] > SMD_THRESHOLD)[0]]),
             ';'.join(feature_name[np.where(tsmd[4] > SMD_THRESHOLD)[0]])
             ])
    results_train_val_test_all = pd.DataFrame(results_train_val_test_all,
                                              columns=['drug', 'name', 'n_treat', 'n_ctrl',
                                                       'loss', 'AUC', 'AUC_IPTW', 'AUC_expected', 'AUC_diff',
                                                       'max_unbalanced_original', 'max_unbalanced_weighted',
                                                       'n_unbalanced_feature', 'n_unbalanced_feature_IPTW',
                                                       'n_feature', 'n_feature_nonzero',
                                                       'EY1_original', 'EY0_original', 'ATE_original',
                                                       'EY1_IPTW', 'EY0_IPTW', 'ATE_IPTW',
                                                       'KM_time_points',
                                                       'KM1_original', 'KM0_original', 'KM1-0_original',
                                                       'KM1-0_original_p',
                                                       'KM1_IPTW', 'KM0_IPTW', 'KM1-0_IPTW', 'KM1-0_IPTW_p',
                                                       'HR_ori', 'HR_ori_CI', 'HR_ori_p',
                                                       'HR_IPTW', 'HR_IPTW_CI', 'HR_IPTW_p',
                                                       'n_treat_IPTW', 'n_ctrl_IPTW',
                                                       'treat_IPTW_stats', 'ctrl_IPTW_stats',
                                                       'treat_PS_stats', 'ctrl_PS_stats',
                                                       'unbalance_feature', 'unbalance_feature_IPTW'],
                                              index=['train', 'test', 'all'])  # 'train', 'val', 'test',

    # SMD_ALL = (max_smd, smd, n_unbalanced_feature, max_smd_weighted, smd_w, n_unbalanced_feature_w)
    print('Unbalanced features SMD:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          np.where(SMD_ALL[1] > SMD_THRESHOLD)[0])
    print('Unbalanced features SMD value:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          SMD_ALL[1][SMD_ALL[1] > SMD_THRESHOLD])
    print('Unbalanced features SMD names:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          feature_name[np.where(SMD_ALL[1] > SMD_THRESHOLD)[0]])
    print('Unbalanced features IPTW-SMD:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          np.where(SMD_ALL[4] > SMD_THRESHOLD)[0])
    print('Unbalanced features IPTW-SMD value:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          SMD_ALL[4][SMD_ALL[4] > SMD_THRESHOLD])
    print('Unbalanced features IPTW-SMD names:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          feature_name[np.where(SMD_ALL[4] > SMD_THRESHOLD)[0]])

    results_train_val_test_all.to_csv(args.save_model_filename + '_results.csv')
    print('Dump to ', args.save_model_filename + '_results.csv')
    if dump_ori:
        with open(args.save_model_filename + '_results.pt', 'wb') as f:
            pickle.dump(results_all_list + [feature_name, ], f)  #
            print('Dump to ', args.save_model_filename + '_results.pt')

    return results_all_list, results_train_val_test_all


# %%  Aux-functions
def logits_to_probability(logits, normalized):
    if normalized:
        if len(logits.shape) == 1:
            return logits
        elif len(logits.shape) == 2:
            return logits[:, 1]
        else:
            raise ValueError
    else:
        if len(logits.shape) == 1:
            return 1 / (1 + np.exp(-logits))
        elif len(logits.shape) == 2:
            prop = softmax(logits, axis=1)
            return prop[:, 1]
        else:
            raise ValueError


def roc_auc_IPTW(y_true, logits_treatment, normalized):
    treated_w, controlled_w = cal_weights(y_true, logits_treatment, normalized=normalized)
    y_pred_prob = logits_to_probability(logits_treatment, normalized)
    weight = np.zeros((len(logits_treatment), 1))
    weight[y_true == 1] = treated_w
    weight[y_true == 0] = controlled_w
    AUC = roc_auc_score(y_true, y_pred_prob, sample_weight=weight)
    return AUC


def roc_auc_expected(logits_treatment, normalized):
    y_pred_prob = logits_to_probability(logits_treatment, normalized)
    weights = np.concatenate([y_pred_prob, 1 - y_pred_prob])
    t = np.concatenate([np.ones_like(y_pred_prob), np.zeros_like(y_pred_prob)])
    p_hat = np.concatenate([y_pred_prob, y_pred_prob])
    AUC = roc_auc_score(t, p_hat, sample_weight=weights)
    return AUC


def cal_weights(golds_treatment, logits_treatment, normalized, stabilized=True, clip=True):
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    logits_treatment = logits_to_probability(logits_treatment, normalized)
    p_T = len(ones_idx[0]) / (len(ones_idx[0]) + len(zeros_idx[0]))

    # comment out p_T scaled IPTW
    if stabilized:
        # stabilized weights:   treated_w.sum() + controlled_w.sum() ~ N
        treated_w, controlled_w = p_T / logits_treatment[ones_idx], (1 - p_T) / (
                1. - logits_treatment[zeros_idx])  # why *p_T here?
    else:
        # standard IPTW:  treated_w.sum() + controlled_w.sum() > N
        treated_w, controlled_w = 1. / logits_treatment[ones_idx], 1. / (
                1. - logits_treatment[zeros_idx])  # why *p_T here? my added test

    treated_w[np.isinf(treated_w)] = 0
    controlled_w[np.isinf(controlled_w)] = 0

    if clip:
        # treated_w = np.clip(treated_w, a_min=1e-06, a_max=50)
        # controlled_w = np.clip(controlled_w, a_min=1e-06, a_max=50)
        amin = np.quantile(np.concatenate((treated_w, controlled_w)), 0.01)
        amax = np.quantile(np.concatenate((treated_w, controlled_w)), 0.99)

        if amax > 50:
            # if there are inf involved in qunatile, returen nan
            amax = np.quantile(np.concatenate((treated_w, controlled_w)), 0.8)
        if amin <= 1e-6:
            amin = np.quantile(np.concatenate((treated_w, controlled_w)), 0.2)

        # print('Using IPTW trim [{}, {}]'.format(amin, amax))
        treated_w = np.clip(treated_w, a_min=amin, a_max=amax)
        controlled_w = np.clip(controlled_w, a_min=amin, a_max=amax)

    # treated_w = np.where(treated_w < 10, treated_w, 25)
    # controlled_w = np.where(controlled_w < 10, controlled_w, 10)

    treated_w, controlled_w = np.reshape(treated_w, (len(treated_w), 1)), np.reshape(controlled_w,
                                                                                     (len(controlled_w), 1))
    return treated_w, controlled_w


def weighted_mean(x, w):
    # input: x: n * d, w: n * 1
    # output: d
    x_w = np.multiply(x, w)
    n_w = np.sum(w)  # w.sum()
    m_w = np.sum(x_w, axis=0) / n_w
    return m_w


def weighted_var(x, w):
    # x: n * d, w: n * 1
    m_w = weighted_mean(x, w)  # d
    # nw, nsw = w.sum(), (w ** 2).sum()
    nw, nsw = np.sum(w), np.sum(w ** 2)
    var = np.multiply((x - m_w) ** 2, w)  # n*d
    var = np.sum(var, axis=0) * (nw / (nw ** 2 - nsw))
    return var


def smd_func(x1, w1, x0, w0, abs=True):
    w_mu1, w_var1 = weighted_mean(x1, w1), weighted_var(x1, w1)
    w_mu0, w_var0 = weighted_mean(x0, w0), weighted_var(x0, w0)
    VAR_w = np.sqrt((w_var1 + w_var0) / 2)
    smd_result = np.divide(
        (w_mu1 - w_mu0),
        VAR_w, out=np.zeros_like(w_mu1), where=VAR_w != 0)
    if abs:
        smd_result = np.abs(smd_result)
    return smd_result


def cal_deviation(hidden_val, golds_treatment, logits_treatment, normalized, verbose=1):
    # covariates, and IPTW
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    treated_w, controlled_w = cal_weights(golds_treatment, logits_treatment, normalized=normalized)
    if verbose:
        print('In cal_deviation: n_treated:{}, n_treated_w:{} |'
              'n_controlled:{}, n_controlled_w:{} |'
              'n:{}, n_w:{}'.format(len(treated_w), treated_w.sum(), len(controlled_w), controlled_w.sum(),
                                    len(golds_treatment), treated_w.sum() + controlled_w.sum()))
    hidden_val = np.asarray(hidden_val)  # original covariates, to be weighted
    hidden_treated, hidden_controlled = hidden_val[ones_idx], hidden_val[zeros_idx]

    # Original SMD
    hidden_treated_mu, hidden_treated_var = np.mean(hidden_treated, axis=0), np.var(hidden_treated, axis=0, ddof=1)
    hidden_controlled_mu, hidden_controlled_var = np.mean(hidden_controlled, axis=0), np.var(hidden_controlled, axis=0,
                                                                                             ddof=1)
    VAR = np.sqrt((hidden_treated_var + hidden_controlled_var) / 2)
    # hidden_deviation = np.abs(hidden_treated_mu - hidden_controlled_mu) / VAR
    # hidden_deviation[np.isnan(hidden_deviation)] = 0  # -1  # 0  # float('-inf') represent VAR is 0
    hidden_deviation = np.divide(
        np.abs(hidden_treated_mu - hidden_controlled_mu),
        VAR, out=np.zeros_like(hidden_treated_mu), where=VAR != 0)

    max_unbalanced_original = np.max(hidden_deviation)

    # Weighted SMD
    hidden_treated_w_mu, hidden_treated_w_var = weighted_mean(hidden_treated, treated_w), weighted_var(hidden_treated,
                                                                                                       treated_w)
    hidden_controlled_w_mu, hidden_controlled_w_var = weighted_mean(hidden_controlled, controlled_w), weighted_var(
        hidden_controlled, controlled_w)
    VAR_w = np.sqrt((hidden_treated_w_var + hidden_controlled_w_var) / 2)
    # hidden_deviation_w = np.abs(hidden_treated_w_mu - hidden_controlled_w_mu) / VAR_w
    # hidden_deviation_w[np.isnan(hidden_deviation_w)] = 0  # -1  # 0
    hidden_deviation_w = np.divide(
        np.abs(hidden_treated_w_mu - hidden_controlled_w_mu),
        VAR_w, out=np.zeros_like(hidden_treated_w_mu), where=VAR_w != 0)

    max_unbalanced_weighted = np.max(hidden_deviation_w)

    return max_unbalanced_original, hidden_deviation, max_unbalanced_weighted, hidden_deviation_w


def cal_ATE(golds_treatment, logits_treatment, golds_outcome, normalized):
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    treated_w, controlled_w = cal_weights(golds_treatment, logits_treatment, normalized)
    if len(golds_outcome.shape) == 1:
        treated_outcome, controlled_outcome = golds_outcome[ones_idx], golds_outcome[zeros_idx]
    elif len(golds_outcome.shape) == 2:
        treated_outcome, controlled_outcome = golds_outcome[ones_idx, 0], golds_outcome[zeros_idx, 0]
        treated_outcome[treated_outcome == -1] = 0  # censor as 0
        controlled_outcome[controlled_outcome == -1] = 0
    else:
        raise ValueError

    treated_outcome_w = np.multiply(treated_outcome, treated_w.squeeze())
    controlled_outcome_w = np.multiply(controlled_outcome, controlled_w.squeeze())

    # ATE original
    UncorrectedEstimator_EY1_val, UncorrectedEstimator_EY0_val = np.mean(treated_outcome), np.mean(controlled_outcome)
    ATE = UncorrectedEstimator_EY1_val - UncorrectedEstimator_EY0_val

    # ATE weighted
    IPWEstimator_EY1_val, IPWEstimator_EY0_val = treated_outcome_w.sum() / treated_w.sum(), controlled_outcome_w.sum() / controlled_w.sum()
    ATE_w = IPWEstimator_EY1_val - IPWEstimator_EY0_val

    # NMI code for bias reference
    IPWEstimator_EY1_val_old, IPWEstimator_EY0_val_old = np.mean(treated_outcome_w), np.mean(controlled_outcome_w)
    ATE_w_old = IPWEstimator_EY1_val_old - IPWEstimator_EY0_val_old

    return (UncorrectedEstimator_EY1_val, UncorrectedEstimator_EY0_val, ATE), (
        IPWEstimator_EY1_val, IPWEstimator_EY0_val, ATE_w)


def cal_survival_KM(golds_treatment, logits_treatment, golds_outcome, normalized):
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    treated_w, controlled_w = cal_weights(golds_treatment, logits_treatment, normalized)
    if len(golds_outcome.shape) == 2:
        treated_outcome, controlled_outcome = golds_outcome[ones_idx, 0], golds_outcome[zeros_idx, 0]
        treated_outcome[treated_outcome == -1] = 0
        controlled_outcome[controlled_outcome == -1] = 0
    else:
        raise ValueError

    # kmf = KaplanMeierFitter()
    T = golds_outcome[:, 1]
    treated_t2e, controlled_t2e = T[ones_idx], T[zeros_idx]
    kmf1 = KaplanMeierFitter(label='Treated').fit(T[ones_idx], event_observed=treated_outcome, label="Treated")
    kmf0 = KaplanMeierFitter(label='Control').fit(T[zeros_idx], event_observed=controlled_outcome, label="Control")

    point_in_time = [180, 365, 540, 730]
    results = survival_difference_at_fixed_point_in_time_test(point_in_time, kmf1, kmf0)
    # results.print_summary()
    survival_1 = kmf1.predict(point_in_time).to_numpy()
    survival_0 = kmf0.predict(point_in_time).to_numpy()
    ate = survival_1 - survival_0
    ate_p = results.p_value

    kmf1_w = KaplanMeierFitter(label='Treated_IPTW').fit(T[ones_idx], event_observed=treated_outcome,
                                                         label="Treated_IPTW", weights=treated_w)
    kmf0_w = KaplanMeierFitter(label='Control_IPTW').fit(T[zeros_idx], event_observed=controlled_outcome,
                                                         label="Control_IPTW", weights=controlled_w)
    results_w = survival_difference_at_fixed_point_in_time_test(point_in_time, kmf1_w, kmf0_w)
    # results_w.print_summary()
    survival_1_w = kmf1_w.predict(point_in_time).to_numpy()
    survival_0_w = kmf0_w.predict(point_in_time).to_numpy()
    ate_w = survival_1_w - survival_0_w
    ate_w_p = results_w.p_value
    # ax = plt.subplot(111)
    # kmf1.plot_survival_function(ax=ax)
    # kmf0.plot_survival_function(ax=ax)
    # kmf1_w.plot_survival_function(ax=ax)
    # kmf0_w.plot_survival_function(ax=ax)
    # plt.show()

    # cox for hazard ratio
    cph = CoxPHFitter()
    event = golds_outcome[:, 0]
    event[event == -1] = 0
    weight = np.zeros(len(golds_treatment))
    weight[ones_idx] = treated_w.squeeze()
    weight[zeros_idx] = controlled_w.squeeze()
    cox_data = pd.DataFrame({'T': T, 'event': event, 'treatment': golds_treatment, 'weights': weight})
    try:
        cph.fit(cox_data, 'T', 'event', weights_col='weights', robust=True)
        HR = cph.hazard_ratios_['treatment']
        CI = np.exp(cph.confidence_intervals_.values.reshape(-1))

        cph_ori = CoxPHFitter()
        cox_data_ori = pd.DataFrame({'T': T, 'event': event, 'treatment': golds_treatment})
        cph_ori.fit(cox_data_ori, 'T', 'event')
        HR_ori = cph_ori.hazard_ratios_['treatment']
        CI_ori = np.exp(cph_ori.confidence_intervals_.values.reshape(-1))
    except:
        cph = HR = CI = None
        cph_ori = HR_ori = CI_ori = None

    return (kmf1, kmf0, ate, survival_1, survival_0, results), \
        (kmf1_w, kmf0_w, ate_w, survival_1_w, survival_0_w, results_w), \
        (HR_ori, CI_ori, cph_ori), \
        (HR, CI, cph)
