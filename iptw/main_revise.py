import sys
# for linux env.
sys.path.insert(0, '..')
import time
from dataset import *
import pickle
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from evaluation import *
import torch.nn.functional as F
import os
from utils import save_model, load_model, check_and_mkdir
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
from ipreprocess.utils import load_icd_to_ccw
from PSModels import mlp, lstm, ml
import itertools
from tqdm import tqdm

import functools
print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--data_dir', type=str, default='../ipreprocess/output/save_cohort_all_loose/')
    parser.add_argument('--treated_drug', type=str)
    parser.add_argument('--controlled_drug', choices=['atc', 'random'], default='random')
    parser.add_argument('--controlled_drug_ratio', type=int, default=3)  # 2 seems not good. keep unchanged
    parser.add_argument("--random_seed", type=int, default=0)

    parser.add_argument('--run_model', choices=['LSTM', 'LR', 'MLP', 'XGBOOST', 'LIGHTGBM'], default='LR')
    parser.add_argument('--med_code_topk', type=int, default=200)
    parser.add_argument('--drug_coding', choices=['rxnorm', 'gpi'], default='rxnorm')
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--stats_exit', action='store_true')
    # Deep PSModels
    parser.add_argument('--batch_size', type=int, default=256)  #768)  # 64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # 0.001
    parser.add_argument('--weight_decay', type=float, default=1e-6)  # )0001)
    parser.add_argument('--epochs', type=int, default=15)  # 30
    # LSTM
    parser.add_argument('--diag_emb_size', type=int, default=128)
    parser.add_argument('--med_emb_size', type=int, default=128)
    parser.add_argument('--med_hidden_size', type=int, default=64)
    parser.add_argument('--diag_hidden_size', type=int, default=64)
    parser.add_argument('--lstm_hidden_size', type=int, default=100)
    # MLP
    parser.add_argument('--hidden_size', type=str, default='', help=', delimited integers')
    # Output
    parser.add_argument('--output_dir', type=str, default='output/debug/')

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
    print('device: ', args.device)
    print('Drug {} cohort: '.format(args.treated_drug))
    print('save_model_filename', args.save_model_filename)

    # %% 1. Load Data
    ## load drug code name mapping
    if args.drug_coding.lower() == 'gpi':
        _fname = os.path.join(args.data_dir, '../_gpi_ingredients_nameset_cnt.pkl')
        print('drug_name file: ', _fname)
        with open(_fname, 'rb') as f:
            # change later, move this file to pickles also
            gpiing_names_cnt = pickle.load(f)
            drug_name = {}
            for key, val in gpiing_names_cnt.items():
                drug_name[key] = '/'.join(val[0])
        print('Using GPI vocabulary, len(drug_name) :', len(drug_name))
        print('trial drug:', args.treated_drug, drug_name.get(args.treated_drug))

    else:
        with open(r'pickles/rxnorm_label_mapping.pkl', 'rb') as f:
            drug_name = pickle.load(f)
            print('Using rxnorm_cui vocabulary, len(drug_name) :', len(drug_name))
            print('trial drug:', args.treated_drug, drug_name.get(args.treated_drug))
    # Load diagnosis code mapping
    icd_to_ccw, icd_to_ccwname, ccw_info = load_icd_to_ccw('../ipreprocess/mapping/CCW_to_use_enriched.json')
    dx_name = ccw_info[1]

    # Load treated triple list and build control triple list
    treated = pickle.load(open(args.data_dir + args.treated_drug + '.pkl', 'rb'))
    controlled = []
    drugfile_in_dir = sorted([x for x in os.listdir(args.data_dir) if
                              (x.split('.')[0].isdigit() and x.split('.')[1] == 'pkl')])
    drug_in_dir = [x.split('.')[0] for x in drugfile_in_dir]
    cohort_size = pickle.load(open(os.path.join(args.data_dir, 'cohorts_size.pkl'), 'rb'))

    # 1-A: build control groups
    n_control_patient = 0
    controlled_drugs_range = []
    n_treat_patient = cohort_size.get(args.treated_drug + '.pkl')
    if args.controlled_drug == 'random':
        print('Control groups: random')
        # sorted for deterministic, listdir seems return randomly
        controlled_drugs = sorted(list(
            set(drugfile_in_dir) -
            set(args.treated_drug + '.pkl')
        ))
        np.random.shuffle(controlled_drugs)

        for c_id in controlled_drugs:
            # n_control_patient += cohort_size.get(c_id)
            # controlled_drugs_range.append(c_id)
            # if n_control_patient >= (args.controlled_drug_ratio + 1) * n_treat_patient:
            #     break

            controlled_drugs_range.append(c_id)
            c = pickle.load(open(args.data_dir + c_id, 'rb'))
            n_c = 0
            n_c_exclude = 0
            for p in c:
                p_drug = sum(p[1][0], [])
                if args.treated_drug not in p_drug:
                    controlled.append(p)
                    n_c += 1
                else:
                    n_c_exclude += 1
            n_control_patient += n_c
            if n_control_patient >= (args.controlled_drug_ratio + 1) * n_treat_patient:
                break

    else:
        print('Control groups: atc level2')
        ATC2DRUG = pickle.load(open(os.path.join('pickles/', 'atcL2_rx.pkl'), 'rb'))  # ATC2DRUG.pkl
        DRUG2ATC = pickle.load(open(os.path.join('pickles/', 'rx_atcL2.pkl'), 'rb'))  # DRUG2ATC.pkl

        if args.drug_coding.lower() == 'rxnorm':
            # if args.stats:
            ## atc drug statistics:
            in_atc = np.array([x in DRUG2ATC for x in drug_in_dir])
            print('Total drugs in dir: {}, {} ({:.2f}%) have atc mapping, {} ({:.2f}%) have not'.format(
                in_atc.shape[0],
                in_atc.sum(),
                in_atc.mean() * 100,
                in_atc.shape[0] - in_atc.sum(),
                (1 - in_atc.mean()) * 100,
            ))
            print('{} rxnorm codes without atc in DRUG2ATC are:\n'.format(len(set(drug_in_dir) - set(DRUG2ATC.keys()))),
                  set(drug_in_dir) - set(DRUG2ATC.keys()))
            ###

        atc_group = set()
        if args.drug_coding.lower() == 'gpi':
            drug_atc = args.treated_drug[:2]
            for d in drug_in_dir:
                if d[:2] == drug_atc:
                    atc_group.add(d)
        else:
            drug_atc = DRUG2ATC.get(args.treated_drug, [])
            for atc in drug_atc:
                if atc in ATC2DRUG:
                    atc_group.update(ATC2DRUG.get(atc))

        if len(atc_group) > 1:
            # atc control may not have n_treat * ratio number of patients
            controlled_drugs = [drug + '.pkl' for drug in atc_group if drug != args.treated_drug]
            controlled_drugs = sorted(list(
                set(drugfile_in_dir) -
                set(args.treated_drug + '.pkl') &
                set(controlled_drugs)
            ))
            np.random.shuffle(controlled_drugs)
        else:
            print("No atcl2 drugs for treated_drug {}, choose random".format(args.treated_drug))
            # all_atc = set(ATC2DRUG.keys()) - set(drug_atc)
            # sample_atc = [atc for atc in list(all_atc) if len(ATC2DRUG.get(atc)) == 1]
            # sample_drug = set()
            # for atc in sample_atc:
            #     for drug in ATC2DRUG.get(atc):
            #         sample_drug.add(drug)
            # controlled_drugs_range = [drug + '.pkl' for drug in sample_drug if drug != args.treated_drug]
            controlled_drugs = sorted(list(
                set(drugfile_in_dir) -
                set(args.treated_drug + '.pkl')
            ))
            np.random.shuffle(controlled_drugs)
            # n_control_patient = 0
            # controlled_drugs_range = []
            # n_treat_patient = cohort_size.get(args.treated_drug + '.pkl')
            # for c_id in controlled_drugs:
            #     n_control_patient += cohort_size.get(c_id)
            #     controlled_drugs_range.append(c_id)
            #     if n_control_patient >= (args.controlled_drug_ratio + 1) * n_treat_patient:
            #         break
        for c_id in controlled_drugs:
            controlled_drugs_range.append(c_id)
            c = pickle.load(open(args.data_dir + c_id, 'rb'))
            n_c = 0
            for p in c:
                p_drug = sum(p[1][0], [])
                if args.treated_drug not in p_drug:
                    controlled.append(p)
                    n_c += 1
            n_control_patient += n_c
            if n_control_patient >= (args.controlled_drug_ratio + 1) * n_treat_patient:
                break

    # for c_drug_id in controlled_drugs_range:
    #     c = pickle.load(open(args.data_dir + c_drug_id, 'rb'))
    #     controlled.extend(c)

    intersect = set(np.asarray(treated)[:, 0]).intersection(set(np.asarray(controlled)[:, 0]))
    controlled = np.asarray([controlled[i] for i in range(len(controlled)) if controlled[i][0] not in intersect])

    controlled_indices = list(range(len(controlled)))
    controlled_sample_index = int(args.controlled_drug_ratio * len(treated))

    np.random.shuffle(controlled_indices)

    controlled_sample_indices = controlled_indices[:controlled_sample_index]

    controlled_sample = controlled[controlled_sample_indices]

    n_user, n_nonuser = len(treated), len(controlled_sample)
    print('#treated: {}, #controls: {}'.format(n_user, n_nonuser),
          '\nactual control/treated:', n_nonuser/n_user,
          '\n(Warning: the args.controlled_drug_ratio is {},'
          ' and the atc control cohort may have less patients than expected)'.format(args.controlled_drug_ratio))

    # 1-B: calculate the statistics of treated v.s. control
    if args.stats or args.stats_exit:
        # demo_feature_vector: [age, sex, race, days_since_mci]
        # triple = (patient,
        #           [rx_codes, dx_codes, demo_feature_vector[0], demo_feature_vector[1], demo_feature_vector[3]],
        #           (outcome, outcome_t2e))
        print('Summarize statistics between treat v.s. control ...')
        from ipreprocess.univariate_statistics import build_patient_characteristics_from_triples, \
            statistics_for_treated_control, build_patient_characteristics_from_triples_v2

        with open('../ipreprocess/mapping/CCW_AD_comorbidity.json') as f:
            ccw_ad_comorbidity = json.load(f)
        ccwcomorid_name = {}
        for name, v in ccw_ad_comorbidity.items():
            ccwcomorid_name[ccw_info[0][name]] = name

        atc2rxnorm = pickle.load(open(os.path.join('pickles', 'atcL2_rx.pkl'), 'rb'))  # ATC2DRUG.pkl
        if args.drug_coding.lower() == 'gpi':
            is_antidiabetic = lambda x: (x[:2] == '27')
            is_antihypertensives = lambda x: (x[:2] == '36')
        else:
            is_antidiabetic = lambda x: x in atc2rxnorm['A10']
            is_antihypertensives = lambda x: x in atc2rxnorm['C02']
        drug_criterion = {'antidiabetic': is_antidiabetic, 'antihypertensives': is_antihypertensives}
        # df_treated = build_patient_characteristics_from_triples(treated, ccwcomorid_name, drug_criterion)
        df_treated = build_patient_characteristics_from_triples_v2(treated, ccwcomorid_name, drug_criterion)
        # df_control = build_patient_characteristics_from_triples(controlled_sample, ccwcomorid_name, drug_criterion)
        df_control = build_patient_characteristics_from_triples_v2(controlled_sample, ccwcomorid_name, drug_criterion)
        add_row = pd.Series({'treat': args.treated_drug,
                             'control': ';'.join([x.split('.')[0] for x in controlled_drugs_range]),
                             'p-value': np.nan},
                            name='file')
        df_stats = statistics_for_treated_control(
            df_treated,
            df_control,
            args.save_model_filename + '_stats.csv',
            add_row)
        print('Characteristics statistic of treated v.s. control, done!')
        if args.stats_exit:
            print('Only run stats! stats_exit! Total Time used:',
                  time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            sys.exit(5)  # stats_exit only run statas

    # 1-C: build pytorch dataset
    print("Constructed Dataset, choose med_code_topk:", args.med_code_topk)
    my_dataset = Dataset(treated, controlled_sample,
                         med_code_topk=args.med_code_topk,
                         diag_name=dx_name,
                         med_name=drug_name)  # int(len(treated)/5)) #150)

    n_feature = my_dataset.DIM_OF_CONFOUNDERS  # my_dataset.med_vocab_length + my_dataset.diag_vocab_length + 3
    feature_name = my_dataset.FEATURE_NAME
    print('n_feature: ', n_feature, ':')
    # print(feature_name)

    train_ratio = 0.7  # 0.5
    val_ratio = 0.1
    print('train_ratio: ', train_ratio,
          'val_ratio: ', val_ratio,
          'test_ratio: ', 1 - (train_ratio + val_ratio))

    dataset_size = len(my_dataset)
    indices = list(range(dataset_size))
    train_index = int(np.floor(train_ratio * dataset_size))
    val_index = int(np.floor(val_ratio * dataset_size))

    np.random.shuffle(indices)

    train_indices, val_indices, test_indices = indices[:train_index], \
                                               indices[train_index:train_index + val_index], \
                                               indices[train_index + val_index:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                             sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                              sampler=test_sampler)
    data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                              sampler=SubsetRandomSampler(indices))

    # %%  LSTM-PS PSModels
    if args.run_model == 'LSTM':
        print("**************************************************")
        print("**************************************************")
        print('LSTM-Attention PS PSModels learning:')
        # paras_grid = {
        #     'hidden_size': [32, 64, 100, 128], #100
        #     'lr': [1e-3],
        #     'weight_decay': [1e-6],
        #     'batch_size': [32, 50, 64, 128],  #50
        # }
        paras_grid = {
            'hidden_size': [32, 64, 100, 128],  # 100
            'lr': [1e-3],
            'weight_decay': [1e-6],
            'batch_size': [50],  # 50
        }
        hyper_paras_names, hyper_paras_v = zip(*paras_grid.items())
        hyper_paras_list = list(itertools.product(*hyper_paras_v))
        print('Model {} Searching Space N={}: '.format(args.run_model, len(hyper_paras_list)), paras_grid)

        best_hyper_paras = None
        best_model = None
        best_auc = float('-inf')
        best_balance = float('inf')
        global_best_auc = float('-inf')
        global_best_balance = float('inf')
        best_model_epoch = -1
        results = []
        i = -1
        i_iter = -1
        for hyper_paras in tqdm(hyper_paras_list):
            i += 1
            hidden_size, lr, weight_decay, batch_size = hyper_paras
            print('In hyper-paras space [{}/{}]...'.format(i, len(hyper_paras_list)))
            print(hyper_paras_names)
            print(hyper_paras)

            train_loader_shuffled = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                                                sampler=train_sampler)
            print('len(train_loader_shuffled): ', len(train_loader_shuffled),
                  'train_loader_shuffled.batch_size: ', train_loader_shuffled.batch_size)

            model_params = dict(
                med_hidden_size=args.med_hidden_size,  # 64
                diag_hidden_size=args.diag_hidden_size,  # 64
                hidden_size=hidden_size,  # 100,
                bidirectional=True,
                med_vocab_size=len(my_dataset.med_code_vocab),
                diag_vocab_size=len(my_dataset.diag_code_vocab),
                diag_embedding_size=args.diag_emb_size,  # 128
                med_embedding_size=args.med_emb_size,  # 128
                # end_index=my_dataset.diag_code_vocab[CodeVocab.END_CODE],
                # pad_index=my_dataset.diag_code_vocab[CodeVocab.PAD_CODE],
            )
            print('Model: LSTM')
            print(model_params)
            model = lstm.LSTMModel(**model_params)
            if args.cuda:
                model = model.to('cuda')
            print(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

            for epoch in tqdm(range(args.epochs)):
                i_iter += 1
                epoch_losses_ipw = []
                for confounder, treatment, outcome in train_loader_shuffled:
                    model.train()
                    # train IPW
                    optimizer.zero_grad()

                    if args.cuda:  # confounder = (diag, med, sex, age)
                        for ii in range(len(confounder)):
                            confounder[ii] = confounder[ii].to('cuda')
                        treatment = treatment.to('cuda')

                    treatment_logits, _ = model(confounder)
                    loss_ipw = F.binary_cross_entropy_with_logits(treatment_logits, treatment.float())

                    loss_ipw.backward()
                    optimizer.step()
                    epoch_losses_ipw.append(loss_ipw.item())

                # just finish 1 epoch
                # scheduler.step()
                epoch_losses_ipw = np.mean(epoch_losses_ipw)

                loss_train, T_train, PS_logits_train, Y_train, X_train = transfer_data_lstm(model, train_loader, cuda=args.cuda)
                loss_val, T_val, PS_logits_val, Y_val, X_val = transfer_data_lstm(model, val_loader, cuda=args.cuda)
                loss_test, T_test, PS_logits_test, Y_test, X_test = transfer_data_lstm(model, test_loader, cuda=args.cuda)

                result_train = _evaluation_helper(X_train, T_train, PS_logits_train, loss_train)
                result_val = _evaluation_helper(X_val, T_val, PS_logits_val, loss_val)
                result_test = _evaluation_helper(X_test, T_test, PS_logits_test, loss_test)

                n_train = len(X_train)
                n_val = len(X_val)
                n_test = len(X_test)

                result_trainval = _evaluation_helper(
                    np.concatenate((X_train, X_val)),
                    np.concatenate((T_train, T_val)),
                    np.concatenate((PS_logits_train, PS_logits_val)),
                    _loss_helper([loss_train, loss_val], [n_train, n_val])
                )

                result_all = _evaluation_helper(
                    np.concatenate((X_train, X_val, X_test)),
                    np.concatenate((T_train, T_val, T_test)),
                    np.concatenate((PS_logits_train, PS_logits_val, PS_logits_test)),
                    _loss_helper([loss_train, loss_val, loss_test], [n_train, n_val, n_test])
                )

                results.append(
                    (i_iter, i, epoch,
                     hyper_paras) + result_train + result_val + result_test + result_trainval + result_all)

                print('HP-i:{}, epoch:{}, loss:{}\n'
                      'trainval-balance:{}, all-balance:{}, val-auc:{}, test-auc:{}'.format(
                    i, epoch, epoch_losses_ipw, result_trainval[5], result_all[5], result_val[1], result_test[1])
                )

                if (result_trainval[5] < best_balance) or \
                        ((result_trainval[5] == best_balance) and (result_val[1] > best_auc)):
                    best_model = model
                    best_hyper_paras = hyper_paras
                    best_auc = result_val[1]
                    best_balance = result_trainval[5]
                    best_model_epoch = epoch
                    print('Save Best PSModel at Hyper-iter[{}/{}]'.format(i, len(hyper_paras_list)),
                          'Epoch: ', epoch, 'trainval-balance:', best_balance, 'val-auc:', best_auc,
                          "all-balance", result_all[5], "test-auc", result_test[1])
                    print(hyper_paras_names)
                    print(hyper_paras)
                    save_model(model, args.save_model_filename, model_params=model_params)

                if result_val[1] > global_best_auc:
                    global_best_auc = result_val[1]

                if result_trainval[5] <= global_best_balance:
                    global_best_balance = result_trainval[5]

        name = ['loss', 'auc', 'max_smd', 'n_unbalanced_feat', 'max_smd_iptw', 'n_unbalanced_feat_iptw']
        col_name = ['i', 'ipara', 'epoch', 'paras'] + \
                   [pre + x for pre in ['train_', 'val_', 'test_', 'trainval_', 'all_'] for x in name]
        results = pd.DataFrame(results, columns=col_name)

        print('Model selection finished! Save Global Best PSModel at Hyper-iter [{}/{}], Epoch: {}'.format(
            i, len(hyper_paras_list), best_model_epoch), 'trainval-balance:', best_balance, 'val-auc:', best_auc,
            'global_best_balance:', global_best_balance, 'global_best_auc:', global_best_auc)

        print(hyper_paras_names)
        print(best_hyper_paras)

        results.to_csv(args.save_model_filename + '_ALL-model-select.csv')
        results_all_list, results_all_df = final_eval_deep(lstm.LSTMModel, args, train_loader, val_loader, test_loader,
                                                           data_loader,
                                                           drug_name, feature_name, n_feature,
                                                           dump_ori=False, lstm=True)

    # %% Logistic regression PS PSModels
    if args.run_model in ['LR', 'XGBOOST', 'LIGHTGBM']:
        print("**************************************************")
        print(args.run_model, ' PS model learning:')

        # print('Train data:')
        # train_x, train_t, train_y = flatten_data(my_dataset, train_indices)
        # print('Validation data:')
        # val_x, val_t, val_y = flatten_data(my_dataset, val_indices)
        # print('Test data:')
        # test_x, test_t, test_y = flatten_data(my_dataset, test_indices)
        print('All data:')
        x, t, y = flatten_data(my_dataset, indices)  # all the data

        # put fixed parameters also into a list e.g. 'objective' : ['binary',]
        if args.run_model == 'LR':
            paras_grid = {
                'penalty': ['l1', 'l2'],
                'C': 10 ** np.arange(-3, 3, 0.5), #0.2),  # 'C': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20],
                'max_iter': [200],  # [100, 200, 500],
                'random_state': [args.random_seed],
            }
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
                'learning_rate': np.arange(0.01, 1, 0.1),
                'num_leaves': np.arange(5, 50, 10),
                'min_child_samples': [200, 250, 300],
                'random_state': [args.random_seed],
            }
        else:
            paras_grid = {}

        # ----2. Learning IPW using PropensityEstimator
        # model = ml.PropensityEstimator(args.run_model, paras_grid).fit(train_x, train_t, val_x, val_t)
        # model = ml.PropensityEstimator(args.run_model, paras_grid).fit_and_test(train_x, train_t, val_x, val_t, test_x,
        #                                                                         test_t)

        model = ml.PropensityEstimator(args.run_model, paras_grid, random_seed=args.random_seed).cross_validation_fit(
            x, t, verbose=0)

        with open(args.save_model_filename, 'wb') as f:
            pickle.dump(model, f)

        model.results.to_csv(args.save_model_filename + '_ALL-model-select.csv')
        model.results_agg.to_csv(args.save_model_filename + '_ALL-model-select-agg.csv')
        # ----3. Evaluation learned PropensityEstimator
        # results_all_list, results_all_df = final_eval_ml(model, args, train_x, train_t, train_y, val_x, val_t, val_y,
        #                                                  test_x, test_t, test_y, x, t, y,
        #                                                  drug_name, feature_name, n_feature, dump_ori=False)
        results_all_list, results_all_df = final_eval_ml_CV_revise(model, args,  x, t, y,
                                                                   drug_name, feature_name, n_feature, dump_ori=False)

    # %%  MLP PS Models
    if args.run_model == 'MLP':
        print("**************************************************")
        print("**************************************************")
        print(args.run_model, ' PS model learning:')

        # PSModels configuration & training
        # paras_grid = {
        #     'hidden_size': [0, 32, 64, 128],
        #     'lr': [1e-2, 1e-3, 1e-4],
        #     'weight_decay': [1e-4, 1e-5, 1e-6],
        #     'batch_size': [32, 64, 128],
        #     'dropout': [0.5],
        # }
        paras_grid = {
            'hidden_size': [32, 64, 128],
            'lr': [1e-3],
            'weight_decay': [1e-5],
            'batch_size': [64],
            'dropout': [0.5],
        }
        hyper_paras_names, hyper_paras_v = zip(*paras_grid.items())
        hyper_paras_list = list(itertools.product(*hyper_paras_v))
        print('Model {} Searching Space N={}: '.format(args.run_model, len(hyper_paras_list)), paras_grid)

        best_hyper_paras = None
        best_model = None
        best_auc = float('-inf')
        best_balance = float('inf')
        global_best_auc = float('-inf')
        global_best_balance = float('inf')
        best_model_epoch = -1
        results = []
        i = -1
        i_iter = -1
        for hyper_paras in tqdm(hyper_paras_list):
            i += 1
            hidden_size, lr, weight_decay, batch_size, dropout = hyper_paras
            print('In hyper-paras space [{}/{}]...'.format(i, len(hyper_paras_list)))
            print(hyper_paras_names)
            print(hyper_paras)

            train_loader_shuffled = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                                                sampler=train_sampler)
            print('len(train_loader_shuffled): ', len(train_loader_shuffled),
                  'train_loader_shuffled.batch_size: ', train_loader_shuffled.batch_size)

            model_params = dict(input_size=n_feature, num_classes=2, hidden_size=hidden_size, dropout=dropout, )
            print('Model: MLP')
            print(model_params)
            model = mlp.MLP(**model_params)
            if args.cuda:
                model = model.to('cuda')
            print(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            # scheduler = torch.optim.lr_scheduler.StepLR( optimizer, step_size=5, gamma=0.1)
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.85, verbose=True)

            for epoch in tqdm(range(args.epochs)):
                i_iter += 1
                epoch_losses_ipw = []
                for confounder, treatment, outcome in train_loader_shuffled:
                    model.train()
                    # train IPW
                    optimizer.zero_grad()

                    dx, rx, sex, age, days = confounder[0], confounder[1], confounder[2], confounder[3], confounder[4]
                    dx, rx = torch.sum(dx, 1), torch.sum(rx, 1)
                    dx = torch.where(dx > 0, 1., 0.)
                    rx = torch.where(rx > 0, 1., 0.)
                    X = torch.cat((dx, rx, sex.unsqueeze(1), age.unsqueeze(1), days.unsqueeze(1)), 1)
                    if args.cuda:
                        X = X.float().to('cuda')
                        treatment = treatment.to('cuda')

                    treatment_logits = model(X)
                    # loss_ipw = F.binary_cross_entropy_with_logits(treatment_logits, treatment.float())
                    loss_ipw = F.cross_entropy(treatment_logits, treatment)
                    loss_ipw.backward()
                    optimizer.step()
                    epoch_losses_ipw.append(loss_ipw.item())

                # just finish 1 epoch
                # scheduler.step()
                epoch_losses_ipw = np.mean(epoch_losses_ipw)

                loss_train, T_train, PS_logits_train, Y_train, X_train = transfer_data(model, train_loader, cuda=args.cuda)
                loss_val, T_val, PS_logits_val, Y_val, X_val = transfer_data(model, val_loader, cuda=args.cuda)
                loss_test, T_test, PS_logits_test, Y_test, X_test = transfer_data(model, test_loader, cuda=args.cuda)

                result_train = _evaluation_helper(X_train, T_train, PS_logits_train, loss_train)
                result_val = _evaluation_helper(X_val, T_val, PS_logits_val, loss_val)
                result_test = _evaluation_helper(X_test, T_test, PS_logits_test, loss_test)

                n_train = len(X_train)
                n_val = len(X_val)
                n_test = len(X_test)

                result_trainval = _evaluation_helper(
                    np.concatenate((X_train, X_val)),
                    np.concatenate((T_train, T_val)),
                    np.concatenate((PS_logits_train, PS_logits_val)),
                    _loss_helper([loss_train, loss_val], [n_train, n_val])
                )

                result_all = _evaluation_helper(
                    np.concatenate((X_train, X_val, X_test)),
                    np.concatenate((T_train, T_val, T_test)),
                    np.concatenate((PS_logits_train, PS_logits_val, PS_logits_test)),
                    _loss_helper([loss_train, loss_val, loss_test], [n_train, n_val, n_test])
                )

                results.append(
                    (i_iter, i, epoch, hyper_paras) + result_train + result_val + result_test + result_trainval + result_all)

                print('HP-i:{}, epoch:{}, loss:{}\n'
                      'trainval-balance:{}, all-balance:{}, val-auc:{}, test-auc:{}'.format(
                    i, epoch, epoch_losses_ipw, result_trainval[5], result_all[5], result_val[1], result_test[1])
                )

                if (result_trainval[5] < best_balance) or \
                        ((result_trainval[5] == best_balance) and (result_val[1] > best_auc)):
                    best_model = model
                    best_hyper_paras = hyper_paras
                    best_auc = result_val[1]
                    best_balance = result_trainval[5]
                    best_model_epoch = epoch
                    print('Save Best PSModel at Hyper-iter[{}/{}]'.format(i, len(hyper_paras_list)),
                          'Epoch: ', epoch, 'trainval-balance:', best_balance, 'val-auc:', best_auc,
                          "all-balance", result_all[5], "test-auc", result_test[1])
                    print(hyper_paras_names)
                    print(hyper_paras)
                    save_model(model, args.save_model_filename, model_params=model_params)

                if result_val[1] > global_best_auc:
                    global_best_auc = result_val[1]

                if result_trainval[5] <= global_best_balance:
                    global_best_balance = result_trainval[5]

        name = ['loss', 'auc', 'max_smd', 'n_unbalanced_feat', 'max_smd_iptw', 'n_unbalanced_feat_iptw']
        col_name = ['i', 'ipara', 'epoch','paras'] + \
                   [pre + x for pre in ['train_', 'val_', 'test_', 'trainval_', 'all_'] for x in name]
        results = pd.DataFrame(results, columns=col_name)

        print('Model selection finished! Save Global Best PSModel at Hyper-iter [{}/{}], Epoch: {}'.format(
            i, len(hyper_paras_list), best_model_epoch), 'trainval-balance:', best_balance, 'val-auc:', best_auc,
            'global_best_balance:', global_best_balance, 'global_best_auc:', global_best_auc)

        print(hyper_paras_names)
        print(best_hyper_paras)

        results.to_csv(args.save_model_filename + '_ALL-model-select.csv')

        results_all_list, results_all_df = final_eval_deep(mlp.MLP, args, train_loader, val_loader, test_loader,
                                                           data_loader,
                                                           drug_name, feature_name, n_feature, dump_ori=False)

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
