import sys
# for linux env.
sys.path.insert(0, '..')
import os
os.environ['CASTLE_BACKEND'] = 'pytorch'
import time
from dataset import *
import pickle
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from evaluation import *
import torch.nn.functional as F

from utils import save_model, load_model, check_and_mkdir
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
from ipreprocess.utils import load_icd_to_ccw
from PSModels import mlp, lstm, ml
import itertools
from tqdm import tqdm
from sklearn.model_selection import KFold
from collections import OrderedDict

import numpy as np
import networkx as nx
import castle
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import PC, GES, ICALiNGAM, GOLEM
from castle.common.priori_knowledge import PrioriKnowledge

import functools
print = functools.partial(print, flush=True)


def get_mediators(G : nx.Graph, treatment, outcome,):
    ## outcome_ancestors = set(nx.ancestors(G, source=outcome))
    ## treatment_descendants = set(nx.descendants(G, source=treatment))
    ## mediators = treatment_descendants.intersection(outcome_ancestors)

    mediators = []
    paths = list(nx.all_simple_paths(G, source=treatment, target=outcome))
    for path in paths:
        if len(path) > 2:
            mediators.extend(path[1:-1])

    return set(mediators)


def get_colliders(G: nx.Graph, treatment, outcome,):
    outcome_descendants = set(nx.descendants(G, source=outcome))
    treatment_descendants = set(nx.descendants(G, source=treatment))
    possible_colliders = outcome_descendants.intersection(treatment_descendants)

    colliders = []
    for possible_collider in possible_colliders:
        paths = nx.all_simple_paths(G, treatment, possible_collider)
        for path in paths:
            if not outcome in path:
                colliders.append(possible_collider)
                break  # avoid duplicates
    return set(colliders)


def get_M_colliders(G: nx.Graph, treatment, outcome,):
    treatment_ancestors = set(nx.ancestors(G, source=treatment))
    outcome_ancestors = set(nx.ancestors(G, source=outcome))
    outcome_ancestors = outcome_ancestors - treatment_ancestors
    s1 = []
    s2 = []
    for x in treatment_ancestors:
        s1.extend(nx.descendants(G, source=x))
    for x in outcome_ancestors:
        s2.extend(nx.descendants(G, source=x))
    s1 = set(s1) - {'Y'}
    s2 = set(s2) - {'T'}
    possible_m = s1.intersection(s2) - outcome_ancestors - treatment_ancestors - {'T', 'Y'}
    return possible_m


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
    # causal discovery part
    parser.add_argument('--noDAGLearn', action='store_true')
    parser.add_argument('--adjustP', action='store_true')
    # Deep PSModels
    parser.add_argument('--batch_size', type=int, default=256)  #768)  # 64)
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
def flatten_data(mdata, data_indices, verbose=1, exclude_set={}):
    print('flatten_data...')
    cov_name = ['Alcohol Use Disorders', 'Anxiety Disorders', 'Depression', 'Diabetes', 'Heart Failure',
                'Hyperlipidemia', 'Hypertension', 'Ischemic Heart Disease', 'Obesity', 'Stroke / Transient Ischemic Attack',
                'Tobacco Use', 'Traumatic Brain Injury and Nonpsychotic Mental Disorders due to Brain Damage',
                'Sleep disorders', 'Periodontitis', 'Menopause']
    # cov_name = list(mdata.FEATURE_NAME[:64])
    if len(exclude_set) > 0:
        cov_name = [x for x in cov_name if x not in exclude_set]
        print('Using exclude_set, ', len(exclude_set), exclude_set)

    print('Finally using selected {} diagnosis covariates:'.format(len(cov_name)), cov_name)

    name_id = {item: key for key, item in mdata.diag_code_vocab.id2name.items()}
    cov_col = [name_id[x] for x in cov_name]

    x, t, y = [], [], []
    for idx in tqdm(data_indices):
        confounder, treatment, outcome = mdata[idx][0], mdata[idx][1], mdata[idx][2]
        dx, rx, sex, age, days = confounder[0], confounder[1], confounder[2], confounder[3], confounder[4]
        dx = np.sum(dx, axis=0)
        dx = np.where(dx > 0, 1, 0)
        # rx = np.sum(rx, axis=0)
        # rx = np.where(rx > 0, 1, 0)
        # x.append(np.concatenate((dx, rx, [sex], [age], [days])))

        x.append(np.concatenate(([sex], [age], dx[cov_col])))

        t.append(treatment)
        y.append(outcome)

    x, t, y = np.asarray(x), np.asarray(t), np.asarray(y)
    if verbose:
        print('...all:', x.shape, 'non-zero ratio:', (x != 0).mean(), 'all-zero:', (x.mean(0) == 0).sum())

    return x, t, y, ['sex', 'age'] + cov_name


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
        x, t, y, _col_name = flatten_data(my_dataset, indices)  # all the data

        if args.noDAGLearn:
            print('...Not using DAG learning algorithm to screen covs')
        else:
            print('...Using DAG learning algorithm to screen covs!')
            yb = np.copy(y[:, 0])
            yb [yb<=0] = 0
            cdata = np.hstack([t.reshape(-1,1), yb.reshape(-1,1), x])
            select_col_name = ['T', 'Y'] + _col_name

            priori = PrioriKnowledge(len(select_col_name))
            priori.add_required_edges([(0, 1), (2, 1), (3, 1), (2, 0), (3, 0)])
            priori.add_forbidden_edges([(1, 0), ]
                                       + [(1, x) for x in range(2, len(select_col_name))]  # all x before Y
                                       + [(x, 2) for x in range(0, len(select_col_name))]  # can not change sex
                                       + [(x, 3) for x in range(0, len(select_col_name))]  # can not change age
                                       )
            if args.adjustP:
                _alpha = 0.05/(len(select_col_name) * (len(select_col_name)-1)/2)
                print('......Using adjusted P value for PC, alpha=', _alpha)
            else:
                _alpha = 0.05
                print('......Not using adjusted P value for PC, alpha=', _alpha)

            model = PC(variant='stable',
                       alpha=_alpha,
                       priori_knowledge=priori)
            # model = GOLEM()
            model.learn(cdata)

            # Print out the learned matrix
            print(model.causal_matrix)
            # Get learned graph
            learned_graph = nx.DiGraph(model.causal_matrix)

            # Relabel the nodes
            MAPPING = {k: v for k, v in zip(range(len(select_col_name)), select_col_name)}
            learned_graph = nx.relabel_nodes(learned_graph, MAPPING, copy=True)

            colliders = get_colliders(learned_graph, 'T', 'Y')
            mediators = get_mediators(learned_graph, 'T', 'Y')
            m_colliders = get_M_colliders(learned_graph, 'T', 'Y')
            print('detected colliders:', len(colliders), colliders)
            print('detected mediators:', len(mediators), mediators)
            print('detected m_colliders:', len(m_colliders), m_colliders)

            print('Causal Discovery Part Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

            nx.write_gexf(learned_graph, args.save_model_filename + '_learned_DAG.gexf')
            exclude_col_med = {'colliders':colliders, 'mediators':mediators, 'm_colliders':m_colliders}
            with open(args.save_model_filename + '_detected_colliders_mediators.pkl', "wb") as outfile:
                pickle.dump(exclude_col_med, outfile)

            # Plot the graph
            ax = plt.subplot(111)
            ax.set_title('{}-{}-{}-{}-{}'.format(args.treated_drug, drug_name.get(args.treated_drug), args.controlled_drug, args.random_seed, args.run_model))
            nx.draw(
                learned_graph,
                with_labels=True,
                node_size=85,
                font_size=6,
                font_color='black',
                edge_color="grey",
                pos=nx.shell_layout(learned_graph)
            )

            plt.savefig(args.save_model_filename + '_learned_DAG.png')
            plt.show()
            plt.clf()

            # exclude detected features
            x, t, y, _col_name = flatten_data(my_dataset, indices, exclude_set=colliders | mediators | m_colliders)  # all the data

        # put fixed parameters also into a list e.g. 'objective' : ['binary',]
        if args.run_model == 'LR':
            paras_grid = {
                'penalty': ['l2'], # 'l1', not using l1 for this small number of covs
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
                'learning_rate': np.arange(0.01, 1, 0.25),
                'num_leaves': np.arange(5, 120, 20),
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

        # with open(args.save_model_filename, 'wb') as f:
        #     pickle.dump(model, f)

        model.results.to_csv(args.save_model_filename + '_ALL-model-select.csv')
        model.results_agg.to_csv(args.save_model_filename + '_ALL-model-select-agg.csv')
        # ----3. Evaluation learned PropensityEstimator
        # results_all_list, results_all_df = final_eval_ml(model, args, train_x, train_t, train_y, val_x, val_t, val_y,
        #                                                  test_x, test_t, test_y, x, t, y,
        #                                                  drug_name, feature_name, n_feature, dump_ori=False)
        results_all_list, results_all_df = final_eval_ml_CV_revise(model, args,  x, t, y,
                                                                   drug_name, feature_name, n_feature, dump_ori=False)

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


