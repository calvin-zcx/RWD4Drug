# import logging
import numpy as np
import torch.utils.data
from vocab import *
from tqdm import tqdm

# logger = logging.getLogger()
DEBUG = False


class Dataset(torch.utils.data.Dataset):
    def __init__(self, treated_patient_list, control_patient_list,
                 diag_code_threshold=None, diag_code_topk=None, diag_name=None,
                 med_code_threshold=None, med_code_topk=None, med_name=None):
        # diag_code_vocab=None, med_code_vocab=None,
        self.treated_patient_list = treated_patient_list
        self.control_patient_list = control_patient_list

        self.diagnoses_visits = []
        self.medication_visits = []
        self.sexes = []
        self.ages = []
        self.days_since_initial = []

        self.outcome = []
        self.treatment = []

        for _, patient_confounder, patient_outcome in tqdm(self.treated_patient_list):
            self.outcome.append(patient_outcome)
            self.treatment.append(1)
            med_visit, diag_visit, age, sex, days = patient_confounder
            self.medication_visits.append(med_visit)
            self.diagnoses_visits.append(diag_visit)
            self.sexes.append(sex)
            self.ages.append(age)
            self.days_since_initial.append(days)

        for _, patient_confounder, patient_outcome in tqdm(self.control_patient_list):
            self.outcome.append(patient_outcome)
            self.treatment.append(0)
            med_visit, diag_visit, age, sex, days = patient_confounder
            self.medication_visits.append(med_visit)
            self.diagnoses_visits.append(diag_visit)
            self.sexes.append(sex)
            self.ages.append(age)
            self.days_since_initial.append(days)

        self.diag_code_vocab = CodeVocab(diag_code_threshold, diag_code_topk, diag_name)
        self.diag_code_vocab.add_patients_visits(self.diagnoses_visits)

        self.med_code_vocab = CodeVocab(med_code_threshold, med_code_topk, med_name)
        self.med_code_vocab.add_patients_visits(self.medication_visits)

        # logger.info('Created Diagnoses Vocab: %s' % self.diag_code_vocab)
        # logger.info('Created Medication Vocab: %s' % self.med_code_vocab)
        print('Created Diagnoses Vocab: %s' % self.diag_code_vocab)
        print('Created Medication Vocab: %s' % self.med_code_vocab)

        self.diag_visit_max_length = max([len(patient_visit) for patient_visit in self.diagnoses_visits])
        self.med_visit_max_length = max([len(patient_visit) for patient_visit in self.medication_visits])

        self.diag_vocab_length = len(self.diag_code_vocab)
        self.med_vocab_length = len(self.med_code_vocab)

        # logger.info('Diagnoses Visit Max Length: %d' % self.diag_visit_max_length)
        # logger.info('Medication Visit Max Length: %d' % self.med_visit_max_length)
        print('Diagnoses Visit Max Length: %d' % self.diag_visit_max_length)
        print('Medication Visit Max Length: %d' % self.med_visit_max_length)

        if DEBUG:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            data_debug = pd.DataFrame(
                data={'treatment': self.treatment, 'age': self.ages, 'sex': self.sexes, 'days': self.days_since_initial})
            sns.histplot(data=data_debug, x='days', hue='treatment', kde=True, bins=25, stat="percent", common_norm=False)
            # sns.histplot(data=data_debug.loc[data_debug['treatment'] == 1, 'age'], kde=True, bins=25, stat="percent")
            # sns.histplot(data=data_debug.loc[data_debug['treatment'] == 0, 'age'], kde=True, bins=25, stat="percent")
            plt.show()

        # self.ages=np.abs(self.ages-np.mean(self.ages))/np.var(self.ages)
        self.ages = (self.ages - np.min(self.ages)) / np.ptp(self.ages)
        self.days_since_initial = (self.days_since_initial - np.min(self.days_since_initial)) / np.ptp(
            self.days_since_initial)
        self.outcome = np.asarray(self.outcome)

        # feature dim: med_visit, diag_visit, age, sex, days
        self.DIM_OF_CONFOUNDERS = len(self.med_code_vocab) + len(self.diag_code_vocab) + 3
        print('DIM_OF_CONFOUNDERS: ', self.DIM_OF_CONFOUNDERS)

        # feature name
        diag_col_name = self.diag_code_vocab.feature_name()
        med_col_name = self.med_code_vocab.feature_name()
        col_name = (diag_col_name, med_col_name, ['sex'], ['age'], ['days'])
        self.FEATURE_NAME = np.asarray(sum(col_name, []))

    def _process_visits(self, visits, max_len_visit, vocab):
        res = np.zeros((max_len_visit, len(vocab)))
        for i, visit in enumerate(visits):
            res[i] = self._process_code(vocab, visit)
        # col_name = [vocab.id2name.get(x, '') for x in range(len(vocab))]
        return res  # , col_name

    def _process_code(self, vocab, codes):
        multi_hot = np.zeros((len(vocab, )), dtype='float')
        for code in codes:
            if code in vocab.code2id:
                multi_hot[vocab.code2id[code]] = 1
        return multi_hot

    def _process_demo(self):
        # add race multi_hot
        # move ages code to here
        pass

    def __getitem__(self, index):
        # Problem: very sparse due to 1. padding a lots of 0, 2. original signals in high-dim.
        # paddedsequence for 1 and graph for 2?
        # should give new self._process_visits and self._process_visits
        # also add more demographics for confounder
        diag = self.diagnoses_visits[index]
        diag = self._process_visits(diag, self.diag_visit_max_length, self.diag_code_vocab)  # T_dx * D_dx

        med = self.medication_visits[index]
        med = self._process_visits(med, self.med_visit_max_length, self.med_code_vocab)  # T_drug * D_drug

        sex = self.sexes[index]
        age = self.ages[index]
        days = self.days_since_initial[index]
        # outcome = self.outcome[index][self.outcome_type]  # no time2event using in the matter rising
        outcome = self.outcome[index]

        treatment = self.treatment[index]

        confounder = (diag, med, sex, age, days)

        return confounder, treatment, outcome

    def __len__(self):
        return len(self.diagnoses_visits)
