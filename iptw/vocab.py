from collections import Counter


class CodeVocab(object):
    END_CODE = '<end>'
    PAD_CODE = '<pad>'
    UNK_CODE = '<unk>'

    def __init__(self, threshold=None, topK=None, code2name=None):
        super().__init__()
        # special_codes = None  # [CodeVocab.END_CODE, CodeVocab.PAD_CODE, CodeVocab.UNK_CODE]
        # self.special_codes = special_codes
        self.code2id = {}
        self.id2code = {}
        self.code2name = code2name  # new 2021-07-12
        self.id2name = {}  # new 2021-07-12

        self.threshold = threshold
        self.topk = topK
        self.code2count = Counter()

        # if self.special_codes is not None:
        #     self.add_code_list(self.special_codes)

    def add_code_list(self, code_list, rebuild=True):
        self.code2count.update(code_list)
        for code in code_list:
            if code not in self.code2id:
                self.code2id[code] = len(self.code2id)

        if rebuild:
            self._rebuild_id2code()

    def add_patients_visits(self, patients_visits):
        for patient in patients_visits:
            for visit in patient:
                self.add_code_list(visit, False)

        if (self.threshold is not None) or (self.topk is not None):
            self._select_most_common_codes()

        self._rebuild_id2code()

    def _rebuild_id2code(self):
        self.id2code = {i: t for t, i in self.code2id.items()}
        if self.code2name is not None:
            self.id2name = {i: self.code2name.get(t, '') for t, i in self.code2id.items()}

    def most_common(self, n=None):
        if isinstance(n, int) and n > 0:
            return self.code2count.most_common(n)
        return self.code2count.most_common()

    def _select_most_common_codes(self):
        n1 = 0
        n2 = 0

        if self.threshold is not None:
            for k, c in self.code2count.most_common():
                if c >= self.threshold:
                    n1 += 1

        if self.topk is not None:
            if self.topk > self.__len__():
                n2 = self.__len__()
            elif self.topk < 0:
                n2 = 0
            else:
                n2 = int(self.topk)

        n = max(n1, n2)
        if n <= 0:
            n = self.__len__()

        selected_codes, selected_codes_cnt = zip(*self.code2count.most_common(n))
        self.code2id = {}
        for code in selected_codes:
            if code not in self.code2id:
                self.code2id[code] = len(self.code2id)

    def get(self, item, default=None):
        return self.code2id.get(item, default)

    def feature_name(self):
        return [self.id2name.get(x, '') for x in range(len(self.code2id))]

    def __getitem__(self, item):
        return self.code2id[item]

    def __contains__(self, item):
        return item in self.code2id

    def __len__(self):
        return len(self.code2id)

    def __str__(self):
        return f'{len(self)} codes'
