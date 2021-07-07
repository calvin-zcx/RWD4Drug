import sys
# for linux env.
sys.path.insert(0,'..')
import pandas as pd
import time
from collections import defaultdict
import re
import pickle
import argparse
import csv
from utils import *
import functools
print = functools.partial(print, flush=True)

def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--input_file',
                        help=', delimited 1 or more drug prescribing files with same format:'
                             ' [0-patientid, 1-date, 2-code, 3-supply days, 4-drug name (if any)]',
                        default=r'../data/florida/dispensing.csv,../data/florida/prescribing.csv')
    # parser.add_argument('--output_file', help='.pkl file of drug taken by patients for cohort building',
    #                     default=r'output/mci_drug_taken_by_patient.pkl')
    args = parser.parse_args()
    return args


def load_drug_mappings_to_ingredients():
    with open(r'pickles/ndc_to_ingredient.pickle', 'rb') as f:
        ndc_to_ing = pickle.load(f)
    print(r'Load pickles/ndc_to_ingredient.pickle done：')
    print('***len(ndc_to_ing): ', len(ndc_to_ing))
    print('***unique ingredients: len(set(ndc_to_ing.values())): ', len(set(ndc_to_ing.values())))

    with open(r'pickles/rxnorm_to_ingredient.pickle', 'rb') as f:
        rxnorm_to_ing = pickle.load(f)
    print(r'Load pickles/rxnorm_to_ingredient.pickle done!')
    print('***len(rxnorm_to_ing): ', len(rxnorm_to_ing))
    print('***unique ingredients: len(set(rxnorm_to_ing.values())): ', len(set(rxnorm_to_ing.values())))

    print('unique ingredients of union set(ndc_to_ing.values()) | set(rxnorm_to_ing.values()):  ', len(
        set(ndc_to_ing.values()) | set(rxnorm_to_ing.values())
    ))
    return ndc_to_ing, rxnorm_to_ing


# %% Build patients drug table
def pre_drug_table(infile):
    """
    Input file format
    florida:
        dispensing
            PATID,DISPENSE_DATE,NDC,DISPENSE_SUP
            11e7506102e51266bb6a0050569ea8fb,2020-05-15,69315090601,10.0
            11e75065dd7117be8cac0050569ea8fb,2020-05-20,60505082901,30.0
        prescribing
            PATID,RX_START_DATE,RXNORM_CUI,RX_DAYS_SUPPLY,RAW_RX_MED_NAME
            11e827a2d4330c5691410050569ea8fb,2020-03-07,798230,5.0,PNEUMOCOCCAL 13-VAL CONJ VACC IM SUSP
            11e823e0622ac5429e470050569ea8fb,2020-02-27,9863,1.0,SODIUM CHLORIDE 0.9 PERCENT IV SOLN
    marketscan:
        ENROLID,FILLDATE,TCGPI_ID,DAYSUPP,DRUG_NAME
        247595501,2009-03-14,58200060100110,30,Nortriptyline HCl
        247595501,2009-11-04,58200060100110,30,Nortriptyline HCl
    :return: mci_drug_taken_by_patient
    """
    print('*******'*5)
    print('pre_drug_table from:', infile)
    ndc_to_ing, rxnorm_to_ing = load_drug_mappings_to_ingredients()
    mci_drug_taken_by_patient = defaultdict(dict)
    # Load from Dispensing table
    n_records = 0
    n_row_no_ingredient_map = 0
    n_no_day_supply = 0
    b_build_vocabulary = False
    with open(infile, 'r', encoding='utf-8') as f:
        col_name = next(csv.reader(f))  # read from first non-name row, above code from 2nd, wrong
        print('read from ', infile, col_name)
        if 'ndc' in col_name[2].lower():
            drug_2_ing = lambda x : ndc_to_ing.get(x)
            print('Using NDC to rxnorm_cui ingredient')
        elif 'rxnorm' in col_name[2].lower():
            drug_2_ing = lambda x: rxnorm_to_ing.get(x)
            print('Using rxnorm_cui to rxnorm_cui ingredient')
        elif 'gpi' in col_name[2].lower():
            # https://www.wolterskluwer.com/en/solutions/medi-span/about/gpi
            drug_2_ing = lambda x: x[:8]
            b_build_vocabulary = True
            gpi_name_cnt = dict()
            gpiing_names_cnt = dict()
            print('Using Drug Base Name of GPI string, namely its first 8 digits')
            print('Set building drug vocabulary flag as true')
        else:
            print('Unknown drug codes')
            raise ValueError

        for row in csv.reader(f):
            n_records += 1
            patid, date, drug, day = row[0], row[1], row[2], row[3]
            try:
                name = row[4]
            except:
                name = ''
            if date and day and (date != 'NULL') and (day != 'NULL'):
                day = int(float(day))
            else:
                day = -1
                n_no_day_supply += 1
            ing = drug_2_ing(drug)
            if ing:
                if ing not in mci_drug_taken_by_patient:
                    mci_drug_taken_by_patient[ing][patid] = {(date, day)}
                else:
                    if patid not in mci_drug_taken_by_patient.get(ing):
                        mci_drug_taken_by_patient[ing][patid] = {(date, day)}
                    else:
                        mci_drug_taken_by_patient[ing][patid].add((date, day))
            else:
                n_row_no_ingredient_map += 1

            if b_build_vocabulary:
                if drug in gpi_name_cnt:
                    gpi_name_cnt[drug][1] += 1
                else:
                    gpi_name_cnt[drug] = [name, 1]

                if ing:
                    if ing in gpiing_names_cnt:
                        gpiing_names_cnt[ing][1] += 1
                        gpiing_names_cnt[ing][0].add(name)
                    else:
                        gpiing_names_cnt[ing] = [{name}, 1]

    print('n_records: ', n_records)
    print('n_no_day_supply: ', n_no_day_supply)
    print('n_row_no_ingredient_map: ', n_row_no_ingredient_map)
    print('Scan n_records: ', n_records)
    print('# of Drugs: {}\t'.format(len(mci_drug_taken_by_patient)))

    try:
        print('dumping...', flush=True)
        il = re.split(r'[\/.]', infile)  #'../data/florida/prescribing.csv' -> ['', '', '', 'data', 'florida', 'prescribing', 'csv']
        ofile = 'output/_mci_drug_taken_by_patient_from_{}_{}.pkl'.format(il[-3], il[-2])
        check_and_mkdir(ofile)
        pickle.dump(mci_drug_taken_by_patient, open(ofile, 'wb'))
        print('dump {} done!'.format(ofile))
    except Exception as e:
        print(e)

    if b_build_vocabulary:
        try:
            # assume there is only one prescription file in marketscan
            pickle.dump(gpi_name_cnt, open('output/_gpi_drug_name_cnt.pkl', 'wb'))
            df = pd.DataFrame([(key, val[0], val[1]) for key, val in gpi_name_cnt.items()],
                              columns=['gpi', 'name', 'cnt'])
            df.to_csv('output/_gpi_drug_name_cnt.csv')
            print('dump output/_gpi_drug_name_cnt.pkl/csv done!')

            pickle.dump(gpiing_names_cnt, open('output/_gpi_ingredients_nameset_cnt.pkl', 'wb'))
            df2 = pd.DataFrame([(key, val[0], val[1]) for key, val in gpiing_names_cnt.items()],
                               columns=['gpi', 'name', 'cnt'])
            df2.to_csv('output/_gpi_ingredients_nameset_cnt.csv')
            print('dump output/_gpi_ingredients_nameset_cnt.pkl/csv done!')
        except Exception as e:
            print(e)

    return mci_drug_taken_by_patient


def _combine_drug_taken_by_patient(a, b):
    mci_drug_taken_by_patient = a
    for ing, taken_by_patient in b.items():
        for patid, take_times_list in taken_by_patient.items():
            for take_time in take_times_list:
                date, day = take_time
                if ing not in mci_drug_taken_by_patient:
                    mci_drug_taken_by_patient[ing][patid] = {(date, day)}
                else:
                    if patid not in mci_drug_taken_by_patient.get(ing):
                        mci_drug_taken_by_patient[ing][patid] = {(date, day)}
                    else:
                        mci_drug_taken_by_patient[ing][patid].add((date, day))
    return mci_drug_taken_by_patient


def generate_and_dump_drug_patient_records(input_file, output_file):
    # 1. generate mci_drug_taken_by_patient from file1, file2 [... if any]
    # 2. combine drugs_patients into one
    start_time = time.time()
    file_list = [x.strip() for x in input_file.split(',')]
    print('generate_and_dump_drug_patient_records, input drug file list: ', file_list)
    results = []
    for file in file_list:
        print('...Processing ', file)
        results.append(pre_drug_table(file))

    mci_drug_taken_by_patient = results[0]
    for i in range(1, len(results)):
        mci_drug_taken_by_patient = _combine_drug_taken_by_patient(mci_drug_taken_by_patient, results[i])

    print('Combine finished, # of Drug = len(mci_drug_taken_by_patient)=', len(mci_drug_taken_by_patient))
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    check_and_mkdir(output_file)
    with open(output_file, 'wb') as f:
        pickle.dump(mci_drug_taken_by_patient, f)  # , pickle.HIGHEST_PROTOCOL)
    print(r'dump {} done!'.format(output_file))
    return mci_drug_taken_by_patient


# def load_drug_patient_records():
#     with open(r'pickles/mci_drug_taken_by_patient_from_dispensing.pkl', 'rb') as f:
#         mci_drug_taken_by_patient_dis = pickle.load(f)
#     print(r'Load pickles/mci_drug_taken_by_patient_from_dispensing.pkl done：')
#     print('***len(mci_drug_taken_by_patient_dis): ', len(mci_drug_taken_by_patient_dis))
#
#     with open(r'pickles/mci_drug_taken_by_patient_from_prescribing.pkl', 'rb') as f:
#         mci_drug_taken_by_patient_pre = pickle.load(f)
#     print(r'Load pickles/mci_drug_taken_by_patient_from_prescribing.pkl done!')
#     print('***len(mci_drug_taken_by_patient_pre): ', len(mci_drug_taken_by_patient_pre))
#
#     with open(r'pickles/mci_drug_taken_by_patient_from_dispensing_plus_prescribing.pkl', 'rb') as f:
#         mci_drug_taken_by_patient_dis_plus_pre = pickle.load(f)
#     print(r'Load pickles/mci_drug_taken_by_patient_from_dispensing_plus_prescribing.pkl done!')
#     print('***len(mci_drug_taken_by_patient_dis_plus_pre): ', len(mci_drug_taken_by_patient_dis_plus_pre))
#
#     return mci_drug_taken_by_patient_dis, mci_drug_taken_by_patient_pre, mci_drug_taken_by_patient_dis_plus_pre


def load_latest_rxnorm_info():
    print('********load_latest_rxnorm_info*********')
    df = pd.read_csv(r'mapping/RXNORM.csv', dtype=str)  # (40157, 4)

    rxnorm_name = {}  # len: 26978
    for index, row in df.iterrows():

        rxnorm = row[r'Class ID'].strip().split('/')[-1]
        name = row[r'Preferred Label']
        rxnorm_name[rxnorm] = name
    print('df.shape:', df.shape, 'len(rxnorm_name):', len(rxnorm_name))
    return rxnorm_name, df


def count_drug_frequency(in_file, out_file, vocab='rxnorm'):

    if vocab == 'gpi':
        with open(r'output/_gpi_ingredients_nameset_cnt.pkl', 'rb') as f:
            gpiing_names_cnt = pickle.load(f)
            drug_name = {}
            for key, val in gpiing_names_cnt.items():
                drug_name[key] = val[0]
        print('Using GPI vocabulary, len(drug_name) :', len(drug_name))
    else:
        drug_name, _ = load_latest_rxnorm_info()
        print('Using rxnorm_cui vocabulary, len(drug_name) :', len(drug_name))

    with open(in_file, 'rb') as f:
        mci_drug_taken_by_patient = pickle.load(f)

    check_and_mkdir(out_file)
    # writer = pd.ExcelWriter(out_file, engine='xlsxwriter')
    drug_patient = []
    for ing, taken_by_patient in mci_drug_taken_by_patient.items():
        n_drug_time = 0
        for patid, take_times_list in taken_by_patient.items():
            n_drug_time += len(take_times_list)
        drug_patient.append((ing, drug_name.get(ing), len(taken_by_patient), n_drug_time))

    pd_drug = pd.DataFrame(drug_patient, columns=['Ingredient', 'label',
                                                  'n_patient_take', 'n_patient_take_times'])
    pd_drug.sort_values(by=['n_patient_take'], inplace=True, ascending=False)
    pd_drug.to_csv(out_file)
    # pd_drug.to_excel(writer)  #, sheet_name=sheet_name[i])
    # writer.save()
    print('count_drug_frequency:\n', in_file, '-->', out_file)
    return pd_drug


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    print(args)
    mci_drug_taken_by_patient = generate_and_dump_drug_patient_records(args.input_file,
                                                                       r'output/mci_drug_taken_by_patient.pkl')
    if 'marketscan' in args.input_file.lower():
        vocab = 'gpi'
    else:
        vocab = 'rxnorm'
    pd_drug = count_drug_frequency(r'output/mci_drug_taken_by_patient.pkl', 'output/drug_prevalence.csv', vocab=vocab)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

