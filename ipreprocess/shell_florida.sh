#!/usr/bin/env bash
DEMO_FILE="../data/florida/demographic.csv"
DIAGNOSIS_FILE="../data/florida/diagnosis.csv"
DRUG_FILE="../data/florida/dispensing.csv,../data/florida/prescribing.csv"

mkdir output
python std1_pre_demo.py --demo_file ${DEMO_FILE}  --dx_file ${DIAGNOSIS_FILE} 2>&1 | tee output/std1_pre_demo.log
python std2_pre_drug.py --input_file ${DRUG_FILE} 2>&1 | tee output/std2_pre_drug.log
python std3_cohort_build.py --min_patients 20 --min_prescription 2 --exposure_interval 30 --followup 730 --baseline 365 --index_minus_init_min 0 --index_minus_init_max 999999999999999 --adrd_minus_index_min 0 --dx_file ${DIAGNOSIS_FILE} --dx_coding ccw --drug_coding rxnorm --save_cohort_all output/save_cohort_all_loose/ 2>&1 | tee output/save_cohort_all_loose.log
python std3_cohort_build.py --min_patients 20 --min_prescription 4 --exposure_interval 180 --followup 730 --baseline 365 --index_minus_init_min 0 --index_minus_init_max 999999999999999 --adrd_minus_index_min 0 --dx_file ${DIAGNOSIS_FILE} --dx_coding ccw --drug_coding rxnorm --save_cohort_all output/save_cohort_all_mediate/ 2>&1 | tee output/save_cohort_all_mediate.log
python std3_cohort_build.py --min_patients 20 --min_prescription 4 --exposure_interval 180 --followup 730 --baseline 365 --index_minus_init_min 0 --index_minus_init_max 365 --adrd_minus_index_min 0 --dx_file ${DIAGNOSIS_FILE} --dx_coding ccw --drug_coding rxnorm --save_cohort_all output/save_cohort_all_stringent/ 2>&1 | tee output/save_cohort_all_stringent.log
python univariate_statistics.py --dx_file ${DIAGNOSIS_FILE} --drug_coding rxnorm 2>&1 | tee output/univariate_statistics.log