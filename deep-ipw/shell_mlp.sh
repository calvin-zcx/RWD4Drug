mkdir -p output/save_cohort_all_loose/log
python main.py --data_dir ../ipreprocess/output/save_cohort_all_loose/ --treated_drug 83367 --controlled_drug random --run_model MLP --output_dir output/save_cohort_all_loose/ --random_seed 0 --drug_coding rxnorm --med_code_topk 150 -stats  2>&1 | tee output/save_cohort_all_loose/log/mlp_83367.log
