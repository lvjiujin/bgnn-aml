python scripts/run.py datasets/elliptic  bgnn --max_seeds 1 \
--repeat_exp 1 --task classification  \
--metric f1  --save_folder ./results/elliptic_bgnn \
--version_num 2.0 > log_test_202109.txt 2>&1
