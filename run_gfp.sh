# python scripts/GWG_2.py configs/gfp_ddg/GWG_2_GFP_stability_linear.yml --logdir logs_mgda_linear_1024 --tag weight_0.0 --linear_weight_1 0.0
# python scripts/GWG_2.py configs/gfp_ddg/GWG_2_GFP_stability_linear.yml --logdir logs_mgda_linear_1024 --tag weight_0.001 --linear_weight_1 0.001
# python scripts/GWG_2.py configs/gfp_ddg/GWG_2_GFP_stability_linear.yml --logdir logs_mgda_linear_1024 --tag weight_0.01 --linear_weight_1 0.01
# python scripts/GWG_2.py configs/gfp_ddg/GWG_2_GFP_stability_linear.yml --logdir logs_mgda_linear_1024 --tag weight_0.1 --linear_weight_1 0.1
# python scripts/GWG_2.py configs/gfp_ddg/GWG_2_GFP_stability_linear.yml --logdir logs_mgda_linear_1024 --tag weight_0.3 --linear_weight_1 0.3
# python scripts/GWG_2.py configs/gfp_ddg/GWG_2_GFP_stability_linear.yml --logdir logs_mgda_linear_1024 --tag weight_0.5 --linear_weight_1 0.5
# python scripts/GWG_2.py configs/gfp_ddg/GWG_2_GFP_stability_linear.yml --logdir logs_mgda_linear_1024 --tag weight_0.7 --linear_weight_1 0.7
# python scripts/GWG_2.py configs/gfp_ddg/GWG_2_GFP_stability_linear.yml --logdir logs_mgda_linear_1024 --tag weight_0.9 --linear_weight_1 0.9
# python scripts/GWG_2.py configs/gfp_ddg/GWG_2_GFP_stability_linear.yml --logdir logs_mgda_linear_1024 --tag weight_0.99 --linear_weight_1 0.99
# python scripts/GWG_2.py configs/gfp_ddg/GWG_2_GFP_stability_linear.yml --logdir logs_mgda_linear_1024 --tag weight_0.999 --linear_weight_1 0.999
# python scripts/GWG_2.py configs/gfp_ddg/GWG_2_GFP_stability_linear.yml --logdir logs_mgda_linear_1024 --tag weight_1.0 --linear_weight_1 1.0

# python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_mgda_linear_1024/GWG_2_GFP_stability_linear_2023_10_24__15_34_43_weight_0.001/samples_20231024-153443/seed_1.csv --num_threads 50
# python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_mgda_linear_1024/GWG_2_GFP_stability_linear_2023_10_24__15_37_01_weight_0.01/samples_20231024-153701/seed_1.csv --num_threads 50
# python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_mgda_linear_1024/GWG_2_GFP_stability_linear_2023_10_24__15_39_07_weight_0.1/samples_20231024-153907/seed_1.csv --num_threads 50
# python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_mgda_linear_1024/GWG_2_GFP_stability_linear_2023_10_24__15_40_56_weight_0.3/samples_20231024-154056/seed_1.csv --num_threads 50
# python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_mgda_linear_1024/GWG_2_GFP_stability_linear_2023_10_24__15_42_42_weight_0.5/samples_20231024-154242/seed_1.csv --num_threads 50
# python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_mgda_linear_1024/GWG_2_GFP_stability_linear_2023_10_24__15_44_31_weight_0.7/samples_20231024-154431/seed_1.csv --num_threads 50
# python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_mgda_linear_1024/GWG_2_GFP_stability_linear_2023_10_24__15_46_16_weight_0.9/samples_20231024-154616/seed_1.csv --num_threads 50
# python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_mgda_linear_1024/GWG_2_GFP_stability_linear_2023_10_24__15_48_00_weight_0.99/samples_20231024-154800/seed_1.csv --num_threads 50
# python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_mgda_linear_1024/GWG_2_GFP_stability_linear_2023_10_24__15_49_42_weight_0.999/samples_20231024-154942/seed_1.csv --num_threads 50
# python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_mgda_linear_1024/GWG_2_GFP_stability_linear_2023_10_24__15_51_21_weight_1.0/samples_20231024-155121/seed_1.csv --num_threads 50
# python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_new/train_smooth_GFP_2023_09_30__13_34_58/samples_20230930-135740/seed_1.csv --num_threads 50 --tag random_selection
# python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_new/train_predictor_stability_0.2_0.4_2023_09_30__10_43_49/samples_20230930-104911/seed_1.csv --num_threads 50 --tag random_selection
python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_baseline_new/adalead_GFP_2023_10_02__16_52_36/samples_seed_42.csv --num_threads 50 --tag random_selection
python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_baseline_new/adalead_stability_2023_10_02__13_41_35/samples_seed_42.csv --num_threads 50 --tag random_selection
python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_baseline_new/mala_approx_GFP_2023_10_01__17_51_17/samples_seed_42.csv --num_threads 50 --tag random_selection
python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_baseline_new/mala_approx_stability_2023_10_01__17_51_27/samples_seed_42.csv
python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_baseline_new/random_sample_GFP_2023_10_01__12_07_24_0.2_0.4/samples_seed_42.csv --num_threads 50 --tag random_selection
python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_baseline_new/random_sample_stability_2023_10_01__12_07_34_0.2_0.4/samples_seed_42.csv --num_threads 50 --tag random_selection
python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_baseline_new/simulated_annealing_GFP_2023_10_01__13_45_04/samples_seed_42.csv --num_threads 50 --tag random_selection
python scripts/evaluate.py configs/gfp_ddg/evaluate.yml --sample_path logs_baseline_new/simulated_annealing_stability_2023_10_01__13_45_07/samples_seed_42.csv --num_threads 50 --tag random_selection
