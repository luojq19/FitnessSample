# python scripts/evaluate_GFP.py configs/gfp_ddg/evaluate.yml --sample_path logs_pref_vec_debug/GWG_2_GFP_stability_pref_vec_2023_10_24__10_14_41_pref_0/samples_20231024-101441/seed_1.csv
# python scripts/evaluate_GFP.py configs/gfp_ddg/evaluate.yml --sample_path logs_pref_vec_debug/GWG_2_GFP_stability_pref_vec_2023_10_24__10_15_10_pref_1/samples_20231024-101510/seed_1.csv
# python scripts/evaluate_GFP.py configs/gfp_ddg/evaluate.yml --sample_path logs_pref_vec_debug/GWG_2_GFP_stability_pref_vec_2023_10_24__10_15_35_pref_2/samples_20231024-101535/seed_1.csv
# python scripts/evaluate_GFP.py configs/gfp_ddg/evaluate.yml --sample_path logs_pref_vec_debug/GWG_2_GFP_stability_pref_vec_2023_10_24__10_18_38_pref_3/samples_20231024-101838/seed_1.csv
# python scripts/evaluate_GFP.py configs/gfp_ddg/evaluate.yml --sample_path logs_pref_vec_debug/GWG_2_GFP_stability_pref_vec_2023_10_24__10_19_26_pref_4/samples_20231024-101926/seed_1.csv
python scripts/evaluate_GFP.py configs/gfp_ddg/evaluate.yml --sample_path logs_pref_vec_debug/GWG_2_GFP_stability_pref_vec_2023_10_24__13_44_54_debug_clip_pref_4/samples_20231024-134454/seed_1.csv

# [2023-10-24 10:54:48,733::evaluate_GFP::INFO] sample path: logs_pref_vec_debug/GWG_2_GFP_stability_pref_vec_2023_10_24__10_14_41_pref_0/samples_20231024-101441/seed_1.csv
# [2023-10-24 10:54:50,032::evaluate_GFP::INFO] Loaded oracle from logs/oracle_GFP_ggs
# running oracle: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32354/32354 [00:14<00:00, 2285.96it/s]
# [2023-10-24 10:55:04,348::evaluate_GFP::INFO] task      mean    median  std     max     min
# [2023-10-24 10:55:04,357::evaluate_GFP::INFO] GFP       1.505   1.484   0.385   3.862   0.136
# [2023-10-24 10:55:04,366::evaluate_GFP::INFO] GFP_normalized    0.078   0.071   0.136   0.908   -0.404
# [2023-10-24 10:55:08,068::evaluate_GFP::INFO] sample path: logs_pref_vec_debug/GWG_2_GFP_stability_pref_vec_2023_10_24__10_15_10_pref_1/samples_20231024-101510/seed_1.csv
# [2023-10-24 10:55:09,442::evaluate_GFP::INFO] Loaded oracle from logs/oracle_GFP_ggs
# running oracle: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23407/23407 [00:11<00:00, 2031.12it/s]
# [2023-10-24 10:55:21,075::evaluate_GFP::INFO] task      mean    median  std     max     min
# [2023-10-24 10:55:21,080::evaluate_GFP::INFO] GFP       1.467   1.444   0.375   3.303   0.068
# [2023-10-24 10:55:21,084::evaluate_GFP::INFO] GFP_normalized    0.065   0.056   0.132   0.711   -0.428
# [2023-10-24 10:55:24,509::evaluate_GFP::INFO] sample path: logs_pref_vec_debug/GWG_2_GFP_stability_pref_vec_2023_10_24__10_15_35_pref_2/samples_20231024-101535/seed_1.csv
# [2023-10-24 10:55:25,826::evaluate_GFP::INFO] Loaded oracle from logs/oracle_GFP_ggs
# running oracle: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15096/15096 [00:08<00:00, 1750.43it/s]
# [2023-10-24 10:55:34,609::evaluate_GFP::INFO] task      mean    median  std     max     min
# [2023-10-24 10:55:34,614::evaluate_GFP::INFO] GFP       1.341   1.334   0.337   3.149   0.188
# [2023-10-24 10:55:34,619::evaluate_GFP::INFO] GFP_normalized    0.020   0.018   0.119   0.657   -0.386
# [2023-10-24 10:55:38,578::evaluate_GFP::INFO] sample path: logs_pref_vec_debug/GWG_2_GFP_stability_pref_vec_2023_10_24__10_18_38_pref_3/samples_20231024-101838/seed_1.csv
# [2023-10-24 10:55:39,868::evaluate_GFP::INFO] Loaded oracle from logs/oracle_GFP_ggs
# running oracle: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9953/9953 [00:06<00:00, 1588.36it/s]
# [2023-10-24 10:55:46,295::evaluate_GFP::INFO] task      mean    median  std     max     min
# [2023-10-24 10:55:46,298::evaluate_GFP::INFO] GFP       1.293   1.289   0.338   2.604   0.038
# [2023-10-24 10:55:46,301::evaluate_GFP::INFO] GFP_normalized    0.003   0.002   0.119   0.465   -0.439
# [2023-10-24 10:55:49,829::evaluate_GFP::INFO] sample path: logs_pref_vec_debug/GWG_2_GFP_stability_pref_vec_2023_10_24__10_19_26_pref_4/samples_20231024-101926/seed_1.csv
# [2023-10-24 10:55:51,252::evaluate_GFP::INFO] Loaded oracle from logs/oracle_GFP_ggs
# running oracle: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28935/28935 [00:14<00:00, 1931.75it/s]
# [2023-10-24 10:56:06,392::evaluate_GFP::INFO] task      mean    median  std     max     min
# [2023-10-24 10:56:06,400::evaluate_GFP::INFO] GFP       1.861   1.758   0.650   4.147   0.264
# [2023-10-24 10:56:06,407::evaluate_GFP::INFO] GFP_normalized    0.204   0.167   0.229   1.008   -0.359