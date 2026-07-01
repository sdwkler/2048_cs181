# qlearning_parallel results (full)

- seed: `181`
- workers: `127`

| experiment | variant | target_mode | feature_mode | average_score | average_steps | time_per_step_ms | rate_1024 | rate_2048 | rate_4096 | rate_8192 | rate_16384 | rate_32768 | td_error_rms_final | td_error_rms_mean | normalized_td_rms_final | average_bias | average_norm_bias | norm_bias_rom | td_error_m_rms_final | td_error_m_rms_mean | model_path | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3-C | V(s')+State (Sampled) | v | state | 2979.5600 | 253.7400 | 0.0840 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 88.2199 | 77.2793 | 0.0288 | -487.6166 | 1.7042 | -0.2414 | 0.0000 | 0.0000 | models/phrase_1/qlearning_runs/20260701_040158/qlearning_3c.pkl | 4031.8691 |
| 3-D | V(s')+Afterstate (Baseline) | v | afterstate | 18731.2800 | 1035.7900 | 0.0530 | 0.7600 | 0.3400 | 0.0200 | 0.0000 | 0.0000 | 0.0000 | 1651.2543 | 2276.2550 | 0.0958 | 34002.0496 | 2.7654 | 1.9762 | 0.0000 | 0.0000 | models/phrase_1/qlearning_runs/20260701_040158/qlearning_3d.pkl | 13115.4453 |
| 3-E | TDA-Full+State (Expected) | tda_full | state | 2979.5600 | 253.7400 | 0.0640 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 157.8709 | 140.1086 | 0.0513 | 1013.9854 | 4.1927 | 0.5021 | 0.0000 | 0.0000 | models/phrase_1/qlearning_runs/20260701_040158/qlearning_3e.pkl | 12433.5811 |
| 3-F | TDA-Full+Afterstate (Paper) | tda_full | afterstate | 17766.8000 | 1004.9300 | 0.0548 | 0.7800 | 0.2000 | 0.0200 | 0.0000 | 0.0000 | 0.0000 | 1314.4810 | 1853.7870 | 0.0756 | 37845.6642 | 4.7053 | 2.3551 | 0.0000 | 0.0000 | models/phrase_1/qlearning_runs/20260701_040158/qlearning_3f.pkl | 38402.9549 |
| 3-G | Downside-MV+Afterstate | mv | afterstate | 17936.5600 | 1003.1700 | 0.1119 | 0.7100 | 0.2600 | 0.0200 | 0.0000 | 0.0000 | 0.0000 | 1693.4890 | 2328.3933 | 0.0983 | 35278.8203 | 3.5943 | 1.9498 | 1471.0007 | 1759.8861 | models/phrase_1/qlearning_runs/20260701_040158/qlearning_3g.pkl | 24195.9202 |
