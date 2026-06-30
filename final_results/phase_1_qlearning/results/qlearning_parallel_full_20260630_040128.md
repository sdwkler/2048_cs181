# qlearning_parallel results (full)

- seed: `181`
- workers: `127`

| experiment | variant | target_mode | feature_mode | average_score | average_steps | time_per_step_ms | rate_1024 | rate_2048 | rate_4096 | rate_8192 | rate_16384 | rate_32768 | td_error_rms_final | td_error_rms_mean | normalized_td_rms_final | average_bias | average_norm_bias | norm_bias_rom | td_error_m_rms_final | td_error_m_rms_mean | model_path | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3-D | V(s')+AfterstateNTuple | v | afterstate | 18731.2800 | 1035.7900 | 0.0774 | 0.7600 | 0.3400 | 0.0200 | 0.0000 | 0.0000 | 0.0000 | 1651.2543 | 2276.2550 | 0.0958 | 34002.0496 | 2.7654 | 1.9762 | 0.0000 | 0.0000 | models/phrase_1/qlearning_runs/20260629_204850/qlearning_3d.pkl | 12693.1886 |
| 3-E | MV(s')+AfterstateNTuple | mv | afterstate | 18767.6800 | 1043.6700 | 0.1506 | 0.7500 | 0.3400 | 0.0100 | 0.0000 | 0.0000 | 0.0000 | 1663.0842 | 2290.8641 | 0.0973 | 34314.1568 | 3.9147 | 1.9342 | 615.1779 | 580.5251 | models/phrase_1/qlearning_runs/20260629_204850/qlearning_3e.pkl | 25923.1489 |
