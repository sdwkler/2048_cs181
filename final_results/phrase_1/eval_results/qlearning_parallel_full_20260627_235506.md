# qlearning_parallel results (full)

- seed: `181`
- workers: `47`

| experiment | variant | target_mode | feature_mode | average_score | average_steps | time_per_step_ms | rate_1024 | rate_2048 | rate_4096 | rate_8192 | rate_16384 | rate_32768 | td_error_rms_final | td_error_rms_mean | normalized_td_rms_final | average_bias | average_norm_bias | norm_bias_rom | td_error_m_rms_final | td_error_m_rms_mean | model_path | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3-A | Q(s,a)+StateNTuple | q | state | 3672.2000 | 291.0600 | 0.0755 | 0.0100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 105.6412 | 96.7808 | 0.0315 | -1329.6817 | 0.5907 | -0.5243 | 0.0000 | 0.0000 | models/phrase_1/qlearning_runs/20260627_225938/qlearning_3a.pkl | 919.1951 |
| 3-B | Q(s,a)+AfterstateNTuple | q | afterstate | 10565.7600 | 673.6500 | 0.0579 | 0.4100 | 0.0100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 242.4887 | 236.3807 | 0.0260 | -5365.4558 | -0.3610 | -0.5337 | 0.0000 | 0.0000 | models/phrase_1/qlearning_runs/20260627_225938/qlearning_3b.pkl | 1539.2822 |
| 3-C | V(s')+StateNTuple | v | state | 2979.5600 | 253.7400 | 0.0561 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 84.5816 | 69.4104 | 0.0282 | -776.3065 | 1.4234 | -0.3844 | 0.0000 | 0.0000 | models/phrase_1/qlearning_runs/20260627_225938/qlearning_3c.pkl | 627.8282 |
| 3-D | V(s')+AfterstateNTuple | v | afterstate | 19508.4400 | 1094.7500 | 0.0443 | 0.8400 | 0.3400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 526.8581 | 654.9319 | 0.0327 | -3723.6605 | 0.2033 | -0.1935 | 0.0000 | 0.0000 | models/phrase_1/qlearning_runs/20260627_225938/qlearning_3d.pkl | 1704.8659 |
| 3-E | MV(s')+AfterstateNTuple | mv | afterstate | 20607.0800 | 1154.3700 | 0.0742 | 0.8200 | 0.3600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 538.0156 | 666.6640 | 0.0337 | -2613.7543 | 0.1824 | -0.1417 | 173.5614 | 120.9606 | models/phrase_1/qlearning_runs/20260627_225938/qlearning_3e.pkl | 3314.5081 |
