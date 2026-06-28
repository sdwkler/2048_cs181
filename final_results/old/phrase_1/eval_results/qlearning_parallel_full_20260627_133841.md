# qlearning_parallel results (full)

- seed: `181`
- workers: `31`

| experiment | variant | target_mode | feature_mode | average_score | average_steps | time_per_step_ms | rate_1024 | rate_2048 | rate_4096 | rate_8192 | rate_16384 | rate_32768 | td_error_rms_final | td_error_rms_mean | normalized_td_rms_final | average_bias | average_norm_bias | model_path | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3-A | Q(s,a)+StateNTuple | q | state | 2027.2000 | 190.4000 | 0.1986 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 92.3349 | 72.4909 | 0.0213 | -1302.9523 | -0.3908 | models\phrase_1\qlearning_runs\20260627_122637\qlearning_3a.pkl | 1068.2123 |
| 3-B | Q(s,a)+AfterstateNTuple | q | afterstate | 7370.8000 | 521.7000 | 0.1417 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 164.4472 | 147.9927 | 0.0213 | -3410.5511 | -0.4169 | models\phrase_1\qlearning_runs\20260627_122637\qlearning_3b.pkl | 1536.9986 |
| 3-C | V(s')+StateNTuple | v | state | 2644.8000 | 233.0000 | 0.1619 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 79.5074 | 61.9108 | 0.0360 | -1164.1229 | 0.7206 | models\phrase_1\qlearning_runs\20260627_122637\qlearning_3c.pkl | 775.5025 |
| 3-D | V(s')+AfterstateNTuple | v | afterstate | 14310.0000 | 864.8000 | 0.0791 | 0.6000 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 288.9508 | 330.0988 | 0.0171 | -6647.0478 | -0.1312 | models\phrase_1\qlearning_runs\20260627_122637\qlearning_3d.pkl | 1896.6557 |
| 3-E | MV(s')+AfterstateNTuple | mv | afterstate | 16017.6000 | 939.3000 | 0.1961 | 0.7000 | 0.2000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 311.2435 | 331.1885 | 0.0178 | -6434.3024 | -0.3015 | models\phrase_1\qlearning_runs\20260627_122637\qlearning_3e.pkl | 4286.3446 |
