# qlearning_parallel results (full)

- seed: `181`
- workers: `31`

| experiment | variant | target_mode | feature_mode | average_score | average_steps | time_per_step_ms | rate_1024 | rate_2048 | rate_4096 | td_error_rms_final | td_error_rms_mean | average_bias | model_path | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3-A | Q(s,a)+StateNTuple | q | state | 2843.6000 | 243.6000 | 0.1565 | 0.0000 | 0.0000 | 0.0000 | 87.6702 | 76.5294 | -1682.2942 | models\qlearning_runs\20260621_160440\qlearning_3a.pkl | 1450.8205 |
| 3-B | Q(s,a)+AfterstateNTuple | q | afterstate | 11808.0000 | 741.2000 | 0.1069 | 0.6000 | 0.0000 | 0.0000 | 203.6742 | 191.4804 | -5785.3778 | models\qlearning_runs\20260621_160440\qlearning_3b.pkl | 3326.2235 |
| 3-C | V(s')+StateNTuple | v | state | 2644.8000 | 233.0000 | 0.1448 | 0.0000 | 0.0000 | 0.0000 | 91.5130 | 80.9804 | -866.0633 | models\qlearning_runs\20260621_160440\qlearning_3c.pkl | 1169.8244 |
| 3-D | V(s')+AfterstateNTuple | v | afterstate | 21270.8000 | 1190.9000 | 0.1221 | 0.8000 | 0.4000 | 0.0000 | 281.0138 | 249.1619 | -8927.4842 | models\qlearning_runs\20260621_160440\qlearning_3d.pkl | 3735.5059 |
| 3-E | MV(s')+AfterstateNTuple | mv | afterstate | 22452.8000 | 1257.5000 | 0.1925 | 0.8000 | 0.3000 | 0.0000 | 379.3694 | 245.4767 | -9791.4242 | models\qlearning_runs\20260621_160440\qlearning_3e.pkl | 8117.5039 |
