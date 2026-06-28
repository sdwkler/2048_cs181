# qlearning_parallel results (full)

- seed: `181`
- workers: `31`

| experiment | variant | target_mode | feature_mode | average_score | average_steps | time_per_step_ms | rate_1024 | rate_2048 | rate_4096 | td_error_rms_final | td_error_rms_mean | average_bias | model_path | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3-A | Q(s,a)+StateNTuple | q | state | 2569.6000 | 231.1000 | 0.6645 | 0.0000 | 0.0000 | 0.0000 | 37.9624 | 29.2489 | -1240.4640 | models\qlearning_runs\20260620_183256\qlearning_3a.pkl | 3702.9249 |
| 3-B | Q(s,a)+AfterstateNTuple | q | afterstate | 2889.2000 | 249.4000 | 0.5875 | 0.0000 | 0.0000 | 0.0000 | 65.0114 | 43.1566 | -1418.9334 | models\qlearning_runs\20260620_183256\qlearning_3b.pkl | 3339.9074 |
| 3-C | V(s')+StateNTuple | v | state | 2644.8000 | 233.0000 | 0.5370 | 0.0000 | 0.0000 | 0.0000 | 49.3748 | 38.2801 | -1803.7726 | models\qlearning_runs\20260620_183256\qlearning_3c.pkl | 3843.7946 |
| 3-D | V(s')+AfterstateNTuple | v | afterstate | 6446.8000 | 446.7000 | 0.5214 | 0.1000 | 0.0000 | 0.0000 | 115.0917 | 84.0682 | -4242.1661 | models\qlearning_runs\20260620_183256\qlearning_3d.pkl | 5918.5296 |
| 3-E | MV(s')+AfterstateNTuple | mv | afterstate | 5183.6000 | 384.9000 | 0.8980 | 0.0000 | 0.0000 | 0.0000 | 95.8945 | 83.6200 | -4126.4220 | models\qlearning_runs\20260620_183256\qlearning_3e.pkl | 13319.1642 |
