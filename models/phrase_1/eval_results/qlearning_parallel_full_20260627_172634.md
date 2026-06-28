# qlearning_parallel results (full)

- seed: `181`
- workers: `31`

| experiment | variant | target_mode | feature_mode | average_score | average_steps | time_per_step_ms | rate_1024 | rate_2048 | rate_4096 | rate_8192 | rate_16384 | rate_32768 | td_error_rms_final | td_error_rms_mean | normalized_td_rms_final | average_bias | average_norm_bias | model_path | train_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3-A | Q(s,a)+StateNTuple | q | state | 4466.0000 | 329.5000 | 0.1738 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 119.4348 | 96.7808 | 0.0337 | -1329.6817 | 0.5907 | models\phrase_1\qlearning_runs\20260627_135045\qlearning_3a.pkl | 3436.9853 |
| 3-B | Q(s,a)+AfterstateNTuple | q | afterstate | 10046.4000 | 647.7000 | 0.1598 | 0.3000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 253.5355 | 236.3807 | 0.0167 | -5365.4558 | -0.3610 | models\phrase_1\qlearning_runs\20260627_135045\qlearning_3b.pkl | 5572.7873 |
| 3-C | V(s')+StateNTuple | v | state | 2644.8000 | 233.0000 | 0.1780 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 89.1864 | 69.4104 | 0.0205 | -776.3065 | 1.4234 | models\phrase_1\qlearning_runs\20260627_135045\qlearning_3c.pkl | 2140.1750 |
| 3-D | V(s')+AfterstateNTuple | v | afterstate | 20423.2000 | 1138.7000 | 0.1247 | 0.9000 | 0.4000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 527.4202 | 654.9319 | 0.0225 | -3723.6605 | 0.2033 | models\phrase_1\qlearning_runs\20260627_135045\qlearning_3d.pkl | 5783.4165 |
| 3-E | MV(s')+AfterstateNTuple | mv | afterstate | 22508.4000 | 1249.2000 | 0.1495 | 0.9000 | 0.4000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 524.5919 | 666.6640 | 0.0304 | -2613.7543 | 0.1824 | models\phrase_1\qlearning_runs\20260627_135045\qlearning_3e.pkl | 12913.2305 |
