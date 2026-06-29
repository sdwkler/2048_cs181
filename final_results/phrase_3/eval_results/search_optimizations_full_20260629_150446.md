# search_optimizations results (full)

- seed: `181`
- workers: `31`

| average_score | average_steps | time_per_step_ms | rate_1024 | rate_2048 | rate_4096 | rate_8192 | rate_16384 | rate_32768 | experiment | variant | depth | eval_type | use_beam_search | use_tt | b_eff | compression_ratio | node_count | tt_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20096.4800 | 1109.1200 | 46.0558 | 0.9400 | 0.3600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2-A1 | Heuristic + Base | 3 | heuristic | False | False | 20.8336 | 0.2073 | 10892.1469 | 0.0000 |
| 13344.0000 | 788.4800 | 22.6948 | 0.6800 | 0.1200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2-A2 | Heuristic + BeamSearch | 3 | heuristic | True | False | 12.6560 | 0.2576 | 2380.9369 | 0.0000 |
| 20096.4800 | 1109.1200 | 32.4420 | 0.9400 | 0.3600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2-A3 | Heuristic + HashDAG | 3 | heuristic | False | True | 15.9734 | 0.4033 | 4586.5713 | 0.0307 |
| 144702.7200 | 6049.1000 | 314.0224 | 1.0000 | 1.0000 | 1.0000 | 0.8400 | 0.0200 | 0.0000 | 2-B1 | NTuple + Base | 3 | ntuple_afterstate | False | False | 18.1587 | 0.2489 | 6937.6646 | 0.0000 |
| 130123.0400 | 5528.8400 | 171.9531 | 1.0000 | 1.0000 | 0.9800 | 0.7600 | 0.0000 | 0.0000 | 2-B2 | NTuple + BeamSearch | 3 | ntuple_afterstate | True | False | 10.9587 | 0.2599 | 1476.0652 | 0.0000 |
| 144702.7200 | 6049.1000 | 148.2334 | 1.0000 | 1.0000 | 1.0000 | 0.8400 | 0.0200 | 0.0000 | 2-B3 | NTuple + HashDAG | 3 | ntuple_afterstate | False | True | 14.4110 | 0.4626 | 3307.3285 | 0.0301 |
