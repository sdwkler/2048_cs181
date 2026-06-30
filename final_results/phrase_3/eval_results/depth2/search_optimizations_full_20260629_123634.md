# search_optimizations results (full)

- seed: `181`
- workers: `31`

| average_score | average_steps | time_per_step_ms | rate_1024 | rate_2048 | rate_4096 | rate_8192 | rate_16384 | rate_32768 | experiment | variant | depth | eval_type | use_beam_search | use_tt | b_eff | compression_ratio | node_count | tt_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12505.1200 | 748.2200 | 4.6959 | 0.6600 | 0.0600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2-A1 | Heuristic + Base | 2 | heuristic | False | False | 14.3415 | 0.5024 | 217.1326 | 0.0000 |
| 8996.4800 | 579.5800 | 4.2723 | 0.3400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2-A2 | Heuristic + BeamSearch | 2 | heuristic | True | False | 8.7873 | 0.5721 | 80.4613 | 0.0000 |
| 12505.1200 | 748.2200 | 4.6299 | 0.6600 | 0.0600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2-A3 | Heuristic + HashDAG | 2 | heuristic | False | True | 14.3308 | 0.5035 | 216.8243 | 0.0005 |
| 113399.1200 | 4890.4800 | 8.9346 | 1.0000 | 0.9800 | 0.9200 | 0.5800 | 0.0000 | 0.0000 | 2-B1 | NTuple + Base | 2 | ntuple_afterstate | False | False | 12.9051 | 0.5526 | 172.9780 | 0.0000 |
| 120719.0400 | 5164.0600 | 8.3669 | 1.0000 | 1.0000 | 0.9400 | 0.6600 | 0.0000 | 0.0000 | 2-B2 | NTuple + BeamSearch | 2 | ntuple_afterstate | True | False | 7.9237 | 0.5957 | 64.3724 | 0.0000 |
| 113399.1200 | 4890.4800 | 9.1436 | 1.0000 | 0.9800 | 0.9200 | 0.5800 | 0.0000 | 0.0000 | 2-B3 | NTuple + HashDAG | 2 | ntuple_afterstate | False | True | 12.8964 | 0.5537 | 172.7594 | 0.0004 |
