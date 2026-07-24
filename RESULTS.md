
## Elsa Reproduction Results

| Model             | Sparsity | ADMM Steps | WikiText-2 |   C4     | Runtime |
|-------------------|---------:|-----------:|-----------:|---------:|--------:|
| facebook/opt-125m |    0%    |      —     |   27.6112  | 26.5213  |    —    |
| facebook/opt-125m |    30%   |    4096    |   29.3321  | 26.7499  |    —    |
| facebook/opt-125m |    40%   |    4096    |   37.5093  | 32.4300  | 3:13:02 |
| facebook/opt-125m |    50%   |    4096    |   31.7825  | 28.4109  |    -    |
| facebook/opt-125m |    60%   |    4096    |   39.6119  | 33.4817  |    —    |
| facebook/opt-125m |    70%   |    4096    |   54.0991  | 41.2454  | 3:08:00 |
| facebook/opt-125m |    80%   |    4096    |   62.7001  | 46.5478  |    —    |
| facebook/opt-125m |    90%   |    4096    |   95.1303  | 61.0246  | 3:04:28 |

## Dense Baseline

| Model             | Sparsity | WikiText-2 | C4      |
|-------------------|---------:|-----------:|--------:|
| facebook/opt-125m |    0%    | 27.6112    | 26.5213 |

## Zero-shot Benchmark Results

## OPT-125M

| Benchmark | 50% | 70% |
|-----------|-----:|-----:|
| ARC Challenge | 0.2287 | 0.2287 |
| ARC Easy | 0.3750 | 0.3582 |
| BoolQ | 0.6153 | 0.6217 |
| HellaSwag | 0.3027 | 0.2845 |
| OpenBookQA | 0.2620 | 0.2520 |
| PIQA | 0.6039 | 0.5963 |
| RACE | 0.2756 | 0.2632 |
| RTE | 0.5451 | 0.4982 |
| WinoGrande | 0.5201 | 0.5130 |

### OPT-125M

| Benchmark | 50% | 70% | 90% |
|-----------|-----:|-----:|-----:|
| ARC Challenge | 0.2304 | 0.2287 | 0.2159 |
| ARC Easy | 0.3750 | 0.3582 | 0.3300 |
| BoolQ | 0.6153 | 0.6217 | 0.4235 |
| HellaSwag | 0.3027 | 0.2845 | 0.2678 |
| OpenBookQA | 0.2620 | 0.2520 | 0.2380 |
| PIQA | 0.6039 | 0.5963 | 0.5609 |
| RACE | 0.2756 | 0.2632 | 0.2450 |
| RTE | 0.5451 | 0.4982 | 0.5199 |
| WinoGrande | 0.5201 | 0.5130 | 0.4862 |
>>>>>>> 1abdeec (Add 70% and 90% zero-shot benchmark results)

### Notes

- Successfully reproduced the OPT-125M 90% sparsity results reported in the ELSA paper.
- Configuration:
  - Learning rate: `2e-4`
  - λ: `1e-3`
  - λ schedule: `cosine`
  - Projection mode: `momentum`
  - Adam β₁/β₂: `0.9 / 0.999`
  - Interval: `32`
  - Training steps: `4096`
  - Effective batch size: `8` (`batch_size=2`, `gradient_accumulation_steps=4`)
  - Precision: `bf16`

## Previous Attempts

| Configuration | WikiText-2 | C4 |
|--------------|-----------:|---:|
| Constant λ schedule + identity projection | 130.1703 | 79.5750 |
| λ = 0.01 (initial run) | 359.0999 | 132.3388 |


