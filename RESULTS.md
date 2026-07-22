
## Elsa Reproduction Results

| Model             | Sparsity | ADMM Steps | WikiText-2 |   C4     | Runtime |
|-------------------|---------:|-----------:|-----------:|---------:|--------:|
| facebook/opt-125m |    0%    |      —     |   27.6112  | 26.5213  |    —    |
| facebook/opt-125m |    30%   |    4096    |   29.3321  | 26.7499  |    —    |
| facebook/opt-125m |    50%   |    4096    |   37.5093  | 32.4300  | 3:13:02 |
| facebook/opt-125m |    50%   |    4096    |   31.7825  | 28.4109  |    -    |
| facebook/opt-125m |    60%   |    4096    |   39.6119  | 33.4817  |    —    |
| facebook/opt-125m |    70%   |    4096    |   54.0991  | 41.2454  | 3:08:00 |
| facebook/opt-125m |    80%   |    4096    |   62.7001  | 46.5478  |    —    |
| facebook/opt-125m |    90%   |    4096    |   95.1303  | 61.0246  | 3:04:28 |

## Dense Baseline

| Model             | Sparsity | WikiText-2 | C4      |
|-------------------|---------:|-----------:|--------:|
| facebook/opt-125m |    0%    | 27.6112    | 26.5213 |

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
