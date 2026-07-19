
## Elsa Reproduction Results

| Model             | Sparsity | ADMM Steps | WikiText-2 |   C4     | Runtime |
|-------------------|---------:|-----------:|-----------:|---------:|--------:|
| facebook/opt-125m |    0%    |      —     |   27.6112  | 26.5213  |    —    |
| facebook/opt-125m |    50%   |    4096    |   37.5093  | 32.4300  | 3:13:02 |
| facebook/opt-125m |    70%   |    4096    |   54.0991  | 41.2454  | 3:08:00 |
| facebook/opt-125m |    90%   |    4096    |   359.0999 | 132.3388 | 3:13:09 |

## Dense Baseline

| Model             | Sparsity | WikiText-2 | C4      |
|-------------------|---------:|-----------:|--------:|
| facebook/opt-125m |    0%    | 27.6112    | 26.5213 |


