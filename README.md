# agent-memory

Minimal experiment layout for dataset loading and baseline evaluation.

```text
dataset_loader/   CSV loading and dataset representations
baseline/         Baseline algorithms
experiments/      Experiment launcher scripts
utils/            Metrics and shared helpers
main.py           CLI entry point
```

Run the baseline with:

```bash
./experiments/run_baseline.sh path/to/data.csv target_column
```
