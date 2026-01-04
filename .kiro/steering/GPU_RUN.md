# GPU Server Instructions

## Connect & Setup

```bash
ssh hamrah-gpu
cd ~/Projects/WM-model
git pull
conda deactivate
source ~/.venv/WM-model/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

## Run Training

```bash
# With generalization validation (recommended)
nohup python -m src.train_with_generalization --config configs/stsf.yaml > train.log 2>&1 &

# Check progress
tail -f train.log

# Or check if running
ps aux | grep python
```

## Configs

| Config | Description |
|--------|-------------|
| `stsf.yaml` | 2-back, category only (fastest) |
| `stmf.yaml` | 2-back, all tasks |
| `mtmf.yaml` | 1,2,3-back, all tasks (full) |
| `attention_*.yaml` | With attention |

## Monitor & Results

```bash
# Watch training
tail -f train.log

# Check GPU usage
nvidia-smi

# Results location
ls experiments/
```
