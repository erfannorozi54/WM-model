# GPU Server Instructions

## Connect & Setup

```bash
ssh hamrah-gpu-internal          # VPN must be up; the alias resolves to an internal IP
cd ~/Projects/WM-model
git pull
conda deactivate                 # only if conda base is active
source ~/.venv/WM-model/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

Before launching anything: `nvidia-smi` to confirm both RTX 3090s are free, and
`uptime` to check the load average — a busy box makes jobs run far longer than
expected (see the BLAS-threading gotcha in `AGENTS.md`).

## Run

Prefer the batch scripts. They resolve the venv themselves, set `PYTHONPATH`, and
write `logs/<stage>_<UTC ts>/00_provenance.log` recording the commit each run
used. Hand-assembled CLIs are how the neural-efficiency results got confounded
twice.

```bash
nohup ./run_training.sh h256            > train.log    2>&1 & disown
nohup ./run_proxy_pipeline.sh baseline  > proxy.log    2>&1 & disown
nohup ./run_analysis.sh h256            > analysis.log 2>&1 & disown
nohup ./run_neural_efficiency.sh        > ne.log       2>&1 & disown
```

`disown` matters — without it the job dies when the SSH session ends.

Single run, no batch:

```bash
nohup python -m src.train_with_generalization --config configs/stsf.yaml > train.log 2>&1 & disown
```

## Configs

| Config | N-values | task_features |
|--------|----------|---------------|
| `stsf.yaml` | `[2]` | `["location"]` — fastest |
| `stmf.yaml` | `[2]` | all three |
| `mtmf.yaml` | `[1, 2, 3]` | all three — the full paper config |
| `attention_*.yaml` / `dual_attention_*.yaml` | same | plus the attention gate |

`configs_128/` mirrors `configs/` with `hidden_size: 128`.
`save_hidden: true` must be set or the analysis pipeline has nothing to read.

## Monitor

```bash
tail -f train.log
tail -f logs/<stage>_<ts>/00_provenance.log    # per-step start/end and rc
nvidia-smi
ps aux | grep python
ls experiments/
```

## Notes

- Local PC venv is `~/.virtualenvs/WM-model`; the server's is `~/.venv/WM-model`.
  `run_common.sh` probes for whichever exists, so the scripts work on both.
- The server runs **UTC**. Comparing its file mtimes against local git timestamps
  without accounting for that has already caused one wrong conclusion.
- SSH to GitHub is blocked from the server; `origin` must be the HTTPS URL.
