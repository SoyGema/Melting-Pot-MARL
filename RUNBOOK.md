# Runbook: Training and Evaluating Commons Harvest Experiments

This document is the step-by-step operational guide for running all experiments in this repo.
For background on what each substrate means scientifically, see [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Project Structure Quick Reference](#2-project-structure-quick-reference)
3. [Training](#3-training)
   - [Basic training command](#31-basic-training-command)
   - [Training all baseline substrates](#32-training-all-baseline-substrates)
   - [Resuming an interrupted run](#33-resuming-an-interrupted-run)
   - [Continuing a completed run](#34-continuing-a-completed-run)
4. [Evaluation](#4-evaluation)
   - [Self-play evaluation (substrate)](#41-self-play-evaluation-substrate)
   - [Self-play evaluation (scenario)](#42-self-play-evaluation-scenario)
   - [Mixed evaluation (focal vs background)](#43-mixed-evaluation-focal-vs-background)
   - [Reading the output CSVs](#44-reading-the-output-csvs)
5. [Priority Experiments](#5-priority-experiments)
   - [Exp 1: Farmer focal vs Open background](#exp-1-farmer-focal-vs-open-background)
   - [Exp 2: No-Zap focal vs Open background](#exp-2-no-zap-focal-vs-open-background)
   - [Exp 3: Dilemma severity axis](#exp-3-dilemma-severity-axis)
6. [Cloud Training (GCP)](#6-cloud-training-gcp)
7. [Finding Your Results](#7-finding-your-results)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Environment Setup

```bash
git clone <this-repo>
cd Melting-Pot-MARL
conda create -n mpc_main python=3.10 -y
conda activate mpc_main
SYSTEM_VERSION_COMPAT=0 pip install dmlab2d
pip install -e .
sh ray_patch.sh
```

Verify the installation:

```bash
python -c "import meltingpot; print('OK')"
```

---

## 2. Project Structure Quick Reference

```
baselines/
  train/
    run_ray_train.py      # entry point for training
    configs.py            # model architecture, stopping conditions
  evaluation/
    evaluate.py           # entry point for evaluation
results/
  torch/
    <substrate_name>/
      PPO_meltingpot_<hash>/
        params.json                  # config used for this run
        checkpoint_000100/           # saved weights every 100 iterations
        checkpoint_000200/
        ...
        results_evals.csv            # written by evaluate.py
        results_evals_timestep.csv   # per-timestep returns (scenario eval)
```

---

## 3. Training

All training runs from `baselines/train/`:

```bash
cd baselines/train
```

### 3.1 Basic training command

```bash
python run_ray_train.py \
  --exp <substrate_name> \
  --no-tune \
  --num_workers 4
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--exp` | `pd_arena` | Substrate to train on (see list below) |
| `--no-tune` | off | Always pass this — disables hyperparameter search |
| `--num_workers` | 2 | Number of rollout workers. Use 4-8 locally, 12 on cloud |
| `--num_gpus` | 0 | Fraction of GPUs. This workload is CPU-bound; GPU not required |
| `--seed` | 123 | Random seed |
| `--results_dir` | `./results` | Where to write checkpoints |
| `--wandb` | False | Enable W&B logging (requires `WANDB_API_KEY` env var) |

Available `--exp` values:

```
commons_harvest__open
commons_harvest__open_disable_zapping
commons_harvest__open_abundance
commons_harvest__open_scarcity
commons_harvest__farmer
commons_harvest__farmer_zap
commons_harvest__closed
commons_harvest__partnership
commons_harvest__private_property
commons_harvest__private_property_pc
pd_arena
al_harvest
clean_up
territory_rooms
day_care
```

Training stops at **100M timesteps** by default (set in `configs.py`). Checkpoints are saved every 100 iterations under `results/torch/<substrate>/`.

### 3.2 Training all baseline substrates

Run each in a separate terminal (or sequentially):

```bash
# Baseline: open commons with zapping
python run_ray_train.py --exp commons_harvest__open --no-tune --num_workers 4

# No punishment control
python run_ray_train.py --exp commons_harvest__open_disable_zapping --no-tune --num_workers 4

# Farmer: positive reward shaping, no zapping
python run_ray_train.py --exp commons_harvest__farmer --no-tune --num_workers 4

# Scarcity: harsh resource pressure
python run_ray_train.py --exp commons_harvest__open_scarcity --no-tune --num_workers 4
```

### 3.3 Resuming an interrupted run

Use this when training was stopped unexpectedly (power cut, Ctrl+C, cloud preemption).
Point `--resume` at the **trial directory** (the one containing `params.json`).

```bash
# Find the trial directory
ls results/torch/commons_harvest__open/

# Resume — picks up from the last checkpoint automatically
python run_ray_train.py \
  --exp commons_harvest__open \
  --no-tune \
  --resume results/torch/commons_harvest__open/PPO_meltingpot_<hash>
```

The run continues from the last saved checkpoint with the same stopping condition.

### 3.4 Continuing a completed run

Use this when training finished (hit 100M timesteps) and you want to train longer,
or when you want to **fine-tune on a different substrate** starting from existing weights.
Point `--restore_checkpoint` at a specific **checkpoint folder**.

```bash
# Find the checkpoint folder
ls results/torch/commons_harvest__open/PPO_meltingpot_<hash>/

# Continue training the same substrate
python run_ray_train.py \
  --exp commons_harvest__open \
  --no-tune \
  --restore_checkpoint results/torch/commons_harvest__open/PPO_meltingpot_<hash>/checkpoint_000500

# Fine-tune on a different substrate (e.g. scarcity) starting from open weights
python run_ray_train.py \
  --exp commons_harvest__open_scarcity \
  --no-tune \
  --restore_checkpoint results/torch/commons_harvest__open/PPO_meltingpot_<hash>/checkpoint_000500
```

---

## 4. Evaluation

All evaluation runs from `baselines/evaluation/`:

```bash
cd baselines/evaluation
```

You need two things before evaluating:
- `--config_dir`: the trial directory containing `params.json`
- `--policies_dir`: the `policies` folder inside a checkpoint

```
results/torch/commons_harvest__open/PPO_meltingpot_<hash>/   ← config_dir
results/torch/commons_harvest__open/PPO_meltingpot_<hash>/checkpoint_000500/policies/  ← policies_dir
```

### 4.1 Self-play evaluation (substrate)

Evaluates the trained population against itself on the substrate (no DeepMind bots).

```bash
python evaluate.py \
  --config_dir results/torch/commons_harvest__open/PPO_meltingpot_<hash> \
  --policies_dir results/torch/commons_harvest__open/PPO_meltingpot_<hash>/checkpoint_000500/policies \
  --num_episodes 10
```

Outputs:
- `results_evals.csv` — one row per episode with per-agent returns
- `results_evals_timestep.csv` — cumulative return at every timestep

### 4.2 Self-play evaluation (scenario)

Evaluates on a specific named scenario (uses DeepMind background bots for some slots).

```bash
python evaluate.py \
  --config_dir results/torch/commons_harvest__open/PPO_meltingpot_<hash> \
  --policies_dir results/torch/commons_harvest__open/PPO_meltingpot_<hash>/checkpoint_000500/policies \
  --eval_on_scenario True \
  --scenario commons_harvest__open_0 \
  --num_episodes 10
```

Available scenarios per substrate (see `baselines/train/configs.py` for full list):

| Substrate | Scenarios |
|---|---|
| `commons_harvest__open` | `commons_harvest__open_0`, `commons_harvest__open_1` |
| `commons_harvest__open_disable_zapping` | `commons_harvest__open_disable_zapping_0`, `..._1` |
| `commons_harvest__farmer` | `commons_harvest__farmer_0`, `commons_harvest__farmer_1` |
| `commons_harvest__open_scarcity` | `commons_harvest__open_scarcity_0`, `..._1` |
| `commons_harvest__open_abundance` | `commons_harvest__open_abundance_0`, `..._1` |

### 4.3 Mixed evaluation (focal vs background)

Evaluates trained agents from one substrate against agents from another.
Focal agents occupy the first N slots; background agents occupy the last M slots.
**This is the main cross-population experiment infrastructure.**

```bash
python evaluate.py \
  --config_dir results/torch/<focal_substrate>/PPO_meltingpot_<hash> \
  --policies_dir results/torch/<focal_substrate>/PPO_meltingpot_<hash>/checkpoint_<N>/policies \
  --background_policies_dir results/torch/<background_substrate>/PPO_meltingpot_<hash>/checkpoint_<N>/policies \
  --num_background_agents 2 \
  --num_episodes 20
```

The output CSV includes per-episode columns:
- `focal_per_capita_return` / `background_per_capita_return`
- `focal_player_returns` / `background_player_returns` (list per agent)
- `focal_apples_eaten` / `background_apples_eaten` (list per agent)
- `focal_zaps_fired` / `background_zaps_fired` (list per agent, 0 if substrate has no zap)

Live episode progress is printed to stdout:
```
Episode 1/20 — focal per capita: 42.31  background per capita: 38.17  focal zaps: 0  bg zaps: 14
```

### 4.4 Reading the output CSVs

```python
import pandas as pd
import ast

# Episode-level results
df = pd.read_csv('results/torch/.../results_evals.csv')

# For mixed eval, columns with lists are stored as strings — parse them:
df['focal_player_returns'] = df['focal_player_returns'].apply(ast.literal_eval)
df['focal_zaps_fired'] = df['focal_zaps_fired'].apply(ast.literal_eval)

# Summary statistics
print(df['focal_per_capita_return'].describe())
print(df['background_per_capita_return'].describe())

# Total zaps across episodes
focal_total_zaps = df['focal_zaps_fired'].apply(sum).sum()
bg_total_zaps    = df['background_zaps_fired'].apply(sum).sum()
print(f"Focal zaps: {focal_total_zaps}, Background zaps: {bg_total_zaps}")
```

---

## 5. Priority Experiments

These require the baseline training runs from Section 3.2 to be complete first.
Replace `PPO_meltingpot_<hash>` and `checkpoint_<N>` with your actual directories.

---

### Exp 1: Farmer focal vs Open background

**Research question**: Can agents trained with positive reward shaping (no zap) survive
against agents that can zap them?

**Step 1** — Train both populations (if not already done):

```bash
python run_ray_train.py --exp commons_harvest__farmer --no-tune --num_workers 4
python run_ray_train.py --exp commons_harvest__open --no-tune --num_workers 4
```

**Step 2** — Run mixed evaluation:

```bash
python evaluate.py \
  --config_dir results/torch/commons_harvest__farmer/PPO_meltingpot_<hash> \
  --policies_dir results/torch/commons_harvest__farmer/PPO_meltingpot_<hash>/checkpoint_<N>/policies \
  --background_policies_dir results/torch/commons_harvest__open/PPO_meltingpot_<hash>/checkpoint_<N>/policies \
  --num_background_agents 2 \
  --num_episodes 20
```

**What to look for**:
- If `focal_per_capita_return` is competitive with `background_per_capita_return`: positive shaping
  survives under aggression — evidence that punishment is not required for cooperation.
- `focal_zaps_fired` should be 0 (farmer has no zap). `background_zaps_fired` shows exploitation.
- `focal_apples_eaten` vs `background_apples_eaten` shows whether farmer agents maintain
  efficient harvesting despite being zapped.

---

### Exp 2: No-Zap focal vs Open background

**Research question**: Does restricting the zap action from an agent's policy make it
vulnerable in a mixed population? (Tests action-space constraints as a safety mechanism.)

**Step 1** — Train both populations (if not already done):

```bash
python run_ray_train.py --exp commons_harvest__open_disable_zapping --no-tune --num_workers 4
python run_ray_train.py --exp commons_harvest__open --no-tune --num_workers 4
```

**Step 2** — Run mixed evaluation:

```bash
python evaluate.py \
  --config_dir results/torch/commons_harvest__open_disable_zapping/PPO_meltingpot_<hash> \
  --policies_dir results/torch/commons_harvest__open_disable_zapping/PPO_meltingpot_<hash>/checkpoint_<N>/policies \
  --background_policies_dir results/torch/commons_harvest__open/PPO_meltingpot_<hash>/checkpoint_<N>/policies \
  --num_background_agents 2 \
  --num_episodes 20
```

**Also run the reverse** (open agents as focal against no-zap background — tests whether
open agents exploit pacifist partners):

```bash
python evaluate.py \
  --config_dir results/torch/commons_harvest__open/PPO_meltingpot_<hash> \
  --policies_dir results/torch/commons_harvest__open/PPO_meltingpot_<hash>/checkpoint_<N>/policies \
  --background_policies_dir results/torch/commons_harvest__open_disable_zapping/PPO_meltingpot_<hash>/checkpoint_<N>/policies \
  --num_background_agents 2 \
  --num_episodes 20
```

**What to look for**:
- Compare `focal_per_capita_return` (no-zap) against `background_per_capita_return` (open).
- If no-zap is significantly lower: action-space constraints create vulnerability.
- If returns are comparable: sustainable harvesting strategy is robust to aggression.

---

### Exp 3: Dilemma severity axis

**Research question**: How does resource scarcity affect harvesting strategy and patch collapse?

**Step 1** — Train all three:

```bash
python run_ray_train.py --exp commons_harvest__open_abundance --no-tune --num_workers 4
python run_ray_train.py --exp commons_harvest__open --no-tune --num_workers 4
python run_ray_train.py --exp commons_harvest__open_scarcity --no-tune --num_workers 4
```

**Step 2** — Evaluate each on its own substrate:

```bash
for SUBSTRATE in commons_harvest__open_abundance commons_harvest__open commons_harvest__open_scarcity; do
  python evaluate.py \
    --config_dir results/torch/${SUBSTRATE}/PPO_meltingpot_<hash> \
    --policies_dir results/torch/${SUBSTRATE}/PPO_meltingpot_<hash>/checkpoint_<N>/policies \
    --num_episodes 20
done
```

**Step 3** — Zero-shot generalization: evaluate open-trained agents on scarcity substrate
(no retraining):

```bash
python evaluate.py \
  --config_dir results/torch/commons_harvest__open_scarcity/PPO_meltingpot_<hash> \
  --policies_dir results/torch/commons_harvest__open/PPO_meltingpot_<hash>/checkpoint_<N>/policies \
  --num_episodes 20
```

**What to look for**:
- `focal_per_capita_return` should decrease along abundance → open → scarcity.
- `focal_apples_eaten` total decreases as patches collapse earlier.
- Zero-shot: how much does per-capita return degrade when open policy faces scarcity?

---

## 6. Cloud Training (GCP)

### Create a preemptible VM

```bash
gcloud compute instances create marl-train \
  --machine-type=c2-standard-16 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --zone=us-central1-a
```

Approximate cost: **~$0.31/hr** (preemptible). Full baseline set (~60 hrs total compute):
~$20.

### Set up the VM

```bash
# SSH in
gcloud compute ssh marl-train

# Install conda + repo
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

git clone <this-repo>
cd Melting-Pot-MARL
conda create -n mpc_main python=3.10 -y
conda activate mpc_main
SYSTEM_VERSION_COMPAT=0 pip install dmlab2d
pip install -e .
sh ray_patch.sh
```

### Run training with more workers

On a 16-vCPU machine, use 12 workers (leave 4 cores for the trainer):

```bash
cd baselines/train
python run_ray_train.py \
  --exp commons_harvest__open \
  --no-tune \
  --num_workers 12
```

### Save results to GCS (so they survive VM termination)

```bash
# After training finishes or at any checkpoint
gsutil -m cp -r ./results gs://your-bucket/marl-results/

# Restore results on a new VM
gsutil -m cp -r gs://your-bucket/marl-results/ ./results
```

### Auto-resume on preemption with a startup script

Create `/home/<user>/startup.sh`:

```bash
#!/bin/bash
source $HOME/miniconda/bin/activate mpc_main
cd $HOME/Melting-Pot-MARL/baselines/train

# Sync latest results from GCS before resuming
gsutil -m rsync -r gs://your-bucket/marl-results/ ../../results/

python run_ray_train.py \
  --exp commons_harvest__open \
  --no-tune \
  --num_workers 12 \
  --resume ../../results/torch/commons_harvest__open/PPO_meltingpot_<hash>
```

Attach as VM startup script:

```bash
gcloud compute instances add-metadata marl-train \
  --metadata-from-file startup-script=/home/<user>/startup.sh
```

---

## 6b. Depletion Sweep (Fig. 5)

Measures what percentage of simulations end with fully-depleted apple patches,
sweeping zapping cooldown values [1, 2, 10, 200, 2000].

**Step 1** — Run once per agent type (replace paths and labels):

```bash
# open agents
python baselines/evaluation/depletion_sweep.py \
  --config_dir  results/torch/commons_harvest__open/PPO_meltingpot_<hash> \
  --policies_dir results/torch/commons_harvest__open/PPO_meltingpot_<hash>/checkpoint_001000/policies \
  --agent_label open \
  --num_episodes 15 \
  --output_dir  eval_results/depletion

# no_zap agents
python baselines/evaluation/depletion_sweep.py \
  --config_dir  results/torch/commons_harvest__open_disable_zapping/PPO_meltingpot_<hash> \
  --policies_dir results/torch/commons_harvest__open_disable_zapping/PPO_meltingpot_<hash>/checkpoint_001000/policies \
  --agent_label no_zap \
  --num_episodes 15 \
  --output_dir  eval_results/depletion

# scarcity agents
python baselines/evaluation/depletion_sweep.py \
  --config_dir  results/torch/commons_harvest__open_scarcity/PPO_meltingpot_<hash> \
  --policies_dir results/torch/commons_harvest__open_scarcity/PPO_meltingpot_<hash>/checkpoint_001000/policies \
  --agent_label scarcity \
  --num_episodes 15 \
  --output_dir  eval_results/depletion
```

Each run saves `<agent_label>_depletion.csv` and a single-agent bar chart to `--output_dir`.

**Step 2** — Combine all CSVs into Fig.5:

```bash
python baselines/evaluation/depletion_sweep.py \
  --plot_only \
  --output_dir eval_results/depletion
```

Saves `eval_results/depletion/fig5_depleted_simulations.png`.

**Output CSV columns**: `cooldown | episode | depleted | total_reward`

**Depletion definition**: an episode is counted as depleted if the total group reward
in the last 1000 steps of the episode is < 1.0 (i.e. no apples were eaten —
permanent patch collapse).

---

## 7. Finding Your Results

After training:

```
results/torch/
  commons_harvest__open/
    PPO_meltingpot_<hash>/
      params.json                   ← use as --config_dir
      progress.csv                  ← training metrics (reward, loss, etc.)
      checkpoint_000100/
        policies/                   ← use as --policies_dir
      checkpoint_000200/
        policies/
      ...
      results_evals.csv             ← written after evaluate.py runs
      results_evals_timestep.csv    ← per-timestep returns (scenario eval only)
```

To find the latest checkpoint:

```bash
ls -lt results/torch/commons_harvest__open/PPO_meltingpot_<hash>/ | grep checkpoint
```

To check training progress:

```python
import pandas as pd
df = pd.read_csv('results/torch/commons_harvest__open/PPO_meltingpot_<hash>/progress.csv')
print(df[['training_iteration', 'timesteps_total', 'episode_reward_mean']].tail(20))
```

---

## 8. Troubleshooting

**`ModuleNotFoundError: No module named 'baselines'`**
Run all scripts from the repo root, not from inside `baselines/`:
```bash
cd /path/to/Melting-Pot-MARL
python baselines/train/run_ray_train.py ...
```

**`Ray already initialized` error**
```bash
ray stop
```

**Evaluation hangs after `ray.init()`**
Add `ray.init(local_mode=True)` for debugging, or check that no other Ray process is running:
```bash
ray status
```

**`params.json` not found**
Point `--config_dir` at the trial directory itself (the one with `params.json`), not the
checkpoint folder inside it.

**Checkpoint not found on resume**
The `--resume` path must be the **trial directory** (contains `params.json`), not a specific
checkpoint. Ray will find the latest checkpoint automatically.

**OOM during training (CPU)**
Reduce `--num_workers`. Each worker holds a full copy of the environment in memory.
For `commons_harvest` substrates with 7 agents, each environment is ~200MB.

**Zap columns all zeros in mixed eval CSV**
Expected for `commons_harvest__farmer` (no zap action) and
`commons_harvest__open_disable_zapping` (zap is inert). The `READY_TO_SHOOT` observation
is absent in these substrates so the tracker correctly records 0.
