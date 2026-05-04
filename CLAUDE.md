# CLAUDE.md — Melting-Pot-MARL Project Guide

## Project overview

Research project studying cooperative behavior in multi-agent reinforcement learning using
[DeepMind Melting Pot](https://github.com/deepmind/meltingpot) as the environment framework.
The focus substrate is **commons_harvest** and its variants (tragedy of the commons dynamics).
Primary research question: how does training condition affect cooperative vs selfish behavior?

Reference paper: Melting Pot 2.0 (arXiv 2211.13746)

## Git workflow

- **Always create PRs against `SoyGema/Melting-Pot-MARL`** (remote: `soygema`)
- `origin` → whymath/Melting-Pot-MARL (do NOT use for PRs)
- Main branch for PRs: `main`

## Environment setup

```bash
conda activate mpc_main   # Python 3.10
# Ray 2.10.0, numpy 1.26.4, torch backend
# After any Ray reinstall, always re-apply patches:
COMPLEX_NET=$(python -c "from ray.rllib.models.torch import complex_input_net; print(complex_input_net.__file__)" 2>/dev/null | grep ".py")
SAMPLE_BATCH=$(python -c "from ray.rllib.policy import sample_batch; print(sample_batch.__file__)" 2>/dev/null | grep ".py")
patch -f "$COMPLEX_NET" < ray_patches/complex_input_net.patch
patch -f "$SAMPLE_BATCH" < ray_patches/sample_batch.patch
```

**Critical**: the ray_patch changes `complex_input_net.py` line 181 so that
`num_outputs = post_fc_stack.num_outputs` (instead of concat_size).
Without it, LSTM input_size mismatch → `RuntimeError: input.size(-1) Expected 144, got 24`.

Run scripts always require `PYTHONPATH` set to repo root:
```bash
env PYTHONPATH=/Users/gema/Documents/Melting-Pot-MARL python baselines/...
```

Trained checkpoints live in `/Users/gema/ray_results/`.

## Code structure

```
meltingpot/
  configs/
    substrates/       # Environment definitions (reward, map, mechanics)
    scenarios/        # Evaluation scenarios (focal + background bot combos)
    bots/             # Pre-trained DeepMind bots (saved_model references)

baselines/
  train/
    run_ray_train.py  # Main training entry point (PPO via RLLib)
    configs.py        # Experiment configs + SUPPORTED_SCENARIOS list
    make_envs.py      # Environment factory
  evaluation/
    evaluate.py       # Evaluation script → saves results_evals.csv
    plot_evaluation.py # Plot focal vs background per capita return (Fig.2-style)
    evaluation_metrics.py # Domain metrics (sustainability, cooperation index...)
  customs/
    policies.py       # EvalPolicy: loads RLLib checkpoint for evaluation
  wrappers/
    meltingpot_wrapper.py         # Main gym wrapper
    downsamplesubstrate_wrapper.py # Downsamples RGB obs by factor 8
    downsamplepolicy_wrapper.py   # Downsamples obs at inference time
```

## Training

```bash
python baselines/train/run_ray_train.py \
  --exp commons_harvest__open \
  --algo ppo --framework torch \
  --num_workers 2 --results_dir ./results
```

Supported `--exp` values: `commons_harvest__open`, `commons_harvest__open_disable_zapping`,
`commons_harvest__private_property`, `commons_harvest__private_property_pc`,
`commons_harvest__closed`, `commons_harvest__partnership`, `commons_harvest__open_abundance`,
`commons_harvest__open_scarcity`, `pd_arena`, `al_harvest`, `clean_up`, `territory_rooms`, `day_care`.

Model config (configs.py): CNN `[[16,[8,8],1],[128,[sprite_x,sprite_y],1]]` + LSTM cell_size=2,
fcnet_hidden=(4,4), post_fcnet_hidden=(16,). Each agent has independent policy.

## Evaluation

```bash
# On substrate only (no background):
python baselines/evaluation/evaluate.py \
  --config_dir <ray_results_run_dir> \
  --policies_dir <ray_results_run_dir>/checkpoint_XXXXXX/policies \
  --num_episodes 10

# On scenario (with DeepMind background bots):
python baselines/evaluation/evaluate.py \
  --eval_on_scenario True \
  --scenario commons_harvest__open_0 \
  --config_dir <run_dir> --policies_dir <run_dir>/checkpoint/policies
```

Output CSV columns: `focal_per_capita_return`, `background_per_capita_return`,
`focal_player_returns` (array), `background_player_returns` (array).

## Plotting (Fig.2-style)

```bash
python baselines/evaluation/plot_evaluation.py \
  --results_dir ./eval_results \       # subdirs named by substrate, each with CSVs per run
  --focal_label "open" \
  --background_label "disable_zapping" \
  --output plot.png
```

Directory structure: `results_dir/<substrate_name>/results_evals_<seed>.csv`

## Trained checkpoints available

| Substrate | Location | Iterations | Reward |
|---|---|---|---|
| commons_harvest__open | ray_results/commons_harvest__open/PPO_meltingpot_f9685... | 578 | 310 |
| commons_harvest__private_property_pc | ray_results/commons_harvest__private_property_pc/... | 629 | 331 |
| commons_harvest__open_disable_zapping | ray_results/commons_harvest__open_disable_zapping/PPO_meltingpot_6fb20... | 551 | 279 |
| commons_harvest__open (v2) | ray_results/commons_harvest__open 2/PPO_meltingpot_41239... | 607 | 265 |
| commons_harvest__closed | ray_results/commons_harvest__closed/... | 405 | 213 |
| commons_harvest__partnership | ray_results/commons_harvest__partnership/... | 140 | 80 |

All checkpoints: Ray 2.10.0, torch framework, downsample=8, obs_space=(11,11,3).

## Key substrate facts (commons_harvest__open)

- Reward: +1 per apple eaten. No proximity shaping. No other reward signal.
- 7 players, role: `"default"` (no farmer/pacifist roles — those are conceptual labels)
- Apples regrow based on neighborhood density → tragedy of commons when over-harvested
- Actions include zap (stun other players). `open_disable_zapping` removes this action.
- Observation space keys: `RGB (11,11,3)`, `COLLECTIVE_REWARD`, `READY_TO_SHOOT`
  - `open` and `open_disable_zapping` have **identical observation spaces** ✓

## Background population design (planned)

Goal: use our own pre-trained agents as background instead of DeepMind bots.
Rationale: fair comparison — same compute budget, same PPO pipeline, different training condition.

Proposed design (mirrors DeepMind's free/pacifist split):
- **"free" focal**: agents trained on `commons_harvest__open` (zap available, self-interested)
- **"pacifist" background**: agents trained on `commons_harvest__open_disable_zapping` (no zap)

This is valid because both substrates have **identical observation spaces** (confirmed above).
The pacifist agents simply never learned to zap — they act cooperatively by construction.
Evaluating focal (free) agents against pacifist background in `commons_harvest__open` substrate
tests generalization: can focal agents maintain performance when paired with cooperative partners?
