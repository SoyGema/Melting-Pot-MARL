"""
depletion_sweep.py

Measures the percentage of fully-depleted simulations across a sweep of
zapping cooldown values for a trained population.  Replicates Fig. 5 from
the project paper (Parreño et al., 2024).

A simulation is classified as "fully depleted" when the total group reward
earned in the final DEPLETION_WINDOW steps is below DEPLETION_THRESHOLD --
a proxy for permanent apple-patch collapse.

Usage (single agent type):
  python baselines/evaluation/depletion_sweep.py \\
    --config_dir  results/torch/commons_harvest__open/PPO_meltingpot_<hash> \\
    --policies_dir results/torch/commons_harvest__open/PPO_meltingpot_<hash>/checkpoint_001000/policies \\
    --agent_label open \\
    --num_episodes 15 \\
    --output_dir   eval_results/depletion

Run once per agent type (open, no_zap, farmer, scarcity, private).
Then combine the per-agent CSVs for the full Fig.5 plot:

  python baselines/evaluation/depletion_sweep.py --plot_only \\
    --output_dir eval_results/depletion
"""

import argparse
import contextlib
import importlib
import json
import os

import dm_env
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ray
import meltingpot

# W&B is optional.  Enable with --wandb flag + WANDB_API_KEY env var.
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
from meltingpot.utils.substrates import substrate as substrate_lib
from baselines.customs.policies import EvalPolicy
from baselines.wrappers.downsamplepolicy_wrapper import DownsamplingPolicyWraper


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COOLDOWNS = [1, 2, 10, 200, 2000]

# Substrate used for evaluation.  All commons_harvest agents share the same
# observation space as commons_harvest__open (confirmed: open and
# open_disable_zapping have identical obs spaces).  This substrate has a
# Zapper component whose cooldownTime we sweep.
EVAL_SUBSTRATE = "commons_harvest__open"

# Depletion detection: if the total group reward in the last DEPLETION_WINDOW
# steps is below DEPLETION_THRESHOLD the episode counts as fully depleted.
DEPLETION_WINDOW    = 1000   # steps  (episode max = 5000)
DEPLETION_THRESHOLD = 1.0    # apples eaten in the window


# ---------------------------------------------------------------------------
# Environment building with patched cooldown
# ---------------------------------------------------------------------------

def _patch_cooldown(settings: dict, cooldown: int) -> dict:
    """Set cooldownTime on every Zapper component in the lab2d settings dict."""
    for obj in settings.get("simulation", {}).get("gameObjects", []):
        for comp in obj.get("components", []):
            if comp.get("component") == "Zapper":
                comp["kwargs"]["cooldownTime"] = cooldown
    return settings


@contextlib.contextmanager
def _build_env(substrate_name: str, cooldown: int, roles):
    """Context manager: build substrate with patched zapping cooldown."""
    module  = importlib.import_module(
        f"meltingpot.configs.substrates.{substrate_name}")
    config  = module.get_config()
    settings = config.lab2d_settings_builder(roles=roles, config=config)

    _patch_cooldown(settings, cooldown)

    # Borrow obs/action metadata from the standard factory.
    factory = meltingpot.substrate.get_factory(substrate_name)
    env = substrate_lib.build_substrate(
        lab2d_settings=settings,
        individual_observations=factory._individual_observations,
        global_observations=factory._global_observations,
        action_table=factory._action_table,
    )
    try:
        yield env
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Policy loading helpers (mirrors evaluate.py)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _load_policies(policies_dir: str, policy_ids: list, scale: int):
    """Load EvalPolicy for each agent id, wrapped with downsampler."""
    with contextlib.ExitStack() as stack:
        yield {
            pid: stack.enter_context(
                DownsamplingPolicyWraper(EvalPolicy(policies_dir, pid), scale))
            for pid in policy_ids
        }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _make_single_agent_timestep(multi_ts: dm_env.TimeStep,
                                 agent_idx: int) -> dm_env.TimeStep:
    """Slice a multi-agent TimeStep into a single-agent view."""
    return dm_env.TimeStep(
        step_type=multi_ts.step_type,
        reward=multi_ts.reward[agent_idx] if multi_ts.reward is not None else None,
        discount=multi_ts.discount[agent_idx] if multi_ts.discount is not None else None,
        observation=multi_ts.observation[agent_idx],
    )


def _episode_is_depleted(reward_history: list) -> bool:
    """Return True if apple patches were permanently wiped out this episode."""
    window = reward_history[-DEPLETION_WINDOW:] if len(reward_history) >= DEPLETION_WINDOW \
             else reward_history
    total = sum(sum(r) for r in window)
    return total < DEPLETION_THRESHOLD


def run_sweep(policies_dir: str, policy_ids: list, scale: int,
              num_episodes: int, substrate_name: str = EVAL_SUBSTRATE) -> pd.DataFrame:
    """
    Sweep cooldown values and return a DataFrame with columns:
      cooldown | episode | depleted | total_reward
    """
    n_agents = len(policy_ids)
    records  = []

    for cooldown in COOLDOWNS:
        print(f"\n--- cooldown = {cooldown} ---")
        with _load_policies(policies_dir, policy_ids, scale) as policies, \
             _build_env(substrate_name, cooldown, ("default",) * n_agents) as env:

            for ep in range(num_episodes):
                # Reset environment and policy states
                multi_ts = env.reset()
                prev_states = {pid: pol.initial_state()
                               for pid, pol in policies.items()}

                reward_history = []
                ep_total_reward = 0.0

                while not multi_ts.step_type.last():
                    # Compute actions for each agent
                    actions = []
                    for i, (pid, pol) in enumerate(sorted(policies.items())):
                        single_ts  = _make_single_agent_timestep(multi_ts, i)
                        action, st = pol.step(single_ts, prev_states[pid])
                        prev_states[pid] = st
                        actions.append(action)

                    multi_ts = env.step(actions)

                    step_rewards = list(multi_ts.reward)
                    reward_history.append(step_rewards)
                    ep_total_reward += sum(step_rewards)

                depleted = _episode_is_depleted(reward_history)
                records.append({
                    "cooldown":     cooldown,
                    "episode":      ep + 1,
                    "depleted":     depleted,
                    "total_reward": ep_total_reward,
                })
                print(f"  ep {ep+1:3d}/{num_episodes}  "
                      f"total_reward={ep_total_reward:7.1f}  "
                      f"depleted={depleted}")
                if _WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        "cooldown":     cooldown,
                        "episode":      ep + 1,
                        "depleted":     int(depleted),
                        "total_reward": ep_total_reward,
                    })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single(df: pd.DataFrame, agent_label: str, output_path: str):
    """Bar chart of % depleted per cooldown for a single agent type."""
    pct = (df.groupby("cooldown")["depleted"].mean() * 100).reindex(COOLDOWNS)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar([str(c) for c in COOLDOWNS], pct.values, color="#3876B4")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Zapping Cooldown")
    ax.set_ylabel("% Depleted Simulations")
    ax.set_title(f"Agent: {agent_label}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved single-agent plot: {output_path}")


def plot_combined(csv_paths: list[str], agent_labels: list[str], output_path: str):
    """Recreate Fig.5: one panel per agent type, cooldown on x-axis."""
    n = len(agent_labels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    fig.suptitle("Percentage of Depleted Simulations")

    for ax, path, label in zip(axes, csv_paths, agent_labels):
        df  = pd.read_csv(path)
        pct = (df.groupby("cooldown")["depleted"].mean() * 100).reindex(COOLDOWNS)
        ax.bar([str(c) for c in COOLDOWNS], pct.values,
               color="#3876B4", label="% depleted")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Zapping Cooldown")
        ax.set_title(f"Agent: {label}")
        if ax is axes[0]:
            ax.set_ylabel("% Depleted Simulations")

    axes[-1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved combined plot: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="Depletion sweep across zapping cooldown values (Fig.5)")

    # -- single-agent run mode
    p.add_argument("--config_dir",   type=str, default=None,
                   help="Trial directory containing params.json")
    p.add_argument("--policies_dir", type=str, default=None,
                   help="Checkpoint policies folder")
    p.add_argument("--agent_label",  type=str, default="agent",
                   help="Short name for this agent type (used in plot title / filename)")
    p.add_argument("--num_episodes", type=int, default=15,
                   help="Episodes per cooldown value")
    p.add_argument("--eval_substrate", type=str,
                   default=EVAL_SUBSTRATE,
                   help=f"Substrate to evaluate on (default: {EVAL_SUBSTRATE})")

    # -- combined plot mode
    p.add_argument("--plot_only",   action="store_true",
                   help="Skip evaluation; just plot all CSVs found in --output_dir")
    p.add_argument("--output_dir",  type=str, default="eval_results/depletion",
                   help="Directory for CSVs and plots")

    # -- W&B
    p.add_argument("--wandb", action="store_true", default=False,
                   help="Log metrics and plots to Weights & Biases.")
    p.add_argument("--wandb_project", type=str, default="meltingpot-eval",
                   help="W&B project name (default: meltingpot-eval).")

    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- combined plot mode ----
    if args.plot_only:
        csvs   = sorted(
            f for f in os.listdir(args.output_dir) if f.endswith("_depletion.csv"))
        labels = [f.replace("_depletion.csv", "") for f in csvs]
        paths  = [os.path.join(args.output_dir, f) for f in csvs]
        if not paths:
            print("No *_depletion.csv files found in", args.output_dir)
            return
        plot_combined(paths, labels,
                      os.path.join(args.output_dir, "fig5_depleted_simulations.png"))
        return

    # ---- single-agent evaluation mode ----
    if not args.config_dir or not args.policies_dir:
        raise ValueError("--config_dir and --policies_dir are required unless --plot_only")

    ray.init(ignore_reinit_error=True)

    # Initialise W&B if requested
    if args.wandb:
        if not _WANDB_AVAILABLE:
            print("WARNING: wandb not installed. Run: pip install wandb")
        else:
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"depletion_{args.agent_label}",
            )

    config = json.load(open(f"{args.config_dir}/params.json"))
    scale  = config["env_config"]["scaled"]
    roles  = config["env_config"]["roles"]
    policy_ids = [f"agent_{i}" for i in range(len(roles))]

    print(f"\nAgent label : {args.agent_label}")
    print(f"Substrate   : {args.eval_substrate}")
    print(f"Episodes    : {args.num_episodes} per cooldown")
    print(f"Cooldowns   : {COOLDOWNS}\n")

    df = run_sweep(
        policies_dir=args.policies_dir,
        policy_ids=policy_ids,
        scale=scale,
        num_episodes=args.num_episodes,
        substrate_name=args.eval_substrate,
    )

    # Save CSV
    csv_path = os.path.join(args.output_dir, f"{args.agent_label}_depletion.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results: {csv_path}")

    # Per-agent summary
    summary = df.groupby("cooldown").agg(
        pct_depleted=("depleted", lambda x: 100 * x.mean()),
        mean_reward=("total_reward", "mean"),
    ).reindex(COOLDOWNS)
    print("\n=== Summary ===")
    print(summary.to_string())

    # Plot single-agent chart
    plot_path = os.path.join(args.output_dir, f"{args.agent_label}_depletion.png")
    plot_single(df, args.agent_label, plot_path)

    # W&B: log summary table + bar chart
    if _WANDB_AVAILABLE and wandb.run is not None:
        summary = df.groupby("cooldown").agg(
            pct_depleted=("depleted", lambda x: 100 * x.mean()),
            mean_reward=("total_reward", "mean"),
        ).reindex(COOLDOWNS).reset_index()
        wandb.log({
            "depletion_summary": wandb.Table(dataframe=summary),
            "depletion_plot":    wandb.Image(plot_path),
        })
        wandb.finish()

    ray.shutdown()


if __name__ == "__main__":
    main()
