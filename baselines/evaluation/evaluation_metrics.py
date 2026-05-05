"""
evaluation_metrics.py

Domain metrics for commons_harvest evaluations.
All functions take numpy arrays produced directly by the evaluation loop
(rewards, zaps, apples eaten, step counts) — no mock data.

Metrics
-------
cooperation_index       Fraction of agent-steps NOT spent zapping (0=full defection, 1=no aggression)
zap_rate                Zaps fired per agent per step
harvest_rate            Apples eaten per agent per step
sustainability_index    Ratio of late-episode to early-episode harvest rate (1=stable, <1=collapsing)
gini_coefficient        Inequality of returns across agents (0=equal, 1=maximal inequality)
episode_depleted        1 if the patch collapsed permanently (zero harvest in final window), else 0
"""

import numpy as np


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def cooperation_index(zaps_fired: np.ndarray, episode_steps: int, n_agents: int) -> float:
    """Fraction of agent-steps with no zap fired.

    1.0 = no agent ever zapped (fully cooperative)
    0.0 = every agent-step included a zap (maximally aggressive)
    """
    agent_steps = episode_steps * n_agents
    if agent_steps == 0:
        return 1.0
    return 1.0 - float(zaps_fired.sum()) / agent_steps


def zap_rate(zaps_fired: np.ndarray, episode_steps: int, n_agents: int) -> float:
    """Zaps per agent per step."""
    agent_steps = episode_steps * n_agents
    return float(zaps_fired.sum()) / agent_steps if agent_steps > 0 else 0.0


def harvest_rate(apples_eaten: np.ndarray, episode_steps: int, n_agents: int) -> float:
    """Apples eaten per agent per step."""
    agent_steps = episode_steps * n_agents
    return float(apples_eaten.sum()) / agent_steps if agent_steps > 0 else 0.0


def sustainability_index(step_rewards: list, split: float = 0.5) -> float:
    """Ratio of per-step harvest rate in the second half vs first half of episode.

    > 1  harvest increased over time (patches recovered / rare)
    = 1  stable harvesting
    < 1  harvest collapsed (patches depleted as episode progressed)
    0    complete collapse in second half

    Parameters
    ----------
    step_rewards : list of floats — total group reward at each env step
    split        : fraction of episode that defines the boundary (default 0.5)
    """
    if len(step_rewards) < 2:
        return 1.0
    mid = int(len(step_rewards) * split)
    first_half  = np.mean(step_rewards[:mid])  if mid > 0 else 0.0
    second_half = np.mean(step_rewards[mid:])
    if first_half <= 0:
        return 1.0 if second_half <= 0 else float('inf')
    return float(second_half) / float(first_half)


def gini_coefficient(returns: np.ndarray) -> float:
    """Gini coefficient of per-agent episode returns.

    0.0 = perfectly equal returns across agents
    1.0 = maximally unequal (one agent takes everything)
    """
    if len(returns) == 0 or returns.sum() == 0:
        return 0.0
    sorted_r = np.sort(np.abs(returns))
    n = len(sorted_r)
    cumulative = np.cumsum(sorted_r)
    return float((2 * np.sum((np.arange(1, n + 1)) * sorted_r)
                  - (n + 1) * cumulative[-1]) / (n * cumulative[-1]))


def episode_depleted(step_rewards: list,
                     window: int = 1000,
                     threshold: float = 1.0) -> int:
    """1 if the episode ended with patch collapse, else 0.

    An episode is classified as depleted when the total group reward
    in the last `window` steps is below `threshold` (no apples eaten
    = permanent patch collapse).
    """
    tail = step_rewards[-window:] if len(step_rewards) >= window else step_rewards
    return int(sum(tail) < threshold)


# ---------------------------------------------------------------------------
# Composite: compute all metrics for one episode
# ---------------------------------------------------------------------------

def compute_episode_metrics(
    focal_returns:  np.ndarray,
    bg_returns:     np.ndarray,
    apples_eaten:   np.ndarray,
    zaps_fired:     np.ndarray,
    step_rewards:   list,
) -> dict:
    """Return a dict of all domain metrics for a single episode.

    Parameters
    ----------
    focal_returns  : per-agent cumulative reward for focal agents      (n_focal,)
    bg_returns     : per-agent cumulative reward for background agents  (n_bg,)
    apples_eaten   : total apples eaten per agent over episode          (n_agents,)
    zaps_fired     : total zaps fired per agent over episode            (n_agents,)
    step_rewards   : total group reward at each env step                list[float]
    """
    n_agents      = len(apples_eaten)
    episode_steps = len(step_rewards)
    all_returns   = np.concatenate([focal_returns, bg_returns])

    return {
        "cooperation_index":    cooperation_index(zaps_fired, episode_steps, n_agents),
        "zap_rate":             zap_rate(zaps_fired, episode_steps, n_agents),
        "harvest_rate":         harvest_rate(apples_eaten, episode_steps, n_agents),
        "sustainability_index": sustainability_index(step_rewards),
        "gini_coefficient":     gini_coefficient(all_returns),
        "episode_depleted":     episode_depleted(step_rewards),
        "total_apples_harvested": int(apples_eaten.sum()),
        "total_zaps_fired":       int(zaps_fired.sum()),
        "episode_steps":          episode_steps,
    }
