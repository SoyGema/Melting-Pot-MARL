"""
evaluation_metrics.py

Domain metrics for commons_harvest evaluations.
All functions take numpy arrays produced directly by the evaluation loop
(rewards, zaps, apples eaten, step counts, positions) — no mock data.

Behavioral metrics (require rewards / zaps / apples)
-----------------------------------------------------
cooperation_index       Fraction of agent-steps NOT spent zapping (0=full defection, 1=no aggression)
zap_rate                Zaps fired per agent per step
harvest_rate            Apples eaten per agent per step
sustainability_index    Ratio of late-episode to early-episode harvest rate (1=stable, <1=collapsing)
gini_coefficient        Inequality of returns across agents (0=equal, 1=maximal inequality)
episode_depleted        1 if the patch collapsed permanently (zero harvest in final window), else 0

Spatiotemporal metrics (require POSITION observations)
------------------------------------------------------
spatial_entropy         Shannon entropy of agent positions over episode (high = spread out / avoidance)
mean_interagent_dist    Mean pairwise distance between agents averaged over steps
patch_revisit_rate      How often agents return to the same grid cell (high = territoriality)
turn_taking_index       Negative temporal cross-correlation of per-patch visits (high = turn-taking)
avoidance_index         Fraction of steps where agents are above mean pairwise distance (spreading out)
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
# Spatiotemporal metrics  (require position_history)
# ---------------------------------------------------------------------------

def spatial_entropy(position_history: np.ndarray, map_width: int = 24, map_height: int = 24) -> float:
    """Shannon entropy of the joint agent position distribution over the episode.

    High entropy = agents spread uniformly across map (avoidance / coalition avoidance).
    Low entropy  = agents cluster in the same locations (competition / co-location).

    Parameters
    ----------
    position_history : (T, N, 2) int array — x,y position of each agent at each step
    """
    T, N, _ = position_history.shape
    # Discretise into cells and count visits
    positions = position_history.reshape(-1, 2)  # (T*N, 2)
    cell_ids  = positions[:, 0] * map_height + positions[:, 1]
    counts    = np.bincount(cell_ids, minlength=map_width * map_height).astype(float)
    probs     = counts / counts.sum()
    probs     = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def mean_interagent_distance(position_history: np.ndarray) -> float:
    """Mean pairwise Euclidean distance between agents, averaged over all steps.

    High = agents spread apart (spatial avoidance emerging as punishment substitute).
    Low  = agents cluster together (competition / co-location).

    Parameters
    ----------
    position_history : (T, N, 2) int array
    """
    T, N, _ = position_history.shape
    if N < 2:
        return 0.0
    dists = []
    for t in range(T):
        pos = position_history[t]  # (N, 2)
        for i in range(N):
            for j in range(i + 1, N):
                dists.append(np.linalg.norm(pos[i] - pos[j]))
    return float(np.mean(dists))


def patch_revisit_rate(position_history: np.ndarray) -> float:
    """Mean fraction of an agent's steps spent in its most-visited cell.

    High = agents return repeatedly to the same location (territorial / farming behavior).
    Low  = agents roam freely (exploratory / migratory behavior).

    Parameters
    ----------
    position_history : (T, N, 2) int array
    """
    T, N, _ = position_history.shape
    rates = []
    for n in range(N):
        positions = position_history[:, n, :]          # (T, 2)
        cell_ids  = positions[:, 0] * 1000 + positions[:, 1]
        counts    = np.bincount(cell_ids)
        rates.append(float(counts.max()) / T)
    return float(np.mean(rates))


def turn_taking_index(position_history: np.ndarray, cell_size: int = 3) -> float:
    """Measures whether agents alternate access to the same spatial regions.

    Computed as the mean negative cross-correlation between any pair of agents'
    presence in the same coarse grid cell, lagged by 1 step.
    Positive values → agents take turns (one leaves when the other arrives).
    Near zero      → no temporal coordination.

    Parameters
    ----------
    position_history : (T, N, 2) int array
    cell_size        : coarse grid cell size in tiles
    """
    T, N, _ = position_history.shape
    if N < 2 or T < 10:
        return 0.0
    # Coarse cell assignment
    coarse = (position_history // cell_size)  # (T, N, 2)
    cell_ids = coarse[:, :, 0] * 1000 + coarse[:, :, 1]  # (T, N)

    correlations = []
    for i in range(N):
        for j in range(i + 1, N):
            same_cell = (cell_ids[:, i] == cell_ids[:, j]).astype(float)  # (T,)
            if same_cell.sum() < 2:
                continue
            # Lag-1 cross-correlation: if agent i is here at t, is agent j here at t+1?
            a, b = same_cell[:-1], same_cell[1:]
            if a.std() > 0 and b.std() > 0:
                corr = float(np.corrcoef(a, b)[0, 1])
                correlations.append(-corr)  # negative = they alternate
    return float(np.mean(correlations)) if correlations else 0.0


def avoidance_index(position_history: np.ndarray) -> float:
    """Fraction of steps where mean inter-agent distance exceeds the episode mean.

    High = agents are more often spread out than average → active avoidance behavior.

    Parameters
    ----------
    position_history : (T, N, 2) int array
    """
    T, N, _ = position_history.shape
    if N < 2:
        return 0.0
    step_dists = []
    for t in range(T):
        pos = position_history[t]
        d = [np.linalg.norm(pos[i] - pos[j])
             for i in range(N) for j in range(i + 1, N)]
        step_dists.append(np.mean(d))
    step_dists = np.array(step_dists)
    return float((step_dists > step_dists.mean()).mean())


def compute_spatiotemporal_metrics(position_history: np.ndarray) -> dict:
    """Return all spatiotemporal metrics for one episode.

    Parameters
    ----------
    position_history : (T, N, 2) int array — POSITION obs stacked over all steps
    """
    return {
        "spatial_entropy":        spatial_entropy(position_history),
        "mean_interagent_dist":   mean_interagent_distance(position_history),
        "patch_revisit_rate":     patch_revisit_rate(position_history),
        "turn_taking_index":      turn_taking_index(position_history),
        "avoidance_index":        avoidance_index(position_history),
    }


# ---------------------------------------------------------------------------
# Composite: compute all metrics for one episode
# ---------------------------------------------------------------------------

def compute_episode_metrics(
    focal_returns:    np.ndarray,
    bg_returns:       np.ndarray,
    apples_eaten:     np.ndarray,
    zaps_fired:       np.ndarray,
    step_rewards:     list,
    position_history: np.ndarray = None,
) -> dict:
    """Return a dict of all domain metrics for a single episode.

    Parameters
    ----------
    focal_returns    : per-agent cumulative reward for focal agents      (n_focal,)
    bg_returns       : per-agent cumulative reward for background agents  (n_bg,)
    apples_eaten     : total apples eaten per agent over episode          (n_agents,)
    zaps_fired       : total zaps fired per agent over episode            (n_agents,)
    step_rewards     : total group reward at each env step                list[float]
    position_history : POSITION obs stacked over steps (T, N, 2) or None
    """
    n_agents      = len(apples_eaten)
    episode_steps = len(step_rewards)
    all_returns   = np.concatenate([focal_returns, bg_returns])

    metrics = {
        "cooperation_index":      cooperation_index(zaps_fired, episode_steps, n_agents),
        "zap_rate":               zap_rate(zaps_fired, episode_steps, n_agents),
        "harvest_rate":           harvest_rate(apples_eaten, episode_steps, n_agents),
        "sustainability_index":   sustainability_index(step_rewards),
        "gini_coefficient":       gini_coefficient(all_returns),
        "episode_depleted":       episode_depleted(step_rewards),
        "total_apples_harvested": int(apples_eaten.sum()),
        "total_zaps_fired":       int(zaps_fired.sum()),
        "episode_steps":          episode_steps,
    }

    if position_history is not None:
        metrics.update(compute_spatiotemporal_metrics(position_history))

    return metrics
