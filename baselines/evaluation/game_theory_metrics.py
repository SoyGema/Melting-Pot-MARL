"""
game_theory_metrics.py

Game-theoretic metrics for characterising emergent behavior in commons_harvest
mixed-motive evaluations.

These functions operate on aggregated episode data (numpy arrays / lists of
per-episode scalars) rather than on the raw step-level loop.  They are designed
to be called post-hoc on the CSVs produced by evaluate.py.

Metrics
-------
nash_deviation              How far observed harvesting is from the Nash Equilibrium
                            (full defection / max harvest rate)
price_of_anarchy            Ratio of social optimum to worst-case equilibrium reward.
                            Quantifies how costly defection is collectively.
evolutionary_stability      Whether the cooperator strategy is ESS: does a cooperator
                            earn more than a rogue in a mixed population?
cooperation_support         Fraction of episodes where cooperators out-earn rogues.
correlated_equilibrium_score  Proxy for whether agents are playing a correlated
                            equilibrium (conditional turn-taking / avoidance).
folk_theorem_index          Estimate of whether LSTM memory is sufficient to sustain
                            conditional cooperation (based on reward trajectory shape).

Usage
-----
import pandas as pd
from baselines.evaluation.game_theory_metrics import compute_game_theory_metrics

# Load mixed eval CSV (focal = cooperators, background = rogues)
df = pd.read_csv("results/.../results_evals.csv")

metrics = compute_game_theory_metrics(
    focal_per_capita=df["focal_per_capita_return"].values,
    background_per_capita=df["background_per_capita_return"].values,
    total_rewards=df["total_apples_harvested"].values,
    sustainability_index=df["sustainability_index"].values,
    turn_taking_index=df["turn_taking_index"].values,   # requires spatiotemporal branch
)
print(metrics)
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def nash_deviation(observed_harvest_rate: float,
                   max_harvest_rate: float) -> float:
    """How far below the Nash Equilibrium (full defection) the agents harvest.

    The NE in a commons dilemma with selfish agents is maximum harvesting speed.
    Any restraint below the NE requires explanation — emergent cooperation,
    spatial avoidance, temporal coordination, etc.

    Returns
    -------
    float in [0, 1]
        0   = agents are playing the NE (full defection)
        1   = agents harvest nothing (maximum restraint)
        Negative values are possible if observed > max (shouldn't happen in practice)
    """
    if max_harvest_rate <= 0:
        return 0.0
    return float((max_harvest_rate - observed_harvest_rate) / max_harvest_rate)


def price_of_anarchy(social_optimum_reward: float,
                     nash_approx_reward: float) -> float:
    """Ratio of the social optimum to the Nash Equilibrium reward.

    Quantifies how costly collective defection is.  A high PoA means there is
    a large gap between what agents could earn cooperating vs what they earn
    when everyone defects — the dilemma is severe.

    Parameters
    ----------
    social_optimum_reward : mean total reward of a fully cooperative population
                            (e.g. mean total_apples_harvested across no_zap episodes)
    nash_approx_reward    : mean total reward when defection dominates
                            (e.g. mean total_apples_harvested at high rogue ratio)

    Returns
    -------
    float >= 1  (1 = no cost to defection, higher = more severe dilemma)
    """
    if nash_approx_reward <= 0:
        return float('inf')
    return float(social_optimum_reward / nash_approx_reward)


def evolutionary_stability(focal_per_capita: np.ndarray,
                            background_per_capita: np.ndarray) -> float:
    """Mean ratio of cooperator return to rogue return across episodes.

    An Evolutionarily Stable Strategy (ESS) cannot be invaded by a mutant
    playing a different strategy.  In the rogue sweep, the cooperator strategy
    is ESS when a single rogue earns less than the cooperators it is paired with.

    Returns
    -------
    float
        > 1  cooperator earns more → cooperator strategy is ESS at this rogue ratio
        = 1  equal fitness → neutral stability
        < 1  rogue earns more → cooperator strategy is NOT ESS, vulnerable to invasion

    Note: compute separately for each rogue ratio (num_background_agents value).
    """
    ratios = []
    for f, b in zip(focal_per_capita, background_per_capita):
        if b > 0:
            ratios.append(f / b)
    return float(np.mean(ratios)) if ratios else float('nan')


def cooperation_support(focal_per_capita: np.ndarray,
                         background_per_capita: np.ndarray) -> float:
    """Fraction of episodes where cooperators out-earn rogues.

    A robust exclusion mechanism should consistently prevent rogues from
    free-riding.  High cooperation_support means the mechanism works
    episode-to-episode, not just on average.

    Returns
    -------
    float in [0, 1]
        1.0  cooperators always out-earn rogues (perfect exclusion)
        0.5  random — no systematic advantage to either strategy
        0.0  rogues always win (complete free-rider exploitation)
    """
    return float(np.mean(np.array(focal_per_capita) > np.array(background_per_capita)))


def correlated_equilibrium_score(turn_taking_index: np.ndarray,
                                  sustainability_index: np.ndarray) -> float:
    """Proxy for whether agents are playing a correlated equilibrium.

    A correlated equilibrium exists when agents condition their strategy on
    shared observable signals.  Turn-taking (alternating patch access) combined
    with sustained resource levels is the signature of a correlated equilibrium
    in a commons dilemma.

    Returns
    -------
    float in [-1, 1]
        Positive  = turn-taking AND sustainability together (correlated equilibrium signal)
        Near zero = one without the other (turn-taking without sustainability, or vice versa)
        Negative  = neither pattern present
    """
    tti = np.array(turn_taking_index)
    si  = np.array(sustainability_index)
    if len(tti) < 2 or tti.std() == 0 or si.std() == 0:
        return 0.0
    # Pearson correlation between turn-taking strength and sustainability
    # across episodes: do episodes with more turn-taking also sustain the resource?
    return float(np.corrcoef(tti, si)[0, 1])


def folk_theorem_index(step_rewards: list,
                        window_fraction: float = 0.1) -> float:
    """Estimate of whether LSTM memory sustains conditional cooperation.

    The Folk Theorem states that in repeated games with sufficiently patient
    players, cooperation can be an equilibrium via conditional strategies
    (tit-for-tat, grim trigger).  The LSTM implements this memory.

    Proxy: if early-episode reward is low (agents start cautious / exploring)
    and late-episode reward is stable or higher, the LSTM learned to condition
    on episode history and sustain cooperation.  If reward monotonically
    declines, the LSTM is not helping — agents defect regardless of history.

    Returns
    -------
    float
        > 0  reward improves or stabilises over episode (LSTM contributing)
        = 0  flat reward (no temporal structure)
        < 0  reward declines monotonically (defection / patch collapse regardless of memory)
    """
    if len(step_rewards) < 10:
        return 0.0
    w = max(1, int(len(step_rewards) * window_fraction))
    early = float(np.mean(step_rewards[:w]))
    late  = float(np.mean(step_rewards[-w:]))
    mid   = float(np.mean(step_rewards))
    if mid <= 0:
        return 0.0
    # Normalised: how much does the late reward exceed the early reward?
    return float((late - early) / mid)


# ---------------------------------------------------------------------------
# Composite: compute all game-theory metrics from episode-level arrays
# ---------------------------------------------------------------------------

def compute_game_theory_metrics(
    focal_per_capita:       np.ndarray,
    background_per_capita:  np.ndarray,
    total_rewards:          np.ndarray,
    sustainability_index:   np.ndarray,
    turn_taking_index:      Optional[np.ndarray] = None,
    max_harvest_rate:       float = 1.0,
    social_optimum_reward:  Optional[float] = None,
) -> dict:
    """Compute all game-theory metrics for a set of evaluation episodes.

    Parameters
    ----------
    focal_per_capita       : per-episode mean return of cooperator (focal) agents
    background_per_capita  : per-episode mean return of rogue (background) agents
    total_rewards          : per-episode total group reward (total_apples_harvested)
    sustainability_index   : per-episode sustainability_index from evaluation_metrics
    turn_taking_index      : per-episode turn_taking_index (optional, from spatiotemporal)
    max_harvest_rate       : theoretical max harvest rate (default 1.0 = 1 apple/agent/step)
    social_optimum_reward  : mean total reward of fully cooperative baseline.
                             If None, uses the 90th percentile of total_rewards as proxy.
    """
    focal_pc = np.array(focal_per_capita)
    bg_pc    = np.array(background_per_capita)
    rewards  = np.array(total_rewards)
    si       = np.array(sustainability_index)

    # Estimate social optimum if not provided
    if social_optimum_reward is None:
        social_optimum_reward = float(np.percentile(rewards, 90))

    # Nash approximation: 10th percentile of rewards (worst collective outcomes)
    nash_approx = float(np.percentile(rewards, 10))

    # Mean observed harvest rate proxy: total_reward / (n_agents * episode_steps)
    # Use sustainability_index < 1 episodes as "collapse" proxy for NE
    observed_harvest_rate = float(np.mean(rewards) / max(rewards.max(), 1))

    tti = turn_taking_index if turn_taking_index is not None else np.zeros(len(si))

    return {
        "nash_deviation":              nash_deviation(observed_harvest_rate, max_harvest_rate),
        "price_of_anarchy":            price_of_anarchy(social_optimum_reward, nash_approx),
        "evolutionary_stability":      evolutionary_stability(focal_pc, bg_pc),
        "cooperation_support":         cooperation_support(focal_pc, bg_pc),
        "correlated_equilibrium_score": correlated_equilibrium_score(np.array(tti), si),
        "folk_theorem_index":          float(np.mean([
            folk_theorem_index(list(si))  # uses sustainability trajectory as proxy
        ])),
        # Summary stats for quick comparison across experiments
        "mean_focal_per_capita":       float(focal_pc.mean()),
        "mean_background_per_capita":  float(bg_pc.mean()),
        "free_rider_advantage":        float(bg_pc.mean() / focal_pc.mean())
                                       if focal_pc.mean() > 0 else float('nan'),
        "social_optimum_reward":       social_optimum_reward,
        "nash_approx_reward":          nash_approx,
    }


# ---------------------------------------------------------------------------
# Rogue ratio sweep analysis
# ---------------------------------------------------------------------------

def analyse_rogue_sweep(results_by_ratio: dict) -> dict:
    """Analyse game-theory metrics across a sweep of rogue ratios.

    Parameters
    ----------
    results_by_ratio : dict mapping num_rogues (int) -> pd.DataFrame
                       Each DataFrame is the output CSV of a mixed eval run.

    Returns
    -------
    dict mapping num_rogues -> game_theory_metrics dict

    Example
    -------
    import pandas as pd
    results = {
        1: pd.read_csv("eval_results/rogue_1/results_evals.csv"),
        2: pd.read_csv("eval_results/rogue_2/results_evals.csv"),
        3: pd.read_csv("eval_results/rogue_3/results_evals.csv"),
        4: pd.read_csv("eval_results/rogue_4/results_evals.csv"),
    }
    sweep = analyse_rogue_sweep(results)
    for n_rogues, m in sweep.items():
        print(f"rogues={n_rogues}  ESS={m['evolutionary_stability']:.2f}  "
              f"PoA={m['price_of_anarchy']:.2f}  support={m['cooperation_support']:.2f}")

    # Find the critical ratio where cooperator strategy stops being ESS
    critical_ratio = next(
        (k for k, m in sorted(sweep.items()) if m["evolutionary_stability"] < 1.0),
        None
    )
    print(f"Critical rogue ratio (ESS breaks): {critical_ratio}")
    """
    summary = {}
    for n_rogues, df in sorted(results_by_ratio.items()):
        tti = df["turn_taking_index"].values if "turn_taking_index" in df.columns else None
        summary[n_rogues] = compute_game_theory_metrics(
            focal_per_capita=df["focal_per_capita_return"].values,
            background_per_capita=df["background_per_capita_return"].values,
            total_rewards=df["total_apples_harvested"].values,
            sustainability_index=df["sustainability_index"].values,
            turn_taking_index=tti,
        )
    return summary
