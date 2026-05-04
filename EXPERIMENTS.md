# Experiments Guide

This document describes all available substrates, what each one manipulates in the social dilemma structure, what experiments you can design, and what emergent behaviors to look for.

---

## Background: The Social Dilemma

All `commons_harvest` substrates instantiate a **tragedy of the commons**: a renewable resource (apples) that regrows based on local density. The fundamental tension is:

- **Cooperate**: harvest sparingly, allow apples to regrow, sustain the commons long-term.
- **Defect**: harvest as fast as possible, outcompete others, deplete the patch.

The Nash equilibrium under selfish play is mutual defection — everyone harvests as fast as possible — which collapses the resource and produces lower total reward than mutual restraint would. This mirrors real commons dilemmas (fisheries, forests, groundwater).

In multi-agent RL this is interesting because:
1. Cooperation is not programmed in — it must emerge from learning.
2. Agents face a **non-stationary environment**: the other agents' policies are also changing.
3. The dilemma has both a **within-episode** dimension (sustain the patch or not) and a **between-episode** dimension (what strategy to commit to at all).

---

## Substrate Reference

### `commons_harvest__open`
**The baseline.** Open map, 7 agents, full zapping enabled (beam length 3, cooldown 2 steps, 4-step respawn). Apple regrowth is density-dependent — a patch with no neighbors stops regrowing permanently.

**Dilemma framing**: Pure multi-player iterated commons dilemma with a punishment mechanism (zapping). Defection strategies: harvest all apples, zap competitors to reduce their collection rate. Cooperation strategies: restrain harvesting, avoid zapping.

**Key parameter**: Regrowth probabilities `[0, 0.0025, 0.005, 0.025]` — patches sustain themselves only if enough nearby apples remain.

---

### `commons_harvest__closed`
Physical walls divide the map into sub-regions. Agents are assigned to regions, limiting cross-region competition.

**Dilemma framing**: Partial enclosure. The commons dilemma is weakened because agents naturally have fewer competitors for their local patch. Tests whether physical partitioning of a commons induces sustainable behavior — analogous to private property without explicit property rights.

**Expected behavior**: Higher individual returns than open, more sustainable harvesting within regions, less zapping.

---

### `commons_harvest__partnership`
Agents are paired into small groups and evaluated together.

**Dilemma framing**: Shifts from group commons dilemma to dyadic (2-player) iterated prisoners' dilemma logic. With fewer agents sharing a patch, the feedback loop between your behavior and your partner's returns is tighter, which theoretically favors conditional cooperation (tit-for-tat style strategies).

---

### `commons_harvest__private_property`
Agents can use zapping to defend specific apple patches. Property rights are not formally enforced — they emerge from the ability to punish intruders.

**Dilemma framing**: Tests whether *de facto* private property rights emerge as a solution to the commons dilemma. An agent that successfully defends a patch now has individual incentive to sustain it. This converts the commons dilemma into an individual optimization problem — but only if property rights are actually enforced, which requires costly zapping.

**Emergent behavior to look for**: Territorial patrolling, boundary defense, patch ownership conventions.

---

### `commons_harvest__private_property_pc`
Partial-coverage private property — a variant with less complete territorial coverage. Intermediate between open commons and full private property.

---

### `commons_harvest__open_disable_zapping`
Open map but zapping is mechanically neutralized (cooldown=10,000, beam length=0). The action space still includes `fireZap` so the observation/action structure matches `commons_harvest__open` — but firing does nothing.

**Dilemma framing**: Removes the enforcement/punishment mechanism entirely. In social dilemma theory, punishment is considered essential for sustaining cooperation at scale. Without it, defectors should be uninhibited. This substrate tests: **does cooperation require the threat of punishment?**

**Key experiment**: Compare episode returns between `open` and `open_disable_zapping` agents. If zapping was actually used as a costly punishment to enforce norms, removing it should reduce average returns. If it was not being used cooperatively, returns may be similar.

**Note on action space**: The `fireZap` action is present but inert. This means policies trained on `open` can be evaluated here without retraining — the observation space is identical, only the effect of one action changes.

---

### `commons_harvest__open_abundance`
Regrowth probabilities are higher than the open baseline. The commons is easy to sustain.

**Dilemma framing**: Tests behavior under **resource abundance**. Behaviorally, when resources are plentiful the cost of cooperating is low and the benefit of defecting is low. We expect more sustainable harvesting, less zapping, and higher per-capita returns.

**Use as a control**: Pairing with `open_scarcity` gives a natural resource-pressure axis to study how scarcity affects strategy.

---

### `commons_harvest__open_scarcity`
Regrowth probabilities ~20x lower than the open baseline: `[0.0, 0.000050, 0.00010, 0.00050]`. Patches regenerate extremely slowly and can collapse irreversibly very quickly.

**Dilemma framing**: Tests behavior under **genuine resource pressure**. The defection payoff (eat fast before others do) is high, but patch collapse is rapid and catastrophic. This is the harshest version of the commons dilemma in this codebase.

**Emergent behaviors to look for**:
- Faster patch collapse than in open (likely)
- Agents learning to space out across patches rather than competing for the same apples
- Possible emergence of inter-patch migration strategies
- Breakdown of any cooperation sustained in the open setting

---

### `commons_harvest__farmer`
No zapping. Each agent has a **Farmer component** that gives +1 reward every 2 steps per visible apple within radius 2. The primary reward is still +1 for eating an apple.

**Dilemma framing**: Reward shaping that directly incentivizes **proximity to apple patches**. An agent near a dense patch earns continuous proximity reward even without eating. This creates an individual incentive to:
1. Stay near patches (farming behavior)
2. Keep patches alive — a depleted patch stops generating proximity reward
3. Prefer patch preservation over rapid extraction

This substrate tests whether intrinsic reward shaping (not punishment) can resolve the commons dilemma. The key difference from other variants: the incentive is **positive** (reward for proximity) rather than **negative** (punishment via zapping).

**Emergent behaviors to look for**:
- Agents clustering near large patches rather than roaming
- Voluntary harvest restraint (the proximity reward offsets the opportunity cost of not eating)
- Emergence of "patch ownership" without any enforcement mechanism
- Stable patches surviving longer than in open

---

### `commons_harvest__farmer_zap`
Farmer variant **with zapping re-enabled** and proximity reward reduced to 0.01 (shaping signal only, not primary reward).

**Dilemma framing**: Combines farming incentives with the enforcement mechanism. This introduces a new strategic layer: does an agent benefit more from staying near a patch (farming) or from zapping competitors (enforcement)? Zapping a competitor away from your patch removes competition for both the apples and the proximity reward.

**Key research question**: Does the farming incentive survive when aggression is possible? Or do agents learn to zap-and-farm?

**Emergent behaviors to look for**:
- Mixed strategies: some agents farm, others zap
- Patch guardians that farm and defend
- Free-rider dynamics: farm near a patch guarded by another agent

---

## Experiment Designs

### 1. Dilemma Severity Axis
Train the same PPO policy across `open_abundance`, `open`, `open_scarcity`. Compare:
- Sustainability (how many steps before patches collapse)
- Per-capita returns
- Zapping rates

```bash
python baselines/train/run_ray_train.py --exp commons_harvest__open_abundance --no-tune
python baselines/train/run_ray_train.py --exp commons_harvest__open --no-tune
python baselines/train/run_ray_train.py --exp commons_harvest__open_scarcity --no-tune
```

### 2. Punishment vs. No Punishment
Train on `open` and `open_disable_zapping`. Cross-evaluate both trained populations on both substrates using the mixed evaluation.

```bash
# Train open agents
python baselines/train/run_ray_train.py --exp commons_harvest__open --no-tune
# Train no-zap agents
python baselines/train/run_ray_train.py --exp commons_harvest__open_disable_zapping --no-tune
# Cross-evaluate: open agents as focal, no-zap agents as background
python baselines/evaluation/evaluate.py \
  --config_dir results/torch/commons_harvest__open/... \
  --policies_dir results/torch/commons_harvest__open/.../policies \
  --background_policies_dir results/torch/commons_harvest__open_disable_zapping/.../policies \
  --num_background_agents 2 --num_episodes 10
```

### 3. Reward Shaping vs. Punishment
Compare `farmer` (positive shaping, no zap) vs. `open` (punishment mechanism). Both are trained independently, then cross-evaluated as focal/background. This isolates whether the mechanism of cooperation (incentive vs. punishment) affects cross-population performance.

### 4. Farmer Robustness
Train farmer agents, then evaluate as focal against `open` background agents who can zap. Do they maintain higher per-capita returns or collapse under aggressive opponents?

```bash
python baselines/evaluation/evaluate.py \
  --config_dir results/torch/commons_harvest__farmer/... \
  --policies_dir results/torch/commons_harvest__farmer/.../policies \
  --background_policies_dir results/torch/commons_harvest__open/.../policies \
  --num_background_agents 2 --num_episodes 10
```

### 5. Generalization Test
Train on `open`, evaluate (zero-shot) on `open_scarcity`. Does the policy generalize to a harder resource environment? How much does per-capita return degrade?

---

## Generalization in Multi-Agent Settings

Generalization in single-agent RL means: does a policy trained on one environment transfer to a new one? In multi-agent RL this gets more complex because:

**1. Co-adaptation**: Agents train against each other, not against a fixed environment. A policy that works well against co-trained partners may fail against unseen partners with different strategies.

**2. Strategy diversity**: Trained populations converge to a specific strategy mix. When you evaluate against a different population (the background population in `run_mixed_evaluation`), you're testing out-of-distribution partner behavior. A policy that exploited its training partners' specific tendencies will generalize poorly.

**3. Substrate generalization**: Moving from `open` to `open_scarcity` changes resource dynamics without changing the action/observation space. A policy that learned "harvest fast" will do well early in an `open` episode but catastrophically in `open_scarcity` where patches collapse permanently.

**4. Role generalization**: The Melting Pot 2.0 evaluation protocol (focal vs. background) specifically tests this. Focal agents are evaluated against DeepMind bots with pre-defined strategies (free agents, pacifists). A policy that only works with co-trained partners fails this test.

The mixed evaluation added in this repo (`--background_policies_dir`) is a middle ground: evaluate against another *learned* policy rather than a hand-coded bot. This lets you test whether cooperation between two learned populations is robust even when those populations were trained independently.

---

## Emergent Behaviors Reference

| Behavior | What it looks like | Which substrates | Significance |
|---|---|---|---|
| Sustainable harvesting | Agents avoid eating the last apple in a patch | open, farmer, scarcity | Cooperative solution to commons dilemma |
| Patch collapse | All apples in a region permanently depleted | scarcity, open | Defection failure mode |
| Punitive zapping | Agent zaps another after being harvested near | open, private_property | Costly enforcement of norms |
| Territorial patrolling | Agent repeatedly traverses the perimeter of a patch | private_property, farmer_zap | De facto property rights |
| Patch farming | Agent hovers near a dense apple patch without always eating | farmer, farmer_zap | Proximity reward exploitation |
| Free-riding | Agent follows others to harvest regrown apples without guarding | open, partnership | Exploitation of cooperators |
| Retaliatory zapping | Agent zaps after being zapped, not after being harvested | open | Tit-for-tat-like punishment |
| Migration | Agents shift between patches as local density drops | open, scarcity | Adaptive resource management |
| Coalition avoidance | Agents spread across map to minimize overlap | scarcity | Individual response to resource pressure |

---

## Priority Experiments (Next Steps)

The following three experiments are ranked by expected impact based on prior project findings
(Parreño et al., 2024) and the current state of the codebase.

### 1. `commons_harvest__farmer` as focal vs `commons_harvest__open` as background

**The most important open question from an AI Safety standpoint.**

`commons_harvest__farmer` is already the farmer-no-zap substrate: agents have the proximity
reward shaping incentive (`rewardForEdibleVisible = 0.1`) but no access to the zap action.
The central safety question this experiment addresses is:

> Can an agent that was never trained to punish, but has a positive sustainability incentive,
> survive and remain competitive when placed against agents that can zap?

This tests action-space asymmetry directly: the background agents can exploit the focal agents
via zapping, while the focal agents can only respond by harvesting efficiently. If farmer agents
maintain competitive returns despite this asymmetry, it is evidence that positive reward shaping
can substitute for punishment as a cooperation mechanism — a qualitatively different and safer
alignment strategy.

Run with the mixed evaluation infrastructure already in place:

```bash
python baselines/evaluation/evaluate.py \
  --config_dir results/torch/commons_harvest__farmer/... \
  --policies_dir results/torch/commons_harvest__farmer/.../policies \
  --background_policies_dir results/torch/commons_harvest__open/.../policies \
  --num_background_agents 2 --num_episodes 20
```

The per-agent zap and apple tracking columns in the output CSV directly measure the asymmetry.

---

### 2. `commons_harvest__open_disable_zapping` focal vs `commons_harvest__open` background

**The punishment asymmetry experiment — infrastructure is ready, agents need retraining.**

Prior results showed `no_zap` was the best-performing trained agent (~10% below the open
baseline), with the lowest inequality and fewest depleted fields. However, the evaluation at
the time could not measure what happened when these pacifist agents were placed against
background agents that *can* zap.

The question: **is a no_zap agent exploited, or does its sustainable harvesting strategy
produce competitive returns even under aggression?**

This is directly relevant to AI Safety: it tests whether restricting a harmful action from
an agent's action space makes it vulnerable in a mixed population — which would argue against
action-space constraints as a safety mechanism — or whether it remains robust.

---

### 3. Retrain `commons_harvest__farmer` with corrected reward

**The most immediately actionable experiment.**

The original farmer results were inconclusive because `rewardForEdibleVisible = 0.01` was too
small relative to the `+1` apple-eating reward to produce a meaningful learning signal. The
substrate now uses `rewardForEdibleVisible = 0.1` (one order of magnitude increase, as
recommended in the project write-up). The hypothesis is that agents will learn to hover near
apple patches rather than deplete them, producing qualitatively different and more sustainable
harvesting behaviour.

This is the baseline that experiments 1 and 2 depend on — without a properly trained farmer
agent, the cross-population evaluation has no signal.
