"""``tanitad.rl`` — RL post-training for the refcv3 planner.

PI commission **D-RL-REFCV3**: *"develop a library of RL methods to post-train
the planner of refcv3 by RL, after doing a research about the best ways and
approaches; it should do the implementation and validate it, to be prepared for
training."*

Published lever: **DiffusionDriveV2** (arXiv 2512.07745) — GRPO RL post-training
from an IL cold start, +3.1 PDMS at fixed ResNet-34 (88.1 -> 91.2), and the raw
top-10 fan floor held at 84.4 where DiffusionDrive fell to 75.3.

⛔ THE TWO RULES THE WHOLE LIBRARY IS BUILT AROUND
--------------------------------------------------
1. **The reward is never the selector's own score.** Structurally enforced by
   ``audit.assert_selector_disjoint`` against ``rewards.FORBIDDEN_REWARD_INPUTS``.
   DDv2's own named defect is selector over-reliance; our SEL-1 winner's-curse
   refusal measured the same thing independently.
2. **The guard must be able to fail.** ``rewards.HACKABLE_WEIGHTS`` (progress
   only) ships as the deliberate-regression arm and ``audit.audit_reward`` MUST
   flag it. An audit that passes it is broken, and every PASS it gave is void.

SCOPE NOTE (read before concluding this contradicts METHOD_LIBRARY.md §2)
-------------------------------------------------------------------------
That document concluded GRPO/RLOO is *dominated* for our stack. It is — **at the
SELECTOR**, whose fan is enumerable and whose metric objective is differentiable
end-to-end, so sampling only adds variance. This library targets the **diffusion
decoder's continuous sampled offsets under a rule-based, non-differentiable
reward**, where the score-function estimator is the only available route. Both
statements are true because they are about different sites; see
``advantage.py``'s docstring for the side-by-side.

Modules
-------
``rewards``    bounded rule-based components, each declaring its own degenerate policy
``advantage``  intra-anchor GRPO + inter-anchor truncated advantage (critic-free)
``audit``      selector-disjointness, the degenerate-policy panel, coverage honesty
``config``     every knob on one frozen dataclass; refuses unsupplied methods
``posttrain``  the loop: frozen trunk, asserted counters, done-marker

Tier: everything here is **T0 training-side**. A capability claim needs T1
(``Project Steering/EVAL_DOCTRINE.md``).
"""

from .advantage import (composite_advantage, grpo_advantage,
                        policy_gradient_loss,
                        truncated_inter_anchor_advantage)
from .audit import (AuditReport, assert_selector_disjoint, audit_reward,
                    degenerate_panel, reference_policy,
                    report_component_coverage)
from .config import METHODS, REQUIREMENTS, PostTrainConfig
from .rewards import (COMPONENTS, DEFAULT_WEIGHTS, FORBIDDEN_REWARD_INPUTS,
                      HACKABLE_WEIGHTS, Kinematics, RewardComponent,
                      RewardSpec, kinematics)
from .posttrain import (SmokeCounters, apply_exploration_noise, rl_objective,
                        run_posttrain, select_trainable)

__all__ = [
    "COMPONENTS", "DEFAULT_WEIGHTS", "FORBIDDEN_REWARD_INPUTS",
    "HACKABLE_WEIGHTS", "Kinematics", "RewardComponent", "RewardSpec",
    "kinematics",
    "composite_advantage", "grpo_advantage", "policy_gradient_loss",
    "truncated_inter_anchor_advantage",
    "AuditReport", "assert_selector_disjoint", "audit_reward",
    "degenerate_panel", "reference_policy", "report_component_coverage",
    "METHODS", "REQUIREMENTS", "PostTrainConfig",
    "SmokeCounters", "apply_exploration_noise", "rl_objective",
    "run_posttrain", "select_trainable",
]
