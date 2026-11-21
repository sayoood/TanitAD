"""P6 / D6 -- the navigation COMMAND the student is conditioned on (SPEC_P6_PAPER_FAITHFUL.md section 2, PI ruling 2026-10-11 P6-N).

The paper's student takes a discrete navigation command, not goal points (DriveZero p. 8: "the ego kinematics and the command are embedded into one
ego token"; the command is "derived from the teacher's goal point", ref [56] = OpenScene). At test NAVSIM supplies `driving_command`, a one-hot over
(LEFT, STRAIGHT, RIGHT, UNKNOWN) -- the order MEASURED in planner.py:184 on the 1,123 selection tokens.

WHAT IS AND IS NOT KNOWN (evidence classes)
* UNVERIFIED: OpenScene's exact goal->command rule. Its source is not in the Library and was not read.
* MEASURED (2026-10-11, `raw/2026-10-11-p6-impl/command_rule_measure.json`, 12,146 navtest tokens, NAVSIM `driving_command` vs the planner's goal census
  `raw/2026-10-01-goal-trigger/goal_census_navtest_full.json`): the best single-threshold rule on the NEAR goal point's bearing reaches 0.9014 on log-disjoint
  held-out halves (threshold 0.245 rad); the far point 0.80; a depth-8 decision tree on all four goal numbers + speed + ego-to-route 0.885. ==> NO goal-derived
  rule tried reaches SPEC_P6 D6's 95 % bar. The rule below is therefore the FALLBACK, and the trainer reports how many commands came from where.
* A row may carry NAVSIM's own command in `row["command"]` (an int 0..3, written by `refe/annotate_commands.py` from the navtrain scene metadata on the
  pod). That value wins; the goal rule only fills rows without one (in practice the goal-augmented twins, whose command must follow their OWN goal).
"""
from __future__ import annotations

import math

COMMANDS = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")
LEFT, STRAIGHT, RIGHT, UNKNOWN = 0, 1, 2, 3
N_COMMANDS = len(COMMANDS)

# bearing of the NEAR goal point (the ego frame is x forward, y LEFT -- planner.py:184 measured +y = LEFT). MEASURED threshold, see the module docstring.
GOAL_RULE_BEARING_RAD = 0.245


def goal_to_command(goal) -> int:
    """The fallback rule: the bearing of the first goal point decides. `goal` = [x1, y1, x2, y2] (or just [x1, y1])."""
    x, y = float(goal[0]), float(goal[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return UNKNOWN
    ang = math.atan2(y, max(x, 1e-6))
    if ang > GOAL_RULE_BEARING_RAD:
        return LEFT
    if ang < -GOAL_RULE_BEARING_RAD:
        return RIGHT
    return STRAIGHT


def command_of_row(row: dict) -> int:
    """NAVSIM's own command if the row carries one (annotate_commands.py), else the goal rule."""
    c = row.get("command")
    if c is not None:
        c = int(c)
        if not 0 <= c < N_COMMANDS:
            raise ValueError(f"row command {c} outside 0..{N_COMMANDS - 1}")
        return c
    return goal_to_command(row["goal"])


def command_source(row: dict) -> str:
    return "row" if row.get("command") is not None else "goal_rule"


def onehot(idx: int) -> list:
    v = [0.0] * N_COMMANDS
    v[int(idx)] = 1.0
    return v


def from_driving_command(one_hot) -> int:
    """NAVSIM's `driving_command` one-hot -> index. An all-zero vector is UNKNOWN (planner._command_name's rule)."""
    v = list(one_hot)
    if len(v) != N_COMMANDS or max(v) <= 0:
        return UNKNOWN
    return max(range(N_COMMANDS), key=lambda i: v[i])
