"""S1b pure rules (numpy only, no nuplan / navsim import): the frozen filter, the CV agent plan, S1's seeded permutation.  Unit-tested in test_s1b_rules.py."""
from __future__ import annotations

import hashlib

import numpy as np

CV_MAX, HUMAN_MIN = 0.8, 0.8           # navtrain: REMOVE if PDMS(CV) > 0.8; REMOVE if PDMS(human) < 0.8   (Dauner et al. 2024 sec. 3.1; NAVSIM_PROTOCOL.md:386-391)
NUM_CV_POSES, CV_DT = 8, 0.5           # devkit constant_velocity_agent.py:15 TrajectorySampling(time_horizon=4, interval_length=0.5) -> 8 poses


def keep_rule(cv: float, human: float) -> bool:
    """literal reading: 'exceeds 0.8' removes (so CV == 0.8 stays); 'less than 0.8' removes (so human == 0.8 stays)."""
    return bool(cv <= CV_MAX and human >= HUMAN_MIN)


def cv_plan(speed: float) -> np.ndarray:
    """ConstantVelocityAgent.compute_trajectory (devkit navsim/agents/constant_velocity_agent.py:41-44): [[(i+1)*dt*v, 0, 0]], float32."""
    return np.array([[(i + 1) * CV_DT * speed, 0.0, 0.0] for i in range(NUM_CV_POSES)], dtype=np.float32)


def perm_for(seed: int, log: str, n: int) -> np.ndarray:
    """S1's apply_U_and_fetch.py permutation: default_rng([seed, int(sha256(log)[:8], 16)]).permutation(n)."""
    r = np.random.default_rng([seed, int(hashlib.sha256(log.encode()).hexdigest()[:8], 16)])
    return r.permutation(n)
