"""Control k2 of PREREG_V8F_PROOF_ADD2: a stack whose only input is the learned prediction with coefficient fixed at 1 reproduces the learned error exactly."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmceil as W
import wmstack as S


def test_control_k2_fixed_coefficient_stack_reproduces_learned_exactly():
    rng = np.random.default_rng(5)
    z0 = rng.standard_normal((500, 7))
    zk = z0 + rng.standard_normal((500, 7))
    lp = z0 + 0.3 * rng.standard_normal((500, 7))
    assert np.abs(W.sqerr(S.fixed_coef_stack_pred(z0, lp, 1.0), zk) - W.sqerr(lp, zk)).max() < 1e-12
    assert np.abs(W.sqerr(S.fixed_coef_stack_pred(z0, lp, 0.0), zk) - W.sqerr(z0, zk)).max() == 0.0   # coefficient 0 = persistence
