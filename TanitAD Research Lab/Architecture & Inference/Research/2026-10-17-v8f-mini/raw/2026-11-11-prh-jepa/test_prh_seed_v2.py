"""Tests for prh_seed_v2 metrics and rules on planted data. PRH_SRC_DIR selects the directory of the code under test (mutation runner)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PRH_SRC_DIR", os.path.dirname(os.path.abspath(__file__))))
import prh_seed_v2 as V  # noqa: E402

N, D, NC = 1200, 24, 30


def _ctx(seed=0, B=40):
    cid = np.repeat(np.arange(NC), N // NC)
    fit_clip = np.arange(NC) % 2 == 0
    is_fit = fit_clip[cid]
    fi, si = np.where(is_fit)[0], np.where(~is_fit)[0]
    sc = np.where(~fit_clip)[0]
    remap = -np.ones(NC, int)
    remap[sc] = np.arange(len(sc))
    nC = len(sc)
    counts = np.random.default_rng(seed).multinomial(nC, np.ones(nC) / nC, size=B).astype(float)
    return {"fit_idx": fi, "score_idx": si, "cid_score": remap[cid[si]], "nC": nC, "counts": counts}


def test_proc_one_on_identical():
    A = np.random.default_rng(0).standard_normal((N, D))
    assert abs(V.metric("PROC", A, A, _ctx())[0] - 1) < 1e-9


def test_proc_lower_for_random_mix():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((N, D))
    Bm = 0.5 * A + 0.5 * rng.standard_normal((N, D))
    c = _ctx()
    assert V.metric("PROC", A, Bm, c)[0] < V.metric("PROC", A, A, c)[0] - 0.2
    # a random rotation-free (non-orthogonal) mix is lower than a pure rotation, which PROC must score as 1
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    assert V.metric("PROC", A, A @ Q, c)[0] > 0.99
    # whitened Procrustes is invariant to any invertible linear map (whitening removes it), so it stays high there
    assert V.metric("PROC", A, A @ rng.standard_normal((D, D)), c)[0] > 0.9


def test_cka_one_on_identical():
    A = np.random.default_rng(2).standard_normal((N, D))
    assert abs(V.metric("CKA", A, A, _ctx())[0] - 1) < 1e-9


def test_cc_one_on_identical_and_boot_shape():
    A = np.random.default_rng(3).standard_normal((N, D))
    pt, boot = V.metric("CC8", A, A, _ctx())
    assert abs(pt - 1) < 1e-9 and boot.shape == (40,)


def _good(**kw):
    v = {"SELF": 1.0, "FLOOR": 0.2, "MIX25": 0.9, "MIX50": 0.6, "MIX75": 0.3, "SHUF": 0.0, "diff50_ci": [0.2, 0.5]}
    v.update(kw)
    return v


def test_admissibility_rule_each_clause():
    assert V.admissible(_good())[1]
    assert not V.admissible(_good(SELF=0.99))[1]
    assert not V.admissible(_good(MIX50=0.95))[1]                       # not monotone
    assert not V.admissible(_good(diff50_ci=[-0.1, 0.5]))[1]            # not separated
    assert not V.admissible(_good(SHUF=0.3))[1]                         # shuffle above floor


def test_choice_rule_planted():
    t = {"CC8": {"admissible": True, "ratio": 5.0}, "CC32": {"admissible": True, "ratio": 9.0}, "PROC": {"admissible": True, "ratio": 2.0}, "CKA": {"admissible": True, "ratio": 20.0}}
    assert V.choose(t) == "CKA"
    t["CKA"]["admissible"] = False
    assert V.choose(t) == "CC32"                                          # PROC not within 10 %
    t["PROC"]["ratio"] = 8.5
    assert V.choose(t) == "PROC"                                          # tie within 10 % goes to PROC
    t["PROC"]["admissible"] = False
    assert V.choose(t) == "CC32"
    assert V.choose({"CC8": {"admissible": False, "ratio": 1.0}}) is None
