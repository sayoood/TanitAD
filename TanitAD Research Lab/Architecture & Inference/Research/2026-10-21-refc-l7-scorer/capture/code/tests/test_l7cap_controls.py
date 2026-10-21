"""Controls (a)/(b)/(c) of the L7 capture on a SYNTHETIC dump whose answers are literals.

Synthetic world: 3 windows, 149 candidates, 8 slots. GT is a straight line. Candidate k is the GT shifted by 0.1*k m in x at EVERY slot, so its
2 s ADE is exactly 0.1*k m. The roll "emitted" candidate 7 on every window (sel_idx = 7), so the own-pick ADE is the literal 0.7 m.

    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' python tests/test_l7cap_controls.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import controls_l7cap as K  # noqa: E402
import fan_roll_r8_l7cap as C  # noqa: E402
import l7cap_hooks as H  # noqa: E402

W, N, NB, SEL = 3, 149, 117, 7


def _world(tmp: str, mutation: str = ""):
    """Write a roll-like dump (ep*.npz + refcv6_extras.npz) and call the REAL `write_banks` to produce fan_extras.npz."""
    H.MUTATION = mutation
    C.ROWS.clear()
    C.LABS.clear()
    ws = np.asarray([7, 12, 17], np.int64)
    slot_x = np.asarray([2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0])
    gt8 = np.stack([slot_x, np.zeros(8)], -1)                                  # [8,2]
    for i in range(W):
        fan = np.repeat(gt8[None], N, 0).copy()
        fan[:, :, 0] += (0.1 * np.arange(N))[:, None]
        row = {"anchor_traj": fan.astype(np.float32), "anchor_logits": np.zeros(NB, np.float32), "refined_logits": np.zeros(N, np.float32),
               "sel_score": np.zeros(N, np.float32), "sel_score_v3": np.zeros(N, np.float32), "traj": fan[SEL].astype(np.float32),
               "traj_base": fan[SEL].astype(np.float32), "goal_dist": np.zeros(N, np.float32), "r8_sub_logits": np.zeros((N, 4), np.float32),
               "reach_keep": np.ones(N, bool), "ceil_keep": np.ones(N, bool), "r8_emit_keep": np.ones(N, bool),
               "sel_idx": np.asarray(SEL), "sel_idx_base": np.asarray(SEL), "r8_cand_src": np.arange(N), "e9_keep": np.ones(N, bool),
               "e9_argmax": np.asarray(SEL), "v0": 9.0, "r8_n_base": NB,
               "r7_bev_tokens": np.zeros((5, 4), np.float32), "r7_bev_pad": np.zeros(5, bool),
               "r7_agent_tokens": np.zeros((3, 4), np.float32), "r7_agent_pad": np.zeros(3, bool)}
        C.ROWS[(0, int(ws[i]) - 7)] = row
    g = np.repeat(gt8[None, :4], W, 0).astype(np.float32)                      # [W,4,2] GT at the 2 s grid
    osp = np.stack([C.ROWS[k]["traj"][:4] for k in sorted(C.ROWS)]).astype(np.float32)
    d = Path(tmp)
    np.savez(d / "ep000.npz", g=g, v0=np.full(W, 9.0, np.float32), os=osp, ws=ws, eid=np.asarray([123]), clip_index=np.asarray([0]))
    side = {"ws": ws, "clip_index": np.zeros(W, np.int64), "nav_true.traj": np.stack([C.ROWS[k]["traj"] for k in sorted(C.ROWS)])}
    np.savez(d / "refcv6_extras.npz", **side)
    with np.load(d / "refcv6_extras.npz") as side:
        C.write_banks(d, side, 1, NB)
    H.MUTATION = ""
    return d


def test_machinery_reads_the_literal():
    with tempfile.TemporaryDirectory() as t:
        d = _world(t)
        r = K.machinery(str(d))
        assert r["PASS"], r
        assert abs(r["mean_own_pick_ade_s2_m"] - 0.7) < 1e-5, r["mean_own_pick_ade_s2_m"]          # literal 0.1 * 7
        assert r["builder_vs_fan_recompute_max_abs_m"] <= 1e-6 and r["builder_vs_emitted_plan_recompute_max_abs_m"] <= 1e-6
        with np.load(d / "fan_extras.npz") as z:
            assert z["is_alloc"].tolist() == [False] * 117 + [True] * 32 and z["anchor_traj"].shape == (3, 149, 8, 2)
            assert z["inference_seed"].tolist() == [1]
            assert not (set(z.files) & H.FORBIDDEN_BANK_KEYS)
        with np.load(d / "l7cap_scene.npz") as s:
            assert set(s.files) == {"clip_index", "ws", "r7_bev_tokens", "r7_bev_pad", "r7_agent_tokens", "r7_agent_pad"}   # no cond, no nav


def test_wrong_candidate_index_goes_red():
    """Control (c): the writer dumps the WRONG candidate index (sel_idx + 1). Machinery must be RED, on the emitted-plan recompute (0.8 vs 0.7)."""
    with tempfile.TemporaryDirectory() as t:
        d = _world(t, mutation="wrong_sel_idx")
        with np.load(d / "fan_extras.npz") as z:
            assert z["sel_idx"].tolist() == [SEL + 1] * W, "the mutation did not take effect"
        r = K.machinery(str(d))
        assert not r["PASS"], "the wrong-index mutant passed the machinery control"
        assert abs(r["builder_vs_emitted_plan_recompute_max_abs_m"] - 0.1) < 1e-5, r["builder_vs_emitted_plan_recompute_max_abs_m"]   # 0.8 - 0.7


def test_k1_empty_comparison_is_not_a_pass():
    """The empty-string-blob hole: two empty (or unreadable) dump dirs compare 'equal'. K1 must read INCONCLUSIVE (PASS False)."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        assert not K.k1_identity(a, [b])["PASS"], "an empty comparison passed K1"
        np.savez(Path(a) / "ep000.npz", os=np.arange(3))
        assert not K.k1_identity(a, [b])["PASS"], "a one-sided array set passed K1"


def test_k1_rule_both_directions():
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2, tempfile.TemporaryDirectory() as t3:
        for d in (t1, t2, t3):
            np.savez(Path(d) / "ep000.npz", os=np.arange(8, dtype=np.float32), ws=np.arange(3))
        assert K.k1_identity(t1, [t2])["PASS"]                                  # bit-identical
        np.savez(Path(t3) / "ep000.npz", os=np.arange(8, dtype=np.float32) + np.float32(6e-5), ws=np.arange(3))
        r = K.k1_identity(t1, [t2, t3])                                         # one unwrapped identical, the other within the envelope
        assert r["PASS"] and abs(r["unwrapped_vs_unwrapped_envelope_m"] - 6e-5) < 1e-6
        np.savez(Path(t3) / "ep000.npz", os=np.arange(8, dtype=np.float32), ws=np.arange(3) + 1)    # an INTEGER array (a pick) differs
        assert not K.k1_identity(t1, [t2, t3])["PASS"], "a differing pick must fail K1"
        np.savez(Path(t3) / "ep000.npz", os=np.arange(8, dtype=np.float32) + np.float32(1e-3), ws=np.arange(3))
        assert not K.k1_identity(t3, [t1])["PASS"], "a float difference with no unwrapped envelope must fail (wrapped != unwrapped)"


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("PASS", name)
    print("ZZL7CCONTROLSPASS", n)
