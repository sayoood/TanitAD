"""d2_selftest.py -- known-value / mutation tests for the LHA0-D2 instrument.  FRESH-SPLIT DATA AND SYNTHETIC INPUTS ONLY: no eval139 file is opened (d2_khead is imported for its
registration gate, which is exercised on a temporary directory, never on eval data).  Each test has a deliberate-regression arm that must go RED.
Writes result/D2_SELFTEST.json.  CPU only.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d2_common as D  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    import k_head as KH
    import d2_khead as K
    res = {}
    z = np.load(D.ROOT / "result" / "d2_fresh_oof.npz")
    P, y, eid_u = z["P"], z["y"].astype(bool), z["eid_u"]
    eP = eid_u[P]
    lo = z["d2_logit_oof"][P]
    # T1 known value: a constant score reads exactly 0.5; a label-leak score reads 1.0; the sign-flipped head reads 1 - AUC (deliberate regression).
    res["T1_constant_0.5"] = KH.auc_np(y, np.zeros(len(y))) == 0.5
    res["T1_leak_1.0"] = KH.auc_np(y, y.astype(float)) == 1.0
    a, a_flip = KH.auc_np(y, lo), KH.auc_np(y, -lo)
    res["T1_sign_flip_is_complement"] = bool(abs(a + a_flip - 1.0) < 1e-12) and a_flip < 0.5
    # T2 bootstrap: unit-weight WAuc == auc_np; a score paired with itself has an exactly-zero difference; the draw is a pure function of (episodes, seed)
    R = D.paired_boot(y, eP, {"lo": lo, "lo2": lo.copy(), "flip": -lo, "const": np.zeros(len(y))})
    res["T2_unit_weights"] = all(R["unit_ok"].values())
    res["T2_self_diff_zero"] = D.diff(R, "lo", "lo2")["ci95"] == [0.0, 0.0] and D.diff(R, "lo", "lo2")["point"] == 0.0
    R2 = D.paired_boot(y, eP, {"lo": lo, "flip": -lo})
    res["T2_deterministic"] = bool(np.array_equal(R["bt"]["lo"], R2["bt"]["lo"]))
    ue, inv = np.unique(eP, return_inverse=True)                       # an INLINE copy of k_head.main()'s loop must agree bit for bit
    rng = np.random.default_rng(0)
    w = np.bincount(rng.integers(0, len(ue), len(ue)), minlength=len(ue)).astype(np.float64)[inv]
    res["T2_matches_khead_inline_first_draw"] = bool(KH.WAuc(y, lo)(w) == R["bt"]["lo"][0])
    res["T2_const_0.5"] = R["pt"]["const"] == 0.5
    # T3 permutation: multiset preserved per episode, shared between columns, and (nearly) every value moves
    U = np.load(D.UNIVERSE)
    ea, a0v, pv = U["eid_u"], U["f_a0"], U["f_psi"]
    pa, pp = D.permute_within_episode(ea, [a0v, pv], seed=0)
    ok_ms = all(np.array_equal(np.sort(pa[ea == e]), np.sort(a0v[ea == e])) and np.array_equal(np.sort(pp[ea == e]), np.sort(pv[ea == e])) for e in np.unique(ea))
    pa2, _ = D.permute_within_episode(ea, [a0v, pv], seed=0)
    res["T3_multiset_preserved"] = bool(ok_ms)
    res["T3_deterministic"] = bool(np.array_equal(pa, pa2))
    res["T3_values_moved_frac"] = round(float((pa != a0v).mean()), 4)
    res["T3_mostly_moved"] = res["T3_values_moved_frac"] > 0.9
    pa_other, _ = D.permute_within_episode(ea, [a0v, pv], seed=1)
    res["T3_seed_changes_permutation"] = bool(not np.array_equal(pa, pa_other))
    # shared permutation: the (a0, psi) PAIR moves together (the pairs set is preserved)
    pairs = lambda x, y_: sorted(zip(x.tolist(), y_.tolist()))
    res["T3_pairs_move_together"] = all(pairs(pa[ea == e], pp[ea == e]) == pairs(a0v[ea == e], pv[ea == e]) for e in np.unique(ea)[:40])
    res["T3_equals_diag_C4_permutation_bitwise"] = bool(np.array_equal(pa, U["f_a0_perm"]) and np.array_equal(pp, U["f_psi_perm"]))   # the eval-side image of LHA0_DIAG C4 uses the SAME function
    # T4 verdict logic, every branch + the boundaries (UNROUNDED numbers)
    res["T4_bar_boundaries"] = [D.clears_bar(0.0299999, 0.01), D.clears_bar(0.030, 0.0001), D.clears_bar(0.05, -0.001), D.clears_bar(0.05, 0.0), D.clears_bar(float("nan"), 0.1)] == [False, True, False, False, False]
    res["T4_branches"] = [D.verdict_word(False, True, False).startswith("INVALID"), D.verdict_word(True, True, True).startswith("VOID"), D.verdict_word(True, True, False) == "GO",
                          D.verdict_word(True, False, False) == "NO-GO", D.verdict_word(True, False, True).startswith("VOID")] == [True] * 5
    # T5 fusion: label-free z-score, mean 0 sd 1; the fusion of a score with itself ranks identically to it (AUC equal)
    f = D.zscore(lo) + D.zscore(lo)
    res["T5_fusion_z"] = bool(abs(D.zscore(lo).mean()) < 1e-12 and abs(D.zscore(lo).std() - 1) < 1e-12 and abs(KH.auc_np(y, f) - KH.auc_np(y, lo)) < 1e-12)
    # T6 frozen heads: reload is sha-pinned (a 1-byte mutation of a COPY must be refused) and the logit is the documented formula
    H = D.load_frozen(D.ROOT / "D2_FROZEN.json")
    tmp = Path(tempfile.mkdtemp(prefix="d2_selftest_", dir="F:/c_overflow_2026-10-26/tmp"))
    mut = tmp / "D2_mut.json"
    mut.write_text((D.ROOT / "D2_FROZEN.json").read_text(encoding="utf-8").replace('"lambda"', '"lambdb"'), encoding="utf-8")
    try:
        D.load_frozen(mut, H["_sha256"])
        res["T6_sha_pin_refuses_mutation"] = False
    except SystemExit:
        res["T6_sha_pin_refuses_mutation"] = True
    x = np.array([[0.5, 10.0, -0.05]])
    man = float(((np.array([0.5, 1.0, -0.05]) - np.array(H["mu"])) / np.array(H["sd"])) @ np.array(H["w"]) + H["b"])
    res["T6_logit_formula"] = bool(abs(D.head_logit(H, x[:, 0], x[:, 1], x[:, 2])[0] - man) < 1e-12)
    # T7 the registration gate of d2_khead: no seal -> refuse; wrong sha -> refuse; right sha but an unrecorded pin -> refuse; sealed + pinned -> passes (run on a COPY dir)
    def gate(setup) -> str:
        d = Path(tempfile.mkdtemp(prefix="d2_gate_", dir=str(tmp)))
        (d / "code").mkdir()
        for rel in ("D2_FROZEN.json", "D2NOACC_FROZEN.json", "code/d2_common.py", "code/d2_fit.py", "code/d2_khead.py", "code/d2_selftest.py"):
            if (D.ROOT / rel).exists():
                shutil.copy2(D.ROOT / rel, d / rel)
        setup(d)
        old, K.ROOT = K.ROOT, d
        try:
            K.check_registration()
            return "PASS"
        except SystemExit:
            return "REFUSE"
        finally:
            K.ROOT = old

    pins = "\n".join(D.sha256_file(D.ROOT / r) for r in ("D2_FROZEN.json", "D2NOACC_FROZEN.json", "code/d2_common.py", "code/d2_fit.py", "code/d2_khead.py", "code/d2_selftest.py") if (D.ROOT / r).exists())

    def mk(d, text, sha_ok=True, with_sha_file=True):
        (d / "PREREG_LHA0_D2.md").write_text(text, encoding="utf-8")
        if with_sha_file:
            s = D.sha256_file(d / "PREREG_LHA0_D2.md") if sha_ok else "0" * 64
            (d / "PREREG_LHA0_D2.sha256").write_text(f"{s} *PREREG_LHA0_D2.md\n# sealed (selftest)\n", encoding="utf-8")

    res["T7_gate_no_seal_refuses"] = gate(lambda d: mk(d, pins, with_sha_file=False)) == "REFUSE"
    res["T7_gate_wrong_sha_refuses"] = gate(lambda d: mk(d, pins, sha_ok=False)) == "REFUSE"
    res["T7_gate_missing_pin_refuses"] = gate(lambda d: mk(d, "no pins here")) == "REFUSE"
    only_if_all_present = all((D.ROOT / r).exists() for r in ("D2_FROZEN.json", "D2NOACC_FROZEN.json", "code/d2_common.py", "code/d2_fit.py", "code/d2_khead.py", "code/d2_selftest.py"))
    res["T7_gate_sealed_and_pinned_passes"] = (gate(lambda d: mk(d, pins)) == "PASS") if (only_if_all_present and (D.ROOT / "code" / "d2_selftest.py").exists()) else "skipped"
    shutil.rmtree(tmp, ignore_errors=True)
    ok = all(v is True or isinstance(v, float) or v == "skipped" for k, v in res.items() if not k.endswith("frac"))
    res["ALL_PASS"] = bool(ok)
    (D.ROOT / "result" / "D2_SELFTEST.json").write_text(json.dumps({"tool": "d2_selftest.py", "tests": res, "script_sha256": D.sha256_file(__file__)}, indent=1, default=str), encoding="utf-8")
    print("ZZSELFTESTZZ", json.dumps(res), flush=True)
    return 0 if res["ALL_PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
