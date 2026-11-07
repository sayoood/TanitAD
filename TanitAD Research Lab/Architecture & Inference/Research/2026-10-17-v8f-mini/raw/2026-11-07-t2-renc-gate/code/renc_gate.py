"""RENC gates M-RENC and G-KEEP (PREREG_T2_RENC.md section 4), with G-ID0 read from box 2. CPU only, dev box.

    python -I renc_gate.py --out-dir <pull of D:/t2v8f/renc/out> --gap48-tokens <GAP48 pull>/tokens --out M_RENC.json [--data <pull of the data meta files>]

For every scene (fold f = first hex digit of sha256(scene id)) at GAP48's ticks (every 10th), with T = the frozen encoder's tokens of the real frame (the GAP48 pull's `real`):
  before   cos(GAP48 render tokens, T)                     = the 0.840 the card names
  M-RENC   cos(E'_other(render), T)    median over held-out pairs: rise >= 0.05 AND level >= 0.90
  G-KEEP   cos(E'_other(real), T)      median over held-out pairs: >= 0.99
Pass needs BOTH. Otherwise RE-FAILS-MECH. Statistic: flattened uncentered float64 cosine (t2a_v8f_encoder.cosine), pooled median; a scene-cluster bootstrap (B 2000, seed 0) is reported beside it and decides nothing.
Deliberate-regression arms (all run in every gate): M-RENC: every scene scored through the encoder of its OWN fold must be flagged by provenance (the scene is in that encoder's train list); its naive numbers are written beside.
G-KEEP: the pairing shifted by one position inside each scene must FAIL 0.99. G-ID0: box 2 ran B = 0 (bit-equal on 100 frames) and a perturbed B (must differ); read from G_ID0.json.
Verdict words: M-RENC-PASS | RE-FAILS-MECH | RENC-INVALID-LEAK | RENC-INVALID-FOLDS | GATE-BROKEN (a regression arm was not flagged, or G-ID0 failed). Exit: 0 pass, 5 fails mechanism, 6 invalid or broken, 4 refused input.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import renc_lib as L

import numpy as np  # noqa: E402
import torch  # noqa: E402

RISE_MIN, LEVEL_MIN, KEEP_MIN = 0.05, 0.90, 0.99
V_PASS, V_FAIL, V_LEAK, V_FOLDS, V_BROKEN = "M-RENC-PASS", "RE-FAILS-MECH", "RENC-INVALID-LEAK", "RENC-INVALID-FOLDS", "GATE-BROKEN"
EXIT = {V_PASS: 0, V_FAIL: 5, V_LEAK: 6, V_FOLDS: 6, V_BROKEN: 6}


def cos_rows(A, B):
    return np.array([L.cosine(A[i], B[i]) for i in range(A.shape[0])], np.float64)


def boot(per_scene, stat, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    S = len(per_scene)
    v = [stat(np.concatenate([per_scene[i] for i in rng.integers(0, S, S)])) for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def load_dir(d):
    return {o["scene"]: o for o in (torch.load(p, map_location="cpu", weights_only=True) for p in sorted(glob.glob(os.path.join(d, "*.pt"))))}


def score(gap, enc_tokens, train_lists, id0, n_boot=2000, meta=None):
    """gap: {scene: {ticks, render, real}} (GAP48); enc_tokens: {'A': {scene: {...}}, 'B': ...}; train_lists: {'A': [scene ids], 'B': [...]}. -> the M_RENC dict."""
    recs = []
    for sid, g in sorted(gap.items()):
        f = L.fold_of(sid)
        o = L.other_fold(f)
        eo, es = enc_tokens[o][sid], enc_tokens[f][sid]
        for e in (eo, es):
            if not torch.equal(e["ticks"].long(), g["ticks"].long()):
                raise ValueError(f"RENC_TICKS_DIFFER {sid[:8]}: the encoder's ticks are not GAP48's")
        t, gr = g["real"].float(), g["render"].float()
        keep_c = cos_rows(eo["real"].float(), t)
        recs.append({"scene": sid, "fold": f, "before": cos_rows(gr, t), "after": cos_rows(eo["render"].float(), t), "keep": keep_c, "keep_shift": cos_rows(eo["real"].float().roll(1, 0), t),
                     "after_same": cos_rows(es["render"].float(), t), "keep_same": cos_rows(es["real"].float(), t), "leak_main": sid in train_lists[o], "leak_same": sid in train_lists[f],
                     "consistent": bool(sid in train_lists[f] and sid not in train_lists[o])})

    def arm(key, base="before"):
        v, b = [r[key] for r in recs], [r["before"] for r in recs]
        c, bc = np.concatenate(v), np.concatenate(b)
        return {"n_pairs": int(c.size), "n_scenes": len(recs), "median": float(np.median(c)), "median_before": float(np.median(bc)), "rise": float(np.median(c) - np.median(bc)),
                "p10": float(np.percentile(c, 10)), "p90": float(np.percentile(c, 90)), "ci_median_scene_cluster_B2000_seed0": boot(v, lambda x: float(np.median(x)), n_boot),
                "scenes_improved": int(sum(float(np.median(r[key])) > float(np.median(r["before"])) for r in recs))}
    m_ren, keep, m_same, keep_sh = arm("after"), arm("keep"), arm("after_same"), arm("keep_shift")
    m_ren["pass_rise"], m_ren["pass_level"] = bool(m_ren["rise"] >= RISE_MIN), bool(m_ren["median"] >= LEVEL_MIN)
    keep["pass"] = bool(keep["median"] >= KEEP_MIN)
    leaks = sorted(r["scene"] for r in recs if r["leak_main"])
    reg_flagged = sorted(r["scene"] for r in recs if r["leak_same"])
    incons = sorted(r["scene"] for r in recs if not r["consistent"])
    keep_reg_fails = bool(keep_sh["median"] < KEEP_MIN)
    id0_ok = bool(id0 and id0.get("pass") and id0.get("regression_arm_flagged"))
    if not reg_flagged or not keep_reg_fails or not id0_ok:
        verdict = V_BROKEN
    elif leaks:
        verdict = V_LEAK
    elif incons:
        verdict = V_FOLDS
    else:
        verdict = V_PASS if (m_ren["pass_rise"] and m_ren["pass_level"] and keep["pass"]) else V_FAIL
    out = {"card": "PREREG_T2_RENC.md", "card_prefix_5544_sha256": L.CARD_PREFIX_SHA256, "gates": "M-RENC + G-KEEP (+ G-ID0 from box 2)", "verdict": verdict, "word_if_failed": V_FAIL,
           "thresholds": {"rise_min": RISE_MIN, "level_min": LEVEL_MIN, "keep_min": KEEP_MIN},
           "statistic": "median over held-out pairs, pooled across scenes, of the flattened uncentered float64 cosine (t2a_v8f_encoder.cosine); targets = the frozen encoder's tokens of the real frames",
           "M_RENC": m_ren, "G_KEEP": keep, "G_ID0": id0,
           "regression_arms": {"M_RENC_same_fold": {"flagged_scenes": len(reg_flagged), "of": len(recs), "flagged": bool(reg_flagged), "naive_median_after": m_same["median"], "naive_rise": m_same["rise"],
                                                    "would_pass_unguarded": bool(m_same["rise"] >= RISE_MIN and m_same["median"] >= LEVEL_MIN)},
                               "G_KEEP_pairing_shifted_by_one": {"median": keep_sh["median"], "fails_as_it_must": keep_reg_fails},
                               "G_ID0_perturbed_B": {"flagged": bool(id0 and id0.get("regression_arm_flagged"))}},
           "leaks": leaks, "inconsistent_fold_lists": incons, "gate_valid": bool(reg_flagged) and keep_reg_fails and id0_ok and not leaks and not incons,
           "per_scene": {r["scene"]: {"fold": r["fold"], "before": float(np.median(r["before"])), "after": float(np.median(r["after"])), "keep": float(np.median(r["keep"]))} for r in recs},
           "tool_sha256": L.sha256_file(os.path.abspath(__file__)), "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.update(meta or {})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="the pull of renc/out: gate_tokens/{A,B}, TRAIN_RESULT_{A,B}.json, G_ID0.json")
    ap.add_argument("--gap48-tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None, help="optional: the meta files of the data job, compared with GAP48 at the gate ticks")
    ap.add_argument("--expect-scenes", type=int, default=48)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print(f"ZZ RENC-GATE-REFUSED RENC_OUT_EXISTS {a.out} (move it aside) ZZ")
        return 4
    d = a.out_dir.replace("\\", "/")
    try:
        gap = {o["scene"]: o for o in (torch.load(p, map_location="cpu", weights_only=True) for p in sorted(glob.glob(os.path.join(a.gap48_tokens, "*.pt"))))}
        if len(gap) != a.expect_scenes:
            raise ValueError(f"RENC_SCENE_COUNT {len(gap)} GAP48 scenes, expected {a.expect_scenes}")
        enc = {f: load_dir(f"{d}/gate_tokens/{f}") for f in L.FOLDS}
        tl = {f: json.load(open(f"{d}/TRAIN_RESULT_{f}.json", encoding="utf-8"))["train_scenes"] for f in L.FOLDS}
        id0 = json.load(open(f"{d}/G_ID0.json", encoding="utf-8"))
        meta = {"inputs": {"out_dir": a.out_dir, "gap48_tokens": a.gap48_tokens}}
        if a.data:
            eq = n = 0
            for sid, g in gap.items():
                m = torch.load(f"{a.data}/{sid[:8]}.meta.pt", map_location="cpu", weights_only=True)
                gi = m["gate_idx"].tolist()
                n += 1
                eq += int(torch.equal(m["target"][gi], g["real"]) and torch.equal(m["render_gate"], g["render"]))
            meta["data_job_vs_gap48"] = {"scenes_compared": n, "scenes_bit_equal_target_and_render": eq}
        res = score(gap, enc, tl, id0, meta=meta)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"ZZ RENC-GATE-REFUSED {e} ZZ")
        return 4
    L.jdump(res, a.out)
    m, k = res["M_RENC"], res["G_KEEP"]
    print(f"ZZ M-RENC {res['verdict']} median before {m['median_before']:.4f} after {m['median']:.4f} rise {m['rise']:+.4f} keep {k['median']:.4f} gate_valid {res['gate_valid']} ZZ")
    return EXIT[res["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
