"""RALIGN gate M-ALIGN (PREREG_T2_RALIGN.md section 4, mechanism): does the cross-fitted map pull the render tokens toward the real-frame tokens on HELD-OUT pairs? CPU only.

    python -I ralign_gate.py --tokens-dir <GAP48 pull>/tokens --map-dir <ralign_fit out-dir> --out M_ALIGN.json [--fold-digit 0|1] [--expect-scenes 48]

For every scene (fold f by the first hex digit of sha256(scene id)) every banked tick pair (render r, real x) is scored two ways with the SAME cosine the policy side uses
(`t2a_v8f_encoder.cosine`: flattened, uncentered, float64): before = cos(r, x), after = cos(map_other(r), x) where map_other is the map fitted on the OTHER fold and applied through
`TokenMap.apply` (the function the T2 policy calls, fp16 output included). The statistic is the median over all held-out pairs, pooled across scenes; a scene-cluster bootstrap
(B 2000, seed 0) is reported beside it and decides nothing.
Pass: rise (median after - median before) >= 0.05 AND median after >= 0.90 (the real-frame 1 s level of gapscale/GAPSCALE.json). Otherwise ALIGN-FAILS-MECH (card section 4: no T2 stage
runs, the next lever is encoder adaptation).
Verdict words: M-ALIGN-PASS | ALIGN-FAILS-MECH | M-ALIGN-INVALID-LEAK (a scene was read under a map whose fold list contains it) | M-ALIGN-INVALID-FOLDS (a scene is not in its own
fold's list) | GATE-BROKEN (the deliberate-regression arm was not flagged: the gate cannot be trusted).
Deliberate-regression arm, run in every gate: the same scoring with the map fitted on the SAME fold. It must be flagged by the provenance check; its naive numbers (what an unguarded gate
would have read) are written beside it so the size of the leak is visible. Exit: 0 pass, 5 fails mechanism, 6 invalid or broken, 4 refused input.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
CODE = os.path.dirname(HERE) + "/code"
for _p in (HERE, CODE):
    if _p not in sys.path:
        sys.path.append(_p)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ralign_lib as L  # noqa: E402
import t2a_v8f_encoder as E  # noqa: E402  (code/, unchanged: `cosine` is the statistic of record)

BOOT_N, BOOT_SEED = 2000, 0
V_PASS, V_FAIL, V_LEAK, V_FOLDS, V_BROKEN = "M-ALIGN-PASS", "ALIGN-FAILS-MECH", "M-ALIGN-INVALID-LEAK", "M-ALIGN-INVALID-FOLDS", "GATE-BROKEN"
EXIT = {V_PASS: 0, V_FAIL: 5, V_LEAK: 6, V_FOLDS: 6, V_BROKEN: 6}


def load_maps(map_dir, shape=L.SHAPE):
    """MAP_A / MAP_B re-hashed against MAPS.json (file and content). -> ({'A': TokenMap, 'B': TokenMap}, MAPS.json dict)."""
    idx = json.load(open(f"{map_dir}/MAPS.json", encoding="utf-8"))
    maps = {f: L.load_map(f"{map_dir}/{idx['maps'][f]['file']}", idx["maps"][f]["file_sha256"], idx["maps"][f]["tensor_sha256"], shape) for f in L.FOLDS}
    return maps, idx


def cosines(R: torch.Tensor, X: torch.Tensor):
    """Per-pair flattened cosine through `t2a_v8f_encoder.cosine`. R, X: [n, P, C]."""
    return np.array([E.cosine(R[i], X[i]) for i in range(R.shape[0])], np.float64)


def score_scene(sid, R, X, maps, digit=0):
    """-> one scene's record: its fold, the three cosine arrays (before, after under the OTHER fold's map, after under its OWN fold's map) and the provenance flags."""
    f = L.fold_of(sid, digit)
    own, oth = maps[f], maps[L.other_fold(f)]
    in_own = sid in own.meta.get("fold_scenes", [])
    in_oth = sid in oth.meta.get("fold_scenes", [])
    return {"scene": sid, "fold": f, "n": int(R.shape[0]), "before": cosines(R, X), "after_cross": cosines(oth.apply(R), X), "after_same": cosines(own.apply(R), X),
            "leak_cross": bool(in_oth), "leak_same": bool(in_own), "consistent": bool(in_own and not in_oth)}


def boot(per_scene, stat, n=BOOT_N, seed=BOOT_SEED):
    """Scene-cluster bootstrap of a function of (before, after) pooled arrays -> [lo, hi] (2.5, 97.5). Answers: would another draw of scenes say this."""
    rng = np.random.default_rng(seed)
    S = len(per_scene)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, S, S)
        vals.append(stat(np.concatenate([per_scene[i][0] for i in idx]), np.concatenate([per_scene[i][1] for i in idx])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def arm(recs, key, leak_key, rise_min=L.RISE_MIN, level_min=L.LEVEL_MIN):
    ps = [(r["before"], r[key]) for r in recs]
    b, a = np.concatenate([p[0] for p in ps]), np.concatenate([p[1] for p in ps])
    mb, ma = float(np.median(b)), float(np.median(a))
    leaks = sorted(r["scene"] for r in recs if r[leak_key])
    out = {"n_pairs": int(b.size), "n_scenes": len(recs), "median_before": mb, "median_after": ma, "rise": ma - mb, "pass_rise": bool(ma - mb >= rise_min), "pass_level": bool(ma >= level_min),
           "after_summary": L.percentile_summary(a), "before_summary": L.percentile_summary(b),
           "ci_scene_cluster_B2000_seed0": {"median_after": boot(ps, lambda b_, a_: float(np.median(a_))), "rise": boot(ps, lambda b_, a_: float(np.median(a_) - np.median(b_)))},
           "scenes_improved": int(sum(float(np.median(r[key])) > float(np.median(r["before"])) for r in recs)),
           "per_scene": {r["scene"]: {"fold": r["fold"], "n": r["n"], "median_before": float(np.median(r["before"])), "median_after": float(np.median(r[key]))} for r in recs},
           "leaks": leaks}
    return out


def decide(main, regression, inconsistent):
    """The verdict word. The regression arm must be flagged (it carries leaks); if it is not, the gate is broken whatever the numbers say."""
    if not regression["leaks"]:
        return V_BROKEN
    if main["leaks"]:
        return V_LEAK
    if inconsistent:
        return V_FOLDS
    return V_PASS if (main["pass_rise"] and main["pass_level"]) else V_FAIL


def gate(pairs, maps, idx=None, digit=0, expect_scenes=48, rise_min=L.RISE_MIN, level_min=L.LEVEL_MIN, meta=None):
    """pairs: iterable of (scene id, render [n,P,C], real [n,P,C]). -> the M_ALIGN dict."""
    recs = [score_scene(sid, R, X, maps, digit) for sid, R, X in pairs]
    if expect_scenes is not None and len(recs) != expect_scenes:
        raise ValueError(f"RALIGN_SCENE_COUNT {len(recs)} scenes read, expected {expect_scenes}")
    if len({r["scene"] for r in recs}) != len(recs):
        raise ValueError("RALIGN_DUPLICATE_SCENE")
    for f in L.FOLDS:
        if maps[f].meta.get("fold") != f or maps[f].meta.get("fold_digit", digit) != digit:
            raise ValueError(f"RALIGN_MAP_FOLD map {f} carries fold {maps[f].meta.get('fold')!r} / digit {maps[f].meta.get('fold_digit')!r}; the gate reads digit {digit}")
    inconsistent = sorted(r["scene"] for r in recs if not r["consistent"])
    main, reg = arm(recs, "after_cross", "leak_cross", rise_min, level_min), arm(recs, "after_same", "leak_same", rise_min, level_min)
    verdict = decide(main, reg, inconsistent)
    out = {"card": "PREREG_T2_RALIGN.md", "card_sha256": L.CARD_SHA256, "gate": "M-ALIGN", "tier": "T2-RA mechanism: render tokens through the OTHER fold's map against real-frame tokens, held-out pairs",
           "verdict": verdict, "thresholds": {"rise_min": rise_min, "level_min": level_min, "level_source": "gapscale/GAPSCALE.json real_within_clip_lag_ticks.k10.median 0.9000 (real frames 1 s apart)"},
           "statistic": "median over all held-out pairs, pooled across scenes, of the flattened uncentered float64 cosine (t2a_v8f_encoder.cosine)", "fold_digit": digit,
           "main": main, "regression_arm": {"what": "the SAME scoring with the map fitted on the SAME fold as the scene", "flagged": bool(reg["leaks"]), "n_leak_scenes": len(reg["leaks"]),
                                            "naive": {"median_before": reg["median_before"], "median_after": reg["median_after"], "rise": reg["rise"], "would_pass_unguarded": bool(reg["pass_rise"] and reg["pass_level"])}},
           "inconsistent_fold_lists": inconsistent, "gate_valid": bool(reg["leaks"]) and not main["leaks"] and not inconsistent,
           "lib_sha256": L.sha256_file(L.__file__), "tool_sha256": L.sha256_file(os.path.abspath(__file__)), "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    if idx is not None:
        out["maps"] = {f: {"file_sha256": idx["maps"][f]["file_sha256"], "tensor_sha256": idx["maps"][f]["tensor_sha256"]} for f in L.FOLDS}
    out.update(meta or {})
    return out


def iter_pairs(tokens_dir, keys=None, shape=L.SHAPE):
    for p in sorted(glob.glob(os.path.join(tokens_dir, "*.pt"))):
        yield L.load_pair_file(p.replace("\\", "/"), shape=shape, **(keys or {}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-dir", required=True)
    ap.add_argument("--map-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold-digit", type=int, default=0, choices=(0, 1))
    ap.add_argument("--expect-scenes", type=int, default=48)
    ap.add_argument("--scene-key", default="scene")
    ap.add_argument("--render-key", default="render")
    ap.add_argument("--real-key", default="real")
    ap.add_argument("--shape", default=f"{L.SHAPE[0]},{L.SHAPE[1]}", help="only a test passes another shape")
    a = ap.parse_args(argv)
    shape = tuple(int(x) for x in a.shape.split(","))
    if os.path.exists(a.out):
        print(f"ZZ RALIGN-GATE-REFUSED RALIGN_OUT_EXISTS {a.out} (move it aside) ZZ")
        return 4
    try:
        maps, idx = load_maps(a.map_dir.replace("\\", "/"), shape)
        res = gate(iter_pairs(a.tokens_dir.replace("\\", "/"), {"scene_key": a.scene_key, "render_key": a.render_key, "real_key": a.real_key}, shape), maps, idx, a.fold_digit, a.expect_scenes,
                   meta={"inputs": {"tokens_dir": a.tokens_dir, "map_dir": a.map_dir}})
    except (ValueError, FileNotFoundError) as e:
        print(f"ZZ RALIGN-GATE-REFUSED {e} ZZ")
        return 4
    L.jdump(res, a.out)
    m = res["main"]
    print(f"ZZ M-ALIGN {res['verdict']} median before {m['median_before']:.4f} after {m['median_after']:.4f} rise {m['rise']:+.4f} regression-flagged {res['regression_arm']['flagged']} ZZ")
    return EXIT[res["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
