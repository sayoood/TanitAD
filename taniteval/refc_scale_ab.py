"""REF-C **scale A/B** — encoder allocation, read at DECISION grade.

THE QUESTION (D-030's middle rung, 2026-07-21): REF-C-base's encoder is
90,458,632 params — within 3.8 % of flagship v1's 87,121,280 — so base-vs-XL is
the program's only near-matched test of *encoder allocation*. Does XL's
199,496,532-param encoder buy real **fan quality**, or is encoder scale simply
not the lever?

WHY ADE IS THE WRONG PRIMARY READ. ADE mixes proposal quality with ranking, and
REF-C's ranking is known-broken (it selects with the t=0 classifier score over the
UN-refined anchors; `refc.py::AnchoredDiffusionDecoder.forward` discards the
denoise passes' own confidences). The proposal-quality read is **oracle-in-fan**:
the ADE of the best proposal the fan actually contained.

THE COVERAGE CONFOUND, AND ITS EXACT CONTROL. base carries **128** anchors, XL
**256**, so a raw oracle comparison charges base for having half the vocabulary.
But base's anchor buffer is a **bit-exact strict prefix** of XL's (verified here
at load: `max|A - B[:128]| == 0`, same FPS script/source/pool-cap/seed), so
restricting XL's fan to its FIRST 128 anchors yields a **coverage-matched**
oracle over the identical vocabulary. Three numbers are therefore reported and
must never be collapsed into one:

    oracle_full      base over 128   vs   XL over 256   (as-deployed, confounded
                                                         by fan width)
    oracle_matched   base over 128   vs   XL over 128   (SAME vocabulary — this
                                                         is the encoder read)
    selected         base over 128   vs   XL over 256   (the leaderboard row;
                                                         ranking included)

INTERVALS. Single-arm = episode-cluster bootstrap over the 40 val episodes;
base-vs-XL = the **paired** version (same windows, so pairing is valid and
strictly more powerful). `taniteval/ci.py`. The legacy `heldout ± ci95` block is
`overlapping_holdout_se`, measured 1.28-2.06x too narrow, and is printed for
continuity ONLY.

⚠️ STANDING CONFOUND, not removable by any control here: base trained on route
labels **v2.1**, XL on **v1** (MODEL_REGISTRY §4.3). Scale, anchor count and
labels move together between these two arms. `oracle_matched` removes the anchor
count; nothing in this script removes the labels.

Run (eval pod):
    python3 /root/taniteval/refc_scale_ab.py dump    --model refc-base-30k   # GPU
    python3 /root/taniteval/refc_scale_ab.py analyze --a refc-base-30k \
                                                     --b refc-xl-30k         # CPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import ci as C  # noqa: E402

RES = Path("/root/taniteval/results")
N_BOOT = 2000


# --------------------------------------------------------------------------- #
# PASS 1 (GPU) — one decode over the canonical val, keeping every proposal      #
# --------------------------------------------------------------------------- #
def dump(key, episodes=40, device="cuda", batch=8):
    """Full-fan dump for ``key``, REUSING ``refc_rerank.dump`` verbatim.

    That function's decode call is byte-identical to ``refc_eval.collect``
    (window 8 / stride 8, nav=follow, v0 through the measurement encoder,
    ``steps = cfg.decoder.diffusion_steps``) and it asserts
    ``sel_idx == argmax(anchor_logits)`` and
    ``waypoints == anchor_traj[sel_idx]``, so the selected row it produces IS the
    published leaderboard row. Nothing is re-implemented here.
    """
    from taniteval import refc_rerank as RR
    RR.BASE_KEY = key                       # read inside dump() at call time
    out = RES / f"fan_{key}.pt"
    return RR.dump(episodes=episodes, device=device, batch=batch, out=out)


# --------------------------------------------------------------------------- #
# PASS 2 (CPU) — oracle / selection decomposition + intervals                   #
# --------------------------------------------------------------------------- #
def _de_all(fan, gt):
    """[B,N] per-proposal ADE@2s (mean over the 4 time horizons)."""
    return torch.linalg.norm(fan - gt[:, None], dim=-1).mean(-1)


def _strata(speed, head_deg):
    """bench.run's own stratum labels, so the rows line up with results/<key>.json."""
    from driving_diagnostic import curvature_bucket
    q = torch.quantile(speed, torch.tensor([1 / 3, 2 / 3]))
    spd = np.array(["low" if float(s) < float(q[0]) else
                    "high" if float(s) >= float(q[1]) else "med" for s in speed])
    curv = np.array([curvature_bucket(float(h)) for h in head_deg])
    return spd, curv


def _boot(v, eid, tag):
    b = C.episode_cluster_bootstrap(np.asarray(v, dtype=np.float64), eid,
                                    n_boot=N_BOOT, seed=0)
    b["metric"] = tag
    return b


def arm_panel(d, n_sub=None):
    """Selected / oracle / gap decomposition for ONE dumped arm.

    ``n_sub`` restricts the fan to its first ``n_sub`` anchors — the
    coverage-matched control. Valid ONLY because base's vocabulary is a bit-exact
    prefix of XL's; the caller asserts that.
    """
    fan, gt, sel = d["fan"], d["gt"], d["sel"]
    if n_sub is not None:
        fan = fan[:, :n_sub]
    de_all = _de_all(fan, gt)                                # [B,N]
    de_or = de_all.min(1).values                             # oracle-in-fan
    per_h = torch.linalg.norm(d["fan"][torch.arange(fan.shape[0]), sel] - gt,
                              dim=-1)                        # [B,4] as SELECTED
    return {"de_sel": per_h.mean(-1).numpy().astype(np.float64),
            "fde": per_h[:, -1].numpy().astype(np.float64),
            "miss": (per_h[:, -1] > 2.0).numpy().astype(np.float64),
            "de_oracle": de_or.numpy().astype(np.float64),
            "n_anchors": int(fan.shape[1])}


def oracle_vs_K(d, Ks=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    """Oracle-in-fan restricted to the FIRST K anchors of the shared FPS order.

    The decisive control for "is the fan lever encoder scale or vocabulary
    WIDTH?". Because the vocabularies nest, the same K means the same anchor set
    in both arms. ⚠️ It still charges XL for a structural cost: XL's winner-takes-
    all training spread its modes over 256 slots, so a prefix restriction removes
    the interstitial anchors nearest ~half its targets. Read the CURVE SHAPE and
    where the arms cross, not a single K in isolation.
    """
    de = _de_all(d["fan"], d["gt"])
    out = {}
    for K in Ks:
        if K > de.shape[1]:
            continue
        v = de[:, :K].min(1).values.numpy().astype(np.float64)
        b = C.episode_cluster_bootstrap(v, d["eid"], n_boot=N_BOOT, seed=0)
        out[int(K)] = {"oracle": b["mean"], "lo": b["lo"], "hi": b["hi"]}
    return out


def analyze(a_key, b_key, out=None):
    da = torch.load(RES / f"fan_{a_key}.pt", map_location="cpu",
                    weights_only=False)
    db = torch.load(RES / f"fan_{b_key}.pt", map_location="cpu",
                    weights_only=False)
    assert da["eid"] == db["eid"], (
        f"{a_key}/{b_key} are not on the same windows — the paired test would be "
        "invalid and the oracle comparison meaningless")
    eid = da["eid"]
    na, nb = int(da["fan"].shape[1]), int(db["fan"].shape[1])

    # the coverage-matched control is only legitimate if the vocabularies nest
    matched_ok = False
    if na <= nb:
        # anchors live in the ckpt buffer; the dump keeps the REFINED fan, so the
        # nesting is checked on the raw vocabulary carried by each checkpoint.
        Aa = torch.load(da["ckpt"], map_location="cpu",
                        weights_only=False)["model"]["decoder.anchors"].float()
        Ab = torch.load(db["ckpt"], map_location="cpu",
                        weights_only=False)["model"]["decoder.anchors"].float()
        matched_ok = bool(torch.equal(Aa, Ab[:Aa.shape[0]]))

    A = arm_panel(da)
    B = arm_panel(db)
    Bm = arm_panel(db, n_sub=na) if matched_ok and na < nb else None

    spd, curv = _strata(da["speed"], da["head_deg"])
    rows = {}

    def pack(tag, va, vb):
        p = C.paired_episode_cluster_bootstrap(va, vb, eid, n_boot=N_BOOT, seed=0)
        return {"a": _boot(va, eid, f"{tag}:{a_key}"),
                "b": _boot(vb, eid, f"{tag}:{b_key}"),
                "paired_a_minus_b": p,
                "per_window_corr": round(float(np.corrcoef(va, vb)[0, 1]), 3)}

    rows["selected_ade2s"] = pack("selected", A["de_sel"], B["de_sel"])
    rows["selected_fde2s"] = pack("fde", A["fde"], B["fde"])
    rows["selected_miss2m"] = pack("miss", A["miss"], B["miss"])
    rows["oracle_in_fan_full"] = pack("oracle_full", A["de_oracle"],
                                      B["de_oracle"])
    if Bm is not None:
        rows["oracle_in_fan_matched"] = pack("oracle_matched", A["de_oracle"],
                                             Bm["de_oracle"])

    def gapstats(P):
        gap = P["de_sel"] - P["de_oracle"]
        return {"sel_gap_mean": round(float(gap.mean()), 4),
                "sel_gap_ci": _boot(gap, eid, "sel_gap"),
                "frac_sel_2x_worse": round(
                    float((P["de_sel"] > 2 * P["de_oracle"]).mean()), 4),
                "n_anchors": P["n_anchors"]}

    out_d = {
        "a": a_key, "b": b_key, "n_windows": len(eid),
        "n_episodes": len(set(eid)), "n_boot": N_BOOT,
        "estimator": "episode_cluster_bootstrap / paired_episode_cluster_bootstrap",
        "anchors": {a_key: na, b_key: nb,
                    "b_vocabulary_is_superset_of_a": matched_ok},
        "gap": {a_key: gapstats(A), b_key: gapstats(B),
                f"{b_key}@{na}": gapstats(Bm) if Bm else None},
        "comparisons": rows,
        "oracle_vs_K": {a_key: oracle_vs_K(da), b_key: oracle_vs_K(db)},
        "strata": {},
        "confound": ("base trained on route labels v2.1, XL on v1 "
                     "(MODEL_REGISTRY 4.3) — scale and labels are entangled; "
                     "oracle_in_fan_matched removes the ANCHOR-COUNT confound "
                     "only, never the label one"),
    }
    for name, lab in (("by_speed", spd), ("by_curvature", curv)):
        blk = {}
        for u in sorted(set(lab.tolist())):
            m = lab == u
            e_sub = [e for e, k in zip(eid, m) if k]
            blk[u] = {
                "n": int(m.sum()),
                f"{a_key}_sel": round(float(A["de_sel"][m].mean()), 4),
                f"{b_key}_sel": round(float(B["de_sel"][m].mean()), 4),
                f"{a_key}_oracle": round(float(A["de_oracle"][m].mean()), 4),
                f"{b_key}_oracle": round(float(B["de_oracle"][m].mean()), 4),
                "paired_sel_a_minus_b": C.paired_episode_cluster_bootstrap(
                    A["de_sel"][m], B["de_sel"][m], e_sub, n_boot=N_BOOT,
                    seed=0),
            }
            if Bm is not None:
                blk[u][f"{b_key}@{na}_oracle"] = round(
                    float(Bm["de_oracle"][m].mean()), 4)
        out_d["strata"][name] = blk

    p = Path(out) if out else RES / f"scaleab_{a_key}_vs_{b_key}.json"
    p.write_text(json.dumps(out_d, indent=2, default=str))

    def line(tag, r):
        pa = r["paired_a_minus_b"]
        print(f"  {tag:<24} {a_key} {r['a']['mean']:.4f} "
              f"[{r['a']['lo']:.4f},{r['a']['hi']:.4f}]   "
              f"{b_key} {r['b']['mean']:.4f} "
              f"[{r['b']['lo']:.4f},{r['b']['hi']:.4f}]   "
              f"delta {pa['delta']:+.4f} [{pa['lo']:+.4f},{pa['hi']:+.4f}] "
              f"{'SEPARATED' if pa['separated'] else 'NOT separated'}")

    print(f"\n=== REF-C SCALE A/B — {a_key} vs {b_key} "
          f"(n={len(eid)} windows / {len(set(eid))} episodes, "
          f"episode-cluster bootstrap B={N_BOOT}) ===\n")
    print(f"  anchors: {a_key} {na} · {b_key} {nb} · nested vocabulary: "
          f"{matched_ok}")
    for k, r in rows.items():
        line(k, r)
    for k, g in out_d["gap"].items():
        if g:
            print(f"  gap[{k:<18}] sel-oracle {g['sel_gap_mean']:.4f} · "
                  f"frac_sel_2x_worse {g['frac_sel_2x_worse']:.4f} "
                  f"({g['n_anchors']} anchors)")
    print("\n  ORACLE-IN-FAN vs FAN WIDTH K (first-K of the shared FPS order)")
    print(f"  {'K':>5} {a_key:>16} {b_key:>16}")
    for K in sorted({int(k) for arm in out_d["oracle_vs_K"].values()
                     for k in arm}):
        ra = out_d["oracle_vs_K"][a_key].get(K)
        rb = out_d["oracle_vs_K"][b_key].get(K)
        sa = ("%.4f" % ra["oracle"]) if ra else "  --  "
        sb = ("%.4f" % rb["oracle"]) if rb else "  --  "
        print("  %5d %16s %16s" % (K, sa, sb))
    print(f"\n-> {p}")
    return out_d


def main():
    ap = argparse.ArgumentParser("refc_scale_ab")
    ap.add_argument("cmd", choices=["dump", "analyze"])
    ap.add_argument("--model", default="refc-base-30k")
    ap.add_argument("--a", default="refc-base-30k")
    ap.add_argument("--b", default="refc-xl-30k")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a.model, episodes=a.episodes, device=a.device, batch=a.batch)
    else:
        analyze(a.a, a.b, out=a.out)


if __name__ == "__main__":
    main()
