"""IDM v4 — the OWED `steer` data-budget retrain (IDM_V3.md §9 escalation 4).

IDM v3 shipped `speed 0.907 / yaw 0.841` but `steer` REGRESSED 0.742 -> 0.408 and
the diagnosis on record is a data-budget effect: A0 (the deployed
`idm_head_v1`) saw ~160 clips, every v3 arm saw **68**. This script runs that
retrain as a pre-registered LADDER rather than a single point, because a ladder
can distinguish "not enough data" from "the recipe is not data-limited at all",
and a single point cannot.

PRE-REGISTRATION — both outcomes are publishable, neither is a failure
--------------------------------------------------------------------
H1  `steer` rises monotonically with the episode budget and the top rung beats
    A0 on a PAIRED read  ->  the v3 regression IS a data-budget effect; the
    steer head can be re-shipped from the top rung.
H0  `steer` is flat in the budget, or rises but does not reach A0  ->  the
    regression is NOT (only) a data-budget effect. That is a real finding about
    the recipe and it retires the standing explanation.
Either way the curve itself is the deliverable.

WHY THIS IS A VALID PAIRED READ
-------------------------------
`idm2_lib.split_tags()` derives the split from whatever sits in the latent dir,
so naively adding episodes would MOVE THE VAL SET and silently destroy every
comparison. Instead the extra training episodes are linked in under prefixes
(`cmx_` / `paix_`) that `split_tags` provably ignores — it selects with
`t.startswith(dom + "_")` for dom in ("pai", "cm"), and neither "cmx_..." nor
"paix_..." starts with "cm_" or "pai_". The 36 val episodes / 4,195 val windows
are therefore BYTE-IDENTICAL to the ones A0's stored predictions were computed
on, and the script asserts that before training.

EPISODE-DISJOINTNESS IS MEASURED, NOT ASSUMED
---------------------------------------------
The pod3 latent cache stores only {z, poses, actions} — no episode_id, no src —
so identity CANNOT be read off metadata. Every candidate training episode is
content-fingerprinted (md5 of the float32 poses) and any episode whose
fingerprint matches a VAL episode is EXCLUDED from training. Measured
2026-07-27: 4 of the 36 val episodes are present in the pod3 cache
(cm_00018, cm_00039, pai_00000, pai_00018) and are excluded. Without this check
the run would have leaked 11 % of its val set — the REF-A I-JEPA failure mode.

LABEL PROTOCOL
--------------
REPAIRED (`idm3_labels.heading_repair`, v_min = 0.5 m/s), i.e. the same protocol
as every v3 arm, extended so the comma extras (`cmx_`) are repaired too — a
`steer` number on unrepaired labels is not comparable to one on repaired labels.
NOTE: the repair applies to the yaw/heading channel; `steer` is not rewritten by
it. It is applied for protocol identity with v3, not because it moves steer.

Estimator: taniteval.ci.(paired_)episode_cluster_bootstrap, unit = EPISODE,
B = 2000. `overlapping_holdout_se` is never called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), "/root/idm2", "/root/taniteval", "/root/idm3",
           "/root/v4eval/stack", "/root/v4eval/stack/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import idm2_lib as L            # noqa: E402
import idm3_arms as A           # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# identity                                                                     #
# --------------------------------------------------------------------------- #
def fingerprint(tag: str) -> str:
    """Content identity of an episode: md5 of its float32 poses."""
    po = L.load_ep(tag)["poses"].float().numpy().astype(np.float32)
    return hashlib.md5(np.ascontiguousarray(po).tobytes()).hexdigest()


def corpus_of(tag: str) -> str:
    return "cm" if tag.startswith(("cm_", "cmx_")) else "pai"


# --------------------------------------------------------------------------- #
# label repair, extended to the linked-in comma extras                         #
# --------------------------------------------------------------------------- #
def repair_labels_ext(setd, v_min=A.V_MIN):
    """`idm3_arms.repair_labels`, but treating `cmx_` as comma2k19.

    The stock version tests `tag.startswith("cm_")`, which is False for the
    linked-in extras and would leave 79 extra comma episodes on the BROKEN
    arctan2-at-standstill heading while the original 64 were repaired — a
    silently mixed label protocol inside one training set.
    """
    S = setd["S"].clone()
    n_changed = 0
    for tag in np.unique(setd["eid"]):
        m = setd["eid"] == tag
        po = L.load_ep(tag)["poses"].float()
        if corpus_of(str(tag)) == "cm":
            yaw = A.heading_repair(po, v_min)[0]
        else:
            yaw = po[:, 2]
        t = setd["tcen"][torch.from_numpy(m)]
        yr = A.yaw_rate_from(yaw, t)
        n_changed += int((yr - S[torch.from_numpy(m), 1]).abs().gt(1e-9).sum())
        S[torch.from_numpy(m), 1] = yr
    return S, n_changed


def prep(setd):
    """Attach exactly what the R0 recipe needs. `mgain`/`ctx` are deliberately
    NOT computed: R0 has ctx=False and phys_scale unset, and `metric_gain_of`
    would need a geometry-table row for every linked-in extra tag."""
    setd["Akin"] = ((setd["Vseq"][:, A.KBUILD + 1] - setd["Vseq"][:, A.KBUILD - 1])
                    / (2 * A.ih.DT))
    setd["S_leg"] = setd["S"].clone()
    setd["S"], nch = repair_labels_ext(setd)
    # normalise the reporting domain: the extras carry dom "cmx"/"paix"
    setd["dom"] = np.array([corpus_of(str(t)) for t in setd["eid"]])
    return nch


# --------------------------------------------------------------------------- #
def nested_ladder(pool_tags, rungs, seed=0):
    """Deterministic NESTED, corpus-stratified subsets: rung i is a subset of
    rung i+1, so the ladder measures the budget and not the sample."""
    rng = np.random.default_rng(seed)
    by = {"cm": [], "pai": []}
    for t in pool_tags:
        by[corpus_of(t)].append(t)
    for c in by:
        by[c] = sorted(by[c])
        rng.shuffle(by[c])
    n_tot = len(pool_tags)
    out = {}
    for r in rungs:
        r = min(r, n_tot)
        # keep the pool's corpus proportions
        n_cm = max(1, round(r * len(by["cm"]) / n_tot))
        n_cm = min(n_cm, len(by["cm"]))
        n_pai = min(r - n_cm, len(by["pai"]))
        out[r] = sorted(by["cm"][:n_cm] + by["pai"][:n_pai])
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", type=int, default=[68, 200, 400, 757])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default="/workspace/idmretrain/out/idm4_steer.json")
    ap.add_argument("--a0-preds", default="/workspace/idmretrain/idm3/out/a0_preds.npy")
    ap.add_argument("--save-ckpt", default="")
    a = ap.parse_args()

    t_start = time.time()
    # ---- 1. the FROZEN v3 split -------------------------------------------
    tr68, va_tags = L.split_tags()
    log(f"v3 split reproduced: {len(tr68)} train / {len(va_tags)} val episodes")
    assert len(va_tags) == 36, f"val set moved! got {len(va_tags)}, expected 36"
    assert len(tr68) == 68, f"v3 train set moved! got {len(tr68)}, expected 68"
    assert all(t.startswith(("cm_", "pai_")) for t in va_tags), \
        "an extra episode leaked into the val split"

    # ---- 2. the expanded pool, with disjointness MEASURED -----------------
    all_tags = L.all_tags()
    extras = sorted(t for t in all_tags if t.startswith(("cmx_", "paix_")))
    log(f"extras linked in: {len(extras)}")
    val_fp = {fingerprint(t): t for t in va_tags}
    tr_fp = {}
    excluded, dup_in_pool = [], []
    seen = {}
    for t in extras:
        fp = fingerprint(t)
        if fp in val_fp:
            excluded.append({"extra": t, "collides_with_val": val_fp[fp]})
            continue
        if fp in seen:
            dup_in_pool.append({"extra": t, "duplicate_of": seen[fp]})
            continue
        seen[fp] = t
        tr_fp[t] = fp
    # the 68 v3 train episodes are in the pool by construction
    for t in tr68:
        fp = fingerprint(t)
        assert fp not in val_fp, f"v3 train/val collision on {t} — split is broken"
        if fp in seen:
            dup_in_pool.append({"extra": seen[fp], "duplicate_of": t})
            tr_fp.pop(seen[fp], None)
        seen[fp] = t
        tr_fp[t] = fp
    pool = sorted(tr_fp)
    log(f"EXCLUDED {len(excluded)} extras colliding with VAL: "
        f"{[e['extra'] for e in excluded]}")
    log(f"EXCLUDED {len(dup_in_pool)} duplicate extras")
    log(f"training pool = {len(pool)} episodes "
        f"(cm {sum(corpus_of(t) == 'cm' for t in pool)}, "
        f"pai {sum(corpus_of(t) == 'pai' for t in pool)})")
    assert not (set(pool) & set(va_tags)), "pool/val tag overlap"
    assert not ({tr_fp[t] for t in pool} & set(val_fp)), \
        "pool/val CONTENT overlap — episode-disjointness violated"

    # ---- 3. val set, built ONCE ------------------------------------------
    va = L.build_set(va_tags, k=A.KBUILD, stride=2, want_seq=True)
    nch = prep(va)
    log(f"val {tuple(va['Z'].shape)}  repaired {nch} yaw labels")
    n_val = int(va["S"].shape[0])

    # ---- 4. A0, on these exact windows ------------------------------------
    a0 = np.load(a.a0_preds, allow_pickle=True).item()
    assert a0["S"].shape[0] == n_val, (
        f"A0 preds are {a0['S'].shape[0]} windows but val is {n_val} — these "
        "are NOT the same windows; the paired read would be invalid")
    Gva = va["S"].numpy().astype(np.float64)
    eid = va["eid"]
    a0_ch = {nm: L.chan_metrics(a0["S"][:, j], Gva[:, j])
             for j, nm in enumerate(L.SCALARS)}
    a0_dom = {nm: {d: L.chan_metrics(a0["S"][va["dom"] == d, j],
                                     Gva[va["dom"] == d, j])
                   for d in ("pai", "cm")}
              for j, nm in enumerate(L.SCALARS)}
    log("A0 on these windows: steer R2 %+.4f (pai %+.4f cm %+.4f)" % (
        a0_ch["steer"]["r2"], a0_dom["steer"]["pai"]["r2"],
        a0_dom["steer"]["cm"]["r2"]))

    res = {
        "meta": {
            "purpose": "IDM_V3.md §9.4 — the owed steer data-budget retrain",
            "label_protocol": "REPAIRED (idm3_labels.heading_repair, v_min=0.5)",
            "recipe": "R0 (k=4 / 9 frames, d_model 256, no winsor, no clip-ctx)",
            "estimator": "taniteval.ci.(paired_)episode_cluster_bootstrap, "
                         "unit=episode, B=2000",
            "val_episodes": va_tags,
            "n_val_windows": n_val,
            "epochs": a.epochs, "seeds": a.seeds,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "disjointness": {
            "excluded_val_collisions": excluded,
            "excluded_duplicates": dup_in_pool,
            "pool_size": len(pool),
            "pool_cm": sum(corpus_of(t) == "cm" for t in pool),
            "pool_pai": sum(corpus_of(t) == "pai" for t in pool),
        },
        "a0_on_these_windows": {
            "channels": {k: v for k, v in a0_ch.items()},
            "per_domain": a0_dom,
        },
        "rungs": {},
    }

    ladder = nested_ladder(pool, a.rungs)
    # rung 68 is pinned to the EXACT v3 train episodes, so the bottom of the
    # ladder is the published control and not a resample of it
    if 68 in ladder:
        ladder[68] = sorted(tr68)

    base = dict(k=4, winsor=False, ctx=False, chan_w=[1, 1, 1, 1], dmodel=256)
    store = {}
    for r in sorted(ladder):
        tags = ladder[r]
        log(f"=== RUNG {r} episodes (cm {sum(corpus_of(t)=='cm' for t in tags)}"
            f" / pai {sum(corpus_of(t)=='pai' for t in tags)}) ===")
        tr = L.build_set(tags, k=A.KBUILD, stride=1, want_seq=True)
        prep(tr)
        log(f"  train windows {tuple(tr['Z'].shape)}")
        rr = {"n_episodes": len(tags),
              "n_cm": sum(corpus_of(t) == "cm" for t in tags),
              "n_pai": sum(corpus_of(t) == "pai" for t in tags),
              "n_train_windows": int(tr["S"].shape[0]),
              "episodes": tags, "seeds": {}}
        Ps = []
        for sd in a.seeds:
            P, meta, sdict = A.train_arm(base, tr, va, seed=sd, epochs=a.epochs)
            ev = A.eval_preds(P, va, va["S"])
            ev["meta"] = meta
            rr["seeds"][str(sd)] = ev
            Ps.append(P)
            c = ev["channels"]
            log("  rung %4d s%d  steer R2 %+.4f (pai %+.4f cm %+.4f) | "
                "speed %+.4f | yaw %+.4f" % (
                    r, sd, c["steer"]["r2"],
                    c["steer"]["per_domain"]["pai"]["r2"],
                    c["steer"]["per_domain"]["cm"]["r2"],
                    c["speed"]["r2"], c["yaw_rate"]["r2"]))
            if a.save_ckpt and r == max(ladder) and sd == a.seeds[0]:
                torch.save({"state_dict": sdict, "config": {"head_kwargs": meta.get("head_kwargs", {})},
                            "rung": r, "seed": sd}, a.save_ckpt)
                log(f"  saved ckpt -> {a.save_ckpt}")
        # seed-mean predictions, then the PAIRED read against A0
        Pm = np.mean([p["S"] for p in Ps], axis=0)
        store[r] = Pm
        rr["seed_mean"] = {}
        for j, nm in enumerate(L.SCALARS):
            m = L.chan_metrics(Pm[:, j], Gva[:, j])
            m["per_domain"] = {d: L.chan_metrics(Pm[va["dom"] == d, j],
                                                 Gva[va["dom"] == d, j])
                               for d in ("pai", "cm")}
            rr["seed_mean"][nm] = m
        rr["paired_vs_a0"] = {}
        for j, nm in enumerate(L.SCALARS):
            d = L.paired_mae(Pm[:, j], a0["S"][:, j], Gva[:, j], eid)
            rr["paired_vs_a0"][nm] = d
            # per corpus, because steer is NOT the same quantity across corpora
            rr["paired_vs_a0"][nm + "_per_domain"] = {
                dm: L.paired_mae(Pm[va["dom"] == dm, j],
                                 a0["S"][va["dom"] == dm, j],
                                 Gva[va["dom"] == dm, j], eid[va["dom"] == dm])
                for dm in ("pai", "cm")}
        st = rr["paired_vs_a0"]["steer"]
        log("  rung %4d SEED-MEAN steer R2 %+.4f | paired dMAE vs A0 %s"
            % (r, rr["seed_mean"]["steer"]["r2"], json.dumps(st)[:160]))
        res["rungs"][str(r)] = rr
        del tr
        L.jdump(res, a.out)      # bank incrementally

    res["meta"]["wall_s"] = round(time.time() - t_start, 1)
    L.jdump(res, a.out)
    log(f"IDM4_DONE {res['meta']['wall_s']}s -> {a.out}")


if __name__ == "__main__":
    main()
