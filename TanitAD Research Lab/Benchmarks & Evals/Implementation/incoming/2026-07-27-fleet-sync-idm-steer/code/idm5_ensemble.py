"""IDM v4 `steer` — SHIP THE 3-SEED ENSEMBLE (closes FLEET_REFILL.md §2.5 item 2).

The 2026-07-27 retrain established the number: rung-757 `steer` R2 **+0.7993**
against the deployed head's **+0.7419**, CI-separated on BOTH corpora
independently. But `idm4_steer.py:295` saves a checkpoint only for
`sd == a.seeds[0]`, so the artifact on disk is **SEED 0 ONLY** while the headline
is the **3-seed ensembled prediction**. A seed-0 file cannot produce the headline
number, and quoting the ensemble number for it would be a fabricated result.

This script closes exactly that gap:

1. Re-derives the rung-757 training pool with the SAME content-fingerprint leak
   check (md5 of float32 poses), and re-asserts the frozen 68/36 v3 split.
2. Trains seeds 0, 1, 2 at rung 757 and saves **all three** state_dicts.
3. Builds the ensemble = MEAN OF PER-SEED SCALAR PREDICTIONS (not a weight
   average — averaging the weights of independently-seeded transformers is not
   the estimator that produced the headline, and would not reproduce it).
4. ⭐ RELOADS the saved checkpoint from disk, re-predicts, and asserts the
   reloaded ensemble reproduces the training-time ensemble. An artifact that
   cannot reproduce its own headline is not shippable, and nothing else here
   proves it can.
5. Re-measures against A0 (the deployed `idm_head_v1`) on the SAME paired,
   episode-disjoint read: per corpus, paired episode-cluster bootstrap, B=2000.
6. Also reports the SEED-0-ONLY numbers, so the currently staged
   `idm_head_v4_steer.pt` can be described by what it actually scores.

ESTIMATOR PIN
-------------
`taniteval.ci.paired_episode_cluster_bootstrap`, unit = EPISODE, B = 2000.
`overlapping_holdout_se` is never called.
⚠️ pod3 carries TWO `taniteval` trees. `/root/taniteval` (the IDM line's copy)
and the repo's pinned tree differ in `ci.py`. This script imports the PINNED
tree explicitly and records both md5s plus the resolved `__file__`, so the
estimator that produced every number below is named, not assumed.

LABEL PROTOCOL
--------------
REPAIRED — `idm3_labels.heading_repair`, `v_min = 0.5 m/s`, extended to the
linked-in `cmx_` extras (`idm4_steer.repair_labels_ext`), i.e. byte-identical to
the protocol of every v3 arm and of the 2026-07-27 retrain. A `steer` number on
unrepaired labels is not comparable to one on repaired labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# ⛔ MEASURED 2026-07-27, and it is why this block looks the way it does.
#
# Putting the pinned harness at sys.path[0] IS NOT ENOUGH. `idm2_lib.py:19` and
# `idm3_arms.py` each run an unconditional
#     sys.path.insert(0, "/root/taniteval")
# at module import, which jumps the stale tree back in FRONT of any pin the
# caller set. First attempt here did exactly that and `taniteval.ci` resolved to
# `/root/taniteval/taniteval/ci.py` — silently, with no error.
#
# ⚠️ `python3 -m taniteval.stack_check --require v5` CANNOT catch this: it pins
# `tanitad`, not `taniteval`. This is STALE_IMPORT_GUARD.md §3.4 residual 1 /
# §8 escalation 3, observed in the field.
#
# The cure is the same one `TANITEVAL_STACK_OVERRIDE` uses for `tanitad`: import
# the pinned package FIRST so `sys.modules` caches it, because the module cache
# beats every later `sys.path.insert`. Then ASSERT the pin survived.
# --------------------------------------------------------------------------- #
_PINNED_TANITEVAL = os.environ.get("TANITEVAL_PINNED",
                                   "/workspace/TanitAD-main/taniteval")
sys.path.insert(0, _PINNED_TANITEVAL)
import taniteval             # noqa: E402  -> caches the PINNED package
import taniteval.ci as TCI   # noqa: E402
_PINNED_CI = TCI.__file__

for _p in (str(HERE), "/root/idm2", "/root/taniteval", "/root/idm3",
           "/root/v4eval/stack", "/root/v4eval/stack/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import idm2_lib as L            # noqa: E402
import idm3_arms as A           # noqa: E402

assert L.tci.__file__ == _PINNED_CI, (
    f"ESTIMATOR PIN DEFEATED: idm2_lib is using {L.tci.__file__}, not the "
    f"pinned {_PINNED_CI}. Every interval below would come from an unpinned "
    "tree — refusing.")
assert TCI.__file__ == _PINNED_CI


RUNG = 757
CFG = dict(k=4, winsor=False, ctx=False, chan_w=[1, 1, 1, 1], dmodel=256)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def md5_file(p):
    try:
        return hashlib.md5(Path(p).read_bytes()).hexdigest()
    except Exception:                                         # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# identity — copied verbatim from idm4_steer so the pool is the SAME pool       #
# --------------------------------------------------------------------------- #
def fingerprint(tag: str) -> str:
    po = L.load_ep(tag)["poses"].float().numpy().astype(np.float32)
    return hashlib.md5(np.ascontiguousarray(po).tobytes()).hexdigest()


def corpus_of(tag: str) -> str:
    return "cm" if tag.startswith(("cm_", "cmx_")) else "pai"


def repair_labels_ext(setd, v_min=A.V_MIN):
    """`idm3_arms.repair_labels` treating `cmx_` as comma2k19 — verbatim from
    idm4_steer.repair_labels_ext. Stock tests `tag.startswith("cm_")`, False for
    the linked-in extras, which would leave 79 comma episodes on the BROKEN
    arctan2-at-standstill heading: a mixed label protocol inside one train set."""
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
    setd["Akin"] = ((setd["Vseq"][:, A.KBUILD + 1] - setd["Vseq"][:, A.KBUILD - 1])
                    / (2 * A.ih.DT))
    setd["S_leg"] = setd["S"].clone()
    setd["S"], nch = repair_labels_ext(setd)
    setd["dom"] = np.array([corpus_of(str(t)) for t in setd["eid"]])
    return nch


def build_head(state_dim, cfg=CFG):
    """The R0 head, rebuilt from explicit kwargs.

    ⚠️ The 2026-07-27 checkpoint stored `config.head_kwargs` from
    `meta.get("head_kwargs", {})`, but `train_arm` never puts `head_kwargs` in
    `meta` — so that field is `{}` and the file does NOT record how to rebuild
    its own head. This function's kwargs are written into the new checkpoint."""
    return A.IDMHeadV3(state_dim=state_dim, window=2 * cfg["k"] + 1,
                       d_model=cfg["dmodel"], use_ctx=cfg["ctx"],
                       side_dim=0, acc_bins=0)


def head_kwargs(state_dim, cfg=CFG):
    return {"state_dim": int(state_dim), "window": 2 * cfg["k"] + 1,
            "d_model": cfg["dmodel"], "use_ctx": bool(cfg["ctx"]),
            "side_dim": 0, "acc_bins": 0,
            "input_slice": [A.KBUILD - cfg["k"], A.KBUILD + cfg["k"] + 1],
            "class": "idm3_arms.IDMHeadV3"}


def metrics_block(P_S, Gva, dom):
    out = {}
    for j, nm in enumerate(L.SCALARS):
        m = L.chan_metrics(P_S[:, j], Gva[:, j])
        m["per_domain"] = {d: L.chan_metrics(P_S[dom == d, j], Gva[dom == d, j])
                           for d in ("pai", "cm")}
        out[nm] = m
    return out


def paired_block(P_S, a0_S, Gva, eid, dom):
    """PAIRED episode-cluster bootstrap vs A0. Per corpus, never pooled alone."""
    out = {}
    for j, nm in enumerate(L.SCALARS):
        out[nm] = L.paired_mae(P_S[:, j], a0_S[:, j], Gva[:, j], eid)
        out[nm + "_per_domain"] = {
            d: L.paired_mae(P_S[dom == d, j], a0_S[dom == d, j],
                            Gva[dom == d, j], eid[dom == d])
            for d in ("pai", "cm")}
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default="/workspace/idmretrain/out/idm5_ensemble.json")
    ap.add_argument("--a0-preds",
                    default="/workspace/idmretrain/idm3/out/a0_preds.npy")
    ap.add_argument("--ckpt",
                    default="/workspace/idmretrain/out/idm_head_v4_steer_ens3.pt")
    ap.add_argument("--prior-json",
                    default="/workspace/idmretrain/out/idm4_steer.json")
    a = ap.parse_args()

    t_start = time.time()
    log(f"estimator pinned -> {TCI.__file__}")

    # ---- 1. the FROZEN v3 split ------------------------------------------- #
    tr68, va_tags = L.split_tags()
    assert len(va_tags) == 36, f"val set moved! got {len(va_tags)}"
    assert len(tr68) == 68, f"v3 train set moved! got {len(tr68)}"
    assert all(t.startswith(("cm_", "pai_")) for t in va_tags), \
        "an extra episode leaked into the val split"
    log(f"v3 split reproduced: {len(tr68)} train / {len(va_tags)} val")

    # ---- 2. the pool, with LEAK MEASURED BY CONTENT ------------------------ #
    all_tags = L.all_tags()
    extras = sorted(t for t in all_tags if t.startswith(("cmx_", "paix_")))
    val_fp = {fingerprint(t): t for t in va_tags}
    tr_fp, excluded, dup_in_pool, seen = {}, [], [], {}
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
    for t in tr68:
        fp = fingerprint(t)
        assert fp not in val_fp, f"v3 train/val collision on {t}"
        if fp in seen:
            dup_in_pool.append({"extra": seen[fp], "duplicate_of": t})
            tr_fp.pop(seen[fp], None)
        seen[fp] = t
        tr_fp[t] = fp
    pool = sorted(tr_fp)
    assert not (set(pool) & set(va_tags)), "pool/val TAG overlap"
    assert not ({tr_fp[t] for t in pool} & set(val_fp)), \
        "pool/val CONTENT overlap — episode-disjointness violated"
    assert len(pool) == RUNG, f"pool is {len(pool)}, expected {RUNG}"
    log(f"LEAK CHECK: {len(excluded)} of {len(va_tags)} val episodes found in the "
        f"pool by CONTENT and excluded -> {[e['extra'] for e in excluded]}")
    log(f"pool = {len(pool)} (cm {sum(corpus_of(t)=='cm' for t in pool)} / "
        f"pai {sum(corpus_of(t)=='pai' for t in pool)}); "
        f"{len(dup_in_pool)} internal duplicates dropped")

    # ---- 3. val, built once ------------------------------------------------ #
    va = L.build_set(va_tags, k=A.KBUILD, stride=2, want_seq=True)
    nch_va = prep(va)
    n_val = int(va["S"].shape[0])
    log(f"val {tuple(va['Z'].shape)}  repaired {nch_va} yaw labels")

    a0 = np.load(a.a0_preds, allow_pickle=True).item()
    assert a0["S"].shape[0] == n_val, (
        f"A0 preds are {a0['S'].shape[0]} windows but val is {n_val} — NOT the "
        "same windows; the paired read would be invalid")
    Gva = va["S"].numpy().astype(np.float64)
    eid, dom = va["eid"], va["dom"]
    a0_m = metrics_block(a0["S"], Gva, dom)
    log("A0 on these windows: steer R2 %+.4f (pai %+.4f cm %+.4f)" % (
        a0_m["steer"]["r2"], a0_m["steer"]["per_domain"]["pai"]["r2"],
        a0_m["steer"]["per_domain"]["cm"]["r2"]))

    res = {
        "meta": {
            "purpose": "FLEET_REFILL.md §2.5 item 2 — ship the 3-SEED ENSEMBLE, "
                       "not the seed-0-only checkpoint",
            "rung": RUNG, "seeds": a.seeds, "epochs": a.epochs,
            "recipe": "R0 (k=4 / 9 frames, d_model 256, no winsor, no clip-ctx)",
            "label_protocol": "REPAIRED (idm3_labels.heading_repair, v_min=0.5, "
                              "extended to cmx_ extras)",
            "ensemble_rule": "MEAN OF PER-SEED SCALAR PREDICTIONS (not a weight "
                             "average)",
            "estimator": "taniteval.ci.(paired_)episode_cluster_bootstrap, "
                         "unit=episode, B=2000",
            "estimator_file": TCI.__file__,
            "estimator_file_used_by_idm2_lib": L.tci.__file__,
            "estimator_md5_pinned": md5_file(TCI.__file__),
            "estimator_md5_idmline_stale_tree": md5_file(
                "/root/taniteval/taniteval/ci.py"),
            "estimator_pin_note":
                "idm2_lib.py:19 and idm3_arms.py each sys.path.insert(0, "
                "'/root/taniteval') at import, which defeats a sys.path pin. The "
                "pinned package is imported FIRST so sys.modules caches it, and "
                "the pin is ASSERTED. stack_check --require v5 cannot catch this "
                "class — it pins `tanitad`, not `taniteval`.",
            "latent_dir_checked_for_leak": str(L.LAT),
            "val_episodes": list(va_tags),
            "n_val_windows": n_val,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "leak_check": {
            "method": "md5 of float32 poses per episode (content, NOT filename)",
            "path_checked": str(L.LAT),
            "n_candidate_extras": len(extras),
            "n_val_episodes": len(va_tags),
            "n_val_episodes_found_in_pool_and_excluded": len(excluded),
            "excluded_val_collisions": excluded,
            "n_internal_duplicates_dropped": len(dup_in_pool),
            "pool_size": len(pool),
            "pool_cm": sum(corpus_of(t) == "cm" for t in pool),
            "pool_pai": sum(corpus_of(t) == "pai" for t in pool),
            "residual_content_overlap": 0,
        },
        "a0_on_these_windows": a0_m,
        "seeds": {},
    }
    L.jdump(res, a.out)

    # ---- 4. train every seed, KEEP EVERY CHECKPOINT ------------------------ #
    tr = L.build_set(pool, k=A.KBUILD, stride=1, want_seq=True)
    prep(tr)
    res["meta"]["n_train_windows"] = int(tr["S"].shape[0])
    res["meta"]["train_episodes"] = pool
    log(f"train windows {tuple(tr['Z'].shape)}")

    Ps, sds = [], []
    for sd in a.seeds:
        P, meta, sdict = A.train_arm(CFG, tr, va, seed=sd, epochs=a.epochs)
        Ps.append(P)
        sds.append(sdict)
        ev = metrics_block(P["S"], Gva, dom)
        res["seeds"][str(sd)] = {"channels": ev, "train_s": meta.get("train_s"),
                                 "params": meta.get("params")}
        log("  seed %d  steer R2 %+.4f (pai %+.4f cm %+.4f) | speed %+.4f | "
            "yaw %+.4f" % (sd, ev["steer"]["r2"],
                           ev["steer"]["per_domain"]["pai"]["r2"],
                           ev["steer"]["per_domain"]["cm"]["r2"],
                           ev["speed"]["r2"], ev["yaw_rate"]["r2"]))
        L.jdump(res, a.out)

    state_dim = int(tr["Z"].shape[-1])
    Pm = np.mean([p["S"] for p in Ps], axis=0)

    # ---- 5. SAVE the ensemble, then RELOAD it and prove it reproduces ------ #
    hk = head_kwargs(state_dim)
    torch.save({
        "kind": "idm_head_v4_steer_ens3",
        "seeds": list(a.seeds),
        "state_dicts": sds,
        "head_kwargs": hk,
        "cfg": CFG,
        "rung": RUNG,
        "ensemble_rule": "mean of the per-seed `scalars` outputs",
        "scalars": list(L.SCALARS),
        "label_protocol": "REPAIRED (heading_repair, v_min=0.5)",
        "provenance": "FLEET_REFILL.md §2.5 / idm5_ensemble.py",
    }, a.ckpt)
    log(f"saved ensemble ckpt -> {a.ckpt} "
        f"({Path(a.ckpt).stat().st_size / 1e6:.1f} MB)")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sl = slice(hk["input_slice"][0], hk["input_slice"][1])
    Zva = va["Z"][:, sl].to(A.DEV).float()
    P_re = []
    for sdict in ck["state_dicts"]:
        h = build_head(ck["head_kwargs"]["state_dim"]).to(A.DEV)
        h.load_state_dict(sdict)
        P_re.append(A.predict(h, Zva, None, None, CFG)["S"])
        del h
    torch.cuda.empty_cache()
    Pm_re = np.mean(P_re, axis=0)
    dmax = float(np.abs(Pm_re - Pm).max())
    res["reload_verification"] = {
        "max_abs_delta_ensemble_pred": dmax,
        "reproduces": bool(dmax < 1e-8),
        "n_state_dicts_in_ckpt": len(ck["state_dicts"]),
        "ckpt": a.ckpt,
        "ckpt_bytes": int(Path(a.ckpt).stat().st_size),
        "ckpt_md5": md5_file(a.ckpt),
    }
    log(f"RELOAD CHECK: max|delta| = {dmax:.3e} -> "
        f"{'REPRODUCES' if dmax < 1e-8 else 'DOES NOT REPRODUCE'}")
    assert dmax < 1e-8, "the saved ensemble does NOT reproduce its own headline"

    # ---- 6. the reads ------------------------------------------------------ #
    res["ensemble_3seed"] = {
        "channels": metrics_block(Pm, Gva, dom),
        "paired_vs_a0": paired_block(Pm, a0["S"], Gva, eid, dom),
    }
    res["seed0_only"] = {
        "note": "what the currently staged idm_head_v4_steer.pt actually scores "
                "— the ensemble number may NOT be quoted for it",
        "channels": metrics_block(Ps[0]["S"], Gva, dom),
        "paired_vs_a0": paired_block(Ps[0]["S"], a0["S"], Gva, eid, dom),
    }
    e = res["ensemble_3seed"]
    log("ENSEMBLE steer R2 %+.4f (pai %+.4f cm %+.4f)" % (
        e["channels"]["steer"]["r2"],
        e["channels"]["steer"]["per_domain"]["pai"]["r2"],
        e["channels"]["steer"]["per_domain"]["cm"]["r2"]))
    log("ENSEMBLE paired dMAE vs A0  pai %s" %
        json.dumps(e["paired_vs_a0"]["steer_per_domain"]["pai"])[:200])
    log("ENSEMBLE paired dMAE vs A0  cm  %s" %
        json.dumps(e["paired_vs_a0"]["steer_per_domain"]["cm"])[:200])

    # ---- 7. determinism vs the 2026-07-27 run ------------------------------ #
    try:
        prior = json.load(open(a.prior_json))
        pr = prior["rungs"][str(RUNG)]
        res["reproduction_of_2026_07_27"] = {
            "prior_seed_mean_steer_r2": pr["seed_mean"]["steer"]["r2"],
            "here_ensemble_steer_r2": e["channels"]["steer"]["r2"],
            "delta": e["channels"]["steer"]["r2"] - pr["seed_mean"]["steer"]["r2"],
            "prior_per_seed_steer_r2": {
                k: v["channels"]["steer"]["r2"] for k, v in pr["seeds"].items()},
            "here_per_seed_steer_r2": {
                k: v["channels"]["steer"]["r2"] for k, v in res["seeds"].items()},
        }
        log("REPRODUCTION: prior seed-mean %+.4f vs here %+.4f (delta %+.6f)" % (
            pr["seed_mean"]["steer"]["r2"], e["channels"]["steer"]["r2"],
            res["reproduction_of_2026_07_27"]["delta"]))
    except Exception as ex:                                   # noqa: BLE001
        res["reproduction_of_2026_07_27"] = {"error": repr(ex)}

    res["meta"]["wall_s"] = round(time.time() - t_start, 1)
    L.jdump(res, a.out)
    log(f"IDM5_DONE {res['meta']['wall_s']}s -> {a.out}")


if __name__ == "__main__":
    main()
