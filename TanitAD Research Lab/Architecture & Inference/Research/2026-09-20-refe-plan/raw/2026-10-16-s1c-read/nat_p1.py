#!/usr/bin/env python3
"""nat_p1.py -- the NATURAL-navtest-population P1 reader of the aux drivable head (PREREG_S1C section 3; weights from raw/2026-10-16-aux-shift-locate/b1_decompose.py S5).

WHY.  The C3 set (1,058 tokens) is failure-ENRICHED (529 u400-zeroed + 529 random) and its city mix is not navtest's.  SHIFT_LOCATE (MEASURED) showed that S1's heads read
0.496 / 0.521 / 0.486 on it and 0.561 / 0.577 / 0.554 once the SAME predictions are re-weighted to the natural navtest population.  PREREG_S1C reads K2 and the budget contrast on BOTH.

WHAT.  Per-frame sufficient statistics (inter / union of the K2 band [0, 40) m) come from the PROGRAMME'S OWN `aux_s1_readout.frame_stats` (not re-derived).  A frame weight
w_i is the iterative-proportional-fit of the 1,058 frames to TWO margins at once -- b1_decompose.rake, copied verbatim (200 iterations, map margin then stratum margin):
    map margin      = the share of each city (LV / BOS / PIT / SG) among ALL 12,146 navtest keys, read from the navtest label store's shard headers;
    stratum margin  = {zero: 529/12,146 = 4.355 %, random: the rest} (tokens.json provenance: n_zero_NCxDAC / n_csv_tokens).
    P1_nat = sum_i w_i * inter_i / sum_i w_i * union_i.   ESS = (sum w)^2 / sum w^2 is reported.   The weights depend on the FRAMES only, never on the model.

ESTIMATOR.  The same one as the harness (aux_s1_readout.boot_weights / ratio_read / diff_read): 10,000 resamples of the 123 LOGS, seed 20260927, 95 % percentile, models held fixed,
a contrast paired on the same resamples.  PRIMARY interval: the weights are FIXED at their point-estimate values.  SENSITIVITY (reported, never gating): the weights are re-fitted inside
every resample to the SAME population margins (b2_standardised_gap_ci's scheme).  It answers "would another draw of LOGS say this?" ONLY: not another training run (H-ESTIM-SEED-1), not
another inference run (the head is deterministic: ASSERTED).  Tier: T0-like (head perception on teacher-forced navtest frames), never driving performance.

IDENTITY (always run when an arm is one of S1's H1-s0 / H1-s1 / H1-s2).  The reader must reproduce SHIFT_LOCATE's banked natural reads 0.56085203693 / 0.57683017391 / 0.55393218323
(rounded 0.561 / 0.577 / 0.554) and ESS 575.3679 to 1e-9, else exit 6 `ZZNATP1_IDENTITY_FAILED`.  The population margins are cross-checked against SHIFT_LOCATE.json's `targets` to 1e-12.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I nat_p1.py --pred H1u10k-s0=head_pred_H1u10k-s0.npz H1-s0=head_pred_H1-s0.npz --contrast H1u10k-s0:H1-s0 --out nat_p1.json
Exit: 0 ok | 3 refused (input guard) | 6 identity failed.   Marker lines: ZZNATP1_OK / ZZNATP1_REFUSED / ZZNATP1_IDENTITY_FAILED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PK = HERE.parent.parent                                  # <plan root>
for _p in (PK / "eval", PK / "refe"):                    # explicit and read-only: the programme's own harness (-I ignores PYTHONPATH)
    sys.path.insert(0, str(_p))
import aux_s1_readout as RO  # noqa: E402  -- the harness: frame_stats, load_head_pred, boot_weights, ratio_read, diff_read, by_log

TOOL = "nat_p1/1"
TIER = "T0-like: head perception on teacher-forced navtest frames, re-weighted to the natural navtest population; never driving performance"
K2_BAND_INDEX = 3                                        # aux_s1_readout.frame_stats: bands = BANDS + [K2_BAND, None]; index 3 = (0, 40)
assert RO.K2_BAND == (0.0, 40.0) and len(RO.BANDS) == 3
IPF_ITERS, IPF_ITERS_BOOT = 200, 60                      # b1_decompose.rake default / b2_standardised_gap_ci.rake_from
IPF_TOL = 1e-9
N_BOOT, SEED = RO.N_BOOT, RO.SEED                        # 10,000 / 20260927
MAPS = ("LV", "BOS", "PIT", "SG")
MAP_SHORT = {"us-nv-las-vegas-strip": "LV", "us-ma-boston": "BOS", "us-pa-pittsburgh-hazelwood": "PIT", "sg-one-north": "SG"}
C3_DIR = PK / "raw" / "2026-10-11-c3-selection-headroom"
SHIFT_DIR = PK / "raw" / "2026-10-16-aux-shift-locate"

# SHIFT_LOCATE.json `targets` (MEASURED 2026-10-16): the population margins the weights are fitted to, frozen so a drifting label store is seen, not absorbed.
FROZEN_SHARE_MAP = {"BOS": 0.30717931829408857, "LV": 0.3345957516877984, "PIT": 0.20064218672814096, "SG": 0.15758274328997202}
FROZEN_SHARE_STRATUM = {"random": 0.9564465667709534, "zero": 0.0435534332290466}
FROZEN_POP = {"BOS": 3731, "LV": 4064, "PIT": 2437, "SG": 1914}            # 12,146 navtest keys (shard headers)
# SHIFT_LOCATE.json S5_standardisation[arm].navtest_natural_population / ess_natural, and R0_navtest_repro (the unweighted C3 read), MEASURED:
BANKED_NATURAL = {"H1-s0": 0.5608520369346311, "H1-s1": 0.5768301739064818, "H1-s2": 0.5539321832326178}
BANKED_C3 = {"H1-s0": 0.4957035913034702, "H1-s1": 0.5210982139151368, "H1-s2": 0.4863984780737079}
BANKED_ESS = 575.3679059904473


class NatRefused(RuntimeError):
    """An input guard failed (population margins drifted, frames missing a label / stratum / city, ...). Nothing is read."""


# ================================================================================================================ the weights (b1_decompose.rake, verbatim)
def rake(maps, strata, margins_map, margins_stratum, iters=IPF_ITERS):
    """iterative proportional fitting of frame weights to a map margin and a stratum margin (shares sum to 1 each).  b1_decompose.py:rake, byte for byte in behaviour."""
    w = np.ones(len(maps))
    for _ in range(iters):
        for lab, marg in ((maps, margins_map), (strata, margins_stratum)):
            if marg is None:
                continue
            tot = w.sum()
            for k, share in marg.items():
                m = lab == k
                s = w[m].sum()
                if s > 0:
                    w[m] *= share * tot / s
    return w / w.mean()


def ess(w) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum())


def ipf_rows(M, groupings, iters=IPF_ITERS_BOOT):
    """The same IPF, applied to every ROW of M [B, N] at once (row b = the frame multiplicities of resample b).  groupings = [(group index per frame [N], shares [G]), ...].
    A group absent from a resample (column sum 0) is skipped exactly as `rake` skips it.  Returns (M_fitted, max abs residual of the margins over resamples that have every group)."""
    M = np.array(M, np.float64, copy=True)
    for _ in range(iters):
        for gix, shares in groupings:
            G = len(shares)
            onehot = np.zeros((M.shape[1], G))
            onehot[np.arange(M.shape[1]), gix] = 1.0
            tot = M.sum(1, keepdims=True)
            S = M @ onehot
            with np.errstate(invalid="ignore", divide="ignore"):
                scale = np.where(S > 0, shares[None, :] * tot / S, 1.0)
            M *= scale[:, gix]
    resid = 0.0
    for gix, shares in groupings:
        G = len(shares)
        onehot = np.zeros((M.shape[1], G))
        onehot[np.arange(M.shape[1]), gix] = 1.0
        S = M @ onehot
        full = (S > 0).all(1)
        if full.any():
            resid = max(resid, float(np.abs(S[full] / M[full].sum(1, keepdims=True) - shares[None, :]).max()))
    return M, resid


# ================================================================================================================ the population (label store shard headers + tokens.json)
def navtest_population(store_dir: str | None = None) -> dict:
    """-> {"log_map": {log: short map}, "count_by_map": {map: keys}, "n": total keys}.  Read from the navtest label store's shard headers (the store the harness pins by sha256)."""
    import aux_label_store as ALS
    store = Path(store_dir or RO.LABEL_DIRS["navtest"])
    shards = sorted((store / "shards").glob("*.aux1"))
    if not shards:
        raise NatRefused(f"no shards under {store / 'shards'}")
    log_map, count = {}, {}
    for f in shards:
        head, _ = ALS.read_shard(f, with_blobs=False)
        m = MAP_SHORT.get(head["map"])
        if m is None:
            raise NatRefused(f"{f.name}: unknown map {head['map']!r}")
        log_map[head["log"]] = m
        count[m] = count.get(m, 0) + len(head["entries"])
    return {"log_map": log_map, "count_by_map": count, "n": int(sum(count.values())), "n_shards": len(shards)}


def build_spec(tokens_json: dict, pop: dict, frozen_check: bool = True) -> dict:
    """The natural-population targets.  `frozen_check` asserts them against SHIFT_LOCATE's banked `targets` (1e-12); tests of the maths turn it off."""
    prov = tokens_json["provenance"]
    n_pop, n_zero = int(prov["n_csv_tokens"]), int(prov["n_zero_NCxDAC"])
    if pop["n"] != n_pop:
        raise NatRefused(f"the navtest store holds {pop['n']} keys, tokens.json provenance says the navtest population is {n_pop}")
    share_map = {m: pop["count_by_map"][m] / pop["n"] for m in MAPS}
    share_stratum = {"zero": n_zero / n_pop, "random": 1 - n_zero / n_pop}
    spec = {"share_map": share_map, "share_stratum": share_stratum, "n_population": n_pop, "n_zero": n_zero, "count_by_map": dict(pop["count_by_map"])}
    if frozen_check:
        bad = {k: (share_map[k], FROZEN_SHARE_MAP[k]) for k in MAPS if abs(share_map[k] - FROZEN_SHARE_MAP[k]) > 1e-12}
        bad.update({k: (share_stratum[k], FROZEN_SHARE_STRATUM[k]) for k in FROZEN_SHARE_STRATUM if abs(share_stratum[k] - FROZEN_SHARE_STRATUM[k]) > 1e-12})
        if bad or dict(pop["count_by_map"]) != FROZEN_POP:
            raise NatRefused(f"the population margins differ from SHIFT_LOCATE's banked targets: {bad or pop['count_by_map']}")
    return spec


def frame_labels(keys, tokens_json: dict, log_map: dict):
    """-> (maps [N], strata [N]) of the frames, in `keys` order.  A frame with no stratum or no city is REFUSED (a silent default would move the weights)."""
    strata, maps = [], []
    for log, tok in keys:
        s = tokens_json["stratum"].get(tok)
        m = log_map.get(log)
        if s not in ("zero", "random") or m is None:
            raise NatRefused(f"frame ({log}, {tok}): stratum {s!r}, city {m!r}")
        strata.append(s)
        maps.append(m)
    return np.asarray(maps), np.asarray(strata)


def natural_weights(maps, strata, spec) -> np.ndarray:
    return rake(maps, strata, spec["share_map"], spec["share_stratum"], IPF_ITERS)


# ================================================================================================================ the reads
def pooled(inter, union, w=None, m=None) -> float:
    """b1_decompose.iou, verbatim: ratio of (weighted) sums over the (optionally masked) frames."""
    m = np.ones(len(inter), bool) if m is None else m
    w = np.ones(len(inter)) if w is None else w
    d = (w[m] * union[m]).sum()
    return float((w[m] * inter[m]).sum() / d) if d > 0 else float("nan")


def arm_stats(pred: dict, source) -> dict:
    """Per-frame K2-band sufficient statistics from the harness's own frame_stats; every frame must be labelled."""
    st = RO.frame_stats(pred, source)
    if not bool(st["valid"].all()):
        raise NatRefused(f"{int((~st['valid']).sum())} frames have no navtest label")
    return {"inter": st[f"inter{K2_BAND_INDEX}"], "union": st[f"union{K2_BAND_INDEX}"], "cells": st[f"cells{K2_BAND_INDEX}"]}


def _logs_index(keys):
    ul = sorted(set(k[0] for k in keys))
    pos = {l: i for i, l in enumerate(ul)}
    return ul, np.asarray([pos[k[0]] for k in keys], np.int64)


def read_population(keys, stats: dict, weights: np.ndarray | None, W: np.ndarray) -> dict:
    """P1 [0, 40) over the frames (weights=None: the C3 set as it is; else the natural population) with the harness's log-cluster bootstrap on the resample matrix W."""
    ul, lix = _logs_index(keys)
    w = np.ones(len(keys)) if weights is None else weights
    num_l = np.bincount(lix, weights=w * stats["inter"], minlength=len(ul))
    den_l = np.bincount(lix, weights=w * stats["union"], minlength=len(ul))
    r = RO.ratio_read(num_l, den_l, W)
    est = pooled(stats["inter"], stats["union"], weights)
    if abs(est - r["est"]) > 1e-12:
        raise NatRefused(f"pooled estimate {est} != ratio of per-log sums {r['est']}")
    return {"est": est, "ci95": r["ci95"], "n_union_weighted": r["n_den"], "n_nan_resamples": r["n_nan_resamples"]}


def read_contrast(keys, st_a: dict, st_b: dict, weights: np.ndarray | None, W: np.ndarray) -> dict:
    """arm A minus arm B on the SAME frames and the SAME log resamples (the harness's diff_read)."""
    ul, lix = _logs_index(keys)
    w = np.ones(len(keys)) if weights is None else weights
    a = [np.bincount(lix, weights=w * st_a[k], minlength=len(ul)) for k in ("inter", "union")]
    b = [np.bincount(lix, weights=w * st_b[k], minlength=len(ul)) for k in ("inter", "union")]
    d = RO.diff_read(a[0], a[1], b[0], b[1], W)
    d["est"] = pooled(st_a["inter"], st_a["union"], weights) - pooled(st_b["inter"], st_b["union"], weights)
    return d


def reraked_reps(keys, maps, strata, spec, W: np.ndarray, stats_list: list) -> tuple:
    """SENSITIVITY: the weights re-fitted INSIDE every resample to the same population margins.  -> ([reps of P1 per stats in stats_list], max margin residual, n resamples with an absent group)."""
    ul, lix = _logs_index(keys)
    M = W[:, lix]                                                # [B, N] frame multiplicities of each resample
    mg = np.asarray([MAPS.index(m) for m in maps])
    sg = np.asarray([0 if s == "zero" else 1 for s in strata])
    groupings = [(mg, np.asarray([spec["share_map"][m] for m in MAPS])), (sg, np.asarray([spec["share_stratum"]["zero"], spec["share_stratum"]["random"]]))]
    Mf, resid = ipf_rows(M, groupings, IPF_ITERS_BOOT)
    absent = 0
    for gix, shares in groupings:
        onehot = np.zeros((M.shape[1], len(shares)))
        onehot[np.arange(M.shape[1]), gix] = 1.0
        absent = max(absent, int((~((M @ onehot) > 0).all(1)).sum()))
    reps = []
    for st in stats_list:
        with np.errstate(invalid="ignore", divide="ignore"):
            reps.append((Mf * st["inter"][None, :]).sum(1) / (Mf * st["union"][None, :]).sum(1))
    return reps, resid, absent


def _ci(reps):
    ci, n_nan = RO._ci(reps)
    return {"ci95": ci, "n_nan_resamples": n_nan}


def per_city(keys, maps, stats, w) -> dict:
    out = {}
    for m in MAPS:
        k = maps == m
        out[m] = {"n_frames": int(k.sum()), "n_logs": len(set(np.asarray([x[0] for x in keys])[k])), "p1_natural_weighted": pooled(stats["inter"], stats["union"], w, k),
                  "p1_unweighted": pooled(stats["inter"], stats["union"], None, k)}
    return out


def identity_block(arm: str, c3_est: float, nat_est: float, ess_v: float) -> dict | None:
    if arm not in BANKED_NATURAL:
        return None
    d_nat, d_c3, d_ess = abs(nat_est - BANKED_NATURAL[arm]), abs(c3_est - BANKED_C3[arm]), abs(ess_v - BANKED_ESS)
    return {"arm": arm, "banked_natural": BANKED_NATURAL[arm], "natural": nat_est, "abs_diff_natural": d_nat, "banked_c3": BANKED_C3[arm], "c3": c3_est, "abs_diff_c3": d_c3,
            "banked_ess": BANKED_ESS, "ess": ess_v, "abs_diff_ess": d_ess, "tolerance": IPF_TOL, "ok": bool(d_nat <= IPF_TOL and d_c3 <= IPF_TOL and d_ess <= 1e-6),
            "source": "raw/2026-10-16-aux-shift-locate/SHIFT_LOCATE.json (b1_decompose.json S5_standardisation, R0_navtest_repro), MEASURED"}


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(preds: dict, contrasts: list, tokens_path: Path, out_path: str | None, n_boot: int = N_BOOT, seed: int = SEED, sensitivity: bool = True,
        store_dir: str | None = None, source=None, frozen_check: bool = True, leak_guard: bool = True) -> dict:
    tokens_json = json.load(open(tokens_path, encoding="utf-8"))
    pop = navtest_population(store_dir)
    spec = build_spec(tokens_json, pop, frozen_check=frozen_check)
    source = source if source is not None else RO.open_labels("navtest")
    loaded = {a: RO.load_head_pred(p) for a, p in preds.items()}
    names = sorted(loaded)
    keys = loaded[names[0]]["keys"]
    for a in names:
        if loaded[a]["keys"] != keys:
            raise NatRefused(f"{a}: (log, token) keys or their order differ from {names[0]}'s")
    if len(set(k[1] for k in keys)) != len(keys) or set(k[1] for k in keys) != set(tokens_json["tokens"]):
        raise NatRefused("the predictions do not cover exactly the C3 tokens of tokens.json")
    if leak_guard:                                              # the harness's own guard: no evaluated key / log may be in the head's training keys
        RO.assert_disjoint(keys, RO.train_keys_for("navtest", RO.open_labels("train")))
    maps, strata = frame_labels(keys, tokens_json, pop["log_map"])
    w = natural_weights(maps, strata, spec)
    ul, _ = _logs_index(keys)
    W = RO.boot_weights(len(ul), n_boot, seed)
    stats = {a: arm_stats(loaded[a], source) for a in names}
    res = {"tool": TOOL, "tier": TIER, "band_m": [0.0, 40.0],
           "estimator": f"paired log-cluster bootstrap, {n_boot} resamples, 95 %, seed {seed}; aux_s1_readout.boot_weights / ratio_read / diff_read (the harness's own); weights FIXED at their point-estimate "
                        f"values (PRIMARY); sensitivity re-fits them inside every resample. Answers 'another draw of LOGS' ONLY",
           "population": {"definition": "the 1,058 C3 frames re-weighted by IPF (200 it.) to navtest's own city mix (all 12,146 keys) x the failure stratum (529 / 12,146)",
                          "spec": spec, "ess": ess(w), "n_frames": len(keys), "n_logs": len(ul), "weight_min": float(w.min()), "weight_max": float(w.max()),
                          "weight_by_cell": {f"{m}|{s}": float(w[(maps == m) & (strata == s)].mean()) for m in MAPS for s in ("zero", "random") if ((maps == m) & (strata == s)).any()},
                          "n_by_cell": {f"{m}|{s}": int(((maps == m) & (strata == s)).sum()) for m in MAPS for s in ("zero", "random")},
                          "sources": {"tokens_json_sha256": sha256_file(tokens_path), "navtest_store_shards": pop["n_shards"], "label_store": RO.LABEL_PINS["navtest"][0]}},
           "arms": {}, "contrasts": {}, "identity": {}}
    rer_cache = {}
    if sensitivity:
        reps, resid, absent = reraked_reps(keys, maps, strata, spec, W, [stats[a] for a in names])
        rer_cache = dict(zip(names, reps))
        res["sensitivity"] = {"scheme": f"weights re-fitted in every resample ({IPF_ITERS_BOOT} IPF iterations from the resample's frame multiplicities) to the SAME population margins",
                              "max_margin_residual": resid, "resamples_with_an_absent_group": absent, "gating": False}
    for a in names:
        c3 = read_population(keys, stats[a], None, W)
        nat = read_population(keys, stats[a], w, W)
        blk = {"c3": c3, "natural": nat, "per_city": per_city(keys, maps, stats[a], w), "input_sha256": None}
        if sensitivity:
            blk["natural"]["sensitivity_rerake"] = _ci(rer_cache[a])
        idn = identity_block(a, c3["est"], nat["est"], ess(w))
        if idn:
            res["identity"][a] = idn
        res["arms"][a] = blk
    for c in contrasts:
        a, b = c
        if a not in stats or b not in stats:
            raise NatRefused(f"contrast {a}:{b} names an arm that was not given")
        blk = {"c3": read_contrast(keys, stats[a], stats[b], None, W), "natural": read_contrast(keys, stats[a], stats[b], w, W)}
        if sensitivity:
            d = rer_cache[a] - rer_cache[b]
            blk["natural"]["sensitivity_rerake"] = _ci(d)
        res["contrasts"][f"{a}-minus-{b}"] = blk
    res["identity_ok"] = all(v["ok"] for v in res["identity"].values()) if res["identity"] else None
    if out_path:
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1, sort_keys=True, default=_json)
    return res


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", nargs="+", required=True, help="ARM=head_pred.npz (the C3 forward's output; ARM = H1-s<k> | H1u10k-s<k> | ...)")
    ap.add_argument("--contrast", nargs="*", default=[], help="A:B => the contrast A minus B (both given as --pred)")
    ap.add_argument("--tokens", default=str(C3_DIR / "tokens.json"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    preds = {}
    for spec in a.pred:
        name, _, path = spec.partition("=")
        RO.parse_arm(name)
        preds[name] = path
    contrasts = [tuple(c.split(":")) for c in a.contrast]
    try:
        res = run(preds, contrasts, Path(a.tokens), a.out, n_boot=a.n_boot, sensitivity=not a.no_sensitivity)
    except (NatRefused, RO.LeakRefused) as e:
        print(f"ZZNATP1_REFUSED {e}", flush=True)
        return 3
    if res["identity"] and not res["identity_ok"]:
        print(f"ZZNATP1_IDENTITY_FAILED {json.dumps({k: v for k, v in res['identity'].items() if not v['ok']}, default=_json)[:600]}", flush=True)
        return 6
    print(f"ZZNATP1_OK arms {sorted(res['arms'])} contrasts {sorted(res['contrasts'])} ess {res['population']['ess']:.4f} identity {sorted(res['identity']) or 'n/a'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
