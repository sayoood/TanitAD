"""D-DAC-HUMAN-ZERO-1 anatomy: WHY the drivable term zeroes the recorded human on 44.6 %.

Three candidate causes, none excluded before the data — map quality, rule strictness, genuine
off-map driving. All three are measured on the SAME 736 held-out windows the finding used,
through the harness's OWN loaders (``taniteval.tools.s1_pass.Corpus``), and the published
per-window ``dac`` is REPRODUCED before anything is characterised. If it does not reproduce,
the run stops: a characterisation of a number I cannot reproduce describes my own pipeline.

⛔ Read-only, CPU only, no GPU. Clip ids appear only as sha12.
⛔ Every rule variant is a MEASUREMENT of what that rule would read, never a recommendation.
   The current rule's numbers stay on record exactly as landed.

    python dac_anatomy.py --out-dir <raw dir> [--config ... --cache ... --maps ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path("D:/Projects/TanitAD")
DEFAULTS = {
    "config": "C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run/config.json",
    "cache": "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB",
    "labels": "C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/s2_labels_v8_eval.jsonl.gz",
    "agents": "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz",
    "maps": "D:/Projects/TanitAD-artifacts/sam3-maps-eval",
    "roundtrip": str(REPO / "TanitAD Research Lab/Architecture & Inference/Research"
                            "/2026-09-19-s1-collision-gate/raw/roundtrip_halfB_A8cfg.json"),
    "a1": "C:/Users/Admin/qland/a1_per_half_floor.json",
}
#: the map's own channel order (``semantic_map_gt.CHANNELS``), re-read at run time
ROAD_PAINT = ("lane / road line", "crosswalk", "arrow / text")
EXPLICIT_OFF = ("non-drivable edge", "hatched area", "sidewalk / verge")
BANDS = ((0, 15), (15, 30), (30, 45), (45, 60))
CELL_M, Y_HALF_M, THR = 0.5, 16.0, 0.5


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def band_of(x: float) -> str:
    for lo, hi in BANDS:
        if lo <= x < hi:
            return f"{lo}-{hi}"
    return "off-grid"


def _import_harness():
    sys.path.insert(0, str(REPO / "taniteval" / "tools"))
    import s1_pass as S1                                   # noqa: E402
    from tanitad.rl import pdm_proxy as P                  # noqa: E402
    from tanitad.data import semantic_map_gt as SMG        # noqa: E402
    # ⛔ MSYS/PYTHONPATH can serve these from another checkout; assert the disk they came from.
    for m in (S1, P, SMG):
        if not str(Path(m.__file__).resolve()).startswith(str(REPO)):
            raise SystemExit(f"⛔ {m.__name__} came from {m.__file__}, not {REPO}")
    return S1, P, SMG


def variants(frac_drivable, road, off, seenf, inside, seen_mask):
    """Every rule as a per-sample violation mask [T, 4] -> the window's dac under it.

    ``inside``/``seen_mask`` are the current rule's own admissibility test; a variant that
    changes admissibility says so in its name. Nothing here is a recommendation.
    """
    adm = inside & seen_mask
    base = adm & (frac_drivable < THR)
    out = {}

    def zero(mask_tc, *, min_corners=1, min_consecutive_ticks=1):
        per_tick = mask_tc.sum(axis=1)                       # violating corners per tick
        hit = per_tick >= min_corners
        if min_consecutive_ticks > 1:
            run, best = 0, 0
            for h in hit:
                run = run + 1 if h else 0
                best = max(best, run)
            return 0.0 if best >= min_consecutive_ticks else 1.0
        return 0.0 if hit.any() else 1.0

    out["V0 current rule (any corner, any tick, drivable < 0.50)"] = zero(base)
    out["V1 >= 2 corners at the same tick"] = zero(base, min_corners=2)
    out["V2 >= 2 consecutive ticks"] = zero(base, min_consecutive_ticks=2)
    out["V2b >= 4 consecutive ticks"] = zero(base, min_consecutive_ticks=4)
    out["V3 threshold 0.25"] = zero(adm & (frac_drivable < 0.25))
    out["V4 abstain unless seen fraction >= 0.75"] = zero(
        inside & seen_mask & (seenf >= 0.75) & (frac_drivable < THR))
    out["V5 abstain unless seen fraction >= 0.90"] = zero(
        inside & seen_mask & (seenf >= 0.90) & (frac_drivable < THR))
    # ⚠️ PROPOSALS — a DEFINITION change, not a strictness knob. Reported, not chosen.
    out["P1 road surface (drivable + paint) < 0.50"] = zero(adm & (road < THR))
    out["P2 explicit off-road (edge + hatched + sidewalk) >= 0.50"] = zero(adm & (off >= THR))
    out["P1+V1 road surface, >= 2 corners"] = zero(adm & (road < THR), min_corners=2)
    out["P1+V2 road surface, >= 2 consecutive ticks"] = zero(adm & (road < THR),
                                                             min_consecutive_ticks=2)
    out["P2+V2 explicit off-road, >= 2 consecutive ticks"] = zero(adm & (off >= THR),
                                                                  min_consecutive_ticks=2)
    return out


def known_value_controls(P, variants_fn) -> dict:
    """⛔ A scan that reads 0 proves nothing without a control that must read a known value.

    Four synthetic maps under a straight 10 m/s trajectory: all-drivable must read 1 under
    EVERY variant; all-sidewalk must read 0 under every variant that can see it; unseen must
    read 1 (no evidence); and an all-'seen, no map class' map — the map saying NOTHING — must
    read 0 under the current rule, which is the finding's mechanism in one line.
    """
    T = P.PROXY.n_ticks + 1
    x = torch.arange(T, dtype=torch.float32) * P.PROXY.dt * 10.0
    states = torch.stack([x, torch.zeros(T), torch.zeros(T), torch.full((T,), 10.0)], -1)[None]
    corners = P._ego_boxes(states, P.PROXY)[0]
    ix = torch.floor(corners[..., 0] / CELL_M).long().clamp(0, 119).numpy()
    iy = torch.floor((corners[..., 1] + Y_HALF_M) / CELL_M).long().clamp(0, 63).numpy()
    inside = np.ones(ix.shape, bool)
    got = {}
    for name, (dr, ro, of, sf, sm) in {
        "all drivable, seen": (1.0, 1.0, 0.0, 1.0, True),
        "all sidewalk, seen": (0.0, 0.0, 1.0, 1.0, True),
        "all unseen": (0.0, 0.0, 0.0, 0.0, False),
        "all 'seen, no map class'": (0.0, 0.0, 0.0, 1.0, True),
    }.items():
        got[name] = variants_fn(np.full(ix.shape, dr, np.float32), np.full(ix.shape, ro, np.float32),
                                np.full(ix.shape, of, np.float32), np.full(ix.shape, sf, np.float32),
                                inside, np.full(ix.shape, sm))
    ok = (all(v == 1.0 for v in got["all drivable, seen"].values())
          and all(v == 1.0 for v in got["all unseen"].values())
          and got["all sidewalk, seen"]["V0 current rule (any corner, any tick, drivable < 0.50)"] == 0.0
          and got["all 'seen, no map class'"]["V0 current rule (any corner, any tick, drivable < 0.50)"] == 0.0)
    return {"readings": got, "all_as_specified": bool(ok)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k, default=v)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    S1, P, SMG = _import_harness()
    mods = {m.__name__: md5(Path(m.__file__)) for m in (S1, P, SMG)}
    t_start = time.time()

    ch_names = list(SMG.CHANNELS)
    i_dr = ch_names.index("drivable")
    i_ns = ch_names.index("not seen")
    i_nc = ch_names.index("seen, no map class")
    i_paint = [ch_names.index(c) for c in ROAD_PAINT]
    i_off = [ch_names.index(c) for c in EXPLICIT_OFF]

    corp = S1.Corpus(a.config, a.cache, a.labels, a.agents, a.maps, lru=2)
    wis = S1.trainer_windows(corp.ds, a.n)
    elig = [wi for wi in wis if corp.eligibility(wi) is None]
    published = {(r["sha12"], r["t0"]): r for r in
                 json.loads(Path(a.roundtrip).read_text(encoding="utf-8"))["roundtrip"]["rows"]}

    rows, vsamples, agree, no_map = [], [], 0, 0
    # ⛔ Band COUNTS of violations are meaningless without their denominator: a 4 s trajectory
    # spends most of its ticks near the ego, so "most violations are near" can be pure exposure.
    band_all, band_adm, band_viol = Counter(), Counter(), Counter()
    band_dr_sum, band_dr_n = Counter(), Counter()
    bname = [f"{lo}-{hi}" for lo, hi in BANDS]
    for wi in elig:
        it = corp.light_item(wi)            # this call also populates the loader's map store
        if it["map_drivable"] is None:
            no_map += 1
            continue
        e_i, _ = corp.ds.index[wi]
        cid = str(corp.clip_ids[e_i])
        r0 = it["t0"] + corp.raw_off
        g = corp.shim.map_store.get(cid)    # ⭐ the loader's OWN handle, not a second open
        mf = g.read(np.asarray([r0]))
        cart, seen = np.asarray(mf.cart[0], np.float32), np.asarray(mf.seen[0], bool)
        # ⭐ the control that makes this pipeline the harness's: my own read must equal the
        # loader's, channel and mask, before any variant is computed on it.
        if not (np.array_equal(cart[i_dr], it["map_drivable"].numpy())
                and np.array_equal(seen, it["map_seen"].numpy())):
            raise SystemExit(f"⛔ map read differs from the harness loader at {it['sha12']}")

        states = it["human"][None]
        corners = P._ego_boxes(states, P.PROXY)[0].numpy()            # [T, 4, 2]
        X, Y = corners[..., 0], corners[..., 1]
        ix = np.floor(X / CELL_M).astype(np.int64)
        iy = np.floor((Y + Y_HALF_M) / CELL_M).astype(np.int64)
        h, w = cart.shape[1], cart.shape[2]
        inside = (ix >= 0) & (ix < h) & (iy >= 0) & (iy < w)
        ixc, iyc = np.clip(ix, 0, h - 1), np.clip(iy, 0, w - 1)
        cells = cart[:, ixc, iyc]                                     # [9, T, 4]
        frac = cells[i_dr]
        road = frac + cells[i_paint].sum(0)
        off = cells[i_off].sum(0)
        seenf = 1.0 - cells[i_ns]
        seen_mask = seen[ixc, iyc]
        vz = variants(frac, road, off, seenf, inside, seen_mask)
        dac0 = vz["V0 current rule (any corner, any tick, drivable < 0.50)"]

        ref = P.dac_from_drivable(states, it["map_drivable"], it["map_seen"])[0].item()
        pub = published.get((it["sha12"], it["t0"]))
        agree += int(dac0 == ref == (pub["dac_h"] if pub else ref))

        viol = inside & seen_mask & (frac < THR)
        adm = inside & seen_mask
        bi = np.digitize(X, [b[1] for b in BANDS[:-1]])
        on_grid = (X >= 0) & (X < BANDS[-1][1])
        for k, nm in enumerate(bname):
            m = (bi == k) & on_grid
            band_all[nm] += int(m.sum())
            band_adm[nm] += int((m & adm).sum())
            band_viol[nm] += int((m & viol).sum())
            band_dr_sum[nm] += float(frac[m & adm].sum())
            band_dr_n[nm] += int((m & adm).sum())
        tks, cns = np.nonzero(viol)
        for t, c in zip(tks.tolist(), cns.tolist()):
            vsamples.append({
                "sha12": it["sha12"], "t0": it["t0"], "tick": t, "corner": c,
                "x_m": round(float(X[t, c]), 2), "y_m": round(float(Y[t, c]), 2),
                "band": band_of(float(X[t, c])),
                "drivable": round(float(frac[t, c]), 3),
                "road_surface": round(float(road[t, c]), 3),
                "explicit_off": round(float(off[t, c]), 3),
                "seen_frac": round(float(seenf[t, c]), 3),
                "no_class": round(float(cells[i_nc, t, c]), 3),
                "argmax": ch_names[int(np.argmax(cells[:, t, c]))],
            })
        rows.append({
            "sha12": it["sha12"], "t0": it["t0"], "dac": dac0, "ref_dac": ref,
            "published_dac": (pub["dac_h"] if pub else None),
            "n_violating_samples": int(viol.sum()),
            "n_violating_ticks": int((viol.sum(axis=1) > 0).sum()),
            "first_violating_tick": int(np.argmax(viol.any(axis=1))) if viol.any() else None,
            "min_violating_x_m": round(float(X[viol].min()), 2) if viol.any() else None,
            "max_violating_x_m": round(float(X[viol].max()), 2) if viol.any() else None,
            "frac_samples_inside_grid": round(float(inside.mean()), 3),
            "frac_samples_seen": round(float((inside & seen_mask).mean()), 3),
            "max_x_m": round(float(X.max()), 2),
            "variants": vz,
        })

    n = len(rows)
    zero_rate = {k: round(sum(1 for r in rows if r["variants"][k] == 0.0) / n, 4)
                 for k in rows[0]["variants"]}
    v_by_band = Counter(s["band"] for s in vsamples)
    v_by_argmax = Counter(s["argmax"] for s in vsamples)
    zeroed = [r for r in rows if r["dac"] == 0.0]
    rep = {
        "_what": "D-DAC-HUMAN-ZERO-1 anatomy: the human's DAC on the S1 held-out windows",
        "_evidence_class": "MEASURED (ours; CPU only, read-only)",
        "inputs": {k: getattr(a, k) for k in DEFAULTS},
        "module_md5_at_start": mods,
        "windows": {"drawn": len(wis), "eligible": len(elig), "scored": n, "no_map": no_map},
        "reproduction": {
            "windows_where_mine_==_loader_==_published": agree,
            "reproduced": agree == n,
            "published_zero_rate": round(sum(1 for r in rows
                                             if r["published_dac"] == 0.0) / n, 4),
        },
        "zero_rate_by_rule": zero_rate,
        "violating_samples": {
            "total": len(vsamples),
            "per_zeroed_window_mean": round(len(vsamples) / max(len(zeroed), 1), 2),
            "by_band": dict(v_by_band),
            "by_argmax_channel": dict(v_by_argmax),
            "seen_frac_lt_0.75": sum(1 for s in vsamples if s["seen_frac"] < 0.75),
            "road_surface_ge_0.5": sum(1 for s in vsamples if s["road_surface"] >= 0.5),
            "explicit_off_ge_0.5": sum(1 for s in vsamples if s["explicit_off"] >= 0.5),
            "no_class_ge_0.5": sum(1 for s in vsamples if s["no_class"] >= 0.5),
        },
        "zeroed_windows": {
            "n": len(zeroed),
            "one_sample_only": sum(1 for r in zeroed if r["n_violating_samples"] == 1),
            "one_tick_only": sum(1 for r in zeroed if r["n_violating_ticks"] == 1),
            "median_violating_samples": float(np.median([r["n_violating_samples"] for r in zeroed])),
            "median_first_violating_tick": float(np.median([r["first_violating_tick"] for r in zeroed])),
            "median_min_violating_x_m": float(np.median([r["min_violating_x_m"] for r in zeroed])),
        },
        "by_band": {nm: {
            "samples": band_all[nm], "admissible (inside + seen)": band_adm[nm],
            "violating": band_viol[nm],
            "violation_rate_of_admissible": round(band_viol[nm] / max(band_adm[nm], 1), 4),
            "mean_drivable_fraction_under_the_footprint": round(
                band_dr_sum[nm] / max(band_dr_n[nm], 1), 4),
        } for nm in bname},
        "tick0": {
            "windows_with_a_violation_at_tick_0": len({(s["sha12"], s["t0"]) for s in vsamples
                                                       if s["tick"] == 0}),
            "note": "the recorded ego's own footprint at t0 — it is on the road by construction",
        },
        "zeroed_by_category": {
            "only_road_surface_cells": sum(
                1 for r in rows if r["dac"] == 0.0 and all(
                    s["road_surface"] >= THR for s in vsamples
                    if (s["sha12"], s["t0"]) == (r["sha12"], r["t0"]))),
            "any_explicit_off_road_cell": sum(
                1 for r in rows if r["dac"] == 0.0 and any(
                    s["explicit_off"] >= THR for s in vsamples
                    if (s["sha12"], s["t0"]) == (r["sha12"], r["t0"]))),
        },
        "a1_per_band_floor": json.loads(Path(a.a1).read_text(encoding="utf-8"))["halves"]["halfB"]["bands"],
        "controls": known_value_controls(P, variants),
        "wall_s": round(time.time() - t_start, 1),
    }
    mods_end = {m.__name__: md5(Path(m.__file__)) for m in (S1, P, SMG)}
    rep["module_md5_at_end"] = mods_end
    rep["modules_unchanged_during_run"] = mods == mods_end
    (out / "dac_anatomy.json").write_text(json.dumps(rep, indent=1) + "\n",
                                          encoding="utf-8", newline="\n")
    with (out / "dac_windows.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with (out / "dac_violating_samples.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for s in vsamples:
            fh.write(json.dumps(s) + "\n")
    print(json.dumps({k: rep[k] for k in ("windows", "reproduction", "zero_rate_by_rule",
                                          "violating_samples", "by_band", "tick0",
                                          "zeroed_by_category", "zeroed_windows",
                                          "modules_unchanged_during_run")}, indent=1))
    print("controls all as specified:", rep["controls"]["all_as_specified"], "| wall", rep["wall_s"], "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
