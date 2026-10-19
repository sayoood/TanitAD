"""RESULT_RA1_FULL.md generator (../PREREG_RA1_FULL.md): renders the readings AND the "Reported, NOT read" list from the banked files.

  python ra1_full_report.py --root <logs/ra1_full> --mini-root <raw/ra1_es> --out RESULT_RA1_FULL.md [--json-out report.json]
                            [--tip <sha>] [--interim] [--family-budget S]

  <root>      runs/st<S>/<run>/eval/{eval.json,perwindow.npz}  (+ runs/st<S>_wd3.0/... for phase 3), full1.json, full_final.json,
              full_final3.json -- exactly what chain_ra1_full_runs.sh pulls.
  <mini-root> the banked 540-clip series (raw/ra1_es/st<S>/<run>/eval/...), for the two descriptive cross-scale rows.

It READS NOTHING NEW: every reading (selection, the A1x gate, K) is copied from ra1_full_verdict.py's JSON, and the seed means it
prints are re-derived from the runs and must equal full1.json's to 1e-9, or the report refuses (it would be reading other files than
the verdict did).  What it computes itself is descriptive only, and says so:
  * the data-scale effect: full - 540-clip run, same arm, budget and training seed, PAIRED over the 141 eval episodes (eval-141
    ADE@6s) and over the 60 holdout episodes, on window-aligned arrays (refused if the windows do not align);
  * the P10-yaw-540 deltas: full run - P10-yaw-540 (the 540-clip runs' prior), paired the same way.
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000. It answers ANOTHER DRAW OF EPISODES only; a cross-scale delta is
also a cross-RUN delta, so it is shown beside the training-seed floors F and flagged only when |delta| > 2 * max(F_full, F_540).
--interim renders whatever runs exist and makes NO selection: no S*, no gate, no K.
Exit code: 0 rendered; 2 on a missing / malformed / inconsistent input."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

CELL = "b0.08g50"
RUNS = {"R0": "R0_s{seed}", "A1": "A1_" + CELL + "_s{seed}", "A1x": "A1x_" + CELL + "_s{seed}"}
SEEDS = (0, 1)
HOLDOUT_SHA = "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9"
N_TRAIN_CACHE = 4572
B = 2000
TIE_M = 0.01
LON = ("speed_mae_mps", "speed_bias_mps", "speed_rmse_mps", "target_speed_acc.within_0.5_mps", "target_speed_acc.within_1.0_mps",
       "target_speed_acc.within_2.0_mps", "along_mae_m", "accel_mae_mps2", "ego_progress.progress_ratio_mean",
       "ego_progress.under_progress_rate")
LAT = ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "cross_mae_m", "cross_final_mae_m")
TAC = ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed")


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_run(d):
    """<d>/eval/eval.json (+ perwindow.npz when present) -> dict, or None when the run has no eval."""
    p = os.path.join(d, "eval", "eval.json")
    if not os.path.exists(p):
        return None
    pw = os.path.join(d, "eval", "perwindow.npz")
    return dict(dir=d, ev=_load(p), pw=dict(np.load(pw, allow_pickle=True)) if os.path.exists(pw) else None)


def scan(runs_root, suffix=""):
    """{S: {arm: {seed: run}}} over runs_root/st<S><suffix>; only directories whose name is exactly st<digits><suffix>."""
    out = {}
    if not os.path.isdir(runs_root):
        return out
    pat = re.compile(r"^st(\d+)" + re.escape(suffix) + "$")
    for name in sorted(os.listdir(runs_root)):
        m = pat.match(name)
        if not m:
            continue
        S = int(m.group(1))
        for arm, tmpl in RUNS.items():
            for seed in SEEDS:
                r = load_run(os.path.join(runs_root, name, tmpl.format(seed=seed)))
                if r is not None:
                    out.setdefault(S, {}).setdefault(arm, {})[seed] = r
    return out


def check_full(grid):
    """One holdout (the Amendment-1 list), one train list, one prior, the full 4,572-clip cache, yaw rate -- or refuse."""
    hold, train, p10 = set(), set(), set()
    for S, arms in grid.items():
        for arm, seeds in arms.items():
            for seed, r in seeds.items():
                ev, where = r["ev"], r["dir"]
                if ev.get("kin_mode") != "yaw":
                    raise ValueError(f"{where}: kin_mode {ev.get('kin_mode')} -- a non-compliant (steer) run")
                if ev.get("arm") != arm:
                    raise ValueError(f"{where}: eval.json says arm {ev.get('arm')}, the directory says {arm}")
                if (ev.get("disjointness") or {}).get("n_train") != N_TRAIN_CACHE:
                    raise ValueError(f"{where}: trained from {(ev.get('disjointness') or {}).get('n_train')} cache clips, not {N_TRAIN_CACHE}")
                if int(ev.get("ckpt_step", -1)) != S:
                    raise ValueError(f"{where}: ckpt_step {ev.get('ckpt_step')} in budget directory st{S}")
                h = ev["holdout"]
                hold.add(h.get("clips_sha256")); train.add(h.get("train_sha256")); p10.add(round(h["ade6_p10"]["mean"], 4))
    for name, s in (("holdout clip list", hold), ("train clip list", train), ("holdout P10 ADE@6s (one prior)", p10)):
        if len(s) > 1 or None in s:
            raise ValueError(f"the runs do not share ONE {name}: {sorted(map(str, s))}")
    if hold and next(iter(hold)) != HOLDOUT_SHA:
        raise ValueError(f"holdout {next(iter(hold))} is not the Amendment-1 holdout {HOLDOUT_SHA}")
    return dict(train_sha256=next(iter(train)) if train else None, holdout_p10_ade6=next(iter(p10)) if p10 else None)


def _aligned(a, b, which):
    pa, pb = a["pw"], b["pw"]
    if pa is None or pb is None:
        raise ValueError(f"perwindow.npz missing: {a['dir'] if pa is None else b['dir']}")
    if which == "eval141":
        same = (pa["clips"].shape == pb["clips"].shape and (pa["clips"] == pb["clips"]).all()
                and (pa["eid"] == pb["eid"]).all() and (pa["t"] == pb["t"]).all())
        return same, pa["eid"]
    same = (a["ev"]["holdout"].get("clips_sha256") == b["ev"]["holdout"].get("clips_sha256")
            and pa["hold_eid"].shape == pb["hold_eid"].shape and (pa["hold_eid"] == pb["hold_eid"]).all() and (pa["hold_t"] == pb["hold_t"]).all())
    return same, pa["hold_eid"]


def paired_runs(a, b, which, key_a, key_b, pairfn):
    """reduce(a[key_a]) - reduce(b[key_b]) on the SAME windows, paired over episodes.  Refuses misaligned windows."""
    ok, eid = _aligned(a, b, which)
    if not ok:
        raise ValueError(f"{which} windows do not align between {a['dir']} and {b['dir']}")
    r = pairfn(np.asarray(a["pw"][key_a], np.float64), np.asarray(b["pw"][key_b], np.float64), eid, n_boot=B, seed=0)
    return {k: (round(float(r[k]), 4) if k in ("delta", "lo", "hi") else r[k]) for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes")
            if k in r}


def fmt_d(d):
    if not isinstance(d, dict) or "delta" not in d:
        return "n/a"
    return f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]{' **sep**' if d.get('separated') else ''}"


def fmt_ci(c, nd=4):
    if not isinstance(c, dict) or "mean" not in c:
        return "n/a"
    return f"{c['mean']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"


def _get(blk, dotted):
    v = blk
    for k in dotted.split("."):
        if not isinstance(v, dict) or k not in v:
            return None
        v = v[k]
    return v


def fam_cell(fam, family, comp):
    blk = fam.get(family) or {}
    c = ((blk.get("ci") or {}).get("components") or {}).get(comp)
    if isinstance(c, dict) and "mean" in c:
        return fmt_ci(c)
    v = _get(blk, comp)
    return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def seedmeans(grid):
    rows = {}
    for S in sorted(grid):
        for arm in ("R0", "A1"):
            seeds = grid[S].get(arm, {})
            if all(s in seeds for s in SEEDS):
                h = [seeds[s]["ev"]["holdout"]["ade6_arm"]["mean"] for s in SEEDS]
                e = [seeds[s]["ev"]["K"]["ade6_arm"]["mean"] for s in SEEDS]
                rows[(arm, S)] = dict(hold=h, hold_mean=sum(h) / 2, hold_floor=abs(h[0] - h[1]), e141=e, e141_floor=abs(e[0] - e[1]))
    return rows


def crosscheck_full1(f1, rows):
    """The report must read the files the verdict read: every seed mean full1.json used is re-derived here to 1e-9."""
    for arm in ("R0", "A1"):
        for S, r in (f1.get("budgets") or {}).get(arm, {}).items():
            if "seed_mean_holdout_ade6" not in r:
                continue
            mine = rows.get((arm, int(S)))
            if mine is None or abs(mine["hold_mean"] - r["seed_mean_holdout_ade6"]) > 1e-9:
                raise ValueError(f"full1.json {arm}@{S} seed-mean holdout ADE {r['seed_mean_holdout_ade6']} != the runs' "
                                 f"{None if mine is None else mine['hold_mean']}: the report is not reading the verdict's files")


def render(a, pairfn) -> tuple[str, dict]:
    runs_root = os.path.join(a.root, "runs")
    grid = scan(runs_root)
    if not grid:
        raise ValueError(f"no runs under {runs_root}")
    prov = check_full(grid)
    rows = seedmeans(grid)
    f1 = f_final = f_final3 = None
    if not a.interim:
        f1 = _load(os.path.join(a.root, "full1.json"))
        crosscheck_full1(f1, rows)
        if os.path.exists(os.path.join(a.root, "full_final.json")):
            f_final = _load(os.path.join(a.root, "full_final.json"))
        if os.path.exists(os.path.join(a.root, "full_final3.json")):
            f_final3 = _load(os.path.join(a.root, "full_final3.json"))
    star = (f1 or {}).get("S_star") or {}
    k_arm = (f_final3 or f_final or f1 or {}).get("K_arm")
    out = dict(schema="ra1-full-report-v1", interim=bool(a.interim), tip=a.tip, provenance=prov, S_star=star, K_arm=k_arm,
               data_scale={}, p10_540={})
    L = []
    w = L.append
    final = f_final3 or f_final
    status = ("INTERIM -- the verdict is not in; NOTHING below is a reading" if a.interim else
              f"FINAL -- outcome **{final.get('outcome')}**" if final else
              f"PHASE 1 ONLY -- full1 next = `{(f1 or {}).get('next')}`; phase 2 not read")
    w("# RESULT — RA-1 on the full corpus (`PREREG_RA1_FULL.md`)\n")
    w(f"**Status:** {status}.  ")
    w(f"**Tier:** T1-class open-loop trajectory (eval-141 = 2,399 windows / 141 episodes; holdout = 60 clips, 4,076 windows).  ")
    w("**Estimator:** paired episode-cluster bootstrap (`taniteval.ci`), B = 2,000 — it answers *another draw of EPISODES*; training-seed "
      "variance is answered only by the floor F = |s0 − s1| (two seeds, so F is itself one draw); the readout is deterministic (no inference seed).  ")
    w(f"**Evidence class:** MEASURED — box 2, code at tip `{a.tip or 'see the chain log'}`; train list sha256 `{(prov['train_sha256'] or '')[:12]}…`, "
      f"holdout `{HOLDOUT_SHA[:12]}…`, holdout P10-yaw-4512 ADE@6s {prov['holdout_p10_ade6']}.  ")
    w("**Generated** by `code/ra1_full_report.py` from `raw/ra1_full/` — every reading is copied from `ra1_full_verdict.py`'s JSON, never re-decided here.\n")

    # ---------------------------------------------------------------- the readings
    w("## The readings (pre-registered order)\n")
    if a.interim:
        w("Not made: the run grid is incomplete or the verdict has not run. No S*, no gate, no K is implied by any number below.\n")
    else:
        w(f"* **Selection (holdout only):** S*_R0 = **{star.get('R0')}**, S*_A1 = **{star.get('A1')}**; edge flags {f1.get('edge')}; "
          f"K arm = **{f1.get('K_arm')}** (tie ≤ {TIE_M} m → R0). `full1.json` reading: {f1.get('reading')}")
        for tag, ff in (("Phase 2", f_final), ("Phase 3 (`--wd 3.0`)", f_final3)):
            if not ff:
                continue
            g = ff.get("A1x_control") or {}
            gd = g.get("delta_A1x_minus_A1_ade6")
            w(f"* **{tag} — reading 2, the A1x gate at S*_A1:** A1x − A1 eval-141 ADE@6s {fmt_d(gd)} → `{g.get('verdict')}`.")
            kr = ff.get("K_read")
            if kr:
                w(f"* **{tag} — reading 3, K:** bar: {kr.get('bar')}. K arm **{kr.get('K_arm')}** → **{'PASS' if kr.get('PASS') else 'FAIL'}**; "
                  f"training-seed floor (A1) {ff.get('training_seed_floor_A1')}.")
                w("\n  | arm | seed | ADE@6s | P10-yaw-4512 | arm − P10 [95 % CI] |\n  |---|---|---|---|---|")
                for nm, row in (("K: " + str(kr.get("K_arm")), kr.get("K_arm_row")), ("other (descriptive)", kr.get("other_arm_DESCRIPTIVE"))):
                    if not isinstance(row, dict):
                        continue
                    for s in ("seed0", "seed1"):
                        c = row.get(s)
                        if isinstance(c, dict):
                            w(f"  | {nm} | {s[-1]} | {c.get('ade6', float('nan')):.4f} | {c.get('p10_ade6', float('nan')):.4f} | {fmt_d(c.get('arm_minus_p10'))} |")
                w("")
            else:
                w(f"* **{tag} — reading 3, K:** NOT READ (the gate did not pass).")
            w(f"* **{tag} outcome `{ff.get('outcome')}`:** {ff.get('reading')}\n")

    # ---------------------------------------------------------------- reading 1 + the selection grid
    w("\n## Reading 1 (descriptive) and the selection grid — holdout selects, eval-141 never does\n")
    w("| arm | steps | holdout ADE@6s s0 / s1 | seed mean | mean − P10-4512 | F_hold | holdout Δ vs P10 s0 | s1 | eval-141 Δ vs P10 s0 (NON-DECIDING) | s1 | F_141 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for (arm, S), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        sd = grid[S][arm]
        mark = " ⭐S*" if star.get(arm) == S else ""
        w(f"| {arm} | {S}{mark} | {r['hold'][0]:.4f} / {r['hold'][1]:.4f} | {r['hold_mean']:.4f} | "
          f"{r['hold_mean'] - prov['holdout_p10_ade6']:+.4f} | {r['hold_floor']:.4f} | "
          f"{fmt_d(sd[0]['ev']['holdout'].get('ade6_arm_minus_p10'))} | {fmt_d(sd[1]['ev']['holdout'].get('ade6_arm_minus_p10'))} | "
          f"{fmt_d(sd[0]['ev']['K'].get('ade6_arm_minus_p10'))} | {fmt_d(sd[1]['ev']['K'].get('ade6_arm_minus_p10'))} | {r['e141_floor']:.4f} |")
    partial = [(arm, S, sorted(grid[S][arm])) for S in sorted(grid) for arm in ("R0", "A1") if arm in grid[S] and len(grid[S][arm]) < 2]
    for arm, S, seeds in partial:
        w(f"| {arm} | {S} | seed {seeds} only — not a row until both seeds exist | | | | | | | | |")
    w("\nF = |s0 − s1| (training-seed floor, one draw). `sep` = the paired episode CI excludes 0. A negative Δ = the arm beats its prior.\n")

    # ---------------------------------------------------------------- data scale (descriptive)
    mini = scan(a.mini_root) if a.mini_root else {}
    w("## The data-scale effect (descriptive, PREREG \"Reported, NOT read\") — full (4,512 clips) − 540-clip run\n")
    if not mini:
        w("NOT COMPUTED: no `--mini-root` given (the banked 540-clip runs).\n")
    else:
        w("Same arm, budget and training seed; same windows (asserted); paired over episodes. A cross-scale Δ is also a cross-RUN Δ, so it is "
          "flagged **> 2F** only when |Δ| exceeds twice the larger training-seed floor of the two scales (eval-141 floors).\n")
        w("| arm | steps | seed | eval-141 ADE@6s full − 540 | holdout ADE@6s full − 540 | 2F (eval-141) | beyond 2F (eval-141) |")
        w("|---|---|---|---|---|---|---|")
        for S in sorted(set(grid) & set(mini)):
            for arm in ("R0", "A1"):
                fs, ms = grid[S].get(arm, {}), mini[S].get(arm, {})
                if not all(s in fs and s in ms for s in SEEDS):
                    continue
                F = max(abs(fs[0]["ev"]["K"]["ade6_arm"]["mean"] - fs[1]["ev"]["K"]["ade6_arm"]["mean"]),
                        abs(ms[0]["ev"]["K"]["ade6_arm"]["mean"] - ms[1]["ev"]["K"]["ade6_arm"]["mean"]))
                for s in SEEDS:
                    d141 = paired_runs(fs[s], ms[s], "eval141", "ade6", "ade6", pairfn)
                    dh = paired_runs(fs[s], ms[s], "holdout", "hold_ade6", "hold_ade6", pairfn)
                    big = bool(d141["separated"] and abs(d141["delta"]) > 2 * F)
                    out["data_scale"][f"{arm}@{S}/s{s}"] = dict(eval141=d141, holdout=dh, two_F=round(2 * F, 4), beyond_2F=big)
                    w(f"| {arm} | {S} | {s} | {fmt_d(d141)} | {fmt_d(dh)} | {2 * F:.4f} | {'yes' if big else 'no'} |")
        unmatched = sorted(set(grid) ^ set(mini))
        w(f"\nBudgets on one scale only (no matched row): {unmatched}.\n")

    # ---------------------------------------------------------------- P10-yaw-540 deltas (descriptive)
    w("## P10-yaw-540 deltas (descriptive) — full run − the 540-clip prior, same windows\n")
    ref = next((mini[S][arm][s] for S in sorted(mini) for arm in ("R0", "A1") for s in SEEDS if s in mini[S].get(arm, {})), None)
    if ref is None:
        w("NOT COMPUTED: no banked 540-clip run to read P10-yaw-540's per-window ADE from.\n")
    else:
        others = [mini[S][arm][s] for S in sorted(mini) for arm in ("R0", "A1") for s in SEEDS if s in mini[S].get(arm, {})]
        for o in others:
            if o["pw"] is None or not np.array_equal(o["pw"]["p10_ade6"], ref["pw"]["p10_ade6"]):
                raise ValueError(f"the 540-clip runs do not share ONE P10-yaw-540 per-window ADE: {o['dir']}")
        w(f"P10-yaw-540 read from `{os.path.relpath(ref['dir'], a.mini_root).replace(os.sep, '/')}` (identical across all {len(others)} banked 540-clip runs).\n")
        w("| arm | steps | seed | eval-141 arm − P10-yaw-540 | holdout arm − P10-yaw-540 |\n|---|---|---|---|---|")
        for (arm, S) in sorted(rows, key=lambda k: (k[0], k[1])):
            for s in SEEDS:
                r = grid[S][arm][s]
                d141 = paired_runs(r, ref, "eval141", "ade6", "p10_ade6", pairfn)
                dh = paired_runs(r, ref, "holdout", "hold_ade6", "hold_p10_ade6", pairfn)
                out["p10_540"][f"{arm}@{S}/s{s}"] = dict(eval141=d141, holdout=dh)
                w(f"| {arm} | {S} | {s} | {fmt_d(d141)} | {fmt_d(dh)} |")
        w("")

    # ---------------------------------------------------------------- four families
    if a.family_budget:
        targets = [("A1", int(a.family_budget)), ("R0", int(a.family_budget))]
    elif star:
        targets = [(k_arm, star.get(k_arm))] + ([("A1", star.get("A1"))] if k_arm != "A1" else [])
    else:
        both = [S for S in sorted(grid) if ("R0", S) in rows and ("A1", S) in rows]
        targets = [("A1", both[-1]), ("R0", both[-1])] if both else []
    w("## The four metric families (reported, NOT read) — T1-class open loop, eval-141\n")
    w("Arm columns and the P10-yaw-4512 column carry **single-arm** episode-cluster intervals: the eval publishes no paired per-family "
      "delta (`perwindow.npz` holds ADE only), so **no arm − P10 family difference is claimed here** — pairing them is a WORK ITEM "
      "(the eval must save per-window family components). The strata rows ARE paired.\n")
    for arm, S in targets:
        seeds = (grid.get(S) or {}).get(arm, {})
        if not all(s in seeds for s in SEEDS):
            w(f"### {arm} @ {S} — NOT RENDERED: both seeds are not in.\n")
            continue
        f0, f1f = seeds[0]["ev"]["families"], seeds[1]["ev"]["families"]
        if "arm" not in f0 or "arm" not in f1f:
            w(f"### {arm} @ {S} — the families block errored: {f0.get('error') or f1f.get('error')} (a WORK ITEM, not a pass)\n")
            continue
        w(f"### {arm} @ {S} steps{' (the K arm)' if arm == k_arm and star else ''}\n")
        w(f"| family | metric | {arm} s0 | {arm} s1 | P10-yaw-4512 |\n|---|---|---|---|---|")
        for fam, comps in (("longitudinal", LON), ("lateral", LAT)):
            for c in comps:
                w(f"| {fam.upper()} | {c} | {fam_cell(f0['arm'], fam, c)} | {fam_cell(f1f['arm'], fam, c)} | {fam_cell(f0['p10'], fam, c)} |")
        for c in TAC:
            for m in ("accuracy", "kappa"):
                cell = lambda f: fmt_ci(((f.get("tactical") or {}).get(c) or {}).get("ci", {}).get(m))
                w(f"| TACTICAL | {c}.{m} | {cell(f0['arm'])} | {cell(f1f['arm'])} | {cell(f0['p10'])} |")
        for c in ("goal_bearing_mae_deg", "goal_range_ratio", "goal_long_bias_m", "goal_lat_bias_m"):
            cell = lambda f: (lambda v: "n/a" if v is None else f"{v:.4f}")(_get(f.get("tactical") or {}, "goal_setting." + c))
            w(f"| TACTICAL | goal_setting.{c} (point) | {cell(f0['arm'])} | {cell(f1f['arm'])} | {cell(f0['p10'])} |")
        for st in ("turn", "stop", "brake"):
            b0, b1 = seeds[0]["ev"]["strata"].get(st, {}), seeds[1]["ev"]["strata"].get(st, {})
            w(f"| TACTICAL stratum | {st}: ADE@6s arm − P10 (PAIRED; n {b0.get('n_windows')} win / {b0.get('n_episodes')} ep) | "
              f"{fmt_d(b0.get('ade6_arm_minus_p10')) if 'ade6_arm_minus_p10' in b0 else b0.get('status')} | "
              f"{fmt_d(b1.get('ade6_arm_minus_p10')) if 'ade6_arm_minus_p10' in b1 else b1.get('status')} | — |")
        sp = f0["arm"].get("strategic") or {}
        w(f"| STRATEGIC | — | **{sp.get('status')}** | **{(f1f['arm'].get('strategic') or {}).get('status')}** | — |")
        dk = (f0["arm"].get("longitudinal") or {}).get("distance_keeping") or {}
        w(f"| LONGITUDINAL | distance_keeping | **{dk.get('status')}** (n {dk.get('n')}) | same | — |\n")
        w(f"* **STRATEGIC:** {sp.get('status')} — {sp.get('ruling')} Underlying: {(sp.get('underlying') or {}).get('status')} "
          f"({(sp.get('underlying') or {}).get('reason', '')[:160]}…). Never a pass, never omitted.")
        w(f"* **Distance-keeping / lead braking:** {dk.get('status')}, n = {dk.get('n')} — {f0.get('lead_braking_stratum_and_distance_keeping')}")
        adm = f0["arm"].get("_longitudinal_claim_admissible")
        w(f"* **Longitudinal admissibility:** `_longitudinal_claim_admissible` = {adm}. "
          + ("The v0 anti-echo control is UNDISCHARGED (no v0 in the window dict), so every LONGITUDINAL number above is a fidelity "
             "diagnostic, **not** a longitudinal capability result." if adm is False else ""))
        w("")

    # ---------------------------------------------------------------- COST / SENS / instruments
    w("## COST and SENS (not PREREG_RA1_FULL readings)\n")
    w("* **SENS:** not in this design (it needs A0).")
    w("* **COST** (T0; bar from PREREG_HJ_STAGE0: L2 − L1 > 0), A1 runs, reported only:\n")
    w("| steps | seed | L2 − L1 native | L2 − L1 matched |\n|---|---|---|---|")
    for S in sorted(grid):
        for s in SEEDS:
            r = grid[S].get("A1", {}).get(s)
            if r and isinstance(r["ev"].get("COST"), dict):
                c = r["ev"]["COST"]
                w(f"| {S} | {s} | {fmt_d(c.get('L2_minus_L1_native'))} | {fmt_d(c.get('L2_minus_L1_matched'))} |")
    w("\n## Instruments, per run\n")
    w("| run | instrument_ok | P10 − P10 | GT ADE max | P10 vs banked refit (m) | untrained params | n_train / overlap | cfg | eval s |")
    w("|---|---|---|---|---|---|---|---|---|")
    bad = []
    for S in sorted(grid):
        for arm in RUNS:
            for s, r in sorted(grid[S].get(arm, {}).items()):
                ev = r["ev"]; c = ev.get("controls") or {}; dj = ev.get("disjointness") or {}
                if not ev.get("instrument_ok"):
                    bad.append(f"st{S}/{arm}_s{s}")
                w(f"| st{S}/{RUNS[arm].format(seed=s)} | {ev.get('instrument_ok')} | {c.get('p10_minus_p10_ok')} | {c.get('gt_ade_max')} | "
                  f"{(c.get('p10_vs_banked_refit') or {}).get('max_abs_diff_m')} | {len((ev.get('param_audit') or {}).get('untrained', []))} | "
                  f"{dj.get('n_train')} / {dj.get('overlap')} | {ev.get('cfg_sha12')} | {ev.get('seconds', 0):.0f} |")
    w("")
    if bad:
        w(f"⛔ **instrument_ok is False for {bad}** — those runs' numbers are not admissible.\n")
    out["instrument_not_ok"] = bad
    return "\n".join(L) + "\n", out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="logs/ra1_full: runs/, full1.json, full_final.json[, full_final3.json]")
    ap.add_argument("--mini-root", default=None, help="the banked 540-clip runs (raw/ra1_es)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--tip", default=None, help="the commit the runs' code was sha-checked against")
    ap.add_argument("--interim", action="store_true", help="render the runs present; read NOTHING (no S*, gate or K)")
    ap.add_argument("--family-budget", default=None, help="render the families at this budget instead of S*")
    a = ap.parse_args(argv)
    try:
        from taniteval.ci import paired_episode_cluster_bootstrap
    except ImportError:                                  # run from the repo checkout: <repo>/<pkg path>/code/this file
        sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "taniteval"))
        from taniteval.ci import paired_episode_cluster_bootstrap
    try:
        md, js = render(a, paired_episode_cluster_bootstrap)
    except (OSError, KeyError, ValueError) as e:
        print(f"REPORT INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(js, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {a.out}" + (f" and {a.json_out}" if a.json_out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
