#!/usr/bin/env python3
"""dk141.py -- LONGITUDINAL distance keeping (headway / time gap / min-TTC / DK violation rate) for any EVAL-141 run.  CPU only, zero training.

    python -I dk141.py read --arm M2b_s0=<eval_M2b_s0.npz> --arm L=<eval_L.npz> [--p10 ref/p10_eval_pred6.npy] [--pair A:B ...] --out raw/DK141_READ.json
                            [--table-md raw/DK141_TABLE.md] [--perwindow-out raw/dk141_perwindow.npz]
    python -I dk141.py controls --out raw/DK141_CONTROLS.json            (known-value controls on the real joined data; rc 3 if any control fails)

WHAT IT IS.  The eval (`stack/scripts/v8f_mini_eval.py run`) writes the arm's plan as `pred` [2399,30,2] (window-origin ego frame, metres, 0.2 s knots; verified:
`eval_M2b_s0.npz` keys pred/traj30/.../ade6/eid).  Distance keeping is therefore a READ of the saved npz, not a rerun.  This file defines NO new metric: every
number is `taniteval.lead_metrics` (`distance_keeping`, `lead_viol`, `lead_progress`) on the arm's `pred` against the SAME logged lead future the human (GT) path is
scored against, joined from the B1 lead block by (clip_id, RAW frame 2t) with the label-free speed proof, via the tip's `v8f_dk_read.py` (sha-pinned, ref/).  What it
adds over `v8f_dk_read.py` (which hard-codes the arms M1/M2b x s0/s1 + P10): ARBITRARY arm names and pairs (B', the LoRA arm L, ...), the TTC-violation rate
(min TTC < 1.5 s, `stack/tanitad/rl/rewards.py:563`), the plan's raw tight-gap rate (time gap < 1.0 s, any human), a per-window DK flag file, and known-value controls.

LAYER OF THE LEAD AGENT (state it with any number): the lead TRACK is the PUBLISHED PhysicalAI-AV feature `obstacle.offline` (3D cuboids), turned into a per-frame
window-origin lead block by `taniteval/tools/build_lead_block_b1.py` (147 v7.2 EVAL clips; K=10 banked block sha c0525943..., K=30 rebuild sha 7a879baa...).  It is
NOT the augmented v7.x label release and NOT the v9 scalar `lead_valid / lead_gap` of `lead_eval139.npz` (that feeds the eval's `viol_LEAD` label-constraint check,
which is neither headway, time gap nor TTC).  Only LEAD windows score; NO_LEAD / NOT_STRAIGHT / NO_LABEL are COUNTED with their reason, never free flow.

TIER: T1-class open loop (the arm is conditioned on its OWN held upper-layer outputs, never on GT; the lead's logged future does not react to the arm).
ESTIMATOR: `taniteval.ci.episode_cluster_bootstrap` (means) / `paired_episode_cluster_bootstrap` (deltas), B 2000, seed 0, over EPISODES.  It answers "another draw
of EPISODES" only -- not another training run, not another inference run (H-ESTIM-SEED-1).  No distance-keeping bar is registered for v8F-mini: nothing here is PASS/FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
HOME = _HERE if os.path.isdir(_HERE + "/ref") else "E:/Projects/TanitAD-artifacts/dagland/motion_c/dk141"     # a mutant copy lives elsewhere and still finds the pins
REF = HOME + "/ref"
SCHEMA = "dk141-v1"
DT = 0.2
TTC_VIOL_S = 1.5                  # stack/tanitad/rl/rewards.py:563 `ttc_min_s` default; stack/scripts/r9_e2_distkeep.py:54
TIGHT_TG_S = 1.0                  # taniteval/lead_metrics.py:433 LEAD_VIOL_TG_S, applied here WITHOUT the human condition
B = 2000
PIN_DK_READ = "77a3faffd052c9d71972a52a2c9d5300ad5f4facdd07f059f083ba018b3e6f57"      # tip 3a8d4761 v8f_dk_read.py
PIN_LEAD_METRICS = "25190ad698bb6d4aff316b61d60ef5e258a2ca141e6b79db484fc92b82a20fb3"
PIN_CI = "528276c5a51a94dc614406a6907b7f0da51081f6d3bd8bf5d8432857393755be"
PIN_BLOCK_K10 = "c0525943605cfd4ab55258bed04046b0e58257bb888544c385095708a0091898"
PIN_BLOCK_K30 = "7a879baa2b6f4c140c2b64c9c098a38f84ecb04926da61fa8e96cfffc33073d9"
PIN_GT = "da38ce866772adc626b6484a140ad1532791eecc5c6e0842c3b7787dba1117e4"
TIER = ("T1-class open loop (self-action: the arm is conditioned on its OWN held upper-layer outputs, never on GT); the lead's logged future does not react, so "
        "'closes on the lead' means 'closes on where the lead was'")
ESTIMATOR = "taniteval.ci.episode_cluster_bootstrap / paired_episode_cluster_bootstrap, B 2000, seed 0, over EPISODES; delta = a - b; answers 'another draw of EPISODES' only"
LAYER = ("lead track = published PhysicalAI-AV `obstacle.offline` via taniteval/tools/build_lead_block_b1.py (B1 per-frame block, 147 v7.2 EVAL clips); "
         "NOT the augmented v7.x label release; NOT the v9 scalar lead labels of lead_eval139.npz")


class Refusal(Exception):
    pass


def refuse(msg):
    raise Refusal(msg)


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _need(p, pin=None):
    if not os.path.exists(p):
        refuse(f"pinned input {p} does not exist")
    got = sha256(p)
    if pin and got != pin:
        refuse(f"{p} is {got}, not the pinned {pin}")
    return got


# ------------------------------------------------------------------------------------------------ boot: pinned instrument
_M = {}


def boot(lm_override: str | None = None):
    """Put the tip's stack + taniteval (tip_export/) first on sys.path, load the sha-pinned v8f_dk_read.py.  ``lm_override`` is a TEST HOOK (mutants only)."""
    if _M:
        return _M
    for p in (HOME + "/tip_export/taniteval", HOME + "/tip_export/stack"):
        if p not in sys.path:
            sys.path.insert(0, p)
    _need(f"{HOME}/tip_export/taniteval/taniteval/ci.py", PIN_CI)
    if lm_override is None:
        _need(f"{HOME}/tip_export/taniteval/taniteval/lead_metrics.py", PIN_LEAD_METRICS)
    _need(f"{REF}/v8f_dk_read.py", PIN_DK_READ)
    import taniteval  # noqa: E402
    if not os.path.abspath(taniteval.__file__).replace("\\", "/").startswith(HOME + "/tip_export/"):
        refuse(f"taniteval imported from {taniteval.__file__}, not the pinned tip export")
    if lm_override is not None:
        spec = importlib.util.spec_from_file_location("taniteval.lead_metrics", lm_override)
        m = importlib.util.module_from_spec(spec)
        sys.modules["taniteval.lead_metrics"] = m
        spec.loader.exec_module(m)
        taniteval.lead_metrics = m
    from taniteval import ci, lead_metrics as lm  # noqa: E402
    spec = importlib.util.spec_from_file_location("v8f_dk_read", f"{REF}/v8f_dk_read.py")
    dk = importlib.util.module_from_spec(spec)
    sys.modules["v8f_dk_read"] = dk
    spec.loader.exec_module(dk)
    _M.update(DK=dk, lm=lm, ci=ci, lm_override=lm_override)
    return _M


# ------------------------------------------------------------------------------------------------ inputs
def load_gt():
    _need(f"{REF}/../gt141.npz", PIN_GT)
    z = np.load(f"{REF}/../gt141.npz")
    return {"gt": z["gt"].astype(np.float64), "eid": z["eid"], "clip": z["clip"].astype(str), "frame": z["frame"].astype(np.int64),
            "v0": z["v0"].astype(np.float64), "dt": float(z["dt"])}


def load_arm(name, path, eid):
    if not os.path.exists(path):
        refuse(f"arm {name}: {path} does not exist")
    z = np.load(path)
    if "pred" not in z.files or "eid" not in z.files:
        refuse(f"arm {name}: {path} carries no pred / eid (keys {z.files}); it is not an eval-141 `run` output")
    if not np.array_equal(z["eid"], eid):
        refuse(f"arm {name}: eid differs from the eval-141 grid (not the 2,399-window grid)")
    return z["pred"].astype(np.float64), (np.asarray(z["ade6"], np.float64) if "ade6" in z.files else None), {"path": os.path.abspath(path), "sha256": sha256(path)}


def blocks_default():
    return [(f"{REF}/b1_eval_lead_block.npz", PIN_BLOCK_K10), (f"{REF}/b1_eval_lead_block_k30.npz", PIN_BLOCK_K30)]


# ------------------------------------------------------------------------------------------------ extra rows (flags per window, rates, pairs)
def _flags(pw, nm, is_lead):
    """Per-window DK flags of ``nm`` ('human' or an arm) from the perwindow arrays.  NaN where the window carries no value."""
    hw, tg, tt = (np.asarray(pw[f"{nm}__{k}"], np.float64) for k in ("headway_min_m", "time_gap_min_s", "min_ttc_s"))
    th = np.asarray(pw["human__time_gap_min_s"], np.float64)
    nan = np.full(hw.shape, np.nan)
    tight = np.where(np.isfinite(tg), (tg < TIGHT_TG_S).astype(float), nan)
    ttcv = np.where(np.isfinite(tt), (tt < TTC_VIOL_S).astype(float), nan)
    with np.errstate(invalid="ignore"):
        at_risk = is_lead & np.isfinite(th) & (th >= TIGHT_TG_S)
        viol = np.where(at_risk, ((tg < TIGHT_TG_S) & np.isfinite(tg)).astype(float), nan)
    return {"tight_tg": tight, "ttc_viol": ttcv, "lead_viol": viol, "at_risk": at_risk}


def extra_rows(pw, arm_names, pairs, n_boot, seed):
    ci = boot()["ci"]
    eid, is_lead = np.asarray(pw["eid"]), np.asarray(pw["is_lead"], bool)
    fl = {nm: _flags(pw, nm, is_lead) for nm in ["human"] + list(arm_names)}
    out = {"definitions": {"ttc_viol_rate": f"share of windows with a finite min-TTC whose min-TTC < {TTC_VIOL_S} s (rewards.py:563 default); a lost lead (plan never inside the corridor behind the lead) has no TTC and is COUNTED, not scored",
                           "tight_tg_rate": f"share of windows with a defined time gap whose time gap < {TIGHT_TG_S} s, ANY human (the plan's own tightness; LEAD-VIOL is the human-conditioned version)",
                           "lead_viol_at_risk": f"registered LEAD-VIOL: plan time gap < {TIGHT_TG_S} s on lead windows where the HUMAN keeps >= {TIGHT_TG_S} s"},
           "arms": {}, "pairs": {}}
    for nm, f in fl.items():
        row = {}
        for key in ("ttc_viol", "tight_tg"):
            v = f[key]
            ok = np.isfinite(v)
            row[key] = ({"n": int(ok.sum()), "n_lead_windows": int(is_lead.sum()), "n_lead_without_value": int((is_lead & ~ok).sum()), "n_flagged": int(v[ok].sum()),
                         **{k: x for k, x in ci.episode_cluster_bootstrap(v[ok], eid[ok], reduce="mean", n_boot=n_boot, seed=seed).items() if k in ("mean", "lo", "hi", "n_episodes")}}
                        if ok.any() else {"n": 0, "status": "NOT-APPLICABLE", "reason": "no window with a value"})
        vv = f["lead_viol"]
        okv = np.isfinite(vv)
        row["lead_viol_at_risk"] = {"n_at_risk": int(f["at_risk"].sum()), "n_flagged": int(vv[okv].sum())}
        out["arms"][nm] = row
    for a, b in pairs:
        if a in fl and b in fl:
            pr = {}
            for key in ("ttc_viol", "tight_tg"):
                va, vb = fl[a][key], fl[b][key]
                ok = np.isfinite(va) & np.isfinite(vb)
                pr[key] = ({"n_used": int(ok.sum()), "n_a_only": int((np.isfinite(va) & ~np.isfinite(vb)).sum()), "n_b_only": int((~np.isfinite(va) & np.isfinite(vb)).sum()),
                            **ci.paired_episode_cluster_bootstrap(va[ok], vb[ok], eid[ok], n_boot=n_boot, seed=seed)} if ok.any()
                           else {"n_used": 0, "status": "NOT-APPLICABLE", "reason": "no window where both have a value"})
            out["pairs"][f"{a}__minus__{b}"] = pr
    return out, fl


def per_window_file(pw, fl, arm_names, key):
    d = {f"{key}__eid": pw["eid"], f"{key}__state": pw["state"], f"{key}__is_lead": pw["is_lead"], f"{key}__speed_mps": pw["speeds"]}
    for nm in ["human"] + list(arm_names):
        for k in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
            d[f"{key}__{nm}__{k}"] = np.asarray(pw[f"{nm}__{k}"], np.float64)
        for k in ("tight_tg", "ttc_viol", "lead_viol"):
            d[f"{key}__{nm}__flag_{k}"] = fl[nm][k]
        d[f"{key}__{nm}__at_risk"] = fl[nm]["at_risk"]
    return d


# ------------------------------------------------------------------------------------------------ the read
def run_read(arms: dict, pairs, blocks, n_boot, seed, p10=True, lm_override=None):
    """arms: name -> (pred, saved_ade6, prov).  Returns (json_dict, perwindow_dict)."""
    M = boot(lm_override)
    DK = M["DK"]
    G = load_gt()
    preds = {n: v[0] for n, v in arms.items()}
    saved = {n: v[1] for n, v in arms.items() if v[1] is not None}
    prov = {n: v[2] for n, v in arms.items()}
    floor_ade = None
    if p10:
        pp = f"{REF}/p10_eval_pred6.npy"
        preds["P10"] = np.load(pp).astype(np.float64)
        prov["P10"] = {"path": pp, "sha256": sha256(pp)}
        floor_ade = np.load(f"{REF}/floors_perwindow.npz")["P10_refit__ade6"]
    names = list(preds)
    pairs = [(a, b) for a, b in pairs if a in preds and b in preds]
    reads, perwindow, extra = {}, {}, {}
    for bp, pin in blocks:
        _need(bp, pin)
        blk = DK.load_block(bp)
        k = int(blk["_k"])
        res = DK.read_dk(preds=preds, gt=G["gt"], eid=G["eid"], clip=G["clip"], frame=G["frame"], v0=G["v0"], block=blk, k=k, dt=G["dt"], n_boot=n_boot, seed=seed,
                         saved_ade6=saved, p10_floor_ade6=floor_ade, pairs=pairs)
        key = f"k{k}"
        pw = res["perwindow"]
        ex, fl = extra_rows(pw, names, pairs, n_boot, seed)
        for nm in names:                                                       # cross-check my flag against the instrument's own count
            nv = res["read"]["arms"][nm]["lead_viol"].get("n_viol")
            mine = int(np.nansum(fl[nm]["lead_viol"]))
            if nv is not None and nv != mine:
                refuse(f"{key}/{nm}: my LEAD-VIOL flags sum to {mine}, lead_metrics.lead_viol says {nv}")
        reads[key] = res["read"]
        extra[key] = ex
        perwindow.update(per_window_file(pw, fl, names, key))
    out = {"schema": SCHEMA, "tier": TIER, "estimator": ESTIMATOR, "lead_agent_layer": LAYER, "n_windows": int(G["eid"].size), "n_episodes": int(np.unique(G["eid"]).size),
           "arms": prov, "pairs": [f"{a}-{b}" for a, b in pairs],
           "pins": {"v8f_dk_read": PIN_DK_READ, "lead_metrics": PIN_LEAD_METRICS, "ci": PIN_CI, "block_k10": PIN_BLOCK_K10, "block_k30": PIN_BLOCK_K30, "gt141": PIN_GT,
                    "tip": "3a8d4761e81b8ab731ffc51157eb5331bb865b83"},
           "no_bar_registered": "no distance-keeping bar is registered for v8F-mini: nothing here is PASS / FAIL; a positive delta = MORE gap/time/TTC, not 'better'",
           "reads": reads, "extra": extra}
    return out, perwindow


# ------------------------------------------------------------------------------------------------ the table
def _f(x, p=2):
    return "n/a" if x is None else f"{x:.{p}f}"


def _ci_s(m, p=2):
    return "n/a" if not m or "mean" not in m else f"{m['mean']:.{p}f} [{m['lo']:.{p}f}, {m['hi']:.{p}f}]"


def table_md(out) -> str:
    L = ["# dk141 table (descriptive; decides nothing)", "", f"Tier: {out['tier']}", f"Estimator: {out['estimator']}", f"Lead agent layer: {out['lead_agent_layer']}", ""]
    for key, r in out["reads"].items():
        ex = out["extra"][key]
        L += [f"## {key}: {r['horizon']['seconds']} s horizon  (window states {r['join']['window_states']}; speed proof max {r['join']['speed_proof_max_mps']:.2e} m/s)", "",
              "| run | n hw | headway m | n tg | time gap s | n TTC (closing) | min-TTC s | TTC<1.5 s: n, rate | tg<1.0 s rate | LEAD-VIOL at-risk | progress |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        rows = [("human", r["human_reference"])] + list(r["arms"].items())
        for nm, a in rows:
            m = a["metrics"]
            lvr = a.get("lead_viol") or {}
            ev = ex["arms"][nm]
            tv, tt = ev["ttc_viol"], ev["tight_tg"]
            lv = ("0 by construction" if nm == "human" else f"{lvr.get('n_viol')}/{lvr.get('n_at_risk')} = {100 * (lvr.get('share_of_at_risk') or 0):.2f} %")
            pg = "1" if nm == "human" else (f"{_f((lvr.get('progress') or {}).get('lead_progress_ratio_clipped'), 3)}")
            L.append(f"| {nm} | {m['headway_min_m']['n']} | {_ci_s(m['headway_min_m'])} | {m['time_gap_min_s']['n']} | {_ci_s(m['time_gap_min_s'])} | "
                     f"{m['min_ttc_s']['n']} ({m['min_ttc_s'].get('n_closing')}) | {_ci_s(m['min_ttc_s'])} | {tv['n']}, {_ci_s(tv, 3)} | {_ci_s(tt, 3)} | {lv} | {pg} |")
        L += ["", "Paired deltas a - b (delta [lo, hi]; sep = CI excludes 0; positive = MORE gap/time/TTC, or MORE violations on the rate rows):", "",
              "| a - b | headway m | time gap s | min-TTC s | TTC<1.5 s rate | tg<1.0 s rate | LEAD-VIOL at-risk |", "|---|---|---|---|---|---|---|"]
        for pk, pr in r["pairs"].items():
            a, b = pk.split("__minus__")
            d = pr["dk"]

            def cell(x, p=3):
                if not x or x.get("status") not in (None, "OK") or "delta" not in x:
                    return "n/a"
                return f"{x['delta']:+.{p}f} [{x['lo']:+.{p}f}, {x['hi']:+.{p}f}] {'sep' if x.get('separated') else 'ns'}"
            e = ex["pairs"].get(pk, {})
            ar = (pr.get("lead_viol") or {}).get("at_risk") or {}
            L.append(f"| {a} - {b} | {cell(d['headway_min_m'], 2)} | {cell(d['time_gap_min_s'])} | {cell(d['min_ttc_s'], 2)} | {cell(e.get('ttc_viol', {}))} | {cell(e.get('tight_tg', {}))} | {cell(ar, 4)} |")
        L.append("")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ controls
def _oracle_headway(preds_dummy, leads, lens, speeds, n_steps=None):
    """INDEPENDENT re-derivation of the headway of a STOPPED-FOREVER ego at the origin (heading 0): min over steps of (lead_x - len/2) where gap >= 0 and |lead_y| < 2.
    Written without `lead_metrics` so a mutated instrument cannot agree with it by accident."""
    lead_x, lead_y = leads[:, :, 0], leads[:, :, 1]
    gap = lead_x - lens[:, None] / 2.0
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(gap) & np.isfinite(lead_y) & (gap >= 0.0) & (np.abs(lead_y) < 2.0)
    g = np.where(ok, gap, np.inf)
    h = g.min(axis=1)
    return np.where(np.isfinite(h) & ok.any(axis=1), h, np.nan)


def make_through(leads, lens, is_lead, k, margin=1e-6):
    """A plan that DRIVES INTO the lead: constant velocity along the bearing of the lead, reaching 1 micrometre short of its rear face at step s*, then continuing through it.
    s* = the last step index <= k//2 - 1 with a finite lead centre.  Windows without a lead stay a straight 10 m/s line (not scored anyway)."""
    n = leads.shape[0]
    plan = np.zeros((n, leads.shape[1], 2))
    plan[:] = (np.arange(1, leads.shape[1] + 1)[None, :, None] * DT) * np.array([10.0, 0.0])[None, None, :]
    used = np.zeros(n, bool)
    for i in np.flatnonzero(is_lead):
        cand = [j for j in range(k // 2) if np.all(np.isfinite(leads[i, j]))]
        if not cand or not np.isfinite(lens[i]):
            continue
        s = cand[-1]
        c = leads[i, s]
        r = float(np.hypot(*c))
        if r < 1e-6:
            continue
        u = c / r
        tgt = c - (lens[i] / 2.0 + margin) * u
        plan[i] = ((np.arange(1, leads.shape[1] + 1) / float(s + 1))[:, None]) * tgt[None, :]
        used[i] = True
    return plan, used


def _chk(c, name, ok, detail):
    c.append({"control": name, "pass": bool(ok), "detail": detail})
    return bool(ok)


def run_controls(n_boot, seed, lm_override=None):
    M = boot(lm_override)
    DK, lm, ci = M["DK"], M["lm"], M["ci"]
    G = load_gt()
    checks, report = [], {}
    # ---- C0: reproduce the BANKED read (dk_read.json, committed at the tip) for M2b_s0 / M2b_s1 / P10 to 1e-9 -------------------------------------------------
    banked = json.load(open(f"{REF}/dk_read_banked.json", encoding="utf-8"))
    arms = {}
    for t in ("M2b_s0", "M2b_s1"):
        arms[t] = load_arm(t, f"{REF}/eval_{t}.npz", G["eid"])
    pairs = [("M2b_s0", "P10"), ("M2b_s1", "P10")]
    out, pw = run_read(arms, pairs, blocks_default(), n_boot, seed, True, lm_override)
    worst, ncmp, worst_pt = 0.0, 0, 0.0
    for key in ("k10", "k30"):
        mine, ref = out["reads"][key], banked["reads"][key]
        for nm in ("M2b_s0", "M2b_s1", "P10"):
            for kk in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
                a, b = mine["arms"][nm]["metrics"][kk], ref["arms"][nm]["metrics"][kk]
                for f in ("mean", "lo", "hi"):
                    worst, ncmp = max(worst, abs(a[f] - b[f])), ncmp + 1
                worst_pt = max(worst_pt, abs(a["mean"] - b["mean"]))
                if a["n"] != b["n"]:
                    worst = max(worst, 1.0)
                    worst_pt = max(worst_pt, 1.0)
            la, lb = mine["arms"][nm]["lead_viol"], ref["arms"][nm]["lead_viol"]
            for f in ("n_viol", "n_at_risk", "share_of_at_risk"):
                worst, ncmp = max(worst, abs(la[f] - lb[f])), ncmp + 1
                worst_pt = max(worst_pt, abs(la[f] - lb[f]))
            worst = max(worst, abs(la["lead_progress_ratio_clipped"] - lb["lead_progress_ratio_clipped"]))
            worst_pt = max(worst_pt, abs(la["lead_progress_ratio_clipped"] - lb["lead_progress_ratio_clipped"]))
        for kk in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
            for f in ("mean", "lo", "hi"):
                worst, ncmp = max(worst, abs(mine["human_reference"]["metrics"][kk][f] - ref["human_reference"]["metrics"][kk][f])), ncmp + 1
        for pk in ("M2b_s0__minus__P10", "M2b_s1__minus__P10"):
            for kk in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
                for f in ("delta", "lo", "hi"):
                    worst, ncmp = max(worst, abs(mine["pairs"][pk]["dk"][kk][f] - ref["pairs"][pk]["dk"][kk][f])), ncmp + 1
                worst_pt = max(worst_pt, abs(mine["pairs"][pk]["dk"][kk]["delta"] - ref["pairs"][pk]["dk"][kk]["delta"]))
        st = mine["join"]["window_states"] == ref["join"]["window_states"]
        _chk(checks, f"C0 window states reproduce the banked read ({key})", st, mine["join"]["window_states"])
    _chk(checks, "C0 banked read reproduced: max |dk141 - banked dk_read.json| over %d cells" % ncmp, worst <= 1e-9, f"max abs diff {worst:.3e} (tol 1e-9)")
    _chk(checks, "C0p point values (means, n, n_viol, progress, delta points) reproduce the banked read (bootstrap-free)", worst_pt <= 1e-9, f"max abs diff {worst_pt:.3e}")
    report["C0_cells"], report["C0_max_abs_diff"], report["C0p_max_abs_diff"] = ncmp, worst, worst_pt
    # ---- the kinematic contract of the human path (the block's own gt_* columns) ----------------------------------------------------------------------------
    c10 = out["reads"]["k10"]["human_reference"]["kinematic_contract"]
    _chk(checks, "C1 kinematic contract: recomputed human headway vs the block's gt_headway_min_m", c10.get("status") == "OK" and c10["median_abs_headway_diff_m"] <= 0.25,
         {k: c10.get(k) for k in ("status", "median_abs_headway_diff_m", "n_both", "n_block_finite")})
    # ---- the human's own distribution (a stated reference) -------------------------------------------------------------------------------------------------
    hum = {}
    for key in ("k10", "k30"):
        d = {}
        for kk in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
            v = np.asarray(pw[f"{key}__human__{kk}"], np.float64)
            v = v[np.isfinite(v)]
            d[kk] = {"n": int(v.size), **{f"p{q}": round(float(np.percentile(v, q)), 3) for q in (10, 25, 50, 75, 90)}, "mean": round(float(v.mean()), 3)}
        for fk in ("tight_tg", "ttc_viol"):
            v = np.asarray(pw[f"{key}__human__flag_{fk}"], np.float64)
            ok = np.isfinite(v)
            d[f"{fk}_rate"] = {"n": int(ok.sum()), "n_flagged": int(v[ok].sum()), "rate": round(float(v[ok].mean()), 4)}
        hum[key] = d
    report["human_own_distribution"] = hum
    bn = {k: banked["reads"][k]["human_reference"]["metrics"]["time_gap_min_s"]["n"] for k in ("k10", "k30")}
    _chk(checks, "C2b tight-gap rate is over exactly the windows with a defined time gap (n equals the banked human time-gap n)", all(hum[k]["tight_tg_rate"]["n"] == bn[k] for k in hum), {"mine": {k: hum[k]["tight_tg_rate"]["n"] for k in hum}, "banked": bn})
    _chk(checks, "C2 the human's own distribution is stated and non-degenerate (headway n>=700, p10<p50<p90)", all(hum[k]["headway_min_m"]["n"] >= 700 and hum[k]["headway_min_m"]["p10"] < hum[k]["headway_min_m"]["p50"] < hum[k]["headway_min_m"]["p90"] for k in hum),
         {k: {"n": hum[k]["headway_min_m"]["n"], "p10/p50/p90": [hum[k]["headway_min_m"][q] for q in ("p10", "p50", "p90")]} for k in hum})
    # ---- synthetic plans on the REAL lead data --------------------------------------------------------------------------------------------------------------
    syn = {}
    for bp, pin in blocks_default():
        blk = DK.load_block(bp)
        k = int(blk["_k"])
        key = f"k{k}"
        J = DK.join_block(blk, G["clip"], G["frame"], G["v0"], k=k, dt=G["dt"])
        hdk = lm.distance_keeping(G["gt"][:, :k], J["leads"], J["lead_lens"], J["speeds"], DT)
        lead = J["is_lead"]
        # STOP: stopped forever at the origin
        stop = np.zeros((G["gt"].shape[0], k, 2))
        dk_s = lm.distance_keeping(stop, J["leads"], J["lead_lens"], J["speeds"], DT)
        ab_s, lv_s = DK.arm_block("STOP", dk_s, hdk, J, G["gt"][:, :k], stop, G["eid"], n_boot, seed)
        orc = _oracle_headway(None, J["leads"], J["lead_lens"], J["speeds"])
        hs = np.asarray(dk_s["headway_min_m"], np.float64)
        same_nan = np.array_equal(np.isfinite(hs), np.isfinite(orc))
        dmax = float(np.nanmax(np.abs(hs - orc))) if np.isfinite(orc).any() else 0.0
        _chk(checks, f"C3 STOP headway == independent oracle, all windows, same NaN pattern ({key})", same_nan and dmax <= 1e-9, f"max |diff| {dmax:.2e}; n finite {int(np.isfinite(orc).sum())}")
        hh = np.asarray(hdk["headway_min_m"], np.float64)
        both = np.isfinite(hs) & np.isfinite(hh)
        dom = hs[both] >= hh[both] - 1e-9
        below = np.flatnonzero(both)[~dom]
        report[f"C3_stop_dominates_human_{key}"] = {"n_both": int(both.sum()), "n_stop_ge_human": int(dom.sum()), "n_exceptions": int((~dom).sum()),
                                                    "exception_max_shortfall_m": (float((hh[both] - hs[both])[~dom].max()) if (~dom).any() else 0.0)}
        _chk(checks, f"C4 STOP keeps >= the human's headway on >=99 % of windows where both have one ({key})", dom.mean() >= 0.99, report[f"C3_stop_dominates_human_{key}"])
        ttcv = np.asarray(dk_s["min_ttc_s"], np.float64)
        n_viol_ttc = int((np.isfinite(ttcv) & (ttcv < TTC_VIOL_S)).sum())
        _chk(checks, f"C5 STOP has ZERO TTC violations (min TTC < {TTC_VIOL_S} s) ({key})", n_viol_ttc == 0, {"n_ttc_finite": int(np.isfinite(ttcv).sum()), "n_viol": n_viol_ttc, "n_closing": int(dk_s.get("n_closing", -1))})
        _chk(checks, f"C6 STOP has ZERO LEAD-VIOL and progress ratio EXACTLY 0.0, the crawl guard fires ({key})",
             lv_s["n_viol"] == 0 and ((lv_s.get("progress") or {}).get("lead_progress_ratio_clipped") == 0.0) and ((lv_s.get("progress") or {}).get("meets_min_progress") is False),
             {"n_viol": lv_s["n_viol"], "n_at_risk": lv_s["n_at_risk"], "progress_clipped": (lv_s.get("progress") or {}).get("lead_progress_ratio_clipped"), "meets_min_progress": (lv_s.get("progress") or {}).get("meets_min_progress")})
        # THROUGH: drives into the lead
        thr, used = make_through(J["leads"], J["lead_lens"], lead, k)
        dk_t = lm.distance_keeping(thr, J["leads"], J["lead_lens"], J["speeds"], DT)
        ab_t, lv_t = DK.arm_block("THROUGH", dk_t, hdk, J, G["gt"][:, :k], thr, G["eid"], n_boot, seed)
        pwt = lv_t["_per_window"]
        at_risk = pwt["at_risk"] == 1.0
        flagged = pwt["viol"] == 1.0
        tg_t = np.asarray(dk_t["time_gap_min_s"], np.float64)
        tt_t = np.asarray(dk_t["min_ttc_s"], np.float64)
        lost = lead & ~np.isfinite(np.asarray(dk_t["headway_min_m"], np.float64))
        n_ar, n_fl = int(at_risk.sum()), int((flagged & at_risk).sum())
        _chk(checks, f"C7 THROUGH flags EVERY at-risk lead window (registered LEAD-VIOL) ({key})", n_ar > 0 and n_fl == n_ar and int((lost & at_risk).sum()) == 0,
             {"n_lead_windows": int(lead.sum()), "plan_built_for": int(used.sum()), "n_at_risk": n_ar, "n_flagged": n_fl, "n_plan_lost_lead_at_risk": int((lost & at_risk).sum()),
              "share_of_at_risk": lv_t["share_of_at_risk"], "share_incl_lost_lead": lv_t["share_incl_lost_lead"]})
        fin_t = lead & np.isfinite(tt_t)
        nt = int(fin_t.sum())
        miss = np.flatnonzero(fin_t & (tt_t >= TTC_VIOL_S))
        ns_in = np.asarray(dk_t["n_steps_in_corridor"])
        _chk(checks, f"C8 THROUGH: >=99 % of lead windows read min TTC < {TTC_VIOL_S} s, and every miss has a SHORT admissible span (<= 4 corridor steps) ({key})",
             nt > 0 and (nt - miss.size) / nt >= 0.99 and bool((ns_in[miss] <= 4).all()),
             {"n_lead_with_ttc": nt, "n_ttc_viol": nt - int(miss.size), "n_miss": int(miss.size), "miss_n_steps_in_corridor": ns_in[miss].tolist(),
              "miss_ttc_s": [round(float(x), 2) for x in tt_t[miss]], "finding": "a window whose plan has ONE admissible corridor step reads TTC censored at the cap (lead_metrics.py:190-191) although its headway is ~0; the time-gap / LEAD-VIOL rows still flag it",
              "n_lost_lead": int(lost.sum()), "max_headway_m": float(np.nanmax(np.asarray(dk_t['headway_min_m'])[lead])),
              "n_tg_lt_1": int((lead & np.isfinite(tg_t) & (tg_t < 1.0)).sum()), "n_tg_defined": int((lead & np.isfinite(tg_t)).sum())})
        pwx = {"eid": G["eid"], "is_lead": lead, "state": J["state"], "speeds": J["speeds"]}
        for nm_, d_ in (("human", hdk), ("STOP", dk_s), ("THROUGH", dk_t)):
            for kk_ in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
                pwx[f"{nm_}__{kk_}"] = np.asarray(d_[kk_], np.float64)
        exx, flx = extra_rows(pwx, ["STOP", "THROUGH"], [("THROUGH", "STOP")], n_boot, seed)
        dlt = exx["pairs"]["THROUGH__minus__STOP"]["ttc_viol"]
        va_, vb_ = flx["THROUGH"]["ttc_viol"], flx["STOP"]["ttc_viol"]
        okc = np.isfinite(va_) & np.isfinite(vb_)
        rt = float(va_[okc].mean() - vb_[okc].mean())                       # the difference of the two rates on the windows where BOTH have a TTC
        _chk(checks, f"C9 paired TTC-violation delta THROUGH - STOP is POSITIVE, separated, and equals the difference of the two arm rates to the estimator's 4 dp ({key})",
             dlt.get("delta", -1) > 0.9 and bool(dlt.get("separated")) and abs(dlt["delta"] - rt) <= 1e-4 and exx["arms"]["STOP"]["ttc_viol"]["n_flagged"] == 0,
             {"delta": dlt.get("delta"), "lo": dlt.get("lo"), "hi": dlt.get("hi"), "diff_of_rates": rt, "n_used": dlt.get("n_used")})
        syn[key] = {"STOP": {"arm_block_metrics": {m: ab_s["metrics"][m] for m in ab_s["metrics"]}, "lead_viol": ab_s["lead_viol"]},
                    "THROUGH": {"arm_block_metrics": {m: ab_t["metrics"][m] for m in ab_t["metrics"]}, "lead_viol": ab_t["lead_viol"],
                                "n_lost_lead": int(lost.sum()), "n_lost_lead_at_risk": int((lost & at_risk).sum())}}
    report["synthetic"] = syn
    allpass = all(c["pass"] for c in checks)
    return {"schema": SCHEMA + "-controls", "mutant_lead_metrics": lm_override, "all_pass": allpass, "n_checks": len(checks), "checks": checks, "report": report}


# ------------------------------------------------------------------------------------------------ io
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        return _clean(o.item())
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("read")
    r.add_argument("--arm", action="append", default=[], help="NAME=path/to/eval_<tag>.npz (repeatable)")
    r.add_argument("--no-p10", action="store_true")
    r.add_argument("--pair", action="append", default=[], help="A:B (delta A - B); default every --arm against P10")
    r.add_argument("--out", required=True)
    r.add_argument("--table-md")
    r.add_argument("--perwindow-out")
    r.add_argument("--n-boot", type=int, default=B)
    r.add_argument("--seed", type=int, default=0)
    c = sp.add_parser("controls")
    c.add_argument("--out", required=True)
    c.add_argument("--n-boot", type=int, default=B)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--mutant-lead-metrics", help="TEST HOOK (mutation runs only): load lead_metrics from this file")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "read":
            if not a.arm:
                refuse("give at least one --arm NAME=path")
            G = load_gt()
            arms = {}
            for s in a.arm:
                if "=" not in s:
                    refuse(f"--arm {s!r} is not NAME=path")
                n, p = s.split("=", 1)
                if n in arms or n in ("P10", "human"):
                    refuse(f"arm name {n!r} is duplicated or reserved")
                arms[n] = load_arm(n, p, G["eid"])
            pairs = [tuple(x.split(":", 1)) for x in a.pair] or [(n, "P10") for n in arms]
            out, pw = run_read(arms, pairs, blocks_default(), a.n_boot, a.seed, not a.no_p10)
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(_clean(out), f, indent=1, default=str)
            if a.table_md:
                with open(a.table_md, "w", encoding="utf-8", newline="\n") as f:
                    f.write(table_md(out))
            if a.perwindow_out:
                np.savez_compressed(a.perwindow_out, **pw)
            print(table_md(out))
            return 0
        res = run_controls(a.n_boot, a.seed, a.mutant_lead_metrics)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(_clean(res), f, indent=1, default=str)
        for ch in res["checks"]:
            print(("PASS " if ch["pass"] else "FAIL ") + ch["control"])
        print("ALL PASS" if res["all_pass"] else "CONTROL FAILURE")
        return 0 if res["all_pass"] else 3
    except Refusal as e:
        print(f"DK141 REFUSED: {e}", file=sys.stderr)
        return 2
    except Exception as e:                                                   # the pinned reader's own guards (v8f_dk_read.Refusal) are refusals too
        if type(e).__name__ == "Refusal":
            print(f"DK141 REFUSED (pinned reader guard): {e}", file=sys.stderr)
            return 2
        raise
    except (OSError, KeyError, ValueError) as e:
        print(f"DK141 INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
