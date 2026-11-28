"""X7 LEAD-VIOL on refcv8's per-frame S6 panels (CPU only; EvalFlyWheel 2026-10-20).

Adapted call-for-call from the landed refcv7 baseline scripts
  FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-11-r7base-fourfamily/code/r7base_stageB.py   (join)
  FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-11-r7base-fourfamily/code/r7base_stageD_leadprog.py (registered progress control)
and the tip's refcv3_arm._distance_keeping (lead_block_common_grid + join_lead_block + lead_metrics.*).
The instrument is taniteval/lead_metrics.py as it stands (blob bd89520d, identical in the C:/lgs/Lfb47a58376 tree and in
refs/remotes/origin/agent/arch-inf-20260803); it is imported, not copied.

usage: python x7_r8.py <out_json>
Nothing is written outside the output directory. No clip id is written: episodes are file indices, the panel identity is a sha12.
"""
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402

OUT_DIR = Path("C:/Users/Admin/refcv8_pf/x7_48852")
WORK = OUT_DIR / "work"
WORK.mkdir(exist_ok=True)
out_json = Path(sys.argv[1])

import tanitad  # noqa: E402
from taniteval import lead_metrics as lm  # noqa: E402
from taniteval import ci as ci_mod  # noqa: E402
from taniteval.ci import episode_cluster_bootstrap  # noqa: E402

print("tanitad", tanitad.__file__, "| lead_metrics", lm.__file__, "| ci", ci_mod.__file__, flush=True)
TIP = "C:/lgs/Lfb47a58376/taniteval/tools/refcv3_arm.py"
_spec = importlib.util.spec_from_file_location("refcv3_arm", TIP)
ra3 = importlib.util.module_from_spec(_spec)
sys.modules["refcv3_arm"] = ra3
_spec.loader.exec_module(ra3)
ra = ra3.ra  # refav1_arm
assert hasattr(lm, "path_arc_length") and hasattr(lm, "lead_viol_verdict") and hasattr(lm, "paired_lead_viol")

LEAD12 = "C:/Users/Admin/qland/landed_hold_x7/artifacts/b1_eval_lead_block_k12_dt0p5.npz"
LEAD12_MD5 = "c346dff3b3d356f55ef2786f327c6c91"
N_BOOT = 2000
SEEDS = (0, 1)
TREES = {
    "refcv8_48852": "C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf",
    "refcv8_40k": "C:/Users/Admin/refcv8_pf/battery40k_pf/step40000ema_pf",
}
R7_KIT = "D:/refcv7_eval_kit/battery/step50400_a6text"
BANKED_S6 = {s: json.load(open(OUT_DIR / "ref" / f"leadprog_s6_s{s}.json", encoding="utf-8"))["arms"] for s in SEEDS}
BANKED_S2 = {s: json.load(open(OUT_DIR / "ref" / f"lead_viol_s{s}.json", encoding="utf-8"))["S2"]["arms"] for s in SEEDS}
BANKED_CTRL = {s: json.load(open(OUT_DIR / "ref" / f"leadprog_s6_s{s}.json", encoding="utf-8"))["controls_internal"] for s in SEEDS}
SURFACES = {
    # name: (panel dir pattern relative to a tree/kit root, banked arms, has registered progress control banked, role)
    "S6": ("s6_s{s}", BANKED_S6, True, "THE BAR SURFACE (6 s): the 2 s block is nearly blind for X7 (X7 package, r7base RESULT s4)"),
    "S2": ("panel_s{s}", BANKED_S2, False, "2 s surface, beside S6 (SPEC s8.7 names S2 and S-ROUTE); refcv7 has ~5-8 violations here, so it is near-blind"),
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha12(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


def load_panel(panel_dir):
    """Same contract as r7_panel.load_panel: ep*.npz sorted, eid = file index."""
    man = json.load(open(os.path.join(panel_dir, "manifest.json"), encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(panel_dir, "ep*.npz")))
    arrs, eid, ws_l, v0_l = {}, [], [], []
    for fi, f in enumerate(files):
        with np.load(f) as z:
            n = z["g"].shape[0]
            eid += [fi] * n
            ws_l.append(np.asarray(z["ws"]).astype(np.int64).reshape(-1))
            v0_l.append(np.asarray(z["v0"], dtype=np.float32).reshape(-1))
            for k in z.files:
                if k in ("eid", "clip_index", "ws", "v0"):
                    continue
                arrs.setdefault(k, []).append(z[k])
    arrs = {k: np.concatenate(v) for k, v in arrs.items()}
    arrs["_ws"] = np.concatenate(ws_l)
    arrs["_v0"] = np.concatenate(v0_l)
    return man, files, arrs, np.asarray(eid)


_LEAD_CACHE = {}


def join(panel_dir, man, files, tag):
    """r7base_stageB.join, call for call (lead_block_common_grid + join_lead_block with identity frame map)."""
    fr = man["corpus"]["frames"]
    raw_off = int(fr["provider_to_raw_frame_offset"])
    dt, k = float(man["grid"]["dt_s"]), int(man["grid"]["k"])
    if (dt, k) not in _LEAD_CACHE:
        _LEAD_CACHE[(dt, k)] = ra3.lead_block_common_grid(LEAD12, dt, k)
    view, idx, meta, info = _LEAD_CACHE[(dt, k)]
    assert view is not None, info
    shim_dir = WORK / f"shim_{tag}"
    shim_dir.mkdir(exist_ok=True)
    shim = []
    for f in files:
        with np.load(f) as d:
            ws = np.asarray(d["ws"]).astype(np.int64).reshape(-1)
            v0 = np.asarray(d["v0"], dtype=np.float32).reshape(-1)
        sf = str(shim_dir / os.path.basename(f))
        np.savez(sf, ws=ws + raw_off, v0=v0)
        shim.append(sf)
    lead = ra.join_lead_block(shim, man, view, idx, k=info["k"], dt=info["dt_s"], frame_of_t=lambda t: t)
    cov = lead.pop("coverage")
    return lead, cov, info, raw_off


def ci_mean(x, ids):
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if not ok.any():
        return {"status": "NOT-APPLICABLE", "n": 0}
    r = episode_cluster_bootstrap(x[ok], [e for e, m in zip(ids, ok) if m], reduce="mean", n_boot=N_BOOT, seed=0)
    return {k: r[k] for k in ("mean", "lo", "hi", "n_windows", "n_episodes", "estimator") if k in r}


def compact(lv):
    """The instrument's own numbers, verbatim (r7base_stageD.compact)."""
    pr = lv["progress"]
    out = {k: lv.get(k) for k in ("status", "n_windows", "n_lead_windows", "n_viol", "share", "n_at_risk",
                                  "share_of_at_risk", "n_plan_lost_lead", "share_incl_lost_lead",
                                  "n_human_tg_undefined", "n_human_tg_below_thr")}
    ci = lv.get("ci") or {}
    out["share_ci"] = {k: ci.get(k) for k in ("mean", "lo", "hi", "n_windows", "n_episodes", "estimator")}
    sc = lv.get("share_of_at_risk_ci") or {}
    out["share_of_at_risk_ci"] = {k: sc.get(k) for k in ("mean", "lo", "hi", "n_windows", "n_episodes", "estimator")}
    out["progress"] = {k: pr.get(k) for k in (
        "status", "n_at_risk", "n", "n_arc_nonfinite", "n_human_arc_degenerate", "min_progress", "min_human_arc_m",
        "decision_reading", "lead_progress_ratio", "lead_progress_ratio_clipped", "progress_ratio_of_means",
        "meets_min_progress", "overshoot_masks_crawl", "n_below_min", "reason")}
    return out


def verdict_compact(v):
    return {k: v.get(k) for k in ("verdict", "bar_decision", "lead_progress_ratio_clipped", "lead_progress_ratio",
                                  "n_progress", "overshoot_masks_crawl", "min_progress", "reason")}


def paired_compact(p):
    ar = p.get("at_risk") or {}
    pg = p.get("progress") or {}
    top = {k: p.get(k) for k in ("status", "n_lead_windows", "n_viol_a", "n_viol_b", "delta", "lo", "hi", "separated",
                                 "n_episodes", "estimator")}
    return {
        "x7_direction_met_at_risk": p.get("x7_direction_met_at_risk"),
        "at_risk_denominator": {k: ar.get(k) for k in ("status", "n_at_risk", "n_viol_a", "n_viol_b", "delta", "lo", "hi",
                                                       "separated", "estimator", "n_episodes")},
        "all_lead_windows_reading": top,
        "progress_paired": {k: pg.get(k) for k in ("status", "n_used", "n_a_only", "n_b_only", "lead_progress_ratio_a",
                                                   "lead_progress_ratio_b", "delta", "lo", "hi", "separated")},
    }


def dk_level(dk, eid):
    pw = dk.get("_per_window", dk)
    return {
        "status": dk.get("status"), "n": dk.get("n"),
        "mean_headway_min_m": ci_mean(pw["headway_min_m"], eid),
        "mean_time_gap_min_s": ci_mean(pw["time_gap_min_s"], eid),
        "mean_min_ttc_s": ci_mean(pw["min_ttc_s"], eid),
        "n_closing": dk.get("n_closing"), "n_time_gap": dk.get("n_time_gap"),
        "censoring_note": dk.get("censoring_note"),
    }


def paired_dk_compact(p):
    out = {}
    for key, blk in p["metrics"].items():
        out[key] = {k: blk.get(k) for k in ("status", "n_used", "n_windows", "n_a_only", "n_b_only", "delta", "lo", "hi",
                                            "separated", "n_episodes")}
    return out


def ade(plan, g):
    return float(np.mean(np.linalg.norm(plan - g, axis=-1)))


# --------------------------------------------------------------------------------------------------------------- #
res = {"meta": {
    "what": "X7 LEAD-VIOL (SPEC_REFCV8 s8.7(a)) + distance keeping, refcv8 per-frame trees vs refcv7-50,400",
    "tier": "T1 self-action OPEN LOOP on logged eval139 frames (NOT driving performance)",
    "training_seeds": "ONE per model (refcv8: 1; refcv7-50,400: 1). Training-seed floor unmeasured.",
    "estimator": ("taniteval.ci episode-cluster bootstrap (single arm) / paired_episode_cluster_bootstrap (deltas), "
                  f"B={N_BOOT}, seed 0. It answers 'another draw of EPISODES' ONLY; it is blind to another TRAINING run "
                  "(H-ESTIM-SEED-1) and 'separated' is necessary, not sufficient, for a lever effect. The inference-seed "
                  "replicate (s0 vs s1) is reported beside it."),
    "instrument": {"lead_metrics": lm.__file__, "lead_metrics_blob": "bd89520d7537aa1c65c23a89708443878b825c14 (equal in repo ref 75ce238d and in C:/lgs/Lfb47a58376)",
                   "ci": ci_mod.__file__, "ci_blob": "54b8895bbc47bf5f990ba6cddf999a91eaca87de (equal in both)",
                   "join": "refcv3_arm.lead_block_common_grid + refav1_arm.join_lead_block from " + TIP +
                           " (function sources identical to refs/remotes/origin/agent/arch-inf-20260803 @75ce238d, sha256-checked)"},
    "lead_block": {"path": LEAD12, "md5": None, "expected_md5": LEAD12_MD5},
    "definition": "LEAD-VIOL: plan min time gap < 1.0 s while the human's >= 1.0 s; bar denominator = share_of_at_risk (PI ruling 2026-10-11); admissible only if lead_progress_ratio_clipped >= 0.90",
    "surfaces": {k: v[3] for k, v in SURFACES.items()},
    "trees": TREES, "refcv7_kit": R7_KIT,
}, "surfaces": {}}
res["meta"]["lead_block"]["md5"] = md5(LEAD12)
assert res["meta"]["lead_block"]["md5"] == LEAD12_MD5, "lead block md5 differs from the one the refcv7 baseline used"
PW = {}  # per-window violation flags kept in memory for the exploratory pooled-draw read (never written)


def run_seed(surface, s):
    pat, banked_all, has_prog_banked, _role = SURFACES[surface]
    banked = banked_all[s]
    t_seed = time.time()
    S = {"panels": {}, "arms": {}, "controls_internal": {}, "paired": {}, "paired_distance_keeping": {}, "checks": {}}
    P = {}
    panels = {name: os.path.join(d, pat.format(s=s)) for name, d in TREES.items()}
    panels["refcv7_own"] = os.path.join(R7_KIT, pat.format(s=s))
    for name, pd in panels.items():
        man, files, arrs, eid = load_panel(pd)
        P[name] = (man, files, arrs, eid)
        clips = [e.get("clip_id") for e in man["episodes"]]
        S["panels"][name] = {"panel_dir": pd, "n_windows": int(arrs["g"].shape[0]), "n_episodes": int(len(files)),
                             "model_step": man["model"].get("step"), "ckpt": man["model"].get("ckpt"),
                             "inference_seed": man["inference_seed"].get("seed"),
                             "nav_feed_mode": (man.get("nav_feed") or {}).get("mode"),
                             "episode_order_sha12": sha12("|".join(map(str, clips)))}
    ref_name = "refcv8_48852"
    _, _, A0, E0 = P[ref_name]
    ident = {}
    for name in P:
        _, _, A, E = P[name]
        ident[name] = {
            "ws_equal": bool(np.array_equal(A["_ws"], A0["_ws"])),
            "v0_equal": bool(np.array_equal(A["_v0"], A0["_v0"])),
            "gt_g_equal": bool(np.array_equal(A["g"], A0["g"])),
            "eid_equal": bool(np.array_equal(E, E0)),
            "episode_order_equal": S["panels"][name]["episode_order_sha12"] == S["panels"][ref_name]["episode_order_sha12"],
            "model_free_arms_equal": {b: bool(np.array_equal(A[b], A0[b])) for b in ("ha0", "ha0_ext", "ha", "stop")},
        }
        assert all(v for k, v in ident[name].items() if k != "model_free_arms_equal"), (name, ident[name])
    S["checks"]["window_grid_identity_vs_refcv8_48852"] = ident
    _, _, A7, _ = P["refcv7_own"]
    slot = {}
    for nm in (f"b_refcv7_50400_s{s}", f"b_refcv7_50400_s{1 - s}"):
        for tree in TREES:
            slot[f"{tree}:{nm}==refcv7_own.os"] = bool(np.array_equal(P[tree][2][nm], A7["os"]))
    S["checks"]["refcv7_baseline_slot"] = slot
    r7_key = f"b_refcv7_50400_s{s}"
    for tree in TREES:
        assert np.array_equal(P[tree][2][r7_key], A7["os"]), f"{tree}: {r7_key} != refcv7's own seed-{s} os"
    joins = {}
    for name in P:
        man, files, arrs, eid = P[name]
        joins[name] = join(panels[name], man, files, f"{surface}_{name}_s{s}")
    lead, cov, info, raw_off = joins[ref_name]
    for name, (ld, cv, inf, ro) in joins.items():
        assert np.array_equal(np.asarray(ld["state"]).astype(str), np.asarray(lead["state"]).astype(str)), name
        assert np.allclose(np.nan_to_num(ld["leads"], nan=-9), np.nan_to_num(lead["leads"], nan=-9)), name
        assert cv["counts"] == cov["counts"] and cv["speed_check"]["max_mps"] == cov["speed_check"]["max_mps"], name
    cols = info["dump_cols"]
    dtv = info["dt_s"]
    state = np.asarray(lead["state"]).astype(str)
    lv = state == "LEAD"
    W = len(state)
    assert W == A0["g"].shape[0] == len(E0)
    eid = E0
    S["join"] = {"raw_offset": raw_off, "instants_s": info["instants_s"], "dt_s": dtv, "k": info["k"],
                 "dump_cols": [int(c) for c in cols], "block_grid_s": info["block_grid_s"],
                 "state_counts": cov["counts"], "n_episodes_ok": cov["n_episodes_ok"], "n_episodes": cov["n_episodes"],
                 "n_windows_no_row": cov["n_windows_no_row"], "speed_check_max_mps": cov["speed_check"]["max_mps"],
                 "identical_across_all_panels": True}
    G = A0["g"][:, cols].astype(np.float64)
    human_dk = lm.distance_keeping(G, lead["leads"], lead["lead_lens"], lead["speeds"], dtv)
    human_arc = lm.path_arc_length(G)
    htg = np.asarray(human_dk.get("_per_window", human_dk)["time_gap_min_s"], dtype=np.float64)
    at_risk = lv & np.isfinite(htg) & (htg >= 1.0)
    S["join"]["n_windows"] = int(W)
    S["join"]["n_lead_windows"] = int(lv.sum())
    S["join"]["n_at_risk_recount"] = int(at_risk.sum())
    S["join"]["n_episodes_with_lead_window"] = int(len(set(eid[lv].tolist())))
    S["join"]["n_episodes_with_at_risk_window"] = int(len(set(eid[at_risk].tolist())))
    S["join"]["human_distance_keeping_level"] = dk_level(human_dk, eid)

    def score(plan_full):
        plan = plan_full[:, cols].astype(np.float64)
        dk = lm.distance_keeping(plan, lead["leads"], lead["lead_lens"], lead["speeds"], dtv)
        lvres = lm.lead_viol(dk, human_dk, eid, lead_valid=lv, n_boot=N_BOOT, seed=0,
                             plan_arc=lm.path_arc_length(plan), human_arc=human_arc)
        return plan, dk, lvres

    flags, dks, plans, arm_src = {}, {}, {}, {}
    for tree in TREES:
        arm_src[f"{tree}.os"] = P[tree][2]["os"]
    arm_src["refcv7_50400.os"] = P["refcv8_48852"][2][r7_key]            # as carried in the refcv8 panel (the paired read's baseline)
    arm_src["refcv7_50400.os@own_panel"] = A7["os"]                      # refcv7's own banked panel
    for b in ("ha0", "ha0_ext", "ha", "stop"):
        arm_src[b] = A0[b]
    for nm, plan_full in arm_src.items():
        plan, dk, lvres = score(plan_full)
        plans[nm], dks[nm], flags[nm] = plan, dk, lvres
        c = compact(lvres)
        c["verdict"] = verdict_compact(lm.lead_viol_verdict(lvres))
        c["distance_keeping_level_all_lead_windows"] = dk_level(dk, eid)
        c["ade_m_all_windows_this_surface"] = ade(plan_full.astype(np.float64), A0["g"].astype(np.float64))
        S["arms"][nm] = c
        print(f"{surface} seed {s} {nm:28s} n_lead {c['n_lead_windows']} viol {c['n_viol']} risk {c['n_at_risk']} "
              f"share_of_at_risk {c['share_of_at_risk']} ratio_clip {c['progress']['lead_progress_ratio_clipped']:.4f} "
              f"{c['verdict']['verdict']}", flush=True)
    # ---- regression controls that must read KNOWN values -------------------------------------------------------
    ci = S["controls_internal"]
    hv = lm.lead_viol(human_dk, human_dk, eid, lead_valid=lv, n_boot=200, seed=0, plan_arc=human_arc, human_arc=human_arc)
    ci["human_as_plan"] = {"lead_progress_ratio": hv["lead_progress_ratio"], "lead_progress_ratio_clipped": hv["lead_progress_ratio_clipped"],
                           "n_viol": hv["n_viol"], "share": hv["share"], "share_of_at_risk": hv["share_of_at_risk"],
                           "expected": "ratio exactly 1.0, n_viol exactly 0", "verdict": lm.lead_viol_verdict(hv)["verdict"],
                           "PASS": bool(hv["lead_progress_ratio"] == 1.0 and hv["n_viol"] == 0)}
    half = 0.5 * G
    hv2 = lm.lead_viol(lm.distance_keeping(half, lead["leads"], lead["lead_lens"], lead["speeds"], dtv), human_dk, eid,
                       lead_valid=lv, n_boot=200, seed=0, plan_arc=lm.path_arc_length(half), human_arc=human_arc)
    ci["half_speed_plan"] = {"lead_progress_ratio": hv2["lead_progress_ratio"], "lead_progress_ratio_clipped": hv2["lead_progress_ratio_clipped"],
                             "n_viol": hv2["n_viol"], "expected": "ratio exactly 0.5, INADMISSIBLE-CRAWL",
                             "verdict": lm.lead_viol_verdict(hv2)["verdict"],
                             "PASS": bool(hv2["lead_progress_ratio"] == 0.5 and lm.lead_viol_verdict(hv2)["verdict"] == "INADMISSIBLE-CRAWL")}
    odd = (eid % 2 == 1)
    mix = np.where(odd[:, None, None], 2.0 * G, 0.0 * G)
    hv3 = lm.lead_viol(lm.distance_keeping(mix, lead["leads"], lead["lead_lens"], lead["speeds"], dtv), human_dk, eid,
                       lead_valid=lv, n_boot=200, seed=0, plan_arc=lm.path_arc_length(mix), human_arc=human_arc)
    ci["crawl_in_half_overshoot_in_half"] = {
        "lead_progress_ratio": hv3["lead_progress_ratio"], "lead_progress_ratio_clipped": hv3["lead_progress_ratio_clipped"],
        "overshoot_masks_crawl": hv3["progress"].get("overshoot_masks_crawl"),
        "expected": "unclipped ~1 (passes), clipped ~0.5, overshoot_masks_crawl True, INADMISSIBLE-CRAWL",
        "verdict": lm.lead_viol_verdict(hv3)["verdict"],
        "PASS": bool(lm.lead_viol_verdict(hv3)["verdict"] == "INADMISSIBLE-CRAWL" and hv3["progress"].get("overshoot_masks_crawl"))}
    st = S["arms"]["stop"]
    ci["stop_plan"] = {"n_viol": st["n_viol"], "lead_progress_ratio": st["progress"]["lead_progress_ratio"],
                       "verdict": st["verdict"]["verdict"], "expected": "n_viol 0, ratio 0, INADMISSIBLE-CRAWL",
                       "PASS": bool(st["n_viol"] == 0 and st["progress"]["lead_progress_ratio"] == 0.0 and st["verdict"]["verdict"] == "INADMISSIBLE-CRAWL")}
    ci["content_asserts"] = {
        "stop_all_zero": bool(np.all(plans["stop"] == 0.0)),
        "g_finite": bool(np.isfinite(G).all()),
        **{f"{nm}_nonzero_finite": bool(np.isfinite(plans[nm]).all() and np.abs(plans[nm]).max() > 1.0)
           for nm in ("refcv8_48852.os", "refcv8_40k.os", "refcv7_50400.os")},
        "refcv8_48852_os_differs_from_refcv7_os": bool(not np.array_equal(plans["refcv8_48852.os"], plans["refcv7_50400.os"])),
        "refcv8_48852_os_differs_from_refcv8_40k_os": bool(not np.array_equal(plans["refcv8_48852.os"], plans["refcv8_40k.os"])),
    }
    assert all(ci["content_asserts"].values()), ci["content_asserts"]
    ci["all_known_value_controls_pass"] = bool(all(ci[k]["PASS"] for k in ("human_as_plan", "half_speed_plan",
                                                                           "crawl_in_half_overshoot_in_half", "stop_plan")))
    # ---- the instrument on refcv7 / model-free arms reproduces the banked r7base numbers ----------------------------
    rec = {}
    for nm, bnm in (("refcv7_50400.os", "os"), ("refcv7_50400.os@own_panel", "os"), ("ha0", "ha0"), ("ha0_ext", "ha0_ext"), ("stop", "stop")):
        a, b = S["arms"][nm], banked[bnm]
        e = {"n_lead": [a["n_lead_windows"], b["n_lead_windows"]], "n_viol": [a["n_viol"], b["n_viol"]],
             "n_at_risk": [a["n_at_risk"], b["n_at_risk"]],
             "share_of_at_risk": [a["share_of_at_risk"], b["share_of_at_risk"]]}
        ok = bool(a["n_lead_windows"] == b["n_lead_windows"] and a["n_viol"] == b["n_viol"] and a["n_at_risk"] == b["n_at_risk"]
                  and a["share_of_at_risk"] == b["share_of_at_risk"])
        if has_prog_banked:
            e["ratio_clipped"] = [a["progress"]["lead_progress_ratio_clipped"], b["progress"]["lead_progress_ratio_clipped"]]
            ok = ok and abs(e["ratio_clipped"][0] - e["ratio_clipped"][1]) < 1e-9
        e["equal"] = ok
        rec[f"{nm} vs banked r7base[{bnm}]"] = e
    S["checks"]["recompute_vs_banked_r7base"] = rec
    if has_prog_banked:
        S["checks"]["controls_equal_banked_r7base_controls"] = bool(
            ci["human_as_plan"]["lead_progress_ratio"] == BANKED_CTRL[s]["human_as_plan"]["lead_progress_ratio"]
            and ci["half_speed_plan"]["lead_progress_ratio"] == BANKED_CTRL[s]["half_speed_plan"]["lead_progress_ratio"])
    # ---- paired refcv8 - refcv7 (+ 48,852 - 40k, + model-free) ---------------------------------------------------------
    pairs = {"refcv8_48852_minus_refcv7_50400": ("refcv8_48852.os", "refcv7_50400.os"),
             "refcv8_40k_minus_refcv7_50400": ("refcv8_40k.os", "refcv7_50400.os"),
             "refcv8_48852_minus_refcv8_40k": ("refcv8_48852.os", "refcv8_40k.os"),
             "refcv8_48852_minus_echo_ha0_ext": ("refcv8_48852.os", "ha0_ext"),
             "refcv8_48852_minus_hold_v0_ha0": ("refcv8_48852.os", "ha0")}
    for pn, (a, b) in pairs.items():
        p = lm.paired_lead_viol(flags[a], flags[b], eid, names=(a, b), n_boot=N_BOOT, seed=0)
        pc = paired_compact(p)
        pc["verdict_of_first_arm_vs_second_as_baseline"] = verdict_compact(lm.lead_viol_verdict(flags[a], paired=p))
        S["paired"][pn] = pc
        d = lm.paired_distance_keeping(dks[a], dks[b], eid, names=(a, b), n_boot=N_BOOT, seed=0)
        S["paired_distance_keeping"][pn] = paired_dk_compact(d)
    S["wall_s"] = round(time.time() - t_seed, 1)
    PW[(surface, s)] = {"eid": eid.copy(), "at_risk": at_risk.copy(), "lv": lv.copy(),
                        **{nm: flags[nm]["_per_window"]["viol"].copy() for nm in ("refcv8_48852.os", "refcv8_40k.os", "refcv7_50400.os")}}
    return S


for surface in SURFACES:
    SS = {"seeds": {}}
    for s in SEEDS:
        SS["seeds"][str(s)] = run_seed(surface, s)
    # ---- inference-seed floor (would another INFERENCE draw say this?) ---------------------------------------------------
    fl = {}
    for nm in ("refcv8_48852.os", "refcv8_40k.os", "refcv7_50400.os"):
        a0 = SS["seeds"]["0"]["arms"][nm]["share_of_at_risk"]
        a1 = SS["seeds"]["1"]["arms"][nm]["share_of_at_risk"]
        f0, f1 = PW[(surface, 0)][nm], PW[(surface, 1)][nm]
        both = np.isfinite(f0) & np.isfinite(f1)
        fl[nm] = {"s0": a0, "s1": a1, "abs_diff": round(abs(a0 - a1), 6),
                  "n_lead_windows_whose_violation_flag_differs_between_the_two_draws": int((f0[both] != f1[both]).sum()),
                  "n_lead_windows": int(both.sum())}
    for pn in ("refcv8_48852_minus_refcv7_50400", "refcv8_40k_minus_refcv7_50400"):
        d0 = SS["seeds"]["0"]["paired"][pn]["at_risk_denominator"]["delta"]
        d1 = SS["seeds"]["1"]["paired"][pn]["at_risk_denominator"]["delta"]
        fl[pn + ".delta_share_of_at_risk"] = {"s0": d0, "s1": d1, "abs_diff": round(abs(d0 - d1), 6)}
    SS["inference_seed_floor"] = {
        "_what": ("would another INFERENCE run say this? refcv8 and refcv7 both sample (DDIM), so the same checkpoint rerun with "
                  "another inference seed moves the plan. Seeds 0 and 1 are the two inference draws on the SAME windows."),
        "rows": fl}
    # ---- EXPLORATORY, POST-HOC (NOT the registered bar): average the two inference draws per window --------------------------
    # The registered bar reads each draw alone; the draws are noisy (sampler). The per-window mean of the two draws' violation
    # flags reduces inference noise only. It does NOT address training-seed or episode variance, and it was decided after seeing
    # the per-draw result. Reported beside the bar, never in place of it.
    from taniteval.ci import paired_episode_cluster_bootstrap
    ex = {}
    d0, d1 = PW[(surface, 0)], PW[(surface, 1)]
    assert np.array_equal(d0["at_risk"], d1["at_risk"])
    sel = d0["at_risk"]
    ids = d0["eid"][sel]
    for nm in ("refcv8_48852.os", "refcv8_40k.os", "refcv7_50400.os"):
        m = 0.5 * (d0[nm][sel] + d1[nm][sel])
        ex[nm] = {"mean_share_of_at_risk_over_two_draws": round(float(m.mean()), 6), "n_at_risk": int(sel.sum())}
    for pn, (a, b) in {"refcv8_48852_minus_refcv7_50400": ("refcv8_48852.os", "refcv7_50400.os"),
                       "refcv8_40k_minus_refcv7_50400": ("refcv8_40k.os", "refcv7_50400.os"),
                       "refcv8_48852_minus_refcv8_40k": ("refcv8_48852.os", "refcv8_40k.os")}.items():
        ma = 0.5 * (d0[a][sel] + d1[a][sel])
        mb = 0.5 * (d0[b][sel] + d1[b][sel])
        r = paired_episode_cluster_bootstrap(ma, mb, list(ids), n_boot=N_BOOT, seed=0)
        ex[pn] = {k: r.get(k) for k in ("delta", "lo", "hi", "separated", "n_episodes", "estimator")}
        ex[pn]["n_at_risk"] = int(sel.sum())
    SS["exploratory_post_hoc_two_draw_mean"] = {
        "status": "EXPLORATORY, POST-HOC. NOT the registered bar (the bar reads each inference draw alone, both must separate).",
        "what": "per-window mean of the violation flag over inference draws s0 and s1, then the same paired episode-cluster bootstrap",
        "answers": "reduces INFERENCE-sampler noise only; still one training seed, still 'another draw of EPISODES' only",
        "rows": ex}
    res["surfaces"][surface] = SS
json.dump(res, open(out_json, "w", encoding="utf-8"), indent=1, default=str)
print("ZZX7_DONE", out_json, flush=True)
