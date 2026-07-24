"""TanitEval — PLAN-FAN **CLIPS**: category-selected short videos, both REF-C arms.

THE ASK (Sayed, 2026-07-21): *"generate also some videos with the nice trajectory
space from the diffusion planner, so I can assess visually the performance"*.

This is the clip driver for :mod:`taniteval.plan_fan` — it does NOT re-implement
the panel. Every pixel is drawn by ``plan_fan.draw_bev`` / ``draw_cam`` /
``draw_colorbar`` / ``draw_legend`` / ``compose``, and the decode is
``plan_fan.episode_planfan`` (which is byte-identical to ``refc_eval.collect``
and keeps the ``sel_idx == argmax(anchor_logits)`` assertion live). What this
module adds is *which windows to look at* and *both arms on the same ones*.

WINDOW SELECTION — chosen for INFORMATION, not for flattery. The windows are
picked from the precomputed full-fan dumps ``results/fan_refc-{base,xl}-30k.pt``
(881 windows x 40 canonical val episodes, bit-identical GT and CV between the
arms — asserted here), so the selection is reproducible from data already on
disk and is not a hand-pick off a video. Six categories, each labelled in the
filename and in the HUD:

  good_selection          argmax at/near the fan's best in BOTH arms — the
                          system working. Filtered to v0 > 4 m/s: a stopped
                          vehicle scores 0.02 m and proves nothing.
  bad_selection_good_fan  sel > 2x oracle-in-fan AND gap > 1 m in BOTH arms.
                          THE failure to see: ``frac_sel_2x_worse`` is 0.4109
                          (base) / 0.4540 (XL), i.e. on ~41-45 % of windows a
                          plan at least twice as good was already in the fan.
  multimodal_junction     |net heading change@2s| > 8 deg with >=5-6 live modes
                          — where a strategic goal could disambiguate.
  high_speed              top speed tertile (>= 13.3 m/s), REF-C's strongest
                          stratum.
  cruise_steady           mid tertile, |a_gt| < 0.3, near-straight — where our
                          arms are weak *relative to CV*.
  braking_longitudinal    a_gt < -1 m/s^2 at v0 > 5 m/s — the longitudinal
                          lever, isolated.

BOTH ARMS, SAME WINDOWS, SAME SCALE. base (128 anchors) and XL (256) were scored
on the identical 881 windows. Each clip is rendered for both, and the BEV range
``xmax`` is computed JOINTLY over both arms' selected plans and the shared GT,
so the two videos are directly comparable frame-for-frame — a per-arm range
would make the fan's apparent spread incomparable, which is the whole point.

CONSTANT VELOCITY is drawn (dotted amber). Every ADE the programme quotes is
relative to CV, so a frame where the white plan sits further from the GT than
the dotted line is a frame the model LOST to two lines of kinematics.

⚠️ PhysicalAI-AV imagery is internal-dev-only: the renders stay on the pod and
in the repo. Never push frames, crops or renders to HF or any external service.

Run (eval pod, GPU free — take /usr/local/bin/gpu_lock.sh):
    PYTHONPATH=/root/taniteval:/root/TanitAD/stack \
      python3 -m taniteval.plan_fan_clips select
    PYTHONPATH=/root/taniteval:/root/TanitAD/stack \
      python3 -m taniteval.plan_fan_clips dump   --arm refc-base-30k
    PYTHONPATH=/root/taniteval:/root/TanitAD/stack \
      python3 -m taniteval.plan_fan_clips dump   --arm refc-xl-30k
    PYTHONPATH=/root/taniteval:/root/TanitAD/stack \
      python3 -m taniteval.plan_fan_clips render          # CPU, both arms
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import loaders, plan_fan as PF                     # noqa: E402
from taniteval.corpus_overlay import (HORIZON, FlatProjector,     # noqa: E402
                                      pretty_man, pretty_route)
from taniteval.direct_overlay import OUT                          # noqa: E402
from taniteval.registry import CORPORA, MODELS                    # noqa: E402

RES = Path("/root/taniteval/results")
VAL_KEY = "physicalai"
ARMS = ("refc-base-30k", "refc-xl-30k")
ARM_LABEL = {"refc-base-30k": "REF-C-base · 128 anchors",
             "refc-xl-30k": "REF-C-XL · 256 anchors"}
WINDOWS_JSON = RES / "planfan_clips_windows.json"
STRIDE = 8                    # the dump protocol's window stride
WINDOW = 8                    # trained context window
PAD = 24                      # frames each side of the scored window (~5 s clip)

# How many windows per category. Weighted by how much each one TEACHES, not
# evenly: the two selection-failure categories and the multimodal ones are what
# a human eye can adjudicate and a scalar cannot, while "good selection" and
# "high speed" are near mode-collapsed and one example each says everything.
N_PER_CAT = {"good_selection": 1, "bad_selection_good_fan": 2,
             "multimodal_junction": 2, "high_speed": 1,
             "cruise_steady": 2, "braking_longitudinal": 1}


# ======================================================================== #
# 1. SELECT — categories off the precomputed fans (CPU, no model)          #
# ======================================================================== #
def _panel(d):
    fan, gt, sel = d["fan"], d["gt"], d["sel"]
    de_all = torch.linalg.norm(fan - gt[:, None], dim=-1).mean(-1)
    p = torch.softmax(d["logits"], dim=1)
    de_or, _ = de_all.min(1)
    return dict(sel=de_all[torch.arange(len(sel)), sel], oracle=de_or,
                top1=p.max(1).values, nmodes=(p > PF.MODE_THRESH).sum(1),
                H=-(p * p.clamp_min(1e-12).log()).sum(1), n=fan.shape[1])


def select(n_per_cat=None, out=WINDOWS_JSON):
    """Pick the render windows. Deterministic; writes the manifest with the
    reason and the per-arm numbers for every window, so the choice is auditable
    and nobody has to trust a screenshot."""
    n_per_cat = dict(N_PER_CAT) if n_per_cat is None else n_per_cat
    D = {k: torch.load(RES / f"fan_{k}.pt", map_location="cpu",
                       weights_only=False) for k in ARMS}
    A, B = D[ARMS[0]], D[ARMS[1]]
    assert A["eid"] == B["eid"], "arms are not on the same windows"
    assert torch.equal(A["gt"], B["gt"]), "GT differs between arms"
    assert torch.equal(A["cv"], B["cv"]), "CV differs between arms"
    eid = A["eid"]
    PA, PB = _panel(A), _panel(B)
    cv = torch.linalg.norm(A["cv"] - A["gt"], dim=-1).mean(-1)

    # window -> frame: the dump walks starts = range(0, T-8-20, 8) per episode,
    # last = start + 7, so the j-th window of an episode ends at frame 8j+7.
    seen, frame = Counter(), []
    for e in eid:
        frame.append(STRIDE * seen[e] + WINDOW - 1)
        seen[e] += 1

    spd, hd, a_gt = A["speed"], A["head_deg"].abs(), A["a_gt"]
    q = torch.quantile(spd, torch.tensor([1 / 3, 2 / 3]))
    rA, rB = PA["sel"] / PA["oracle"].clamp_min(1e-6), \
        PB["sel"] / PB["oracle"].clamp_min(1e-6)
    gA, gB = PA["sel"] - PA["oracle"], PB["sel"] - PB["oracle"]

    cats = [
        ("good_selection",
         (rA < 1.15) & (rB < 1.15) & (PA["sel"] < 0.6) & (PB["sel"] < 0.6)
         & (spd > 4.0),
         PA["sel"] + PB["sel"],
         "argmax <=1.15x oracle in BOTH arms, sel<0.6 m, moving (v0>4 m/s)"),
        ("bad_selection_good_fan",
         (rA > 2.0) & (rB > 2.0) & (gA > 1.0) & (gB > 1.0), -(gA + gB),
         "sel >2x oracle-in-fan AND gap >1 m in BOTH arms (the 41 % failure)"),
        ("multimodal_junction",
         (hd > 8.0) & (PB["nmodes"] >= 6) & (PA["nmodes"] >= 5),
         -(PA["H"] + PB["H"]),
         "|net heading@2s| >8 deg with >=5/6 modes>1% — left/straight/right live"),
        ("high_speed", spd >= q[1], -spd, "top speed tertile (REF-C's best)"),
        ("cruise_steady",
         (spd > q[0]) & (spd < q[1]) & (a_gt.abs() < 0.3) & (hd < 2.0),
         -(PA["sel"] + PB["sel"]),
         "mid tertile, |a_gt|<0.3 m/s^2, near-straight — where we are weak vs CV"),
        ("braking_longitudinal", (a_gt < -1.0) & (spd > 5.0), a_gt,
         "a_gt < -1 m/s^2 at v0 > 5 m/s — the longitudinal lever"),
    ]

    picked, used_ep = [], Counter()
    for cat, mask, key, why in cats:
        idx = torch.nonzero(mask).flatten()
        idx = idx[torch.argsort(key[idx])]
        taken, cat_eps = 0, set()
        for i in idx.tolist():
            if taken >= n_per_cat.get(cat, 1):
                break
            # Spread across episodes: never twice from the same episode WITHIN a
            # category (two windows of the same clip mostly show the same thing
            # — the first pass put both high-speed picks in ep31), at most 2
            # clips from any episode overall, and never overlapping frames.
            if eid[i] in cat_eps or used_ep[eid[i]] >= 2:
                continue
            if any(p["ep"] == eid[i] and abs(p["frame"] - frame[i]) < 2 * PAD
                   for p in picked):
                continue
            cat_eps.add(eid[i])
            picked.append(dict(
                w=i, ep=int(eid[i]), frame=int(frame[i]), cat=cat, why=why,
                v0=round(float(spd[i]), 2),
                head_deg=round(float(A["head_deg"][i]), 2),
                a_gt=round(float(a_gt[i]), 3), cv_ade=round(float(cv[i]), 3),
                arms={k: dict(sel=round(float(P["sel"][i]), 3),
                              oracle=round(float(P["oracle"][i]), 3),
                              ratio=round(float(r[i]), 2),
                              top1=round(float(P["top1"][i]), 3),
                              H=round(float(P["H"][i]), 2),
                              modes=int(P["nmodes"][i]))
                      for k, P, r in ((ARMS[0], PA, rA), (ARMS[1], PB, rB))}))
            used_ep[eid[i]] += 1
            taken += 1

    meta = dict(
        source={k: str(RES / f"fan_{k}.pt") for k in ARMS},
        n_windows=len(eid), n_episodes=len(set(eid)),
        corpus=VAL_KEY, stride=STRIDE, window=WINDOW, pad=PAD,
        speed_tertiles=[round(float(q[0]), 3), round(float(q[1]), 3)],
        arm_summary={k: dict(
            n_anchors=int(P["n"]), mean_sel=round(float(P["sel"].mean()), 4),
            mean_oracle=round(float(P["oracle"].mean()), 4),
            mean_gap=round(float((P["sel"] - P["oracle"]).mean()), 4),
            frac_sel_2x_worse=round(float((P["sel"] > 2 * P["oracle"])
                                          .float().mean()), 4),
            mean_modes=round(float(P["nmodes"].float().mean()), 3))
            for k, P in ((ARMS[0], PA), (ARMS[1], PB))},
        cv_mean_ade=round(float(cv.mean()), 4), clips=picked)
    Path(out).write_text(json.dumps(meta, indent=1))
    print(f"[select] {len(picked)} windows -> {out}")
    for p in picked:
        a, b = p["arms"][ARMS[0]], p["arms"][ARMS[1]]
        print(f"  {p['cat']:<24} ep{p['ep']:02d} f{p['frame']:03d} "
              f"v0 {p['v0']:5.1f} hd {p['head_deg']:+6.1f} a {p['a_gt']:+5.2f} "
              f"| base {a['sel']:5.2f}/{a['oracle']:4.2f} ({a['ratio']:4.1f}x) "
              f"| xl {b['sel']:5.2f}/{b['oracle']:4.2f} ({b['ratio']:4.1f}x) "
              f"| cv {p['cv_ade']:5.2f}")
    return meta


# ======================================================================== #
# 2. DUMP — one GPU pass per arm over the selected clip frame ranges        #
# ======================================================================== #
def _entry(key):
    e = [m for m in MODELS if m["key"] == key]
    assert e, f"{key} not in the registry"
    assert e[0]["arch"] == "refc", f"{key} is arch={e[0]['arch']}, not refc"
    return e[0]


def dump(arm, device="cuda", batch=8, windows=WINDOWS_JSON):
    """Stride-1 decode over each clip's frame range, full proposal set kept."""
    meta = json.loads(Path(windows).read_text())
    e = _entry(arm)
    L = loaders.load(e, device)
    model, step = L["model"], L["step"]
    assert L["step_readout"] is None, "REF-C must have no grounded step_readout"
    assert not model.cfg.refc1, "refc1 ckpt: horizons are distances, not times"
    assert tuple(model.cfg.trajectory.horizons) == PF.WP_STEPS
    # PLANNER_VIZ_CONCEPT.md line 33: with grounded_selector=False the fan
    # colours ARE the selection score. If a future ckpt turns it on they are
    # not, and the panel would be a lie. Refuse rather than draw.
    assert not model.cfg.grounded_selector, (
        "grounded_selector=True: the selection score is conf + a progress "
        "proxy, so the fan colours (softmax of conf) are NOT the selection "
        "score. Refusing to render.")
    steps = model.cfg.decoder.diffusion_steps
    window = int(model.cfg.window)
    assert window == WINDOW, f"trained window {window} != protocol {WINDOW}"
    anchors = model.decoder.anchors.detach().float().cpu()
    n_anchors = int(anchors.shape[0])
    print(f"[load] {arm} step={step} anchors={n_anchors} denoise={steps} "
          f"window={window} graft_maneuver={model.cfg.graft_maneuver} "
          f"grounded_selector={model.cfg.grounded_selector}", flush=True)

    corp = [c for c in CORPORA if c["key"] == meta["corpus"]][0]
    files = sorted(Path(corp["root"]).glob("ep_*.pt"))
    from tanitad.data.mixing import load_episode

    store = {}
    for c in meta["clips"]:
        ep = load_episode(str(files[c["ep"]]), mmap=True)
        recs = PF.episode_planfan(model, ep, device, window, steps, batch=batch,
                                  t_lo=c["frame"] - PAD, t_hi=c["frame"] + PAD,
                                  want_cv=True)
        assert c["frame"] in recs, (
            f"ep{c['ep']} f{c['frame']} not decodable (episode too short?)")
        for t, r in recs.items():                 # vocabulary-only coverage floor
            dv = torch.linalg.norm(anchors - r["gt_wp"][None], dim=-1).mean(-1)
            r["vocab_ade"] = float(dv.min())
            r["is_anchor"] = (t == c["frame"])
        # sanity: the clip's scored frame must reproduce the dumped fan row
        print(f"[clip] {c['cat']:<24} ep{c['ep']:02d} f{c['frame']:03d} "
              f"{len(recs):3d} frames  anchor sel {recs[c['frame']]['ade']:.3f} "
              f"(fan dump {c['arms'][arm]['sel']:.3f})  oracle "
              f"{recs[c['frame']]['oracle_ade']:.3f} "
              f"(dump {c['arms'][arm]['oracle']:.3f})", flush=True)
        store[f"{c['cat']}|{c['ep']}|{c['frame']}"] = recs
        del ep
        torch.cuda.empty_cache()

    out = RES / f"planfan_clips_{arm}.pt"
    torch.save(dict(arm=arm, step=step, n_anchors=n_anchors, steps=steps,
                    window=window, anchors=anchors, clips=store), out)
    print(f"[dump] {len(store)} clips -> {out}", flush=True)
    return out


# ======================================================================== #
# 3. RENDER — CPU; both arms, JOINT BEV range so the pair is comparable     #
# ======================================================================== #
def render(arms=ARMS, fps=10, windows=WINDOWS_JSON, stills=True, only=None):
    meta = json.loads(Path(windows).read_text())
    D = {a: torch.load(RES / f"planfan_clips_{a}.pt", map_location="cpu",
                       weights_only=False) for a in arms}
    corp = [c for c in CORPORA if c["key"] == meta["corpus"]][0]
    files = sorted(Path(corp["root"]).glob("ep_*.pt"))
    proj = FlatProjector(HORIZON[meta["corpus"]])
    from tanitad.data.mixing import load_episode
    OUT.mkdir(parents=True, exist_ok=True)
    vdir = OUT / "planfan-clips"
    vdir.mkdir(parents=True, exist_ok=True)

    summary = []
    for c in meta["clips"]:
        key = f"{c['cat']}|{c['ep']}|{c['frame']}"
        if only and c["cat"] != only:
            continue
        R = {a: D[a]["clips"][key] for a in arms}
        ts = sorted(set.intersection(*[set(R[a]) for a in arms]))
        # JOINT, arm-independent-where-possible BEV range: cover v0*2 s + margin
        # and never clip the GT or EITHER arm's selected plan. One range for the
        # pair, held fixed for the clip — otherwise the two videos cannot be
        # compared and the fan's spread jitters frame to frame.
        need = max([max(R[arms[0]][t]["v0"] for t in ts) * 2.0 + 8.0, 20.0]
                   + [float(R[arms[0]][t]["gt_wp"][:, 0].max()) + 5.0
                      for t in ts]
                   + [float(R[a][t]["fan"][R[a][t]["sel"]][:, 0].max()) + 5.0
                      for a in arms for t in ts])
        xmax = min(90.0, 5.0 * math.ceil(need / 5.0))
        ep = load_episode(str(files[c["ep"]]), mmap=True)

        for a in arms:
            recs, d = R[a], D[a]
            m_sel = sum(recs[t]["ade"] for t in ts) / len(ts)
            m_or = sum(recs[t]["oracle_ade"] for t in ts) / len(ts)
            m_vo = sum(recs[t]["vocab_ade"] for t in ts) / len(ts)
            m_cv = sum(recs[t]["cv_ade"] for t in ts) / len(ts)
            n2x = sum(recs[t]["ade"] > 2 * recs[t]["oracle_ade"] for t in ts)
            ncv = sum(recs[t]["ade"] > recs[t]["cv_ade"] for t in ts)
            ctx = dict(
                model_key=a, step=d["step"], corpus=meta["corpus"],
                title=f"{ARM_LABEL[a]} · plan fan · [{c['cat'].upper()}]",
                n_anchors=d["n_anchors"], steps=d["steps"], window=d["window"],
                h_max=math.log(d["n_anchors"]), proj=proj,
                anchors=d["anchors"].tolist(), ep=c["ep"], tag=c["cat"],
                xmax=xmax, mean_ade=m_sel, mean_oracle=m_or,
                legend_notes=[
                    f"score = softmax over ALL {d['n_anchors']} anchor logits",
                    "(H19 maneuver prior included) from the t=0 classifier",
                    f"pass; geometry is post-{d['steps']}-denoise-step — those",
                    "passes' own confidences are discarded by the decoder.",
                    "polylines: ego -> the 4 waypoints, straight segments",
                    "(drawing device); all metrics use the waypoints only.",
                    f"clip: {n2x}/{len(ts)} frames pick a plan >2x worse than",
                    f"one already in the fan; {ncv}/{len(ts)} lose to CV."])
            short = "base" if "base" in a else "xl"
            name = (f"planfan_{c['cat']}_ep{c['ep']:02d}_f{c['frame']:03d}"
                    f"_{short}")
            fdir = vdir / f"_frames_{name}"
            fdir.mkdir(parents=True, exist_ok=True)
            for n, t in enumerate(ts):
                r = recs[t]
                r["rgb"] = ep.frames[t, -3:].permute(1, 2, 0).numpy()
                im = PF.compose(r, ctx)
                im.save(fdir / f"f{n:04d}.png")
                if stills and r.get("is_anchor"):
                    im.save(vdir / f"{name}_ANCHOR.png")
                r["rgb"] = None
            mp4 = vdir / f"{name}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-r", str(fps), "-i", str(fdir / "f%04d.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
                 "-movflags", "+faststart", str(mp4)],
                check=True, capture_output=True)
            shutil.rmtree(fdir)
            row = dict(clip=name, cat=c["cat"], arm=a, ep=c["ep"],
                       anchor_frame=c["frame"], frames=len(ts), fps=fps,
                       xmax_m=xmax, n_anchors=d["n_anchors"],
                       mean_sel=round(m_sel, 4), mean_oracle=round(m_or, 4),
                       mean_vocab=round(m_vo, 4), mean_cv=round(m_cv, 4),
                       frames_sel_2x_worse=int(n2x), frames_lost_to_cv=int(ncv),
                       anchor=dict(
                           sel=round(recs[c["frame"]]["ade"], 4),
                           oracle=round(recs[c["frame"]]["oracle_ade"], 4),
                           vocab=round(recs[c["frame"]]["vocab_ade"], 4),
                           cv=round(recs[c["frame"]]["cv_ade"], 4),
                           top1=round(recs[c["frame"]]["top1"], 4),
                           H=round(recs[c["frame"]]["H"], 3),
                           modes=recs[c["frame"]]["n_modes"],
                           man=recs[c["frame"]]["man"],
                           route=recs[c["frame"]]["route"],
                           v0=round(recs[c["frame"]]["v0"], 3)),
                       mp4=str(mp4),
                       size_mb=round(mp4.stat().st_size / 1e6, 2))
            summary.append(row)
            print(f"[video] {mp4.name} {len(ts)}f {row['size_mb']} MB  "
                  f"sel {m_sel:.3f} oracle {m_or:.3f} cv {m_cv:.3f}  "
                  f"2x-worse {n2x}/{len(ts)}  lost-to-CV {ncv}/{len(ts)}",
                  flush=True)
        del ep

    p = RES / "planfan_clips_summary.json"
    p.write_text(json.dumps(dict(meta={k: v for k, v in meta.items()
                                       if k != "clips"},
                                 clips=summary), indent=1))
    print(f"\nPLAN_FAN_CLIPS_DONE  {len(summary)} videos -> {vdir}\n-> {p}")
    return summary


def main():
    ap = argparse.ArgumentParser("plan_fan_clips")
    ap.add_argument("cmd", choices=["select", "dump", "render"])
    ap.add_argument("--arm", default="refc-xl-30k")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--per-cat", type=int, default=None,
                    help="override N_PER_CAT uniformly (default: the weighted "
                         "per-category counts)")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    if a.cmd == "select":
        select(n_per_cat=None if a.per_cat is None else
               {k: a.per_cat for k in N_PER_CAT})
    elif a.cmd == "dump":
        dump(a.arm, device=a.device, batch=a.batch)
    else:
        render(fps=a.fps, only=a.only)


if __name__ == "__main__":
    main()
