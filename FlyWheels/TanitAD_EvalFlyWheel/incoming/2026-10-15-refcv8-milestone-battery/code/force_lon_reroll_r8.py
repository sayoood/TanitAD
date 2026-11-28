"""force_lon_reroll_r8.py -- the INTERVENTION that tests whether refcv8's S6 false brake is decided at the tactical longitudinal head.

CPU ONLY.  Three stages (one file, so the pre-registration, the roll and the analysis cannot drift apart):

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 R8_NO_POSE_SYNC=1 PYTHONIOENCODING=utf-8 \
    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
      python -X utf8 force_lon_reroll_r8.py select  [--fb-max 250] [--ct-max 250]      # windows + arms, from the banked tree
      python -X utf8 force_lon_reroll_r8.py roll    [--limit-windows N] [--tag smoke]  # DETACH this one (nohup); resumable
      python -X utf8 force_lon_reroll_r8.py analyze [--tag smoke]                      # --analyze-only: no model, no roll

WHAT IT TESTS (pre-registered by the Master Mind before any forced number existed; the table below is NOT editable here)
  Association already MEASURED (`raw/s6_lon_decomp/VERDICT_ANALYST.md`): refcv8 EMA-30k loses to refcv7-50,400 on S6 by ADE
  +0.1658 [+0.0268, +0.3153] m because it BRAKES WHERE THE HUMAN CRUISED -- GT-cruising windows carry 127 % of the delta, windows
  where the tactical lon head argmax is BRAKE_TO carry 109 %.  The head's posteriors feed the planner
  (`refc_v3.py:1999-2027` `hypothesis_posteriors` -> `r8_feed` / `lon_prior`).  Here the head is INTERVENED ON, eval-only, with
  `RefCV3Model.set_r8_force(lon6=...)` (`refc_v3.py:2081`), on the SAME windows, SAME weights (EMA-30k), SAME sampler seed per window.

  FB ("false brake") = GT-cruising (the decomposition's dv class `cruising`) AND head argmax BRAKE_TO.
  CT ("control")     = GT-cruising AND head argmax NOT BRAKE_TO; a seeded random sample, capped at min(|FB used|, 250).
  Arms (CPU, per-window sampler seed IDENTICAL across arms):
    N   native, no force.
    F1  force the LON6 class mapped from the GT tactical label `lon_v7` (oracle tactical lon).  LON6_V7_IDS = (0, 1, 3, 4, 5, 7)
        (`stack/tanitad/refs/refcv8_conditioning.py:63-64`): the class whose v7 id equals `lon_v7`.  A window whose `lon_v7` is
        unlabelled (-100) or is a v7 class that is NOT a LON6 hypothesis (2 YIELD_MERGE, 6 ADAPT_SPEED_FOR_CURVE: masked,
        `LON_MASKED_V7_IDS`) is EXCLUDED from F1 and COUNTED.
    F2  force CRUISE (LON6 index 1, v7 id 1) on every FB window.  Not run on CT.
  Extra arms (CONTROLS / diagnostics, added by the implementer, they do not enter the committed decision table):
    N2     native at a second sampler seed -> the sampler-noise floor of "same model, same window, different draw".
    Fown   force each window to its OWN head-argmax LON6 class (argmax of `hypothesis_posteriors`).        -> KB
    Fbrake force BRAKE_TO on CT windows (a positive control: the force must be able to CAUSE the defect).   -> KD
  Metrics on the S6 slots vs GT: ADE, |along-track| at 6 s, signed along-track at 6 s; plus refcv7-50,400's banked error on the
  same windows (the panel's embedded `b_refcv7_50400_s0` arm, read exactly as `s6_lon_decomp.py` reads it).
  PRIMARY: CLOSURE on FB = (ADE_N - ADE_F1) / (ADE_N - ADE_R7), episode-cluster bootstrap (ratio of resampled sums, B 2000).
  COMMITTED OUTCOMES (verbatim from the brief):
    HEAD         closure point >= 0.50 AND CI lower bound > 0: the false brake is decided at the tactical lon head.
    DECODER      closure CI upper bound < 0.25: forcing the right tactical class does not fix it; the operative decoder brakes on its own.
    INCONCLUSIVE anything else; report the point, CI, n and what n would be needed.

TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  ESTIMATOR: episode-cluster bootstrap over the
episodes present (models FIXED): answers "another draw of EPISODES" only -- NOT another training run (one training seed), NOT another
inference run (the sampler seed is fixed per window; N2 reports the draw-to-draw floor).
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PKG = HERE.parent
OUT = PKG / "raw" / "force_lon"
SCR = Path("E:/refcv8_eval_kit/force_lon_scratch")                  # raw-id material and big tensors stay OFF the package/C:
ARCH = "C:/Users/Admin/refcv8_eval_archive/battery/step30000ema_unc"
CKPT = "E:/refcv8_eval_kit/ckpt/ckpt_ema_30000.pt"
CONFIG = "E:/refcv8_eval_kit/ckpt/config.json"
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
QUESTION = ("episode-cluster bootstrap over the episodes present in the window set, models FIXED, B 2000, seed 0: answers 'another draw of "
            "EPISODES' only; NOT another training run (single training seed per arm), NOT another inference run (the sampler seed is fixed per "
            "window; arm N2 reports the draw-to-draw floor)")
N_BOOT = 2000
BOOT_SEED = 0
SEL_SEED = 20261018                                                  # the window-sampling seed (fixed here, before any forced number)
INFER_SEED = 0
N2_SEED_OFFSET = 7919
LON6 = ("FOLLOW", "CRUISE", "BRAKE_TO", "CREEP", "HOLD", "ACCELERATE")        # refcv8_conditioning.py:63
LON6_V7_IDS = (0, 1, 3, 4, 5, 7)                                              # refcv8_conditioning.py:64
CRUISE6, BRAKE6 = LON6.index("CRUISE"), LON6.index("BRAKE_TO")
BRAKE_V7 = 3                                                                  # TACTICAL_LON_ACTIONS_V7[3] == "BRAKE_TO" (asserted at run time)
#: BASE arms run on EVERY window of the set (the committed table needs only these); CONTROL arms run on a seeded subset of --kb-n windows per set.
BASE_FB, BASE_CT = ("F1", "F2"), ("F1",)
CTRL_FB, CTRL_CT = ("N2", "Fown"), ("N2", "Fown", "Fbrake")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, p)


def window_seed(infer_seed: int, e_i: int, t: int) -> int:
    """One sampler seed per WINDOW, identical for every arm of that window (N2 adds N2_SEED_OFFSET to infer_seed)."""
    h = hashlib.sha256(f"{infer_seed}|{e_i}|{t}".encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31 - 1)


def f1_class(lon_v7: int):
    """The LON6 index forced by F1, or (None, reason).  Mapping = LON6_V7_IDS (refcv8_conditioning.py:64)."""
    if lon_v7 < 0:
        return None, "unlabelled (-100)"
    if lon_v7 in LON6_V7_IDS:
        return LON6_V7_IDS.index(lon_v7), None
    return None, f"v7 class {lon_v7} is masked (not a LON6 hypothesis)"


def apportion(counts: dict, n_total: int, rng) -> dict:
    """Proportional allocation of n_total over episodes (largest remainder; never above an episode's own count)."""
    tot = sum(counts.values())
    if n_total >= tot:
        return dict(counts)
    q = {e: counts[e] * n_total / tot for e in counts}
    take = {e: int(np.floor(q[e])) for e in counts}
    left = n_total - sum(take.values())
    order = sorted(counts, key=lambda e: (-(q[e] - take[e]), rng.random()))
    for e in order[:left]:
        take[e] += 1
    return take


# ====================================================================================================== stage: select
def stage_select(a) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[force_lon] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    import r8_battery_patch as BP
    L, Pm = BP.apply()
    ra = Pm.RR.ra3().ra
    import pair_read_r8 as PR
    import s6_lon_decomp as SD
    from tanitad.models import vocab_v7 as V7
    assert V7.TACTICAL_LON_ACTIONS_V7[BRAKE_V7] == "BRAKE_TO" and V7.TACTICAL_LON_ACTIONS_V7[1] == "CRUISE"
    T = SD.load_tree(ARCH, 0, L.sha12, PR, ra)
    r7x = SD.load_r7_extras(BP.R7_BASE_ROOT, T)
    T["r7_v_max_ms"] = r7x["v_max_ms"]
    parts, _dv = SD.partitions(T)
    dvc = np.asarray(parts["dvclass"][0])
    cruising = dvc == "cruising"
    pred = T["tac_lon_pred"]
    fb_all = cruising & (pred == BRAKE_V7)
    pool = cruising & (pred != BRAKE_V7)
    rng = np.random.default_rng(SEL_SEED)
    ep = np.asarray(T["ep_index"])

    def by_ep(mask):
        d = {}
        for i in np.nonzero(mask)[0]:
            d.setdefault(int(ep[i]), []).append(int(i))
        return d

    def pick(mask, n):
        d = by_ep(mask)
        take = apportion({e: len(v) for e, v in d.items()}, n, rng)
        sel = []
        for e in sorted(d):
            idx = list(d[e])
            rng.shuffle(idx)
            sel += idx[:take[e]]
        return sorted(sel)

    fb_sel = pick(fb_all, a.fb_max) if a.fb_max and fb_all.sum() > a.fb_max else sorted(np.nonzero(fb_all)[0].tolist())
    n_ct = min(len(fb_sel), a.ct_max)
    ct_sel = pick(pool, n_ct)
    ids = T["clip"]
    entries = []
    for setname, sel in (("FB", fb_sel), ("CT", ct_sel)):
        for i in sel:
            c, why = f1_class(int(T["lon_v7"][i]))
            entries.append({"set": setname, "row": int(i), "clip": ids[i], "ws": int(T["ws"][i]), "ep_index": int(ep[i]),
                            "lon_v7": int(T["lon_v7"][i]), "f1_class": c, "f1_excluded_reason": why,
                            "head_argmax_v7": int(pred[i]), "v0": float(T["v0"][i]),
                            "arms": list(BASE_FB if setname == "FB" else BASE_CT), "control_subset": False})
    for setname, ctrl in (("FB", CTRL_FB), ("CT", CTRL_CT)):         # control arms on a seeded subset; chunk ids = a seeded balanced split
        es = [e for e in entries if e["set"] == setname]
        keep = set(rng.permutation(len(es))[:a.kb_n if a.kb_n else len(es)].tolist())
        order = rng.permutation(len(es))
        for k, e in enumerate(es):
            e["chunk"] = int(order[k] % a.n_chunks)
            if k in keep:
                e["arms"] = e["arms"] + list(ctrl)
                e["control_subset"] = True
    for e in entries:
        if e["f1_class"] is None:
            e["arms"] = [x for x in e["arms"] if x != "F1"]
    windows = {}
    for e in entries:
        windows.setdefault(e["clip"], []).append(e["ws"])
    windows = {c: sorted(v) for c, v in windows.items()}
    SCR.mkdir(parents=True, exist_ok=True)
    jdump({"entries": entries, "windows": windows, "sel_seed": SEL_SEED}, SCR / "plan_raw.json")
    sha = L.sha12

    def cnt(mask):
        return int(mask.sum())

    stats = {
        "tool": "force_lon_reroll_r8.py select", "generated": time.strftime("%FT%T"), "tier": TIER, "sel_seed": SEL_SEED,
        "banked_tree": ARCH, "n_s6_windows": int(len(pred)), "n_s6_episodes": int(len(set(ep.tolist()))),
        "cruising_windows": cnt(cruising), "FB_all": cnt(fb_all), "FB_all_episodes": int(len(set(ep[fb_all].tolist()))),
        "CT_pool": cnt(pool), "CT_pool_episodes": int(len(set(ep[pool].tolist()))),
        "FB_used": len(fb_sel), "FB_used_episodes": int(len(set(ep[fb_sel].tolist()))),
        "CT_used": len(ct_sel), "CT_used_episodes": int(len(set(ep[ct_sel].tolist()))),
        "fb_max": a.fb_max, "ct_max": a.ct_max, "kb_n": a.kb_n, "n_chunks": a.n_chunks,
        "control_subset_sizes": {s_: sum(1 for e in entries if e["set"] == s_ and e["control_subset"]) for s_ in ("FB", "CT")},
        "F1_class_vs_head_class_FB": {"F1_class_is_BRAKE_TO": sum(1 for e in entries if e["set"] == "FB" and e["f1_class"] == BRAKE6), "F1_class_is_CRUISE": sum(1 for e in entries if e["set"] == "FB" and e["f1_class"] == CRUISE6)},
        "F1_excluded": {s: {"n_set": sum(1 for e in entries if e["set"] == s),
                            "n_excluded": sum(1 for e in entries if e["set"] == s and e["f1_class"] is None),
                            "by_reason": {r: sum(1 for e in entries if e["set"] == s and e["f1_excluded_reason"] == r)
                                          for r in sorted({e["f1_excluded_reason"] for e in entries if e["f1_excluded_reason"]})}}
                        for s in ("FB", "CT")},
        "F1_class_counts": {s: {LON6[k]: sum(1 for e in entries if e["set"] == s and e["f1_class"] == k) for k in range(6)} for s in ("FB", "CT")},
        "arms_per_set": {s: sorted({x for e in entries if e["set"] == s for x in e["arms"]}) for s in ("FB", "CT")},
        "n_forward_calls_per_window_mean": {s: 1 + float(np.mean([len(e["arms"]) for e in entries if e["set"] == s])) for s in ("FB", "CT")},
        "head_argmax_hist_in_CT_pool": {V7.TACTICAL_LON_ACTIONS_V7[k]: int(((pred == k) & pool).sum()) for k in range(8)},
        "dvclass_counts_all_S6": {k: int((dvc == k).sum()) for k in sorted(set(dvc.tolist()))},
        "note": "raw clip ids are in E:/refcv8_eval_kit/force_lon_scratch/plan_raw.json only; this file is sha12-keyed",
    }
    jdump(stats, OUT / "select_stats.json")
    jdump([{**{k: v for k, v in e.items() if k not in ("clip", "row")}, "clip_sha12": sha(e["clip"])} for e in entries], OUT / "plan_sha12.json")
    print(json.dumps({k: stats[k] for k in ("FB_all", "FB_used", "CT_pool", "CT_used", "F1_excluded", "arms_per_set",
                                              "n_forward_calls_per_window_mean", "control_subset_sizes")}, indent=1))
    print("ZZFLSELECTOKZZ")
    return 0


# ====================================================================================================== stage: roll
def _res_path(tag):
    return SCR / f"roll_{tag}.json"


def stage_roll_driver(a) -> int:
    """Run chunks 0..N-1 as sequential child processes.  The success test is the ARTIFACT (every planned window of the chunk is in the result
    file), never an exit code.  A chunk that falls short is retried (twice); a model reload per chunk is the price of robustness."""
    import subprocess
    plan = jload(SCR / "plan_raw.json")["entries"]
    if a.limit_windows:
        raise SystemExit("[force_lon] --driver with --limit-windows is not supported (use --chunk for the smoke)")
    print(f"ZZFLDRIVERSTARTZZ chunks={a.driver} entries={len(plan)} t={time.strftime('%FT%T')}", flush=True)
    for c in range(a.driver):
        want = {f"{e['clip']}|{e['ws']}" for e in plan if e.get("chunk") == c}
        for attempt in range(3):
            res_p = _res_path(a.tag)
            have = set(jload(res_p)["windows"]) if res_p.exists() else set()
            miss = want - have
            if not miss:
                break
            print(f"ZZFLCHUNKZZ chunk={c} attempt={attempt} missing={len(miss)}/{len(want)} t={time.strftime('%FT%T')}", flush=True)
            subprocess.run([sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()), "roll", "--tag", a.tag, "--chunk", str(c),
                            "--save-every", str(a.save_every)], check=False)
        have = set(jload(_res_path(a.tag))["windows"]) if _res_path(a.tag).exists() else set()
        if want - have:
            print(f"ZZFLCHUNKFAILZZ chunk={c} still_missing={len(want - have)}", flush=True)
    have = set(jload(_res_path(a.tag))["windows"]) if _res_path(a.tag).exists() else set()
    allw = {f"{e['clip']}|{e['ws']}" for e in plan}
    print(f"ZZFLDRIVERENDZZ rolled={len(have & allw)}/{len(allw)} t={time.strftime('%FT%T')}", flush=True)
    return 0


def stage_roll(a) -> int:
    if a.driver > 0:
        return stage_roll_driver(a)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[force_lon] set CUDA_VISIBLE_DEVICES=-1 (CPU only; the shared card belongs to another job)")
    if os.environ.get("R8_NO_POSE_SYNC", "") not in ("1", "true", "yes"):
        raise SystemExit("[force_lon] R8_NO_POSE_SYNC=1 is required: the banked tree is the STANDALONE (refcv7-comparable) clock")
    import r8_battery_patch as BP
    BP.apply_child()                                   # r8_repoint.apply(): dev-box remap, launch tree, pose-sync policy, seam guard
    import roll_seed_r7 as RS                          # noqa: F401  (battery code; its loader import is now re-pointed)
    import run_battery_r7 as RB
    import r7_roll as RR
    import torch
    L = RR.L
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    _orig = L.build_model

    def _fp32(*aa, **kk):                              # exactly roll_seed_r7's --smoke-cpu-fp32-trunk
        m, c, ar, r = _orig(*aa, **kk)
        for mod in m.modules():
            lv = getattr(mod, "memory_levers", None)
            if isinstance(lv, dict) and "bf16" in lv:
                lv["bf16"] = False
                lv["channels_last"] = False
        r["smoke_note"] = "trunk bf16/NHWC OFF (CPU fp32 trunk, as roll_seed_r7 --smoke-cpu-fp32-trunk)"
        return m, c, ar, r
    L.build_model = _fp32
    from tanitad.refs import refcv8_conditioning as C8
    assert tuple(C8.LON6) == LON6 and tuple(C8.LON6_V7_IDS) == LON6_V7_IDS, "LON6 mapping drifted from the pre-registered one"

    plan = jload(SCR / "plan_raw.json")
    entries = plan["entries"]
    if a.limit_windows:
        # smoke: the first N entries, alternating FB / CT so both sets and the control arms are exercised
        fb = [e for e in entries if e["set"] == "FB"]
        ct = [e for e in entries if e["set"] == "CT"]
        entries = [x for pair in zip(fb, ct) for x in pair][:a.limit_windows]
    if a.chunk >= 0:
        entries = [e for e in entries if e.get("chunk") == a.chunk]
    res_p = _res_path(a.tag)
    res = jload(res_p) if res_p.exists() else {"windows": {}}
    done_keys = set(res["windows"])
    W = 8                                              # cfg.core.window (asserted below once the model is built)
    clip_ids = RB._clip_ids_by_index(RB.KIT_EVAL)
    ci = {c: i for i, c in enumerate(clip_ids)}
    todo = [e for e in entries if f"{e['clip']}|{e['ws']}" not in done_keys]
    print(f"ZZFLSTARTZZ tag={a.tag} entries={len(entries)} done_already={len(entries) - len(todo)} todo={len(todo)} "
          f"threads={torch.get_num_threads()} t={time.strftime('%FT%T')}", flush=True)
    if not todo:
        print("ZZFLDONEZZ nothing to do", flush=True)
        return 0
    windows = {}
    for e in todo:
        windows.setdefault(e["clip"], []).append(e["ws"])
    windows = {c: sorted(v) for c, v in windows.items()}
    st = RR.install_patches(config_path=CONFIG, windows=windows, frame_memo=True, obedience=False, vmaxzero=False,
                            with_navflip=True, filteroff=False)
    dump_dir = SCR / f"dump_{a.tag}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    plan_by_key = {}
    for e in todo:
        plan_by_key[(ci[e["clip"]], int(e["ws"]) - (W - 1))] = e
    clip_of = {k: clip_ids[k[0]] for k in plan_by_key}
    prog = {"n_done": 0, "n_todo": len(todo), "t0": time.time()}
    _Base = RR.Refcv6Proxy

    class ForceProxy(_Base):
        """The battery's proxy, plus the forced re-rolls.  ARM N goes through the battery's own `__call__` (so the row-0 record is the
        same object the banked `os` is); the other arms re-call the model on the SAME inputs with the SAME per-window sampler seed."""

        def __call__(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
            d = self.__dict__
            if nav_cmd is None:
                raise RuntimeError("[force_lon] unexpected nav-null call (the roll is configured --no-navzero)")
            key = tuple(self.ctx.index)
            ent = plan_by_key.get(key)
            if ent is None:
                raise RuntimeError(f"[force_lon] window {key} has no plan entry")
            model, n, dev = d["model"], int(frames.shape[0]), frames.device
            t_w = time.time()
            rec = {"set": ent["set"], "arms": {}}
            torch.manual_seed(window_seed(INFER_SEED, key[0], key[1]))
            out = _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
            ex = d["extras"][key]["nav_true"]
            lon_logits = np.asarray(ex["tacv6_lon_logits"], dtype=np.float64)
            own6 = int(np.argmax(lon_logits[list(LON6_V7_IDS)]))
            rec["head_logits_v7"] = lon_logits.tolist()
            rec["head_argmax_v7"] = int(np.argmax(lon_logits))
            rec["own6"] = own6

            def grab(o, forced):
                r = {"traj": o["traj"][0].detach().float().cpu().numpy().tolist(), "sel_idx": int(o["sel_idx"][0])}
                if forced is not None:
                    r["forced6"] = int(forced)
                if "r8_hyp_alloc" in o and torch.is_tensor(o["r8_hyp_alloc"]):
                    lon = (o["r8_hyp_alloc"][0] % len(LON6)).detach().cpu().numpy()
                    r["alloc_lon6_hist"] = np.bincount(lon, minlength=len(LON6)).tolist()
                return r
            rec["arms"]["N"] = grab(out, None)
            rec["arms"]["N"]["wall_s"] = round(time.time() - t_w, 2)
            specs = {"F1": (0, ent["f1_class"]), "F2": (0, CRUISE6), "N2": (N2_SEED_OFFSET, None), "Fown": (0, own6),
                     "Fbrake": (0, BRAKE6)}
            for arm in ent["arms"]:
                off, cls = specs[arm]
                kw2 = dict(kw)
                kw2.update(self._inputs(n, dev))           # ego_poses is one-shot: re-supplied for every call
                torch.manual_seed(window_seed(INFER_SEED + off, key[0], key[1]))
                t_a = time.time()
                if cls is not None:
                    model.set_r8_force(lon6=int(cls))
                with torch.no_grad():
                    o = model(frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **kw2)
                if getattr(model, "_r8_force", None) is not None:
                    raise RuntimeError(f"[force_lon] arm {arm}: the force was NOT consumed by the forward (window {key})")
                rec["arms"][arm] = grab(o, cls)
                rec["arms"][arm]["wall_s"] = round(time.time() - t_a, 2)
            rec["wall_s"] = round(time.time() - t_w, 2)
            res["windows"][f"{clip_of[key]}|{key[1] + W - 1}"] = rec
            prog["n_done"] += 1
            if prog["n_done"] % int(a.save_every) == 0 or prog["n_done"] == prog["n_todo"]:
                jdump(res, res_p)
                el = time.time() - prog["t0"]
                eta = el / prog["n_done"] * (prog["n_todo"] - prog["n_done"])
                print(f"ZZFLPROGZZ {prog['n_done']}/{prog['n_todo']} elapsed={el:.0f}s eta={eta:.0f}s last_window_wall={rec['wall_s']}s", flush=True)
            return out

    RR.Refcv6Proxy = ForceProxy                        # install_patches' load_model resolves the name at call time
    argv = ["--ckpt", CKPT, "--config", CONFIG, "--episodes", RB.KIT_EVAL, "--labels", RB.KIT_LABELS,
            "--nav-source", "v72", "--grid", "2s", "--action-units", "steer", "--window-stride", "1", "--with-oracle-sel",
            "--dump-dir", str(dump_dir), "--dump-only", "--out", str(dump_dir / "_unused.json"),
            "--device", "cpu", "--infer-seed", str(INFER_SEED), "--seed", "0", "--arm", "refcv7", "--lru", "8",
            "--with-navflip", "--no-navzero"]
    R = RR.ra3()
    t0 = time.time()
    R.main(argv)
    jdump(res, res_p)
    mrec = st.get("model_record") or {}
    jdump({"tool": "force_lon_reroll_r8.py roll", "tag": a.tag, "ckpt": CKPT, "device": "cpu", "trunk": "fp32 (bf16/NHWC off)",
           "n_windows": len(res["windows"]), "wall_s": round(time.time() - t0, 1), "infer_seed": INFER_SEED, "n2_offset": N2_SEED_OFFSET,
           "state_dict": (mrec.get("state_dict") or {}).get("strict") if isinstance(mrec.get("state_dict"), dict) else None,
           "pose_clock": "standalone (R8_NO_POSE_SYNC=1)", "finished": time.strftime("%FT%T")}, SCR / f"roll_{a.tag}.meta.json")
    print(f"ZZFLDONEZZ n={len(res['windows'])} wall={time.time() - t0:.0f}s", flush=True)
    return 0


# ====================================================================================================== stage: analyze
METRICS = (("ade_m", "ADE 0-6 s", "m"), ("LON_along_abs_final_m", "|along-track| at 6 s", "m"),
           ("along_signed_final_m", "signed along-track at 6 s (P_x - G_x; negative = SHORT)", "m"))
HEAD_POINT, HEAD_LO_GT, DECODER_HI_LT = 0.50, 0.0, 0.25            # the committed outcome table (do not edit)


def decide(point, lo, hi) -> str:
    """The COMMITTED outcome table, verbatim from the brief."""
    if point is None or lo is None or hi is None:
        return "INCONCLUSIVE (closure undefined: the denominator is not positive)"
    if point >= HEAD_POINT and lo > HEAD_LO_GT:
        return "HEAD"
    if hi < DECODER_HI_LT:
        return "DECODER"
    return "INCONCLUSIVE"


def n_needed(point, lo, hi, n_win) -> dict:
    """ESTIMATED: windows needed if the point stayed and the CI half-width scaled as 1/sqrt(n) (an assumption, not a measurement)."""
    out = {"assumption": "CI half-width ~ 1/sqrt(n_windows); the point estimate stays; ESTIMATED, not measured"}
    if None in (point, lo, hi):
        return out
    if point >= HEAD_POINT and lo <= HEAD_LO_GT:
        out["to_reach_HEAD"] = float(n_win * ((point - lo) / point) ** 2)
    elif point < HEAD_POINT:
        out["to_reach_HEAD"] = "unreachable at this point estimate (point < 0.50)"
    if point < DECODER_HI_LT and hi >= DECODER_HI_LT:
        out["to_reach_DECODER"] = float(n_win * ((hi - point) / (DECODER_HI_LT - point)) ** 2)
    elif point >= DECODER_HI_LT:
        out["to_reach_DECODER"] = "unreachable at this point estimate (point >= 0.25)"
    return out


def ratio_boot(num, den, eid, SD):
    """Ratio of resampled sums (= ratio of means: the common n cancels) on episode-cluster draws; also returns the draw count with den<=0."""
    C, ep_idx, n_ep = SD.boot_counts(eid)
    sn = np.bincount(ep_idx, weights=num, minlength=n_ep)
    sd = np.bincount(ep_idx, weights=den, minlength=n_ep)
    nb, db = C @ sn, C @ sd
    ok = db > 0
    r = nb[ok] / db[ok]
    pt = float(num.sum() / den.sum()) if den.sum() > 0 else None
    lo, hi = SD.pct(r) if r.size else (None, None)
    return {"point": pt, "lo": lo, "hi": hi, "n_windows": int(len(num)), "n_episodes": int(n_ep),
            "n_draws_den_le_0": int((~ok).sum()), "frac_draws_den_le_0": float((~ok).mean()),
            "num_sum_over_n": float(num.mean()), "den_sum_over_n": float(den.mean())}


def mean_ci(x, eid, SD):
    C, ep_idx, n_ep = SD.boot_counts(eid)
    sx = np.bincount(ep_idx, weights=x, minlength=n_ep)
    sn = np.bincount(ep_idx, minlength=n_ep).astype(np.float64)
    mb = (C @ sx) / (C @ sn)
    lo, hi = SD.pct(mb)
    return {"mean": float(np.mean(x)), "lo": lo, "hi": hi, "n_windows": int(len(x)), "n_episodes": int(n_ep)}


def stage_analyze(a) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[force_lon] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    import r8_battery_patch as BP
    L, Pm = BP.apply()
    ra = Pm.RR.ra3().ra
    import pair_read_r8 as PR
    import s6_lon_decomp as SD
    t0 = time.time()
    sha = L.sha12
    roll = jload(_res_path(a.tag))
    T = SD.load_tree(ARCH, 0, sha, PR, ra)
    S1 = SD.load_tree(ARCH, 1, sha, PR, ra)
    if S1["keys"] != T["keys"]:
        raise SystemExit("[force_lon] seed-0 and seed-1 S6 trees disagree on window identity")
    slots, dt = list(T["slots"]), T["dt"]
    key2row = {f"{c}|{w}": i for i, (c, w) in enumerate(zip(T["clip"], T["ws"].tolist()))}
    ex = np.load(os.path.join(ARCH, "dump_s0", "refcv6_extras.npz"))
    keys = [k for k in roll["windows"] if k in key2row]
    if len(keys) != len(roll["windows"]):
        raise SystemExit("[force_lon] a rolled window is not an S6 window of the banked tree")
    keys.sort(key=lambda k: key2row[k])
    rows = np.asarray([key2row[k] for k in keys])
    n = len(rows)
    eid = np.asarray(T["eid"])[rows]
    G = T["g"][rows]
    setv = np.asarray([roll["windows"][k]["set"] for k in keys])
    isFB, isCT = setv == "FB", setv == "CT"

    def arm_P(arm):
        """[n, 6, 2] plan at the S6 slots (NaN rows where the arm was not run) + the presence mask."""
        P = np.full((n, len(slots), 2), np.nan, dtype=np.float32)
        have = np.zeros(n, bool)
        for j, k in enumerate(keys):
            r = roll["windows"][k]["arms"].get(arm)
            if r is not None:
                P[j] = np.asarray(r["traj"], dtype=np.float32)[slots]
                have[j] = True
        return P, have

    def metrics_of(P, have):
        out = {m: np.full(n, np.nan) for m, _l, _u in METRICS}
        if have.any():
            wm = SD.window_metrics(PR, ra, P[have], G[have], dt)
            for m, _l, _u in METRICS:
                out[m][have] = wm[m]
        return out

    arms = {}
    for nm in ("N", "F1", "F2", "N2", "Fown", "Fbrake"):
        arms["CPU_" + nm] = arm_P(nm)
    arms["GPU_N_banked"] = (T["os"][rows], np.ones(n, bool))
    arms["GPU_N_banked_s1"] = (S1["os"][rows], np.ones(n, bool))
    arms["R7_banked"] = (T["ref"][rows], np.ones(n, bool))
    M = {nm: metrics_of(P, h) for nm, (P, h) in arms.items()}
    HAVE = {nm: h for nm, (_P, h) in arms.items()}

    def cell(new, ref, sel):
        """new - ref, paired episode-cluster bootstrap (taniteval.ci) on the selected windows finite on both sides."""
        m = sel & np.isfinite(new) & np.isfinite(ref)
        if m.sum() < 2:
            return {"status": "NO DATA", "n_windows": int(m.sum())}
        r = PR.paired_cell(new[m], ref[m], list(eid[m]))
        r["separated"] = bool(r["lo"] > 0 or r["hi"] < 0)
        return r

    rep = {"tool": "force_lon_reroll_r8.py analyze", "generated": time.strftime("%FT%T"), "tag": a.tag, "tier": TIER, "estimator": QUESTION,
           "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "banked_tree": ARCH, "ckpt": CKPT, "device": "CPU (fp32 trunk), per-window sampler seed",
           "n_windows": n, "n_FB": int(isFB.sum()), "n_CT": int(isCT.sum()), "n_episodes_FB": int(len(set(eid[isFB].tolist()))),
           "n_episodes_CT": int(len(set(eid[isCT].tolist()))), "arms_present": {nm: int(h.sum()) for nm, h in HAVE.items()},
           "slots_S6": slots, "dt": dt, "controls": {}, "cells": {}}
    # ---------------------------------------------------------------- sample description (n, F1 exclusions)
    fb_f1 = isFB & HAVE["CPU_F1"]
    sst = jload(OUT / "select_stats.json") if (OUT / "select_stats.json").exists() else {}
    rep["sample"] = {"FB_total_in_banked_S6": sst.get("FB_all"), "FB_rolled": int(isFB.sum()), "FB_with_F1": int(fb_f1.sum()),
                     "FB_F1_excluded": int(isFB.sum() - fb_f1.sum()), "CT_rolled": int(isCT.sum()),
                     "CT_with_F1": int((isCT & HAVE["CPU_F1"]).sum()), "CT_F1_excluded": int(isCT.sum() - (isCT & HAVE["CPU_F1"]).sum()),
                     "select_stats": sst}
    # ---------------------------------------------------------------- per-set arm means (+ episode-cluster CI)
    means = {}
    for setname, sel in (("FB", isFB), ("CT", isCT)):
        means[setname] = {}
        for nm in M:
            means[setname][nm] = {}
            for mk, _l, _u in METRICS:
                m = sel & np.isfinite(M[nm][mk])
                means[setname][nm][mk] = mean_ci(M[nm][mk][m], eid[m], SD) if m.sum() >= 2 else {"n_windows": int(m.sum())}
    rep["means"] = means
    # ---------------------------------------------------------------- paired cells
    CELLS = {
        "FB": [("N_minus_R7", "CPU_N", "R7_banked"), ("F1_minus_N", "CPU_F1", "CPU_N"), ("F2_minus_N", "CPU_F2", "CPU_N"),
               ("F1_minus_R7", "CPU_F1", "R7_banked"), ("F2_minus_R7", "CPU_F2", "R7_banked"),
               ("N2_minus_N", "CPU_N2", "CPU_N"), ("Fown_minus_N", "CPU_Fown", "CPU_N"),
               ("GPU_N_banked_minus_R7", "GPU_N_banked", "R7_banked"), ("CPU_N_minus_GPU_N_banked", "CPU_N", "GPU_N_banked"),
               ("GPU_N_banked_s1_minus_s0", "GPU_N_banked_s1", "GPU_N_banked")],
        "CT": [("N_minus_R7", "CPU_N", "R7_banked"), ("F1_minus_N", "CPU_F1", "CPU_N"), ("N2_minus_N", "CPU_N2", "CPU_N"),
               ("Fown_minus_N", "CPU_Fown", "CPU_N"), ("Fbrake_minus_N", "CPU_Fbrake", "CPU_N"),
               ("CPU_N_minus_GPU_N_banked", "CPU_N", "GPU_N_banked"), ("GPU_N_banked_s1_minus_s0", "GPU_N_banked_s1", "GPU_N_banked")],
    }
    for setname, sel in (("FB", isFB), ("CT", isCT)):
        rep["cells"][setname] = {}
        for cname, nw, rf in CELLS[setname]:
            rep["cells"][setname][cname] = {mk: cell(M[nw][mk], M[rf][mk], sel) for mk, _l, _u in METRICS}

    # ---------------------------------------------------------------- PRIMARY: closure on FB (windows with F1)
    def closure(arm, sel, mk="ade_m", base="CPU_N"):
        m = sel & np.isfinite(M[arm][mk]) & np.isfinite(M[base][mk]) & np.isfinite(M["R7_banked"][mk])
        if m.sum() < 2:
            return {"status": "NO DATA", "n_windows": int(m.sum())}
        r = ratio_boot(M[base][mk][m] - M[arm][mk][m], M[base][mk][m] - M["R7_banked"][mk][m], eid[m], SD)
        r["metric"], r["base"] = mk, base
        return r
    prim = closure("CPU_F1", isFB)
    prim["verdict"] = decide(prim.get("point"), prim.get("lo"), prim.get("hi"))
    prim["n_needed"] = n_needed(prim.get("point"), prim.get("lo"), prim.get("hi"), prim.get("n_windows", 0))
    rep["PRIMARY_closure_F1_on_FB"] = prim
    plan_ent = {f"{e['clip']}|{e['ws']}": e for e in jload(SCR / "plan_raw.json")["entries"]}
    f1c = np.asarray([(plan_ent[k]["f1_class"] if plan_ent[k]["f1_class"] is not None else -1) for k in keys])
    own6 = np.asarray([roll["windows"][k]["own6"] for k in keys])
    rep["closure_split_by_whether_F1_changes_the_class"] = {
        "what": "F1 forces a one-hot posterior. Where the oracle class EQUALS the head's own class, F1 is a pure SHARPENING (soft -> one-hot) and no class "
                "information was added; where it DIFFERS, F1 injects the oracle class. Splitting the primary statistic separates the two.",
        "F1_class_differs_from_CPU_head_class": closure("CPU_F1", isFB & (f1c >= 0) & (f1c != own6)),
        "F1_class_equals_CPU_head_class (pure sharpening)": closure("CPU_F1", isFB & (f1c >= 0) & (f1c == own6)),
        "FB_F1_class_counts": {LON6[k]: int(((f1c == k) & isFB).sum()) for k in range(6)},
        "FB_CPU_head_class_counts": {LON6[k]: int(((own6 == k) & isFB).sum()) for k in range(6)}}
    rep["closure_secondary"] = {
        "F2_on_FB_all_windows": closure("CPU_F2", isFB),
        "F1_on_FB_|along|6s": closure("CPU_F1", isFB, "LON_along_abs_final_m"),
        "F2_on_FB_|along|6s": closure("CPU_F2", isFB, "LON_along_abs_final_m"),
        "F1_on_FB_numerator_vs_BANKED_GPU_N (mixes devices: sensitivity only)": closure("CPU_F1", isFB, "ade_m", "GPU_N_banked"),
        "F2_on_FB_numerator_vs_BANKED_GPU_N (mixes devices: sensitivity only)": closure("CPU_F2", isFB, "ade_m", "GPU_N_banked"),
        "N2_as_the_numerator_arm_on_FB (a NULL intervention, another sampler draw: the noise floor of this statistic)": closure("CPU_N2", isFB),
    }
    # ---------------------------------------------------------------- CONTROLS
    ctl = rep["controls"]
    full_gpu = ex["nav_true.traj"][np.asarray(T["extras_rows"])[rows]].astype(np.float64)
    full_cpu = np.stack([np.asarray(roll["windows"][k]["arms"]["N"]["traj"], dtype=np.float64) for k in keys])
    dADE = M["CPU_N"]["ade_m"] - M["GPU_N_banked"]["ade_m"]
    d_s1 = M["GPU_N_banked_s1"]["ade_m"] - M["GPU_N_banked"]["ade_m"]
    d_n2 = M["CPU_N2"]["ade_m"] - M["CPU_N"]["ade_m"]

    def spread(x, sel):
        x = x[sel & np.isfinite(x)]
        if x.size == 0:
            return {"n": 0}
        return {"n": int(x.size), "mean": float(x.mean()), "rms": float(np.sqrt((x ** 2).mean())),
                "sd": float(x.std(ddof=1)) if x.size > 1 else None,
                "se_of_mean_iid": float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else None}
    allw = np.ones(n, bool)
    sp = {"all": spread(dADE, allw), "FB": spread(dADE, isFB), "CT": spread(dADE, isCT)}
    wp = np.abs(arms["CPU_N"][0].astype(np.float64) - arms["GPU_N_banked"][0].astype(np.float64))
    ctl["KA"] = {
        "what": "CPU native (arm N, per-window sampler seed) vs the banked GPU `os` (a GLOBAL sampler stream consumed sequentially over 4,754 "
                "windows, bf16 trunk): BIT-IDENTITY IS IMPOSSIBLE (different device RNG AND different draw order), so this is a statistical "
                "agreement check against the brief's 0.05 m bar.",
        "max_abs_waypoint_diff_S6_slots_m": float(wp.max()), "median_abs_waypoint_diff_S6_slots_m": float(np.median(wp)),
        "max_abs_waypoint_diff_all_8_slots_m": float(np.abs(full_cpu - full_gpu).max()),
        "ADE_diff_CPU_minus_GPU": sp, "bar_from_brief_abs_mean_ADE_diff_m": 0.05,
        "bar_exceeded": {k: (bool(abs(v["mean"]) > 0.05) if v.get("n") else None) for k, v in sp.items()},
        "NOISE_CONTEXT_same_windows": {
            "GPU_seed1_minus_GPU_seed0_banked_os (two independent GPU draws of the SAME model)": {"all": spread(d_s1, allw), "FB": spread(d_s1, isFB), "CT": spread(d_s1, isCT)},
            "CPU_N2_minus_CPU_N (two independent CPU draws)": {"all": spread(d_n2, allw), "FB": spread(d_n2, isFB), "CT": spread(d_n2, isCT)}},
        "per_window_ADE_diff_CPU_minus_GPU_quantiles": {str(q): float(np.quantile(dADE, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "head_argmax_agreement_CPU_vs_banked_GPU_8way": float(np.mean([roll["windows"][k]["head_argmax_v7"] == int(T["tac_lon_pred"][key2row[k]]) for k in keys])),
        "head_logit_max_abs_diff_CPU_vs_GPU": float(np.max([np.abs(np.asarray(roll["windows"][k]["head_logits_v7"]) -
                                                                   ex["nav_true.tacv6_lon_logits"][T["extras_rows"][key2row[k]]]).max() for k in keys])),
        "FB_windows_whose_CPU_head_argmax_is_still_BRAKE_TO": (float(np.mean([roll["windows"][k]["head_argmax_v7"] == BRAKE_V7
                                                                               for k in np.asarray(keys)[isFB]])) if isFB.any() else None)}

    if (OUT / "KA_NOISE_FLOOR_BANKED.json").exists():
        kf = jload(OUT / "KA_NOISE_FLOOR_BANKED.json")["sets"]
        ctl["KA"]["BANKED_GPU_seed1_minus_seed0_floor_on_the_FULL_planned_window_set"] = {
            s_: {"n_windows": kf[s_]["n_windows"], "mean_diff": kf[s_]["mean_diff_s1_minus_s0"]["mean"], "ci": [kf[s_]["mean_diff_s1_minus_s0"]["lo"], kf[s_]["mean_diff_s1_minus_s0"]["hi"]],
                 "iid_se": kf[s_]["iid_se_of_mean_diff"], "per_window_rms": kf[s_]["per_window_diff_rms"],
                 "frac_mode_flips_gt_1m": kf[s_]["frac_windows_with_abs_diff_gt_1m (mode flips)"]} for s_ in kf}
        ctl["KA"]["reading"] = ("The brief's 0.05 m bar is applied AS WRITTEN above (bar_exceeded). The banked floor shows two independent GPU draws of the same "
                                "model already differ by this much on this window set, so an exceedance is read against the floor: a CPU-vs-GPU mean "
                                "difference inside the floor's CI is what a faithful re-roll looks like; outside it would be the gross-mismatch signal.")

    def disp(arm):
        P, have = arms[arm]
        d = np.full(n, np.nan)
        d[have] = np.linalg.norm(P[have].astype(np.float64) - arms["CPU_N"][0][have].astype(np.float64), axis=-1).mean(1)
        return d
    D = {nm: disp("CPU_" + nm) for nm in ("F1", "F2", "N2", "Fown", "Fbrake")}

    def dcell(x, y, sel):
        m = sel & np.isfinite(D[x]) & np.isfinite(D[y])
        if m.sum() < 2:
            return {"status": "NO DATA", "n_windows": int(m.sum())}
        r = PR.paired_cell(D[x][m], D[y][m], list(eid[m]))
        r["separated"] = bool(r["lo"] > 0 or r["hi"] < 0)
        return r

    def dmean(nm, sel):
        m = sel & np.isfinite(D[nm])
        return mean_ci(D[nm][m], eid[m], SD) if m.sum() >= 2 else None
    kb1 = dcell("Fown", "F2", isFB)
    kb2 = dcell("Fown", "Fbrake", isCT)
    ctl["KB"] = {
        "what": "displacement of the plan from native N (mean over the 6 S6 slots of the L2 distance, m). Forcing the head's OWN argmax class must move "
                "the plan LESS than forcing a DIFFERENT class (FB: F2 = CRUISE vs own = BRAKE_TO; CT: Fbrake = BRAKE_TO vs own), proving the force "
                "reaches the planner and that the F1 shift is not sampler noise (N2 = another draw).",
        "mean_displacement_m": {s: {nm: dmean(nm, sel) for nm in D} for s, sel in (("FB", isFB), ("CT", isCT))},
        "FB_own_minus_F2": kb1, "FB_own_minus_F1": dcell("Fown", "F1", isFB), "FB_F1_minus_N2": dcell("F1", "N2", isFB),
        "FB_F2_minus_N2": dcell("F2", "N2", isFB), "CT_own_minus_Fbrake": kb2, "CT_F1_minus_N2": dcell("F1", "N2", isCT),
        "PASS_FB (own-class displacement < different-class displacement, CI upper < 0)": (bool(kb1["hi"] < 0) if "hi" in kb1 else None),
        "PASS_CT (same, CT)": (bool(kb2["hi"] < 0) if "hi" in kb2 else None)}
    reach = {}
    for nm in ("F1", "F2", "Fown", "Fbrake"):
        sh, shN = [], []
        for k in keys:
            r = roll["windows"][k]["arms"].get(nm)
            rn = roll["windows"][k]["arms"]["N"]
            if r is None or "alloc_lon6_hist" not in r or "forced6" not in r or "alloc_lon6_hist" not in rn:
                continue
            h, hn = np.asarray(r["alloc_lon6_hist"], float), np.asarray(rn["alloc_lon6_hist"], float)
            sh.append(h[r["forced6"]] / max(h.sum(), 1))
            shN.append(hn[r["forced6"]] / max(hn.sum(), 1))
        reach[nm] = {"n": len(sh), "mean_share_of_allocated_candidates_in_forced_class": (float(np.mean(sh)) if sh else None),
                     "same_class_share_under_native_N": (float(np.mean(shN)) if shN else None)}
    ctl["force_reaches_candidate_allocation"] = reach
    ctl["fraction_windows_selected_candidate_index_changed_vs_N"] = {
        nm: float(np.mean([roll["windows"][k]["arms"][nm]["sel_idx"] != roll["windows"][k]["arms"]["N"]["sel_idx"]
                           for k in keys if nm in roll["windows"][k]["arms"]]))
        for nm in ("F1", "F2", "N2", "Fown", "Fbrake") if any(nm in roll["windows"][k]["arms"] for k in keys)}
    ct_f1 = rep["cells"]["CT"]["F1_minus_N"]["ade_m"]
    ctl["CT_no_degradation"] = {"F1_minus_N_ADE": ct_f1,
                                "degrades_beyond_CI (CI lower bound > 0)": (bool(ct_f1["lo"] > 0) if "lo" in ct_f1 else None),
                                "note": "F2 is not run on CT by design (the brief); Fbrake on CT is the positive control KD, not a degradation test"}
    ctl["KD_positive_control"] = {"what": "forcing BRAKE_TO on correctly-cruising CT windows: ADE must rise and the plan must end SHORT (signed along < 0)",
                                  "cells": rep["cells"]["CT"]["Fbrake_minus_N"]}
    sm = lambda s, c, mk: rep["cells"][s][c][mk]
    rep["headline"] = {
        "tier": TIER, "estimator": QUESTION,
        "PRIMARY_closure_F1_on_FB": {k: prim.get(k) for k in ("point", "lo", "hi", "n_windows", "n_episodes", "verdict", "n_needed")},
        "ADE_N_minus_R7_on_FB (the denominator)": sm("FB", "N_minus_R7", "ade_m"),
        "ADE_F1_minus_N_on_FB (numerator, sign flipped)": sm("FB", "F1_minus_N", "ade_m"),
        "ADE_F2_minus_N_on_FB": sm("FB", "F2_minus_N", "ade_m"),
        "ADE_F1_minus_N_on_CT": sm("CT", "F1_minus_N", "ade_m"),
        "KA_bar_exceeded": ctl["KA"]["bar_exceeded"], "KB": {k: v for k, v in ctl["KB"].items() if k.startswith("PASS")}}
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["per_window_sha12"] = [{"clip_sha12": sha(k.split("|")[0]), "ws": int(k.split("|")[1]), "set": str(setv[j]),
                                **{f"{nm}:{mk}": (None if not np.isfinite(M[nm][mk][j]) else round(float(M[nm][mk][j]), 5))
                                   for nm in M for mk in ("ade_m", "along_signed_final_m")}} for j, k in enumerate(keys)]
    OUT.mkdir(parents=True, exist_ok=True)
    jdump(rep, OUT / ("FORCE_LON.json" if a.tag == "full" else f"FORCE_LON_{a.tag}.json"))
    write_md(rep, OUT / ("FORCE_LON.md" if a.tag == "full" else f"FORCE_LON_{a.tag}.md"))
    print(json.dumps(rep["headline"], indent=1, default=str))
    print("ZZFLANALYZEDONEZZ")
    return 0


def _f(x, nd=4):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, (int, float)) else str(x))


def _ci(c, nd=4):
    if not isinstance(c, dict) or "delta" not in c:
        return "n/a"
    return f"{c['delta']:+.{nd}f} [{c['lo']:+.{nd}f}, {c['hi']:+.{nd}f}] {'SEP' if c.get('separated') else 'n.s.'} (n {c.get('n_windows')}/{c.get('n_episodes')} ep)"


def write_md(rep, path):
    p = rep["PRIMARY_closure_F1_on_FB"]
    ka, kb = rep["controls"]["KA"], rep["controls"]["KB"]
    nz = ka["NOISE_CONTEXT_same_windows"]
    g = nz["GPU_seed1_minus_GPU_seed0_banked_os (two independent GPU draws of the SAME model)"]["all"]
    c2 = nz["CPU_N2_minus_CPU_N (two independent CPU draws)"]["all"]
    out = []
    out.append(f"# FORCE_LON -- does forcing the tactical lon head fix refcv8's S6 false brake?  (tag {rep['tag']})")
    out.append("")
    out.append(f"Stamp for every number: **{rep['tier']}**. Model: refcv8-r101-s0 EMA step 30,000 on CPU (fp32 trunk), per-window sampler seed identical "
               f"across arms. Estimator: {rep['estimator']}.")
    out.append("")
    out.append("## Verdict (committed table: HEAD = closure >= 0.50 with CI lower > 0 | DECODER = CI upper < 0.25 | else INCONCLUSIVE)")
    out.append("")
    out.append(f"**{p.get('verdict')}** -- closure of F1 on FB = **{_f(p.get('point'), 3)}** [{_f(p.get('lo'), 3)}, {_f(p.get('hi'), 3)}], "
               f"n {p.get('n_windows')} windows / {p.get('n_episodes')} episodes (draws with a non-positive denominator: {p.get('n_draws_den_le_0')}). "
               f"Numerator ADE_N - ADE_F1 = {_f(p.get('num_sum_over_n'))} m; denominator ADE_N - ADE_R7 = {_f(p.get('den_sum_over_n'))} m.")
    out.append("")
    out.append(f"n needed (ESTIMATED): `{json.dumps(p.get('n_needed'))}`")
    out.append("")
    s = rep["sample"]
    out.append(f"Sample: FB rolled {s['FB_rolled']} (of {s['FB_total_in_banked_S6']} in banked S6), with F1 {s['FB_with_F1']} (F1-excluded, unlabelled lon_v7: "
               f"{s['FB_F1_excluded']}); CT rolled {s['CT_rolled']}, with F1 {s['CT_with_F1']} (excluded {s['CT_F1_excluded']}).")
    out.append("")
    out.append("## Paired cells (episode-cluster bootstrap, taniteval.ci; new - ref; lower ADE is better)")
    for st in ("FB", "CT"):
        out.append(f"\n### {st}")
        out.append("| cell | ADE 0-6 s (m) | \\|along\\| 6 s (m) | signed along 6 s (m) |")
        out.append("|---|---|---|---|")
        for c, v in rep["cells"][st].items():
            out.append(f"| {c} | {_ci(v['ade_m'])} | {_ci(v['LON_along_abs_final_m'])} | {_ci(v['along_signed_final_m'])} |")
    out.append("\n## Secondary closures")
    for k, v in rep["closure_secondary"].items():
        out.append(f"* {k}: point {_f(v.get('point'), 3)} [{_f(v.get('lo'), 3)}, {_f(v.get('hi'), 3)}] n {v.get('n_windows')}")
    sp_ = rep["closure_split_by_whether_F1_changes_the_class"]
    out.append("\n## Primary split: does F1 change the class, or only sharpen the posterior?")
    for k in ("F1_class_differs_from_CPU_head_class", "F1_class_equals_CPU_head_class (pure sharpening)"):
        v = sp_[k]
        out.append(f"* {k}: closure {_f(v.get('point'), 3)} [{_f(v.get('lo'), 3)}, {_f(v.get('hi'), 3)}] n {v.get('n_windows')} / {v.get('n_episodes')} ep")
    out.append(f"* FB oracle (F1) class counts `{json.dumps(sp_['FB_F1_class_counts'])}`; FB CPU head class counts `{json.dumps(sp_['FB_CPU_head_class_counts'])}`")
    out.append("\n## Controls")
    out.append(f"* **KA** (CPU N vs banked GPU `os`): max abs waypoint diff {ka['max_abs_waypoint_diff_S6_slots_m']:.3f} m (S6 slots) / "
               f"{ka['max_abs_waypoint_diff_all_8_slots_m']:.3f} m (all 8 slots); mean ADE diff all {_f(ka['ADE_diff_CPU_minus_GPU']['all'].get('mean'))} m "
               f"(iid SE {_f(ka['ADE_diff_CPU_minus_GPU']['all'].get('se_of_mean_iid'))}), FB {_f(ka['ADE_diff_CPU_minus_GPU']['FB'].get('mean'))}, "
               f"CT {_f(ka['ADE_diff_CPU_minus_GPU']['CT'].get('mean'))}; 0.05 m bar exceeded: {ka['bar_exceeded']}. Noise context on the same windows: GPU "
               f"seed1-seed0 mean {_f(g.get('mean'))} rms {_f(g.get('rms'))}; CPU N2-N mean {_f(c2.get('mean'))} rms {_f(c2.get('rms'))}. "
               f"Head argmax agreement CPU vs GPU {ka['head_argmax_agreement_CPU_vs_banked_GPU_8way']:.3f}; FB windows still BRAKE_TO on CPU "
               f"{_f(ka['FB_windows_whose_CPU_head_argmax_is_still_BRAKE_TO'], 3)}.")
    fl = ka.get("BANKED_GPU_seed1_minus_seed0_floor_on_the_FULL_planned_window_set")
    if fl:
        out.append("* **KA floor (banked, two independent GPU draws of the same model, the FULL planned 500-window set)**: " +
                   "; ".join(f"{k} mean {v['mean_diff']:+.4f} m CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] iid SE {v['iid_se']:.3f}, per-window rms {v['per_window_rms']:.2f} m, "
                              f"mode flips (>1 m) {v['frac_mode_flips_gt_1m']:.3f}" for k, v in fl.items()) + f". {ka.get('reading', '')}")
    out.append(f"* **KB** (the force reaches the planner): FB own-class minus F2 displacement {_ci(kb['FB_own_minus_F2'])} -> "
               f"PASS_FB {kb['PASS_FB (own-class displacement < different-class displacement, CI upper < 0)']}; CT own minus Fbrake "
               f"{_ci(kb['CT_own_minus_Fbrake'])} -> PASS_CT {kb['PASS_CT (same, CT)']}. F1 displacement over the sampler-noise displacement (FB): {_ci(kb['FB_F1_minus_N2'])}.")
    out.append(f"* candidate allocation: `{json.dumps(rep['controls']['force_reaches_candidate_allocation'])}`")
    out.append(f"* selected-candidate index changed vs N: `{json.dumps(rep['controls']['fraction_windows_selected_candidate_index_changed_vs_N'])}`")
    cd = rep["controls"]["CT_no_degradation"]
    out.append(f"* **CT no-degradation** (F1 on cruising non-brake windows): {_ci(cd['F1_minus_N_ADE'])}; degrades beyond CI: {cd['degrades_beyond_CI (CI lower bound > 0)']}.")
    kd = rep["controls"]["KD_positive_control"]["cells"]
    out.append(f"* **KD** (Fbrake on CT must CAUSE the defect): ADE {_ci(kd['ade_m'])}; signed along {_ci(kd['along_signed_final_m'])}.")
    out.append("\n## What this does NOT answer")
    out.append("One training seed (H-ESTIM-SEED-1); one fixed sampler seed per window (N2 reports the draw-to-draw floor); CPU fp32 trunk vs the banked GPU "
               "bf16 trunk (KA); FB is defined by the head's argmax in the BANKED GPU roll (a selection on the association side); open loop on logged "
               "frames -- not driving performance.")
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="stage", required=True)
    s = sp.add_parser("select")
    s.add_argument("--fb-max", type=int, default=250)
    s.add_argument("--ct-max", type=int, default=250)
    s.add_argument("--kb-n", type=int, default=100, help="control arms (N2, Fown, Fbrake) run on a seeded subset of N windows per set (0 = all)")
    s.add_argument("--n-chunks", type=int, default=5)
    r = sp.add_parser("roll")
    r.add_argument("--tag", default="full")
    r.add_argument("--limit-windows", type=int, default=0)
    r.add_argument("--save-every", type=int, default=5)
    r.add_argument("--chunk", type=int, default=-1, help="roll only the plan entries with this chunk id (-1 = all)")
    r.add_argument("--driver", type=int, default=0, help="N>0: run chunks 0..N-1 as sequential CHILD processes (resumable; asserts on the artifact)")
    z = sp.add_parser("analyze")
    z.add_argument("--tag", default="full")
    a = ap.parse_args()
    return {"select": stage_select, "roll": stage_roll, "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
