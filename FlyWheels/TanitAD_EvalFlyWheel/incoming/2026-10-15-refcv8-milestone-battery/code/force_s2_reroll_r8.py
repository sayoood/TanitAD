"""force_s2_reroll_r8.py -- the S2 INTERVENTION: is refcv8's near-term (0-2 s) over-run in slow / decelerating windows decided by the TIMING of the
tactical lon class, by its IDENTITY, or by the decoder?   NEW file; `force_lon_reroll_r8.py` is imported (loader, model build, `set_r8_force`,
per-window sampler seed, chunked resumable driver, ratio bootstrap), never edited and never copied.

CPU ONLY.  Four stages (one file, so the pre-registration, the roll and the analysis cannot drift apart):

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 R8_NO_POSE_SYNC=1 PYTHONIOENCODING=utf-8 \
    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
      python -X utf8 force_s2_reroll_r8.py select                 # S2A / S2B from the banked panel (e1_decomp_r8's own loader + dv2 rule)
      python -X utf8 force_s2_reroll_r8.py roll --driver 4        # DETACH (the launcher does): 4 sequential child processes, resumable
      python -X utf8 force_s2_reroll_r8.py status                 # artifact read only: rolled windows / planned windows
      python -X utf8 force_s2_reroll_r8.py analyze                # no model, no roll: numpy on the banked arrays + the roll artifact

PRE-REGISTERED 2026-10-18 (Master Mind brief); `decide()` was in this file from its creation at 21:16:56 +0200 and pinned by test_force_s2_r8.py (created
21:19:01) BEFORE any real number of this roll existed.  The table below is NOT editable here; `decide()` implements it and nothing else.

CONTEXT (MEASURED, raw/e1_decomp/E1_DECOMP.md): refcv8 EMA-30k, capped, ceiling-compliant S2 set (4,159 windows / 138 ep): E1 = os - ha0_ext S2 ADE
= -0.0085 [-0.0229, +0.0065], not separated.  Its gross loss +0.0264 is 72 % in slow / slowing windows (S2C +0.0189): refcv8 ends ~+0.25 m too FAR at
2 s where the human stays slow or keeps braking; its lon head argmax is ACCELERATE in 72 % of slow_both windows.

WINDOWS.  S2A = dv2 in {slow_both, stopping} (434), S2B = dv2 decelerating (424), both on the compliant S2 set, seed 0 (dv2 = the dv-class rule of
`s6_lon_decomp.partitions` on v2 = gt_future_ext[:, 19, 3], applied by `e1_decomp_r8.build_partitions` -- imported, not copied).  Each set is
subsampled to <= 220, stratified by episode (seeded), and n is recorded.

ARMS (same per-window sampler seed across arms):
  N        native, no force.
  F1       force the LON6 class mapped from the GT tactical label `lon_v7` (`force_lon_reroll_r8.f1_class`, LON6_V7_IDS = (0,1,3,4,5,7)); windows whose
           `lon_v7` is unlabelled (-100) or a masked v7 class are EXCLUDED from F1 and COUNTED.
  F_slow   force HOLD (LON6 index 4) on S2A and BRAKE_TO (LON6 index 2) on S2B.  Indices verified from source:
           stack/tanitad/refs/refcv8_conditioning.py:63  LON6 = ("FOLLOW","CRUISE","BRAKE_TO","CREEP","HOLD","ACCELERATE")  -> BRAKE_TO = 2, HOLD = 4.
           (The mechanism is `RefCV3Model.set_r8_force(lon6=...)`, stack/tanitad/refs/refc_v3.py:2081.)
  Fown     (100-window seeded subset per set) force each window's OWN head-argmax LON6 class.
  N2       (same 100-window subset) native at a second sampler seed = the inference-noise floor.
METRIC per window: d = ADE_arm(S2 0-2 s) - ADE_echo(S2), echo = the panel's `ha0_ext` for that window (as e1_decomp_r8 reads it); S2 = the first four
model slots (0.5 / 1.0 / 1.5 / 2.0 s).

PRIMARY STATISTIC (pre-registered): on S2A u S2B, closure_X = (mean d_N - mean d_X) / mean d_N, paired episode-cluster bootstrap CI (B 2000), for
X in {F1, F_slow}.  COMMITTED OUTCOMES (verbatim from the brief):
  TIMING-CONTROLLABLE  closure_Fslow >= 0.50 with CI lower bound > 0, AND closure_F1 < 0.50 -- the near-term plan obeys the class; the defect is the lon
                       class's TIMING (the label covers NOW+2..8 s).
  IDENTITY             closure_F1 >= 0.50 with CI lower bound > 0 -- the head picks the wrong class.
  DECODER              both closures' CI upper bounds < 0.25 -- forcing the class does not move the near-term plan.
  INCONCLUSIVE         otherwise (report n_needed).
IMPLEMENTATION READINGS of the table (they add no threshold and move none): (i) closure_F1 is computed on the windows that HAVE an F1 arm (labelled,
LON6-mapped), N paired on the same windows -- the committed text says unlabelled windows are excluded; closure_F_slow is computed on every window
that has the arm.  The like-for-like F_slow closure on the F1 windows is reported BESIDE it, because the populations differ.  (ii) an undefined closure
(non-positive denominator, or no data) makes the whole outcome INCONCLUSIVE, never a pass.  (iii) the bounds are compared as written: point >= 0.50,
lower > 0, upper < 0.25, point < 0.50.
BESIDE the primary: per-set closures, KA (CPU N vs the banked GPU `os`, against the banked inference floor), KB (Fown changes the plan less than
F_slow), and what E1 would read if S2A u S2B took the F_slow plans (ARITHMETIC, labelled as such, not a result).

TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  ESTIMATOR: paired episode-cluster bootstrap over the episodes
present (models FIXED, draws identical to taniteval.ci._draws): answers "another draw of EPISODES" only -- NOT another training run (one training
seed), NOT another inference run (the sampler seed is fixed per window; N2 reports the draw-to-draw floor).

HOW F_slow REUSES THE force_lon ROLL WITHOUT EDITING IT.  `force_lon_reroll_r8.stage_roll`'s proxy builds `specs["F2"] = (0, CRUISE6)` from the MODULE
GLOBAL `CRUISE6` on every call.  Each chunk of this plan holds ONE set, and the child process rebinds `FL.CRUISE6` to that set's slow class before
calling `FL.stage_roll`; the arm named "F2" in the roll artifact therefore means F_slow here.  The rebinding is guarded twice: statically (the spec
line must still read that way, `assert_fl_internals`) and by CONTENT (the analysis refuses unless every F2 record carries `forced6` equal to its
set's slow class).
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PKG = HERE.parent
OUT = PKG / "raw" / "force_s2"
SCR = Path("E:/refcv8_eval_kit/force_s2_scratch")                    # raw-id material and big tensors stay OFF the package/C:

import force_lon_reroll_r8 as FL  # noqa: E402  (imported, never edited)

TIER = FL.TIER
QUESTION = ("paired episode-cluster bootstrap over the episodes present in the window set, models FIXED, B 2000, seed 0 (draws identical to "
            "taniteval.ci._draws): answers 'another draw of EPISODES' only; NOT another training run (single training seed), NOT another inference "
            "run (the sampler seed is fixed per window; arm N2 reports the draw-to-draw floor)")
SEL_SEED = 20261019                                                   # the window-sampling seed, fixed here before any forced number exists
MAX_PER_SET = 220
KB_N = 100
CHUNKS_PER_SET = 2                                                    # 2 sets x 2 = 4 chunks; every chunk holds ONE set
SETS = ("S2A", "S2B")
SET_DEFS = {"S2A": ("slow_both", "stopping"), "S2B": ("decelerating",)}
SET_EXPECTED_N = {"S2A": 434, "S2B": 424}                             # literals from raw/e1_decomp/VERDICT_ANALYST.md section 4
SLOW_CLASS6 = {"S2A": 4, "S2B": 2}                                    # HOLD on S2A, BRAKE_TO on S2B (refcv8_conditioning.py:63, LON6 index)
S2_NSLOT = 4                                                          # S2 = model slots 0..3 = 0.5 / 1.0 / 1.5 / 2.0 s (panel grid '2s')
CLOSE_POINT, CLOSE_LO_GT, DEC_HI_LT = 0.50, 0.0, 0.25               # the committed outcome table (do not edit)
KA_BAR_M = 0.05
#: BRIEF literal of the compliant E1 cell (seed 0): the saved arrays must reproduce it (control R).  Written as literals, never derived.
E1_LITERAL = {"delta": -0.0085, "lo": -0.0229, "hi": 0.0065, "n_windows": 4159, "n_episodes": 138}
N_BOOT = FL.N_BOOT


# ====================================================================================================== small helpers
def clean(x):
    """JSON-safe copy: numpy scalars / arrays -> python (json.dump(default=str) would turn np.int64 into a STRING)."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def bind_fl() -> None:
    """force_lon's driver, roll and result paths all read ITS module global `SCR`; point it at this roll's scratch."""
    FL.SCR = SCR


def assert_fl_internals() -> None:
    """The F_slow mechanism (module-global CRUISE6 read at call time) must still be what force_lon's source says.  Fails in 2 s, not after 3 h."""
    src = inspect.getsource(FL.stage_roll)
    if '"F2": (0, CRUISE6)' not in src:
        raise SystemExit("[force_s2] force_lon_reroll_r8.stage_roll no longer builds specs['F2'] = (0, CRUISE6): the F_slow rebinding would be a no-op")
    if re.search(r"^\s*(global\s+)?CRUISE6\s*=", src, re.M):
        raise SystemExit("[force_s2] stage_roll assigns CRUISE6 itself: the module-global rebinding would be shadowed")
    if FL.stage_roll.__globals__ is not vars(FL):
        raise SystemExit("[force_s2] stage_roll does not resolve names in force_lon_reroll_r8's own module dict")
    if (FL.LON6[SLOW_CLASS6["S2A"]], FL.LON6[SLOW_CLASS6["S2B"]]) != ("HOLD", "BRAKE_TO"):
        raise SystemExit("[force_s2] LON6 indices drifted: SLOW_CLASS6 no longer names HOLD / BRAKE_TO")


@contextlib.contextmanager
def slow_class_bound(set_name: str):
    """Rebind force_lon's CRUISE6 to this set's slow class for the duration of a chunk (restored afterwards, also on error)."""
    old = FL.CRUISE6
    FL.CRUISE6 = SLOW_CLASS6[set_name]
    try:
        yield
    finally:
        FL.CRUISE6 = old


# ====================================================================================================== COMMITTED DECISION TABLE
def decide(f1, fslow) -> str:
    """The COMMITTED outcome table, verbatim from the brief.  f1 / fslow: dicts with point, lo, hi (the closure and its episode-cluster CI) or None."""
    for c in (f1, fslow):
        if not c or c.get("point") is None or c.get("lo") is None or c.get("hi") is None:
            return "INCONCLUSIVE (closure undefined: no data or the denominator is not positive)"
    if fslow["point"] >= CLOSE_POINT and fslow["lo"] > CLOSE_LO_GT and f1["point"] < CLOSE_POINT:
        return "TIMING-CONTROLLABLE"
    if f1["point"] >= CLOSE_POINT and f1["lo"] > CLOSE_LO_GT:
        return "IDENTITY"
    if f1["hi"] < DEC_HI_LT and fslow["hi"] < DEC_HI_LT:
        return "DECODER"
    return "INCONCLUSIVE"


def n_needed_s2(f1, fslow, n_f1, n_fs) -> dict:
    """ESTIMATED windows needed (CI half-width ~ 1/sqrt(n), point held), per arm, via force_lon's n_needed (same 0.50 / 0.25 thresholds)."""
    out = {}
    for nm, c, n in (("F1", f1, n_f1), ("F_slow", fslow, n_fs)):
        if not c or c.get("point") is None:
            out[nm] = {"status": "closure undefined"}
            continue
        r = FL.n_needed(c["point"], c["lo"], c["hi"], n)
        out[nm] = {"assumption": r["assumption"], "to_reach_closure_ge_0.50_with_lo_gt_0": r.get("to_reach_HEAD"),
                   "to_reach_hi_lt_0.25": r.get("to_reach_DECODER")}
    return out


# ====================================================================================================== pure selection logic
def assign_sets(dv2, mask) -> dict:
    """{'S2A': bool[n], 'S2B': bool[n]}: the dv2 rule on the COMPLIANT windows only."""
    d = np.asarray([str(x) for x in dv2])
    m = np.asarray(mask, dtype=bool)
    return {s: m & np.isin(d, SET_DEFS[s]) for s in SETS}


def stratified_pick(set_mask, ep, n, rng) -> list:
    """<= n rows of `set_mask`, proportional over episodes (largest remainder, never above an episode's own count), shuffled within episode."""
    by = {}
    for i in np.nonzero(set_mask)[0]:
        by.setdefault(int(ep[i]), []).append(int(i))
    take = FL.apportion({e: len(v) for e, v in by.items()}, n, rng)
    sel = []
    for e in sorted(by):
        idx = list(by[e])
        rng.shuffle(idx)
        sel += idx[:take[e]]
    return sorted(sel)


def build_plan(ref: dict, clip_of_ep: dict, *, seed=SEL_SEED, max_per_set=MAX_PER_SET, kb_n=KB_N, chunks_per_set=CHUNKS_PER_SET,
               expect_n=SET_EXPECTED_N) -> tuple:
    """Plan entries + stats from the banked S2 arrays (pure: no I/O, no model)."""
    ep, ws, mask = np.asarray(ref["ep_index"]), np.asarray(ref["ws"]), np.asarray(ref["mask"], bool)
    sets = assign_sets(ref["dv2"], mask)
    if expect_n is not None:
        got = {s: int(sets[s].sum()) for s in SETS}
        if got != dict(expect_n):
            raise SystemExit(f"[force_s2] regenerated sets {got} != the analyst's {dict(expect_n)} (VERDICT_ANALYST.md section 4): refusing to plan")
    rng = np.random.default_rng(seed)
    entries = []
    for si, s in enumerate(SETS):
        rows = stratified_pick(sets[s], ep, max_per_set, rng) if sets[s].sum() > max_per_set else sorted(np.nonzero(sets[s])[0].tolist())
        es = []
        for i in rows:
            c, why = FL.f1_class(int(ref["lon_v7"][i]))
            clip = clip_of_ep[int(ep[i])]
            es.append({"set": s, "row": int(i), "clip": clip, "ws": int(ws[i]), "ep_index": int(ep[i]), "dv2": str(ref["dv2"][i]),
                       "lon_v7": int(ref["lon_v7"][i]), "f1_class": c, "f1_excluded_reason": why, "head_argmax_v7": int(ref["tac"][i]),
                       "v0": float(ref["v0"][i]), "slow_class6": SLOW_CLASS6[s], "clip_sha12": str(ref["csha"][i]),
                       "arms": (["F1"] if c is not None else []) + ["F2"], "control_subset": False})
        keep = set(rng.permutation(len(es))[:kb_n if kb_n else len(es)].tolist())
        order = rng.permutation(len(es))
        for k, e in enumerate(es):
            e["chunk"] = int(si * chunks_per_set + order[k] % chunks_per_set)
            if k in keep:
                e["arms"] = e["arms"] + ["N2", "Fown"]
                e["control_subset"] = True
        entries += es
    st = {"sel_seed": seed, "max_per_set": max_per_set, "kb_n": kb_n, "chunks_per_set": chunks_per_set, "n_chunks": len(SETS) * chunks_per_set}
    for s in SETS:
        es = [e for e in entries if e["set"] == s]
        st[s] = {"n_in_compliant_S2_set": int(sets[s].sum()), "n_episodes_in_set": int(len(set(ep[sets[s]].tolist()))),
                 "n_used": len(es), "n_episodes_used": len({e["ep_index"] for e in es}),
                 "n_with_F1": sum(1 for e in es if e["f1_class"] is not None),
                 "F1_excluded_by_reason": {r: sum(1 for e in es if e["f1_excluded_reason"] == r)
                                           for r in sorted({e["f1_excluded_reason"] for e in es if e["f1_excluded_reason"]})},
                 "F1_class_counts": {FL.LON6[k]: sum(1 for e in es if e["f1_class"] == k) for k in range(6)},
                 "head_argmax_v7_hist": {str(k): sum(1 for e in es if e["head_argmax_v7"] == k) for k in range(8)},
                 "n_control_subset": sum(1 for e in es if e["control_subset"]),
                 "chunks": sorted({e["chunk"] for e in es}),
                 "forced_forwards_per_window_mean": float(np.mean([len(e["arms"]) for e in es])) if es else None}
    nf = sum(len(e["arms"]) for e in entries)
    st["ESTIMATED_wall_s"] = {"basis": "force_lon HANDOFF: N forward incl. trunk ~10 s, each forced forward ~5 s (CPU, 8 threads); ESTIMATED, not measured here",
                              "total": 10.0 * len(entries) + 5.0 * nf}
    return entries, st


# ====================================================================================================== stage: select
def load_s2_arrays() -> tuple:
    """The banked S2 arrays of seed 0 / 1 through e1_decomp_r8's OWN loader (imported).  Needs the commit headroom the launcher waits for."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[force_s2] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    import clamp_posthoc_r8 as CP
    import e1_decomp_r8 as E1
    NR, _BP, PR, ra, sha12 = CP.get_readers()
    tools = {"NR": NR, "PR": PR, "ra": ra, "sha12": sha12}
    rc = CP.run_cfg()
    comp_json = FL.jload(str(E1.COMPLIANT_JSON))
    if not comp_json["controls"]["ALL_PASS"]:
        raise SystemExit("[force_s2] the compliant reread's own controls did not pass: refusing")
    cx0 = E1.load_seed(0, tools, rc, comp_json)
    cx1 = E1.load_seed(1, tools, rc, comp_json)
    for cx in (cx0, cx1):
        ok = (cx["sha"] == E1.MASK_SHA and int(cx["mask_full"].sum()) == 4159 and len(cx["mask_full"]) == 4754
              and len(np.unique(cx["eid"][cx["mask_full"]])) == 138)
        if not ok:
            raise SystemExit(f"[force_s2] control M failed for seed {cx['seed']}: the compliant mask is not the banked one")
    if not (np.array_equal(cx0["ep_index"], cx1["ep_index"]) and np.array_equal(cx0["ws"], cx1["ws"])):
        raise SystemExit("[force_s2] seed-1 panel window order != seed-0")
    n = len(cx0["mask_full"])
    _p, pinfo = E1.build_partitions(cx0, cx0["mask_full"], {"labels": np.full(n, "other", dtype=object)})
    pdir = f"{E1.ARCHIVE}/{E1.TREE}/panel_s0"
    man, _files, ep_of = NR.panel_layout(pdir)
    slots = [int(x) for x in man["grid"]["slots"]]
    if slots != list(range(S2_NSLOT)) or abs(float(man["grid"]["dt_s"]) - 0.5) > 1e-9:
        raise SystemExit(f"[force_s2] S2 grid is not slots 0..3 at dt 0.5: {man['grid']}")
    cid = {int(e["episode_index"]): str(e["clip_id"]) for e in man["episodes"]}
    for ei, c in zip(cx0["ep_index"].tolist(), cx0["csha"].tolist()):
        if sha12(cid[int(ei)]) != c:
            raise SystemExit("[force_s2] clip id <-> sha12 mismatch between the manifest and the panel")
    ref = {"ep_index": cx0["ep_index"], "ws": cx0["ws"], "mask": cx0["mask_full"], "dv2": np.asarray([str(x) for x in pinfo["dv2_all"]]),
           "lon_v7": cx0["lon_v7"], "tac": cx0["tac"], "v0": cx0["v0"], "csha": np.asarray([str(x) for x in cx0["csha"]]),
           "eid": np.asarray([str(x) for x in cx0["eid"]]), "G": cx0["A"]["g"].astype(np.float32), "os0": cx0["A"]["os"].astype(np.float32),
           "os1": cx1["A"]["os"].astype(np.float32), "ha": cx0["A"]["ha0_ext"].astype(np.float32)}
    return ref, cid


def save_ref(ref: dict, cid: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, **ref, cid_ep=np.asarray(sorted(cid), dtype=np.int64), cid_clip=np.asarray([cid[k] for k in sorted(cid)]))
    os.replace(tmp, path)


def load_ref(path) -> tuple:
    z = np.load(path, allow_pickle=False)
    ref = {k: z[k] for k in z.files if not k.startswith("cid_")}
    return ref, {int(e): str(c) for e, c in zip(z["cid_ep"], z["cid_clip"])}


def stage_select(a) -> int:
    assert_fl_internals()                                  # fail HERE in 2 s, not after the 3-hour roll, if force_lon's source drifted
    if a.ref_npz:
        ref, cid = load_ref(a.ref_npz)
    else:
        ref, cid = load_s2_arrays()
        save_ref(ref, cid, SCR / "s2_ref.npz")
    entries, st = build_plan(ref, cid, max_per_set=a.max_per_set, kb_n=a.kb_n, chunks_per_set=a.chunks_per_set,
                             expect_n=None if a.no_expect_counts else SET_EXPECTED_N)
    SCR.mkdir(parents=True, exist_ok=True)
    FL.jdump(clean({"entries": entries, "sel_seed": SEL_SEED}), SCR / "plan_raw.json")
    st.update({"tool": "force_s2_reroll_r8.py select", "generated": time.strftime("%FT%T%z"), "tier": TIER,
               "note": "raw clip ids are in E:/refcv8_eval_kit/force_s2_scratch/plan_raw.json only; plan_sha12.json is sha12-keyed"})
    FL.jdump(clean(st), OUT / "select_stats.json")
    FL.jdump(clean([{k: v for k, v in e.items() if k not in ("clip", "row")} for e in entries]), OUT / "plan_sha12.json")
    print(json.dumps(clean({s: {k: st[s][k] for k in ("n_in_compliant_S2_set", "n_used", "n_with_F1", "n_control_subset", "chunks")} for s in SETS}), indent=1))
    print("ZZFS2SELECTOKZZ")
    return 0


# ====================================================================================================== stage: roll (force_lon's, by import)
def stage_roll(a) -> int:
    bind_fl()
    assert_fl_internals()
    if a.limit_windows:
        raise SystemExit("[force_s2] --limit-windows is not supported (the plan's chunks are single-set; use --chunk)")
    if a.driver > 0:
        n_chunks = 1 + max(int(e["chunk"]) for e in FL.jload(SCR / "plan_raw.json")["entries"])
        if a.driver != n_chunks:
            raise SystemExit(f"[force_s2] --driver {a.driver} but the plan holds {n_chunks} chunks: a smaller driver would silently skip a whole set")
        FL.__file__ = str(Path(__file__).resolve())      # force_lon's driver spawns `<its __file__> roll --chunk c`: point it at THIS file
        return FL.stage_roll_driver(a)
    if a.chunk < 0:
        raise SystemExit("[force_s2] roll needs --driver N (the whole plan) or --chunk c (one single-set chunk)")
    entries = FL.jload(SCR / "plan_raw.json")["entries"]
    sets = {e["set"] for e in entries if e.get("chunk") == a.chunk}
    if len(sets) != 1:
        raise SystemExit(f"[force_s2] chunk {a.chunk} holds sets {sorted(sets)}: a chunk must hold exactly ONE set (the F_slow class is bound per chunk)")
    with slow_class_bound(next(iter(sets))):
        return FL.stage_roll(a)


def stage_status(a) -> int:
    bind_fl()
    plan = FL.jload(SCR / "plan_raw.json")["entries"]
    p = FL._res_path(a.tag)
    have = set(FL.jload(p)["windows"]) if p.exists() else set()
    allw = {f"{e['clip']}|{e['ws']}" for e in plan}
    print(f"ZZFS2STATUSZZ rolled={len(have & allw)}/{len(allw)}")
    return 0


# ====================================================================================================== stage: analyze
def ade_of(P, G):
    return np.linalg.norm(np.asarray(P, dtype=np.float64) - np.asarray(G, dtype=np.float64), axis=-1).mean(-1)


def content_check(entries, windows) -> dict:
    """The recorded `forced6` of every rolled arm must be what the plan says it forced (the rebinding is verified by CONTENT, not by trust)."""
    bad, missing, n = [], [], 0
    for e in entries:
        k = f"{e['clip']}|{e['ws']}"
        w = windows.get(k)
        if w is None:
            continue
        n += 1
        arms = w.get("arms", {})
        for nm in e["arms"]:
            if nm not in arms:
                missing.append((e["clip_sha12"], e["ws"], nm))
                continue
            f6 = arms[nm].get("forced6")
            want = {"F2": e["slow_class6"], "F1": e["f1_class"], "Fown": w.get("own6"), "N2": None}.get(nm)
            if f6 != want:
                bad.append({"clip_sha12": e["clip_sha12"], "ws": e["ws"], "arm": nm, "forced6": f6, "expected": want})
        if "N" not in arms:
            missing.append((e["clip_sha12"], e["ws"], "N"))
    return {"n_windows_checked": n, "n_forced6_mismatch": len(bad), "mismatch_head": bad[:5], "n_missing_arm_records": len(missing),
            "missing_head": missing[:5], "PASS": bool(n > 0 and not bad and not missing)}


def analyze_core(ref: dict, entries: list, windows: dict, SD, expect_e1=None) -> dict:  # noqa: C901
    """Pure numpy: the banked S2 arrays + the plan + the roll artifact -> the full report (verdict, per-set closures, KA, KB, controls, E1 arithmetic)."""
    rows_all = np.asarray([e["row"] for e in entries])
    keys_all = [f"{e['clip']}|{e['ws']}" for e in entries]
    have = np.asarray([k in windows for k in keys_all])
    ents = [e for e, h in zip(entries, have) if h]
    keys = [k for k, h in zip(keys_all, have) if h]
    rows = rows_all[have]
    n = len(ents)
    rep = {"tool": "force_s2_reroll_r8.py analyze", "tier": TIER, "estimator": QUESTION, "n_boot": N_BOOT, "boot_seed": FL.BOOT_SEED,
           "n_planned": len(entries), "n_rolled": n, "controls": {}}
    rep["coverage"] = {s: {"planned": sum(1 for e in entries if e["set"] == s), "rolled": sum(1 for e in ents if e["set"] == s)} for s in SETS}
    rep["coverage"]["PARTIAL"] = bool(n < len(entries))
    if n < 2:
        rep["PRIMARY"] = {"verdict": "INCONCLUSIVE (fewer than 2 rolled windows)"}
        return rep
    eid = np.asarray(ref["eid"])[rows]
    G = np.asarray(ref["G"], np.float64)[rows]
    ade_echo = ade_of(np.asarray(ref["ha"])[rows], G)
    ade_gpu0, ade_gpu1 = ade_of(np.asarray(ref["os0"])[rows], G), ade_of(np.asarray(ref["os1"])[rows], G)
    setv = np.asarray([e["set"] for e in ents])
    isA, isB = setv == "S2A", setv == "S2B"
    union = isA | isB
    own6 = np.asarray([windows[k].get("own6", -1) for k in keys])
    f1c = np.asarray([-1 if e["f1_class"] is None else e["f1_class"] for e in ents])
    traj = {}
    ade = {}
    for nm in ("N", "F1", "F2", "N2", "Fown"):
        P = np.full((n, S2_NSLOT, 2), np.nan)
        for j, k in enumerate(keys):
            r = windows[k]["arms"].get(nm)
            if r is not None:
                P[j] = np.asarray(r["traj"], dtype=np.float64)[:S2_NSLOT]
        traj[nm] = P
        ade[nm] = ade_of(P, G)                                 # NaN rows where the arm was not run
    ade["GPU_N_banked"], ade["GPU_N_banked_s1"] = ade_gpu0, ade_gpu1
    ade["echo"] = ade_echo
    rep["content_check"] = content_check(ents, windows)
    rep["sample"] = {s: {"planned": rep["coverage"][s]["planned"], "rolled": rep["coverage"][s]["rolled"],
                         "with_F1": int(((setv == s) & np.isfinite(ade["F1"])).sum()),
                         "F1_excluded": int(((setv == s) & ~np.isfinite(ade["F1"])).sum()),
                         "with_controls (Fown, N2)": int(((setv == s) & np.isfinite(ade["Fown"])).sum())} for s in SETS}

    def mci(x, sel):
        m = sel & np.isfinite(x)
        return FL.mean_ci(x[m], eid[m], SD) if m.sum() >= 2 else {"n_windows": int(m.sum())}

    def dcell(new, ref_, sel):
        m = sel & np.isfinite(new) & np.isfinite(ref_)
        if m.sum() < 2:
            return {"status": "NO DATA", "n_windows": int(m.sum())}
        r = FL.mean_ci((new - ref_)[m], eid[m], SD)
        r["delta"], r["separated"] = r.pop("mean"), bool(r["lo"] > 0 or r["hi"] < 0)
        return r

    def closure(arm, sel, base="N"):
        m = sel & np.isfinite(ade[arm]) & np.isfinite(ade[base])
        if m.sum() < 2:
            return {"status": "NO DATA", "point": None, "lo": None, "hi": None, "n_windows": int(m.sum())}
        r = FL.ratio_boot(ade[base][m] - ade[arm][m], ade[base][m] - ade_echo[m], eid[m], SD)
        r["metric"], r["base"] = "ade_m (S2 0-2 s)", base
        return r

    # ---- controls: R (the saved arrays reproduce the brief's E1 cell), content (forced6), N-vs-banked (KA below)
    if expect_e1 is not None:
        comp = np.asarray(ref["mask"], bool)
        Gf = np.asarray(ref["G"], np.float64)
        dfull = ade_of(np.asarray(ref["os0"]), Gf) - ade_of(np.asarray(ref["ha"]), Gf)
        c = FL.mean_ci(dfull[comp], np.asarray(ref["eid"])[comp], SD)
        got = {"delta": round(c["mean"], 4), "lo": round(c["lo"], 4), "hi": round(c["hi"], 4), "n_windows": c["n_windows"], "n_episodes": c["n_episodes"]}
        rep["controls"]["R_E1_cell_from_saved_arrays"] = {"got": got, "brief_literal": expect_e1, "PASS": bool(got == expect_e1)}
    # ---- per-arm means of d (vs the echo) on each set and the union
    d = {nm: ade[nm] - ade_echo for nm in ("N", "F1", "F2", "N2", "Fown", "GPU_N_banked")}
    rep["means_d_vs_echo"] = {sn: {nm: mci(d[nm], sel) for nm in d} for sn, sel in (("S2A", isA), ("S2B", isB), ("S2A_u_S2B", union))}
    # ---- PRIMARY
    f1 = closure("F1", union)
    fs = closure("F2", union)
    f1_has = union & np.isfinite(ade["F1"])
    prim = {"closure_F1": f1, "closure_Fslow": fs, "closure_Fslow_like_for_like_on_F1_windows": closure("F2", f1_has),
            "n_needed": n_needed_s2(f1, fs, f1.get("n_windows", 0), fs.get("n_windows", 0))}
    prim["verdict"] = decide(f1, fs)
    if not rep["content_check"]["PASS"]:
        prim["verdict_per_committed_table"] = prim["verdict"]
        prim["verdict"] = "REFUSED (content check failed: a recorded forced6 differs from the plan, or an arm record is missing)"
    rep["PRIMARY"] = prim
    rep["per_set_closures"] = {sn: {"F1": closure("F1", sel), "F_slow": closure("F2", sel)} for sn, sel in (("S2A", isA), ("S2B", isB))}
    rep["secondary"] = {
        "N2_as_the_numerator_arm (a NULL intervention, another sampler draw: the noise floor of this statistic)": closure("N2", union),
        "F_slow_numerator_vs_BANKED_GPU_N (mixes devices: sensitivity only)": (
            lambda m: FL.ratio_boot(ade["GPU_N_banked"][m] - ade["F2"][m], ade["GPU_N_banked"][m] - ade_echo[m], eid[m], SD)
            if m.sum() >= 2 else {"status": "NO DATA"})(union & np.isfinite(ade["F2"])),
        "cells_ADE_S2_arm_minus_N": {sn: {nm: dcell(ade[nm], ade["N"], sel) for nm in ("F1", "F2", "N2", "Fown")}
                                     for sn, sel in (("S2A", isA), ("S2B", isB), ("S2A_u_S2B", union))}}
    # ---- KA: CPU native vs the banked GPU `os`, against the banked inference floor (same windows)
    dA = ade["N"] - ade_gpu0
    ds1 = ade_gpu1 - ade_gpu0
    dn2 = ade["N2"] - ade["N"]

    def spread(x, sel):
        x = x[sel & np.isfinite(x)]
        if x.size == 0:
            return {"n": 0}
        return {"n": int(x.size), "mean": float(x.mean()), "rms": float(np.sqrt((x ** 2).mean())),
                "sd": float(x.std(ddof=1)) if x.size > 1 else None, "se_of_mean_iid": float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else None}
    allw = np.ones(n, bool)
    sp = {"all": spread(dA, allw), "S2A": spread(dA, isA), "S2B": spread(dA, isB)}
    wp = np.linalg.norm(traj["N"] - np.asarray(ref["os0"], np.float64)[rows], axis=-1)      # L2 per waypoint
    rep["controls"]["KA"] = {
        "what": "CPU native (arm N, per-window sampler seed, fp32 trunk) vs the banked GPU `os` (global sequential sampler stream, bf16 trunk) on the S2 slots: "
                "BIT-IDENTITY IS IMPOSSIBLE, so this is a statistical agreement check against the brief's 0.05 m bar, read against the banked inference floor.",
        "ADE_diff_CPU_N_minus_GPU_os": sp, "bar_abs_mean_ADE_diff_m": KA_BAR_M,
        "bar_exceeded": {k: (bool(abs(v["mean"]) > KA_BAR_M) if v.get("n") else None) for k, v in sp.items()},
        "max_L2_waypoint_diff_S2_slots_m": float(wp.max()), "median_L2_waypoint_diff_S2_slots_m": float(np.median(wp)),
        "BANKED_INFERENCE_FLOOR_GPU_seed1_minus_seed0_os_same_windows": {"all": spread(ds1, allw), "S2A": spread(ds1, isA), "S2B": spread(ds1, isB)},
        "CPU_N2_minus_CPU_N_two_CPU_draws (control subset)": {"all": spread(dn2, allw), "S2A": spread(dn2, isA), "S2B": spread(dn2, isB)},
        "head_argmax_agreement_CPU_vs_banked_GPU_8way": float(np.mean([windows[k].get("head_argmax_v7") == int(ref["tac"][r]) for k, r in zip(keys, rows)])),
        "reading": "The 0.05 m bar is applied AS WRITTEN (bar_exceeded). A CPU-vs-GPU mean difference inside the banked floor is what a faithful re-roll "
                   "looks like; the Master Mind reads an exceedance against that floor."}
    # ---- KB: forcing the head's OWN class moves the plan less than forcing a DIFFERENT class (F_slow)
    D = {nm: np.linalg.norm(traj[nm] - traj["N"], axis=-1).mean(1) for nm in ("F1", "F2", "N2", "Fown")}
    kb1 = dcell(D["Fown"], D["F2"], union)
    rep["controls"]["KB"] = {
        "what": "displacement of the S2 plan from native N (mean over the 4 slots of the L2 distance, m).  Forcing the head's OWN argmax class must move the "
                "plan LESS than forcing F_slow (proves the force reaches the planner and that the shift is not sampler noise, N2 = another draw).",
        "mean_displacement_m": {sn: {nm: (mci(D[nm], sel)) for nm in D} for sn, sel in (("S2A", isA), ("S2B", isB), ("S2A_u_S2B", union))},
        "own_minus_Fslow": kb1, "Fslow_minus_N2": dcell(D["F2"], D["N2"], union), "F1_minus_N2": dcell(D["F1"], D["N2"], union),
        "PASS (own-class displacement < F_slow displacement, CI upper < 0)": (bool(kb1["hi"] < 0) if "hi" in kb1 else None),
        "frac_windows_where_F_slow_class_equals_the_head_own_class": float(np.mean(own6[union & (own6 >= 0)] == np.asarray(
            [e["slow_class6"] for e in ents])[union & (own6 >= 0)])) if (union & (own6 >= 0)).any() else None}
    # ---- E1 arithmetic: what the compliant E1 cell would read if S2A u S2B took the F_slow plans (SHIFT form; ARITHMETIC, NOT a result)
    comp = np.asarray(ref["mask"], bool)
    Gf = np.asarray(ref["G"], np.float64)
    dfull = ade_of(np.asarray(ref["os0"]), Gf) - ade_of(np.asarray(ref["ha"]), Gf)
    eidf = np.asarray(ref["eid"])
    sets_full = assign_sets(ref["dv2"], comp)
    shift = {}
    dprime = dfull.copy()
    for sn, sel in (("S2A", isA), ("S2B", isB)):
        m = sel & np.isfinite(ade["F2"]) & np.isfinite(ade["N"])
        if m.sum():
            shift[sn] = float((ade["F2"][m] - ade["N"][m]).mean())
            dprime[sets_full[sn]] += shift[sn]
    e1_native = FL.mean_ci(dfull[comp], eidf[comp], SD)
    e1_if = FL.mean_ci(dprime[comp], eidf[comp], SD)
    rep["ARITHMETIC_E1_if_S2A_u_S2B_took_the_Fslow_plans"] = {
        "LABEL": "ARITHMETIC, NOT A RESULT.  SHIFT form: each full set's banked d is moved by the rolled subsample's mean (ADE_F_slow - ADE_N), CPU vs CPU, "
                 "and the E1 cell is re-read on the SAME draws.  The CI is conditional on that shift being exact for the unrolled windows.",
        "shift_per_set_m": shift, "E1_native_banked_GPU_os": e1_native, "E1_if_Fslow": e1_if,
        "separated_lt0_if": bool(e1_if["hi"] < 0), "n_windows": int(comp.sum())}
    # ---- per-window table (sha12 keyed; no raw clip ids)
    rep["per_window_sha12"] = [{"clip_sha12": e["clip_sha12"], "ws": e["ws"], "set": e["set"], "dv2": e["dv2"],
                                **{f"ade:{nm}": (None if not np.isfinite(ade[nm][j]) else round(float(ade[nm][j]), 5))
                                   for nm in ("N", "F1", "F2", "N2", "Fown", "echo", "GPU_N_banked")}} for j, e in enumerate(ents)]
    rep["headline"] = {"tier": TIER, "estimator": QUESTION, "verdict": prim["verdict"],
                       **{k: {q: prim[k].get(q) for q in ("point", "lo", "hi", "n_windows", "n_episodes")} for k in ("closure_F1", "closure_Fslow")},
                       "KA_bar_exceeded": rep["controls"]["KA"]["bar_exceeded"],
                       "KB_PASS": rep["controls"]["KB"]["PASS (own-class displacement < F_slow displacement, CI upper < 0)"],
                       "content_check_PASS": rep["content_check"]["PASS"], "PARTIAL": rep["coverage"]["PARTIAL"]}
    return rep


def _f(x, nd=4):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, (int, float)) else str(x))


def _clo(c):
    if not isinstance(c, dict) or c.get("point") is None:
        return "undefined"
    return f"{c['point']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] (n {c['n_windows']} / {c['n_episodes']} ep)"


def write_md(rep: dict, path) -> None:
    p = rep["PRIMARY"]
    out = [f"# FORCE_S2 -- is refcv8's near-term over-run in slow / slowing windows a TIMING, IDENTITY or DECODER defect?  (tag {rep.get('tag')})", ""]
    out.append(f"Stamp for every number: **{rep['tier']}**.  Model: refcv8-r101-s0 EMA step 30,000 on CPU (fp32 trunk), per-window sampler seed identical across "
               f"arms.  Estimator: {rep['estimator']}.")
    out.append("")
    cov = rep["coverage"]
    out.append(f"Coverage: rolled {rep['n_rolled']} of {rep['n_planned']} planned (S2A {cov['S2A']['rolled']}/{cov['S2A']['planned']}, S2B "
               f"{cov['S2B']['rolled']}/{cov['S2B']['planned']}){'  **PARTIAL ROLL**' if cov['PARTIAL'] else ''}.  Content check (recorded forced6 == plan): "
               f"**{'PASS' if rep['content_check']['PASS'] else 'FAIL'}**.")
    out.append("")
    out.append("## Verdict (committed table: TIMING-CONTROLLABLE | IDENTITY | DECODER | INCONCLUSIVE)")
    out.append("")
    out.append(f"**{p['verdict']}**")
    out.append("")
    out.append(f"* closure_F_slow on S2A u S2B: {_clo(p['closure_Fslow'])}")
    out.append(f"* closure_F1 on the windows that have an F1 arm: {_clo(p['closure_F1'])}")
    out.append(f"* like-for-like (F_slow on the F1 windows): {_clo(p['closure_Fslow_like_for_like_on_F1_windows'])}")
    out.append(f"* n needed (ESTIMATED): `{json.dumps(clean(p['n_needed']))}`")
    out.append("")
    out.append("## Per-set closures")
    for sn, v in rep["per_set_closures"].items():
        out.append(f"* {sn}: F1 {_clo(v['F1'])}; F_slow {_clo(v['F_slow'])}")
    out.append("")
    out.append("## Mean d = ADE_arm - ADE_echo (S2 0-2 s, m; episode-cluster CI)")
    for sn, v in rep["means_d_vs_echo"].items():
        out.append(f"* {sn}: " + "; ".join(f"{nm} {_f(c.get('mean'))} [{_f(c.get('lo'))}, {_f(c.get('hi'))}] n {c.get('n_windows')}" for nm, c in v.items()))
    ka, kb = rep["controls"]["KA"], rep["controls"]["KB"]
    fl = ka["BANKED_INFERENCE_FLOOR_GPU_seed1_minus_seed0_os_same_windows"]["all"]
    out.append("")
    out.append("## Controls")
    out.append(f"* **KA** (CPU N vs banked GPU `os`): mean ADE diff all {_f(ka['ADE_diff_CPU_N_minus_GPU_os']['all'].get('mean'))} m (iid SE "
               f"{_f(ka['ADE_diff_CPU_N_minus_GPU_os']['all'].get('se_of_mean_iid'))}); 0.05 m bar exceeded: {ka['bar_exceeded']}.  Banked floor (GPU seed1 - seed0, same "
               f"windows): mean {_f(fl.get('mean'))} rms {_f(fl.get('rms'))}.  Median L2 waypoint diff {ka['median_L2_waypoint_diff_S2_slots_m']:.4f} m, max "
               f"{ka['max_L2_waypoint_diff_S2_slots_m']:.3f} m.  Head argmax agreement {ka['head_argmax_agreement_CPU_vs_banked_GPU_8way']:.3f}.")
    own = kb["own_minus_Fslow"]
    out.append(f"* **KB** (Fown moves the plan less than F_slow): displacement diff {_f(own.get('delta'))} [{_f(own.get('lo'))}, {_f(own.get('hi'))}] -> PASS "
               f"{kb['PASS (own-class displacement < F_slow displacement, CI upper < 0)']}.")
    r = rep["controls"].get("R_E1_cell_from_saved_arrays")
    if r:
        out.append(f"* **R** (saved arrays reproduce the brief's E1 cell): {r['got']} vs {r['brief_literal']} -> {'PASS' if r['PASS'] else 'FAIL'}.")
    ar = rep["ARITHMETIC_E1_if_S2A_u_S2B_took_the_Fslow_plans"]
    out.append("")
    out.append("## E1 if S2A u S2B took the F_slow plans -- ARITHMETIC, NOT A RESULT")
    out.append(f"{ar['LABEL']}")
    out.append(f"* shift per set (m): `{json.dumps(clean(ar['shift_per_set_m']))}`; E1 native {_f(ar['E1_native_banked_GPU_os'].get('mean'))} "
               f"[{_f(ar['E1_native_banked_GPU_os'].get('lo'))}, {_f(ar['E1_native_banked_GPU_os'].get('hi'))}] -> if F_slow {_f(ar['E1_if_Fslow'].get('mean'))} "
               f"[{_f(ar['E1_if_Fslow'].get('lo'))}, {_f(ar['E1_if_Fslow'].get('hi'))}] (separated < 0: {ar['separated_lt0_if']}).")
    out.append("")
    out.append("## What this does NOT answer")
    out.append("One training seed (H-ESTIM-SEED-1); one fixed sampler seed per window (N2 reports the draw-to-draw floor); CPU fp32 trunk vs the banked GPU bf16 "
               "trunk (KA); the sets are a stratified SUBSAMPLE of S2A / S2B; closure_F1 and closure_F_slow are on different populations (like-for-like beside); "
               "open loop on logged frames -- not driving performance.  A timing verdict identifies WHERE the defect lives; the fix itself (time-aware "
               "tactical conditioning) is a retraining decision for the PI.")
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


def stage_analyze(a) -> int:
    bind_fl()
    import s6_lon_decomp as SD
    ref, _cid = load_ref(a.ref_npz or (SCR / "s2_ref.npz"))
    entries = FL.jload(SCR / "plan_raw.json")["entries"]
    roll = FL.jload(FL._res_path(a.tag))
    rep = analyze_core(ref, entries, roll["windows"], SD, expect_e1=None if a.no_expect_e1 else E1_LITERAL)
    rep.update({"generated": time.strftime("%FT%T%z"), "tag": a.tag, "ckpt": FL.CKPT, "device": "CPU (fp32 trunk), per-window sampler seed"})
    name = "FORCE_S2" if a.tag == "full" else f"FORCE_S2_{a.tag}"
    OUT.mkdir(parents=True, exist_ok=True)
    FL.jdump(clean(rep), OUT / f"{name}.json")
    write_md(rep, OUT / f"{name}.md")
    print(json.dumps(clean(rep["headline"]), indent=1))
    print("ZZFS2ANALYZEDONEZZ")
    return 0


# ====================================================================================================== CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="stage", required=True)
    s = sp.add_parser("select")
    s.add_argument("--ref-npz", default="", help="skip the panel load and plan from this saved ref (resume / test aid)")
    s.add_argument("--max-per-set", type=int, default=MAX_PER_SET)
    s.add_argument("--kb-n", type=int, default=KB_N)
    s.add_argument("--chunks-per-set", type=int, default=CHUNKS_PER_SET)
    s.add_argument("--no-expect-counts", action="store_true", help="TEST ONLY: skip the 434 / 424 set-size control")
    r = sp.add_parser("roll")
    r.add_argument("--tag", default="full")
    r.add_argument("--limit-windows", type=int, default=0)
    r.add_argument("--save-every", type=int, default=2)
    r.add_argument("--chunk", type=int, default=-1)
    r.add_argument("--driver", type=int, default=0)
    t = sp.add_parser("status")
    t.add_argument("--tag", default="full")
    z = sp.add_parser("analyze")
    z.add_argument("--tag", default="full")
    z.add_argument("--ref-npz", default="")
    z.add_argument("--no-expect-e1", action="store_true", help="TEST ONLY: skip control R (the brief's E1 literal)")
    a = ap.parse_args(argv)
    return {"select": stage_select, "roll": stage_roll, "status": stage_status, "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
