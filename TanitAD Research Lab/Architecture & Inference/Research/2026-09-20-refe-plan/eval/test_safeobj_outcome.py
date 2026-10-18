#!/usr/bin/env python3
"""Self-test of eval/safeobj_outcome.py: bars G, L, E and the outcome table O1-O7 of PREREG_SAFEOBJ.md WITH ADDENDUM 1 (PREREG_SAFEOBJ.addendum1.md, R1-R4) applied mechanically.
Run: python eval/test_safeobj_outcome.py -> ZZTEST_PASS <n>.   No pod, no GPU, no network.

WHICH FIXTURE COMES FROM WHICH PRODUCER (the "built, tested, unreachable" rule: a reader tested on a schema its author invented passes its test and reads nothing)
  REAL PRODUCERS
    * the per-check record / kill_checks.jsonl / KILLED.json (K2 K3 K4 K6) / kill_replay.json = `rls1_checks.RlValChecker.__call__`, `monitor`, `replay` (as eval/test_safeobj_read.py does it)
    * rlval_vs_A0.json (A0 / CTRL / R)  and  rlval_B_vs_ownA.json (A / B, base A, 10,000 resamples)  = `rl_val_eval.run`, hashing the arm's real scores/ files
    * safeobj_S.json = `safeobj_read.read_arm` + `safeobj_read.write_record`
    * the dual's trajectory in sft.jsonl: the lambda / pos_updates / cap_hits / lam_seen_max counters, the `data` event's epo_con block and the `done` event's state are the REAL `scorer_finetune.EpoDual`
      (`step`, `built_facts`, `state_dict`).  The train-row assembly around it is the trainer's own loop text, COPIED here and pinned by a source check (`pin_trainer_row_writer_*`): the loop is inside
      `main()` and cannot be called alone.
    * ONE real `main()` run (test_rl_epo.smoke, CPU stubs for the GPU parts only) writes a real sft.jsonl; `read_log` must read its data / train / done rows (`real_main_*`).
  SYNTHETIC (no producer exists, said here and at each use)
    * navtest_E.json (IMPL step 9, the navtest families adapter, is not built): every E scenario.
    * exact-boundary numbers (dose.json's alpha 0.25 / 0.375 / 0.5 points, a PDMS point exactly on G(t), dyadic L ties, a window violation of exactly 0.001): no run emits an exact value, so the
      producer's file is rewritten with the number (both files that carry it, kept in agreement) and safeobj_S.json is re-written by the real reader.
    * KILLED.json code K-MON (the chain's monitor-failure kill is a shell-level event, no python producer): both the monitor-died record and a K-MON record that CARRIES the real monitor's K3 reason
      (the second is the real K3 KILLED.json with its code rewritten).
    * the L controls' dyadic deltas (rlval_vs_A0.json subset).

  R0e-pick B3 (the constraint's registered T is PER ARM, never read from the log) -- fixtures:
    REAL: scorer_finetune.main() run twice on CPU (`--epo-con-temp` unnamed -> T 0.1, and `--epo-con-temp 0.01`); the REAL EpoDual built at 0.01 behind full arm directories (o1_t001, r1d_o1_t001);
          the REAL registered argv of R0d / R1o in eval/rls1d_arms.json; the REAL banked R0d trainer log, read against the banked pre-B3 outcome record.
    SYNTHETIC: corrupted `temp` values (no trainer writes a bool or a string), the arms files other than rls1d_arms.json.

Then: SCENARIOS per outcome O1-O7 and one per PRECEDENCE rule of addendum R2 (O6 beats O1, O6 beats O7, g3-alone is O7 not O4, O7 beats a g5 breach, the overlap is O4+O5, the gap is O4 flagged SLACK, a K-MON
kill with the K3 reason is K3), BOUNDARIES, the VERTEX test of addendum R1 (alpha 0.25 reads G FAIL by 0.00125; alpha 0.375 and 0.5 sit on F and read PASS), bars L and E, the joint rule, the CLI end to end,
the REFUSAL of an addendum that is not the registered bytes, the PINS (every committed threshold against the prereg TEXT, the ADDENDUM text and their sha256, and against the source constants), and MUTATIONS
of the tool's source, each of which must turn the checks it exists for RED (precedence reordered, positive part for the signed mean, window moved, printed vertices used, addendum sha mismatch not refused, ...).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                    # eval/
PKGROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PKGROOT / "refe"))
import test_safeobj_read as TR  # noqa: E402  (its fixture machinery: Fx, the producers' records)
import safeobj_read as SR  # noqa: E402
import rl_val_eval as RV  # noqa: E402
import rls1_checks as RC  # noqa: E402

TOOL_PATH = HERE / "safeobj_outcome.py"
PREREG = PKGROOT / "raw" / "2026-10-17-safety-objective" / "PREREG_SAFEOBJ.md"
ADD = PKGROOT / "raw" / "2026-10-17-safety-objective" / "PREREG_SAFEOBJ.addendum1.md"
DOSE = PKGROOT / "raw" / "2026-10-14-rls1b-attribution" / "dose.json"
SF_PATH = PKGROOT / "refe" / "scorer_finetune.py"
BANKED_R0D_LOG = PKGROOT / "raw" / "2026-10-18-r0d-outcome" / "raw" / "sft.jsonl"      # the REAL R0d trainer log (a pre-PICKCON trainer: its epo_con block has no `temp` key)
EPO_TEMP = 0.1                                                                          # --epo-temp's default = the registered constraint T (PREREG section 2); pinned against the prereg text, the trainer and the tool below
N = TR.N
TAGS = TR.TAGS
TAG_UPD = {"step0": 0, "update250": 250, "update500": 500, "update750": 750, "update1000": 1000}


# ============================================================ fixtures: B's picks per variant (A picks P1 everywhere); every P0 / P3 / P4 set sits in its OWN log (even indices, or 101..)
def _mk(p0=(), p3=(), p4=()):
    p = [1] * N
    for i in p0:
        p[i] = 0
    for i in p3:
        p[i] = 3
    for i in p4:
        p[i] = 4
    return p


def _ev(a, n):
    return [2 * i for i in range(a, a + n)]


G_VARIANTS = {                                                                                  # PDMS gain per P0 set = +0.0833 pts; a P3 set = TTC -0.4, PDMS -0.2167
    "gain_hi": _mk(p0=_ev(0, 20)),                                                              # PDMS +1.667, TTC 0
    "gain_lo": _mk(p0=_ev(0, 6)),                                                               # PDMS +0.500, TTC 0, lower bound > 0
    "flat": _mk(p0=_ev(0, 1)),                                                                  # PDMS +0.083, lower bound exactly 0
    "gain_hi_ttc": _mk(p0=_ev(0, 20), p3=[101]),                                                # PDMS +1.45, TTC -0.4  (bar there: 0.7945)
    "gain_lo_ttc": _mk(p0=_ev(0, 12), p3=[101]),                                                # PDMS +0.783, TTC -0.4, lower bound > 0, below the bar
    "pdmsdrop": _mk(p4=_ev(0, 9)),                                                              # PDMS -0.6 (K4), TTC / NC / g3 clean
    "pdmsdrop_ttc": _mk(p4=_ev(0, 9), p3=[101, 103]),                                           # PDMS drop (K4 breach) AND TTC -0.8 (a g5 breach)
    "g3g5": _mk(p4=list(range(0, 140)), p3=[141, 143]),                                         # shortest-bin share 0.56 AND TTC -0.8
    "k1": _mk(p3=[101]), "k2": _mk(p3=[101, 103]), "k3": _mk(p3=[101, 103, 105]), "k4": _mk(p3=[101, 103, 105, 107]),          # TTC -0.4 k  (the L controls)
}
TR.VARIANTS.update(G_VARIANTS)


class World:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.fx = TR.Fx(tmp / "fx")
        self._ba = {}
        self.n_arm = 0
        import scorer_finetune as sf          # the REAL trainer module (its EpoDual is the dual's producer)
        self.sf = sf

    def ba(self, variant, n_boot=10000):
        """rl_val_eval.run for arms A (P1 everywhere) and B (variant), base A: the paired B - own-A read with its interval (cached; ~13 s at the registered 10,000 resamples)."""
        key = (variant, n_boot)
        if key not in self._ba:
            a = self.fx.root / "ref" / "scores_update250_A.npz"
            arms = {"A": self.fx._load(a), "B": self.fx._load(self.fx.bfile[variant])}
            self._ba[key] = RV.run(arms, self.fx.pdm, self.fx.terms, self.fx.paths, "navsim_v1_lane", self.fx.cfg, TR.SCENES, True, "A", n_boot)
        return copy.deepcopy(self._ba[key])


# ============================================================ the trainer's log: the REAL dual, the trainer's own loop text
def sim_log(W, total, g_ttc, g_nc=lambda u: 0.0, *, accum=2, ramp=200, last=None, nan_at=None, done=True, ramp_in_data=None, con_temp=None):
    """-> {"data", "start", "trains": [row...], "done": row or None}.  The dual is scorer_finetune.EpoDual; the accumulation / row lines are the trainer's (scorer_finetune.main, the loop)."""
    dual = W.sf.EpoDual(("ttc", "nc"), 0.0, 4.0, 10.0, 0.0, temp=EPO_TEMP if con_temp is None else con_temp)          # B3: con_temp = the arm's --epo-con-temp (None = inherits --epo-temp, the registered 0.1).  PICKCON: scorer_finetune.main builds the dual with the constraint's T (--epo-con-temp, else --epo-temp); its built_facts() block carries `temp`
    con_acc, con_log, trains = {"n": 0}, {"n": 0}, []
    upd, end = 0, (total if last is None else last)
    while upd < end:
        for _ in range(accum):
            gt, gn = g_ttc(upd), g_nc(upd)
            parts = {"con": dual.lam["ttc"] * gt + dual.lam["nc"] * gn, "g_ttc": gt, "g_nc": gn, "cA_ttc": 0.9, "cB_ttc": 0.9 - gt, "cA_nc": 0.95, "cB_nc": 0.95 - gn,
                     "p1gap_ttc": 0.0, "p1gap_nc": 0.0, "hinge_ttc": 0.1, "hinge_nc": 0.1}
            for k_, v_ in parts.items():                                  # --- scorer_finetune.main: `for k_, v_ in con_last.items(): con_acc[k_] = con_acc.get(k_, 0.0) + v_`
                con_acc[k_] = con_acc.get(k_, 0.0) + v_
            con_acc["n"] += 1
        dual.step(con_acc)                                                # --- `dual.step(con_acc)`
        for k_, v_ in con_acc.items():                                    # --- `con_log[k_] = con_log.get(k_, 0.0) + v_`
            con_log[k_] = con_log.get(k_, 0.0) + v_
        con_acc = {"n": 0}
        upd += 1
        if upd % 25 == 0:
            row = {"event": "train", "update": upd, "of": total, "lr": 1e-4, "bceA": float("nan") if nan_at == upd else 0.31, "lossB": 0.42, "listnet": 0.2, "s_per_update": 10.0,
                   "epo": 0.1, "epo_J": 0.2, "epo_kl": 0.01, "epo_lam_pg": min(1.0, upd / ramp)}
            nc_ = max(con_log["n"], 1)
            cm_ = lambda k_: con_log.get(k_, 0.0) / nc_                    # noqa: E731
            row.update({"con": cm_("con"), "con_lam_ttc": dual.lam["ttc"], "con_lam_nc": dual.lam["nc"], "con_g_ttc": cm_("g_ttc"), "con_g_nc": cm_("g_nc"),
                        "con_cA_ttc": cm_("cA_ttc"), "con_cB_ttc": cm_("cB_ttc"), "con_cA_nc": cm_("cA_nc"), "con_cB_nc": cm_("cB_nc"),
                        "con_p1_gap_ttc": cm_("p1gap_ttc"), "con_p1_gap_nc": cm_("p1gap_nc"), "con_hinge_ttc": cm_("hinge_ttc"), "con_hinge_nc": cm_("hinge_nc"),
                        "con_cap_hits": dual.cap_hits, "con_lam_pos_updates": dual.pos_updates,
                        "con_lam_max_ttc": dual.lam_seen_max["ttc"], "con_lam_max_nc": dual.lam_seen_max["nc"]})
            con_log = {"n": 0}
            trains.append(row)
    return {"data": {"event": "data", "train_sets": 4000, "epo": {"temp": EPO_TEMP, "ramp": ramp if ramp_in_data is None else ramp_in_data, "lam": 1.0}, "epo_con": dual.built_facts()},
            "start": {"event": "start", "updates": total, "micro_per_update": accum, "batch": 8, "train_sets": 4000},
            "trains": trains, "done": ({"event": "done", "updates": upd, "hours": 3.4, "epo_con": dual.state_dict()} if done else None), "dual": dual}


# the profiles (g per update; lambda moves by 4 g per update, clamps at [0, 10])
HELD = (lambda u: 0.002 if u < 50 else 0.0, lambda u: 0.0)                     # lambda -> 0.4 and stays: lambda > 0 on every post-ramp update, violation 0
CAP_FAST = (lambda u: 0.05, lambda u: 0.0)                                     # lambda hits 10 at update 50, violation 0.05
CAP_THEN_HELD = (lambda u: 0.05 if u < 60 else 0.0, lambda u: 0.0)             # hits lambda_max, then holds: violation 0, share 1
LAG = (lambda u: 0.003, lambda u: 0.0)                                         # violation 0.003, lambda only ~6 at update 525
INERT = (lambda u: -0.01, lambda u: -0.01)                                     # lambda == 0 at every update
GAP = (lambda u: 0.01 if u < 20 else -0.01, lambda u: 0.0)                     # lambda back to 0 by update 40: share 0, no cap, violation < 0
LATE_CAP = (lambda u: 0.002 if u < 50 else (0.0 if u < 600 else 0.05), lambda u: 0.0)          # window before u500 holds; lambda hits lambda_max only after u600
TERMS_SPLIT = (lambda u: 0.0015 if u < 600 else 0.0, lambda u: -0.001)         # window: max of the terms 0.0015 > 0.001, sum 0.0005 <= 0.001
NC_VIOL = (lambda u: 0.0, lambda u: 0.003)                                     # the NC term alone violates (0.003): 'only TTC is judged' must not pass this
WIN_EDGE = (lambda u: 0.002 if u < 50 else (0.0044 if 400 <= u < 425 else 0.0), lambda u: 0.0)          # window before u500 = rows 425 / 450 / 475 / 500: only row 425 violates; its mean over the four is 0.0011
LAST_STEP_NC = (lambda u: 0.0, lambda u: 0.01 if u == 999 else 0.0)
LAST_STEP = (lambda u: 0.01 if u == 999 else 0.0, lambda u: 0.0)               # lambda becomes > 0 only AFTER the last update: pos_updates 0 but lam_seen_max > 0
WIN_NEWEST = (lambda u: 0.002 if u < 50 else (0.0044 if 475 <= u < 500 else 0.0), lambda u: 0.0)          # window before u500 = rows 425 / 450 / 475 / 500: only row 500 (the NEWEST) violates; mean 0.0011
OSC = (lambda u: 0.002 if u < 50 else (0.01 if (400 <= u < 425 or 450 <= u < 475) else (-0.01 if (425 <= u < 450 or 475 <= u < 500) else 0.0)), lambda u: 0.0)    # the window's four rows read +0.01 -0.01 +0.01 -0.01:
#   the SIGNED mean is exactly 0 (held); the mean of the positive parts is 0.005 (not held).  lambda swings between 0.4 and 1.4 and never reaches 0 or lambda_max
K3_LATE = (lambda u: 0.002 if u < 50 else (0.01 if 650 <= u < 750 else 0.0), lambda u: 0.0)                # violation only in the 100 updates before u750 (the FIRST of the K3 pair u750 / u1000), none before u250


def clamp_to_zero_at(u_star):
    """lambda = 1.0 from update 5 and held (g 0) until a huge negative g clamps it to 0.0 at update u_star, then g -0.01 keeps it there: lambda > 0 on steps 1..u_star, so with ramp 200 and U 1000
    the post-ramp count is u_star - 199 of 800."""
    return (lambda u: 0.05 if u < 5 else (-10.0 if u == u_star else (0.0 if u < u_star else -0.01))), (lambda u: 0.0)


def write_sft(d: Path, log, dumps, *, cut=None):
    """sft.jsonl as the trainer writes it: data, start, the step0 dump, then per tag the train rows up to its update and its dump; the remaining rows; the `done` event.  dumps = [score_dump rows]."""
    rows = [log["data"], log["start"]]
    by_tag = {r["tag"]: r for r in dumps}
    pending = list(log["trains"])
    if "step0" in by_tag:
        rows.append(by_tag["step0"])
    for t in TAGS:
        if t in by_tag:
            while pending and pending[0]["update"] <= TAG_UPD[t]:
                rows.append(pending.pop(0))
            rows.append(by_tag[t])
    rows += pending
    if log["done"] is not None:
        rows.append(log["done"])
    (d / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jsave(p, o):
    Path(p).write_text(json.dumps(o, indent=1), encoding="utf-8")


def rewrite_rows(d: Path, edit):
    """edit(row) -> row (SYNTHETIC: used for exact-boundary values no run emits)."""
    rows = [json.loads(ln) for ln in (d / "sft.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    (d / "sft.jsonl").write_text("".join(json.dumps(edit(r)) + "\n" for r in rows), encoding="utf-8")


def write_S(d: Path):
    """the REAL reader writes safeobj_S.json."""
    SR.write_record(str(d), SR.read_arm(str(d)))


def write_ba(W, d: Path, variant, n_boot=10000):
    """rl_val_eval's own B - own-A read of the arm's REAL scores/ files (+ the `inputs.scores_sha256` block rl_val_eval.main adds)."""
    res = W.ba(variant, n_boot)
    sc = d / "scores"
    res["inputs"] = {"scores_sha256": {"A": RV.sha256_file(str(sc / "scores_update1000_A.npz")), "B": RV.sha256_file(str(sc / "scores_update1000_B.npz"))}}
    with open(d / "rlval_B_vs_ownA.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)


def build_arm2(W, root: Path, name: str, arm: str, variants: dict, log, *, kill=None, live=True, replay=True, ba=True, ba_variant=None, S=True, rlval_variant="last"):
    """an arm directory the way the chain leaves it, with the trainer's log merged in BEFORE the real monitor / replay read it.  kill in (None, "K6")."""
    fx = W.fx
    d = TR.build_arm(fx, root, name, variants, arm=arm, live=False, replay=False, rlval_variant=rlval_variant)
    dumps = [json.loads(ln) for ln in (d / "sft.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    write_sft(d, log, dumps)
    vt = {"step0": "same", **{t: v for t, v in variants.items() if v}}

    def check_fn(tag, files):
        return {**fx.rec(vt[tag]), "tag": tag}
    if live:
        ticks = iter([0.0] + [7 * 3600.0] * 100)
        kw = {"now_fn": lambda: next(ticks)} if kill == "K6" else {"now_fn": lambda: 0.0}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            RC.monitor(arm, str(d), check_fn, 1, lambda pid: False, lambda pid: None, sleep_fn=lambda s: None, **kw)
    if replay:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code, why, recs = RC.replay(arm, str(d), check_fn)
        with open(d / "kill_replay.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump({"arm": arm, "kill": code, "why": why, "checks": recs}, f, indent=1)
    if ba and variants.get("update1000"):
        write_ba(W, d, ba_variant or variants["update1000"])
    if S:
        write_S(d)
    return d


def ctrl_dir(W, root: Path, name: str, variant: str):
    """an arm directory holding only the chain's u1000 read (bar L's input): the REAL rl_val_eval.run for A0 / CTRL / R, with the REAL scores/ files hashed."""
    d = root / name
    sc = d / "scores"
    sc.mkdir(parents=True)
    fx = W.fx
    RV.write_scores_npz(str(sc / "scores_update1000_A.npz"), TR.KEYS, [TR.CK] * N, TR._scores(TR.A_PICK))
    shutil.copy(fx.bfile[variant], sc / "scores_update1000_B.npz")
    RV.write_scores_npz(str(sc / "scores_update1000_base.npz"), TR.KEYS, [TR.CK] * N, TR._scores(TR.A_PICK))
    res = fx.rlval(variant)
    res["inputs"] = {"scores_sha256": {"A0": RV.sha256_file(str(sc / "scores_update1000_base.npz")), "CTRL": RV.sha256_file(str(sc / "scores_update1000_A.npz")), "R": RV.sha256_file(str(sc / "scores_update1000_B.npz"))}}
    jsave(d / "rlval_vs_A0.json", res)
    return d


def synth_ctrl_dir(root: Path, name: str, ttc_delta: float, pdms_delta: float = 0.0):
    """SYNTHETIC: a dyadic u1000 B - A so that L's ties are exact in binary (no run emits one)."""
    d = root / name
    d.mkdir(parents=True)
    jsave(d / "rlval_vs_A0.json", {"arms": {"CTRL": {"subscores_x100": {"TTC": 100.0}, "PDMS_x100": 50.0}, "R": {"subscores_x100": {"TTC": 100.0 + ttc_delta}, "PDMS_x100": 50.0 + pdms_delta}}})
    return d


def override_u1000(d: Path, pdms_mean, pdms_lo, ttc_mean, pdms_hi=None):
    """SYNTHETIC: put an exact B - own-A point into BOTH files that carry it (kept in agreement), then let the real reader re-write safeobj_S.json."""
    r = jload(d / "rlval_vs_A0.json")
    r["arms"]["R"]["PDMS_x100"] = r["arms"]["CTRL"]["PDMS_x100"] + pdms_mean
    r["arms"]["R"]["subscores_x100"]["TTC"] = r["arms"]["CTRL"]["subscores_x100"]["TTC"] + ttc_mean
    jsave(d / "rlval_vs_A0.json", r)
    b = jload(d / "rlval_B_vs_ownA.json")
    q = b["paired_vs_base"]["B"]["quantities"]
    q["PDMS"]["mean"], q["PDMS"]["ci95"] = pdms_mean, [pdms_lo, pdms_hi if pdms_hi is not None else pdms_mean + 1.0]
    q["TTC"]["mean"] = ttc_mean
    jsave(d / "rlval_B_vs_ownA.json", b)
    write_S(d)


def copy_arm(src: Path, dst_root: Path, name: str):
    dst = dst_root / name / src.name
    shutil.copytree(src, dst)
    return dst


# ============================================================ the scenarios
ALL4 = lambda v: dict(zip(TAGS, (v,) * 4))                           # noqa: E731


def with_(base, **kw):
    return {**base, **kw}


def make_world_arms(W: World, root: Path):
    """-> {name: dir} (built once; the tool never writes into these; scenarios that need a changed directory copy it)."""
    A = {}
    B = lambda name, arm, variants, log, **kw: build_arm2(W, root / "arms", name, arm, variants, log, **kw)               # noqa: E731
    L1000 = lambda prof, **kw: sim_log(W, 1000, *prof, **kw)                                                              # noqa: E731
    L_K = lambda prof, last=525, **kw: sim_log(W, 1000, *prof, last=last, done=False, **kw)                               # noqa: E731
    K3V = {"update250": "ttc", "update500": "ttc"}                                                                       # two consecutive g5 breaches: the real monitor K3-kills at update500
    # ---- completed arms, bar S PASS
    A["o1"] = B("o1", "R0d", ALL4("gain_hi"), L1000(HELD))
    A["o1_ttc"] = B("o1_ttc", "R0d", ALL4("gain_hi_ttc"), L1000(HELD))
    A["o2"] = B("o2", "R0d", ALL4("gain_lo"), L1000(HELD))
    A["o2_ttc"] = B("o2_ttc", "R0d", ALL4("gain_lo_ttc"), L1000(HELD))
    A["o3_flat"] = B("o3_flat", "R0d", ALL4("flat"), L1000(HELD))
    A["o3_same"] = B("o3_same", "R0d", ALL4("same"), L1000(HELD))                                                        # the IMPL_PLAN selftest: an arm equal to its A
    A["o3_o6_tie"] = B("o3_o6_tie", "R0d", ALL4("same"), L1000(INERT))                                                   # S holds AND lambda == 0 at every update: O3 and O6 both match
    A["o1_inert"] = B("o1_inert", "R0d", ALL4("gain_hi"), L1000(INERT))                                                  # O1 and O6
    # ---- completed arms, a single non-consecutive breach (S FAIL, no kill) -> the log decides O4 / O5 / ...
    BR500 = with_(ALL4("gain_hi"), update500="ttc")
    A["o4"] = B("o4", "R0d", BR500, L1000(HELD))
    A["o4_share_exact"] = B("o4_share_exact", "R0d", BR500, L1000(clamp_to_zero_at(349)))                                # post-ramp share THROUGH u500 EXACTLY 0.5 (150 / 300; the whole run would read 150 / 800)
    A["gap_share_below"] = B("gap_share_below", "R0d", BR500, L1000(clamp_to_zero_at(348)))                              # 149 / 300: just below -> O4 flagged SLACK
    A["viol_exact"] = B("viol_exact", "R0d", BR500, L1000(HELD))                                                          # edited below to a window violation of exactly 0.001
    A["viol_over"] = B("viol_over", "R0d", BR500, L1000(HELD))                                                            # edited below to 0.0010000001
    A["o5_viol_cap"] = B("o5_viol_cap", "R0d", BR500, L1000(CAP_FAST))                                                    # lambda reaches lambda_max AND violation 0.05
    A["o5_cap_o4_tie"] = B("o5_cap_o4_tie", "R0d", BR500, L1000(CAP_THEN_HELD))                                          # cap reached, then held: O4 and O5 both match
    A["cap_after_breach"] = B("cap_after_breach", "R0d", BR500, L1000(LATE_CAP))                                         # lambda reaches lambda_max only AFTER the first breached check: not O5 (R3: through the first breach)
    A["terms_max"] = B("terms_max", "R0d", BR500, L1000(TERMS_SPLIT))                                                     # the window means of the two terms are 0.0015 and -0.001: held only if the SUM were judged
    A["share_through_breach"] = B("share_through_breach", "R0d", BR500, L1000(clamp_to_zero_at(450)))                    # whole run 251 / 800, THROUGH u500 251 / 300: O4 (R3: counted through the first breach)
    A["osc_signed"] = B("osc_signed", "R0d", BR500, L1000(OSC))                                                           # signed window mean 0 (held); the positive parts would read 0.005 (not held)
    A["win_newest"] = B("win_newest", "R0d", BR500, L1000(WIN_NEWEST))                                                    # violation in the NEWEST row of the window only
    A["o5_nc_viol"] = B("o5_nc_viol", "R0d", BR500, L1000(NC_VIOL))                                                       # violation through the NC term only
    A["win_edge"] = B("win_edge", "R0d", BR500, L1000(WIN_EDGE))                                                           # violation in the OLDEST row of the 100-update window only
    A["o3_last_step"] = B("o3_last_step", "R0d", ALL4("flat"), L1000(LAST_STEP))                                          # lambda == 0 at every update the dual USED, > 0 after the last step: not O6
    A["o3_last_step_nc"] = B("o3_last_step_nc", "R0d", ALL4("flat"), L1000(LAST_STEP_NC))                                  # the same through the NC term only
    A["gap"] = B("gap", "R0d", BR500, L1000(GAP))                                                                         # breached, violation < 0.001, share 0, no cap: no row matches
    A["o6_breach"] = B("o6_breach", "R0d", BR500, L1000(INERT))                                                           # breached and lambda == 0 everywhere: O6 only
    A["o7_g3"] = B("o7_g3", "R0d", with_(ALL4("gain_hi"), update500="share"), L1000(GAP))                                 # a g3-only breach, O4 / O5 do not match: O7 only
    A["o7_g3_o4"] = B("o7_g3_o4", "R0d", with_(ALL4("gain_hi"), update500="share"), L1000(HELD))                          # a g3-only breach AND O4: tie
    A["o7_g3g5"] = B("o7_g3g5", "R0d", with_(ALL4("gain_hi"), update500="g3g5"), L1000(HELD))                             # g3 AND g5 at the same check: not 'g3 alone' -> O4
    A["o7_beats_o4"] = B("o7_beats_o4", "R0d", {"update250": "pdmsdrop_ttc", "update500": "pdmsdrop"}, L1000(HELD), live=False, ba=False)     # a g5 breach at u250 AND a replay K4 kill at u500: O7 beats O4
    A["o4_u1000"] = B("o4_u1000", "R0d", with_(ALL4("gain_hi"), update1000="ttc"), L1000(HELD), live=False)                # the ONLY breach is the final check, update 1000 (the replay row locates it)
    # ---- killed arms (the real monitor kills; the log ends where the trainer was when it did)
    A["o4_k3"] = B("o4_k3", "R0d", K3V, L_K(HELD), replay=False, ba=False)
    A["o5_k3"] = B("o5_k3", "R0d", K3V, L_K(CAP_FAST), replay=False, ba=False)
    A["o5_lag_k3"] = B("o5_lag_k3", "R0d", K3V, L_K(LAG), replay=False, ba=False)
    A["o6_k3"] = B("o6_k3", "R0d", K3V, L_K(INERT), replay=False, ba=False)
    A["gap_k3"] = B("gap_k3", "R0d", K3V, L_K(GAP), replay=False, ba=False)
    A["o7_k6"] = B("o7_k6", "R0d", {"update250": "gain_hi"}, L_K(HELD, last=275), kill="K6", replay=False, ba=False)
    A["o6_beats_o7"] = B("o6_beats_o7", "R0d", {"update250": "gain_hi"}, L_K(INERT, last=275), kill="K6", replay=False, ba=False)           # a K6 kill of an arm whose lambda never moved: O6 beats O7
    A["k3_late_pair"] = B("k3_late_pair", "R0d", {"update250": "ttc", "update500": "same", "update750": "ttc", "update1000": "ttc"}, L1000(K3_LATE), replay=False, ba=False)      # an isolated breach at u250, the K3 pair u750 / u1000
    A["iso_two"] = B("iso_two", "R0d", {"update250": "ttc", "update500": "same", "update750": "ttc", "update1000": "same"}, L1000(K3_LATE), ba=False)      # two ISOLATED breaches (u250, u750), no K3 pair, no kill: the window ends at the FIRST (u250)
    A["o7_k4"] = B("o7_k4", "R0d", {"update250": "pdmsdrop", "update500": "pdmsdrop"}, L_K(HELD), replay=False, ba=False)
    A["o7_k2"] = B("o7_k2", "R0d", {"update250": "gain_hi"}, L_K(HELD, last=325, nan_at=300), replay=False, ba=False)
    A["o7_k4_retro"] = B("o7_k4_retro", "R0d", {"update250": "pdmsdrop", "update500": "pdmsdrop"}, L1000(HELD), live=False, ba=False)                 # a COMPLETED run whose post-hoc REPLAY reads a K4 kill (no live KILLED.json)
    A["n_k4_replay_unfinished"] = B("n_k4_replay_unfinished", "R0d", {"update250": "pdmsdrop", "update500": "pdmsdrop"}, L_K(HELD), live=False, ba=False)   # the same replay kill, but the log has no `done` event and no KILLED.json
    d = B("kmon_monitor_died", "R0d", {"update250": "gain_hi"}, L_K(HELD, last=275), replay=False, ba=False)
    jsave(d / "KILLED.json", {"arm": "R0d", "code": "K-MON", "why": "kill rules unenforceable: monitor died rc 137", "at": "t"})          # SYNTHETIC: a shell-level event (rls1_pod.sh supervise_arm), no python producer
    write_S(d)
    A["kmon_monitor_died"] = d
    # ---- second constrained arm (seed 1), for the joint rule
    A["r1d_o1"] = B("r1d_o1", "R1d", ALL4("gain_hi"), L1000(HELD))
    A["r1d_o2"] = B("r1d_o2", "R1d", ALL4("gain_lo"), L1000(HELD))
    # ---- R0e-pick B3: the same arms as o1 / r1d_o1, but the REAL EpoDual was built at --epo-con-temp 0.01 (the data event's epo_con.temp = 0.01; the EPO reward term's `epo.temp` stays 0.1)
    A["o1_t001"] = B("o1_t001", "R0d", ALL4("gain_hi"), L1000(HELD, con_temp=0.01))
    A["r1d_o1_t001"] = B("r1d_o1_t001", "R1d", ALL4("gain_hi"), L1000(HELD, con_temp=0.01))
    # ---- exact-boundary edits of the window violation (SYNTHETIC rows: the sum of 25-update means cannot be made exact by a run)
    for nm, val in (("viol_exact", 0.001), ("viol_over", 0.0010000001)):
        rewrite_rows(A[nm], lambda r, val=val: ({**r, "con_g_ttc": val} if (r.get("event") == "train" and r["update"] in (425, 450, 475, 500)) else r))
    return A


# ============================================================ the checks
def dose_pts():
    full = jload(DOSE)["full"]
    return tuple((float(full[a]["TTC"]["mean"]), float(full[a]["PDMS"]["mean"])) for a in ("0.25", "0.375", "0.5"))


def chk_read(T, d, name="R0d", dose=True, registered_temp=None):
    """dose = True (the real, sha256-pinned dose.json), False / None (none given), or a path.  registered_temp = the arm's registered constraint T (None = the tool's default, the PREREG 0.1)."""
    ctx = T.Ctx()
    path = str(DOSE) if dose is True else (None if dose in (False, None) else str(dose))
    kw = {} if registered_temp is None else {"registered_temp": registered_temp, "temp_source": {"kind": "test"}}
    return T.read_constrained(name, str(d), ctx, T.load_dose(path, ctx), **kw)


def RUN(T, dirs, **kw):
    """T.run with the registered addendum and dose.json unless overridden."""
    kw.setdefault("dose_path", str(DOSE))
    kw.setdefault("addendum_path", str(ADD))
    return T.run({k: str(v) for k, v in dirs.items()}, **kw)


def run_suite(src, W, A, root, tag, *, e2e=False):
    """-> the names of the failing checks for the tool whose source is `src`."""
    bad = []
    tpath = root / f"tool_{tag}" / "safeobj_outcome.py"
    tpath.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "safeobj_read.py", tpath.parent / "safeobj_read.py")
    try:
        T = load_tool(src, tpath)
    except Exception as e:                                                                                                   # noqa: BLE001
        return [f"IMPORT:{type(e).__name__}:{str(e)[:60]}"]

    def t(name, fn):
        try:
            ok = bool(fn())
        except Exception:                                                                                                    # noqa: BLE001
            ok = False
        if not ok:
            bad.append(name)
    for nm, fn in CHECKS:
        t(nm, lambda fn=fn: fn(T, W, A, root, tag))
    if e2e:
        try:
            for nm, ok in cli_checks(tpath, A, root, tag):
                t(f"e2e:{nm}", lambda ok=ok: ok)
        except Exception as e:                                                                                               # noqa: BLE001
            bad.append(f"e2e:CRASH:{type(e).__name__}")
    return bad


def load_tool(src: str, where: Path):
    where.write_text(src, encoding="utf-8", newline="\n")
    m = types.ModuleType("safeobj_outcome_under_test")
    m.__file__ = str(where)
    exec(compile(src, str(where), "exec"), m.__dict__)
    return m


CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def outcome_is(key, want, S=None, G=None, shadowed=None, flags=None, extra=None):
    """register a scenario check: arm A[key] reads outcome `want` (and, when given, the S / G verdicts, the rows that matched but lost to the precedence, the outcome flags)."""
    def fn(T, W, A, root, tag):
        r = chk_read(T, A[key])
        assert r["outcome"] == want, (key, r["outcome"], r.get("table", {}).get("why"), r["problems"][:3])
        if S is not None:
            assert r["S"]["verdict"] == S, (key, r["S"]["verdict"])
        if G is not None:
            assert r["G"]["verdict"] == G, (key, r["G"])
        if shadowed is not None:
            assert r["table"]["shadowed"] == shadowed, (key, r["table"]["shadowed"])
        if flags is not None:
            assert r["outcome_flags"] == flags, (key, r["outcome_flags"])
        if extra:
            assert extra(r), (key, "the extra check failed")
        return True
    CHECKS.append((f"outcome:{key}->{want}", fn))


# ---- the seven outcomes, each from a real-producer arm
outcome_is("o1", "O1", S="PASS", G="PASS", flags=[])
outcome_is("o1_ttc", "O1", S="PASS", G="PASS", extra=lambda r: abs(r["G"]["bar_at_t"] - 0.79610) < 1e-5 and abs(r["G"]["ttc_point_delta_x100"]["value"] + 0.4) < 1e-9)       # F between the alpha 0.375 and 0.5 vertices
outcome_is("o2", "O2", S="PASS", G="FAIL", extra=lambda r: r["G"]["lower_bound_gt_0"] is True and r["G"]["point_ge_bar"] is False)
outcome_is("o2_ttc", "O2", S="PASS", G="FAIL", extra=lambda r: r["G"]["lower_bound_gt_0"] is True and r["G"]["point_ge_bar"] is False)
outcome_is("o3_flat", "O3", S="PASS", G="FAIL", extra=lambda r: r["G"]["lower_bound_gt_0"] is False)
outcome_is("o3_same", "O3", S="PASS", G="FAIL", extra=lambda r: r["G"]["pdms_ci95_lower"]["value"] == 0.0 and r["G"]["pdms_point_delta_x100"]["value"] == 0.0)          # an arm equal to its A
outcome_is("o4", "O4", S="FAIL", flags=[], extra=lambda r: r["facts"]["breached"] is True and r["first_breached_check"] == "update500")
outcome_is("o4_k3", "O4", S="FAIL", flags=[], extra=lambda r: r["state"] == "killed" and r["first_breached_check"] == "update250" and r["G"]["verdict"] == "NOT_READ" and r["kill_rules"] == ["K3"]
           and r["log"]["window_overlaps_ramp"] is True and any("overlaps the lambda_pg ramp" in n for n in r["log"]["notes"]))        # a K3 kill at u250: the window 151..250 overlaps the ramp: DISCLOSED, not moved
outcome_is("o5_k3", "O5", S="FAIL", extra=lambda r: r["log"]["cap_hit"] is True and r["log"]["violation_ok"] is False)
outcome_is("o5_viol_cap", "O5", S="FAIL")
outcome_is("o5_lag_k3", "O5", S="FAIL", extra=lambda r: r["log"]["cap_hit"] is False and r["log"]["violation_ok"] is False)                    # the 'lagging' clause alone
outcome_is("o6_k3", "O6", S="FAIL", extra=lambda r: r["log"]["lam_zero"] is True)
outcome_is("o6_breach", "O6", S="FAIL", shadowed=[])
outcome_is("o7_k6", "O7", extra=lambda r: r["S"]["verdict"] == "NOT_READ" and r["kill_codes"] == ["K6"] and r["kill_rules"] == ["K6"])
outcome_is("o7_k4", "O7", extra=lambda r: r["kill_codes"] == ["K4"])
outcome_is("o7_k2", "O7", extra=lambda r: r["kill_codes"] == ["K2"])
outcome_is("o7_k4_retro", "O7", extra=lambda r: r["kill_codes"] == ["K4"] and r["state"] == "complete")                                        # a kill that only the post-hoc replay recorded
outcome_is("o4_u1000", "O4", S="FAIL", extra=lambda r: r["first_breached_check"] == "update1000" and r["breach_kinds"] == ["g5"] and r["log"]["window_overlaps_ramp"] is False)             # window = the 4 rows ending at update 1000
outcome_is("o7_g3", "O7", S="FAIL", extra=lambda r: r["breach_kinds"] == ["g3"] and r["facts"]["g3_alone"] is True)
outcome_is("o7_g3g5", "O4", S="FAIL", extra=lambda r: r["breach_kinds"] == ["g3", "g5"] and r["facts"]["g3_alone"] is False)                       # g3 AND g5 is not 'g3 alone'
outcome_is("o5_nc_viol", "O5", S="FAIL", extra=lambda r: r["log"]["numbers"]["window_g_nc_mean"]["value"] > 0.001 and r["log"]["numbers"]["window_g_ttc_mean"]["value"] <= 0.001)
outcome_is("win_edge", "O5", S="FAIL", extra=lambda r: r["log"]["violation_ok"] is False and 0.001 < r["log"]["numbers"]["window_g_ttc_mean"]["value"] < 0.0012)         # the OLDEST row of the window only
outcome_is("win_newest", "O5", S="FAIL", extra=lambda r: r["log"]["violation_ok"] is False and 0.001 < r["log"]["numbers"]["window_g_ttc_mean"]["value"] < 0.0012)       # the NEWEST row of the window only
outcome_is("osc_signed", "O4", S="FAIL", flags=[], extra=lambda r: r["log"]["violation_ok"] is True and abs(r["log"]["numbers"]["window_g_ttc_mean"]["value"]) < 1e-12 and r["log"]["columns_read"] == {"ttc": "con_g_ttc", "nc": "con_g_nc"})       # the SIGNED mean (R3), the column named
outcome_is("o3_last_step_nc", "O3", S="PASS", extra=lambda r: r["log"]["lam_zero"] is False and r["log"]["numbers"]["lam_pos_updates_last"]["value"] == 0 and r["log"]["numbers"]["lam_seen_max_nc_last"]["value"] > 0 and r["log"]["numbers"]["lam_seen_max_ttc_last"]["value"] == 0)
outcome_is("o3_last_step", "O3", S="PASS", extra=lambda r: r["log"]["lam_zero"] is False and r["log"]["numbers"]["lam_pos_updates_last"]["value"] == 0 and r["log"]["numbers"]["lam_seen_max_ttc_last"]["value"] > 0)
# ---- addendum R2: the fixed precedence O6 > O7 > O4/O5 > O1/O2/O3 (one scenario per rule)
outcome_is("o3_o6_tie", "O6", S="PASS", shadowed=["O3"], flags=[])                                                    # O6 beats O3 (the old tie)
outcome_is("o1_inert", "O6", S="PASS", shadowed=["O1"], extra=lambda r: r["G"]["verdict"] == "PASS")               # O6 beats O1 (an arm that PASSES S and G but whose lambda never moved)
outcome_is("o6_beats_o7", "O6", shadowed=["O7"], extra=lambda r: r["kill_codes"] == ["K6"] and r["log"]["lam_zero"] is True)    # O6 beats O7 (a K6 kill)
outcome_is("o7_g3_o4", "O7", S="FAIL", shadowed=[], extra=lambda r: r["breach_kinds"] == ["g3"] and r["facts"]["g3_alone"] is True and r["facts"]["breached"] is False and r["table"]["rows"]["O4"] is False and r["table"]["rows"]["O5"] is False)   # 'g3 alone' is O7, NOT O4
outcome_is("o7_beats_o4", "O7", S="FAIL", shadowed=["O4"], extra=lambda r: r["kill_codes"] == ["K4"] and r["breach_kinds"] == ["g5"] and r["facts"]["breached"] is True)    # a g5 breach AND a K4 kill: O7 beats O4
outcome_is("o5_cap_o4_tie", "O4+O5", S="FAIL", shadowed=[], flags=[], extra=lambda r: r["table"]["rows"]["O4"] is True and r["table"]["rows"]["O5"] is True)      # the overlap: both rows
outcome_is("gap", "O4", S="FAIL", shadowed=[], flags=["SLACK"], extra=lambda r: any("SLACK" in n for n in r["notes"]) and r["table"]["rows"]["O5"] is False)         # the gap: O4 flagged SLACK
outcome_is("gap_k3", "O4", S="FAIL", flags=["SLACK"])
outcome_is("iso_two", "O4", S="FAIL", extra=lambda r: r["first_breached_check"] == "update250" and r["log"]["violation_ok"] is True and abs(r["log"]["numbers"]["window_g_ttc_mean"]["value"]) < 1e-12)     # R3: the window ends at the FIRST breached check; the later isolated breach is not it
outcome_is("kmon_monitor_died", "O7", extra=lambda r: r["kill_codes"] == ["K-MON"] and r["kill_rules"] == ["K-MON"])       # a K-MON kill whose reason is NOT the K3 rule is not K3: O7 (A-O9)
outcome_is("kmon_k3", "O4", flags=[], extra=lambda r: r["kill_codes"] == ["K-MON"] and r["kill_rules"] == ["K3"] and r["S"]["verdict"] == "NOT_READ" and r["first_breached_check"] == "update250"
           and r["facts"]["breached"] is True and r["facts"]["other_kill"] is False)                                     # a K-MON kill that recorded the K3 reason COUNTS as K3
# ---- addendum R3: the first breached check of a K3 kill is the FIRST of the two consecutive breaches
outcome_is("k3_late_pair", "O5", S="FAIL", extra=lambda r: r["first_breached_check"] == "update750" and r["first_check_with_any_breach"] == "update250" and r["log"]["violation_ok"] is False
           and r["log"]["numbers"]["window_end_update"]["value"] == 750 and r["log"]["window_overlaps_ramp"] is False)
# ---- boundaries of the O4 clauses: the post-ramp share of lambda > 0 THROUGH the first breached check (150 / 300 = 0.5 exactly, 149 / 300), the window violation (0.001 exactly, just over)
outcome_is("o4_share_exact", "O4", S="FAIL", flags=[], extra=lambda r: r["log"]["numbers"]["post_ramp_share"]["value"] == 0.5 and r["log"]["lam_pos_share_ok"] is True and r["log"]["numbers"]["post_ramp_updates"]["value"] == 300)
outcome_is("gap_share_below", "O4", S="FAIL", flags=["SLACK"], extra=lambda r: r["log"]["lam_pos_share_ok"] is False and r["log"]["numbers"]["post_ramp_updates_with_lambda_gt_0"]["value"] == 149)
outcome_is("viol_exact", "O4", S="FAIL", flags=[], extra=lambda r: r["log"]["violation_ok"] is True and r["log"]["numbers"]["window_g_ttc_mean"]["value"] == 0.001)
outcome_is("viol_over", "O5", S="FAIL", extra=lambda r: r["log"]["violation_ok"] is False)
# ---- addendum R3: everything is read THROUGH the first breached check, never over the whole run
outcome_is("cap_after_breach", "O4", S="FAIL", flags=[], extra=lambda r: r["log"]["cap_hit"] is False and r["log"]["numbers"]["lam_seen_max_ttc_last"]["value"] >= 10.0)          # lambda_max is reached at u600+, after the u500 breach
outcome_is("terms_max", "O5", S="FAIL", extra=lambda r: r["log"]["numbers"]["window_g_ttc_mean"]["value"] > 0.001 and r["log"]["numbers"]["window_g_nc_mean"]["value"] < 0 and r["log"]["violation_ok"] is False)     # BOTH terms must hold: the sum would
outcome_is("share_through_breach", "O4", S="FAIL", flags=[], extra=lambda r: r["log"]["numbers"]["post_ramp_updates_with_lambda_gt_0"]["value"] == 251 and r["log"]["numbers"]["post_ramp_updates"]["value"] == 300)


# ============================================================ the modified directories, the exact-boundary arms, the L controls, the E files
def _edit_json(p, fn):
    o = jload(p)
    fn(o)
    jsave(p, o)


def _drop_rows(d: Path, pred):
    rows = [json.loads(ln) for ln in (d / "sft.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    (d / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if not pred(r)), encoding="utf-8")


def make_mods(W: World, A: dict, root: Path):
    m = root / "mods"
    m.mkdir()
    base = A["o1"]

    def mod(name, fn, src=None):
        d = copy_arm(src or base, m, name)
        fn(d)
        A[name] = d
        return d
    # ---- bar S: the file the chain acted on must be the record the reader writes now
    mod("n_S_missing", lambda d: (d / "safeobj_S.json").unlink())
    mod("n_S_stale", lambda d: jsave(d / "rlval_vs_A0.json", jload(d / "rlval_vs_A0.json")))                                       # same numbers, different bytes: the reader's input sha256 moved
    mod("n_S_handedit", lambda d: jsave(d / "safeobj_S.json", {**jload(d / "safeobj_S.json"), "S": "FAIL"}))                       # edited, not re-hashed
    mod("n_S_notread", lambda d: ((d / "kill_replay.json").unlink(), write_S(d)))                                                  # the reader itself says NOT_READ
    # ---- addendum R2.2 / A-O9: a K-MON record that CARRIES the real monitor's K3 reason (the real K3 KILLED.json with its code rewritten; SYNTHETIC: no producer writes this)
    def kmon_k3(d):
        k = jload(d / "KILLED.json")
        assert k["code"] == "K3" and "g3/g5 bound breached at update250 and update500" in k["why"], k
        jsave(d / "KILLED.json", {**k, "code": "K-MON"})
        write_S(d)
    mod("kmon_k3", kmon_k3, src=A["o4_k3"])
    # ---- the trainer's log
    mod("n_log_cols", lambda d: rewrite_rows(d, lambda r: {k: v for k, v in r.items() if k != "con_lam_pos_updates"} if r.get("event") == "train" else r))
    mod("n_log_missing", lambda d: (d / "sft.jsonl").unlink())
    mod("n_log_lam0", lambda d: rewrite_rows(d, lambda r: {**r, "epo_con": {**r["epo_con"], "lam0": 0.5}} if r.get("event") == "data" else r))
    mod("n_log_eta", lambda d: rewrite_rows(d, lambda r: {**r, "epo_con": {**r["epo_con"], "lr": 2.0}} if r.get("event") == "data" else r))
    mod("n_log_ramp", lambda d: rewrite_rows(d, lambda r: {**r, "epo": {**r["epo"], "ramp": 100}} if r.get("event") == "data" else r))
    # PICKCON: the data event's epo_con block gained `temp` (the constraint's T).  legacy_block = the shape of every pre-PICKCON trainer (the banked R0d log): no `temp` key, still read.
    # n_log_temp = SO-PICKCON's own T (0.02) in a block that is otherwise the registered one: a second lever, refused.  n_log_temp_none = a dual built without a T (main always sets it).
    mod("legacy_block", lambda d: rewrite_rows(d, lambda r: {**r, "epo_con": {k: v for k, v in r["epo_con"].items() if k != "temp"}} if r.get("event") == "data" else r))
    mod("n_log_temp", lambda d: rewrite_rows(d, lambda r: {**r, "epo_con": {**r["epo_con"], "temp": 0.02}} if r.get("event") == "data" else r))
    mod("n_log_temp_none", lambda d: rewrite_rows(d, lambda r: {**r, "epo_con": {**r["epo_con"], "temp": None}} if r.get("event") == "data" else r))
    mod("n_log_gap", lambda d: _drop_rows(d, lambda r: r.get("event") == "train" and r["update"] == 100))
    mod("n_log_done", lambda d: rewrite_rows(d, lambda r: {**r, "epo_con": {**r["epo_con"], "pos_updates": r["epo_con"]["pos_updates"] + 1}} if r.get("event") == "done" else r))
    mod("n_log_cap_inconsistent", lambda d: rewrite_rows(d, lambda r: {**r, "con_cap_hits": 3} if (r.get("event") == "train" and r["update"] >= 500) else r))
    mod("n_log_decreasing", lambda d: rewrite_rows(d, lambda r: {**r, "con_lam_pos_updates": 0} if (r.get("event") == "train" and r["update"] == 600) else r))
    mod("n_log_half_line", lambda d: (d / "sft.jsonl").write_text((d / "sft.jsonl").read_text(encoding="utf-8").rstrip("\n") + '\n{"event": "train", "update": 1025, "con_l', encoding="utf-8"))          # a SIGKILLed trainer's half line: tolerated
    # ---- bar G: the B - own-A read
    mod("n_ba_missing", lambda d: (d / "rlval_B_vs_ownA.json").unlink())
    mod("n_ba_other_dumps", lambda d: _edit_json(d / "rlval_B_vs_ownA.json", lambda b: b["inputs"]["scores_sha256"].__setitem__("A", hashlib.sha256(b"C0's dumps").hexdigest())))
    mod("n_ba_other_B", lambda d: _edit_json(d / "rlval_B_vs_ownA.json", lambda b: b["inputs"]["scores_sha256"].__setitem__("B", hashlib.sha256(b"another B").hexdigest())))
    mod("n_ba_disagrees", lambda d: _edit_json(d / "rlval_B_vs_ownA.json", lambda b: b["paired_vs_base"]["B"]["quantities"]["PDMS"].__setitem__("mean", b["paired_vs_base"]["B"]["quantities"]["PDMS"]["mean"] + 0.05)))
    mod("n_ba_ttc_disagrees", lambda d: _edit_json(d / "rlval_B_vs_ownA.json", lambda b: b["paired_vs_base"]["B"]["quantities"]["TTC"].__setitem__("mean", -0.3)))
    mod("n_ba_base", lambda d: _edit_json(d / "rlval_B_vs_ownA.json", lambda b: b.__setitem__("base", "B")))
    mod("n_ba_garbage", lambda d: (d / "rlval_B_vs_ownA.json").write_text("[1, 2]", encoding="utf-8"))
    mod("n_ba_sets", lambda d: _edit_json(d / "rlval_B_vs_ownA.json", lambda b: b["eligibility"].__setitem__("sets_used", 7)))
    def wrong_both(d):                                                  # BA and rlval_vs_A0.json agree with each other on a dump hash that is NOT the file on disk
        h = hashlib.sha256(b"a dump that is not on this disk").hexdigest()
        _edit_json(d / "rlval_B_vs_ownA.json", lambda b: b["inputs"]["scores_sha256"].__setitem__("A", h))
        _edit_json(d / "rlval_vs_A0.json", lambda b: b["inputs"]["scores_sha256"].__setitem__("CTRL", h))
        write_S(d)
    mod("n_ba_wrong_both", wrong_both)
    mod("n_ba_estimator", lambda d: write_ba(W, d, "gain_hi", n_boot=50))                                                         # the REAL rl_val_eval.run at 50 resamples: not the registered estimator

    # ---- the arm's state
    def incomplete(d):
        for f in ("kill_replay.json", "rlval_vs_A0.json", "rlval_B_vs_ownA.json", "safeobj_S.json", "kill_checks.jsonl"):
            if (d / f).exists():
                (d / f).unlink()
        _drop_rows(d, lambda r: r.get("event") == "done" or (r.get("event") == "score_dump" and r.get("tag") in ("update750", "update1000")) or (r.get("event") == "train" and r["update"] > 600))
    mod("pending_incomplete", incomplete)
    A["n_dir_missing"] = m / "no_such_dir"
    # ---- bar G exact boundaries at dose.json's EXACT vertices (SYNTHETIC numbers written into the producers' files)
    dose = jload(DOSE)["full"]
    P = dose_pts()
    for nm, (pd_, lo_, tt_) in {"b_v1_eq": (0.60, 0.2, P[0][0]), "b_v1_below": (0.60 - 1e-9, 0.2, P[0][0]),                 # at vertex 1 the +0.60 floor binds (F = +0.59875)
                                "b_v2_eq": (P[1][1], 0.2, P[1][0]), "b_v2_below": (P[1][1] - 1e-9, 0.2, P[1][0]),
                                "b_v3_eq": (P[2][1], 0.2, P[2][0]), "b_v3_below": (P[2][1] - 1e-9, 0.2, P[2][0]),
                                "b_lb_zero": (3.0, 0.0, 0.0), "b_lb_tiny": (3.0, 1e-12, 0.0)}.items():
        d = copy_arm(base, m, nm)
        override_u1000(d, pd_, lo_, tt_)
        A[nm] = d
    for a_ in ("0.25", "0.375", "0.5"):
        d = copy_arm(base, m, f"dose_{a_}")
        override_u1000(d, dose[a_]["PDMS"]["mean"], dose[a_]["PDMS"]["ci95"][0], dose[a_]["TTC"]["mean"], dose[a_]["PDMS"]["ci95"][1])
        A[f"dose_{a_}"] = d
    # ---- bar L: controls with a real producer's u1000 read; the exact ties are SYNTHETIC dyadic deltas
    c = root / "ctrl"
    c.mkdir()
    for nm in ("same", "k1", "k2", "k4"):
        A[f"c_{nm}"] = ctrl_dir(W, c, f"c_{nm}", nm)

    def L_real(R0, R0d, R1d, R1o):
        return {"R0": A[f"c_{R0}"], "R0d": A[f"c_{R0d}"], "R1d": A[f"c_{R1d}"], "R1o": A[f"c_{R1o}"]}
    A["L_pass"] = L_real("k4", "same", "k1", "k4")
    A["L_fail_all_equal"] = L_real("k4", "k4", "k4", "k4")
    A["L_fail_floor"] = L_real("k4", "same", "same", "k2")
    s = root / "synth"
    s.mkdir()

    def L_syn(nm, R0, R0d, R1d, R1o):
        A[nm] = {k: synth_ctrl_dir(s, f"{nm}_{k}", v) for k, v in (("R0", R0), ("R0d", R0d), ("R1d", R1d), ("R1o", R1o))}
    L_syn("L_tie", -1.0, -0.5, -0.25, -1.0)                       # seed 0: e0 = 0.5 = 2 F_seed exactly -> not >
    L_syn("L_above_tie", -1.0, -0.498046875, -0.25, -1.0)         # e0 = 0.501953125 > 2 * 0.248046875
    L_syn("L_all_zero", -1.0, -1.0, -1.0, -1.0)                   # F = 0 and both effects exactly 0: not > 0
    L_syn("L_seed1_fails_alone", -2.0, 2.0, 0.5, -1.0)            # e0 = 4 > 2 F (F = 1.5); e1 = 1.5 <= 2 F = 3
    L_syn("L_seed0_fails_alone", -1.0, 0.5, 2.0, -2.0)            # e1 = 4 > 2 F (F = 1.5); e0 = 1.5 <= 2 F = 3
    A["L_pass_with_pdms_cost"] = {k: synth_ctrl_dir(s, f"Lpc_{k}", t, p_) for k, t, p_ in (("R0", -2.0, 3.0), ("R0d", 2.0, -2.0), ("R1d", 1.75, 0.5), ("R1o", -2.0, -7.0))}      # TTC passes (4.0 and 3.75 > 2 x 0.25), PDMS moves a lot
    A["L_pending"] = {k: v for k, v in A["L_pass"].items() if k != "R1o"}
    A["L_no_R0"] = {k: v for k, v in A["L_pass"].items() if k != "R0"}
    d = copy_arm(A["c_k4"], root / "ctrl", "c_k4_nolog")
    (d / "rlval_vs_A0.json").unlink()
    A["L_no_rlval"] = {**A["L_pass"], "R1o": d}
    d = copy_arm(A["c_k4"], root / "ctrl", "c_k4_otherB")
    (d / "scores" / "scores_update1000_B.npz").write_bytes(b"not the dump that was read")
    A["L_other_dump"] = {**A["L_pass"], "R1o": d}
    # ---- bar E: SYNTHETIC files (no producer: IMPL step 9 is not built), written per check
    (root / "E").mkdir()
    return A


def E_base():
    sub = lambda m, lo, hi: {"mean": m, "ci95": [lo, hi]}                       # noqa: E731
    pair = lambda p: {"PDMS": sub(*p), "NC": sub(0.1, -0.1, 0.3), "DAC": sub(0.0, -0.2, 0.2), "EP": sub(1.0, 0.5, 1.5), "TTC": sub(0.05, -0.3, 0.4), "C": sub(0.0, -0.1, 0.1), "DDC": sub(0.0, -0.1, 0.1)}
    return {"tier": "T1", "estimator": "paired log-cluster bootstrap, 10000 resamples, 95 %, seed 20260927", "n_tokens": 12146, "n_logs": 136,
            "pairs": {"R0d": pair((0.75, 0.30, 1.20)), "R1d": pair((0.5, 0.10, 0.90))}, "pooled": {"PDMS": sub(0.625, 0.35, 0.9), "resample": "joint"}, "floors": {"irep_abs_delta": 0.05}}


def E_file(root, name, edit=None, sync=True):
    """sync: the pooled mean follows the two pairs' means (the contract: pooled = their mean) unless the case is about exactly that."""
    o = E_base()
    if edit:
        edit(o)
    if sync and isinstance(o.get("pooled"), dict) and isinstance(o["pooled"].get("PDMS"), dict):
        o["pooled"]["PDMS"]["mean"] = (o["pairs"]["R0d"]["PDMS"]["mean"] + o["pairs"]["R1d"]["PDMS"]["mean"]) / 2.0
    p = root / "E" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    jsave(p, o)
    return str(p)


# ============================================================ bars G, L, E, the joint rule, provenance
def _G(T, d, name="R0d"):
    return chk_read(T, d, name)["G"]


@check("G:frontier_vertices_and_flat_ends")
def _(T, W, A, root, tag):
    P = dose_pts()
    f = lambda t: T.frontier(t, P)          # noqa: E731
    lin = lambda t, a, b: a[1] + (t - a[0]) * (b[1] - a[1]) / (b[0] - a[0])          # noqa: E731
    return (f(P[0][0]) == 0.60 and f(P[1][0]) == P[1][1] and f(P[2][0]) == P[2][1] and f(0.0) == 0.60 and f(0.7) == 0.60 and f(-0.9) == P[2][1] and f(-5.0) == P[2][1]
            and abs(f(-0.15) - lin(-0.15, P[0], P[1])) < 1e-12 and abs(f(-0.4) - lin(-0.4, P[1], P[2])) < 1e-12 and abs(f(-0.37) - lin(-0.37, P[1], P[2])) < 1e-12
            and f(-0.034) == 0.60 and f(-0.0359) == 0.60 and f(-0.0361) > 0.60 and f(-0.0361) == lin(-0.0361, P[0], P[1]))          # the floor binds until F crosses +0.60 at t = -0.03602


@check("G:frontier_is_monotone_and_never_below_the_floor")
def _(T, W, A, root, tag):
    P = dose_pts()
    ts = [-0.6 + 0.0013 * i for i in range(600)]
    v = [T.frontier(t, P) for t in ts]
    return all(b <= a + 1e-12 for a, b in zip(v, v[1:])) and min(v) >= 0.60 and max(v) <= P[2][1]


@check("G:boundary_arms_equal_to_the_bar_pass_just_below_fails")
def _(T, W, A, root, tag):
    out = []
    for k, want in (("b_v1_eq", "PASS"), ("b_v1_below", "FAIL"), ("b_v2_eq", "PASS"), ("b_v2_below", "FAIL"), ("b_v3_eq", "PASS"), ("b_v3_below", "FAIL")):
        g = _G(T, A[k])
        out.append(g["verdict"] == want and (g["verdict"] == "PASS") == g["point_ge_bar"] and g["lower_bound_gt_0"] is True)
    return all(out)


@check("G:lower_bound_exactly_zero_fails_a_hair_above_passes")
def _(T, W, A, root, tag):
    return _G(T, A["b_lb_zero"])["verdict"] == "FAIL" and _G(T, A["b_lb_zero"])["lower_bound_gt_0"] is False and _G(T, A["b_lb_tiny"])["verdict"] == "PASS"


@check("G:a_point_above_the_bar_with_lower_bound_zero_is_not_a_pass")
def _(T, W, A, root, tag):
    ctx = T.Ctx()
    g = T.bar_G({"pdms_mean": {"value": 9.0}, "pdms_lo": {"value": 0.0}, "pdms_hi": {"value": 10.0}, "ttc_mean": {"value": 0.0}}, [], T.load_dose(str(DOSE), ctx), ctx)
    return g["verdict"] == "FAIL" and g["point_ge_bar"] is True and g["lower_bound_gt_0"] is False


# ---- addendum R1: the VERTEX test (replaces the IMPL_PLAN selftest, which was wrong: alpha 0.25 does not sit on the boundary)
@check("G:vertex_alpha_0.25_reads_FAIL_by_0.00125")
def _(T, W, A, root, tag):
    r = chk_read(T, A["dose_0.25"])
    g = r["G"]
    P = dose_pts()
    return (g["verdict"] == "FAIL" and g["lower_bound_gt_0"] is True and g["point_ge_bar"] is False and g["bar_at_t"] == 0.60 and g["pdms_point_delta_x100"]["value"] == P[0][1]
            and abs(g["margin_point_minus_bar"] - (P[0][1] - 0.60)) < 1e-15 and abs(g["margin_point_minus_bar"] + 0.00125) < 1e-5 and abs(g["margin_point_minus_bar"] + 0.0012495138888889) < 1e-12
            and r["outcome"] == "O2" and r["outcome_flags"] == [])


@check("G:vertex_alpha_0.375_sits_exactly_on_F_and_reads_PASS")
def _(T, W, A, root, tag):
    r = chk_read(T, A["dose_0.375"])
    g = r["G"]
    P = dose_pts()
    return (g["verdict"] == "PASS" and g["point_ge_bar"] is True and g["margin_point_minus_bar"] == 0.0 and g["bar_at_t"] == P[1][1] == g["pdms_point_delta_x100"]["value"] and g["bar_at_t"] > 0.60 and r["outcome"] == "O1")


@check("G:vertex_alpha_0.5_sits_exactly_on_F_and_reads_PASS")
def _(T, W, A, root, tag):
    r = chk_read(T, A["dose_0.5"])
    g = r["G"]
    P = dose_pts()
    return (g["verdict"] == "PASS" and g["point_ge_bar"] is True and g["margin_point_minus_bar"] == 0.0 and g["bar_at_t"] == P[2][1] == g["pdms_point_delta_x100"]["value"] and g["bar_at_t"] > 0.60 and r["outcome"] == "O1")


@check("G:the_record_names_the_vertices_their_file_and_sha256_and_the_read_it_judged")
def _(T, W, A, root, tag):
    r = chk_read(T, A["o1"])
    g = r["G"]
    dose_sha = hashlib.sha256(DOSE.read_bytes()).hexdigest()
    ba = A["o1"] / "rlval_B_vs_ownA.json"
    return (dose_sha == T.DOSE_SHA256 and g["vertices_source"]["sha256"] == dose_sha and g["vertices_source"]["pinned_sha256"] == dose_sha and len(g["vertices"]) == 3
            and all(v["pdms_x100"]["sha256"] == dose_sha and v["ttc_x100"]["sha256"] == dose_sha and v["ttc_x100"]["source"] == os.path.abspath(str(DOSE)) for v in g["vertices"])
            and g["read"]["file"] == os.path.abspath(ba) and g["read"]["sha256"] == hashlib.sha256(ba.read_bytes()).hexdigest() and g["floor"] == 0.60
            and g["pdms_point_delta_x100"]["sha256"] == g["read"]["sha256"] and g["printed_vertices_round_match"] is True
            and g["disclosures"]["A-G2"].startswith("NOT_COMPUTED_BY_THIS_TOOL") and g["pool"]["pool_matches_registered"] is False)          # the fixture pool is 250 sets, not 3,000


@check("G:no_dose_json_is_NOT_READ_and_the_outcome_follows")
def _(T, W, A, root, tag):
    r = chk_read(T, A["o1"], dose=False)
    return r["G"]["verdict"] == "NOT_READ" and any("no dose.json" in p for p in r["G"]["problems"]) and r["outcome"] == "NOT_READ"


@check("G:a_dose_json_that_is_not_the_pinned_bytes_is_NOT_READ")
def _(T, W, A, root, tag):
    full = jload(DOSE)
    full["full"]["0.25"]["PDMS"]["mean"] += 0.01
    p = root / f"dose_tampered_{tag}.json"
    jsave(p, full)
    r = chk_read(T, A["o1"], dose=p)
    r2 = chk_read(T, A["o1"], dose=root / "no_such_dose.json")
    return (r["G"]["verdict"] == "NOT_READ" and any("not the pinned" in x for x in r["G"]["problems"]) and r["outcome"] == "NOT_READ"
            and r2["G"]["verdict"] == "NOT_READ" and any("not found" in x for x in r2["G"]["problems"]))


@check("G:dose_json_points_are_the_addendum_numbers_and_round_to_the_printed_vertices")
def _(T, W, A, root, tag):
    pts = T.dose_points(jload(DOSE))
    txt = ADD.read_text(encoding="utf-8")
    m = re.findall(r"α (0\.\d+) \(−(\d\.\d+), \+(\d\.\d+)\)", txt)
    assert [a for a, _, _ in m] == ["0.25", "0.375", "0.5"], m
    for (a, tt, pp), (t_, p_) in zip(m, pts):
        assert abs(-float(tt) - t_) < 5e-7 and abs(float(pp) - p_) < 5e-7, (a, tt, pp, t_, p_)
    return ([tuple(round(v, 2) for v in p) for p in pts] == [tuple(p) for p in T.G_POINTS_PRINTED] and T.dose_points({"full": {}}) is None and T.dose_points([]) is None
            and pts[0][1] < 0.60 < pts[1][1])


@check("G:killed_arm_has_no_u1000_so_G_is_NOT_READ_and_not_pending")
def _(T, W, A, root, tag):
    g = _G(T, A["o4_k3"])
    return g["verdict"] == "NOT_READ" and "killed before u1000" in g["problems"][0]


# ---- fail closed: every missing / broken / unpinned input reads NOT_READ and never an outcome
def notread(key, where, needle=None):
    def fn(T, W, A, root, tag):
        r = chk_read(T, A[key])
        assert r["outcome"] == "NOT_READ", (key, r["outcome"], r["problems"][:2])
        text = " | ".join(r["problems"] + (r.get("G") or {}).get("problems", []) + (r.get("log") or {}).get("problems", []) + (r.get("log") or {}).get("notes", []))
        assert (needle is None) or (needle in text), (key, needle, text[:400])
        if where == "S":
            assert r["S"]["verdict"] == "NOT_READ"
        if where == "G":
            assert r["G"]["verdict"] == "NOT_READ" and r["S"]["verdict"] == "PASS"
        if where == "log":
            assert r["log"]["readable"] is False and r["S"]["verdict"] == "PASS"
        return True
    CHECKS.append((f"notread:{key}", fn))


notread("n_S_missing", "S", "safeobj_S.json is missing")
notread("n_S_stale", "S", "stale")
notread("n_S_handedit", "S", "not self-consistent")
notread("n_S_notread", "S")
notread("n_log_cols", "log", "lacks finite")
notread("n_log_lam0", "log", "not the registered")
notread("n_log_eta", "log", "not the registered")
notread("n_log_ramp", "log", "--epo-ramp")
notread("n_log_temp", "log", "not the registered")
notread("n_log_temp_none", "log", "not the registered")
notread("n_log_gap", "log", "contiguous")
notread("n_log_done", "log", "`done` event")
notread("n_k4_replay_unfinished", None, "run's end is unknown")                                                       # fail closed: a replay kill beside a log that just stops is not an O7 until the end of the run is known
notread("n_log_cap_inconsistent", "log", "disagree about reaching lambda_max")
notread("n_log_decreasing", "log", "DECREASES")
notread("n_ba_missing", "G", "rlval_B_vs_ownA.json is missing")
notread("n_ba_other_dumps", "G", "NOT the ones")
notread("n_ba_other_B", "G", "NOT the ones")
notread("n_ba_wrong_both", None, "NOT this arm's scores")
notread("n_ba_disagrees", "G", "disagree")
notread("n_ba_ttc_disagrees", "G", "disagree")
notread("n_ba_base", "G", "not an rl_val_eval read")
notread("n_ba_garbage", "G")
notread("n_ba_sets", "G", "different numbers of sets")
notread("n_ba_estimator", "G", "not the registered one")


@check("notread:n_log_missing_is_not_pending_when_the_chain_already_read_the_arm")
def _(T, W, A, root, tag):
    r = chk_read(T, A["n_log_missing"])
    return r["outcome"] == "NOT_READ" and r["state"] == "unclear"


@check("notread:n_dir_missing")
def _(T, W, A, root, tag):
    r = chk_read(T, A["n_dir_missing"])
    return r["outcome"] == "NOT_READ" and r["state"] == "missing"


@check("tolerated:a_half_written_final_log_line_is_a_note_not_a_problem")
def _(T, W, A, root, tag):
    r = chk_read(T, A["n_log_half_line"])
    return r["outcome"] == "O1" and any("half-written" in n for n in r["log"]["notes"])


@check("tolerated:a_legacy_data_block_without_temp_is_read_exactly_as_before_PICKCON")
def _(T, W, A, root, tag):
    first = next(json.loads(ln) for ln in (A["legacy_block"] / "sft.jsonl").read_text(encoding="utf-8").splitlines() if '"event": "data"' in ln)
    assert "temp" not in first["epo_con"] and set(first["epo_con"]) == set(T.REGISTERED_EPO_CON), first["epo_con"]          # the fixture IS the legacy shape
    r, ref = chk_read(T, A["legacy_block"]), chk_read(T, A["o1"])
    return r["outcome"] == ref["outcome"] == "O1" and r["log"]["readable"] is True and r["log"]["problems"] == []


@check("compat:the_banked_R0d_trainer_log_has_the_legacy_epo_con_block_and_is_read")
def _(T, W, A, root, tag):
    if not BANKED_R0D_LOG.exists():                                      # a package shipped without raw/ (a pod): the synthetic legacy arm above still runs
        return True
    data = next(json.loads(ln) for ln in BANKED_R0D_LOG.read_text(encoding="utf-8").splitlines() if '"event": "data"' in ln)
    assert "temp" not in data["epo_con"] and data["epo_con"] == T.REGISTERED_EPO_CON, data["epo_con"]                     # REAL producer: the pre-PICKCON trainer's block IS the registered eight-key block
    lg = T.read_log(str(BANKED_R0D_LOG), None, T.Ctx())
    return lg["readable"] is True and lg["problems"] == []


# ============================================================ R0e-pick B3: the constraint's registered T is PER ARM, from the arm's registration, never from the log
T001 = 0.01
RLS1D_ARMS = HERE / "rls1d_arms.json"                                                     # the REAL registered argv of R0d / R1d / R1o
BANKED_R0D_RECORD = PKGROOT / "raw" / "2026-10-18-r0d-outcome" / "raw" / "safeobj_outcome.json"          # the pre-B3 tool's record of the banked R0d arm


def real_run_t001(W, root):
    # a SECOND real scorer_finetune.main() run, identical to real_run's argv plus --epo-con-temp 0.01 (the producer of a nine-key block whose temp is NOT the default)
    if not hasattr(W, "_real_t001"):
        import test_rl_epo as TE
        td = root / "real_main_t001"
        td.mkdir()
        W._real_t001 = TE.smoke(SF_PATH.read_text(encoding="utf-8"), str(td), epo=True, epo_lam=1.0, native7=True, terms=True, tag="oc_real_t001",
                                extra=("--epo-con", "g5", "--epo-con-eps", "0", "--epo-con-lr", "4", "--epo-con-max", "10", "--epo-con-lam0", "0.5", "--epo-con-temp", "0.01"))
        W._real_t001_path = td / "out" / "sft.jsonl"
    return W._real_t001, W._real_t001_path


def _read_real(T, p, temp=None):
    kw = {} if temp is None else {"registered_temp": temp}
    return T.read_log(str(p), 25, T.Ctx(), registered={**T.REGISTERED_EPO_CON, "lam0": 0.5}, registered_ramp=2, **kw)           # the argv of the smoke runs (lam0 0.5, ramp 2)


def _says(lg, *needles):
    return any(all(n in x for n in needles) for x in lg["problems"])


def _data_row(p):
    return next(json.loads(ln) for ln in Path(p).read_text(encoding="utf-8").splitlines() if '"event": "data"' in ln)


def _log_with_temp(src, dst, value, drop=False):
    # SYNTHETIC: src's sft.jsonl with the data event's epo_con.temp replaced by `value` (or the key dropped)
    rows = [json.loads(ln) for ln in Path(src).read_text(encoding="utf-8").splitlines() if ln.strip()]
    for r in rows:
        if r.get("event") == "data":
            if drop:
                r["epo_con"].pop("temp", None)
            else:
                r["epo_con"]["temp"] = value
    Path(dst).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return dst


def _arms_file(root, name, arms):
    f = root / name
    f.write_text(json.dumps({"schema": "synthetic", "arms": {k: {"argv": v} for k, v in arms.items()}}), encoding="utf-8")
    return f


@check("con_temp:a_REAL_trainer_run_at_epo_con_temp_0.01_is_read_under_a_registered_0.01")
def _(T, W, A, root, tag):
    res, p = real_run_t001(W, root)
    assert res["run"] == 0 and p.exists()
    d = _data_row(p)
    assert d["epo_con"]["temp"] == T001 and d["epo"]["temp"] == EPO_TEMP, (d["epo_con"], d["epo"])          # the producer's own block: the constraint at 0.01, the EPO reward term still at 0.1
    lg = _read_real(T, p, T001)
    return lg["readable"] is True and lg["problems"] == []


@check("con_temp:a_registered_0.1_REFUSES_the_REAL_0.01_run_and_a_registered_0.01_REFUSES_the_REAL_default_run")
def _(T, W, A, root, tag):
    _, p1 = real_run_t001(W, root)
    _, p0 = real_run(W, root)
    for lg in (_read_real(T, p1), _read_real(T, p1, 0.1)):                                    # the tool's default and an explicit 0.1: the 0.01 run is not that arm
        assert lg["readable"] is False and _says(lg, "not the registered", "registered constraint T = 0.1"), lg["problems"]
    lg = _read_real(T, p0, T001)
    assert lg["readable"] is False and _says(lg, "not the registered", "registered constraint T = 0.01"), lg["problems"]
    return _read_real(T, p0, 0.1)["readable"] is True and _read_real(T, p0)["readable"] is True          # the reverse still reads: the default run IS the 0.1 arm


@check("con_temp:an_arm_whose_REAL_EpoDual_ran_at_0.01_reads_O1_under_a_registered_0.01_and_NOT_READ_under_the_default_or_an_explicit_0.1")
def _(T, W, A, root, tag):
    assert _data_row(A["o1_t001"] / "sft.jsonl")["epo_con"]["temp"] == T001                  # the fixture is what it says
    r = chk_read(T, A["o1_t001"], registered_temp=T001)
    assert r["outcome"] == "O1" and r["log"]["readable"] is True and r["log"]["problems"] == [], (r["outcome"], r["problems"][:2])
    assert r["registered_con_temp"]["value"] == T001
    for kw in ({}, {"registered_temp": 0.1}):
        r = chk_read(T, A["o1_t001"], **kw)
        assert r["outcome"] == "NOT_READ" and r["log"]["readable"] is False and _says(r["log"], "not the registered"), (kw, r["outcome"])
    return True


@check("con_temp:an_arm_whose_log_says_0.1_is_NOT_READ_under_a_registered_0.01_and_reads_O1_under_0.1")
def _(T, W, A, root, tag):
    r = chk_read(T, A["o1"], registered_temp=T001)
    assert r["outcome"] == "NOT_READ" and r["log"]["readable"] is False and _says(r["log"], "not the registered", "registered constraint T = 0.01"), r["log"]["problems"]
    return chk_read(T, A["o1"], registered_temp=0.1)["outcome"] == "O1" and chk_read(T, A["o1"])["outcome"] == "O1"


@check("con_temp:a_registered_0.01_REFUSES_a_legacy_block_with_no_temp_(a_trainer_that_predates_the_flag_ran_the_constraint_at_0.1)")
def _(T, W, A, root, tag):
    r = chk_read(T, A["legacy_block"], registered_temp=T001)
    assert r["outcome"] == "NOT_READ" and _says(r["log"], "no `temp` key", "0.01"), r["log"]["problems"]
    return chk_read(T, A["legacy_block"], registered_temp=0.1)["outcome"] == "O1" and chk_read(T, A["legacy_block"])["outcome"] == "O1"


@check("con_temp:a_registered_0.01_REFUSES_every_other_temp_in_the_log_and_a_registered_1.0_REFUSES_a_bool_True")
def _(T, W, A, root, tag):
    src = A["o1_t001"] / "sft.jsonl"
    for nm, v in (("t01", 0.1), ("t02", 0.02), ("off", 0.0100001), ("zero", 0.0), ("neg", -0.01), ("none", None), ("str", "0.01"), ("lst", [0.01]), ("nan", float("nan"))):
        lg = T.read_log(str(_log_with_temp(src, root / f"ct_{tag}_{nm}.jsonl", v)), None, T.Ctx(), registered_temp=T001)
        assert lg["readable"] is False and _says(lg, "not the registered"), (nm, lg["problems"])
    ok = T.read_log(str(_log_with_temp(src, root / f"ct_{tag}_ok.jsonl", 0.01)), None, T.Ctx(), registered_temp=T001)
    assert ok["readable"] is True and ok["problems"] == [], ok["problems"]                    # the control: the same rewrite with the registered value reads
    lg = T.read_log(str(_log_with_temp(src, root / f"ct_{tag}_bool.jsonl", True)), None, T.Ctx(), registered_temp=1.0)
    assert lg["readable"] is False and _says(lg, "not the registered"), lg["problems"]       # True == 1.0 in Python: a bool must not vouch for a registered 1.0
    ok1 = T.read_log(str(_log_with_temp(src, root / f"ct_{tag}_one.jsonl", 1.0)), None, T.Ctx(), registered_temp=1.0)
    return ok1["readable"] is True


@check("con_temp:each_arm_is_read_against_its_OWN_registered_T_(R0d_0.01_beside_R1d_0.1_and_both_0.01)")
def _(T, W, A, root, tag):
    rec = RUN(T, {"R0d": A["o1_t001"], "R1d": A["r1d_o1"]}, con_temps={"R0d": T001})
    assert rec["arms"]["R0d"]["outcome"] == "O1" and rec["arms"]["R1d"]["outcome"] == "O1" and rec["joint"]["replication"] == "REPLICATED_O1_BOTH", [rec["arms"][n]["outcome"] for n in rec["arms"]]
    assert rec["arms"]["R0d"]["registered_con_temp"]["value"] == T001 and "registered_con_temp" not in rec["arms"]["R1d"]
    assert rec["registered"]["epo_con_temp_by_arm"] == {"R0d": {"value": T001, "source": {"kind": "caller"}}} and rec["registered"]["epo_con_temp_default"] == 0.1
    both = RUN(T, {"R0d": A["o1_t001"], "R1d": A["r1d_o1_t001"]}, con_temps={"R0d": T001, "R1d": T001})
    assert both["joint"]["replication"] == "REPLICATED_O1_BOTH"
    wrong = RUN(T, {"R0d": A["o1_t001"], "R1d": A["r1d_o1"]}, con_temps={"R0d": T001, "R1d": T001})          # R1d's log says 0.1: it is not its registered arm
    return wrong["arms"]["R0d"]["outcome"] == "O1" and wrong["arms"]["R1d"]["outcome"] == "NOT_READ" and wrong["joint"]["replication"] != "REPLICATED_O1_BOTH"


@check("con_temp:naming_no_T_writes_the_record_it_always_wrote_and_registering_the_default_changes_nothing_that_is_read")
def _(T, W, A, root, tag):
    a, b, c = (RUN(T, {"R0d": A["o1"]}, **kw) for kw in ({}, {"con_temps": {}}, {"con_temps": {"R0d": 0.1}}))
    assert a["record_sha256"] == b["record_sha256"], "an empty registration must write the same record"
    assert "epo_con_temp_by_arm" not in a["registered"] and "registered_con_temp" not in a["arms"]["R0d"]
    ra, rc = a["arms"]["R0d"], c["arms"]["R0d"]
    return ra["outcome"] == rc["outcome"] == "O1" and ra["log"] == rc["log"] and ra["facts"] == rc["facts"] and ra["table"] == rc["table"]


@check("con_temp:a_registration_that_is_not_a_positive_finite_number_for_a_constrained_arm_is_refused_before_anything_is_read")
def _(T, W, A, root, tag):
    for bad in ({"R1o": T001}, {"R0": T001}, {"R9": T001}, {"R0d": 0}, {"R0d": -0.1}, {"R0d": None}, {"R0d": True}, {"R0d": "0.01"}, {"R0d": float("nan")}, {"R0d": float("inf")}):
        try:
            RUN(T, {"R0d": A["o1"]}, con_temps=bad)
        except ValueError:
            continue
        raise AssertionError(("accepted", bad))
    for bad_t in (0, -1.0, None, True, "0.1", float("nan"), float("inf")):
        lg = T.read_log(str(A["o1"] / "sft.jsonl"), None, T.Ctx(), registered_temp=bad_t)
        assert lg["readable"] is False and _says(lg, "positive finite number"), (bad_t, lg["problems"])
    return True


@check("con_temp:the_banked_R0d_log_reads_exactly_as_its_banked_pre_B3_record_says_and_is_REFUSED_under_a_registered_0.01")
def _(T, W, A, root, tag):
    if not (BANKED_R0D_LOG.exists() and BANKED_R0D_RECORD.exists()):                         # a package shipped without raw/: the synthetic legacy arm above still runs
        return True
    banked = jload(BANKED_R0D_RECORD)["arms"]["R0d"]
    c = int(banked["first_breached_check"][len("update"):])
    lg, lg1, lg01 = (T.read_log(str(BANKED_R0D_LOG), c, T.Ctx(), **kw) for kw in ({}, {"registered_temp": 0.1}, {"registered_temp": T001}))
    assert lg == lg1 and lg["readable"] is True and lg["problems"] == [], lg["problems"]
    want = banked["log"]
    for k in ("readable", "problems", "notes", "lam_zero", "lam_pos_share_ok", "cap_hit", "violation_ok", "window_overlaps_ramp", "columns_read"):
        assert lg[k] == want[k], (k, lg[k], want[k])
    assert {k: v["value"] for k, v in lg["numbers"].items()} == {k: v["value"] for k, v in want["numbers"].items()}
    assert all(v["sha256"] == want["numbers"][k]["sha256"] for k, v in lg["numbers"].items())          # the same bytes the pre-B3 tool read on the pod
    return lg01["readable"] is False and _says(lg01, "no `temp` key", "0.01")


@check("con_temp:the_arms_file_reader_resolves_the_REAL_registered_argv_of_R0d_to_0.1_by_default_and_refuses_R1o_whose_constraint_is_off")
def _(T, W, A, root, tag):
    t, src = T.con_temp_from_arms_file(str(RLS1D_ARMS), "R0d")
    assert t == 0.1 and src["resolved"] == "default" and src["tokens"] == {"--epo-con-temp": None, "--epo-temp": None}, (t, src)
    assert src["sha256"] == hashlib.sha256(RLS1D_ARMS.read_bytes()).hexdigest() and src["arm"] == "R0d"
    for arm in ("R1o", "R9"):
        try:
            T.con_temp_from_arms_file(str(RLS1D_ARMS), arm)
        except ValueError:
            continue
        raise AssertionError(("accepted", arm))
    return True


@check("con_temp:the_arms_file_reader_resolves_the_T_the_trainer_would_(con_temp_else_epo_temp_else_default)_and_refuses_an_ambiguous_or_constraint_less_registration")
def _(T, W, A, root, tag):
    base = ["--epo", "--epo-con", "g5", "--epo-con-eps", "0"]
    arms = {"named": base + ["--epo-con-temp", "0.01"], "named_eq": base + ["--epo-con-temp=0.02"], "inherit": base + ["--epo-temp", "0.05"], "both": base + ["--epo-temp", "0.05", "--epo-con-temp", "0.01"], "default": base,
            "twice": base + ["--epo-con-temp", "0.01", "--epo-con-temp", "0.02"], "novalue": base + ["--epo-con-temp"], "nan": base + ["--epo-con-temp", "nan"], "word": base + ["--epo-con-temp", "cold"],
            "zero": base + ["--epo-con-temp", "0"], "off": ["--epo", "--epo-con", "off"], "noconstraint": ["--epo"]}
    f = _arms_file(root, f"arms_{tag}.json", arms)
    for k, v in {"named": 0.01, "named_eq": 0.02, "inherit": 0.05, "both": 0.01, "default": 0.1}.items():
        got, src = T.con_temp_from_arms_file(str(f), k)
        assert got == v, (k, got, v)
    assert T.con_temp_from_arms_file(str(f), "named")[1]["resolved"] == "--epo-con-temp" and T.con_temp_from_arms_file(str(f), "inherit")[1]["resolved"] == "--epo-temp"
    for k in ("twice", "novalue", "nan", "word", "zero", "off", "noconstraint", "absent_arm"):
        try:
            T.con_temp_from_arms_file(str(f), k)
        except ValueError:
            continue
        raise AssertionError(("accepted", k))
    try:
        T.con_temp_from_arms_file(str(root / f"no_such_arms_{tag}.json"), "named")
    except ValueError:
        return True
    raise AssertionError("a missing arms file was accepted")


@check("pending:an_unfinished_arm_is_PENDING_and_never_an_outcome")
def _(T, W, A, root, tag):
    r = chk_read(T, A["pending_incomplete"])
    return r["outcome"] == "PENDING" and r["state"] == "incomplete" and r["S"] is None and r["G"] is None


@check("pending:PENDING_arm_gives_a_PENDING_joint")
def _(T, W, A, root, tag):
    rec = RUN(T, {"R0d": A["pending_incomplete"]})
    return rec["arms"]["R0d"]["outcome"] == "PENDING" and rec["joint"]["replication"] == "PENDING"


# ---- bar L
def L_run(T, dirs):
    return T.bar_L({k: str(v) for k, v in dirs.items()}, T.Ctx())


@check("L:pass_real_producer_both_seeds_clear_2xF_seed")
def _(T, W, A, root, tag):
    L = L_run(T, A["L_pass"])
    t = L["ttc"]
    return (L["verdict"] == "PASS" and abs(t["effect_seed0"] - 1.6) < 1e-9 and abs(t["effect_seed1"] - 1.2) < 1e-9 and abs(t["F_seed"] - 0.4) < 1e-9 and t["seed0_ok"] and t["seed1_ok"]
            and all(L["inputs"][n]["ttc_B_minus_A_x100"]["sha256"] for n in ("R0", "R0d", "R1d", "R1o")))


@check("L:seed1_fails_alone_SYNTHETIC_dyadic")
def _(T, W, A, root, tag):
    L = L_run(T, A["L_seed1_fails_alone"])
    t = L["ttc"]
    return L["verdict"] == "FAIL" and t["seed0_ok"] is True and t["seed1_ok"] is False and t["effect_seed0"] == 4.0 and t["effect_seed1"] == 1.5 and t["F_seed"] == 1.5


@check("L:seed0_fails_alone_SYNTHETIC_dyadic")
def _(T, W, A, root, tag):
    L = L_run(T, A["L_seed0_fails_alone"])
    t = L["ttc"]
    return L["verdict"] == "FAIL" and t["seed0_ok"] is False and t["seed1_ok"] is True and t["effect_seed1"] == 4.0 and t["effect_seed0"] == 1.5 and t["F_seed"] == 1.5


@check("L:four_equal_real_arms_have_no_lever_effect")
def _(T, W, A, root, tag):
    L = L_run(T, A["L_fail_all_equal"])
    return L["verdict"] == "FAIL" and abs(L["ttc"]["effect_seed0"]) < 1e-9 and abs(L["ttc"]["effect_seed1"]) < 1e-9 and L["ttc"]["seed0_ok"] is False and L["ttc"]["seed1_ok"] is False


@check("L:a_large_seed_floor_fails_a_positive_effect")
def _(T, W, A, root, tag):
    L = L_run(T, A["L_fail_floor"])
    return L["verdict"] == "FAIL" and L["ttc"]["effect_seed1"] > 0 and L["ttc"]["effect_seed1"] < L["ttc"]["floor_2xF_seed"] and abs(L["ttc"]["F_seed"] - 0.8) < 1e-9


@check("L:exact_tie_effect_equal_2xF_is_not_a_pass_SYNTHETIC_dyadic")
def _(T, W, A, root, tag):
    L = L_run(T, A["L_tie"])
    t = L["ttc"]
    return L["verdict"] == "FAIL" and t["effect_seed0"] == 0.5 and t["floor_2xF_seed"] == 0.5 and t["seed0_ok"] is False and t["seed1_ok"] is True


@check("L:a_hair_above_the_tie_passes_SYNTHETIC_dyadic")
def _(T, W, A, root, tag):
    L = L_run(T, A["L_above_tie"])
    t = L["ttc"]
    return L["verdict"] == "PASS" and t["effect_seed0"] > t["floor_2xF_seed"] and t["seed0_ok"] and t["seed1_ok"]


@check("L:zero_effect_with_zero_floor_is_not_a_pass_SYNTHETIC")
def _(T, W, A, root, tag):
    L = L_run(T, A["L_all_zero"])
    return L["verdict"] == "FAIL" and L["ttc"]["F_seed"] == 0.0 and L["ttc"]["effect_seed0"] == 0.0


@check("L:PENDING_until_R1o_exists_and_never_PASS")
def _(T, W, A, root, tag):
    L = L_run(T, A["L_pending"])
    return L["verdict"] == "PENDING" and "R1o" in L["why"]


@check("L:NOT_READ_without_the_banked_R0_or_a_control_read_or_with_another_dump")
def _(T, W, A, root, tag):
    return (L_run(T, A["L_no_R0"])["verdict"] == "NOT_READ" and L_run(T, A["L_no_rlval"])["verdict"] == "NOT_READ" and L_run(T, A["L_other_dump"])["verdict"] == "NOT_READ"
            and any("sha256 differs" in p for p in L_run(T, A["L_other_dump"])["problems"]))


@check("L:the_PDMS_cost_is_reported_not_a_bar_with_its_own_yardstick_beside_the_TTC_one")
def _(T, W, A, root, tag):
    L = L_run(T, A["L_pass_with_pdms_cost"])
    c = L["pdms"]["cost_report_not_a_bar"]
    return ("cost_report_not_a_bar" in L["pdms"] and c["note"].startswith("PREREG A-L1") and "seed0_ok" not in L["pdms"] and L["decided_on"].startswith("TTC only")
            and c["yardstick_2xF_seed_pdms"] == 20.0 and c["yardstick_2xF_seed_ttc_printed_beside"] == 0.5 and c["seed0_abs_effect_gt_2xF"] is False and L["pdms"]["effect_seed0"] == -5.0 and L["pdms"]["effect_seed1"] == 7.5)


@check("L:the_PDMS_numbers_never_move_the_verdict_SYNTHETIC")
def _(T, W, A, root, tag):
    s = Path(tempfile.mkdtemp(dir=root))
    base = {k: synth_ctrl_dir(s, f"x_{k}", v, pdms_delta=pd) for (k, v), pd in zip((("R0", -1.0), ("R0d", -0.5), ("R1d", -0.25), ("R1o", -1.0)), (3.0, -2.0, 0.5, -7.0))}
    return (L_run(T, base)["verdict"] == "FAIL" and L_run(T, A["L_tie"])["verdict"] == "FAIL"
            and L_run(T, A["L_pass_with_pdms_cost"])["verdict"] == "PASS")          # a lever that passes on TTC passes L whatever it costs in PDMS: decided on TTC ONLY


# ---- bar E (SYNTHETIC: no producer exists)
@check("E:PENDING_when_no_navtest_read_exists_and_never_ADOPT")
def _(T, W, A, root, tag):
    r = T.bar_E(None, T.Ctx())
    return r["verdict"] == "PENDING" and "IMPL step 9" in r["why"]


@check("E:NOT_READ_for_a_named_file_that_does_not_exist")
def _(T, W, A, root, tag):
    return T.bar_E(str(root / "E" / "nope.json"), T.Ctx())["verdict"] == "NOT_READ"


def E_case(name, edit, verdict, clauses=None, problem=None, rl=0.1, sync=True):
    """rl = the RL-val training-seed floor handed to bar_E (None = unavailable)."""
    def fn(T, W, A, root, tag):
        r = T.bar_E(E_file(root, f"{name}_{tag}", edit, sync), T.Ctx(), None if rl is None else {"value": rl, "terms": {}}, "the test gave none" if rl is None else None)
        assert r["verdict"] == verdict, (name, r["verdict"], r.get("problems"), r.get("clauses"))
        for k, v in (clauses or {}).items():
            assert r["clauses"][k] is v, (name, k, r["clauses"])
        if problem:
            assert any(problem in p for p in r["problems"]), (name, r["problems"])
        return True
    CHECKS.append((f"E:{name}->{verdict}", fn))


def _set(path, val):
    def e(o):
        cur = o
        for k in path[:-1]:
            cur = cur[k]
        cur[path[-1]] = val
    return e


def _both(*es):
    def e(o):
        for f in es:
            f(o)
    return e


C1, C2, C3, C4 = "1_each_ge_0.25_and_mean_ge_0.5", "2_pooled_lower_gt_0", "3_mean_contrast_gt_2x_max_floor", "4_ttc_nc_upper_ge_0_each_pair"
PDMS_PAIR = lambda a, b: _both(_set(["pairs", "R0d", "PDMS", "mean"], a), _set(["pairs", "R1d", "PDMS", "mean"], b))          # noqa: E731
E_case("adopt_base", None, "ADOPT", {C1: True, C2: True, C3: True, C4: True})
E_case("each_exactly_025_and_mean_exactly_05", PDMS_PAIR(0.25, 0.75), "NOT_PROVEN", {C1: True, C3: False})
E_case("mean_exactly_05_two_halves", PDMS_PAIR(0.5, 0.5), "ADOPT", {C1: True, C3: True})
E_case("one_pair_below_025_mean_ok", PDMS_PAIR(1.0, 0.1875), "NOT_PROVEN", {C1: False})
E_case("mean_below_05", PDMS_PAIR(0.5, 0.25), "NOT_PROVEN", {C1: False})
E_case("pooled_lower_exactly_zero", _set(["pooled", "PDMS", "ci95"], [0.0, 0.9]), "NOT_PROVEN", {C2: False})
E_case("pooled_lower_a_hair_above_zero", _set(["pooled", "PDMS", "ci95"], [1e-9, 0.9]), "ADOPT")
# ---- addendum R4 / A-E3: the seed floor is the larger of the RL-val training-seed floor and the navtest seed floor |R0d - R1d|, then the I-rep; each tie is a NOT_PROVEN, a hair below clears
E_case("floor_tie_irep", _set(["floors", "irep_abs_delta"], 0.3125), "NOT_PROVEN", {C3: False})
E_case("floor_tie_navtest_seed_floor", PDMS_PAIR(0.78125, 0.46875), "NOT_PROVEN", {C3: False, C1: True})           # |0.78125 - 0.46875| = 0.3125 = mean / 2 exactly
E_case("floor_tie_rlval_seed_floor", None, "NOT_PROVEN", {C3: False}, rl=0.3125)
E_case("floor_just_below_tie_irep", _set(["floors", "irep_abs_delta"], 0.3), "ADOPT")
E_case("floor_just_below_tie_rlval", None, "ADOPT", None, rl=0.3)
E_case("seed_floor_abs_in_the_file_is_IGNORED_the_tool_derives_the_floors", _set(["floors", "seed_floor_abs"], 100.0), "ADOPT", {C3: True}, "seed_floor_abs")
E_case("a_huge_irep_in_the_file_still_counts", _set(["floors", "irep_abs_delta"], 100.0), "NOT_PROVEN", {C3: False})
E_case("rlval_floor_unavailable_clause3_is_unread_so_the_file_is_NOT_READ", None, "NOT_READ", None, "clause 3 is unread", rl=None)
E_case("rlval_floor_unavailable_but_REFUTED_needs_no_floor", _set(["pooled", "PDMS", "ci95"], [-0.5, 0.2499]), "REFUTED", None, None, rl=None)
# ---- clause 4: PER PAIR, the TTC and NC upper bounds (four bounds)
E_case("ttc_upper_adverse_in_one_pair", _set(["pairs", "R1d", "TTC", "ci95"], [-0.9, -0.001]), "NOT_PROVEN", {C4: False})
E_case("nc_upper_adverse_in_one_pair", _set(["pairs", "R0d", "NC", "ci95"], [-0.9, -0.001]), "NOT_PROVEN", {C4: False})
E_case("nc_upper_exactly_zero_is_not_adverse", _set(["pairs", "R0d", "NC", "ci95"], [-0.9, 0.0]), "ADOPT")
E_case("ep_adverse_does_not_block_clause_4_which_is_TTC_and_NC", _set(["pairs", "R0d", "EP", "ci95"], [-1.0, -0.001]), "ADOPT", {C4: True})
E_case("refuted_when_pooled_upper_below_025", _set(["pooled", "PDMS", "ci95"], [-0.5, 0.2499]), "REFUTED")
E_case("pooled_upper_exactly_025_is_not_refuted", _both(PDMS_PAIR(0.25, 0.25), _set(["pooled", "PDMS", "ci95"], [-0.5, 0.25])), "NOT_PROVEN")
E_case("adopt_and_refuted_both_is_inconsistent", _set(["pooled", "PDMS", "ci95"], [0.1, 0.2]), "NOT_READ")
E_case("subscore_missing_clause5", lambda o: o["pairs"]["R1d"].pop("EP"), "NOT_READ", None, "clause 5")
E_case("estimator_not_registered", _set(["estimator"], "paired log-cluster bootstrap, 50 resamples, seed 20260927"), "NOT_READ")
E_case("tier_not_T1", _set(["tier"], "T0"), "NOT_READ")
E_case("extra_pair", lambda o: o["pairs"].__setitem__("R1o", o["pairs"]["R1d"]), "NOT_READ")
E_case("malformed_interval", _set(["pairs", "R0d", "TTC", "ci95"], [0.5, -0.5]), "NOT_READ")
E_case("floor_missing", lambda o: o["floors"].pop("irep_abs_delta"), "NOT_READ")
E_case("pooled_missing", lambda o: o.pop("pooled"), "NOT_READ")
# ---- addendum R4 / A-E1: the pooled contrast IS the mean of the pairs' contrasts, on the same 12,146 tokens, resampled JOINTLY over the 136 logs
E_case("pooled_mean_is_not_the_mean_of_the_two_pairs", _set(["pooled", "PDMS", "mean"], 0.7), "NOT_READ", None, "mean of the two pairs", sync=False)
E_case("pooled_resampled_independently_is_refused", _set(["pooled", "resample"], "independent"), "NOT_READ", None, "JOINTLY")
E_case("pooled_resample_undeclared_is_refused", lambda o: o["pooled"].pop("resample"), "NOT_READ", None, "JOINTLY")
E_case("n_tokens_is_not_12146", _set(["n_tokens"], 12000), "NOT_READ", None, "12146")
E_case("n_logs_is_not_136", _set(["n_logs"], 135), "NOT_READ", None, "136")


@check("E:every_number_carries_its_source_and_sha256")
def _(T, W, A, root, tag):
    p = E_file(root, f"prov_{tag}")
    r = T.bar_E(p, T.Ctx(), {"value": 0.1, "terms": {}})
    sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return all(v["sha256"] == sha and v["source"] == os.path.abspath(p) for v in r["numbers"].values()) and len(r["numbers"]) == 7


@check("E:the_RL_val_seed_floor_is_derived_from_L_inputs_PDMS_both_terms_and_run_hands_it_to_E")
def _(T, W, A, root, tag):
    L = L_run(T, A["L_pass_with_pdms_cost"])
    fl, why = T.rlval_seed_floor(L)
    f0, why0 = T.rlval_seed_floor({"verdict": "PENDING"})
    p = E_file(root, f"fromL_{tag}")
    rec = RUN(T, {"R0d": A["o1"], "R1d": A["r1d_o1"], "R0": A["c_k4"], "R1o": A["c_k4"]}, navtest_e=p)
    e = rec["E"]
    return (fl["value"] == 10.0 and fl["terms"] == {"abs_R0d_minus_R1d": 2.5, "abs_R0_minus_R1o": 10.0} and why is None and f0 is None and "bar L is PENDING" in why0
            and e["verdict"] == "ADOPT" and e["floor"]["rlval_seed_floor"]["value"] == rec["L"]["pdms"]["F_seed"] and e["floor"]["max_of_all"] == max(0.25, 0.05, rec["L"]["pdms"]["F_seed"]))


@check("E:run_without_R0_and_R1o_cannot_read_clause_3")
def _(T, W, A, root, tag):
    p = E_file(root, f"noL_{tag}")
    rec = RUN(T, {"R0d": A["o1"], "R1d": A["r1d_o1"]}, navtest_e=p)
    return rec["E"]["verdict"] == "NOT_READ" and any("clause 3 is unread" in x for x in rec["E"]["problems"]) and rec["L"]["verdict"] == "PENDING"


# ---- the joint rule of section 6's last paragraph
def joint_case(name, k0, k1, want):
    def fn(T, W, A, root, tag):
        dirs = {n: A[k] for n, k in (("R0d", k0), ("R1d", k1)) if k}
        rec = RUN(T, dirs)
        assert rec["joint"]["replication"] == want, (name, rec["joint"], {n: a["outcome"] for n, a in rec["arms"].items()})
        return True
    CHECKS.append((f"joint:{name}->{want}", fn))


joint_case("both_O1", "o1", "r1d_o1", "REPLICATED_O1_BOTH")
joint_case("r0d_O1_r1d_not", "o1", "r1d_o2", "NOT_REPLICATED")
joint_case("r1d_O1_r0d_not", "o2", "r1d_o1", "NOT_APPLICABLE")
joint_case("neither", "o2", "r1d_o2", "NOT_APPLICABLE")
joint_case("only_R0d", "o1", None, "PENDING")


# ---- provenance and the record
def walk(o, fn):
    if isinstance(o, dict):
        fn(o)
        for v in o.values():
            walk(v, fn)
    elif isinstance(o, list):
        for v in o:
            walk(v, fn)


@check("record:every_number_names_its_source_file_and_that_files_current_sha256")
def _(T, W, A, root, tag):
    rec = RUN(T, {"R0d": A["o1"], "R1d": A["r1d_o1"], **{k: v for k, v in A["L_pass"].items() if k in ("R0", "R1o")}})
    seen = []
    walk(rec, lambda o: seen.append(o) if ("value" in o and "source" in o) else None)
    ok = len(seen) > 20
    for n in seen:
        ok = ok and Path(n["source"]).exists() and n["sha256"] == hashlib.sha256(Path(n["source"]).read_bytes()).hexdigest() and len(n["sha256"]) == 64
    return ok


@check("record:the_record_hash_is_self_consistent_and_inputs_are_listed")
def _(T, W, A, root, tag):
    rec = RUN(T, {"R0d": A["o1"]})
    body = json.dumps({k: v for k, v in rec.items() if k != "record_sha256"}, sort_keys=True, separators=(",", ":"), default=str)
    return (rec["record_sha256"] == hashlib.sha256(body.encode()).hexdigest() and any(p.endswith("rlval_B_vs_ownA.json") for p in rec["inputs_sha256"]) and all(len(v or "") == 64 for v in rec["inputs_sha256"].values())
            and any(p.endswith("PREREG_SAFEOBJ.addendum1.md") for p in rec["inputs_sha256"]) and any(p.endswith("dose.json") for p in rec["inputs_sha256"]))


OLD_IDS = {"A-G1", "A-G2", "A-G3", "A-O1", "A-O2", "A-O3", "A-O4", "A-O5", "A-O6", "A-O7", "A-O8", "A-O9", "A-O10", "A-L1", "A-L2", "A-E1", "A-E2", "A-E3", "A-E4"}
OPEN_IDS = {"A-E3b", "A-G2b", "A-O9b", "A-G3b", "A-E1b", "A-L1b"}


@check("record:lists_ONLY_the_readings_the_addendum_does_not_settle_and_names_what_it_settled")
def _(T, W, A, root, tag):
    rec = RUN(T, {"R0d": A["o1"]})
    ids = {a["id"] for a in rec["prereg_ambiguities"]}
    return ids == OPEN_IDS and not (ids & OLD_IDS) and set(rec["settled_by_addendum"]) == OLD_IDS and all(a["text"] for a in rec["prereg_ambiguities"]) and len(rec["prereg_ambiguities"]) == len(OPEN_IDS)


@check("record:every_record_carries_the_addendum_name_and_its_sha256")
def _(T, W, A, root, tag):
    rec = RUN(T, {"R0d": A["o1"], "R1d": A["r1d_o1"]})
    sha = hashlib.sha256(ADD.read_bytes()).hexdigest()
    return (rec["addendum"] == "PREREG_SAFEOBJ.addendum1.md" and rec["addendum_sha256"] == sha == T.ADDENDUM_SHA256 and rec["addendum_file"]["sha256_registered"] == sha
            and all(a["addendum"] == "PREREG_SAFEOBJ.addendum1.md" and a["addendum_sha256"] == sha for a in rec["arms"].values()))


@check("refuse:an_addendum_that_is_not_the_registered_bytes_is_REFUSED_and_the_registered_one_is_accepted")
def _(T, W, A, root, tag):
    bad = root / f"addendum_edited_{tag}.md"
    bad.write_bytes(ADD.read_bytes() + b"\n")
    one = root / f"addendum_one_char_{tag}.md"
    one.write_bytes(ADD.read_bytes().replace(b"stays **+0.60 exactly**", b"stays **+0.61 exactly**"))
    out = []
    for path in (bad, one, root / f"no_such_addendum_{tag}.md", None):
        try:
            T.run({"R0d": str(A["o1"])}, dose_path=str(DOSE), addendum_path=None if path is None else str(path))
            out.append(False)
        except T.AddendumRefused:
            out.append(True)
    rec = RUN(T, {"R0d": A["o1"]})
    return all(out) and len(out) == 4 and rec["addendum_sha256"] == T.ADDENDUM_SHA256 and rec["arms"]["R0d"]["outcome"] == "O1"


@check("record:a_wrong_prereg_file_is_a_problem")
def _(T, W, A, root, tag):
    p = root / f"fake_prereg_{tag}.md"
    p.write_text("not the registered text", encoding="utf-8")
    bad = RUN(T, {"R0d": A["o1"]}, prereg_path=str(p))
    good = RUN(T, {"R0d": A["o1"]}, prereg_path=str(PREREG))
    return any("not the registered" in x for x in bad["problems"]) and not any("not the registered" in x for x in good["problems"]) and good["prereg_sha256_checked"] == T.PREREG_SHA256


@check("record:an_unusable_dose_json_is_a_problem_and_NOT_READ_not_a_crash_and_not_a_silent_fallback")
def _(T, W, A, root, tag):
    p = root / f"bad_dose_{tag}.json"
    p.write_text("{", encoding="utf-8")
    rec = RUN(T, {"R0d": A["o1"]}, dose_path=str(p))
    rec2 = RUN(T, {"R0d": A["o1"]}, dose_path=str(root / "no_dose.json"))
    rec3 = RUN(T, {"R0d": A["o1"]}, dose_path="")
    return (all(r["arms"]["R0d"]["outcome"] == "NOT_READ" and r["arms"]["R0d"]["G"]["verdict"] == "NOT_READ" for r in (rec, rec2, rec3)) and any("not the pinned" in x for x in rec["problems"])
            and any("not found" in x for x in rec2["problems"]) and any("no dose.json given" in x for x in rec3["problems"]))


@check("record:the_tool_never_raises_on_a_garbage_arm_directory")
def _(T, W, A, root, tag):
    d = root / f"garbage_{tag}"
    d.mkdir()
    (d / "sft.jsonl").write_text('{"event": "done"}\n', encoding="utf-8")
    (d / "KILLED.json").write_text("[]", encoding="utf-8")
    r = T.read_constrained("R0d", str(d), T.Ctx(), None)
    return r["outcome"] in ("NOT_READ", "NO_ROW", "O7") and r["outcome"] not in ("O1", "O2", "O3")


# ---- pure classification: the table of section 6 with addendum R2's fixed precedence as a truth table (no arm involved)
def facts(**kw):
    base = {"s_pass": False, "g_lb_pos": None, "g_point_ok": None, "breached": False, "violation_ok": None, "lam_pos_share_ok": None, "cap_hit": None, "lam_zero": False, "other_kill": False, "g3_alone": False}
    base.update(kw)
    return base


@check("table:truth_table_of_section_6_with_the_addendum_precedence")
def _(T, W, A, root, tag):
    c = T.classify
    S = dict(s_pass=True, g_lb_pos=True, g_point_ok=True)
    BR = dict(breached=True)
    cases = [
        # ---- not breached: O1 / O2 / O3 exhaustive and exclusive given S
        (facts(**S), "O1", []), (facts(s_pass=True, g_lb_pos=True, g_point_ok=False), "O2", []), (facts(s_pass=True, g_lb_pos=False, g_point_ok=False), "O3", []),
        (facts(s_pass=True, g_lb_pos=False, g_point_ok=True), "O3", []), (facts(s_pass=True, g_lb_pos=False, g_point_ok=None), "O3", []),            # Kleene: a known False settles O3 beside an unknown
        # ---- breached: exactly one of O4 / O5, both, neither (SLACK)
        (facts(**BR, violation_ok=True, lam_pos_share_ok=True, cap_hit=False), "O4", []), (facts(**BR, violation_ok=False, lam_pos_share_ok=True, cap_hit=False), "O5", []),
        (facts(**BR, violation_ok=True, lam_pos_share_ok=False, cap_hit=True), "O5", []), (facts(**BR, violation_ok=True, lam_pos_share_ok=True, cap_hit=True), "O4+O5", []),
        (facts(**BR, violation_ok=True, lam_pos_share_ok=False, cap_hit=False), "O4", ["SLACK"]), (facts(**BR, violation_ok=False, lam_pos_share_ok=False, cap_hit=False), "O5", []),
        (facts(**BR, violation_ok=False, lam_pos_share_ok=None, cap_hit=None), "O5", []),              # a known violation settles O5; O4 is False
        # ---- the precedence: O6 > O7 > O4/O5 > O1/O2/O3
        (facts(**BR, violation_ok=True, lam_pos_share_ok=False, cap_hit=False, lam_zero=True), "O6", []),                              # O6 beats the SLACK O4
        (facts(**S, lam_zero=True), "O6", []), (facts(other_kill=True, lam_zero=True), "O6", []), (facts(g3_alone=True, lam_zero=True), "O6", []),
        (facts(other_kill=True), "O7", []), (facts(g3_alone=True), "O7", []),
        (facts(**BR, violation_ok=True, lam_pos_share_ok=True, cap_hit=False, other_kill=True), "O7", []),                              # O7 beats O4
        (facts(**BR, violation_ok=False, lam_pos_share_ok=True, cap_hit=True, g3_alone=True), "O7", []),                                # ... and O5, and O4+O5
        (facts(**BR, violation_ok=True, lam_pos_share_ok=True, cap_hit=True, other_kill=True), "O7", []),
        # ---- three-valued: an unknown that could change the answer reads NOT_READ
        (facts(lam_zero=None, other_kill=True), "NOT_READ", []),                                                                        # O6 outranks O7: an unreadable O6 cannot be skipped
        (facts(lam_zero=None, **S), "NOT_READ", []), (facts(other_kill=None), "NOT_READ", []), (facts(g3_alone=None), "NOT_READ", []),
        (facts(breached=None), "NOT_READ", []), (facts(s_pass=True, g_lb_pos=None, g_point_ok=None), "NOT_READ", []), (facts(s_pass=True, g_lb_pos=True, g_point_ok=None), "NOT_READ", []),
        (facts(**BR, violation_ok=None, lam_pos_share_ok=None, cap_hit=True), "NOT_READ", []),                                          # O5 holds but O4 is open: O5 or O4+O5
        (facts(**BR, violation_ok=True, lam_pos_share_ok=False, cap_hit=None), "NOT_READ", []),                                         # O5 open: O4 SLACK or O5
        (facts(**BR, violation_ok=True, lam_pos_share_ok=True, cap_hit=None), "NOT_READ", []),
        (facts(**BR, violation_ok=None, lam_pos_share_ok=True, cap_hit=False), "NOT_READ", []),
        # ---- inconsistent facts are a TOOL defect, never an outcome
        (facts(), "NO_ROW", []),                                                                                                       # not breached, S not PASS, nothing else: no row
    ]
    bad = [(i, c(f)["outcome"], c(f)["flags"], want, wf) for i, (f, want, wf) in enumerate(cases) if (c(f)["outcome"], c(f)["flags"]) != (want, wf)]
    assert not bad, bad
    d = c(facts(**S, lam_zero=True))
    assert d["decided_at"] == "O6" and d["shadowed"] == ["O1"] and d["precedence"] == ["O6", "O7", "O45", "O123"], d
    d = c(facts(**BR, violation_ok=True, lam_pos_share_ok=True, cap_hit=False, g3_alone=True))
    assert d["decided_at"] == "O7" and d["shadowed"] == ["O4"], d
    return True


# ============================================================ PINS: every committed threshold against the prereg TEXT, the ADDENDUM text and their sha256, and the trainer / producers it is read from
def prereg_text():
    return PREREG.read_text(encoding="utf-8")


def addendum_text():
    """the addendum with every run of whitespace collapsed (a phrase may span a line break)."""
    return " ".join(ADD.read_text(encoding="utf-8").split())


def _one(rx, txt, flags=0):
    m = re.findall(rx, txt, flags)
    assert len(m) == 1, (rx, m)
    return m[0]


@check("pin:the_prereg_file_is_the_registered_bytes")
def _(T, W, A, root, tag):
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    return sha == T.PREREG_SHA256 == (PREREG.parent / "PREREG_SAFEOBJ.sha256").read_text(encoding="utf-8").split()[0]


@check("pin:the_addendum_is_the_registered_bytes_and_names_the_prereg_it_amends")
def _(T, W, A, root, tag):
    sha = hashlib.sha256(ADD.read_bytes()).hexdigest()
    txt = addendum_text()
    return (sha == T.ADDENDUM_SHA256 == (ADD.parent / "PREREG_SAFEOBJ.addendum1.sha256").read_text(encoding="utf-8").split()[0] and T.ADDENDUM_NAME == ADD.name == "PREREG_SAFEOBJ.addendum1.md"
            and f"sha256 `{T.PREREG_SHA256[:8]}…`" in txt and "R0d has NOT been launched" in txt)


@check("pin:G_vertices_floor_lower_bound_and_point_rule_against_the_prereg_text")
def _(T, W, A, root, tag):
    txt = prereg_text()
    line = _one(r"(F is the linear interpolation of dose\.json's Pareto points[^\n]*)", txt)
    pts = tuple((float(a), float(b)) for a, b in re.findall(r"\((-\d\.\d+), \+(\d\.\d+)\)", line))
    assert pts == tuple(T.G_POINTS_PRINTED), (pts, T.G_POINTS_PRINTED)
    assert f"G(t) = max(+{T.G_FLOOR:.2f}, F(t))" in txt and "held flat beyond -0.47" in txt
    assert "The PDMS lower 95 % bound is > 0" in txt and T.G_LOWER_BOUND_MIN == 0.0 and "the PDMS point is ≥ G(t)" in txt
    assert "the bar is +0.60 at TTC ≥ -0.03, +0.71 at -0.27, and +0.84 at the bound" in txt
    return True


@check("pin:addendum_R1_dose_json_sha256_alphas_floor_comparison_estimator_and_the_pool")
def _(T, W, A, root, tag):
    txt = addendum_text()
    assert "sha256 `735a6d909bc0ef41…`" in txt and T.DOSE_SHA256.startswith("735a6d909bc0ef41") and hashlib.sha256(DOSE.read_bytes()).hexdigest() == T.DOSE_SHA256
    assert "key `full[alpha].{TTC,PDMS}.mean` (×100)" in txt and T.DOSE_ALPHAS == ("0.25", "0.375", "0.5")
    assert "F uses dose.json's own Pareto points at full precision" in txt and "F is held flat beyond the last vertex" in txt
    assert "stays **+0.60 exactly**" in txt and T.G_FLOOR == 0.60 and "The comparison stays **≥** as written" in txt
    assert "10,000 resamples, seed 20260927" in txt and (T.N_BOOT, T.SEED) == (10000, 20260927)
    assert "(3,000 sets / 75 logs)" in txt and (T.RLVAL_SETS, T.RLVAL_LOGS) == (3000, 75)
    assert "`--base A`" in txt and "t** = that read's TTC point delta" in txt
    assert "reported beside G as a disclosure" in txt
    return True


@check("pin:addendum_R2_precedence_order_O4_O5_overlap_and_gap_the_K3_rule_and_g3_alone")
def _(T, W, A, root, tag):
    txt = addendum_text()
    pos = [txt.index(x) for x in ("1. **O6 INERT.**", "2. **O7 OTHER KILL / g3 ALONE.**", "3. **Breached**", "4. **Not breached**")]
    assert pos == sorted(pos) and T.PRECEDENCE == ("O6", "O7", "O45", "O123") and tuple(T.STAGES) == ("O6", "O7", "O45", "O123"), (pos, T.PRECEDENCE)
    assert "λ_TTC = λ_NC = 0.0 at every logged update" in txt and "**wins over every other row**" in txt
    assert "Any kill whose recorded rule is not K3 (K2, K4, K6, or any other code) is O7." in txt
    assert "(`K-MON`) **applying the K3 rule** counts as **K3**, judged by its recorded reason, not by its code" in txt
    assert "no TTC or NC g5 breach at any check" in txt and "any K3 kill was triggered by g3 breaches only" in txt and "This is O7, not O4/O5" in txt
    assert "(K3 kill, or S fails with at least one g5 breach)" in txt
    assert "**both match** (held in-sample, λ > 0 on ≥ 50 %, AND λ reached λ_max) → **O4+O5**" in txt
    assert "**neither matches** (held in-sample, λ > 0 on < 50 %, never at λ_max) → **O4, flagged SLACK**" in txt
    assert "(S holds, no kill): O1 if G holds; O2 if the PDMS lower bound is > 0 and the point is < G(t); O3 if the lower bound is ≤ 0" in txt
    return True


@check("pin:addendum_R3_signed_window_mean_the_window_the_ramp_and_the_through_the_first_breach_counts")
def _(T, W, A, root, tag):
    txt = addendum_text()
    assert "the **signed** mean over the window of the per-update constraint value g_k that the dual step consumes" in txt
    assert "**Held** ⇔ both terms' window means are ≤ 0.001" in txt and T.O4_VIOLATION_MAX == 0.001 and "The tool names the logged column it reads" in txt
    assert "the 100 logged updates immediately before the first breached check" in txt and T.O4_WINDOW_UPDATES == 100
    assert "For a K3 kill, that is the first of the two consecutive breaches" in txt and "the overlap is disclosed as a flag; the window does not move" in txt
    assert "post-ramp = updates with λ_pg = 1, counted from the end of the ramp **through the first breached check**" in txt
    assert "λ_TTC or λ_NC ≥ λ_max at any update from 0 through the first breached check" in txt
    assert "through the last logged update. A killed arm's unlogged tail (fewer than 25 updates) is disclosed" in txt and T.LOG_EVERY == 25
    assert "on ≥ 50 % of post-ramp" in prereg_text() and T.O4_LAM_POS_SHARE_MIN == 0.5
    return True


@check("pin:addendum_R4_L_on_TTC_only_E_pooled_jointly_clause_4_per_pair_and_the_conservative_seed_floor")
def _(T, W, A, root, tag):
    txt = addendum_text()
    assert "**L is decided on TTC only.**" in txt and "reported as the lever's cost" in txt
    assert "it reads NOT_APPLICABLE: no lever claim" in txt
    assert "on the same 12,146 tokens" in txt and "resampling the 136 navtest logs **jointly** for both pairs" in txt and (T.NAVTEST_TOKENS, T.NAVTEST_LOGS) == (12146, 136)
    assert "12,146 tokens, 136 logs" in prereg_text()
    assert "**per pair**. For EACH pair, the TTC and NC upper 95 % bounds are ≥ 0: four bounds" in txt
    assert "the larger of §7's RL-val training-seed floor and the navtest seed floor |(R0d B − A) − (R1d B − A)|" in txt and "That is the conservative reading" in txt
    return True


@check("pin:O_table_thresholds_against_the_prereg_text")
def _(T, W, A, root, tag):
    txt = prereg_text()
    row4 = _one(r"(\| \*\*O4 BREACHED[^\n]*)", txt)
    assert float(_one(r"is ≤ (0\.\d+),", row4)) == T.O4_VIOLATION_MAX and int(_one(r"averaged over the (\d+) updates", row4)) == T.O4_WINDOW_UPDATES
    assert float(_one(r"on ≥ (\d+) % of post-ramp", row4)) / 100.0 == T.O4_LAM_POS_SHARE_MIN
    row5 = _one(r"(\| \*\*O5 BREACHED[^\n]*)", txt)
    assert "λ reached λ_max OR the in-sample violation over that window is > 0.001" in row5 and T.O4_VIOLATION_MAX == 0.001
    assert "λ_TTC = λ_NC = 0.0 at every update" in _one(r"(\| \*\*O6 INERT[^\n]*)", txt)
    row7 = _one(r"(\| \*\*O7 OTHER KILL[^\n]*)", txt)
    assert tuple(re.findall(r"K\d", row7.split("|")[2])) == ("K2", "K4", "K6") and "g3 alone" in row7, row7
    assert re.findall(r"^\| \*\*O(\d) ", txt, re.M) == [str(i) for i in range(1, 8)]
    assert "R0d: S and G" in _one(r"(\| \*\*O1 SAFE[^\n]*)", txt) and "the PDMS lower bound is > 0 but the point is below G(t)" in _one(r"(\| \*\*O2 SAFE[^\n]*)", txt)
    assert "the PDMS lower bound is ≤ 0" in _one(r"(\| \*\*O3 SAFE[^\n]*)", txt)
    assert "If R0d is O1 and R1d is not, the result is **NOT REPLICATED**" in txt
    assert tuple(T.DETERMINATE) == ("O1", "O2", "O3", "O4", "O5", "O6", "O7", "O4+O5")
    return True


@check("pin:registered_dual_values_and_ramp_against_the_prereg_text_and_the_trainer")
def _(T, W, A, root, tag):
    txt = prereg_text()
    eps = float(_one(r"\| eps \| \*\*(\d+\.\d+)\*\*", txt))
    eta = float(_one(r"\| dual step eta \| \*\*(\d+\.\d+)\*\*", txt))
    lmax = float(_one(r"\| λ_max \| \*\*(\d+\.\d+)\*\*", txt))
    lam0 = float(_one(r"\| λ0 \| \*\*(\d+\.\d+)\*\*", txt))
    reg = T.REGISTERED_EPO_CON
    assert (reg["eps"], reg["lr"], reg["max"], reg["lam0"]) == (eps, eta, lmax, lam0) and T.LAM_MAX == lmax
    assert W.sf.EpoDual(("ttc", "nc"), eps, eta, lmax, lam0, temp=T.REGISTERED_CON_TEMP).built_facts() == {**reg, "temp": T.REGISTERED_CON_TEMP}      # the REAL dual's BUILT facts are the registered ones (+ PICKCON's `temp`)
    src = SF_PATH.read_text(encoding="utf-8")
    assert int(_one(r'"--epo-ramp", type=int, default=(\d+)', src)) == T.REGISTERED_RAMP
    t_prereg = float(_one(r"`epo_policy` at T = (\d+\.\d+)", txt))                                                       # PICKCON: the registered constraint T, from the prereg text, the trainer's --epo-temp default and the test
    assert T.REGISTERED_CON_TEMP == EPO_TEMP == t_prereg == float(_one(r'"--epo-temp", type=float, default=(\d+\.\d+)', src))
    arms = PKGROOT / "eval" / "rls1d_arms.json"
    if arms.exists():
        assert "--epo-ramp" not in arms.read_text(encoding="utf-8")                                                      # the registered argv does not name it: the BUILT ramp is the default
    return True


@check("pin:L_E_floors_thresholds_and_estimator_against_the_prereg_text")
def _(T, W, A, root, tag):
    txt = prereg_text()
    assert "is > 2 × F_seed" in txt and T.L_FLOOR_MULT == 2.0
    assert "each ≥ +0.25 and their mean ≥ +0.5" in txt and (T.E_EACH_MIN, T.E_MEAN_MIN) == (0.25, 0.5)
    assert "the mean contrast is > 2 × max(seed floor, I-rep |Δ|)" in txt and T.E_FLOOR_MULT == 2.0
    assert "REFUTED iff the pooled upper bound is < +0.25" in txt and T.E_REFUTED_UPPER == 0.25
    assert "the pooled lower bound is > 0" in txt and "the TTC and NC upper bounds are ≥ 0" in txt
    assert "paired log-cluster bootstrap, **10,000 resamples, seed 20260927**" in txt and (T.N_BOOT, T.SEED) == (10000, 20260927) == (RV.N_BOOT, RV.SEED)
    assert "F_seed = max(|ΔTTC(R0 vs R1o)|, |ΔTTC(R0d vs R1d)|)" in txt
    return True


@check("pin:dose_json_vertices_are_the_printed_ones_rounded")
def _(T, W, A, root, tag):
    return [tuple(round(v, 2) for v in p) for p in T.dose_points(jload(DOSE))] == [tuple(p) for p in T.G_POINTS_PRINTED]


@check("pin:trainer_log_cadence_columns_and_counter_semantics")
def _(T, W, A, root, tag):
    src = SF_PATH.read_text(encoding="utf-8")
    assert src.count("if upd % 25 == 0:") == 1 and T.LOG_EVERY == 25
    blk = src[src.index('row.update({"con": cm_("con")'):]
    blk = blk[:blk.index("con_log = {\"n\": 0}")]
    cols = set(re.findall(r'"(con[A-Za-z_0-9]*)"\s*:', blk))
    sim = sim_log(W, 25, *HELD)["trains"][0]
    assert cols == {k for k in sim if k.startswith("con")}, (sorted(cols), sorted(k for k in sim if k.startswith("con")))
    assert set(T.CON_COLS) <= cols
    # the counters the tool's arithmetic assumes (O4's post-ramp share, O5's cap, O6's 'at every update')
    assert "if any(v > 0.0 for v in self.lam.values()):\n            self.pos_updates += 1" in src
    assert "if raw >= self.lam_max:\n                self.cap_hits += 1" in src and "self.lam_seen_max[c] = max(self.lam_seen_max[c], self.lam[c])" in src
    assert '"con_lam_ttc": dual.lam["ttc"]' in src
    # the update numbering the post-ramp count assumes: update index u uses lam_pg = min(1, u / ramp); the row at update number U has run indices 0 .. U-1
    assert "return 1.0 if ramp <= 0 else min(1.0, upd / ramp)" in src and "dual.step(con_acc)" in src
    return True


@check("pin:the_trainer_loop_lines_sim_log_copies")
def _(T, W, A, root, tag):
    src = SF_PATH.read_text(encoding="utf-8")
    return all(x in src for x in ("con_acc[k_] = con_acc.get(k_, 0.0) + v_", "dual.step(con_acc)", "con_log[k_] = con_log.get(k_, 0.0) + v_", "cm_ = lambda k_: con_log.get(k_, 0.0) / nc_",
                                  'log({"event": "done", "updates": upd,'))


@check("pin:K3_reason_text_is_the_real_monitors_and_the_chains_K_MON_reasons_are_not_the_K3_rule")
def _(T, W, A, root, tag):
    r3 = lambda tag_: {"tag": tag_, "g3_ok": True, "g5_ok": False, "k4_breach": False, "k5_breach": False}          # noqa: E731  (a g5 breach)
    r4 = lambda tag_: {"tag": tag_, "g3_ok": True, "g5_ok": True, "k4_breach": True, "k5_breach": False}          # noqa: E731  (a PDMS drop)
    code, why = RC.judge("R0d", r3("update500"), [r3("update250")])                   # the REAL monitor's own K3 / K4 reason texts
    code4, why4 = RC.judge("R0d", r4("update500"), [r4("update250")])
    pod = (PKGROOT / "refe" / "rls1_pod.sh").read_text(encoding="utf-8")
    return (code == "K3" and T.kill_rule("K-MON", why) == "K3" and T.kill_rule("K3", why) == "K3" and T.kill_rule("K3", "") == "K3"
            and code4 == "K4" and T.kill_rule("K4", why4) == "K4" and T.kill_rule("K-MON", why4) == "K-MON"
            and 'why="kill rules unenforceable: monitor died rc $MON_RC"' in pod and 'why="kill rules unenforceable: monitor not READY within ${ready_s}s"' in pod
            and T.kill_rule("K-MON", "kill rules unenforceable: monitor died rc 137") == "K-MON" and T.kill_rule("K-MON", "kill rules unenforceable: monitor not READY within 600s") == "K-MON"
            and T.kill_rule("K6", "more than 6.0 h on the A40") == "K6" and T.kill_rule("K2", "non-finite loss at update 300") == "K2" and T.kill_rule("K-MON", None) == "K-MON")


@check("pin:file_names_against_the_chain")
def _(T, W, A, root, tag):
    rc = (PKGROOT / "refe" / "rls1_checks.py").read_text(encoding="utf-8")
    return ('kill("K2"' in rc and 'kill("K6"' in rc and 'return "K3"' in rc and 'return "K4"' in rc and T.S_NAME == SR.OUT_NAME == "safeobj_S.json" and '"safeobj_S.json"' in rc
            and T.RLVAL_NAME == "rlval_vs_A0.json" and "rlval_vs_A0.json" in (PKGROOT / "refe" / "rls1_pod.sh").read_text(encoding="utf-8") and T.SR.CHECK_TAGS == TAGS and T.BA_NAME == "rlval_B_vs_ownA.json")


@check("pin:the_reader_symbols_the_tool_imports_exist")
def _(T, W, A, root, tag):
    return all(hasattr(SR, n) for n in ("judge_check", "_rows", "CHECK_TAGS", "read_arm", "write_record", "sha256_file", "OUT_NAME")) and T.SR is SR


@check("pin:the_chain_does_not_write_a_B_minus_ownA_interval_so_the_extra_rl_val_eval_call_is_needed")
def _(T, W, A, root, tag):
    pod = (PKGROOT / "refe" / "rls1_pod.sh").read_text(encoding="utf-8")
    return "rlval_B_vs_ownA" not in pod and "--scores A0=$SC/scores_update1000_base.npz CTRL=$SC/scores_update1000_A.npz R=$SC/scores_update1000_B.npz --base A0" in pod


# ============================================================ a REAL main() run writes a real sft.jsonl: the tool must read its data / train / done rows
def real_run(W, root):
    if not hasattr(W, "_real"):
        import test_rl_epo as TE
        td = root / "real_main"
        td.mkdir()
        W._real = TE.smoke(SF_PATH.read_text(encoding="utf-8"), str(td), epo=True, epo_lam=1.0, native7=True, terms=True, tag="oc_real",
                           extra=("--epo-con", "g5", "--epo-con-eps", "0", "--epo-con-lr", "4", "--epo-con-max", "10", "--epo-con-lam0", "0.5"))
        W._real_path = td / "out" / "sft.jsonl"
    return W._real, W._real_path


@check("real_main:read_log_reads_the_real_trainers_data_train_and_done_rows")
def _(T, W, A, root, tag):
    res, p = real_run(W, root)
    assert res["run"] == 0 and p.exists()
    reg = {**T.REGISTERED_EPO_CON, "lam0": 0.5}                                           # the argv of this run (non-circular: the block comes from the argv, the run's block must equal it)
    lg = T.read_log(str(p), 25, T.Ctx(), registered=reg, registered_ramp=2)                                          # c = 25: the only row there is
    assert lg["readable"] and lg["problems"] == [], lg["problems"]
    n = lg["numbers"]
    assert n["last_logged_update"]["value"] == 25 and n["ramp"]["value"] == 2
    assert lg["lam_zero"] is False and lg["cap_hit"] is False and n["lam_pos_updates_last"]["value"] == 25          # lam0 0.5 > 0 at the start of every one of the 25 updates
    assert n["lam_seen_max_ttc_last"]["value"] >= 0.5
    assert any("ramp end" in x for x in lg["notes"]) and lg["lam_pos_share_ok"] is None                               # no train row AT update 2: the share is unreadable, said so
    return True


@check("real_main:the_real_trainers_epo_con_block_carries_temp_equal_to_the_registered_T_and_to_epo_temp")
def _(T, W, A, root, tag):
    res, p = real_run(W, root)
    d = next(json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if '"event": "data"' in ln)
    return d["epo_con"].get("temp") == T.REGISTERED_CON_TEMP == d["epo"]["temp"] and {k for k in d["epo_con"] if k != "temp"} == set(T.REGISTERED_EPO_CON)       # --epo-con-temp unset: the constraint inherits --epo-temp


@check("real_main:the_real_run_is_refused_under_the_registered_values")
def _(T, W, A, root, tag):
    res, p = real_run(W, root)
    lg = T.read_log(str(p), None, T.Ctx())                                                                         # registered lam0 0.0 / ramp 200: this run is neither
    return lg["readable"] is False and any("not the registered" in x for x in lg["problems"]) and any("--epo-ramp" in x for x in lg["problems"])


@check("real_main:the_real_done_events_state_matches_the_last_row_and_a_tampered_one_is_caught")
def _(T, W, A, root, tag):
    res, p = real_run(W, root)
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    done = [r for r in rows if r.get("event") == "done"]
    last = [r for r in rows if r.get("event") == "train"][-1]
    assert done and done[0]["epo_con"]["pos_updates"] == last["con_lam_pos_updates"] and done[0]["epo_con"]["cap_hits"] == last["con_cap_hits"] and done[0]["updates"] == last["update"]
    q = root / f"real_tamper_{tag}.jsonl"
    q.write_text("".join(json.dumps({**r, "epo_con": {**r["epo_con"], "pos_updates": 3}} if r.get("event") == "done" else r) + "\n" for r in rows), encoding="utf-8")
    lg = T.read_log(str(q), None, T.Ctx(), registered={**T.REGISTERED_EPO_CON, "lam0": 0.5}, registered_ramp=2)
    return lg["readable"] is False and any("`done` event" in x for x in lg["problems"])


# ============================================================ the CLI, end to end
def cli_checks(tpath, A, root, tag):
    chk = []

    def run(*args):
        return subprocess.run([sys.executable, "-X", "utf8", str(tpath), *args], capture_output=True, text=True, timeout=300)
    out = root / f"cli_{tag}"
    out.mkdir(exist_ok=True)
    extras = ["--dose", str(DOSE), "--prereg", str(PREREG), "--addendum", str(ADD)]
    add_sha = hashlib.sha256(ADD.read_bytes()).hexdigest()
    r = run("--arm", f"R0d={A['o1']}", "--arm", f"R1d={A['r1d_o1']}", "--arm", f"R0={A['c_k4']}", "--arm", f"R1o={A['c_k4']}", "--out", str(out / "a.json"), *extras)
    f = out / "a.json"
    rec = jload(f) if f.exists() else {}
    chk.append(("rc0_when_every_constrained_arm_has_a_determinate_outcome", r.returncode == 0))
    chk.append(("prints_the_outcome_joint_and_L_lines", all(x in r.stdout for x in ("ZZSAFEOBJ_OUTCOME R0d O1", "ZZSAFEOBJ_OUTCOME R1d O1", "ZZSAFEOBJ_JOINT REPLICATED_O1_BOTH", "ZZSAFEOBJ_BAR L PASS", "ZZSAFEOBJ_BAR E PENDING"))))
    chk.append(("prints_the_addendum_sha256_on_the_file_line", f"addendum_sha256={add_sha}" in r.stdout and "ZZSAFEOBJ_FILE" in r.stdout))
    chk.append(("writes_the_record_with_a_self_consistent_hash", bool(rec) and rec["record_sha256"] == hashlib.sha256(json.dumps({k: v for k, v in rec.items() if k != "record_sha256"}, sort_keys=True,
                                                                                                                       separators=(",", ":"), default=str).encode()).hexdigest()
                                                                  and rec["arms"]["R0d"]["outcome"] == "O1" and rec["L"]["verdict"] == "PASS" and rec["E"]["verdict"] == "PENDING" and rec["addendum_sha256"] == add_sha))
    r = run("--arm", f"R0d={A['o5_cap_o4_tie']}", "--out", str(out / "b.json"), *extras)
    chk.append(("rc0_and_the_O4+O5_line_for_the_overlap", r.returncode == 0 and "ZZSAFEOBJ_OUTCOME R0d O4+O5" in r.stdout and (out / "b.json").exists()))
    r = run("--arm", f"R0d={A['gap']}", "--out", str(out / "b2.json"), *extras)
    chk.append(("rc0_and_the_SLACK_flag_is_printed", r.returncode == 0 and "ZZSAFEOBJ_OUTCOME R0d O4 [SLACK]" in r.stdout))
    r = run("--arm", f"R0d={A['o3_o6_tie']}", "--out", str(out / "b3.json"), *extras)
    chk.append(("rc0_and_O6_beats_O3", r.returncode == 0 and "ZZSAFEOBJ_OUTCOME R0d O6" in r.stdout))
    r = run("--arm", f"R0d={A['n_dir_missing']}", "--out", str(out / "c.json"), *extras)
    chk.append(("rc2_for_NOT_READ", r.returncode == 2 and "ZZSAFEOBJ_OUTCOME R0d NOT_READ" in r.stdout))
    chk.append(("rc3_without_a_constrained_arm", run("--arm", f"R1o={A['c_k4']}").returncode == 3 and run().returncode in (2, 3)))
    chk.append(("rc3_for_a_duplicate_arm", run("--arm", f"R0d={A['o1']}", "--arm", f"R0d={A['o1']}", "--out", str(out / "d.json")).returncode == 3))
    chk.append(("rc3_for_an_unknown_arm_name", run("--arm", f"R9={A['o1']}", "--out", str(out / "e.json")).returncode == 3 and run("--arm", "R0d", "--out", str(out / "e.json")).returncode == 3))
    bad = out / "addendum_edited.md"
    bad.write_bytes(ADD.read_bytes() + b" ")
    r = run("--arm", f"R0d={A['o1']}", "--out", str(out / "g.json"), "--dose", str(DOSE), "--prereg", str(PREREG), "--addendum", str(bad))
    chk.append(("rc4_refused_for_a_wrong_addendum_and_no_record_written", r.returncode == 4 and "ZZSAFEOBJ_REFUSED" in r.stdout and not (out / "g.json").exists()))
    r = run("--arm", f"R0d={A['o1']}", "--out", str(out / "g2.json"), "--dose", str(DOSE), "--prereg", str(PREREG))
    chk.append(("rc4_refused_when_no_addendum_is_given_and_none_is_beside_the_tool", r.returncode == 4 and "ZZSAFEOBJ_REFUSED" in r.stdout and not (out / "g2.json").exists()))
    ef = E_file(root, f"cli_{tag}")
    r = run("--arm", f"R0d={A['o1']}", "--arm", f"R1d={A['r1d_o1']}", "--arm", f"R0={A['c_k4']}", "--arm", f"R1o={A['c_k4']}", "--navtest-e", ef, "--out", str(out / "f.json"), *extras)
    rec = jload(out / "f.json") if (out / "f.json").exists() else {}
    chk.append(("navtest_e_file_is_read_end_to_end_ADOPT", r.returncode == 0 and "ZZSAFEOBJ_BAR E ADOPT" in r.stdout and rec.get("E", {}).get("verdict") == "ADOPT"))
    # ---- R0e-pick B3: the arm's registered constraint T on the command line (--con-temp), or from its registered argv (--arms-file / --arm-reg)
    def rj(name):
        return jload(out / name) if (out / name).exists() else {}
    af = out / "arms_con_temp.json"
    af.write_text(json.dumps({"arms": {"R0e-pick": {"argv": ["--epo-con", "g5", "--epo-con-temp", "0.01"]}, "R0e-id": {"argv": ["--epo-con", "g5", "--epo-con-temp", "0.1"]}, "R1o": {"argv": ["--epo-con", "off"]}}}), encoding="utf-8")
    af_sha = hashlib.sha256(af.read_bytes()).hexdigest()
    r = run("--arm", f"R0d={A['o1_t001']}", "--con-temp", "R0d=0.01", "--out", str(out / "t1.json"), *extras)
    rec = rj("t1.json")
    chk.append(("con_temp_cli:--con-temp_R0d=0.01_reads_the_0.01_arm_O1_rc0_and_records_the_registration",
                r.returncode == 0 and "ZZSAFEOBJ_OUTCOME R0d O1" in r.stdout and rec.get("arms", {}).get("R0d", {}).get("registered_con_temp", {}).get("value") == 0.01
                and rec.get("registered", {}).get("epo_con_temp_by_arm", {}).get("R0d", {}).get("source", {}).get("kind") == "cli"))
    r = run("--arm", f"R0d={A['o1_t001']}", "--out", str(out / "t2.json"), *extras)
    chk.append(("con_temp_cli:the_0.01_arm_without_--con-temp_is_NOT_READ_rc2", r.returncode == 2 and "ZZSAFEOBJ_OUTCOME R0d NOT_READ" in r.stdout))
    r = run("--arm", f"R0d={A['o1']}", "--con-temp", "R0d=0.01", "--out", str(out / "t3.json"), *extras)
    chk.append(("con_temp_cli:the_0.1_arm_under_--con-temp_R0d=0.01_is_NOT_READ_rc2", r.returncode == 2 and "ZZSAFEOBJ_OUTCOME R0d NOT_READ" in r.stdout))
    r = run("--arm", f"R0d={A['o1']}", "--con-temp", "R0d=0.1", "--out", str(out / "t4.json"), *extras)
    chk.append(("con_temp_cli:the_0.1_arm_under_--con-temp_R0d=0.1_reads_O1_rc0", r.returncode == 0 and "ZZSAFEOBJ_OUTCOME R0d O1" in r.stdout))
    r = run("--arm", f"R0d={A['o1_t001']}", "--arm", f"R1d={A['r1d_o1']}", "--con-temp", "R0d=0.01", "--out", str(out / "t5.json"), *extras)
    chk.append(("con_temp_cli:R0d_at_0.01_beside_R1d_at_the_default_are_both_read_and_the_joint_replicates",
                r.returncode == 0 and "ZZSAFEOBJ_OUTCOME R0d O1" in r.stdout and "ZZSAFEOBJ_OUTCOME R1d O1" in r.stdout and "ZZSAFEOBJ_JOINT REPLICATED_O1_BOTH" in r.stdout))
    r = run("--arm", f"R0d={A['o1_t001']}", "--arms-file", str(af), "--arm-reg", "R0d=R0e-pick", "--out", str(out / "t6.json"), *extras)
    src = rj("t6.json").get("registered", {}).get("epo_con_temp_by_arm", {}).get("R0d", {}).get("source", {})
    chk.append(("con_temp_cli:--arms-file_--arm-reg_R0d=R0e-pick_registers_0.01_from_the_registered_argv_and_records_the_file_sha256",
                r.returncode == 0 and "ZZSAFEOBJ_OUTCOME R0d O1" in r.stdout and src.get("kind") == "arms_file" and src.get("arm") == "R0e-pick" and src.get("sha256") == af_sha and af_sha in rj("t6.json").get("inputs_sha256", {}).values()))
    r = run("--arm", f"R0d={A['o1_t001']}", "--arms-file", str(af), "--arm-reg", "R0d=R0e-id", "--out", str(out / "t7.json"), *extras)
    chk.append(("con_temp_cli:R0e-id_is_registered_at_0.1_so_the_0.01_log_is_NOT_READ_rc2", r.returncode == 2 and "ZZSAFEOBJ_OUTCOME R0d NOT_READ" in r.stdout))
    r = run("--arm", f"R0d={A['o1_t001']}", "--con-temp", "R0d=0.01", "--arms-file", str(af), "--arm-reg", "R0d=R0e-pick", "--out", str(out / "t8.json"), *extras)
    chk.append(("con_temp_cli:--con-temp_and_the_arms_file_may_agree", r.returncode == 0 and rj("t8.json").get("registered", {}).get("epo_con_temp_by_arm", {}).get("R0d", {}).get("source", {}).get("kind") == "cli+arms_file"))
    bad_usage = {
        "a_non_number": ["--arm", f"R0d={A['o1']}", "--con-temp", "R0d=abc"], "zero": ["--arm", f"R0d={A['o1']}", "--con-temp", "R0d=0"], "negative": ["--arm", f"R0d={A['o1']}", "--con-temp", "R0d=-1"],
        "nan": ["--arm", f"R0d={A['o1']}", "--con-temp", "R0d=nan"], "inf": ["--arm", f"R0d={A['o1']}", "--con-temp", "R0d=inf"], "no_equals": ["--arm", f"R0d={A['o1']}", "--con-temp", "0.01"],
        "an_unconstrained_slot": ["--arm", f"R0d={A['o1']}", "--arm", f"R1o={A['c_k4']}", "--con-temp", "R1o=0.01"], "an_arm_that_is_not_given": ["--arm", f"R0d={A['o1']}", "--con-temp", "R1d=0.01"],
        "twice": ["--arm", f"R0d={A['o1']}", "--con-temp", "R0d=0.01", "--con-temp", "R0d=0.01"],
        "the_constraint_off_arm": ["--arm", f"R0d={A['o1']}", "--arms-file", str(af), "--arm-reg", "R0d=R1o"], "an_absent_arm": ["--arm", f"R0d={A['o1']}", "--arms-file", str(af), "--arm-reg", "R0d=nope"],
        "an_arms_file_without_--arm-reg": ["--arm", f"R0d={A['o1']}", "--arms-file", str(af)], "an_arm-reg_without_--arms-file": ["--arm", f"R0d={A['o1']}", "--arm-reg", "R0d=R0e-pick"],
        "disagreeing_registrations": ["--arm", f"R0d={A['o1_t001']}", "--con-temp", "R0d=0.1", "--arms-file", str(af), "--arm-reg", "R0d=R0e-pick"],
    }
    for nm, args in bad_usage.items():
        rr = run(*args, "--out", str(out / "t9.json"), *extras)
        chk.append((f"con_temp_cli:rc3_usage_error_for_{nm}", rr.returncode == 3 and "usage:" in rr.stdout))
    chk.append(("con_temp_cli:a_usage_error_writes_no_record", not (out / "t9.json").exists()))
    cp = out / "cp" / "R0d"
    shutil.copytree(A["o1"], cp)
    r = run("--arm", f"R0d={cp}", *extras)
    chk.append(("default_out_is_beside_the_first_arm_dir", r.returncode == 0 and (out / "cp" / "safeobj_outcome.json").exists()))
    return chk


# ============================================================ mutations of the tool's source; each must turn the checks it exists for RED
def MU(name, old, new, expect):
    return (name, old, new, set(expect))


TRUTH = "table:truth_table_of_section_6_with_the_addendum_precedence"
PIN_G = "pin:G_vertices_floor_lower_bound_and_point_rule_against_the_prereg_text"
PIN_R1 = "pin:addendum_R1_dose_json_sha256_alphas_floor_comparison_estimator_and_the_pool"
PIN_R2 = "pin:addendum_R2_precedence_order_O4_O5_overlap_and_gap_the_K3_rule_and_g3_alone"
PIN_R3 = "pin:addendum_R3_signed_window_mean_the_window_the_ramp_and_the_through_the_first_breach_counts"
PIN_R4 = "pin:addendum_R4_L_on_TTC_only_E_pooled_jointly_clause_4_per_pair_and_the_conservative_seed_floor"
PIN_O = "pin:O_table_thresholds_against_the_prereg_text"
PIN_K3 = "pin:K3_reason_text_is_the_real_monitors_and_the_chains_K_MON_reasons_are_not_the_K3_rule"
REFUSE = "refuse:an_addendum_that_is_not_the_registered_bytes_is_REFUSED_and_the_registered_one_is_accepted"
VERTEX = ("G:vertex_alpha_0.25_reads_FAIL_by_0.00125", "G:vertex_alpha_0.375_sits_exactly_on_F_and_reads_PASS", "G:vertex_alpha_0.5_sits_exactly_on_F_and_reads_PASS")
SIGNED_OR_SUM = 'out["violation_ok"] = bool(max(g["ttc"], g["nc"]) <= O4_VIOLATION_MAX)'
WINDOW = "need = [c - O4_WINDOW_UPDATES + LOG_EVERY * i for i in range(1, O4_WINDOW_UPDATES // LOG_EVERY + 1)]"
C3_LINE = 'c3 = bool(mean_contrast > E_FLOOR_MULT * floor) if floor is not None else None'
K3_RX_LINE = 'K3_REASON_RX = re.compile(r"g3/g5 bound breached at (update\\d+) and (update\\d+)")'
MUTATIONS = [
    # ---- bar G (addendum R1)
    MU("G floor raised to 0.65", "G_FLOOR = 0.60 ", "G_FLOOR = 0.65 ", {"G:frontier_vertices_and_flat_ends", PIN_G, VERTEX[0]}),
    MU("G floor lowered to 0.50 (the pin and the alpha-0.25 vertex must see it)", "G_FLOOR = 0.60 ", "G_FLOOR = 0.50 ",{PIN_G, VERTEX[0], "G:frontier_vertices_and_flat_ends"}),
    MU("the PRINTED (2-decimal) vertices are used for the bar", 'bar = frontier(t, dose["points"])', "bar = frontier(t, G_POINTS_PRINTED)", {*VERTEX, "G:boundary_arms_equal_to_the_bar_pass_just_below_fails"}),
    MU("the dose.json alphas are the wrong ones", 'DOSE_ALPHAS = ("0.25", "0.375", "0.5")', 'DOSE_ALPHAS = ("0.25", "0.375", "0.625")', {VERTEX[2], PIN_R1, "G:dose_json_points_are_the_addendum_numbers_and_round_to_the_printed_vertices"}),
    MU("dose.json's sha256 is not checked", 'if sha != DOSE_SHA256:\n        out["problems"]', 'if False:\n        out["problems"]',
       {"G:a_dose_json_that_is_not_the_pinned_bytes_is_NOT_READ", "record:an_unusable_dose_json_is_a_problem_and_NOT_READ_not_a_crash_and_not_a_silent_fallback"}),
    MU("the pinned dose.json sha256 is wrong", 'DOSE_SHA256 = "735a6d909bc0ef41', 'DOSE_SHA256 = "835a6d909bc0ef41', {PIN_R1, VERTEX[1], "G:the_record_names_the_vertices_their_file_and_sha256_and_the_read_it_judged"}),
    MU("G point rule strict (> for >=)", "lb_pos, pt_ok = lo > G_LOWER_BOUND_MIN, pt >= bar", "lb_pos, pt_ok = lo > G_LOWER_BOUND_MIN, pt > bar", {"G:boundary_arms_equal_to_the_bar_pass_just_below_fails", VERTEX[1], VERTEX[2]}),
    MU("G lower bound >= 0 passes", "lb_pos, pt_ok = lo > G_LOWER_BOUND_MIN, pt >= bar", "lb_pos, pt_ok = lo >= G_LOWER_BOUND_MIN, pt >= bar", {"G:lower_bound_exactly_zero_fails_a_hair_above_passes", "outcome:o3_flat->O3"}),
    MU("G passes on the lower bound OR the point", 'rec = {"verdict": "PASS" if (lb_pos and pt_ok) else "FAIL"', 'rec = {"verdict": "PASS" if (lb_pos or pt_ok) else "FAIL"',
       {"G:a_point_above_the_bar_with_lower_bound_zero_is_not_a_pass", "G:lower_bound_exactly_zero_fails_a_hair_above_passes", "outcome:o2->O2"}),
    MU("G ignores the arm's TTC delta", '    bar = frontier(t, dose["points"])\n', '    bar = frontier(0.0, dose["points"])\n', {"outcome:o1_ttc->O1", "outcome:o2_ttc->O2", VERTEX[1]}),
    MU("the BA read is not tied to the chain's dumps", 'if not (h_ba.get("A") and h_ba.get("A") == h_r.get("CTRL") and h_ba.get("B") == h_r.get("R")):', "if False:", {"notread:n_ba_other_dumps", "notread:n_ba_other_B"}),
    MU("the BA read is not tied to the dumps on disk", 'if ((d.get("inputs") or {}).get("scores_sha256") or {}).get(name) != ctx.sha(fp):', "if False:", {"notread:n_ba_wrong_both"}),
    MU("the BA read is not cross-checked with the chain's u1000 read", "if abs(ttc_r - vals[3]) > TOL or abs(pdms_r - vals[0]) > TOL:", "if False:", {"notread:n_ba_disagrees", "notread:n_ba_ttc_disagrees"}),
    MU("the BA read's set count is not compared", 'if (d.get("eligibility") or {}).get("sets_used") != (r.get("eligibility") or {}).get("sets_used"):', "if False:", {"notread:n_ba_sets"}),
    MU("the BA estimator is not pinned", 'if f"{N_BOOT} resamples" not in est or f"seed {SEED}" not in est:\n            problems.append(f"{BA_NAME}: the estimator', 'if False:\n            problems.append(f"{BA_NAME}: the estimator', {"notread:n_ba_estimator"}),
    MU("the BA base arm is not checked", 'if d.get("tool") != "rl_val_eval/1" or d.get("rule") != "navsim_v1_lane" or d.get("base") != "A" or set(d.get("arms") or {}) != {"A", "B"}:', "if False:", {"notread:n_ba_base"}),
    MU("a killed arm gets a u1000 read", 'if rec["state"] == "killed":\n        rec["G"]', 'if False:\n        rec["G"]', {"G:killed_arm_has_no_u1000_so_G_is_NOT_READ_and_not_pending"}),
    # ---- the outcome table: the rows (section 6)
    MU("O4 and O5 swapped", '        "O4": and3(f["breached"], f["violation_ok"], f["lam_pos_share_ok"]),\n        "O5": and3(f["breached"], or3(f["cap_hit"], not3(f["violation_ok"]))),',
       '        "O5": and3(f["breached"], f["violation_ok"], f["lam_pos_share_ok"]),\n        "O4": and3(f["breached"], or3(f["cap_hit"], not3(f["violation_ok"]))),', {"outcome:o4->O4", "outcome:o5_k3->O5", TRUTH}),
    MU("O5 drops the cap clause", 'or3(f["cap_hit"], not3(f["violation_ok"]))', 'not3(f["violation_ok"])', {"outcome:o5_cap_o4_tie->O4+O5", TRUTH}),
    MU("O5 drops the violation clause", 'or3(f["cap_hit"], not3(f["violation_ok"]))', 'f["cap_hit"]', {"outcome:o5_lag_k3->O5", "outcome:viol_over->O5", TRUTH}),
    MU("O4 drops the lambda-share clause", 'and3(f["breached"], f["violation_ok"], f["lam_pos_share_ok"])', 'and3(f["breached"], f["violation_ok"])', {"outcome:gap_share_below->O4", "outcome:gap->O4", TRUTH}),
    MU("O4 drops the violation clause", 'and3(f["breached"], f["violation_ok"], f["lam_pos_share_ok"])', 'and3(f["breached"], f["lam_pos_share_ok"])', {"outcome:viol_over->O5", TRUTH}),
    MU("O6 never matches", '"O6": f["lam_zero"],', '"O6": False,', {"outcome:o6_k3->O6", "outcome:o3_o6_tie->O6", "outcome:o1_inert->O6"}),
    MU("O7 loses 'g3 alone'", '"O7": or3(f["other_kill"], f["g3_alone"]),', '"O7": f["other_kill"],', {"outcome:o7_g3->O7", "outcome:o7_g3_o4->O7", TRUTH}),
    MU("a kill whose rule is not K3 is ignored", 'other_kill = True if (rules - {"K3"}) else (None if kill_unknown else False)', "other_kill = False",
       {"outcome:o7_k6->O7", "outcome:o7_k4->O7", "outcome:o7_k2->O7", "outcome:kmon_monitor_died->O7", "outcome:o7_beats_o4->O7"}),
    MU("'g3 alone' counts a g3 breach beside a g5 breach", 'g3_alone = (("g3" in kinds) and ("g5" not in kinds)) if kinds else None', 'g3_alone = ("g3" in kinds) if kinds else None', {"outcome:o7_g3g5->O4"}),
    # ---- the fixed precedence (addendum R2): each ordering that is NOT the registered one must go RED where it matters
    MU("precedence reordered: O7 before O6", 'PRECEDENCE = ("O6", "O7", "O45", "O123")', 'PRECEDENCE = ("O7", "O6", "O45", "O123")', {"outcome:o6_beats_o7->O6", TRUTH, PIN_R2}),
    MU("precedence reordered: O4/O5 before O6 and O7", 'PRECEDENCE = ("O6", "O7", "O45", "O123")', 'PRECEDENCE = ("O45", "O6", "O7", "O123")', {"outcome:o6_breach->O6", "outcome:o7_beats_o4->O7", TRUTH, PIN_R2}),
    MU("precedence reordered: O1/O2/O3 before the rest", 'PRECEDENCE = ("O6", "O7", "O45", "O123")', 'PRECEDENCE = ("O123", "O6", "O7", "O45")', {"outcome:o1_inert->O6", "outcome:o3_o6_tie->O6", TRUTH, PIN_R2}),
    MU("precedence reordered: O4/O5 before O7", 'PRECEDENCE = ("O6", "O7", "O45", "O123")', 'PRECEDENCE = ("O6", "O45", "O7", "O123")', {"outcome:o7_beats_o4->O7", TRUTH, PIN_R2}),
    MU("precedence reordered: O1/O2/O3 before O4/O5 (an equivalent mutation: the pin must see it)", 'PRECEDENCE = ("O6", "O7", "O45", "O123")', 'PRECEDENCE = ("O6", "O7", "O123", "O45")', {PIN_R2}),
    MU("the SLACK gap is NOT_READ", 'return ("match", "O4", ("SLACK",))', 'return ("unknown", "O4/O5")', {"outcome:gap->O4", "outcome:gap_k3->O4", "outcome:gap_share_below->O4", TRUTH}),
    MU("the SLACK flag is dropped", 'return ("match", "O4", ("SLACK",))', 'return ("match", "O4", ())', {"outcome:gap->O4", "outcome:gap_k3->O4", "outcome:gap_share_below->O4", TRUTH}),
    MU("the O4/O5 overlap is reported as O4 only", 'return ("match", "O4+O5", ())', 'return ("match", "O4", ())', {"outcome:o5_cap_o4_tie->O4+O5", TRUTH}),
    MU("O4+O5 is not a determinate outcome (exit code)", 'DETERMINATE = ("O1", "O2", "O3", "O4", "O5", "O6", "O7", "O4+O5")', 'DETERMINATE = ("O1", "O2", "O3", "O4", "O5", "O6", "O7")',
       {"e2e:rc0_and_the_O4+O5_line_for_the_overlap", PIN_O}),
    MU("an unreadable row is ignored", '        if r[0] == "unknown":\n', '        if False:\n', {"notread:n_log_cols", "notread:n_log_gap", TRUTH}),
    MU("a K-MON kill is judged by its code, not by its recorded reason", 'if isinstance(why, str) and K3_REASON_RX.search(why):', "if False:", {"outcome:kmon_k3->O4", PIN_K3}),
    MU("any recorded reason counts as the K3 rule", K3_RX_LINE, 'K3_REASON_RX = re.compile(r"")', {"outcome:kmon_monitor_died->O7", "outcome:o7_k4->O7", PIN_K3}),
    MU("Kleene AND returns True beside an unknown", "    return None if any(x is None for x in xs) else True\n\n\ndef or3", "    return True\n\n\ndef or3", {TRUTH}),
    MU("Kleene OR returns False beside an unknown", "    return None if any(x is None for x in xs) else False\n\n\ndef not3", "    return False\n\n\ndef not3", {TRUTH}),
    # ---- the trainer's log (addendum R3)
    MU("violation threshold 0.001 -> 0.01", "O4_VIOLATION_MAX = 0.001", "O4_VIOLATION_MAX = 0.01", {"outcome:viol_over->O5", "outcome:win_edge->O5", PIN_O, PIN_R3}),
    MU("violation strict (< for <=)", SIGNED_OR_SUM, 'out["violation_ok"] = bool(max(g["ttc"], g["nc"]) < O4_VIOLATION_MAX)', {"outcome:viol_exact->O4"}),
    MU("only the TTC term is judged", SIGNED_OR_SUM, 'out["violation_ok"] = bool(g["ttc"] <= O4_VIOLATION_MAX)', {"outcome:o5_nc_viol->O5"}),
    MU("the SUM of the two terms is judged", SIGNED_OR_SUM, 'out["violation_ok"] = bool(g["ttc"] + g["nc"] <= O4_VIOLATION_MAX)', {"outcome:terms_max->O5"}),
    MU("the POSITIVE PART is averaged instead of the signed value", 'g = {t: math.fsum(by_u[u][f"con_g_{t}"] for u in need) / len(need) for t in ("ttc", "nc")}',
       'g = {t: math.fsum(max(by_u[u][f"con_g_{t}"], 0.0) for u in need) / len(need) for t in ("ttc", "nc")}', {"outcome:osc_signed->O4", "outcome:terms_max->O5"}),
    MU("the window is 50 updates", "O4_WINDOW_UPDATES = 100", "O4_WINDOW_UPDATES = 50", {"outcome:win_edge->O5", PIN_O, PIN_R3}),
    MU("the window is MOVED one row earlier", WINDOW, "need = [c - O4_WINDOW_UPDATES + LOG_EVERY * i for i in range(0, O4_WINDOW_UPDATES // LOG_EVERY)]", {"outcome:win_newest->O5", "outcome:osc_signed->O4"}),
    MU("the window is MOVED one row later", WINDOW, "need = [c - O4_WINDOW_UPDATES + LOG_EVERY * i for i in range(2, O4_WINDOW_UPDATES // LOG_EVERY + 2)]", {"outcome:win_edge->O5"}),
    MU("the window ends at the last check, not the first breached one", 'c_upd = int(c_tag[len("update"):]) if c_tag else None', "c_upd = 1000 if c_tag else None", {"outcome:win_edge->O5", "outcome:o4_k3->O4", "outcome:k3_late_pair->O5"}),
    MU("the first breached check is the LAST one", "first = next((t for t in SR.CHECK_TAGS if t in per_tag), None)", "first = next((t for t in reversed(SR.CHECK_TAGS) if t in per_tag), None)", {"outcome:iso_two->O4"}),
    MU("a K3 kill uses the first breached check, not the first of the consecutive pair", "c_tag = (kill_pair_first(per_tag) if k3_kill else first_tag) if breached else None", "c_tag = first_tag if breached else None", {"outcome:k3_late_pair->O5"}),
    MU("a K3 kill uses the LAST of the consecutive pair", "return next((a for a, b in zip(SR.CHECK_TAGS[:-1], SR.CHECK_TAGS[1:]) if a in per_tag and b in per_tag), None)",
       "return next((b for a, b in zip(SR.CHECK_TAGS[:-1], SR.CHECK_TAGS[1:]) if a in per_tag and b in per_tag), None)", {"outcome:o4_k3->O4", "outcome:k3_late_pair->O5"}),
    MU("the monitor's rows are not used to find the breach", 'if isinstance(r, dict) and r.get("tag") in SR.CHECK_TAGS and r["tag"] not in per_tag:', "if False:", {"outcome:o4_k3->O4", "outcome:o5_k3->O5"}),
    MU("a breach at the FINAL check is not located", 'first = next((t for t in SR.CHECK_TAGS if t in per_tag), None)', 'first = next((t for t in SR.CHECK_TAGS[:3] if t in per_tag), None)', {"outcome:o4_u1000->O4"}),
    MU("the log is read without a first breached check", "lg = read_log(sft, c_upd, ctx, registered_temp=registered_temp)", "lg = read_log(sft, None, ctx, registered_temp=registered_temp)", {"outcome:o4->O4", "outcome:o5_k3->O5", "outcome:viol_over->O5"}),
    MU("the window overlap with the ramp is not disclosed", 'out["window_overlaps_ramp"] = bool(c - O4_WINDOW_UPDATES + 1 <= ramp)', 'out["window_overlaps_ramp"] = False', {"outcome:o4_k3->O4"}),
    MU("lambda share threshold 0.5 -> 0.6", "O4_LAM_POS_SHARE_MIN = 0.5", "O4_LAM_POS_SHARE_MIN = 0.6", {"outcome:o4_share_exact->O4", PIN_O, PIN_R3}),
    MU("lambda share strict (> for >=)", 'out["lam_pos_share_ok"] = bool(k >= O4_LAM_POS_SHARE_MIN * m)', 'out["lam_pos_share_ok"] = bool(k > O4_LAM_POS_SHARE_MIN * m)', {"outcome:o4_share_exact->O4"}),
    MU("the share counts from update 0, not from the ramp end", 'm, k = c - ramp, rc["con_lam_pos_updates"] - by_u[ramp]["con_lam_pos_updates"]', 'm, k = c, rc["con_lam_pos_updates"]', {"outcome:gap_share_below->O4", "outcome:o4_share_exact->O4"}),
    MU("the share is counted over the WHOLE RUN, not through the first breach", 'm, k = c - ramp, rc["con_lam_pos_updates"] - by_u[ramp]["con_lam_pos_updates"]',
       'm, k = U - ramp, last["con_lam_pos_updates"] - by_u[ramp]["con_lam_pos_updates"]', {"outcome:share_through_breach->O4", "outcome:o4_share_exact->O4"}),
    MU("lambda_max is read over the WHOLE RUN, not through the first breach", 'out["cap_hit"] = bool(rc["con_cap_hits"] > 0 or rc["con_lam_max_ttc"] >= lam_max_cfg or rc["con_lam_max_nc"] >= lam_max_cfg)',
       'out["cap_hit"] = bool(last["con_cap_hits"] > 0 or last["con_lam_max_ttc"] >= lam_max_cfg or last["con_lam_max_nc"] >= lam_max_cfg)', {"outcome:cap_after_breach->O4"}),
    MU("the cap is never seen", 'out["cap_hit"] = bool(rc["con_cap_hits"] > 0 or rc["con_lam_max_ttc"] >= lam_max_cfg or rc["con_lam_max_nc"] >= lam_max_cfg)', 'out["cap_hit"] = False',
       {"outcome:o5_cap_o4_tie->O4+O5", "outcome:o5_k3->O5"}),
    MU("O6 reads only pos_updates", 'out["lam_zero"] = bool(last["con_lam_pos_updates"] == 0 and last["con_lam_max_ttc"] == 0.0 and last["con_lam_max_nc"] == 0.0)', 'out["lam_zero"] = bool(last["con_lam_pos_updates"] == 0)',
       {"outcome:o3_last_step->O3"}),
    MU("O6 reads only the TTC maximum", 'out["lam_zero"] = bool(last["con_lam_pos_updates"] == 0 and last["con_lam_max_ttc"] == 0.0 and last["con_lam_max_nc"] == 0.0)', 'out["lam_zero"] = bool(last["con_lam_pos_updates"] == 0 and last["con_lam_max_ttc"] == 0.0)',
       {"outcome:o3_last_step_nc->O3"}),
    MU("a MISSING log column is read as zero", "miss = [k for k in CON_COLS if not _num(r.get(k))]", "miss = [r.setdefault(k, 0.0) for k in CON_COLS if k not in r] and []", {"notread:n_log_cols"}),
    MU("the BUILT dual is not compared with the registered one", 'if data.get("epo_con") != registered:', "if False:", {"notread:n_log_lam0", "notread:n_log_eta", "real_main:the_real_run_is_refused_under_the_registered_values"}),
    # PICKCON: the data event's epo_con block carries `temp`; a legacy block (no key) and a block at the registered T read as the registered block, any other T is refused
    MU("the constraint's registered T is not enforced (any numeric temp passes)", 'if isinstance(t, (int, float)) and not isinstance(t, bool) and t == registered_temp:',
       'if isinstance(t, (int, float)) and not isinstance(t, bool):', {"notread:n_log_temp", "con_temp:an_arm_whose_log_says_0.1_is_NOT_READ_under_a_registered_0.01_and_reads_O1_under_0.1"}),
    MU("a legacy block (no temp) is refused", "    return blk          # PICKCON-LEGACY", "    return None          # PICKCON-LEGACY",
       {"tolerated:a_legacy_data_block_without_temp_is_read_exactly_as_before_PICKCON", "compat:the_banked_R0d_trainer_log_has_the_legacy_epo_con_block_and_is_read"}),
    MU("a block at the registered T is not read as the registered block (temp never stripped)", 'return {k: v for k, v in blk.items() if k != "temp"}', "return blk",
       {"outcome:o1->O1", "real_main:read_log_reads_the_real_trainers_data_train_and_done_rows"}),
    # ---- R0e-pick B3: the constraint's registered T is per arm, from the arm's registration, never from the log
    MU("the per-arm comparison is deleted: the T compared against is the PREREG 0.1 again", 'if isinstance(t, (int, float)) and not isinstance(t, bool) and t == registered_temp:',
       'if isinstance(t, (int, float)) and not isinstance(t, bool) and t == REGISTERED_CON_TEMP:',
       {"con_temp:a_REAL_trainer_run_at_epo_con_temp_0.01_is_read_under_a_registered_0.01", "con_temp:an_arm_whose_REAL_EpoDual_ran_at_0.01_reads_O1_under_a_registered_0.01_and_NOT_READ_under_the_default_or_an_explicit_0.1",
        "con_temp:a_registered_0.01_REFUSES_every_other_temp_in_the_log_and_a_registered_1.0_REFUSES_a_bool_True"}),
    MU("a bool is accepted as the registered T (True == 1.0)", 'if isinstance(t, (int, float)) and not isinstance(t, bool) and t == registered_temp:', 'if isinstance(t, (int, float)) and t == registered_temp:',
       {"con_temp:a_registered_0.01_REFUSES_every_other_temp_in_the_log_and_a_registered_1.0_REFUSES_a_bool_True"}),
    MU("the registered T is inferred from the log itself", '        blk0 = data.get("epo_con")\n', '        blk0 = data.get("epo_con")\n        registered_temp = blk0.get("temp", registered_temp) if isinstance(blk0, dict) else registered_temp\n',
       {"con_temp:a_registered_0.1_REFUSES_the_REAL_0.01_run_and_a_registered_0.01_REFUSES_the_REAL_default_run", "con_temp:an_arm_whose_log_says_0.1_is_NOT_READ_under_a_registered_0.01_and_reads_O1_under_0.1",
        "con_temp:a_registered_0.01_REFUSES_every_other_temp_in_the_log_and_a_registered_1.0_REFUSES_a_bool_True"}),
    MU("a legacy block (no temp) is accepted under a registered T that is not 0.1", 'if registered_temp != REGISTERED_CON_TEMP and isinstance(blk0, dict) and "temp" not in blk0:', "if False:",
       {"con_temp:a_registered_0.01_REFUSES_a_legacy_block_with_no_temp_(a_trainer_that_predates_the_flag_ran_the_constraint_at_0.1)",
        "con_temp:the_banked_R0d_log_reads_exactly_as_its_banked_pre_B3_record_says_and_is_REFUSED_under_a_registered_0.01"}),
    MU("read_log compares the block against the PREREG 0.1, not the arm's T", 'data = dict(data, epo_con=_registered_view(blk0, registered_temp))', 'data = dict(data, epo_con=_registered_view(blk0))',
       {"con_temp:a_REAL_trainer_run_at_epo_con_temp_0.01_is_read_under_a_registered_0.01"}),
    MU("read_constrained does not hand the arm's T to read_log", 'lg = read_log(sft, c_upd, ctx, registered_temp=registered_temp)', "lg = read_log(sft, c_upd, ctx)",
       {"con_temp:an_arm_whose_REAL_EpoDual_ran_at_0.01_reads_O1_under_a_registered_0.01_and_NOT_READ_under_the_default_or_an_explicit_0.1"}),
    MU("run() does not hand the arm's registered T to read_constrained", 'registered_temp=con_temps.get(n, REGISTERED_CON_TEMP), temp_source=', 'registered_temp=REGISTERED_CON_TEMP, temp_source=',
       {"con_temp:each_arm_is_read_against_its_OWN_registered_T_(R0d_0.01_beside_R1d_0.1_and_both_0.01)", "e2e:con_temp_cli:--con-temp_R0d=0.01_reads_the_0.01_arm_O1_rc0_and_records_the_registration"}),
    MU("one T for every arm (the first registered T is used for each slot)", 'registered_temp=con_temps.get(n, REGISTERED_CON_TEMP), temp_source=', 'registered_temp=next(iter(con_temps.values()), REGISTERED_CON_TEMP), temp_source=',
       {"con_temp:each_arm_is_read_against_its_OWN_registered_T_(R0d_0.01_beside_R1d_0.1_and_both_0.01)", "e2e:con_temp_cli:R0d_at_0.01_beside_R1d_at_the_default_are_both_read_and_the_joint_replicates"}),
    MU("the record forgets the registration", '        rec["registered"]["epo_con_temp_by_arm"] = {', '        rec["registered"]["epo_con_temp_by_arm_FORGOTTEN"] = {',
       {"con_temp:each_arm_is_read_against_its_OWN_registered_T_(R0d_0.01_beside_R1d_0.1_and_both_0.01)", "e2e:con_temp_cli:--con-temp_R0d=0.01_reads_the_0.01_arm_O1_rc0_and_records_the_registration"}),
    MU("a registration that is not a positive finite number is accepted by run()", '        if not _num(t) or t <= 0:\n            raise ValueError(f"the registered constraint T of {n} must be', '        if False:\n            raise ValueError(f"the registered constraint T of {n} must be',
       {"con_temp:a_registration_that_is_not_a_positive_finite_number_for_a_constrained_arm_is_refused_before_anything_is_read"}),
    MU("a registration for an arm that is not constrained is accepted by run()", '        if n not in CONSTRAINED:\n            raise ValueError(f"a registered constraint T was given for', '        if False:\n            raise ValueError(f"a registered constraint T was given for',
       {"con_temp:a_registration_that_is_not_a_positive_finite_number_for_a_constrained_arm_is_refused_before_anything_is_read"}),
    MU("read_log accepts a registered T that is not a positive finite number", '    if not _num(registered_temp) or registered_temp <= 0:\n', '    if False:\n',
       {"con_temp:a_registration_that_is_not_a_positive_finite_number_for_a_constrained_arm_is_refused_before_anything_is_read"}),
    MU("an arms file arm whose constraint is OFF has a T", '    if econ is None or econ == "off":', "    if False:",
       {"con_temp:the_arms_file_reader_resolves_the_REAL_registered_argv_of_R0d_to_0.1_by_default_and_refuses_R1o_whose_constraint_is_off",
        "con_temp:the_arms_file_reader_resolves_the_T_the_trainer_would_(con_temp_else_epo_temp_else_default)_and_refuses_an_ambiguous_or_constraint_less_registration"}),
    MU("the arms file reader ignores --epo-con-temp", "    raw = named if named is not None else epo", "    raw = epo",
       {"con_temp:the_arms_file_reader_resolves_the_T_the_trainer_would_(con_temp_else_epo_temp_else_default)_and_refuses_an_ambiguous_or_constraint_less_registration"}),
    MU("the CLI parses --con-temp and drops it", 'a.addendum or a0, con_temps=temps, con_temp_sources=tsrc)', "a.addendum or a0)",
       {"e2e:con_temp_cli:--con-temp_R0d=0.01_reads_the_0.01_arm_O1_rc0_and_records_the_registration", "e2e:con_temp_cli:--arms-file_--arm-reg_R0d=R0e-pick_registers_0.01_from_the_registered_argv_and_records_the_file_sha256"}),
    MU("--con-temp and the arms file may disagree", "        if n in temps and temps[n] != t:", "        if False:", {"e2e:con_temp_cli:rc3_usage_error_for_disagreeing_registrations"}),
    MU("--con-temp 0 / nan / negative is accepted", "if n not in CONSTRAINED or n not in dirs or n in temps or t is None or not _num(t) or t <= 0:", "if n not in CONSTRAINED or n not in dirs or n in temps or t is None:",
       {"e2e:con_temp_cli:rc3_usage_error_for_zero", "e2e:con_temp_cli:rc3_usage_error_for_nan"}),
    MU("the BUILT ramp is not compared with the registered one","if ramp != registered_ramp or isinstance(ramp, bool):", "if False:", {"notread:n_log_ramp", "real_main:the_real_run_is_refused_under_the_registered_values"}),
    MU("the train rows need not be contiguous", "if ups != list(range(LOG_EVERY, LOG_EVERY * len(trains) + 1, LOG_EVERY)):", "if False:", {"notread:n_log_gap"}),
    MU("the cumulative columns may decrease", "                if b[k] < a[k]:", "                if False:", {"notread:n_log_decreasing"}),
    MU("cap_hits and lam_seen_max may disagree", 'if (r["con_cap_hits"] > 0) != (r["con_lam_max_ttc"] >= lam_max_cfg or r["con_lam_max_nc"] >= lam_max_cfg):', "if False:", {"notread:n_log_cap_inconsistent"}),
    MU("the done event is not cross-checked", 'if (ec.get("pos_updates"), ec.get("cap_hits")', 'if False and (ec.get("pos_updates"), ec.get("cap_hits")', {"notread:n_log_done", "real_main:the_real_done_events_state_matches_the_last_row_and_a_tampered_one_is_caught"}),
    MU("a half-written final line is a problem", '(out["notes"] if (bad == 1 and last_bad) else pb)', "pb", {"tolerated:a_half_written_final_log_line_is_a_note_not_a_problem"}),
    MU("the trainer's cadence is 50", "LOG_EVERY = 25 ", "LOG_EVERY = 50 ", {"pin:trainer_log_cadence_columns_and_counter_semantics", "outcome:o4->O4"}),
    # ---- the state and S
    MU("a stale S file is accepted", 'elif fileS.get("record_sha256") != S["record_sha256"]:', "elif False:", {"notread:n_S_stale"}),
    MU("a hand-edited S file is accepted", 'if fileS.get("record_sha256") != hashlib.sha256(body.encode("utf-8")).hexdigest():', "if False:", {"notread:n_S_handedit"}),
    MU("a missing S file is accepted", "    if not os.path.exists(sp):\n        s_problem", "    if False:\n        s_problem", {"notread:n_S_missing"}),
    MU("a run with reads but no end is PENDING", 'rec["state"] = "killed" if killed_file else ("complete" if has_done else ("unclear" if has_reads else "incomplete"))',
       'rec["state"] = "killed" if killed_file else ("complete" if has_done else "incomplete")', {"notread:n_log_missing_is_not_pending_when_the_chain_already_read_the_arm"}),
    MU("an unfinished arm reads O1", 'rec["outcome"] = "PENDING"\n        pb.append("the run has neither', 'rec["outcome"] = "O1"\n        pb.append("the run has neither', {"pending:an_unfinished_arm_is_PENDING_and_never_an_outcome", "pending:PENDING_arm_gives_a_PENDING_joint"}),
    MU("a replay-only kill is ignored", 'if isinstance(rep, dict) and rep.get("kill"):', "if False:", {"outcome:o7_k4_retro->O7"}),
    MU("an unfinished log with reads is taken as complete", '("unclear" if has_reads else "incomplete")', '("complete" if has_reads else "incomplete")', {"notread:n_k4_replay_unfinished"}),
    MU("--navtest-e is dropped by run()", "rec[\"E\"] = bar_E(navtest_e, ctx, floor, why)", "rec[\"E\"] = bar_E(None, ctx, floor, why)", {"e2e:navtest_e_file_is_read_end_to_end_ADOPT"}),
    # ---- bar L (addendum R4: TTC only)
    MU("L floor multiple 2 -> 1", "L_FLOOR_MULT = 2.0", "L_FLOOR_MULT = 1.0", {"L:exact_tie_effect_equal_2xF_is_not_a_pass_SYNTHETIC_dyadic", "L:a_large_seed_floor_fails_a_positive_effect", "pin:L_E_floors_thresholds_and_estimator_against_the_prereg_text"}),
    MU("L seed-0 tie passes (>= for >)", 't["effect_seed0"] > L_FLOOR_MULT * t["F_seed"]', 't["effect_seed0"] >= L_FLOOR_MULT * t["F_seed"]', {"L:exact_tie_effect_equal_2xF_is_not_a_pass_SYNTHETIC_dyadic"}),
    MU("F_seed is the smaller term", 'f_seed = max(abs(v["R0"] - v["R1o"]), abs(v["R0d"] - v["R1d"]))', 'f_seed = min(abs(v["R0"] - v["R1o"]), abs(v["R0d"] - v["R1d"]))', {"L:a_large_seed_floor_fails_a_positive_effect"}),
    MU("F_seed uses only the unconstrained pair", 'f_seed = max(abs(v["R0"] - v["R1o"]), abs(v["R0d"] - v["R1d"]))', 'f_seed = abs(v["R0"] - v["R1o"])', {"L:exact_tie_effect_equal_2xF_is_not_a_pass_SYNTHETIC_dyadic"}),
    MU("L needs only seed 0", 'out["verdict"] = "PASS" if (t["seed0_ok"] and t["seed1_ok"]) else "FAIL"', 'out["verdict"] = "PASS" if t["seed0_ok"] else "FAIL"', {"L:seed1_fails_alone_SYNTHETIC_dyadic"}),
    MU("L also needs the PDMS effect > 0 (L decided on TTC AND PDMS)", 'out["verdict"] = "PASS" if (t["seed0_ok"] and t["seed1_ok"]) else "FAIL"',
       'out["verdict"] = "PASS" if (t["seed0_ok"] and t["seed1_ok"] and out["pdms"]["effect_seed0"] > 0) else "FAIL"', {"L:the_PDMS_numbers_never_move_the_verdict_SYNTHETIC"}),
    MU("L effect has the wrong sign", 'eff0, eff1 = v["R0d"] - v["R0"], v["R1d"] - v["R1o"]', 'eff0, eff1 = v["R0"] - v["R0d"], v["R1o"] - v["R1d"]', {"L:pass_real_producer_both_seeds_clear_2xF_seed"}),
    MU("L is PENDING never: R1o is not required", 'if "R1o" not in arm_dirs:\n        return {"verdict": "PENDING"', 'if False:\n        return {"verdict": "PENDING"', {"L:PENDING_until_R1o_exists_and_never_PASS"}),
    MU("L PENDING written as PASS", '{"verdict": "PENDING", "why": "R1o has not run', '{"verdict": "PASS", "why": "R1o has not run', {"L:PENDING_until_R1o_exists_and_never_PASS"}),
    MU("L controls are not tied to their own dumps", 'if os.path.exists(fp) and ((r.get("inputs") or {}).get("scores_sha256") or {}).get("CTRL" if f_.endswith("A.npz") else "R") != ctx.sha(fp):', "if False:", {"L:NOT_READ_without_the_banked_R0_or_a_control_read_or_with_another_dump"}),
    # ---- bar E (addendum R4)
    MU("E PENDING written as ADOPT", '{"verdict": "PENDING", "why": "no navtest read exists yet', '{"verdict": "ADOPT", "why": "no navtest read exists yet', {"E:PENDING_when_no_navtest_read_exists_and_never_ADOPT"}),
    MU("E: each pair >= 0.125", "E_EACH_MIN, E_MEAN_MIN = 0.25, 0.5", "E_EACH_MIN, E_MEAN_MIN = 0.125, 0.5", {"E:one_pair_below_025_mean_ok->NOT_PROVEN", PIN_R4.replace(PIN_R4, "pin:L_E_floors_thresholds_and_estimator_against_the_prereg_text")}),
    MU("E: mean >= 0.3", "E_EACH_MIN, E_MEAN_MIN = 0.25, 0.5", "E_EACH_MIN, E_MEAN_MIN = 0.25, 0.3", {"E:mean_below_05->NOT_PROVEN", "pin:L_E_floors_thresholds_and_estimator_against_the_prereg_text"}),
    MU("E: pooled lower bound >= 0", 'c2 = bool(pooled["ci95"][0] > 0)', 'c2 = bool(pooled["ci95"][0] >= 0)', {"E:pooled_lower_exactly_zero->NOT_PROVEN"}),
    MU("E: floor multiple 2 -> 1", "E_FLOOR_MULT = 2.0", "E_FLOOR_MULT = 1.0", {"E:floor_tie_irep->NOT_PROVEN", "E:floor_tie_navtest_seed_floor->NOT_PROVEN", "pin:L_E_floors_thresholds_and_estimator_against_the_prereg_text"}),
    MU("E: floor tie passes (>= for >)", "c3 = bool(mean_contrast > E_FLOOR_MULT * floor)", "c3 = bool(mean_contrast >= E_FLOOR_MULT * floor)", {"E:floor_tie_irep->NOT_PROVEN", "E:floor_tie_navtest_seed_floor->NOT_PROVEN", "E:floor_tie_rlval_seed_floor->NOT_PROVEN"}),
    MU("E: the floor is the smaller of the terms", "floor = max(floor_terms) if rlval_floor is not None else None", "floor = min(floor_terms) if rlval_floor is not None else None",
       {"E:floor_tie_irep->NOT_PROVEN", "E:floor_tie_navtest_seed_floor->NOT_PROVEN", "E:floor_tie_rlval_seed_floor->NOT_PROVEN"}),
    MU("E: the navtest seed floor |R0d - R1d| is ignored", 'floor_terms = [navtest_seed, fl["irep_abs_delta"]] +', 'floor_terms = [fl["irep_abs_delta"]] +', {"E:floor_tie_navtest_seed_floor->NOT_PROVEN"}),
    MU("E: the RL-val seed floor is ignored", '+ ([rlval_floor["value"]] if rlval_floor is not None else [])', "+ []", {"E:floor_tie_rlval_seed_floor->NOT_PROVEN"}),
    MU("E: the floor is taken from the file's seed_floor_abs", 'floor_terms = [navtest_seed, fl["irep_abs_delta"]] +', 'floor_terms = [navtest_seed, fl["irep_abs_delta"], fl.get("seed_floor_abs", 0.0)] +',
       {"E:seed_floor_abs_in_the_file_is_IGNORED_the_tool_derives_the_floors->ADOPT"}),
    MU("E: an unreadable clause 3 is True", C3_LINE, "c3 = bool(mean_contrast > E_FLOOR_MULT * floor) if floor is not None else True", {"E:rlval_floor_unavailable_clause3_is_unread_so_the_file_is_NOT_READ->NOT_READ", "E:run_without_R0_and_R1o_cannot_read_clause_3"}),
    MU("E: clause 4 dropped", 'c4 = bool(all(pairs[n][s]["ci95"][1] >= 0 for n in CONSTRAINED for s in ("TTC", "NC")))', "c4 = True", {"E:ttc_upper_adverse_in_one_pair->NOT_PROVEN", "E:nc_upper_adverse_in_one_pair->NOT_PROVEN"}),
    MU("E: clause 4 strict", 'c4 = bool(all(pairs[n][s]["ci95"][1] >= 0 for n in CONSTRAINED for s in ("TTC", "NC")))', 'c4 = bool(all(pairs[n][s]["ci95"][1] > 0 for n in CONSTRAINED for s in ("TTC", "NC")))', {"E:nc_upper_exactly_zero_is_not_adverse->ADOPT"}),
    MU("E: clause 4 checks one pair only", 'for n in CONSTRAINED for s in ("TTC", "NC")))', 'for n in CONSTRAINED[:1] for s in ("TTC", "NC")))', {"E:ttc_upper_adverse_in_one_pair->NOT_PROVEN"}),
    MU("E: clause 4 reads all six sub-scores", 'c4 = bool(all(pairs[n][s]["ci95"][1] >= 0 for n in CONSTRAINED for s in ("TTC", "NC")))', 'c4 = bool(all(pairs[n][s]["ci95"][1] >= 0 for n in CONSTRAINED for s in SUBS))',
       {"E:ep_adverse_does_not_block_clause_4_which_is_TTC_and_NC->ADOPT"}),
    MU("E: refuted threshold 0.3", "E_REFUTED_UPPER = 0.25", "E_REFUTED_UPPER = 0.3", {"E:pooled_upper_exactly_025_is_not_refuted->NOT_PROVEN", "pin:L_E_floors_thresholds_and_estimator_against_the_prereg_text"}),
    MU("E: refuted at <=", 'refuted = bool(pooled["ci95"][1] < E_REFUTED_UPPER)', 'refuted = bool(pooled["ci95"][1] <= E_REFUTED_UPPER)', {"E:pooled_upper_exactly_025_is_not_refuted->NOT_PROVEN"}),
    MU("E: ADOPT and REFUTED together accepted", "        if adopt is True and refuted:", "        if False:", {"E:adopt_and_refuted_both_is_inconsistent->NOT_READ"}),
    MU("E: a missing sub-score is accepted", 'miss = [s for s in ("PDMS",) + SUBS if not _ci(pairs[n].get(s))]', "miss = []", {"E:subscore_missing_clause5->NOT_READ"}),
    MU("E: the estimator is not pinned", 'if f"{N_BOOT} resamples" not in est or f"seed {SEED}" not in est:\n            pb.append', 'if False:\n            pb.append', {"E:estimator_not_registered->NOT_READ"}),
    MU("E: any tier accepted", 'if not isinstance(d, dict) or d.get("tier") != "T1":', "if False:", {"E:tier_not_T1->NOT_READ"}),
    MU("E: the pooled mean need not be the mean of the pairs", 'elif not pb and abs(d["pooled"]["PDMS"]["mean"] - (pairs["R0d"]["PDMS"]["mean"] + pairs["R1d"]["PDMS"]["mean"]) / 2.0) > E_POOL_TOL:', "elif False:",
       {"E:pooled_mean_is_not_the_mean_of_the_two_pairs->NOT_READ"}),
    MU("E: the pooled interval need not be a JOINT resample", 'elif (d.get("pooled") or {}).get("resample") != "joint":', "elif False:", {"E:pooled_resampled_independently_is_refused->NOT_READ", "E:pooled_resample_undeclared_is_refused->NOT_READ"}),
    MU("E: n_tokens / n_logs are not pinned", 'if not isinstance(d, dict) or d.get("n_tokens") != NAVTEST_TOKENS or d.get("n_logs") != NAVTEST_LOGS:', "if False:", {"E:n_tokens_is_not_12146->NOT_READ", "E:n_logs_is_not_136->NOT_READ"}),
    # ---- the addendum itself (addendum: the tool refuses to run unless it is the registered file)
    MU("an addendum sha256 mismatch is NOT refused", '    if sha != ADDENDUM_SHA256:\n        raise AddendumRefused', '    if False:\n        raise AddendumRefused', {REFUSE, "e2e:rc4_refused_for_a_wrong_addendum_and_no_record_written"}),
    MU("a missing addendum file is NOT refused", '    if not os.path.isfile(path):\n        raise AddendumRefused(f"the addendum {path} does not exist")', '    if False:\n        raise AddendumRefused(f"the addendum {path} does not exist")', {REFUSE}),
    MU("no addendum given is NOT refused", '    if not path:\n        raise AddendumRefused("no addendum file given', '    if False:\n        raise AddendumRefused("no addendum file given', {REFUSE, "e2e:rc4_refused_when_no_addendum_is_given_and_none_is_beside_the_tool"}),
    MU("the registered addendum sha256 is wrong", 'ADDENDUM_SHA256 = "3c6a25c1', 'ADDENDUM_SHA256 = "4c6a25c1', {"pin:the_addendum_is_the_registered_bytes_and_names_the_prereg_it_amends", REFUSE}),
    MU("the record is not stamped with the addendum", '"addendum": add["name"], "addendum_sha256": add["sha256_checked"], "addendum_file": add,', '"addendum_file": add,', {"record:every_record_carries_the_addendum_name_and_its_sha256"}),
    MU("an arm record is not stamped with the addendum", '"addendum": ADDENDUM_NAME, "addendum_sha256": ADDENDUM_SHA256, "state": None', '"state": None', {"record:every_record_carries_the_addendum_name_and_its_sha256"}),
    MU("the ambiguities the addendum settled are still listed", 'PREREG_AMBIGUITIES = [\n    {"id": "A-E3b"', 'PREREG_AMBIGUITIES = [\n    {"id": "A-G1", "where": "x", "text": "x"},\n    {"id": "A-E3b"', {"record:lists_ONLY_the_readings_the_addendum_does_not_settle_and_names_what_it_settled"}),
    # ---- the joint rule, provenance, the CLI
    MU("NOT REPLICATED is not recognised", 'elif o0 == "O1":\n        j = {"replication": "NOT_REPLICATED"', 'elif False:\n        j = {"replication": "NOT_REPLICATED"', {"joint:r0d_O1_r1d_not->NOT_REPLICATED"}),
    MU("REPLICATED needs only R0d", 'elif o0 == "O1" and o1 == "O1":', 'elif o0 == "O1":', {"joint:r0d_O1_r1d_not->NOT_REPLICATED"}),
    MU("numbers lose their sha256", 'return {"value": value, "source": os.path.abspath(path), "sha256": self.sha(path)}', 'return {"value": value, "source": os.path.abspath(path), "sha256": None}',
       {"record:every_number_names_its_source_file_and_that_files_current_sha256", "E:every_number_carries_its_source_and_sha256"}),
    MU("the record hash is not over the record", 'body = json.dumps({k: v for k, v in rec.items() if k != "record_sha256"}, sort_keys=True, separators=(",", ":"), default=str)', 'body = "{}"',
       {"record:the_record_hash_is_self_consistent_and_inputs_are_listed", "e2e:writes_the_record_with_a_self_consistent_hash"}),
    MU("a wrong prereg file is not noticed", "        if sha != PREREG_SHA256:", "        if False:", {"record:a_wrong_prereg_file_is_a_problem"}),
    MU("the registered prereg sha is wrong", 'PREREG_SHA256 = "b404c2fb', 'PREREG_SHA256 = "c404c2fb', {"pin:the_prereg_file_is_the_registered_bytes", "record:a_wrong_prereg_file_is_a_problem"}),
    MU("the registered lr is wrong", '"lr": 4.0, "max": 10.0, "lam0": 0.0, "ref": "armA_detached"}      # PREREG section 2', '"lr": 2.0, "max": 10.0, "lam0": 0.0, "ref": "armA_detached"}      # PREREG section 2',
       {"outcome:o1->O1", "pin:registered_dual_values_and_ramp_against_the_prereg_text_and_the_trainer"}),
    MU("exit code 0 for a NOT_READ arm", "return 0 if done else 2", "return 0", {"e2e:rc2_for_NOT_READ"}),
    MU("a duplicate --arm is accepted", "if n not in ARM_NAMES or not d or n in dirs:", "if n not in ARM_NAMES or not d:", {"e2e:rc3_for_a_duplicate_arm"}),
    MU("the record file is not written", "    write_record(out, rec)\n    for n, r in rec", "    for n, r in rec", {"e2e:writes_the_record_with_a_self_consistent_hash", "e2e:default_out_is_beside_the_first_arm_dir"}),
]
MUTATIONS = [m for m in MUTATIONS if m[3]]                                      # an entry with no expected RED location is a draft, not a mutation


# ============================================================ main
def build_world(td: Path):
    W = World(td)
    A = make_world_arms(W, td)
    make_mods(W, A, td)
    return W, A


def load_arms_from(td: Path):
    """SOBJ_OUTCOME_KEEP development convenience ONLY: reuse an already built tree (the registered 10,000-resample reads take ~80 s)."""
    raise NotImplementedError


def main() -> int:
    src = TOOL_PATH.read_text(encoding="utf-8")
    n = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        W, A = build_world(td)
        # ---- the unmutated tool: every check, the CLI included
        bad = run_suite(src, W, A, td, "base", e2e=True)
        assert bad == [], f"the UNMUTATED tool fails: {bad}"
        n += 1
        names = {nm for nm, _ in CHECKS}
        n_checks = len(CHECKS)
        for i, (name, old, new, expect) in enumerate(MUTATIONS):
            assert src.count(old) == 1, f"mutation anchor x{src.count(old)}: {name}: {old[:80]!r}"
            unknown = {x for x in expect if x not in names and not x.startswith("e2e:")}
            assert not unknown, f"mutation '{name}' expects checks that do not exist: {sorted(unknown)}"
            got = run_suite(src.replace(old, new), W, A, td, f"m{i}", e2e=any(x.startswith("e2e:") for x in expect))
            assert got, f"mutation '{name}' stayed GREEN"
            assert expect <= set(got), f"mutation '{name}' went RED but NOT where it exists to: expected {sorted(expect)}, got {got}"
            print(f"  RED  {name}  ->  {len(got)} check(s): {', '.join(got[:3])}{' ...' if len(got) > 3 else ''}", flush=True)
            n += 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite of {n_checks} checks + the CLI end to end, {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
