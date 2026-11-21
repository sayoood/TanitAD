"""PREREG_V8F_STRAT1 study checker (BUILD_SPEC section 8; the pattern of ``obd_check.py``). It maps the seven launch-gate checks onto the trainer's own selfcheck:

  G-HYG    V8FMiniConfig refuses bad values / unknown keys                      (tests: test_v8f_strat1.py::test_hygiene_*)           -> ``hyg``  (runs the refusals here)
  G-DVB    trainer ``declared_vs_built`` incl. ``bad_ctx`` + FLAG_CHECKS          -> ``gate`` asserts every argv lever against ``selfcheck.levers``
  G-LIVE   ``live`` / ``ctx_head_probe`` / ``nav_source_probe`` / inherited ``acc_probe`` ``otband_probe`` ``dagger_probe``             -> ``gate``
  G-CLOCK  label time: ``ctx_label_arrays`` alignment asserts (trainer) + ``build_navtext_nav.py`` G-CLOCK-NAV (its report)                -> ``gate`` reads the nav report sha
  G-EVAL   a checkpoint loads through each evaluator's own build path, strictly; built levers == the selfcheck's; a cfg that drops ctx_heads is REFUSED -> ``geval``
  G-CKPT   save / load round trip bit-equal                                       (tests + the post check)                              -> ``post``
  G-SUITE  pytest on a clean export of the branch tip, 0 regressions vs cdbff971   (run by the MM in the pod environment)                 -> NOT run here

Subcommands: ``argv TAG --root R --out O`` (render a frozen argv), ``gate TAG --selfcheck F`` (problems), ``post TAG --init I --final F`` (moved identity), ``geval --ckpt C --selfcheck F`` and ``token``.
NOT built here (UNVERIFIED, the MM's launch loop): the pilot timing / chain runner / pod preflight of ``obd_check.py``. This file only DECIDES; it launches nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(REPO, "stack"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

#: the control's code of record (PREREG: the OBD-CTL recipe): the frozen box-2 argvs the STRAT1 argvs extend (read-only input; box-2 paths are mapped to --root)
BASE_ARGVS = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/obd/box2/frozen/argvs_obd.json"
BASE_SHA256 = None            # pinned by the MM when the frozen file is copied into the chain's folder (UNVERIFIED here: read in place)
STRAT_LABELS_SHA256 = "8055a321df6db1725ea17abde2999d48bcb68d43f1a480a69ed42dd0cf184eff"
NINE = ("--nav-source", "--nav-text-npz", "--nav-text-sha256", "--ctx-heads", "--w-ctx", "--ctx-labels", "--ctx-labels-sha256", "--ctx-lr-mult", "--ctx-detach")
KINDS = ("C0", "N1", "S1", "S1d", "N1R", "S1R", "S1dR", "N1Z", "N1D", "N1Dx", "N1DW4", "N1DW2", "N1DC", "N1DH", "N1DM")
#: PREREG_V8F_NAVRB2: the R kinds = the base kind + the five --nav-rb flags (N1R / S1R / S1dR: the real channel; N1Z: the same argv with the channel all 0.0, a same-seed N1 replicate with the new code)
RB_BASE = {"N1R": "N1", "S1R": "S1", "S1dR": "S1d", "N1Z": "N1", "N1D": "N1", "N1Dx": "N1", "N1DW4": "N1", "N1DW2": "N1", "N1DC": "N1", "N1DH": "N1", "N1DM": "N1"}
#: PREREG_V8F_NAVRBD: the D kinds = an R kind (N1R) + exactly `--nav-rb-feat exitdist` and the 15-column npz + sha (N1D: the real channel; N1Dx: the deliberate-regression arm, rows deranged across clips). An R / non-RB kind never names exitdist.
D_KINDS = ("N1D", "N1Dx")
#: PREREG_V8F_N1DW: the W kinds = N1D's argv (the real 15-column channel, the same npz + sha) + exactly `--rb-window-w <w>` (N1DW4: w = 4, N1DW2: w = 2). They read the D channel (DCHAN_KINDS = every kind with the 15-column table); no
#: existing kind's argv moves, and a non-W kind never names the flag.
W_KINDS = ("N1DW4", "N1DW2")
W_VALUE = {"N1DW4": "4", "N1DW2": "2"}
RB_W_FLAG = "--rb-window-w"
#: PREREG_V8F_N1DC: the C kind = N1D's argv (the real 15-column channel, the same npz + sha) + exactly `--o-code-skip on` (the skip from T's code into O's plan head). It reads the D channel too; no existing kind's argv moves, and a non-C kind
#: never names the flag, builds o_code or carries its probe.
C_KINDS = ("N1DC",)
C_FLAG = "--o-code-skip"
#: PREREG_V8F_N1DH: the H kind = N1D's argv (the real 15-column channel, the same npz + sha) + exactly `--aux-route cond --w-aux-route 0.5` (the auxiliary route-decision head on O's plan feature whose detached posterior conditions the plan
#: head; the card's weight, untuned; `ce` is not built). It reads the D channel too, never names `--o-code-skip` or `--rb-window-w`; no existing kind's argv moves, and a non-H kind never names the flags, builds the modules or carries the probe.
H_KINDS = ("N1DH",)
H_FLAGS = ("--aux-route", "--w-aux-route")
H_WEIGHT = "0.5"
#: PREREG_V8F_N1DM: the M kind = N1D's argv (the real 15-column channel, the same npz + sha) + exactly `--aux-route modes --w-aux-route 0.5` (route head + K = 3 zero-init residual plan branches selected softly by the detached posterior; no
#: route_emb). It reads the D channel too, never names `--o-code-skip` or `--rb-window-w`; no existing kind's argv moves (N1DH still renders `cond`), and a non-M kind never builds route_br or carries the modes probe.
M_KINDS = ("N1DM",)
M_MODE = "modes"
DCHAN_KINDS = D_KINDS + W_KINDS + C_KINDS + H_KINDS + M_KINDS
RB_FEAT_FLAG = "--nav-rb-feat"
RBD_NPZ, RBD_REPORT = "navtext_nav_rbd.npz", "navtext_nav_rbd_report.json"                  # the 15-column table and its builder report (build_navtext_nav_rbd.py), in <inputs>/strat1/
RBD_GROUPS = ("c0", "c1_6", "c7_12", "c13", "c14")
RB_KINDS = tuple(RB_BASE)
RB_FLAGS = ("--nav-rb", "--nav-rb-npz", "--nav-rb-sha256", "--nav-rb-variant", "--nav-rb-lr-mult")


def rb_names(kind):
    """The channel table and builder-report file names of an R / D kind (D kinds read the 15-column table)."""
    return (RBD_NPZ, RBD_REPORT) if kind in DCHAN_KINDS else ("navtext_nav_rb.npz", "navtext_nav_rb_report.json")


RB_LR_MULT = "10"                                                         # card section 0, item 2 (MM ruling 2)
NAV_KINDS = ("N1", "S1", "S1d") + RB_KINDS                                # the arms that read the text nav
CTX_KINDS = ("S1", "S1d", "S1R", "S1dR")                                  # the arms with the ctx heads on; S1d = S1 with --ctx-detach on (PREREG_V8F_STRAT1 SEEDS read COSTS-HEADS)
SEEDS = {"s0": {"base": "OBD-CTL_s0"}, "s1": {"base": "OBD-CTL_s1"}, "rep": {"base": "OBD-CTL_s0", "seed": "2"},      # rep = init 0, bank s0, dagger-seed 0, seed 2 (OBD-NULL_rep's form)
         "s10": {"base": "OBD-CTL_s0", "seed": "10"}, "s11": {"base": "OBD-CTL_s1", "seed": "11"}}      # PREREG_V8F_STRAT1 SEEDS-DISAGREE: seed 10 = init 0 (bank s0), seed 11 = init 1 (bank s1), rep's form; PI go 2026-11-16
RUN_STEPS = 3000
MOVED = {"adapter": ("adapter", "pos", "time"), "S": ("read_s", "head_s"), "T": ("read_t", "trunk_t", "h_goal", "h_lat", "h_lon", "h_cons"), "O": ("read_o", "head_o"), "WM": ("wm.",)}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def flag_value(toks, name):
    return toks[toks.index(name) + 1] if name in toks else None


def base_kind(kind):
    """The STRAT1 kind an R kind extends (N1R -> N1); a STRAT1 kind is its own base."""
    return RB_BASE.get(kind, kind)


def render(kind, seed, root, out, nav_npz="", ctx_npz="", base_argvs=BASE_ARGVS, cache_dir=None, rb_npz=""):
    """The frozen argv of an arm: the OBD-CTL argv of that seed, box-2 paths mapped under ``root`` (the pod layout: ``root/runs/v8f_mini_full/X`` -> ``root/X``, the tracks next to ``root``, the cache at ``cache_dir``), plus the nine
    STRAT1 flags (ALL named, the control included). An R kind (PREREG_V8F_NAVRB2) renders its base kind's argv byte for byte and APPENDS the five --nav-rb flags (``rb_npz`` = the channel table); no other kind's argv moves."""
    assert kind in KINDS and seed in SEEDS
    base = json.load(open(base_argvs, encoding="utf-8"))[SEEDS[seed]["base"]]["argv"]
    r = root.rstrip("/")
    BOX2 = "D:" + chr(92) + "tanitad"
    t = [x.replace(BOX2, r).replace(chr(92), "/") if isinstance(x, str) and x.startswith(BOX2) else x for x in base]
    t = [x.replace(r + "/runs/v8f_mini_full/", r + "/") if isinstance(x, str) else x for x in t]
    t = [r + "/native_tracks.pt" if x == r + "/cache_full/native_tracks.pt" else x for x in t]
    if cache_dir:
        t = [cache_dir if x == r + "/cache_full" else x for x in t]
    t[t.index("--out") + 1] = out
    if "seed" in SEEDS[seed]:
        t[t.index("--seed") + 1] = SEEDS[seed]["seed"]
    bk = base_kind(kind)
    nav = ["--nav-source", "text", "--nav-text-npz", nav_npz, "--nav-text-sha256", sha_file(nav_npz) if os.path.exists(nav_npz) else "<sha of navtext_nav.npz>"] if bk in NAV_KINDS else         ["--nav-source", "v9", "--nav-text-npz", "", "--nav-text-sha256", ""]
    ctx = ["--ctx-heads", "on", "--w-ctx", "0.5", "--ctx-labels", ctx_npz, "--ctx-labels-sha256", STRAT_LABELS_SHA256, "--ctx-lr-mult", "10", "--ctx-detach", "on" if bk == "S1d" else "off"] if bk in CTX_KINDS else         ["--ctx-heads", "off", "--w-ctx", "0", "--ctx-labels", "", "--ctx-labels-sha256", "", "--ctx-lr-mult", "1", "--ctx-detach", "off"]
    rb = ["--nav-rb", "on", "--nav-rb-npz", rb_npz, "--nav-rb-sha256", sha_file(rb_npz) if rb_npz and os.path.exists(rb_npz) else "<sha of navtext_nav_rb.npz>",
          "--nav-rb-variant", "zero" if kind == "N1Z" else "real", "--nav-rb-lr-mult", RB_LR_MULT] if kind in RB_KINDS else []
    if kind == "N1Dx":
        rb[rb.index("--nav-rb-variant") + 1] = "shuffle"                       # PREREG_V8F_NAVRBD: the deliberate-regression arm
    if kind in DCHAN_KINDS:
        rb += [RB_FEAT_FLAG, "exitdist"]                                       # PREREG_V8F_NAVRBD: the ONE new flag, appended after the N1R flags (an N1R render is untouched)
    if kind in W_KINDS:
        rb += [RB_W_FLAG, W_VALUE[kind]]                                       # PREREG_V8F_N1DW: the ONE new flag of a W kind, appended after N1D's (an N1D render is untouched)
    if kind in C_KINDS:
        rb += [C_FLAG, "on"]                                                   # PREREG_V8F_N1DC: the ONE new flag of the C kind, appended after N1D's (an N1D render is untouched)
    if kind in H_KINDS:
        rb += [H_FLAGS[0], "cond", H_FLAGS[1], H_WEIGHT]                       # PREREG_V8F_N1DH: the two flags of the H kind, appended after N1D's (an N1D render is untouched)
    if kind in M_KINDS:
        rb += [H_FLAGS[0], M_MODE, H_FLAGS[1], H_WEIGHT]                       # PREREG_V8F_N1DM: the two flags of the M kind (same flags as N1DH, the mode differs), appended after N1D's
    return t + nav + ctx + rb


def pairing_problems(kind, toks, ref_toks):
    """PREREG_V8F_NAVRBD pairing rule: a D kind's argv equals its paired N1R argv PLUS exactly the new flag (--nav-rb-feat exitdist), a NEW channel npz + sha (the 15-column table) and its own --out; N1Dx also moves
    --nav-rb-variant to shuffle. Every other token (seed, steps, data, lr, the N1 flags, the other four --nav-rb flags) must be equal, in order. ``ref_toks`` = the banked N1R argv of the same seed."""
    if kind not in D_KINDS:
        return [f"{kind} is not a D kind"]
    bad, toks, ref = [], list(toks), list(ref_toks)
    if RB_FEAT_FLAG in ref:
        bad.append("the reference (N1R) argv names --nav-rb-feat (N1R's argv is the 1-column channel and does not)")
    if toks.count(RB_FEAT_FLAG) != 1 or flag_value(toks, RB_FEAT_FLAG) != "exitdist":
        return bad + [f"the {kind} argv must name {RB_FEAT_FLAG} exactly once, as exitdist"]
    i = toks.index(RB_FEAT_FLAG)
    toks = toks[:i] + toks[i + 2:]
    if len(toks) != len(ref):
        return bad + [f"argv length {len(toks)} (without the feat flag) != the N1R argv's {len(ref)}"]
    free = {"--out", "--nav-rb-npz", "--nav-rb-sha256"} | ({"--nav-rb-variant"} if kind == "N1Dx" else set())
    diff = []
    for j, (x, y) in enumerate(zip(toks, ref)):
        if x != y and not (j > 0 and toks[j - 1] in free and ref[j - 1] == toks[j - 1]):
            diff.append((j, y, x))
    if diff:
        bad.append(f"the {kind} argv differs from N1R's beyond the new flag / npz / sha / out: {diff[:4]}")
    if flag_value(toks, "--nav-rb-sha256") in (None, "", flag_value(ref, "--nav-rb-sha256")):
        bad.append("--nav-rb-sha256 is missing or equals the N1R channel's (a D arm reads the NEW 15-column table)")
    if flag_value(toks, "--nav-rb-npz") in (None, "", flag_value(ref, "--nav-rb-npz")):
        bad.append("--nav-rb-npz is missing or equals the N1R channel's")
    if flag_value(toks, "--nav-rb-variant") != ("shuffle" if kind == "N1Dx" else flag_value(ref, "--nav-rb-variant")):
        bad.append(f"--nav-rb-variant {flag_value(toks, '--nav-rb-variant')!r} (N1D = the reference's {flag_value(ref, '--nav-rb-variant')!r}, N1Dx = shuffle)")
    return bad


def w_pairing_problems(kind, toks, ref_toks):
    """PREREG_V8F_N1DW pairing rule: a W kind's argv equals its paired N1D argv (the banked one of the same seed) PLUS exactly ``--rb-window-w <w>`` (w = 4 for N1DW4, 2 for N1DW2), with its own --out and nothing else different:
    the same seed, steps, data, lr, nav flags, channel npz + sha, variant and feat. ``ref_toks`` = the banked N1D argv."""
    if kind not in W_KINDS:
        return [f"{kind} is not a W kind"]
    bad, toks, ref = [], list(toks), list(ref_toks)
    if RB_W_FLAG in ref:
        bad.append(f"the reference (N1D) argv names {RB_W_FLAG} (N1D's argv is the unweighted arm and does not)")
    if flag_value(ref, RB_FEAT_FLAG) != "exitdist":
        bad.append("the reference argv is not an N1D argv (--nav-rb-feat exitdist missing)")
    if toks.count(RB_W_FLAG) != 1 or flag_value(toks, RB_W_FLAG) != W_VALUE[kind]:
        return bad + [f"the {kind} argv must name {RB_W_FLAG} exactly once, as {W_VALUE[kind]}"]
    i = toks.index(RB_W_FLAG)
    toks = toks[:i] + toks[i + 2:]
    if len(toks) != len(ref):
        return bad + [f"argv length {len(toks)} (without the weight flag) != the N1D argv's {len(ref)}"]
    diff = [(j, y, x) for j, (x, y) in enumerate(zip(toks, ref)) if x != y and not (j > 0 and toks[j - 1] == "--out" and ref[j - 1] == "--out")]
    if diff:
        bad.append(f"the {kind} argv differs from N1D's beyond the weight flag / out: {diff[:4]}")
    return bad


def w_gate_problems(kind, toks, sc):
    """PREREG_V8F_N1DW G-DVB / G-LIVE read of a selfcheck.json: a W kind names the flag once with its kind's value, builds exactly that value (levers.rb_window_w), and carries the probe's controls (term > 0, O's head gradient > 0,
    the all-False mask EXACTLY 0.0 in term, total and gradient, the repeat control EXACTLY 0.0). A non-W kind names no --rb-window-w, builds no such lever and carries no probe."""
    bad = []
    lv = sc.get("levers", {})
    if kind not in W_KINDS:
        if RB_W_FLAG in toks:
            bad.append(f"a {kind} argv names {RB_W_FLAG}")
        if "rb_window_w" in lv or "rb_window_probe" in sc or "rb_window" in lv:
            bad.append(f"a {kind} arm builds or probes the roundabout-window weight")
        return bad
    want = float(W_VALUE[kind])
    if toks.count(RB_W_FLAG) != 1 or flag_value(toks, RB_W_FLAG) != W_VALUE[kind]:
        bad.append(f"{RB_W_FLAG}: arm {kind} needs it named once as {W_VALUE[kind]}, argv {flag_value(toks, RB_W_FLAG)!r} (a flag whose default disables it must be named)")
    if lv.get("rb_window_w") != want:
        bad.append(f"levers.rb_window_w: arm {kind} needs {want}, built {lv.get('rb_window_w')!r}")
    pr = sc.get("rb_window_probe") or {}
    if not pr.get("pass"):
        bad.append("rb_window_probe missing or not pass")
    if pr.get("rb_window_w_declared") != want:
        bad.append(f"rb_window_probe.rb_window_w_declared {pr.get('rb_window_w_declared')!r} != {want}")
    for key in ("term_forced", "head_o_grad_diff"):
        v = pr.get(key)
        if not isinstance(v, (int, float)) or not v > 0.0:
            bad.append(f"rb_window_probe.{key} {v!r} (must be > 0)")
    for key in ("control_allfalse_term", "control_allfalse_total_diff", "control_allfalse_grad_diff", "control_repeat_total_diff", "control_repeat_grad_diff", "control_prior_grad_diff"):
        if pr.get(key) != 0.0:
            bad.append(f"rb_window_probe.{key} {pr.get(key)!r} (must be exactly 0.0)")
    if pr.get("term_at_unit_weight") is not None:
        bad.append("rb_window_probe: the term is computed at w = 1.0 (a declared-off lever must add no term)")
    return bad


def c_pairing_problems(kind, toks, ref_toks):
    """PREREG_V8F_N1DC pairing rule: the C kind's argv equals its paired N1D argv (the banked one of the same seed) PLUS exactly ``--o-code-skip on``, with its own --out and nothing else different: the same seed, steps, data, lr, nav
    flags, channel npz + sha, variant and feat. ``ref_toks`` = the banked N1D argv."""
    if kind not in C_KINDS:
        return [f"{kind} is not a C kind"]
    bad, toks, ref = [], list(toks), list(ref_toks)
    if C_FLAG in ref:
        bad.append(f"the reference (N1D) argv names {C_FLAG} (N1D's argv is the arm without the skip and does not)")
    if RB_W_FLAG in ref or RB_W_FLAG in toks:
        bad.append(f"an argv names {RB_W_FLAG} (N1DC runs at the weight's default, which builds nothing)")
    if flag_value(ref, RB_FEAT_FLAG) != "exitdist":
        bad.append("the reference argv is not an N1D argv (--nav-rb-feat exitdist missing)")
    if toks.count(C_FLAG) != 1 or flag_value(toks, C_FLAG) != "on":
        return bad + [f"the {kind} argv must name {C_FLAG} exactly once, as on"]
    i = toks.index(C_FLAG)
    toks = toks[:i] + toks[i + 2:]
    if len(toks) != len(ref):
        return bad + [f"argv length {len(toks)} (without the skip flag) != the N1D argv's {len(ref)}"]
    diff = [(j, y, x) for j, (x, y) in enumerate(zip(toks, ref)) if x != y and not (j > 0 and toks[j - 1] == "--out" and ref[j - 1] == "--out")]
    if diff:
        bad.append(f"the {kind} argv departs from N1D's beyond the skip flag / out: {diff[:4]}")
    return bad


def c_gate_problems(kind, toks, sc):
    """PREREG_V8F_N1DC G-DVB / G-LIVE read of a selfcheck.json: the C kind names ``--o-code-skip on`` once (a flag whose default disables it must be named), builds it (levers.o_code_skip), and carries the probe's controls (o_code.weight gradient
    > 0 from the total and from O's ade, the OCODE group's G-LIVE gradient > 0, the identity at the zero init, the plan moves when the weight is set, and every exact-0.0 control). A non-C kind names none, builds none and carries no probe."""
    bad = []
    lv, live = sc.get("levers", {}), sc.get("live", {})
    if kind not in C_KINDS:
        if C_FLAG in toks:
            bad.append(f"a {kind} argv names {C_FLAG}")
        if "o_code_skip" in lv or "o_code_probe" in sc:
            bad.append(f"a {kind} arm builds or probes the o_code skip")
        return bad
    if toks.count(C_FLAG) != 1 or flag_value(toks, C_FLAG) != "on":
        bad.append(f"{C_FLAG}: arm {kind} needs it named once as on, argv {flag_value(toks, C_FLAG)!r} (a flag whose default disables it must be named)")
    if lv.get("o_code_skip") != "on":
        bad.append(f"levers.o_code_skip: arm {kind} needs on, built {lv.get('o_code_skip')!r}")
    pr = sc.get("o_code_probe") or {}
    if not pr.get("pass"):
        bad.append("o_code_probe missing or not pass")
    for key in ("grad_total", "grad_ade", "plan_move_when_weight_set"):
        v = pr.get(key)
        if not isinstance(v, (int, float)) or not v > 0.0:
            bad.append(f"o_code_probe.{key} {v!r} (must be > 0)")
    for key in ("control_tside_grad", "control_repeat_grad_diff", "control_prior_grad", "control_ade_grad_on_upper"):      # the last: the N1D recipe is --detach-upper, so O's ade must put exactly 0.0 on S / T
        if pr.get(key) != 0.0:
            bad.append(f"o_code_probe.{key} {pr.get(key)!r} (must be exactly 0.0)")
    for key in ("identity_terms_differing", "identity_outputs_differing"):
        if pr.get(key) != []:
            bad.append(f"o_code_probe.{key} {pr.get(key)!r} (must be an empty list: at the zero init the model is the parent's)")
    if not (live.get("groups_grad") or {}).get("OCODE", 0.0) > 0.0:
        bad.append("G-LIVE: the OCODE group has no gradient")
    return bad


def h_pairing_problems(kind, toks, ref_toks):
    """PREREG_V8F_N1DH pairing rule: the H kind's argv equals its paired N1D argv (the banked one of the same seed) PLUS exactly ``--aux-route cond --w-aux-route 0.5``, with its own --out and nothing else different: the same seed, steps,
    data, lr, nav flags, channel npz + sha, variant and feat. Neither argv names ``--o-code-skip`` or ``--rb-window-w``. ``ref_toks`` = the banked N1D argv."""
    if kind not in H_KINDS:
        return [f"{kind} is not an H kind"]
    bad, toks, ref = [], list(toks), list(ref_toks)
    for fl in (*H_FLAGS, C_FLAG, RB_W_FLAG):
        if fl in ref:
            bad.append(f"the reference (N1D) argv names {fl} (N1D's argv is the arm without it and does not)")
    for fl in (C_FLAG, RB_W_FLAG):
        if fl in toks:
            bad.append(f"the {kind} argv names {fl} (N1DH runs at that lever's default, which builds nothing)")
    if flag_value(ref, RB_FEAT_FLAG) != "exitdist":
        bad.append("the reference argv is not an N1D argv (--nav-rb-feat exitdist missing)")
    if toks.count(H_FLAGS[0]) != 1 or flag_value(toks, H_FLAGS[0]) != "cond" or toks.count(H_FLAGS[1]) != 1 or flag_value(toks, H_FLAGS[1]) != H_WEIGHT:
        return bad + [f"the {kind} argv must name {H_FLAGS[0]} exactly once as cond and {H_FLAGS[1]} exactly once as {H_WEIGHT}"]
    for fl in H_FLAGS:
        i = toks.index(fl)
        toks = toks[:i] + toks[i + 2:]
    if len(toks) != len(ref):
        return bad + [f"argv length {len(toks)} (without the two route flags) != the N1D argv's {len(ref)}"]
    diff = [(j, y, x) for j, (x, y) in enumerate(zip(toks, ref)) if x != y and not (j > 0 and toks[j - 1] == "--out" and ref[j - 1] == "--out")]
    if diff:
        bad.append(f"the {kind} argv departs from N1D's beyond the two route flags / out: {diff[:4]}")
    return bad


def h_gate_problems(kind, toks, sc):
    """PREREG_V8F_N1DH G-DVB / G-LIVE read of a selfcheck.json: the H kind names ``--aux-route cond`` and ``--w-aux-route 0.5`` once each (a flag whose default disables it must be named), builds them (levers.aux_route / w_aux_route), and
    carries the probe's controls (route_head.weight CE gradient > 0, route_emb.weight gradient > 0 with head_o live, the identity at the zero init, the plan moves when the embedding is set, and every exact-0.0 control: the decoder term on
    route_head (the detach), the T/S-side terms, the repeat, the prior, the batch with no own-code row), the ROUTE group's G-LIVE gradient and the ``route`` term. A non-H kind names none, builds none and carries no probe."""
    bad = []
    lv, live = sc.get("levers", {}), sc.get("live", {})
    if kind in M_KINDS:
        return bad                                                             # PREREG_V8F_N1DM: the M kind is read by m_gate_problems
    if kind not in H_KINDS:
        for fl in H_FLAGS:
            if fl in toks:
                bad.append(f"a {kind} argv names {fl}")
        if "aux_route" in lv or "w_aux_route" in lv or "aux_route_probe" in sc:
            bad.append(f"a {kind} arm builds or probes the aux-route head")
        return bad
    if toks.count(H_FLAGS[0]) != 1 or flag_value(toks, H_FLAGS[0]) != "cond":
        bad.append(f"{H_FLAGS[0]}: arm {kind} needs it named once as cond, argv {flag_value(toks, H_FLAGS[0])!r} (a flag whose default disables it must be named)")
    if toks.count(H_FLAGS[1]) != 1 or flag_value(toks, H_FLAGS[1]) != H_WEIGHT:
        bad.append(f"{H_FLAGS[1]}: arm {kind} needs it named once as {H_WEIGHT} (the card's weight), argv {flag_value(toks, H_FLAGS[1])!r}")
    if lv.get("aux_route") != "cond":
        bad.append(f"levers.aux_route: arm {kind} needs cond, built {lv.get('aux_route')!r}")
    if lv.get("w_aux_route") != float(H_WEIGHT):
        bad.append(f"levers.w_aux_route: arm {kind} needs {H_WEIGHT}, built {lv.get('w_aux_route')!r}")
    pr = sc.get("aux_route_probe") or {}
    if not pr.get("pass"):
        bad.append("aux_route_probe missing or not pass")
    for key in ("head_grad_total", "head_grad_route", "emb_grad_total", "emb_grad_ade", "plan_move_when_emb_set", "own_rows_probe"):
        v = pr.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not v > 0.0:
            bad.append(f"aux_route_probe.{key} {v!r} (must be > 0)")
    for key in ("control_decoder_grad_on_head", "control_tside_grad_on_head", "control_tside_grad_on_emb", "control_repeat_grad_diff", "control_prior_grad", "control_no_own_row_route_term", "control_no_own_row_head_grad"):
        if pr.get(key) != 0.0:
            bad.append(f"aux_route_probe.{key} {pr.get(key)!r} (must be exactly 0.0)")
    for key in ("identity_outputs_differing", "identity_terms_differing"):
        if pr.get(key) != []:
            bad.append(f"aux_route_probe.{key} {pr.get(key)!r} (must be an empty list: at the zero init the model is the parent's)")
    if not (live.get("groups_grad") or {}).get("ROUTE", 0.0) > 0.0:
        bad.append("G-LIVE: the ROUTE group has no gradient")
    if "route" not in (live.get("terms") or {}):
        bad.append("G-LIVE: the route loss term is not among the terms the loop logged")
    return bad


def hyg_h():
    """PREREG_V8F_N1DH G-HYG: the config refuses the unbuilt ce mode, a non-finite or negative weight, cond with a per-unit code mix or a cut interface, and an unknown key."""
    from tanitad.models import v8f_mini as M
    n = 0
    for kw in (dict(aux_route="ce"), dict(aux_route="cond", w_aux_route=float("nan")), dict(aux_route="cond", w_aux_route=-1.0), dict(aux_route="cond", w_aux_route=0.5, code_mix="kind"),
               dict(aux_route="cond", w_aux_route=0.5, o_interface="cut"), dict(aux_routez="cond")):
        try:
            M.V8FMiniConfig(**kw)
        except (ValueError, TypeError):
            n += 1
    return [] if n == 6 else [f"only {n} of 6 bad aux_route configs were refused"]


def m_pairing_problems(kind, toks, ref_toks):
    """PREREG_V8F_N1DM pairing rule: the M kind's argv equals its paired N1D argv (the banked one of the same seed) PLUS exactly ``--aux-route modes --w-aux-route 0.5``, with its own --out and nothing else different: the same seed, steps,
    data, lr, nav flags, channel npz + sha, variant and feat. Neither argv names ``--o-code-skip`` or ``--rb-window-w``; the reference names no route flag. (N1DH's own argv, ``--aux-route cond``, is NOT an M reference.)"""
    if kind not in M_KINDS:
        return [f"{kind} is not an M kind"]
    bad, toks, ref = [], list(toks), list(ref_toks)
    for fl in (*H_FLAGS, C_FLAG, RB_W_FLAG):
        if fl in ref:
            bad.append(f"the reference (N1D) argv names {fl} (N1D's argv is the arm without it and does not)")
    for fl in (C_FLAG, RB_W_FLAG):
        if fl in toks:
            bad.append(f"the {kind} argv names {fl} (N1DM runs at that lever's default, which builds nothing)")
    if flag_value(ref, RB_FEAT_FLAG) != "exitdist":
        bad.append("the reference argv is not an N1D argv (--nav-rb-feat exitdist missing)")
    if toks.count(H_FLAGS[0]) != 1 or flag_value(toks, H_FLAGS[0]) != M_MODE or toks.count(H_FLAGS[1]) != 1 or flag_value(toks, H_FLAGS[1]) != H_WEIGHT:
        return bad + [f"the {kind} argv must name {H_FLAGS[0]} exactly once as {M_MODE} and {H_FLAGS[1]} exactly once as {H_WEIGHT}"]
    for fl in H_FLAGS:
        i = toks.index(fl)
        toks = toks[:i] + toks[i + 2:]
    if len(toks) != len(ref):
        return bad + [f"argv length {len(toks)} (without the two route flags) != the N1D argv's {len(ref)}"]
    diff = [(j, y, x) for j, (x, y) in enumerate(zip(toks, ref)) if x != y and not (j > 0 and toks[j - 1] == "--out" and ref[j - 1] == "--out")]
    if len(diff):
        bad.append(f"the {kind} argv departs from N1D's beyond the two route flags / out: {diff[:4]}")
    return bad


def m_gate_problems(kind, toks, sc):
    """PREREG_V8F_N1DM G-DVB / G-LIVE read of a selfcheck.json: the M kind names ``--aux-route modes`` and ``--w-aux-route 0.5`` once each (a flag whose default disables it must be named), builds them (levers.aux_route / w_aux_route), and
    carries the probe's controls: route_head.weight CE gradient > 0; EVERY one of the K branches a gradient > 0 at the real init AND with seeded weights; the plan moves when the branches are set AND when p is permuted across rows; the identity at
    the zero init (the plan and every loss term but route / total are the model-without-the-head's); and every exact-0.0 control (the decoder term on route_head: the detach; the T/S-side terms; the repeat; the prior; the batch with no own-code row);
    the ROUTE group's G-LIVE gradient and the ``route`` term. A non-M kind is read by the H gate, which refuses any aux_route flag, lever or probe on it."""
    bad = []
    lv, live = sc.get("levers", {}), sc.get("live", {})
    pr = sc.get("aux_route_probe") or {}
    if kind not in M_KINDS:
        return bad                                                             # a non-M kind: the H / D / W / C gates read it (the H gate refuses any aux_route flag, lever or probe on a non-H kind)
    if toks.count(H_FLAGS[0]) != 1 or flag_value(toks, H_FLAGS[0]) != M_MODE:
        bad.append(f"{H_FLAGS[0]}: arm {kind} needs it named once as {M_MODE}, argv {flag_value(toks, H_FLAGS[0])!r} (a flag whose default disables it must be named)")
    if flag_value(toks, H_FLAGS[1]) != H_WEIGHT or toks.count(H_FLAGS[1]) != 1:
        bad.append(f"{H_FLAGS[1]}: arm {kind} needs it named once as {H_WEIGHT} (the card's weight), argv {flag_value(toks, H_FLAGS[1])!r}")
    if lv.get("aux_route") != M_MODE:
        bad.append(f"levers.aux_route: arm {kind} needs {M_MODE}, built {lv.get('aux_route')!r}")
    if float(lv.get("w_aux_route", -1.0)) != float(H_WEIGHT):
        bad.append(f"levers.w_aux_route: arm {kind} needs {H_WEIGHT}, built {lv.get('w_aux_route')!r}")
    if not pr.get("pass"):
        bad.append("aux_route_probe missing or not pass")
    for key in ("head_grad_total", "head_grad_route", "emb_grad_total", "emb_grad_ade", "plan_move_when_emb_set", "plan_move_when_p_permuted", "own_rows_probe"):
        v = pr.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not v > 0.0:
            bad.append(f"aux_route_probe.{key} {v!r} (must be > 0)")
    for key in ("real_init_emb_each", "emb_each"):
        v = pr.get(key)
        if not (isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0.0 for x in v)):
            bad.append(f"aux_route_probe.{key} {v!r} (must be 3 values, every one > 0: each branch reaches the loss)")
    zero_keys = ("control_decoder_grad_on_head", "control_tside_grad_on_head", "control_tside_grad_on_emb", "control_repeat_grad_diff", "control_prior_grad", "control_no_own_row_route_term", "control_no_own_row_head_grad")
    for key in zero_keys:
        if pr.get(key) != 0.0:
            bad.append(f"aux_route_probe.{key} {pr.get(key)!r} (must be exactly 0.0)")
    ident_keys = ("identity_outputs_differing", "identity_terms_differing")
    for key in ident_keys:
        if pr.get(key) != []:
            bad.append(f"aux_route_probe.{key} {pr.get(key)!r} (must be an empty list: at the zero init the model is the parent's)")
    grp = (live.get("groups_grad") or {}).get("ROUTE", 0.0)
    if not grp > 0.0:
        bad.append("G-LIVE: the ROUTE group has no gradient")
    if "route" not in set(live.get("terms") or {}):
        bad.append("G-LIVE: the route loss term is not among the terms the loop logged")
    return bad


def hyg_m():
    """PREREG_V8F_N1DM G-HYG: the config refuses the unbuilt ce mode, an unknown mode, a non-finite or negative weight, modes with a per-unit code mix or a cut interface, and an unknown key."""
    from tanitad.models import v8f_mini as M
    n = 0
    for kw in (dict(aux_route="ce"), dict(aux_route="mode"), dict(aux_route="modes", w_aux_route=float("nan")), dict(aux_route="modes", w_aux_route=-1.0), dict(aux_route="modes", w_aux_route=0.5, code_mix="kind"),
               dict(aux_route="modes", w_aux_route=0.5, o_interface="cut"), dict(aux_routez="modes")):
        try:
            M.V8FMiniConfig(**kw)
        except (ValueError, TypeError):
            n += 1
    return [] if n == 7 else [f"only {n} of 7 bad aux_route (modes) configs were refused"]


def hyg_c():
    """PREREG_V8F_N1DC G-HYG: the config refuses a non-bool skip flag and an unknown key."""
    from tanitad.models import v8f_mini as M
    n = 0
    for kw in (dict(o_code_skip="on"), dict(o_code_skip=1), dict(o_code_skipz=True)):
        try:
            M.V8FMiniConfig(**kw)
        except (ValueError, TypeError):
            n += 1
    return [] if n == 3 else [f"only {n} of 3 bad o_code_skip configs were refused"]


def hyg_w():
    """PREREG_V8F_N1DW G-HYG: the config refuses a weight below 1, a non-finite weight and an unknown key."""
    from tanitad.models import v8f_mini as M
    n = 0
    for kw in (dict(rb_window_w=0.5), dict(rb_window_w=float("nan")), dict(rb_window_w=float("inf")), dict(rb_window_wz=2.0)):
        try:
            M.V8FMiniConfig(**kw)
        except (ValueError, TypeError):
            n += 1
    return [] if n == 4 else [f"only {n} of 4 bad rb_window_w configs were refused"]


def argv_sha(toks):
    return hashlib.sha256(json.dumps(list(toks)).encode("utf-8")).hexdigest()


def gate_problems(kind, toks, sc, steps=RUN_STEPS):
    """The G-DVB + G-LIVE read of a selfcheck.json against its argv. [] = pass."""
    bad = []
    bk = base_kind(kind)
    miss = [f for f in NINE if f not in toks]
    if miss:
        bad.append(f"the argv does not name {miss}: a flag whose default disables it must be named")
    if sc.get("declared_vs_built") != "PASS":
        bad.append("declared_vs_built is not PASS")
    lv, live = sc.get("levers", {}), sc.get("live", {})
    if not live.get("pass"):
        bad.append("G-LIVE (live) did not pass")
    if sc.get("steps_total") != steps or flag_value(toks, "--steps") != str(steps):
        bad.append(f"steps_total {sc.get('steps_total')} / argv --steps {flag_value(toks, '--steps')} != {steps}")
    nsp = sc.get("nav_source_probe") or {}
    if not nsp.get("pass"):
        bad.append("nav_source_probe missing or not pass")
    want_src = "text" if bk in NAV_KINDS else "v9"
    if nsp.get("source") != want_src or flag_value(toks, "--nav-source") != want_src:
        bad.append(f"nav source: arm {kind} needs {want_src}, probe read {nsp.get('source')}, argv {flag_value(toks, '--nav-source')}")
    if (nsp.get("n_text_ne_v9", 0) > 0) != (want_src == "text"):
        bad.append(f"n_text_ne_v9 {nsp.get('n_text_ne_v9')} is inconsistent with source {want_src}")
    on = bk in CTX_KINDS
    want_det = "on" if bk == "S1d" else "off"
    if flag_value(toks, "--ctx-detach") != want_det or lv.get("ctx_detach", "off") != want_det:
        bad.append(f"ctx_detach: arm {kind} needs {want_det}, argv {flag_value(toks, '--ctx-detach')}, built {lv.get('ctx_detach', 'off')}")
    if lv.get("ctx_heads", "off") != ("on" if on else "off") or flag_value(toks, "--ctx-heads") != ("on" if on else "off"):
        bad.append(f"ctx_heads: arm {kind} needs {'on' if on else 'off'}, built {lv.get('ctx_heads', 'off')}")
    if on:
        cp = sc.get("ctx_head_probe") or {}
        if not cp.get("pass"):
            bad.append("ctx_head_probe missing or not pass")
        if cp.get("control_head_grad_w0") != 0.0 or cp.get("control_allmasked_head_grad") != 0.0 or cp.get("control_adapter_grad_detach_on") != 0.0:
            bad.append("a ctx exact-0.0 control is not exactly 0.0")
        ag = cp.get("adapter_grad_ctx")                                  # S1: the ctx loss reaches the shared adapter (> 0); S1d: exactly 0.0 (the detach)
        if not isinstance(ag, (int, float)) or (ag != 0.0 if bk == "S1d" else not ag > 0.0):
            bad.append(f"adapter_grad_ctx {ag!r}: arm {kind} needs " + ("exactly 0.0" if bk == "S1d" else "> 0.0"))
        if not (live.get("groups_grad") or {}).get("CTX", 0.0) > 0.0:
            bad.append("G-LIVE: the CTX group has no gradient")
        if float(lv.get("w_ctx", 0)) != float(flag_value(toks, "--w-ctx")) or lv.get("ctx_dim") != 41 or not lv.get("ctx_vision_only_source"):
            bad.append("built ctx levers (w_ctx / ctx_dim 41 / vision-only source) differ from the argv")
        if (sc.get("ctx_labels") or {}).get("sha256") != STRAT_LABELS_SHA256:
            bad.append("the loaded ctx label file is not sha 8055a321")
    elif "ctx_head_probe" in sc or "ctx_head." in json.dumps(sc.get("levers", {})):
        bad.append("a ctx-off arm carries a ctx probe / lever")
    for k in ("acc_probe", "dagger_probe", "otband_probe"):                 # the inherited probes (OBD-CTL's gate read them)
        if not isinstance(sc.get(k), dict) or sc[k].get("skipped"):
            bad.append(f"inherited probe {k} is missing or skipped")
    if "--detach-upper" not in toks or (sc.get("otband_probe") or {}).get("detach_upper") is not True:      # the BUILT value: the otband probe records cfg.detach_upper and, when it is on,
        bad.append("--detach-upper is not built (the OBD-CTL recipe)")                                       # asserts O's terms put no gradient on S / T (levers has no such key: reading it refused every real mint)
    bad += rb_gate_problems(kind, toks, sc)
    bad += w_gate_problems(kind, toks, sc)                                   # PREREG_V8F_N1DW: the W kinds' weight (a non-W kind names none and builds none)
    bad += c_gate_problems(kind, toks, sc)                                   # PREREG_V8F_N1DC: the C kind's skip (a non-C kind names none, builds none and carries no probe)
    bad += m_gate_problems(kind, toks, sc)                                   # PREREG_V8F_N1DM: the M kind's modes head (a non-M kind builds no route_br and carries no modes probe)
    bad += h_gate_problems(kind, toks, sc)                                   # PREREG_V8F_N1DH: the H kind's route head (a non-H kind names none, builds none and carries no probe)
    return bad


def rb_gate_problems(kind, toks, sc):
    """PREREG_V8F_NAVRB2 G-DVB / G-LIVE for the channel: an R kind names all five flags, builds the lever, and reads the probe's controls (gradient > 0, the zero channel EXACTLY 0.0, the step-0 identity); N1Z reads the all-zero
    channel (gradient EXACTLY 0.0 and the weight declared dead). A non-R kind names none, builds nothing and carries no probe."""
    bad = []
    lv, live, info = sc.get("levers", {}), sc.get("live", {}), sc.get("info") or {}
    if kind not in RB_KINDS:
        if any(f in toks for f in RB_FLAGS) or RB_FEAT_FLAG in toks:
            bad.append(f"a {kind} argv names a --nav-rb flag")
        if lv.get("nav_rb", "off") != "off" or "navrb_probe" in sc or info.get("nav_rb") is not None or "nav_rb_feat" in lv:
            bad.append(f"a {kind} arm builds or probes the nav_rb channel")
        return bad
    miss = [f for f in RB_FLAGS if f not in toks]
    if miss:
        bad.append(f"the argv does not name {miss}: a flag whose default disables it must be named")
    dk = kind in DCHAN_KINDS
    if dk and RB_FEAT_FLAG not in toks:
        bad.append(f"the argv does not name {RB_FEAT_FLAG}: a flag whose default disables it must be named (a {kind} arm is the exitdist channel)")
    if dk and RB_FEAT_FLAG in toks and flag_value(toks, RB_FEAT_FLAG) != "exitdist":
        bad.append(f"nav_rb_feat: arm {kind} needs exitdist, argv {flag_value(toks, RB_FEAT_FLAG)}")
    if dk and lv.get("nav_rb_feat") != "exitdist":
        bad.append(f"levers.nav_rb_feat: arm {kind} needs exitdist, built {lv.get('nav_rb_feat')!r}")
    if not dk and flag_value(toks, RB_FEAT_FLAG) == "exitdist":
        bad.append(f"{kind} is an R kind: it refuses --nav-rb-feat exitdist (the 15-column channel is the D kinds')")
    if not dk and lv.get("nav_rb_feat", "state") != "state":
        bad.append(f"{kind}: built nav_rb_feat {lv.get('nav_rb_feat')!r} (an R kind builds the 1-column channel)")
    zero = kind == "N1Z"
    want_var = "zero" if zero else ("shuffle" if kind == "N1Dx" else "real")
    if flag_value(toks, "--nav-rb") != "on" or lv.get("nav_rb") != "on":
        bad.append(f"nav_rb: arm {kind} needs on, argv {flag_value(toks, '--nav-rb')}, built {lv.get('nav_rb')}")
    if flag_value(toks, "--nav-rb-variant") != want_var:
        bad.append(f"nav_rb_variant: arm {kind} needs {want_var}, argv {flag_value(toks, '--nav-rb-variant')}")
    if flag_value(toks, "--nav-rb-lr-mult") != RB_LR_MULT:
        bad.append(f"nav_rb_lr_mult: the card says {RB_LR_MULT}, argv {flag_value(toks, '--nav-rb-lr-mult')}")
    rec = info.get("nav_rb") or {}
    if rec.get("sha256") != flag_value(toks, "--nav-rb-sha256") or not rec.get("sha256"):
        bad.append("info.nav_rb.sha256 differs from the argv --nav-rb-sha256 (or is missing)")
    if rec.get("variant") != want_var:
        bad.append(f"info.nav_rb.variant {rec.get('variant')!r} != {want_var}")
    if zero and rec.get("n_on") != 0:
        bad.append(f"N1Z: the loaded channel has {rec.get('n_on')} rows on (must be 0)")
    if not zero and not (rec.get("n_on") or 0) > 0:
        bad.append(f"{kind}: the loaded channel has no row on")
    if dk:
        if rec.get("feat") != "exitdist" or rec.get("width") != 15:
            bad.append(f"info.nav_rb.feat {rec.get('feat')!r} / width {rec.get('width')!r} (need exitdist / 15)")
        gno = rec.get("group_n_on") or {}
        if set(gno) != set(RBD_GROUPS) or any(not (gno.get(g) or 0) > 0 for g in RBD_GROUPS):
            bad.append(f"info.nav_rb.group_n_on {gno!r}: every column group needs n_on > 0 in the loaded array")
        if kind == "N1Dx" and not rec.get("donor_map_sha256"):
            bad.append("N1Dx: info.nav_rb has no donor_map_sha256")
    pr = sc.get("navrb_probe") or {}
    if not pr.get("pass"):
        bad.append("navrb_probe missing or not pass")
    if dk:                                                                     # PREREG_V8F_NAVRBD Z2: the gradient PER COLUMN GROUP, the zero control exactly 0.0 per group, the batch's own counts printed
        if pr.get("feat") != "exitdist" or pr.get("width") != 15:
            bad.append(f"navrb_probe.feat {pr.get('feat')!r} / width {pr.get('width')!r} (need exitdist / 15)")
        for key, need in (("group_grads_planted", "gt"), ("group_control_zero_channel", "zero"), ("group_grads", "have"), ("group_n_on", "have")):
            gv = pr.get(key)
            if not isinstance(gv, dict) or set(gv) != set(RBD_GROUPS):
                bad.append(f"navrb_probe.{key} {gv!r}: needs the five groups {RBD_GROUPS}")
            elif need == "gt":
                lowg = [g for g in RBD_GROUPS if not (isinstance(gv[g], (int, float)) and gv[g] > 0.0)]
                if lowg:
                    bad.append(f"navrb_probe.{key}: no gradient on column groups {lowg} (must be > 0 on every group)")
            elif need == "zero":
                nzg = [g for g in RBD_GROUPS if gv[g] != 0.0]
                if nzg:
                    bad.append(f"navrb_probe.{key}: the zero-channel control reads non-zero on {nzg} (must be exactly 0.0 per group)")
        gg, gn, unread = pr.get("group_grads"), pr.get("group_n_on"), set(pr.get("group_unread_in_batch") or [])
        if isinstance(gg, dict) and isinstance(gn, dict) and set(gg) == set(gn) == set(RBD_GROUPS):
            dead = [g for g in RBD_GROUPS if (gn[g] or 0) > 0 and not gg[g] > 0.0 and g not in unread]
            if dead:
                bad.append(f"navrb_probe: groups {dead} have non-zero rows in the real batch but no gradient and are not named unread")
    if pr.get("step0_identity") is not True:
        bad.append(f"navrb_probe.step0_identity {pr.get('step0_identity')!r} (must be True at the zero init)")
    if pr.get("keys_read_has_nav_rb") is not True:
        bad.append("navrb_probe: forward_infer does not read nav_rb")
    g = pr.get("e_nav_rb_grad")
    if not isinstance(g, (int, float)) or not g > 0.0:
        bad.append(f"navrb_probe.e_nav_rb_grad {g!r} (must be > 0)")
    if pr.get("control_e_nav_rb_grad_zero_channel") != 0.0:
        bad.append(f"navrb_probe zero-channel control {pr.get('control_e_nav_rb_grad_zero_channel')!r} (must be exactly 0.0)")
    ng = (live.get("groups_grad") or {}).get("NAVRB")
    if zero:
        if ng != 0.0 or "e_nav_rb.weight" not in (live.get("declared_dead") or []):
            bad.append(f"N1Z: live NAVRB group grad {ng!r} (must be exactly 0.0) and e_nav_rb.weight must be declared dead")
    elif not isinstance(ng, (int, float)) or not ng > 0.0:
        bad.append(f"G-LIVE: the NAVRB group has no gradient ({ng!r})")
    return bad


def group_weight_norms(w):
    """PREREG_V8F_NAVRBD: the L1 norm of each column group of a 15-column e_nav_rb.weight (a group whose column was never non-zero in training reads exactly 0.0: named in the report, not a failure)."""
    return {g: float(w[:, lo:hi].abs().sum()) for g, (lo, hi) in {"c0": (0, 1), "c1_6": (1, 7), "c7_12": (7, 13), "c13": (13, 14), "c14": (14, 15)}.items()}


def moved_identity(init_sd, fin_sd, kind):
    """The post check: every non-ctx group moved from the init; ``ctx_head.*`` exists and moved iff S1 / S1d; no ctx key in C0 / N1."""
    import torch
    bad = []
    for g, pre in MOVED.items():
        ks = [k for k in fin_sd if k.startswith(pre) and k in init_sd]
        if ks and all(torch.equal(init_sd[k], fin_sd[k]) for k in ks):
            bad.append(f"group {g} did not move")
    ck = [k for k in fin_sd if k.startswith("ctx_head.")]
    rb = [k for k in fin_sd if k.startswith("e_nav_rb.")]                      # PREREG_V8F_NAVRB2: the channel weight starts at EXACTLY 0.0 (it is not in the M2b init)
    if kind in RB_KINDS:
        w = fin_sd.get("e_nav_rb.weight")
        if w is None:
            bad.append(f"{kind} checkpoint has no e_nav_rb.weight")
        elif kind == "N1Z":
            if not bool((w == 0).all()):
                bad.append("N1Z: e_nav_rb.weight moved off 0.0 (the channel was all zero)")
        elif not bool((w != 0).any()):
            bad.append(f"{kind}: e_nav_rb.weight did not move off its zero init")
        if w is not None and kind in DCHAN_KINDS and tuple(w.shape)[1:] != (15,):
            bad.append(f"{kind}: e_nav_rb.weight is {tuple(w.shape)} (needs [d, 15])")
        if w is not None and kind not in DCHAN_KINDS and tuple(w.shape)[1:] != (1,):
            bad.append(f"{kind}: e_nav_rb.weight is {tuple(w.shape)} (needs [d, 1])")
    elif rb:
        bad.append(f"e_nav_rb keys {rb[:2]} in a {kind} checkpoint")
    oc = [k for k in fin_sd if k.startswith("o_code.")]                         # PREREG_V8F_N1DC: the skip's weight starts at EXACTLY 0.0 (it is not in the M2b init) and must have moved in a C kind; no other kind has it
    if kind in C_KINDS:
        w = fin_sd.get("o_code.weight")
        if w is None:
            bad.append(f"{kind} checkpoint has no o_code.weight")
        else:
            if not bool((w != 0).any()):
                bad.append(f"{kind}: o_code.weight did not move off its zero init")
            if len(tuple(w.shape)) != 2 or int(w.shape[1]) != 60 or ("head_o.0.weight" in fin_sd and int(w.shape[0]) != int(fin_sd["head_o.0.weight"].shape[1])):
                bad.append(f"{kind}: o_code.weight is {tuple(w.shape)} (needs [n_q * d, 60], n_q * d = head_o's input width)")
    elif oc:
        bad.append(f"o_code keys {oc[:2]} in a {kind} checkpoint")
    rr = [k for k in fin_sd if k.startswith(("route_head.", "route_emb.", "route_br."))]     # PREREG_V8F_N1DH: route_emb starts at EXACTLY 0.0 (not in the M2b init) and must have moved in an H kind; route_head starts at its seeded default and must have moved; no other kind has them
    if kind in H_KINDS:
        if any(k.startswith("route_br.") for k in fin_sd):
            bad.append(f"{kind} checkpoint carries route_br keys (the cond arm has no plan branches)")
        wh, bh, we = fin_sd.get("route_head.weight"), fin_sd.get("route_head.bias"), fin_sd.get("route_emb.weight")
        if wh is None or bh is None or we is None:
            bad.append(f"{kind} checkpoint lacks route_head.weight / route_head.bias / route_emb.weight")
        else:
            if not bool((we != 0).any()):
                bad.append(f"{kind}: route_emb.weight did not move off its zero init")
            if len(tuple(wh.shape)) != 2 or int(wh.shape[0]) != 3 or len(tuple(we.shape)) != 2 or int(we.shape[1]) != 3 or int(we.shape[0]) != int(wh.shape[1]) or tuple(bh.shape) != (3,) or (
                    "head_o.0.weight" in fin_sd and int(we.shape[0]) != int(fin_sd["head_o.0.weight"].shape[1])):
                bad.append(f"{kind}: route modules are {tuple(wh.shape)} / {tuple(bh.shape)} / {tuple(we.shape)} (need [3, n_q * d] / [3] / [n_q * d, 3], n_q * d = head_o's input width)")
    elif kind in M_KINDS:                                               # PREREG_V8F_N1DM: route_head (seeded default, moved) + K = 3 branches that START at EXACTLY 0.0 and must have moved; NO route_emb
        wh, bh = fin_sd.get("route_head.weight"), fin_sd.get("route_head.bias")
        brs = [fin_sd.get(f"route_br.{k}.weight") for k in range(3)]
        if wh is None or bh is None or any(w is None for w in brs):
            bad.append(f"{kind} checkpoint lacks route_head.weight / route_head.bias / route_br.0..2.weight")
        else:
            for k, w in enumerate(brs):
                if not bool((w != 0).any()):
                    bad.append(f"{kind}: route_br.{k}.weight did not move off its zero init")
            do = int(fin_sd["head_o.2.weight"].shape[0]) if "head_o.2.weight" in fin_sd else None
            if len(tuple(wh.shape)) != 2 or int(wh.shape[0]) != 3 or tuple(bh.shape) != (3,) or any(len(tuple(w.shape)) != 2 or int(w.shape[1]) != int(wh.shape[1]) or (do is not None and int(w.shape[0]) != do) for w in brs) or (
                    "head_o.0.weight" in fin_sd and int(wh.shape[1]) != int(fin_sd["head_o.0.weight"].shape[1])):
                bad.append(f"{kind}: route modules are {tuple(wh.shape)} / {tuple(bh.shape)} / {[tuple(w.shape) for w in brs]} (need [3, n_q * d] / [3] / 3 x [head_o's output width, n_q * d])")
        if any(k.startswith("route_emb.") for k in fin_sd):
            bad.append(f"{kind} checkpoint carries route_emb keys (the modes arm has no conditioning embedding)")
    elif rr:
        bad.append(f"route keys {rr[:2]} in a {kind} checkpoint")
    if base_kind(kind) in CTX_KINDS:
        if not ck:
            bad.append(f"{kind} checkpoint has no ctx_head keys")
        else:                                                           # the head's seeded init (CTX_INIT_SEED, forked RNG) is the fresh start state: it must have moved
            from tanitad.models import v8f_mini as M
            d = int(fin_sd["ctx_head.0.weight"].shape[1])
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(M.CTX_INIT_SEED)
                fresh = M._mlp(d, d, M.CTX_DIM).state_dict()
            if all(torch.equal(fresh[k[len("ctx_head."):]], fin_sd[k]) for k in ck):
                bad.append("ctx_head did not move from its seeded init")
    elif ck:
        bad.append(f"ctx_head keys {ck[:2]} in a {kind} checkpoint")
    return bad


def geval(ckpt_path, selfcheck_levers=None):
    """G-EVAL: load through the evaluators' own build path (``V8FMiniConfig(**ck['cfg'])`` -> ``V8FMini`` -> STRICT ``load_state_dict``); built levers equal the selfcheck's; the deliberate-regression arm (a cfg that drops
    ``ctx_heads`` from a ctx checkpoint) must be REFUSED. Returns (problems, record)."""
    import torch
    from tanitad.models import v8f_mini as M
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bad, rec = [], {}

    def build(cfgd):
        m = M.V8FMini(M.V8FMiniConfig(**cfgd), M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
        m.load_state_dict(ck["model"])
        return m
    try:
        m = build(ck["cfg"])
    except Exception as e:                                              # noqa: BLE001
        return [f"strict load failed: {type(e).__name__}: {str(e)[:200]}"], rec
    lv = m.built_levers()
    rec["built_levers"] = {k: lv[k] for k in lv if k.startswith(("ctx", "w_ctx", "nav_rb", "o_code", "aux_route", "w_aux_route"))}
    if selfcheck_levers is not None:
        for k in [k for k in set(lv) | set(selfcheck_levers) if k.startswith(("ctx", "w_ctx")) or k in ("navonly", "wm", "detach_upper", "nav_rb", "nav_rb_feat", "o_code_skip", "aux_route", "w_aux_route")]:
            if lv.get(k) != selfcheck_levers.get(k):
                bad.append(f"built lever {k}: evaluator {lv.get(k)!r} vs selfcheck {selfcheck_levers.get(k)!r}")
    if ck["cfg"].get("ctx_heads") == "on":
        try:
            build({**ck["cfg"], "ctx_heads": "off", "w_ctx": 0.0})
            bad.append("regression arm: a cfg that drops ctx_heads was NOT refused (the evaluator would silently skip the head)")
        except Exception:                                               # noqa: BLE001
            rec["regression_arm_refused"] = True
    if ck["cfg"].get("nav_rb") == "on":                                  # PREREG_V8F_NAVRB2: a cfg that drops the channel from a channel checkpoint must be refused (strict load: unexpected e_nav_rb.weight)
        try:
            build({**ck["cfg"], "nav_rb": "off"})
            bad.append("regression arm: a cfg that drops nav_rb was NOT refused (the evaluator would silently skip the channel)")
        except Exception:                                               # noqa: BLE001
            rec["regression_arm_nav_rb_refused"] = True
    if ck["cfg"].get("nav_rb") == "on" and ck["cfg"].get("nav_rb_feat", "state") != "state":      # PREREG_V8F_NAVRBD (G-GEN): a cfg that drops the feat from a 15-column checkpoint must be refused (size mismatch of e_nav_rb.weight)
        try:
            build({**ck["cfg"], "nav_rb_feat": "state"})
            bad.append("regression arm: a cfg that drops nav_rb_feat was NOT refused (the evaluator would build the 1-column channel)")
        except Exception:                                               # noqa: BLE001
            rec["regression_arm_nav_rb_feat_refused"] = True
    if ck["cfg"].get("aux_route", "off") != "off":                        # PREREG_V8F_N1DH: a cfg that drops the head from a route checkpoint must be refused (strict load: unexpected route_head.weight)
        try:
            build({**ck["cfg"], "aux_route": "off", "w_aux_route": 0.0})
            bad.append("regression arm: a cfg that drops aux_route was NOT refused (the evaluator would silently skip the head)")
        except Exception:                                               # noqa: BLE001
            rec["regression_arm_aux_route_refused"] = True
    if ck["cfg"].get("o_code_skip"):                                     # PREREG_V8F_N1DC: a cfg that drops the skip from a skip checkpoint must be refused (strict load: unexpected o_code.weight)
        try:
            build({**ck["cfg"], "o_code_skip": False})
            bad.append("regression arm: a cfg that drops o_code_skip was NOT refused (the evaluator would silently skip the path)")
        except Exception:                                               # noqa: BLE001
            rec["regression_arm_o_code_skip_refused"] = True
    return bad, rec


def hyg_rb():
    """PREREG_V8F_NAVRB2 G-HYG: the channel's config refuses a bad mode and an unknown key."""
    from tanitad.models import v8f_mini as M
    n = 0
    for kw in (dict(nav_rb="maybe"), dict(nav_rbz="on"), dict(nav_rb_feat="wide")):
        try:
            M.V8FMiniConfig(**kw)
        except (ValueError, TypeError):
            n += 1
    return [] if n == 3 else [f"only {n} of 3 bad nav_rb configs were refused"]


def hyg():
    from tanitad.models import v8f_mini as M
    n = 0
    for kw in (dict(ctx_heads="maybe"), dict(ctx_detach="x"), dict(w_ctx=-1.0), dict(w_ctx=float("nan")), dict(nav_source="osm")):
        try:
            M.V8FMiniConfig(**kw)
        except ValueError:
            n += 1
    try:
        M.V8FMiniConfig(ctx_headz="on")
    except TypeError:
        n += 1
    return [] if n == 6 else [f"only {n} of 6 bad configs were refused"]


# ---- PREREG_V8F_STRAT1 launch gate: token, tree manifest, data bindings, G-SUITE result, mint -------------------------------------------------------------------------------------------------
TOKEN_SCHEMA = "strat1-gate-token-v1"
WRITTEN_BY = "strat1_check.py mint"
DATA_FLAGS = ("--v9-train", "--lead-train", "--speed-train", "--join-train", "--p10", "--tracks", "--nav-text-npz", "--ctx-labels", "--init-ckpt")
DATA_FLAGS_RB = ("--nav-rb-npz",)                                           # PREREG_V8F_NAVRB2: bound only when the argv names it (an R kind); kept out of DATA_FLAGS so no STRAT1 fixture or token moves
REQUIRED_CHECKS = ("G-HYG", "G-DVB", "G-LIVE", "G-CLOCK", "G-EVAL", "G-CKPT", "G-SUITE")
TREE_DIRS = (("stack", "tanitad"), ("stack", "scripts"), ("stack", "tests"), ("taniteval",))
TREE_EXT = (".py", ".txt", ".json", ".sh")
TRAINER_REL = os.path.join("stack", "scripts", "v8f_mini_train.py")
EVAL_GRID_REL = ("TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini", "raw", "eval_grid_windows.json")


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_entry(p):
    """size, sha256 and md5 of one input file (None when absent)."""
    if not p or not os.path.isfile(p):
        return None
    return {"size": os.path.getsize(p), "sha256": sha_file(p), "md5": md5_file(p)}


def tree_manifest(root):
    """sha256 of every code / test file the run executes (``stack/tanitad``, ``stack/scripts``, ``stack/tests``, ``taniteval``; no caches), keyed by the path relative to ``root``."""
    out = {}
    for parts in TREE_DIRS:
        base = os.path.join(root, *parts)
        for dp, dn, fn in os.walk(base):
            dn[:] = sorted(d for d in dn if d != "__pycache__")
            for f in sorted(fn):
                if f.endswith(TREE_EXT):
                    full = os.path.join(dp, f)
                    out[os.path.relpath(full, root).replace(os.sep, "/")] = sha_file(full)
    return out


def tree_manifest_sha(root):
    return hashlib.sha256(json.dumps(tree_manifest(root), sort_keys=True).encode("utf-8")).hexdigest()


def data_binding(toks):
    """Every data input the argv names: its size, sha256 and md5 (recomputed now). Returns (dict, problems). A named path that does not exist is a problem; an empty value is not named."""
    d, bad = {}, []
    for fl in DATA_FLAGS + DATA_FLAGS_RB:
        v = flag_value(toks, fl)
        if not v:
            continue
        e = hash_entry(v)
        if e is None:
            bad.append(f"{fl} {v!r} is not a file")
        else:
            d[fl] = {"path": v, **e}
    bank = flag_value(toks, "--dagger-bank")
    if bank:
        if not os.path.isdir(bank):
            bad.append(f"--dagger-bank {bank!r} is not a directory")
        else:
            for f in sorted(os.listdir(bank)):
                if f.endswith((".npz", ".json")):
                    d["bank/" + f] = {"path": os.path.join(bank, f), **hash_entry(os.path.join(bank, f))}
            bj = os.path.join(bank, "bank.json")
            if not os.path.isfile(bj):
                bad.append("the dagger bank has no bank.json")
            elif flag_value(toks, "--dagger-bank-sha256") != sha_file(bj):
                bad.append("--dagger-bank-sha256 differs from the sha256 of bank.json")
    cache = flag_value(toks, "--cache-dir")
    if cache:
        for split in ("train", "eval"):
            m = os.path.join(cache, split, "manifest.jsonl")
            e = hash_entry(m)
            if e is None:
                bad.append(f"the cache has no {split}/manifest.jsonl ({m})")
            else:
                d[f"cache/{split}/manifest.jsonl"] = {"path": m, **e}
    nav = flag_value(toks, "--nav-text-npz")
    if nav and flag_value(toks, "--nav-text-sha256") != d.get("--nav-text-npz", {}).get("sha256"):
        bad.append("--nav-text-sha256 differs from the sha256 of the nav npz")
    cl = flag_value(toks, "--ctx-labels")
    if cl and flag_value(toks, "--ctx-labels-sha256") != d.get("--ctx-labels", {}).get("sha256"):
        bad.append("--ctx-labels-sha256 differs from the sha256 of the ctx label file")
    if flag_value(toks, "--nav-rb-npz") and flag_value(toks, "--nav-rb-sha256") != d.get("--nav-rb-npz", {}).get("sha256"):
        bad.append("--nav-rb-sha256 differs from the sha256 of the nav_rb npz")
    return d, bad


def suite_problems(path, commit):
    """The G-SUITE result file written by ``suite``: 0 regressions of the new tree against the base, for this commit. Returns (sha256 of the file or None, problems)."""
    try:
        r = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None, ["no G-SUITE result file"]
    bad = []
    if r.get("schema") != "strat1-gsuite-result-v1" or r.get("written_by") != "strat1_check.py suite":
        bad.append("the G-SUITE result file was not written by the checker")
    if r.get("n_regressions") != 0 or r.get("regressions"):
        bad.append(f"G-SUITE has {r.get('n_regressions')} regressions")
    if not r.get("new", {}).get("n_tests") or not r.get("base", {}).get("n_tests"):
        bad.append("the G-SUITE result has an empty side")
    if r.get("new_commit") != commit:
        bad.append(f"the G-SUITE result is for commit {r.get('new_commit')}, not {commit}")
    return sha_file(path), bad


def nav_report_problems(kind, nav_npz, report_path):
    """G-CLOCK-NAV: the nav builder's report must read PASS on train and eval, its regression arm must FAIL, and it must describe this npz."""
    if base_kind(kind) not in NAV_KINDS:
        return []
    try:
        r = json.load(open(report_path, encoding="utf-8"))
    except (OSError, ValueError):
        return ["no nav builder report"]
    bad = []
    for sp in ("train", "eval"):
        if not r.get(sp, {}).get("G-CLOCK-NAV_pass"):
            bad.append(f"G-CLOCK-NAV {sp} is not pass")
        if r.get(sp, {}).get("G-CLOCK-NAV_regression_arm_plus_travelled", {}).get("pass") is not False:
            bad.append(f"G-CLOCK-NAV {sp} regression arm did not fail as required")
    if r.get("navtext_nav_sha256") != sha_file(nav_npz):
        bad.append("the nav report describes a different npz")
    return bad


def rb_report_problems(kind, rb_npz, report_path):
    """PREREG_V8F_NAVRB2 G-CLOCK-RB: the channel builder's report must read PASS on train and eval, its regression arm (state read 5 steps = 1 s ahead) must FAIL, and it must describe this npz. Only R kinds read it."""
    if kind not in RB_KINDS:
        return []
    try:
        r = json.load(open(report_path, encoding="utf-8"))
    except (OSError, ValueError):
        return ["no nav_rb builder report"]
    if kind in DCHAN_KINDS:                                                     # PREREG_V8F_NAVRBD: the D builder's own clock check (card section 6, G-CLOCK-RBD), same layout, same rule: PASS and the 1-s-ahead regression arm FAILS
        bad = []
        for sp in ("train", "eval"):
            if not r.get(sp, {}).get("G-CLOCK-RBD_pass"):
                bad.append(f"G-CLOCK-RBD {sp} is not pass")
            if r.get(sp, {}).get("G-CLOCK-RBD_regression_arm_shift_5", {}).get("pass") is not False:
                bad.append(f"G-CLOCK-RBD {sp} regression arm did not fail as required")
        if r.get("table_sha256") != sha_file(rb_npz):
            bad.append("the nav_rbd report describes a different npz")
        return bad
    bad = []
    for sp in ("train", "eval"):
        if not r.get(sp, {}).get("G-CLOCK-RB_pass"):
            bad.append(f"G-CLOCK-RB {sp} is not pass")
        if r.get(sp, {}).get("G-CLOCK-RB_regression_arm_shift_5", {}).get("pass") is not False:
            bad.append(f"G-CLOCK-RB {sp} regression arm did not fail as required")
    if r.get("navtext_nav_rb_sha256") != sha_file(rb_npz):
        bad.append("the nav_rb report describes a different npz")
    return bad


def binding(kind, toks, root, commit, suite_path, trainer=None, nav_report="", rb_report=""):
    """Everything the token binds, recomputed NOW. Returns (dict, problems)."""
    d, bad = data_binding(toks)
    tr = trainer or os.path.join(root, TRAINER_REL)
    grid = os.path.join(root, *EVAL_GRID_REL)                                 # the trainer's DAGGER_EVAL_GRID (K-B3 refuses a launch without it): a data input of the tree, bound when present
    if os.path.isfile(grid):
        d["eval_grid_windows.json"] = {"path": grid, **hash_entry(grid)}
    ssha, sbad = suite_problems(suite_path, commit)
    bad = list(bad) + sbad
    if nav_report:
        bad += nav_report_problems(kind, flag_value(toks, "--nav-text-npz"), nav_report)
    if kind in RB_KINDS:
        bad += rb_report_problems(kind, flag_value(toks, "--nav-rb-npz"), rb_report) if rb_report else ["no nav_rb builder report named for an R kind"]
    b = {"kind": kind, "commit": commit, "argv_sha256": argv_sha(toks), "data": d, "tree_manifest_sha256": tree_manifest_sha(root),
         "trainer_sha256": sha_file(tr) if os.path.isfile(tr) else None, "gsuite_result_sha256": ssha,
         "checker_sha256": sha_file(os.path.abspath(__file__)),
         "nav_report_sha256": sha_file(nav_report) if nav_report and os.path.isfile(nav_report) else None}
    if kind in RB_KINDS:                                                       # R kinds only: a STRAT1 kind's token body keeps its key set
        b["nav_rb_report_sha256"] = sha_file(rb_report) if rb_report and os.path.isfile(rb_report) else None
    if b["trainer_sha256"] is None:
        bad.append("trainer not found")
    return b, bad


def body_sha(tok):
    return hashlib.sha256(json.dumps({k: v for k, v in tok.items() if k != "body_sha256"}, sort_keys=True).encode("utf-8")).hexdigest()


def write_token(path, b, checks, selfcheck_sha):
    """The ONLY writer of a token (called by ``mint`` after every required check read PASS). ``checks`` maps each required check to a record whose ``pass`` is True."""
    miss = [c for c in REQUIRED_CHECKS if not (checks.get(c) or {}).get("pass")]
    if miss:
        raise ValueError(f"refusing to write a token: checks not PASS: {miss}")
    tok = {"schema": TOKEN_SCHEMA, "written_by": WRITTEN_BY, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "selfcheck_sha256": selfcheck_sha, "checks": checks, **b}
    tok["body_sha256"] = body_sha(tok)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tok, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return tok


def token_problems(token_path, kind, toks, root, commit, suite_path, selfcheck_path="", trainer=None, nav_report="", rb_report=""):
    """[] iff a token written by ``mint`` still binds this arm kind, commit, argv, every data file, the tree, the trainer, the checker and the G-SUITE result. Each differing key is named."""
    try:
        tok = json.load(open(token_path, encoding="utf-8"))
    except (OSError, ValueError):
        return ["no gate token"]
    p = []
    if tok.get("schema") != TOKEN_SCHEMA or tok.get("written_by") != WRITTEN_BY:
        p.append("token not written by the checker")
    if tok.get("body_sha256") != body_sha(tok):
        p.append("token body was edited (body_sha256)")
    miss = [c for c in REQUIRED_CHECKS if not (tok.get("checks", {}).get(c) or {}).get("pass")]
    if miss:
        p.append(f"token checks not PASS: {miss}")
    want, bad = binding(kind, toks, root, commit, suite_path, trainer, nav_report, rb_report)
    p += list(bad)
    for k, v in want.items():
        if k == "data":
            for name in sorted(set(v) | set(tok.get("data", {}))):
                a, b2 = v.get(name), tok.get("data", {}).get(name)
                if a != b2:
                    p.append(f"token data {name} differs from the current file")
        elif tok.get(k) != v:
            p.append(f"token {k} differs from the current {k}")
    if selfcheck_path and (not os.path.isfile(selfcheck_path) or sha_file(selfcheck_path) != tok.get("selfcheck_sha256")):
        p.append("selfcheck.json differs from the token's")
    return p


# ---- G-SUITE comparison ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def junit_results(d):
    """{test id: 'pass'|'fail'|'error'|'skip'} over every shard xml of a junit directory."""
    import glob
    import xml.etree.ElementTree as ET
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.xml"))):
        for tc in ET.parse(f).getroot().iter("testcase"):
            tid = (tc.get("classname") or "") + "::" + (tc.get("name") or "")
            st = "pass"
            for ch in tc:
                if ch.tag in ("failure", "error", "skipped"):
                    st = {"failure": "fail", "error": "error", "skipped": "skip"}[ch.tag]
                    break
            out[tid] = st
    return out


def suite_compare(base_dir, new_dir, new_commit, base_commit):
    b, n = junit_results(base_dir), junit_results(new_dir)
    bad = ("fail", "error")
    reg = sorted(t for t, s in n.items() if s in bad and b.get(t) == "pass")
    newonly = sorted(t for t in n if t not in b)
    return {"schema": "strat1-gsuite-result-v1", "written_by": "strat1_check.py suite", "base_commit": base_commit, "new_commit": new_commit,
            "base": {"n_tests": len(b), "n_pass": sum(s == "pass" for s in b.values()), "n_fail": sum(s == "fail" for s in b.values()), "n_error": sum(s == "error" for s in b.values()), "n_skip": sum(s == "skip" for s in b.values())},
            "new": {"n_tests": len(n), "n_pass": sum(s == "pass" for s in n.values()), "n_fail": sum(s == "fail" for s in n.values()), "n_error": sum(s == "error" for s in n.values()), "n_skip": sum(s == "skip" for s in n.values())},
            "n_regressions": len(reg), "regressions": reg,
            "new_test_failures": sorted(t for t in newonly if n[t] in bad),
            "fail_on_both": sorted(t for t, s in n.items() if s in bad and b.get(t) in bad),
            "missing_in_new": sorted(t for t in b if t not in n), "n_new_only": len(newonly)}


# ---- mint: the checker runs the trainer's own selfcheck and a short resume run, reads every check, and only then writes the token ----------------------------------------------------------
def _run(py, trainer, toks, cwd, env_extra, log):
    import subprocess
    env = dict(os.environ)
    env.update(env_extra)
    with open(log, "ab") as lf:
        return subprocess.run([py, trainer] + list(toks), cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode


def _with(toks, **kv):
    t = list(toks)
    for k, v in kv.items():
        f = "--" + k.replace("_", "-")
        if f in t:
            t[t.index(f) + 1] = str(v)
        else:
            t += [f, str(v)]
    return t


def arm_tag(kind, seed, steps):
    return f"{kind}_{seed}" if steps == RUN_STEPS else f"pilot{steps}_{kind}_{seed}"


def mint(tag, kind, seed, a):
    """Mint ``<runs>/<tag>/gate/gate_token.json``. Returns the list of problems (empty = token written)."""
    rd = os.path.join(a.runs, tag)
    gd = os.path.join(rd, "gate")
    out = os.path.join(rd, "train")
    os.makedirs(gd, exist_ok=True)
    frozen = os.path.join(a.inputs, "frozen", "argvs_obd.json")             # the frozen OBD-CTL argvs, shipped to the pod with the other inputs (the dev-box path is not there)
    toks = render(kind, seed, a.inputs, out, os.path.join(a.inputs, "strat1", "navtext_nav.npz"), os.path.join(a.inputs, "strat1", "strat_labels_table.npz"), base_argvs=frozen, cache_dir=a.cache_dir,
                  rb_npz=os.path.join(a.inputs, "strat1", rb_names(kind)[0]) if kind in RB_KINDS else "")
    if a.steps != RUN_STEPS:                                                  # a timing pilot: the SAME argv with --steps moved (nothing else), its own token
        toks = _with(toks, steps=a.steps)
    json.dump(toks, open(os.path.join(rd, "argv.json"), "w", encoding="utf-8"))
    nav_report = os.path.join(a.inputs, "strat1", "navtext_nav_report.json")
    rb_report = os.path.join(a.inputs, "strat1", rb_names(kind)[1]) if kind in RB_KINDS else ""
    b, bad = binding(kind, toks, a.root, a.commit, a.suite, nav_report=nav_report if base_kind(kind) in NAV_KINDS else "", rb_report=rb_report)
    if bad:
        return ["binding: " + x for x in bad]
    trainer = os.path.join(a.root, TRAINER_REL)
    env = {"PYTHONPATH": os.path.join(a.root, "stack") + os.pathsep + a.root, "CUDA_VISIBLE_DEVICES": a.gpu, "HF_HUB_DISABLE_XET": "1"}
    checks = {"G-SUITE": {"pass": True, "result_sha256": b["gsuite_result_sha256"]}, "frozen_argvs_sha256": {"pass": True, "sha256": sha_file(frozen)}}
    h = hyg() + (hyg_rb() if kind in RB_KINDS else []) + (hyg_w() if kind in W_KINDS else []) + (hyg_c() if kind in C_KINDS else []) + (hyg_h() if kind in H_KINDS else []) + (hyg_m() if kind in M_KINDS else [])
    checks["G-HYG"] = {"pass": not h, "problems": h}
    if base_kind(kind) in NAV_KINDS:
        checks["G-CLOCK"] = {"pass": True, "nav_report_sha256": b["nav_report_sha256"], "note": "G-CLOCK-NAV read from the builder report (train + eval PASS, regression arm FAILS); ctx label alignment asserts run inside the trainer's selfcheck"}
        if kind in RB_KINDS:
            checks["G-CLOCK"]["nav_rb_report_sha256"] = b["nav_rb_report_sha256"]
            checks["G-CLOCK"]["note"] += "; G-CLOCK-RB read from the nav_rb builder report (train + eval PASS, regression arm FAILS)"
    else:
        checks["G-CLOCK"] = {"pass": True, "note": "control: no new label clock; the inherited label clock is the OBD-CTL recipe's"}
    # selfcheck-only on the exact argv (--out moved to the gate folder)
    gsc = os.path.join(gd, "selfcheck_run")
    shutil.rmtree(gsc, ignore_errors=True)
    glog = os.path.join(gd, "gate.log")
    rc = _run(a.py, trainer, _with(toks, out=gsc) + ["--selfcheck-only"], a.root, env, glog)
    scp = os.path.join(gsc, "selfcheck.json")
    if rc != 0 or not os.path.isfile(scp):
        return [f"the trainer's --selfcheck-only exited rc={rc} and {'wrote' if os.path.isfile(scp) else 'did not write'} selfcheck.json (log {glog})"]
    shutil.copy2(scp, os.path.join(gd, "selfcheck.json"))
    sc = json.load(open(scp, encoding="utf-8"))
    gp = gate_problems(kind, toks, sc, a.steps)
    dvb = sc.get("declared_vs_built") == "PASS" and not gp
    checks["G-DVB"] = {"pass": dvb, "declared_vs_built": sc.get("declared_vs_built"), "problems": gp}
    checks["G-LIVE"] = {"pass": bool((sc.get("live") or {}).get("pass")) and not gp, "problems": gp, "groups_grad": (sc.get("live") or {}).get("groups_grad")}
    # a short resume run on the same argv: G-CKPT (round trip, resume) and G-EVAL (the evaluators' build path)
    short = os.path.join(gd, "short")
    shutil.rmtree(short, ignore_errors=True)
    st = _with(toks, out=short, steps=4, ckpt_every=2, log_every=1)
    slog = os.path.join(gd, "short.log")
    import torch
    rc1 = _run(a.py, trainer, st, a.root, {**env, "V8F_TRAIN_CRASH_AT": "3"}, slog)
    ck = os.path.join(short, "ckpt_last.pt")
    ok1 = rc1 != 0 and os.path.isfile(ck)
    step1 = torch.load(ck, map_location="cpu", weights_only=False).get("step") if ok1 else None
    rc2 = _run(a.py, trainer, st, a.root, env, slog) if ok1 else -1
    fin = os.path.join(short, "ckpt_last.pt")
    ckrec = {"crash_rc": rc1, "resume_rc": rc2, "step_at_crash_ckpt": step1}
    if ok1 and rc2 == 0:
        c1 = torch.load(fin, map_location="cpu", weights_only=False)
        ckrec["final_step"] = c1.get("step")
        sd = c1["model"]
        torch.save(sd, os.path.join(gd, "rt.pt"))
        rt = torch.load(os.path.join(gd, "rt.pt"), map_location="cpu", weights_only=False)
        eq = set(rt) == set(sd) and all(torch.equal(rt[k], sd[k]) for k in sd)
        os.remove(os.path.join(gd, "rt.pt"))
        ckrec["roundtrip_bit_equal"] = bool(eq)
        init_sd = torch.load(flag_value(toks, "--init-ckpt"), map_location="cpu", weights_only=False)["model"]
        mv = moved_identity(init_sd, sd, kind)
        ckrec["moved_identity_problems"] = mv
        ge, rec = geval(fin, sc.get("levers"))
        checks["G-EVAL"] = {"pass": not ge, "problems": ge, **rec, "scope": "strict load through the evaluators' build path, built levers == selfcheck levers, regression arm (cfg drops ctx_heads) refused; the 8-window forward_infer parity is NOT run here"}
        checks["G-CKPT"] = {"pass": bool(eq) and not mv and ckrec.get("final_step") is not None and step1 is not None and int(ckrec["final_step"]) > int(step1), **ckrec}
    else:
        checks["G-EVAL"] = {"pass": False, "problems": ["no checkpoint to load"]}
        checks["G-CKPT"] = {"pass": False, **ckrec, "problems": ["the crash run did not leave a checkpoint, or the resume run failed (logs in the gate folder)"]}
    shutil.rmtree(short, ignore_errors=True)                                  # own temporary data
    shutil.rmtree(gsc, ignore_errors=True)
    failed = [c for c in REQUIRED_CHECKS if not (checks.get(c) or {}).get("pass")]
    json.dump(checks, open(os.path.join(gd, "checks.json"), "w", encoding="utf-8"), indent=1)
    if failed:
        return [f"check {c} did not pass: {json.dumps(checks.get(c))[:600]}" for c in failed]
    write_token(os.path.join(gd, "gate_token.json"), b, checks, sha_file(os.path.join(gd, "selfcheck.json")))
    return []


def run_arm(tag, kind, a):
    """The runner: REFUSE unless the token binds exactly the argv on disk, the commit, every data file and the tree; then start the trainer under the same argv. Returns the process return code (2 = refused)."""
    import subprocess
    rd = os.path.join(a.runs, tag)
    try:
        toks = json.load(open(os.path.join(rd, "argv.json"), encoding="utf-8"))
    except (OSError, ValueError):
        print(f"REFUSED {tag}: no argv.json")
        return 2
    nav_report = os.path.join(a.inputs, "strat1", "navtext_nav_report.json") if base_kind(kind) in NAV_KINDS else ""
    rb_report = os.path.join(a.inputs, "strat1", rb_names(kind)[1]) if kind in RB_KINDS else ""
    trainer = a.trainer or os.path.join(a.root, TRAINER_REL)
    bad = token_problems(os.path.join(rd, "gate", "gate_token.json"), kind, toks, a.root, a.commit, a.suite, os.path.join(rd, "gate", "selfcheck.json"), trainer, nav_report, rb_report)
    if bad:
        print(f"REFUSED {tag}: " + "; ".join(bad))
        return 2
    print(f"TOKEN-OK {tag}")
    env = dict(os.environ)
    env.update({"PYTHONPATH": os.path.join(a.root, "stack") + os.pathsep + a.root, "CUDA_VISIBLE_DEVICES": a.gpu, "HF_HUB_DISABLE_XET": "1"})
    os.makedirs(os.path.join(rd, "logs"), exist_ok=True)
    with open(os.path.join(rd, "logs", "train.log"), "ab") as lf:
        rc = subprocess.run([a.py, trainer] + list(toks), cwd=a.root, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("argv"); p.add_argument("kind", choices=KINDS); p.add_argument("seed", choices=tuple(SEEDS)); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--nav-npz", default=""); p.add_argument("--ctx-npz", default=""); p.add_argument("--rb-npz", default="")
    p = sub.add_parser("gate"); p.add_argument("kind", choices=KINDS); p.add_argument("--selfcheck", required=True); p.add_argument("--argv-json", required=True)
    p = sub.add_parser("post"); p.add_argument("kind", choices=KINDS); p.add_argument("--init", required=True); p.add_argument("--final", required=True)
    p = sub.add_parser("geval"); p.add_argument("--ckpt", required=True); p.add_argument("--selfcheck", default="")
    p = sub.add_parser("pair"); p.add_argument("kind", choices=D_KINDS + W_KINDS + C_KINDS + H_KINDS + M_KINDS); p.add_argument("--argv-json", required=True); p.add_argument("--ref-argv-json", required=True)
    sub.add_parser("hyg")
    for nm in ("mint", "run"):
        p = sub.add_parser(nm); p.add_argument("kind", choices=KINDS); p.add_argument("seed", choices=tuple(SEEDS))
        p.add_argument("--root", required=True, help="the exported tree (holds stack/ and taniteval/)"); p.add_argument("--inputs", required=True); p.add_argument("--cache-dir", required=True)
        p.add_argument("--runs", required=True); p.add_argument("--commit", required=True); p.add_argument("--suite", required=True, help="the G-SUITE result file written by `suite`")
        p.add_argument("--steps", type=int, default=RUN_STEPS); p.add_argument("--py", default=sys.executable); p.add_argument("--gpu", default="0"); p.add_argument("--trainer", default="")
    p = sub.add_parser("suite"); p.add_argument("--base-junit", required=True); p.add_argument("--new-junit", required=True); p.add_argument("--base-commit", required=True); p.add_argument("--new-commit", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "argv":
        print(json.dumps(render(a.kind, a.seed, a.root, a.out, a.nav_npz, a.ctx_npz, rb_npz=a.rb_npz)))
        return 0
    if a.cmd == "gate":
        bad = gate_problems(a.kind, json.load(open(a.argv_json, encoding="utf-8")), json.load(open(a.selfcheck, encoding="utf-8")))
    elif a.cmd == "pair":
        bad = (w_pairing_problems if a.kind in W_KINDS else c_pairing_problems if a.kind in C_KINDS else h_pairing_problems if a.kind in H_KINDS else m_pairing_problems if a.kind in M_KINDS else pairing_problems)(a.kind, json.load(open(a.argv_json, encoding="utf-8")), json.load(open(a.ref_argv_json, encoding="utf-8")))
    elif a.cmd == "post":
        import torch
        fin = torch.load(a.final, map_location="cpu", weights_only=False)["model"]
        bad = moved_identity(torch.load(a.init, map_location="cpu", weights_only=False)["model"], fin, a.kind)
        if a.kind in DCHAN_KINDS and "e_nav_rb.weight" in fin and fin["e_nav_rb.weight"].shape[1] == 15:
            print(json.dumps({"group_weight_l1": group_weight_norms(fin["e_nav_rb.weight"])}))      # PREREG_V8F_NAVRBD: a group that reads exactly 0.0 is named here, not failed
    elif a.cmd == "suite":
        r = suite_compare(a.base_junit, a.new_junit, a.new_commit, a.base_commit)
        json.dump(r, open(a.out, "w", encoding="utf-8"), indent=1)
        print(json.dumps({k: r[k] for k in ("base", "new", "n_regressions", "n_new_only")}))
        return 0 if r["n_regressions"] == 0 else 1
    elif a.cmd == "mint":
        tag = arm_tag(a.kind, a.seed, a.steps)
        bad = mint(tag, a.kind, a.seed, a)
        print("TOKEN-WRITTEN " + tag if not bad else "REFUSED: " + "; ".join(bad))
        return 0 if not bad else 1
    elif a.cmd == "run":
        return run_arm(arm_tag(a.kind, a.seed, a.steps), a.kind, a)
    elif a.cmd == "geval":
        bad, rec = geval(a.ckpt, json.load(open(a.selfcheck, encoding="utf-8"))["levers"] if a.selfcheck else None)
        print(json.dumps(rec))
    else:
        bad = hyg()
    print("PASS" if not bad else "REFUSED: " + "; ".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
