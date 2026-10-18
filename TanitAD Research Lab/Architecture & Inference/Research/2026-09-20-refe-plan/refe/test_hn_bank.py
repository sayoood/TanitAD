#!/usr/bin/env python3
"""Tests for refe/hn_bank.py (P6-HN-S checklist item 1; raw/2026-10-18-p6hn-prereg/PREREG_P6HN_SCREEN.md sections 2, 4.3, 5).
Run:  PYTHONIOENCODING=utf-8 python refe/test_hn_bank.py   ->   ZZTEST_PASS <n>   (prints the mutation table).   CPU only, no pod, no checkpoint, no nuPlan, no real data.

WHAT IS PROVEN (CLAUDE.md "built, tested, and unreachable from its caller": a mechanism that WORKS is not a mechanism that RUNS -- the wiring into the trainer is checklist item 3 and
is NOT proved here; what is proved here is the module the trainer will import):
  G  GEOMETRY (gate I2).  On a straight and a curved toy path every offset's mean |cross-track| to the anchor is |delta| (median error <= 0.01 m), time and heading are kept, delta > 0 is on the
     LEFT, the normal from the stored yaw and from a central-difference tangent agree, the STATIONARY rule (path < 2.0 m, counted from the origin) skips, and the builder's call graph reaches NO map API --
     with a control that must find one in the labeller's modules.
  L  LABELS.  The labelling call is `onpolicy_relabel_pdm.label_log` BY REFERENCE (no copied body in hn_bank.py), the label_version of every row sent and of the manifest is 5, I1 (the same call on the
     fan reproduces independently produced labels bit for bit) is live, the hard-negative rule has its truth table (NC = 0.5 is NOT one; an anchor with PDMS 0 contributes none), the bank file is
     deterministic, round-trips, and is REFUSED on another partition / label_version / fan fingerprint / changed bytes, a missing key, or another ckpt_step.  SY reports the share with its estimator.
  T  THE TERM, on a tiny scorer built from the REAL CrossBlock and the REAL REFe.score_trajectories: the fan call is untouched by the bank (bit-identical with and without), a plain append DOES move
     it (the defect the masked form removes), the masked hn logits equal an independent per-layer oracle (and, at ONE decoder layer, the last row of a real 65-candidate call), they differ from the
     plain-append logits, the gradient of the term alone is finite and non-zero on score_q_mlp / score_dec / score_head rows 0-5 and EXACTLY 0.0 on the lane row and on a separate arm-A copy, an empty
     bank gives exactly 0.0 loss and gradients, the weight is 1/(B*K*6), the pad-perturbation identity holds BIT FOR BIT even with NaN / inf junk in the pad slots, and the draft's duplicate-row
     identity holds on a scorer with the self-attention neutralised.
  P  PIN.  `hn_bank.score_set(blocked=None)` equals `REFe.score_trajectories` bit for bit, and that function's body is hash-pinned here, so the masked call cannot drift from it silently.
Every mutation below edits hn_bank.py SOURCE (each snippet must occur EXACTLY once, else the test errors) and must turn >= 1 expected check RED.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
import textwrap
import types

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M  # noqa: E402

SRC_PATH = os.path.join(HERE, "hn_bank.py")
SCORE_TRAJ_BODY_SHA = "423e1e9348a0f3b3728e8c9e78c5af279cf77d4129800db5583b0caf3128b6b0"
MAP_RE = re.compile(r"(?i)(map_?api|map_?root|get_?map|drivable|lane|centerline|map_?location|route_?lane|polygon)")
LABELLER_MODULES = ("navtrain_scenarios.py", "navsim_pdm_targets.py")


# ------------------------------------------------------------------------------------------------ helpers
def check(bad, name, fn):
    try:
        ok = bool(fn())
    except Exception as e:                                            # noqa: BLE001 -- an exception IS a failed check
        ok = False
        if os.environ.get("HNT_DEBUG"):
            print("  [check raised]", name, type(e).__name__, str(e)[:200])
    if not ok:
        bad.append(name)


def load_module(src: str, tag: str):
    mod = types.ModuleType(f"hn_bank_{tag}")
    mod.__file__ = SRC_PATH
    exec(compile(src, f"hn_bank_{tag}.py", "exec"), mod.__dict__)    # noqa: S102 -- the module under test, possibly mutated
    return mod


def idents(src: str) -> set:
    """every identifier-like token of a source's AST: names, attributes, args, keyword names, defs, import parts (strings and comments are NOT read)"""
    out = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.keyword) and n.arg:
            out.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                out.update(a.name.split("."))
        elif isinstance(n, ast.ImportFrom):
            out.update((n.module or "").split("."))
            out.update(a.name for a in n.names)
    return out


def called_names_in_functions(src: str) -> dict:
    """{function name: set of called names} -- to find a copied labeller body (score_plans / executed_plans8 / ... called from inside hn_bank.py)"""
    res = {}
    for f in ast.walk(ast.parse(src)):
        if isinstance(f, ast.FunctionDef):
            calls = set()
            for n in ast.walk(f):
                if isinstance(n, ast.Call):
                    fn = n.func
                    calls.add(fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "?"))
            res[f.name] = calls
    return res


class Tiny(nn.Module):
    """the three scorer modules built from the REAL CrossBlock, forward = the REAL REFe.score_trajectories (the ScorerCopy construction)"""

    def __init__(self, w=32, heads=4, depth=2, T=20, d=3, nout=7):
        super().__init__()
        self.score_q_mlp = nn.Sequential(nn.Linear(T * d, w), nn.GELU(), nn.Linear(w, w))
        self.score_dec = nn.ModuleList([M.CrossBlock(w, heads, w) for _ in range(depth)])
        self.score_head = nn.Linear(w, nout)

    def forward(self, trajs, sctx):
        return M.REFe.score_trajectories(self, trajs, sctx)


def mk(seed=0, **kw):
    torch.manual_seed(seed)
    return Tiny(**kw)


def neutralise_self_attn(m):
    for b in m.score_dec:
        b.self_attn.out_proj.weight.data.zero_()
        b.self_attn.out_proj.bias.data.zero_()
    return m


def data(seed=1, B=3, K=64, H=8, P=40, w=32, valid=((0, 0), (0, 1), (0, 2), (1, 2), (1, 5))):
    g = torch.Generator().manual_seed(seed)
    fan = torch.randn(B, K, 20, 3, generator=g)
    hn = torch.randn(B, H, 20, 3, generator=g)
    tg = torch.rand(B, H, 6, generator=g)
    ctx = torch.randn(B, P, w, generator=g)
    mk_ = torch.zeros(B, H, dtype=torch.bool)
    for b, h in valid:
        mk_[b, h] = True
    return fan, hn, tg, mk_, ctx


def oracle_hn_logits(m, fan, hn_h, ctx):
    """INDEPENDENT per-layer oracle of 'one hn row read against the deployed fan': the fan stream runs the normal fan-only block; the hn row's query attends to the fan's states at the INPUT of
    each layer plus its own.  Written from the block's sub-modules and the textbook formula, not from hn_bank."""
    sf = m.score_q_mlp(fan.flatten(2))
    sh = m.score_q_mlp(hn_h[:, None].flatten(2))
    for blk in m.score_dec:
        both = torch.cat([sf, sh], 1)
        h = blk.n1(both)
        q = sh + blk.self_attn(h[:, -1:], h, h, need_weights=False)[0]
        q = q + blk.cross(blk.n2(q), ctx, ctx, need_weights=False)[0]
        q = q + blk.mlp(blk.n3(q))
        sh = q
        sf = blk(sf, ctx)
    return m.score_head(sh)[:, 0]


def plain_append(m, fan, hn, ctx):
    """the draft's original wording: append the hn rows to the candidate set and score ONE set of K + H"""
    return m(torch.cat([fan, hn], 1), ctx)


# ------------------------------------------------------------------------------------------------ the fake labeller (a stand-in for label_log: same job/return contract)
OBS = np.array([10.0, -1.0])


def fake_cols(traj3):
    """deterministic labels from geometry only: NC 0 within 0.7 m of a fixed obstacle, 0.5 within 1.0 m, else 1; DAC 0 when |y| > 0.5 anywhere; ep depends on mean |yaw| (so a dropped yaw changes it)"""
    t = np.asarray(traj3, np.float64)
    out = {k: [] for k in ("nc", "dac", "ep", "ttc", "c", "ddc", "pdms")}
    for p in t:
        d = float(np.linalg.norm(p[:, :2] - OBS, axis=-1).min())
        nc = 0.0 if d < 0.7 else (0.5 if d < 1.0 else 1.0)
        dac = 0.0 if float(np.abs(p[:, 1]).max()) > 0.5 else 1.0
        ep = round(1.0 - min(1.0, float(np.abs(p[:, 2]).mean())), 5)
        out["nc"].append(nc)
        out["dac"].append(dac)
        out["ep"].append(ep)
        out["ttc"].append(1.0)
        out["c"].append(1.0)
        out["ddc"].append(1.0)
        out["pdms"].append(round(nc * dac * (5 * ep + 5 + 2) / 12.0, 5))
    return out


CALLS: list = []


def fake_label_fn(job):
    log, rows = job
    CALLS.append((log, [dict(r) for r in rows]))
    out = []
    for r in rows:
        t = np.concatenate([np.asarray(r["traj"], np.float64), np.asarray(r["yaw"], np.float64)[..., None]], -1)
        rec = {"kind": "pdm_targets", "key": [r["log_name"], r["token"], int(r["step"]), int(r["rank"])], "ckpt_step": int(r["ckpt_step"]),
               "label_version": int(r.get("label_version", 1)), "plans": "executed (A7 repair)"}
        rec.update(fake_cols(t))
        out.append(json.dumps(rec) + "\n")
    return out, 0, 0, None


def straight(n=20, v=8.0, T=0.2, y0=0.0):
    x = v * T * np.arange(1, n + 1)
    return np.stack([x, np.full(n, y0), np.zeros(n)], -1)


def arc(n=20, R=50.0, ds=1.2):
    th = ds * np.arange(1, n + 1) / R
    return np.stack([R * np.sin(th), R * (1 - np.cos(th)), th], -1)


def anchors_for(n_logs=3, per_log=4, mix_stationary=True):
    rows = []
    for li in range(n_logs):
        for j in range(per_log):
            a = straight()
            if mix_stationary and j == 3:
                a = straight(v=0.5)                                       # 20 steps * 0.1 m = 2.0 m? v*T = 0.1 -> path 2.0 m exactly at n=20: shortened below
                a[:, 0] = np.linspace(0.05, 1.9, 20)                      # 1.9 m: STATIONARY
            rows.append({"log_name": f"log{li}", "token": f"t{li}{j}", "step": 0, "rank": 0, "ckpt_step": 10075, "traj": a[:, :2].tolist(), "yaw": a[:, 2].tolist()})
    return rows


def fan_rows_and_banked(n=4, M_=6):
    """independent 'RL-S1' labels: built by hand-formatting the rows and calling the fake labeller directly (NOT through hn_bank)"""
    rng = np.random.default_rng(3)
    rows, banked = [], {}
    for i in range(n):
        traj = rng.normal(0, 0.4, (M_, 20, 2)) + np.stack([np.linspace(0, 20, 20), np.zeros(20)], -1)[None]
        yaw = rng.normal(0, 0.05, (M_, 20))
        r = {"log_name": f"log{i % 2}", "token": f"f{i}", "step": 0, "rank": 0, "ckpt_step": 10075, "traj": np.round(traj, 5).tolist(), "yaw": np.round(yaw, 6).tolist()}
        rows.append(r)
        out, *_ = fake_label_fn((r["log_name"], [dict(r, label_version=5)]))
        rec = json.loads(out[0])
        banked[(r["log_name"], r["token"], 0, 0)] = {c: rec[c] for c in ("nc", "dac", "ep", "ttc", "c", "ddc", "pdms")}
    return rows, banked


# ================================================================================================ the suite
def suite(H, tag: str) -> list:
    bad: list = []
    tmp = tempfile.mkdtemp(prefix=f"hnb_{tag}_")
    hsrc = getattr(H, "__source__", None) or open(SRC_PATH, encoding="utf-8").read()

    # ------------------------------------------------------------------ G: geometry (I2)
    def cross_track(anchor, off):
        """mean |distance| of offset points to the anchor POLYLINE (independent of how the offset was built)"""
        a = anchor[:, :2]
        res = []
        for p in off[:, :2]:
            best = 1e9
            for s0, s1 in zip(a[:-1], a[1:]):
                d = s1 - s0
                t = np.clip(np.dot(p - s0, d) / max(np.dot(d, d), 1e-12), 0, 1)
                best = min(best, float(np.linalg.norm(p - (s0 + t * d))))
            res.append(best)
        return float(np.mean(res))

    def geom_ok(anchor):
        off = H.make_offsets(anchor)
        errs = [abs(cross_track(anchor, off[i]) - abs(d)) for i, d in enumerate(H.DELTAS)]
        kept = all(np.array_equal(off[i][:, 2], anchor[:, 2]) and off[i].shape == anchor.shape for i in range(len(H.DELTAS)))
        return float(np.median(errs)) <= 0.01 and max(errs) <= 0.02 and kept

    check(bad, "G_straight", lambda: geom_ok(straight()))
    check(bad, "G_curved", lambda: geom_ok(arc()) and geom_ok(arc(R=-60.0, ds=1.0) if False else arc(R=80.0)))

    def left_is_positive():
        off = H.make_offsets(straight())
        i_pos, i_neg = H.DELTAS.index(0.30), H.DELTAS.index(-0.30)
        return abs(off[i_pos][:, 1].mean() - 0.30) < 1e-9 and abs(off[i_neg][:, 1].mean() + 0.30) < 1e-9 and np.allclose(off[i_pos][:, 0], straight()[:, 0])

    check(bad, "G_side_left_positive", left_is_positive)

    def normals_agree():
        a = arc(R=60.0)
        n_yaw = H.left_normals(a)
        n_tan = H.left_normals(a[:, :2])
        # curved path, central differences: a few mrad apart; and a yaw-less anchor goes through the tangent branch with the heading filled in
        ang = np.abs(np.arccos(np.clip((n_yaw * n_tan).sum(-1), -1, 1)))
        full = H.anchor_array(a[:, :2])
        return float(ang.max()) < 0.03 and np.isfinite(full[:, 2]).all() and np.abs(full[:, 2] - a[:, 2]).max() < 0.03

    check(bad, "G_normal_yaw_vs_tangent", normals_agree)

    def stationary_rule():
        rows = anchors_for(1, 4)
        below = H.path_length(np.stack([np.linspace(0.05, 1.99, 20), np.zeros(20)], -1)) < H.STATIONARY_M
        above = H.path_length(np.stack([np.linspace(0.05, 2.01, 20), np.zeros(20)], -1)) >= H.STATIONARY_M
        CALLS.clear()
        keys, arrays, man = H.build_bank(rows, partition_id="p", fan_fingerprint="f", label_fn=fake_label_fn)
        sent = {r["token"] for _lg, rs in CALLS for r in rs}
        stat_tok = {r["token"] for r in rows if H.path_length(H.anchor_array(r["traj"], r["yaw"])) < 2.0}
        status = {k[1]: int(s) for k, s in zip(keys, arrays["status"])}
        return (below and above and stat_tok and not (stat_tok & sent) and all(status[t] == H.STATUS_STATIONARY for t in stat_tok)
                and man["label_counters"]["requested"] == len(rows) - len(stat_tok))

    check(bad, "G_stationary_rule", stationary_rule)

    def path_from_origin():
        # one step of 2.5 m from the origin: path length counts the origin -> p0 leg
        a = np.zeros((20, 3))
        a[:, 0] = 2.5
        return abs(H.path_length(a) - 2.5) < 1e-9

    check(bad, "G_path_starts_at_origin", path_from_origin)

    # ------------------------------------------------------------------ G: no map API in the builder (with a control)
    def nomap():
        mine = {t for t in idents(hsrc) if MAP_RE.search(t)}
        return not mine

    def nomap_control():
        n = 0
        for fn in LABELLER_MODULES:
            p = os.path.join(HERE, fn)
            n += len({t for t in idents(open(p, encoding="utf-8").read()) if MAP_RE.search(t)})
        return n >= 2                                                    # the scanner MUST find map identifiers in the labeller's modules, or a clean scan proves nothing

    check(bad, "G_no_map_api_in_builder", nomap)
    check(bad, "G_no_map_api_control_finds_one_in_labeller", nomap_control)

    # ------------------------------------------------------------------ L: labels
    def labeller_by_reference():
        import onpolicy_relabel_pdm as ORP
        calls = called_names_in_functions(hsrc)
        copied = [f for f, cs in calls.items() if cs & {"score_plans", "executed_plans8", "metric_cache_parts", "build_scenarios_for_log", "simulate_proposals"}]
        defs = {n.name for n in ast.walk(ast.parse(hsrc)) if isinstance(n, ast.FunctionDef)}
        return H.labeller() is ORP.label_log and not copied and "label_log" not in defs

    check(bad, "L_label_log_by_reference", labeller_by_reference)

    def label_version_five():
        CALLS.clear()
        keys, arrays, man = H.build_bank(anchors_for(1, 3), partition_id="p", fan_fingerprint="f", label_fn=fake_label_fn)
        sent = [r["label_version"] for _lg, rs in CALLS for r in rs]
        return H.LABEL_VERSION == 5 and man["label_version"] == 5 and sent and set(sent) == {5}

    check(bad, "L_label_version_5", label_version_five)

    def i1_identity():
        rows, banked = fan_rows_and_banked()
        ok = H.label_identity_check(rows, banked, fake_label_fn)
        wrong = {k: dict(v, pdms=[x + 1e-5 for x in v["pdms"]]) for k, v in banked.items()}
        neg = H.label_identity_check(rows, wrong, fake_label_fn)
        return ok["ok"] and ok["n_identical"] == len(rows) and not neg["ok"]

    check(bad, "L_I1_label_identity_live", i1_identity)

    def rule_truth_table():
        def lab(anchor_pdms, nc, dac, pdms):
            l9 = np.full((9, 7), 1.0, np.float32)
            l9[0, 6] = anchor_pdms
            l9[1] = [nc, dac, 1, 1, 1, 1, pdms]
            return H.hard_negative_rule(l9)
        h, s = lab(1.0, 0.0, 1.0, 0.0)
        c1 = bool(h[0]) and int(s[0]) == 2                               # NC-only
        h, s = lab(1.0, 1.0, 0.0, 0.0)
        c2 = bool(h[0]) and int(s[0]) == 1                               # DAC-only
        h, s = lab(1.0, 0.0, 0.0, 0.0)
        c3 = bool(h[0]) and int(s[0]) == 3                               # both
        h, s = lab(1.0, 0.5, 1.0, 0.5)
        c4 = not bool(h[0])                                              # NC = 0.5 is NOT a hard negative
        h, s = lab(0.0, 0.0, 1.0, 0.0)
        c5 = not bool(h[0])                                              # the anchor scores 0: contributes none
        h, s = lab(1.0, 1.0, 1.0, 0.0)
        c6 = not bool(h[0])                                              # PDMS 0 from some other component (neither NC nor DAC zero): not one
        h, s = lab(1.0, 1.0, 1.0, 0.7)
        c7 = not bool(h[0])
        h, s = lab(1.0, 0.5, 1.0, 0.0)
        c7 = c7 and not bool(h[0])                                       # NC = 0.5 with PDMS 0 from another component: still not one
        h, s = H.hard_negative_rule(np.full((9, 7), np.nan, np.float32))
        c8 = not h.any()                                                 # unlabelled rows are never hard negatives
        return all((c1, c2, c3, c4, c5, c6, c7, c8))

    check(bad, "L_hard_negative_rule_truth_table", rule_truth_table)

    def bank_roundtrip():
        rows = anchors_for(3, 4)
        p1, p2 = os.path.join(tmp, "b1.hnbank"), os.path.join(tmp, "b2.hnbank")
        kw = dict(partition_id="part-A", fan_fingerprint="fan-1", label_fn=fake_label_fn)
        b1 = H.mint_bank(rows, p1, **kw)
        b2 = H.mint_bank(rows, p2, **kw)
        same = open(p1, "rb").read() == open(p2, "rb").read()             # deterministic bytes
        man = json.load(open(p1 + ".manifest.json", encoding="utf-8"))
        keys = [(r["log_name"], r["token"], 0, 0) for r in rows]
        bt = b1.batch(keys)
        non_stat = [i for i, r in enumerate(rows) if r["token"][-1] != "3"]
        # straight anchor: delta -0.60 -> NC-only? (obstacle at (10,-1)): see the expected sub-types
        sub = bt["hn_sub"][non_stat[0]].tolist()
        exp_sub = [3, 2, 0, 0, 0, 0, 0, 1]                                # d=-0.60 both, -0.45 NC-only, -0.30/-0.15 NC=0.5, +0.15.. clear, +0.60 DAC-only (|y| > 0.5)
        sums = int(bt["hn_m"].sum())
        return (same and man["sha256"] == H.sha256_file(p1) and man["n_rows"] == len(rows) and bt["hn_xy"].shape == (len(rows), 8, 20, 3) and bt["hn_tg"].shape == (len(rows), 8, 6)
                and sub == exp_sub and sums == b1.n_rows() and sums == man["counts"]["hard_negative_rows"] and b1.scenes_with_hn() == 9
                and b1.n_rows((1,)) == 9 and b1.n_rows((2,)) == 9 and b1.n_rows((3,)) == 9 and b2.n_rows() == b1.n_rows()
                and b1.hn_block()["rows"] == b1.n_rows() and b1.hn_block()["max_per_sample"] == 8 and b1.hn_block()["comps"] == 6)

    check(bad, "L_bank_roundtrip_deterministic_and_counted", bank_roundtrip)

    def sdac_filter():
        rows = anchors_for(1, 3, mix_stationary=False)
        b = H.mint_bank(rows, os.path.join(tmp, "b3.hnbank"), partition_id="p", fan_fingerprint="f", label_fn=fake_label_fn)
        keys = [(r["log_name"], r["token"], 0, 0) for r in rows]
        full, dac_only = b.batch(keys), b.batch(keys, subtypes=(1,))
        return int(full["hn_m"].sum()) == 9 and int(dac_only["hn_m"].sum()) == 3 and bool((dac_only["hn_sub"][dac_only["hn_m"]] == 1).all())

    check(bad, "L_S_DAC_subtype_filter", sdac_filter)

    def cli_yield():
        rows = anchors_for(2, 4)
        p = os.path.join(tmp, "b5.hnbank")
        H.mint_bank(rows, p, partition_id="part-A", fan_fingerprint="fan-1", label_fn=fake_label_fn)
        out = os.path.join(tmp, "sy.json")
        rc = H.main(["yield", "--bank", p, "--partition-id", "part-A", "--label-version", "5", "--fan-fingerprint", "fan-1", "--out", out])
        rep = json.load(open(out, encoding="utf-8"))
        try:
            H.main(["yield", "--bank", p, "--partition-id", "other", "--out", out])
            refused = False
        except H.BankMismatch:
            refused = True
        # 6 labelled scenes of 8 carry hard negatives (3 each); 2 are stationary
        return rc == 0 and rep["n_scenes"] == 8 and rep["scenes_with_hn"] == 6 and abs(rep["share_with_hn"] - 0.75) < 1e-12 and not rep["sparse"] and rep["n_stationary"] == 2 and refused

    check(bad, "L_cli_yield_report_and_partition_refusal", cli_yield)

    def refusals():
        rows = anchors_for(1, 2, mix_stationary=False)
        p = os.path.join(tmp, "b4.hnbank")
        H.mint_bank(rows, p, partition_id="part-A", fan_fingerprint="fan-1", label_fn=fake_label_fn)
        res = []
        for exp in ({"partition_id": "part-B"}, {"label_version": 4}, {"fan_fingerprint": "fan-2"}):
            try:
                H.load_bank(p, expect=exp)
                res.append(False)
            except H.BankMismatch:
                res.append(True)
        good = H.load_bank(p, expect={"partition_id": "part-A", "label_version": 5, "fan_fingerprint": "fan-1"})
        with open(p, "r+b") as f:                                          # stale_bank: bytes changed after minting
            f.seek(-4, 2)
            b = f.read(1)
            f.seek(-4, 2)
            f.write(bytes([b[0] ^ 1]))
        try:
            H.load_bank(p)
            res.append(False)
        except H.BankMismatch:
            res.append(True)
        k0 = (rows[0]["log_name"], rows[0]["token"], 0, 0)
        try:
            good.batch([("nolog", "nope", 0, 0)])
            res.append(False)
        except H.BankMismatch:
            res.append(True)
        try:
            good.batch([k0], ckpt_steps=[99])
            res.append(False)
        except H.BankMismatch:
            res.append(True)
        return all(res) and good.batch([k0], ckpt_steps=[10075])["hn_m"].shape == (1, 8)

    check(bad, "L_bank_refused_on_partition_label_fan_bytes_missing_key_ckpt", refusals)

    def wrong_label_version_from_labeller():
        def bad_fn(job):
            out, nf, ns, err = fake_label_fn(job)
            return [json.dumps(dict(json.loads(l), label_version=4)) + "\n" for l in out], nf, ns, err
        try:
            H.build_bank(anchors_for(1, 2, mix_stationary=False), partition_id="p", fan_fingerprint="f", label_fn=bad_fn)
            return False
        except H.BankMismatch:
            return True

    check(bad, "L_labeller_echo_label_version_checked", wrong_label_version_from_labeller)

    def sy_report():
        N = 40
        keys = [(f"log{i % 8}", f"t{i}", 0, 0, 1) for i in range(N)]
        lab9 = np.full((N, 9, 7), 1.0, np.float32)
        def bank_with(n_hn):
            l9 = lab9.copy()
            for i in range(n_hn):
                l9[i, 1, 1] = 0.0
                l9[i, 1, 6] = 0.0
            return H.HNBank(keys, {"status": np.zeros(N, np.int8), "traj9": np.zeros((N, 9, 20, 3), np.float32), "lab9": l9}, {"sha256": "x"})
        r4 = H.yield_report(bank_with(4), n_boot=300)                    # 4/40 = 0.10 exactly: NOT sparse (rule is '< 0.10')
        r3 = H.yield_report(bank_with(3), n_boot=300)
        r8 = H.yield_report(bank_with(8), n_boot=300)
        return (abs(r4["share_with_hn"] - 0.10) < 1e-12 and not r4["sparse"] and r3["sparse"] and not r8["sparse"] and r4["scenes_with_hn"] == 4 and r4["n_logs"] == 8
                and r4["share_ci95"][0] <= r4["share_with_hn"] <= r4["share_ci95"][1] and "estimator" in r4 and r4["subtype_rows"]["DAC-only"] == 4
                and len(r4["rate_per_delta_and_side"]) == 8 and r4["anchors_with_pdms_zero"] == 0)

    check(bad, "L_SY_share_estimator_and_threshold_boundary", sy_report)

    # ------------------------------------------------------------------ P: the pin
    def pinned():
        import model as MM
        fn = ast.parse(textwrap.dedent(inspect.getsource(MM.REFe.score_trajectories))).body[0]
        body = fn.body[1:] if isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) else fn.body
        sha = hashlib.sha256("\n".join(ast.dump(n) for n in body).encode()).hexdigest()
        m = mk(0)
        fan, hn, tg, mk_, ctx = data()
        return sha == SCORE_TRAJ_BODY_SHA and torch.equal(H.score_set(m, fan, ctx), m(fan, ctx)) and torch.equal(H.score_set(m, fan, ctx, None), M.REFe.score_trajectories(m, fan, ctx))

    check(bad, "P_score_set_equals_score_trajectories_and_source_pinned", pinned)

    # ------------------------------------------------------------------ T: the term
    def fan_untouched():
        m = mk(0).eval()
        fan, hn, tg, mk_, ctx = data()
        with torch.no_grad():
            f0 = m(fan, ctx)
            H.hn_term(m, fan, hn, tg, mk_, ctx)                          # the term runs; the fan call is a separate call
            f1 = m(fan, ctx)
            app = plain_append(m, fan, hn, ctx)[:, :fan.shape[1]]
        return torch.equal(f0, f1) and float((app - f0).abs().max()) > 1e-4          # the defect is real: a plain append moves the fan rows

    check(bad, "T_fan_call_bit_identical_and_plain_append_moves_it", fan_untouched)

    def equiv_oracle():
        ok = []
        for mode in ("train", "eval"):
            m = getattr(mk(0, depth=3), mode)()
            fan, hn, tg, mk_, ctx = data()
            with torch.no_grad():
                lg = H.hn_scores(m, fan, hn, mk_, ctx)
                ref = torch.stack([oracle_hn_logits(m, fan, hn[:, h], ctx) for h in range(hn.shape[1])], 1)
            ok.append(float((lg - ref).abs()[mk_].max()) < 1e-5)
        return all(ok)

    check(bad, "T_masked_logits_equal_per_layer_oracle_depth3", equiv_oracle)

    def equiv65_depth1():
        m = mk(0, depth=1).eval()
        fan, hn, tg, mk_, ctx = data()
        K = fan.shape[1]
        with torch.no_grad():
            lg = H.hn_scores(m, fan, hn, mk_, ctx)
            ref = torch.stack([m(torch.cat([fan, hn[:, h:h + 1]], 1), ctx)[:, K] for h in range(hn.shape[1])], 1)
        return float((lg - ref).abs()[mk_].max()) < 1e-5

    check(bad, "T_masked_logits_equal_real_65_candidate_call_at_depth1", equiv65_depth1)

    def hn_differ():
        m = mk(0, depth=3).eval()
        fan, hn, tg, mk_, ctx = data()
        with torch.no_grad():
            lg = H.hn_scores(m, fan, hn, mk_, ctx)
            app = plain_append(m, fan, hn, ctx)[:, fan.shape[1]:]
        return float((lg - app)[mk_].abs().max()) > 1e-4 and float((lg[0, 0] - lg[0, 1]).abs().max()) > 1e-4

    check(bad, "T_hn_rows_differ_from_plain_append_and_from_each_other", hn_differ)

    def grads():
        mB, mA = mk(0), mk(0)
        fan, hn, tg, mk_, ctx = data()
        f = H.probe_fields(mB, mA, fan, hn, tg, mk_, ctx)
        g = f["grad_abs_sum"]
        return (f["grads_finite"] and g["score_q_mlp"] > 0 and g["score_dec"] > 0 and g["score_head_rows_0_5"] > 0 and g["score_head_row_6_lane"] == 0.0 and g["armA_all"] == 0.0
                and f["n_hn"] == 5 and f["loss"] > 0 and f["fan_two_call_equals_one_call"])

    check(bad, "T_grad_table_nonzero_and_exact_zero_controls", grads)

    def empty_bank():
        m = mk(0)
        fan, hn, tg, mk_, ctx = data()
        out = H.hn_term(m, fan, hn, tg, torch.zeros_like(mk_), ctx)
        out.loss.backward()
        return float(out.loss) == 0.0 and int(out.n_hn) == 0 and all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in m.parameters())

    check(bad, "T_empty_bank_exact_zero_loss_and_grads", empty_bank)

    def weight_and_dup_identity():
        """the draft's identity on a scorer with NO set-level self-attention: a bank row that duplicates fan row j adds exactly row j's share of bce_fan"""
        m = neutralise_self_attn(mk(0))
        fan, hn, tg, mk_, ctx = data(valid=())
        B, K = fan.shape[0], fan.shape[1]
        j = 17
        hn = hn.clone()
        tg = tg.clone()
        mk_ = torch.zeros_like(mk_)
        mk_[1, 0] = True
        hn[1, 0] = fan[1, j]
        with torch.no_grad():
            fl = m(fan, ctx).float()
            tgt = torch.rand(6)
            tg[1, 0] = tgt
            out = H.hn_term(m, fan, hn, tg, mk_, ctx)
            share = torch.nn.functional.binary_cross_entropy_with_logits(fl[1, j, :6], tgt, reduction="sum") / (B * K * 6)
            # the same row twice (two valid dup rows) doubles the term
            mk2 = mk_.clone()
            mk2[1, 1] = True
            hn2, tg2 = hn.clone(), tg.clone()
            hn2[1, 1] = fan[1, j]
            tg2[1, 1] = tgt
            out2 = H.hn_term(m, fan, hn2, tg2, mk2, ctx)
        return abs(float(out.loss) - float(share)) <= 1e-7 * max(1.0, abs(float(share))) and abs(float(out2.loss) - 2 * float(share)) <= 2e-7

    check(bad, "T_weight_1_over_BK6_and_duplicate_row_identity_without_self_attention", weight_and_dup_identity)

    def pad_identity():
        m = mk(0)
        fan, hn, tg, mk_, ctx = data()
        def run(hn_, tg_):
            for p in m.parameters():
                p.grad = None
            out = H.hn_term(m, fan, hn_, tg_, mk_, ctx)
            out.loss.backward()
            return out.loss.detach().clone(), [p.grad.clone() for p in m.parameters()]
        l0, g0 = run(hn, tg)
        hn2, tg2 = hn.clone(), tg.clone()
        pad = ~mk_
        hn2[pad] = float("nan")
        hn2[0, 7, 3, 0] = float("inf")
        tg2[pad] = float("nan")
        l1, g1 = run(hn2, tg2)
        hn3, tg3 = hn.clone(), tg.clone()
        hn3[pad] = 1e6 * torch.randn_like(hn3[pad])
        tg3[pad] = 5.0 * torch.randn_like(tg3[pad])
        l2, g2 = run(hn3, tg3)
        return (torch.equal(l0, l1) and torch.equal(l0, l2) and all(torch.equal(a, b) for a, b in zip(g0, g1)) and all(torch.equal(a, b) for a, b in zip(g0, g2))
                and bool(torch.isfinite(l1)))

    check(bad, "T_pad_perturbation_identity_bit_for_bit_with_nan_inf_junk", pad_identity)

    def mask_shape():
        b = H.hn_attn_mask(64, 8)
        fan_sees_fan = (~b[:64, :64]).all()
        fan_blind_to_hn = b[:64, 64:].all()
        hn_sees_fan = (~b[64:, :64]).all()
        hn_self = all(not bool(b[64 + h, 64 + h]) for h in range(8))
        hn_not_other = all(bool(b[64 + h, 64 + g]) for h in range(8) for g in range(8) if g != h)
        return bool(fan_sees_fan and fan_blind_to_hn and hn_sees_fan and hn_self and hn_not_other)

    check(bad, "T_attention_mask_shape", mask_shape)

    def autocast_bf16():
        m = mk(0)
        fan, hn, tg, mk_, ctx = data()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = H.hn_term(m, fan, hn, tg, mk_, ctx)
        out.loss.backward()
        return bool(torch.isfinite(out.loss)) and out.loss.dtype == torch.float32 and m.score_head.weight.grad[6].abs().sum() == 0

    check(bad, "T_runs_under_bf16_autocast_float32_loss", autocast_bf16)
    return bad


# ================================================================================================ mutations
MUTATIONS = [
    ("normal sign flipped (right-normal)", [("return np.stack([-np.sin(yaw), np.cos(yaw)], -1)", "return np.stack([np.sin(yaw), -np.cos(yaw)], -1)")], ["G_side_left_positive"]),
    ("STATIONARY rule dropped", [("if path_length(a) < STATIONARY_M:", "if False:")], ["G_stationary_rule"]),
    ("path length not started from the origin", [("pts = np.concatenate([np.zeros((1, 2)), xy], 0)", "pts = xy")], ["G_path_starts_at_origin"]),
    ("label_version 4", [("LABEL_VERSION = 5 ", "LABEL_VERSION = 4 ")], ["L_label_version_5"]),
    ("a copied labeller body", [("    import onpolicy_relabel_pdm as ORP\n    return ORP.label_log\n",
                                 "    def label_log(job):\n        import navsim_pdm_targets as PT\n        return PT.score_plans(None, PT.executed_plans8(None))\n    return label_log\n")],
     ["L_label_log_by_reference"]),
    ("label row drops the yaw", [('"traj": t[..., :2].tolist(), "yaw": t[..., 2].tolist()}', '"traj": t[..., :2].tolist(), "yaw": np.zeros_like(t[..., 2]).tolist()}')], ["L_I1_label_identity_live"]),
    ("NC = 0.5 counted as a hard negative", [("    nc0 = nc == 0\n", "    nc0 = nc <= 0.5\n")], ["L_hard_negative_rule_truth_table"]),
    ("anchor PDMS > 0 condition dropped", [(" & anchor_ok[..., None]\n    sub", "\n    sub")], ["L_hard_negative_rule_truth_table"]),
    ("loader skips the partition / label / fan check", [("        if man.get(k) != v:\n", "        if False:\n")], ["L_bank_refused_on_partition_label_fan_bytes_missing_key_ckpt"]),
    ("loader skips the sha256 check", [('    if got != man.get("sha256"):\n', "    if False:\n")], ["L_bank_refused_on_partition_label_fan_bytes_missing_key_ckpt"]),
    ("a missing key is served as an empty row", [("                if strict:\n                    raise BankMismatch(f\"key {k} is not in the bank\")\n", "                if False:\n                    pass\n")],
     ["L_bank_refused_on_partition_label_fan_bytes_missing_key_ckpt"]),
    ("labeller's label_version echo unchecked", [("            if int(rec[\"label_version\"]) != int(want[k][\"label_version\"]):\n", "            if False:\n")], ["L_labeller_echo_label_version_checked"]),
    ("SY threshold <= instead of <", [('"sparse": bool(pt < min_share)', '"sparse": bool(pt <= min_share)')], ["L_SY_share_estimator_and_threshold_boundary"]),
    ("a map identifier planted in the builder", [("TOOL = \"hn_bank/1\"\n", "TOOL = \"hn_bank/1\"\nmap_api = None\n")], ["G_no_map_api_in_builder"]),
    ("the term is deleted", [("    loss = bce.sum() / float(B * K * N_COMP)\n", "    loss = bce.sum() * 0.0\n")], ["T_grad_table_nonzero_and_exact_zero_controls", "T_weight_1_over_BK6_and_duplicate_row_identity_without_self_attention"]),
    ("weight 1/(B*6) instead of 1/(B*K*6)", [("float(B * K * N_COMP)", "float(B * N_COMP)")], ["T_weight_1_over_BK6_and_duplicate_row_identity_without_self_attention"]),
    ("pad rows unmasked in the loss", [("    bce = torch.where(keep[..., None], bce, torch.zeros_like(bce))\n", "")], ["T_empty_bank_exact_zero_loss_and_grads", "T_weight_1_over_BK6_and_duplicate_row_identity_without_self_attention"]),
    ("pad rows not sanitised before the call", [("    xy = torch.where(keep[..., None, None], hn_xy.to(fan_xy.dtype), torch.zeros_like(hn_xy, dtype=fan_xy.dtype))\n", "    xy = hn_xy.to(fan_xy.dtype)\n")],
     ["T_pad_perturbation_identity_bit_for_bit_with_nan_inf_junk"]),
    ("pad targets not sanitised", [("    tg = torch.where(keep[..., None], hn_tg.to(torch.float32), torch.zeros_like(hn_tg, dtype=torch.float32))\n", "    tg = hn_tg.to(torch.float32)\n")],
     ["T_pad_perturbation_identity_bit_for_bit_with_nan_inf_junk"]),
    ("attention mask dropped (everything sees everything)", [("    blocked = (j >= K) & (j != i)\n", "    blocked = (j >= K) & (j != i) & (j < K)\n")],
     ["T_masked_logits_equal_per_layer_oracle_depth3", "T_attention_mask_shape"]),
    ("fan rows see the hn rows", [("    blocked = (j >= K) & (j != i)\n", "    blocked = (j >= K) & (j != i) & (i >= K)\n")], ["T_masked_logits_equal_per_layer_oracle_depth3", "T_attention_mask_shape"]),
    ("hn rows see each other", [("    blocked = (j >= K) & (j != i)\n", "    blocked = (j >= K) & (i < K)\n")], ["T_masked_logits_equal_per_layer_oracle_depth3", "T_attention_mask_shape"]),
    ("hn rows get a lane target (7 components)", [("    bce = F.binary_cross_entropy_with_logits(logits[..., :N_COMP], tg[..., :N_COMP], reduction=\"none\")\n",
                                                   "    bce = F.binary_cross_entropy_with_logits(logits, F.pad(tg, (0, 1)), reduction=\"none\")\n")], ["T_grad_table_nonzero_and_exact_zero_controls"]),
    ("the second call replaced by a plain append of hn to the fan set", [("    return score_set(scorer, both, sctx, blocked)[:, K:]\n", "    return score_set(scorer, both, sctx, None)[:, K:]\n")],
     ["T_masked_logits_equal_per_layer_oracle_depth3", "T_masked_logits_equal_real_65_candidate_call_at_depth1"]),
]


def apply_edits(src: str, edits, name: str) -> str:
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise AssertionError(f"mutation {name!r}: snippet occurs {n} times (need exactly 1): {old[:70]!r}")
        src = src.replace(old, new)
    return src


def main() -> int:
    src = open(SRC_PATH, encoding="utf-8").read()
    H = load_module(src, "real")
    H.__source__ = src
    real = suite(H, "real")
    assert not real, f"the unmutated hn_bank.py FAILS its own checks: {real}"
    n, problems = 1, []
    print(f"{'mutation':70s} caught-by (failing checks)")
    for name, edits, expect in MUTATIONS:
        m_src = apply_edits(src, edits, name)
        try:
            ast.parse(m_src)
            Hm = load_module(m_src, f"m{n}")
            Hm.__source__ = m_src
            failing = suite(Hm, f"m{n}")
        except (Exception, SystemExit) as e:                           # noqa: BLE001 -- a mutated source that breaks the suite's own setup is RED, and said so
            failing = [f"suite setup raised {type(e).__name__}: {str(e)[:80]}"]
        if not failing:
            problems.append(f"mutation {name!r} stayed GREEN: the suite cannot see it")
            continue
        for c in expect:
            if c not in failing and not any(f.startswith("suite setup raised") for f in failing):
                problems.append(f"mutation {name!r}: the check {c} must go RED and stayed green (RED: {failing[:8]})")
        print(f"{name[:70]:70s} {', '.join(failing[:3])}", flush=True)
        n += 1
    if problems:
        print("\n".join("PROBLEM " + p for p in problems))
        print(f"ZZTEST_FAIL {len(problems)} problem(s)")
        return 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite + {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
