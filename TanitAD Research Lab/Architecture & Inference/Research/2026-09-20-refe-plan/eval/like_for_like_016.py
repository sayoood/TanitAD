#!/usr/bin/env python3
"""LIKE-FOR-LIKE PDMS READOUT FOR REFe SNAPSHOT 016: snapshot 015 re-read WITH SPEC_NAVTEST Amendment 7's repair on
the SAME 200 tokens (W3's A1_sub200 subset), so 015 -> 016 on the sub200 learning curve is repaired-vs-repaired.
CPU ONLY: no model run, no GPU, torch is never imported (asserted and recorded in every output).

WHY. The sub200 curve read PDMS 61.18 after epoch 15 and 76.52 after epoch 16, but 016 is the FIRST point scored under
Amendment 7 (adopted 2026-09-27 ~14:30 Berlin): on the planner's native [20, 3] output the EXECUTED plan's last heading
is repaired, heading[19] := heading[18] (refe/planner.py `repair_last_heading`, `REFePlanner.executed`). 015 was scored
without it. The repair alone measured +16.20 [+12.83, +19.52] PDMS on 923 fresh tokens at epoch 15
(eval/raw/a7_confirm/a7_confirm_ep015.json), so the +15.34 step mixes the repair with one epoch of training.
The repair acts AFTER selection (`infer` picks k from the scorer's logits on the raw proposals; `executed` repairs
proposal k), so the pick is unchanged and 015-repaired is what today's planner executes on snapshot 015.

STAGES (each prints a ZZLFL marker; `all` = build -> score -> analyze):
  build     GATE A -- recover each token's shipped pick k from the ep015 NATIVE dump (stop_candidate_probe.py's
            stop_candidate_dump.npz: key `traj` [200, 64, 20, 3], token-order key `token`) by EXACT match:
            to_navsim(traj[token, k]) as float32 (the seam writer's dtype) must equal the shipped ep015 seam's 8 poses
            BIT-FOR-BIT, and uniquely -- 0 or >1 matches on any token -> STOP (the failing tokens are written out).
            GATE R -- the selection rule (and `select`) each point was picked with, read from its seam report AND its
            pipeline proposal dump; different rules -> NOT like-for-like, and no delta is read.
            Writes two seams in the pipeline's npz format (token, fingerprint, poses, sampling, arm): the REBUILT
            UNREPAIRED one, to_navsim(traj[token, k]), and the REPAIRED one, to_navsim(repair_last_heading(
            traj[token, k])) with planner.py's OWN function, compiled from its source so torch is never imported
            (an independent literal heading[19] := heading[18] must agree bit-for-bit).
            GATE B -- repaired x, y bit-identical to the shipped seam on all 8 poses of all tokens; headings 0.5-3.5 s
            bit-identical; the 4.0 s heading bit-identical to the native heading[18]; so ONLY the 4.0 s heading differs.
  score     both seams through eval/score_navtest_refe.py UNCHANGED (W3's harness: SeamAgentV1, metric_cache_navtest,
            PYTHONHASHSEED=1, guards C1-C3) in the pipeline's interpreter (EC.DRIVERL_PY), called exactly as
            eval_checkpoint.py step 2 calls it; at most 2 concurrent harness runs (the CPU is shared). Resumable: a
            PASS log newer than its seam is kept.
  analyze   re-derives gates A, R and B from the files, then
            GATE C -- the rebuilt unrepaired seam's harness scores reproduce the pipeline's own ep015 scores (PDMS
            61.18) token-for-token (every sub-score, |diff| <= 1e-12);
            GATE D -- every harness run PASS with every token valid: the two new runs, and the pipeline's own 015 and
            016 runs being compared.
            THE STATISTIC (reported; no decision rides on it): per-token PDMS(016 repaired) - PDMS(015 repaired), mean
            over the 200 tokens, paired log-cluster bootstrap over the logs, 10,000 resamples, percentile 95 %, seed
            20260927 (eval/snapshot_pair_under_rule.py's estimator). On the SAME draws: the repair's own delta
            PDMS(015 repaired) - PDMS(015 shipped), and the curve's published step PDMS(016) - PDMS(015 shipped).
            ESTIMATOR CONTROL: the same function re-reads Amendment 7's banked interval from its own CSVs.
            The interval answers ONE question -- another draw of EPISODES (logs). One checkpoint each, one deterministic
            forward: it says nothing about training variance (another run of epoch 16) or inference variance.
  selftest  deliberate-regression arms: mutated in-memory copies must turn gates A, B, C, R, a64 and b64 RED while the
            unmutated inputs stay GREEN; the estimator must return exactly [c, c] for a constant difference c.

EXTENSION (coordinator, 2026-09-27: did the planner's BEST proposal degenerate or stabilise?), only after the pick
readout's gates pass; `all64` = build64 -> score64 -> analyze64:
  build64   ALL 64 proposals of ep015 from the same native dump. GATE a64 -- the unrepaired rebuild of every proposal
            equals the pipeline's own ep015 E-6 poses (proptable/sub200_ep015/table.npz `proposals`, which equal its
            proposals.npz) bit-for-bit, 12,800 of 12,800. GATE b64 -- only the 4.0 s heading differs (as gate B, per
            proposal). TIE -- the pick column equals the pick readout's repaired seam. Writes 64 seams.
  score64   the 64 seams through score_navtest_refe.py UNCHANGED, at most 2 concurrent runs; resumable.
  analyze64 G3 (64 PASS, every token valid, each tied to its seam file) and c64 (the pick column reproduces the pick
            readout's repaired harness scores), then the repaired E-6 table proptable/sub200_ep015_repaired/table.npz
            in proposal_table.py's format (ep015's logits, pick and rule), then eval/oracle_trend.py UNCHANGED:
            --pairs 015:016 015_repaired:016 015:015_repaired (best-of-64, mean-of-64, share >= 80; paired log-cluster
            bootstrap, seed 20260927). The position-only oracle ADE needs no re-read (the repair moves no position).

FOUR FAMILIES: not recomputed. They read positions only (the NAVSIM family adapter drops the heading column; Amendment
7 ran families6.py on a shipped and a repaired seam and found the blocks identical), and GATE B proves that
015-repaired carries 015-shipped's positions bit-for-bit -- so the pipeline's existing per-snapshot family readouts for
015 and 016 are already like-for-like.

    python eval/like_for_like_016.py all --workers 2
    python eval/like_for_like_016.py selftest
Outputs: eval/raw/like_for_like_016/ (build_record.json, like_for_like_016.json, selftest.json, harness/, logs/,
the repaired seam npz, MANIFEST.md) and eval/RESULT_like_for_like_016.md.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_checkpoint as EC        # noqa: E402  the interpreter + env the pipeline scores with (stdlib only)
import refe_navtest_seam as SEAM    # noqa: E402  the pipeline's own converter `to_navsim` (numpy only; run, not re-implemented)

PKG = os.path.dirname(HERE)
PLANNER_PY = os.path.join(PKG, "refe", "planner.py")
TOK = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
       "A1_sub200_tokens.json")
TOK_MD5 = "114deb5bf6d2631a9cdd6e8c0724273d"
A, B = "sub200_ep015", "sub200_ep016"           # the earlier point (shipped WITHOUT the repair), the later (WITH it)
DUMP = f"{EC.DATA}/proptable/{A}/stop_candidate_dump.npz"
SEAM_OF = {A: f"{EC.DATA}/seams/refe_{A}.npz", B: f"{EC.DATA}/seams/refe_{B}.npz"}
REPORT_OF = {A: f"{EC.DATA}/seams/refe_{A}.report.json", B: f"{EC.DATA}/seams/refe_{B}.report.json"}
PROP_OF = {A: f"{EC.DATA}/proptable/{A}/proposals.npz", B: f"{EC.DATA}/proptable/{B}/proposals.npz"}
POINT_OF = {A: f"{EC.DATA}/points/{A}.json", B: f"{EC.DATA}/points/{B}.json"}
CSV_OF = {A: f"{EC.DATA}/score/refe_{A}/refe_{A}.csv", B: f"{EC.DATA}/score/refe_{B}/refe_{B}.csv"}
FAM_OF = {A: f"{EC.DATA}/points/{A}/4_families.json", B: f"{EC.DATA}/points/{B}/4_families.json"}
CKPT_OF = {A: "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt",
           B: "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch016.pt"}
CKPT_MD5_A = "d7c59f4f2fbcbde3e2dec8f67d63a7e7"        # the md5 a7_confirm.py asserts for the same snapshot
TAG = "lfl016"
LABELS = {"unrepaired": f"refe_{TAG}_ep015_unrepaired", "repaired": f"refe_{TAG}_ep015_repaired"}
# the harness passes the seam path through a Hydra override, so the seams live in the data dir (no spaces in the path)
SEAMS = {k: f"{EC.DATA}/seams/{v}.npz" for k, v in LABELS.items()}
RAW = os.path.join(HERE, "raw", "like_for_like_016")
LOGDIR = os.path.join(RAW, "logs")
HARNESS_COPY = os.path.join(RAW, "harness")
BUILD_JSON = os.path.join(RAW, "build_record.json")
OUT_JSON = os.path.join(RAW, "like_for_like_016.json")
SELFTEST_JSON = os.path.join(RAW, "selftest.json")
REPAIRED_COPY = os.path.join(RAW, f"{LABELS['repaired']}.npz")
RESULT_MD = os.path.join(HERE, "RESULT_like_for_like_016.md")
MANIFEST = os.path.join(RAW, "MANIFEST.md")
SEED = 20260927
SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
       "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
HEAD = ("NC", "DAC", "EP", "TTC", "C", "DDC")
TOL = 1e-12                                             # the harness's own C1 tolerance
# the estimator control: Amendment 7's banked interval, re-read from its own artifacts
A7_JSON = os.path.join(HERE, "raw", "a7_confirm", "a7_confirm_ep015.json")
A7_TOK = os.path.join(HERE, "raw", "a5_confirm", "a5_confirm_tokens.json")
A7_SEAM = f"{EC.DATA}/seams/refe_a7confirm_ep015.npz"
A7_CSV = {"shipped": f"{EC.DATA}/score/refe_a7confirm_ep015/refe_a7confirm_ep015.csv",
          "repaired": f"{EC.DATA}/score/refe_a7confirm_ep015_repaired/refe_a7confirm_ep015_repaired.csv"}
# EXTENSION (coordinator, 2026-09-27): ALL 64 proposals of ep015, repaired, as an E-6 table that eval/oracle_trend.py
# reads UNCHANGED -- it loads proptable/sub200_ep<x>/table.npz, so x = "015_repaired" names this table
NAME64 = "sub200_ep015_repaired"
WD64 = f"{EC.DATA}/proptable/{NAME64}"
TABLE64 = f"{WD64}/table.npz"
SEAM64_DIR = f"{EC.DATA}/seams/proptable/{NAME64}"
TABLE_OF = {A: f"{EC.DATA}/proptable/{A}/table.npz", B: f"{EC.DATA}/proptable/{B}/table.npz"}
GATES_OF = {A: f"{EC.DATA}/proptable/{A}/gates.json", B: f"{EC.DATA}/proptable/{B}/gates.json"}
BUILD64_JSON = os.path.join(RAW, "build64_record.json")
OUT64_JSON = os.path.join(RAW, "like_for_like_016_64.json")
TREND_JSON = os.path.join(RAW, "oracle_trend_ep015rep_ep016.json")
SCORES64_COPY = os.path.join(RAW, f"table_{NAME64}_scores.npz")
TREND_PAIRS = ("015:016", "015_repaired:016", "015:015_repaired")
M64 = 64


def label64(k):
    return f"refe_{NAME64}_p{k:02d}"


def seam64(k):
    return f"{SEAM64_DIR}/{label64(k)}.npz"


# the exploration that motivated Amendment 7, on these same 200 tokens (a reported reproduction, never a gate)
EXPL_SEAM = f"{EC.DATA}/seams/proptable/{A}_slow/refe_{A}_f100_pickhold.npz"
EXPL_CSV = f"{EC.DATA}/score/refe_{A}_f100_pickhold/refe_{A}_f100_pickhold.csv"
EXPL_JSON = os.path.join(HERE, "raw", "e6_sub200_ep015", "slow_copies.json")


# ------------------------------------------------------------------------------------------------ helpers
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def bits_equal(x, y) -> bool:
    """BIT identity (stricter than ==: -0.0 and 0.0 differ), for arrays of one dtype and shape."""
    x, y = np.asarray(x), np.asarray(y)
    return (x.dtype == y.dtype and x.shape == y.shape
            and np.ascontiguousarray(x).tobytes() == np.ascontiguousarray(y).tobytes())


def no_torch():
    return {"torch_imported": "torch" in sys.modules, "python": sys.executable,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}


def planner_repair():
    """planner.py's OWN `repair_last_heading`, compiled from its source WITHOUT importing the module (which imports
    torch and the nuPlan devkit): the shipped function body runs here, not a re-implementation, and its numpy branch
    is the one taken. Returns (fn, provenance)."""
    src = open(PLANNER_PY, encoding="utf-8").read()
    fns = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "repair_last_heading"]
    if len(fns) != 1:
        raise RuntimeError(f"planner.py defines repair_last_heading {len(fns)} times at module level")
    seg = ast.get_source_segment(src, fns[0])
    ns = {"np": np, "torch": types.SimpleNamespace(Tensor=type("NotATorchTensor", (), {}))}
    exec(compile(ast.Module(body=fns, type_ignores=[]), PLANNER_PY, "exec"), ns)
    return ns["repair_last_heading"], {"planner_py": PLANNER_PY, "planner_py_sha256": sha256(PLANNER_PY),
                                       "function_source_sha256": hashlib.sha256(seg.encode("utf-8")).hexdigest(),
                                       "function_first_line": seg.splitlines()[0]}


def literal_repair(x):
    """the registered repair written out independently: heading[19] := heading[18]; nothing else changes"""
    q = np.array(x, copy=True)
    q[..., 19, 2] = q[..., 18, 2]
    return q


def nav(batch20):
    """[n, 20, 3] -> [n, 8, 3] float32: the seam writer's conversion (refe_navtest_seam.to_navsim) and dtype"""
    return np.stack([SEAM.to_navsim(x).astype(np.float32) for x in batch20])


def read_csv(p):
    rows = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["token"] == "average":
            continue
        rows[r["token"]] = (r["valid"] == "True", float(r["score"]), tuple(float(r[c]) for c in SUB))
    return rows


def status_of(txt):
    for line in reversed(txt.splitlines()):
        if line.startswith("{") and '"status"' in line:
            return json.loads(line)
    return None


def csv_of(label):
    return f"{EC.DATA}/score/{label}/{label}.csv"


def token_log():
    return json.load(open(TOK, encoding="utf-8"))["token_log"]


# ------------------------------------------------------------------------------------------------ the gates (pure)
def recover_matches(traj, seam_poses):
    """for every token i: every k with to_navsim(traj[i, k]) as float32 BIT-identical to the shipped seam row"""
    return [[k for k in range(traj.shape[1]) if bits_equal(SEAM.to_navsim(traj[i, k]).astype(np.float32), seam_poses[i])]
            for i in range(traj.shape[0])]


def gate_a(matches, tokens):
    bad = [{"token": tokens[i], "n_matches": len(m), "matching_k": m} for i, m in enumerate(matches) if len(m) != 1]
    n = len(matches)
    return {"what": "each token's shipped pick recovered from the ep015 native dump: to_navsim(traj[token, k]) as float32 "
                    "must equal the shipped ep015 seam's 8 poses bit-for-bit, for exactly ONE k",
            "tokens_with_unique_exact_match": f"{sum(len(m) == 1 for m in matches)}/{n}",
            "tokens_with_0_matches": sum(len(m) == 0 for m in matches),
            "tokens_with_more_than_1_match": sum(len(m) > 1 for m in matches),
            "failing_tokens": bad, "pass": not bad and n > 0}


def gate_b(rep, shipped, native_pick):
    """rep, shipped [n, 8, 3] float32 NAVSIM poses; native_pick [n, 20, 3] float32 (the UNREPAIRED picked proposal)"""
    xy = bits_equal(rep[..., :2], shipped[..., :2])
    early = bits_equal(rep[:, :7, 2], shipped[:, :7, 2])
    h4 = bits_equal(rep[:, 7, 2], native_pick[:, 18, 2])
    changed = rep[:, 7, 2] != shipped[:, 7, 2]
    return {"what": "repaired seam vs the shipped ep015 seam: x, y bit-identical on all 8 poses of all tokens; headings "
                    "0.5-3.5 s bit-identical; the 4.0 s heading bit-identical to the native heading[18] -- so only the "
                    "4.0 s heading can differ",
            "x_y_bit_identical_all_8_poses": xy, "heading_bit_identical_0.5_to_3.5s": early,
            "heading_4.0s_bit_identical_to_native_heading18": h4,
            "tokens_whose_4.0s_heading_changed": int(changed.sum()), "n": int(rep.shape[0]),
            "pass": bool(xy and early and h4)}


def gate_c(rows_new, rows_pipe, status, tokens):
    missing = [t for t in tokens if t not in rows_new or t not in rows_pipe]
    ok_rows = [t for t in tokens if t in rows_new and t in rows_pipe]
    d = max((abs(rows_new[t][1] - rows_pipe[t][1]) for t in ok_rows), default=float("inf"))
    dsub = max((max(abs(x - y) for x, y in zip(rows_new[t][2], rows_pipe[t][2])) for t in ok_rows),
               default=float("inf"))
    valid_new = sum(1 for t in ok_rows if rows_new[t][0])
    valid_pipe = sum(1 for t in ok_rows if rows_pipe[t][0])
    n = len(tokens)
    mean_new = 100 * float(np.mean([rows_new[t][1] for t in ok_rows])) if ok_rows else None
    mean_pipe = 100 * float(np.mean([rows_pipe[t][1] for t in ok_rows])) if ok_rows else None
    return {"what": "the rebuilt UNREPAIRED seam, scored today by the unchanged harness, reproduces the pipeline's own "
                    "ep015 scores token-for-token (PDMS and all six sub-scores, |diff| <= 1e-12)",
            "harness_status": (status or {}).get("status"), "tokens_missing": len(missing),
            "valid_rebuilt": f"{valid_new}/{n}", "valid_pipeline": f"{valid_pipe}/{n}",
            "max_abs_pdms_diff": d, "max_abs_subscore_diff": dsub,
            "pdms_rebuilt_x100": None if mean_new is None else round(mean_new, 4),
            "pdms_pipeline_x100": None if mean_pipe is None else round(mean_pipe, 4),
            "pass": bool((status or {}).get("status") == "PASS" and not missing and valid_new == n and valid_pipe == n
                         and d <= TOL and dsub <= TOL)}


def gate_r(rules):
    """rules: {point: {"report": (rule, select), "dump": (rule, select)}}"""
    vals = {(str(v["report_rule"]), str(v["report_select"])) for v in rules.values()} | \
           {(str(v["dump_rule"]), str(v["dump_select"])) for v in rules.values()}
    return {"what": "the selection rule and `select` each point was picked with, from its seam report AND its pipeline "
                    "proposal dump: one rule for both points, or the readout is not like-for-like",
            "distinct_rule_select_pairs": sorted([list(v) for v in vals]),
            "pass": len(vals) == 1 and "None" not in {x for v in vals for x in v}}


def cluster_draws(tokens, tlog, boot, seed=SEED):
    """eval/snapshot_pair_under_rule.py's draws: resample the LOGS with replacement, keep each drawn log's tokens"""
    logs = np.array([tlog[t] for t in tokens])
    ul = np.unique(logs)
    idx = {l: np.where(logs == l)[0] for l in ul}
    rng = np.random.default_rng(seed)
    return [np.concatenate([idx[l] for l in rng.choice(ul, size=len(ul), replace=True)]) for _ in range(boot)], int(len(ul))


def paired(d, draws):
    bs = np.array([100 * d[ix].mean() for ix in draws])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"delta": round(100 * float(d.mean()), 2), "ci95": [round(lo, 2), round(hi, 2)],
            "separated": bool(lo > 0 or hi < 0), "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
            "tied": int((d == 0).sum()), "delta_unrounded": 100 * float(d.mean()), "ci95_unrounded": [lo, hi]}


# ------------------------------------------------------------------------------------------------ inputs
def load_inputs():
    """every input, with the structural checks that make the token axis one axis. The dump's per-token arrays are
    returned RE-INDEXED BY TOKEN into the shipped ep015 seam's order (never by an assumed common order)."""
    Dz = np.load(DUMP, allow_pickle=False)
    SA, SB = np.load(SEAM_OF[A]), np.load(SEAM_OF[B])
    dt, sa, sb = [str(t) for t in Dz["token"]], [str(t) for t in SA["token"]], [str(t) for t in SB["token"]]
    tl = token_log()
    di = {t: i for i, t in enumerate(dt)}
    struct = {"dump_traj_shape": list(Dz["traj"].shape), "dump_traj_dtype": str(Dz["traj"].dtype),
              "dump_token_key": "token", "n_tokens": len(sa),
              "dump_holds_every_ep015_seam_token_once": len(di) == len(dt) and all(t in di for t in sa)
                                                          and len(dt) == len(sa),
              "dump_tokens_same_order_as_ep015_seam": dt == sa,
              "ep015_ep016_seams_same_token_order": sa == sb,
              "ep015_ep016_seams_same_fingerprints": bool(np.array_equal(SA["fingerprint"], SB["fingerprint"])),
              "seam_tokens_equal_token_file": sorted(sa) == sorted(json.load(open(TOK, encoding="utf-8"))["tokens"]),
              "every_token_has_a_log": all(t in tl for t in sa),
              "tokens_unique": len(set(sa)) == len(sa)}
    ok = (Dz["traj"].ndim == 4 and Dz["traj"].shape[2:] == (20, 3)
          and struct["dump_holds_every_ep015_seam_token_once"] and struct["ep015_ep016_seams_same_token_order"]
          and struct["ep015_ep016_seams_same_fingerprints"] and struct["seam_tokens_equal_token_file"]
          and struct["every_token_has_a_log"] and struct["tokens_unique"])
    struct["pass"] = bool(ok)
    ri = np.array([di.get(t, 0) for t in sa], dtype=np.int64)
    D = {"traj": Dz["traj"][ri], "pick_infer": Dz["pick_infer"][ri], "traj2_maxdiff": Dz["traj2_maxdiff"][ri],
         "token": Dz["token"][ri], "meta": Dz["meta"]}
    return D, SA, SB, sa, struct


def rules_record():
    out = {}
    for p in (A, B):
        rep = json.load(open(REPORT_OF[p], encoding="utf-8"))
        P = np.load(PROP_OF[p], allow_pickle=False)
        out[p] = {"report_rule": rep.get("rule"), "report_select": rep.get("select"),
                  "dump_rule": str(P["rule"]) if "rule" in P.files else None,
                  "dump_select": str(P["select"]) if "select" in P.files else None,
                  "report_repair_last_heading": rep.get("repair_last_heading", "ABSENT (the report predates Amendment 7)"),
                  "dump_repair_last_heading": (bool(P["repair_last_heading"]) if "repair_last_heading" in P.files
                                               else "ABSENT (the dump predates Amendment 7)"),
                  "ckpt": rep.get("ckpt"), "rows": rep.get("rows"), "misses": rep.get("misses")}
    return out


# ------------------------------------------------------------------------------------------------ build
def build_all():
    """pure: every array and every gate of the build, from the files"""
    D, SA, SB, toks, struct = load_inputs()
    traj = D["traj"]
    matches = recover_matches(traj, SA["poses"])
    GA = gate_a(matches, toks)
    rules = rules_record()
    GR = gate_r(rules)
    res = {"structure": struct, "gate_A": GA, "gate_R": GR, "rules": rules}
    if not (struct["pass"] and GA["pass"]):
        return res, None
    k = np.array([m[0] for m in matches], dtype=np.int64)
    native_pick = traj[np.arange(len(toks)), k]                           # [n, 20, 3] float32, UNREPAIRED
    fn, prov = planner_repair()
    rep_native = fn(native_pick)
    lit = literal_repair(native_pick)
    unrep = nav(native_pick)
    rep = nav(rep_native)
    GB = gate_b(rep, SA["poses"], native_pick)
    Pd = np.load(PROP_OF[A], allow_pickle=False)
    pix = {str(t): i for i, t in enumerate(Pd["token"])}
    pipe_pick = np.array([int(Pd["pick"][pix[t]]) for t in toks])
    meta = json.loads(str(D["meta"]))
    res.update({
        "gate_B": GB,
        "repair_function": dict(prov, **{
            "planner_fn_equals_independent_literal_bitwise": bits_equal(rep_native, lit),
            "planner_fn_changes_only_heading19": bool(bits_equal(rep_native[:, :19], native_pick[:, :19])
                                                      and bits_equal(rep_native[:, 19, :2], native_pick[:, 19, :2]))}),
        "cross_checks_reported": {
            "recovered_k_equals_pipeline_proposal_dump_pick": f"{int((k == pipe_pick).sum())}/{len(toks)}",
            "recovered_k_equals_dump_pick_infer": f"{int((k == D['pick_infer']).sum())}/{len(toks)}",
            "rebuilt_unrepaired_seam_bit_identical_to_shipped_ep015_seam": bits_equal(unrep, SA["poses"]),
            "repaired_conversion_also_matches_shipped_seam_on_tokens": int(sum(bits_equal(rep[i], SA["poses"][i])
                                                                             for i in range(len(toks)))),
            "dump_meta": {kk: meta.get(kk) for kk in ("ckpt", "tokens", "planner_rule", "planner_select", "rows",
                                                      "n_misses", "gpu_finished_local", "extra_forward")},
            "dump_second_forward_max_abs_pose_diff": float(np.max(D["traj2_maxdiff"])),
            "what": "the pick recovered by exact match against the pipeline's own record of its pick (proposals.npz, "
                    "written by the 09:30 forward) and the dump's; the dump's forward repeated itself bit-for-bit "
                    "(traj2_maxdiff), consistent with a deterministic forward"}})
    arrays = {"k": k, "native_pick": native_pick, "unrepaired": unrep, "repaired": rep,
              "token": SA["token"], "fingerprint": SA["fingerprint"], "sampling": SA["sampling"]}
    return res, arrays


def build(a) -> int:
    got = md5(TOK)
    if got != TOK_MD5:
        print(f"ZZLFL_FAIL token file md5 {got} != {TOK_MD5}"); return 1
    os.makedirs(RAW, exist_ok=True)
    res, arr = build_all()
    res["_label"] = "like-for-like 016: build record (gates A, R, B; the two seams)"
    res["built_local"] = now()
    res["inputs"] = {"dump": DUMP, "dump_sha256": sha256(DUMP), "ep015_seam": SEAM_OF[A],
                     "ep015_seam_sha256": sha256(SEAM_OF[A]), "tokens": TOK, "tokens_md5": TOK_MD5,
                     "ckpt_md5": {A: md5(CKPT_OF[A]), B: md5(CKPT_OF[B])},
                     "ckpt_015_md5_equals_a7_asserted": md5(CKPT_OF[A]) == CKPT_MD5_A}
    ok = bool(res["structure"]["pass"] and res["gate_A"]["pass"])
    if ok:
        ok = bool(res["gate_B"]["pass"] and res["repair_function"]["planner_fn_equals_independent_literal_bitwise"]
                  and res["cross_checks_reported"]["rebuilt_unrepaired_seam_bit_identical_to_shipped_ep015_seam"])
    if ok:
        for key in ("unrepaired", "repaired"):
            np.savez(SEAMS[key], token=arr["token"], fingerprint=arr["fingerprint"], poses=arr[key],
                     sampling=arr["sampling"], arm=np.array(f"REFe_{TAG}_ep015_{key}"))
        shutil.copyfile(SEAMS["repaired"], REPAIRED_COPY)
        res["seams"] = {key: {"path": SEAMS[key], "md5": md5(SEAMS[key]), "sha256": sha256(SEAMS[key])}
                        for key in SEAMS}
        res["seams"]["repaired"]["repo_copy"] = REPAIRED_COPY
        res["seams"]["repaired"]["repo_copy_md5_equal"] = md5(REPAIRED_COPY) == res["seams"]["repaired"]["md5"]
        res["recovered_pick_k"] = {str(t): int(kk) for t, kk in zip(arr["token"], arr["k"])}
    res["cpu_only"] = no_torch()
    json.dump(res, open(BUILD_JSON, "w", encoding="utf-8"), indent=1)
    ga, gr = res["gate_A"], res["gate_R"]
    print(f"  gate A: {ga['tokens_with_unique_exact_match']} unique exact matches; 0-match {ga['tokens_with_0_matches']}, "
          f">1-match {ga['tokens_with_more_than_1_match']} -> pass={ga['pass']}", flush=True)
    print(f"  gate R: rule/select pairs {gr['distinct_rule_select_pairs']} -> pass={gr['pass']}", flush=True)
    if "gate_B" in res:
        gb = res["gate_B"]
        print(f"  gate B: x/y {gb['x_y_bit_identical_all_8_poses']}, h 0.5-3.5 s {gb['heading_bit_identical_0.5_to_3.5s']}, "
              f"h 4.0 s == native h[18] {gb['heading_4.0s_bit_identical_to_native_heading18']}; changed on "
              f"{gb['tokens_whose_4.0s_heading_changed']}/{gb['n']} -> pass={gb['pass']}", flush=True)
    if not ok:
        print("ZZLFL_FAIL build (a gate failed; see build_record.json) -- STOP"); return 1
    print(f"  wrote {SEAMS['unrepaired']} (md5 {res['seams']['unrepaired']['md5']}) and {SEAMS['repaired']} "
          f"(md5 {res['seams']['repaired']['md5']}); torch imported: {res['cpu_only']['torch_imported']}", flush=True)
    print("ZZLFL_BUILD_OK")
    return 0


# ------------------------------------------------------------------------------------------------ score
def score(a) -> int:
    if a.workers not in (1, 2):
        print("ZZLFL_FAIL at most 2 concurrent harness runs (the CPU is shared)"); return 1
    for k_, p in SEAMS.items():
        if not os.path.exists(p):
            print(f"ZZLFL_FAIL seam {k_} missing -- run build first"); return 1
    os.makedirs(LOGDIR, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor
    # the pipeline's step-2 environment; CUDA hidden from the outer process too (the harness hides it from the devkit)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")

    def one(k_):
        lb, log = LABELS[k_], os.path.join(LOGDIR, f"score_{k_}.log")
        for attempt in range(a.tries):
            if (os.path.exists(log) and os.path.exists(csv_of(lb))
                    and os.path.getmtime(log) > os.path.getmtime(SEAMS[k_])):
                st = status_of(open(log, encoding="utf-8", errors="replace").read())
                if st and st.get("status") == "PASS":
                    return k_, st
            if attempt:
                time.sleep(60)                                  # the W3 RAM guard: give the box a minute
            print(f"  harness {lb}: attempt {attempt + 1} started {now()}", flush=True)
            EC.run([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", lb, "--seam", SEAMS[k_], "--tokens", TOK,
                    "--out", f"{EC.DATA}/score"], HERE, env, log)
        st = status_of(open(log, encoding="utf-8", errors="replace").read()) if os.path.exists(log) else None
        return k_, st
    res = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for k_, st in ex.map(one, list(LABELS)):
            res[k_] = st
            print(f"  scored {k_}: {st.get('status') if st else None} valid rows "
                  f"{st.get('csv_valid_rows') if st else None} PDMS {(st or {}).get('summary_x100_4dp', {}).get('PDMS')}"
                  f"  {now()}", flush=True)
    ok = all(st and st.get("status") == "PASS" for st in res.values())
    print(f"ZZLFL_SCORE_{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ analyze
def same_path(x, y):
    return bool(x and y and os.path.normcase(os.path.abspath(x)) == os.path.normcase(os.path.abspath(y)))


def pipeline_run(p, toks):
    P = json.load(open(POINT_OF[p], encoding="utf-8"))
    sc = P.get("score") or {}
    rows = read_csv(CSV_OF[p])
    nvalid = sum(1 for t in toks if t in rows and rows[t][0])
    return {"label": sc.get("label"), "status": sc.get("status"), "csv_valid_rows": sc.get("csv_valid_rows"),
            "tokens_valid": f"{nvalid}/{len(toks)}", "summary_PDMS_x100": (sc.get("summary_x100_4dp") or {}).get("PDMS"),
            "csv": CSV_OF[p], "point_record": POINT_OF[p], "point_tokens": P.get("tokens"),
            "point_tokens_are_this_token_file": same_path(P.get("tokens"), TOK), "point_ckpt": P.get("ckpt"),
            "point_ckpt_is_this_snapshot": same_path(P.get("ckpt"), CKPT_OF[p])}


def estimator_control(boot):
    """the SAME estimator on Amendment 7's own CSVs must re-read its banked interval exactly"""
    if not all(os.path.exists(p) for p in (A7_JSON, A7_TOK, A7_SEAM, *A7_CSV.values())):
        return {"status": "NOT RUN -- an A7 artifact is missing", "pass": None}
    banked = json.load(open(A7_JSON, encoding="utf-8"))["statistic"]
    stok = [str(t) for t in np.load(A7_SEAM)["token"]]
    rows = {k_: read_csv(p) for k_, p in A7_CSV.items()}
    d = np.array([rows["repaired"][t][1] - rows["shipped"][t][1] for t in stok])
    draws, nl = cluster_draws(stok, json.load(open(A7_TOK, encoding="utf-8"))["token_log"], boot)
    got = paired(d, draws)
    ok = got["delta"] == banked["delta"] and got["ci95"] == banked["ci95"]
    return {"what": "this script's estimator on Amendment 7's own per-token CSVs (923 tokens) must reproduce the "
                    "banked delta and interval exactly",
            "banked": {"delta": banked["delta"], "ci95": banked["ci95"], "source": A7_JSON},
            "reread": {"delta": got["delta"], "ci95": got["ci95"], "n_logs": nl}, "pass": bool(ok)}


def exploration_reproduction(rep_poses, rows_rep, toks):
    out = {"what": "Amendment 7's motivating exploration on THESE 200 tokens (slow_copy_probe.py, the shipped pick with "
                   "heading[19] := heading[18], 'pickhold'): a reported reproduction, not a gate"}
    try:
        E = np.load(EXPL_SEAM)
        same_order = [str(t) for t in E["token"]] == toks
        out["seam"] = EXPL_SEAM
        out["poses_bit_identical_to_this_repaired_seam"] = bool(same_order and bits_equal(E["poses"], rep_poses))
        er = read_csv(EXPL_CSV)
        out["csv"] = EXPL_CSV
        out["per_token_max_abs_pdms_diff_vs_this_run"] = max(abs(er[t][1] - rows_rep[t][1]) for t in toks)
        ej = json.load(open(EXPL_JSON, encoding="utf-8"))["repair_last_heading_shipped_pick"]["hold"]
        out["banked"] = {"pdms": ej.get("pdms"), "minus_shipped": ej.get("minus_shipped"),
                         "ci95_vs_shipped": ej.get("ci95_vs_shipped"), "source": EXPL_JSON}
    except Exception as e:                                          # absent, and said so -- never silently dropped
        out["status"] = f"NOT AVAILABLE: {type(e).__name__}: {e}"
    return out


def where_the_difference_comes_from(rows_a, rows_b, toks):
    """DESCRIPTIVE: each token's PDMS change classified by WHICH NAVSIM-v1 terms moved (PDMS = NC x DAC x (5 EP + 5 TTC
    + 2 C) / 12; DDC has weight 0) -- a multiplier, TTC or comfort, or only ego progress -- and each class's
    contribution to the mean difference (the contributions sum to it)"""
    n, cls = len(toks), {}
    for t in toks:
        (_, pa, (nca, daca, epa, ttca, ca, _)), (_, pb, (ncb, dacb, epb, ttcb, cb, _)) = rows_a[t], rows_b[t]
        d = pb - pa
        if d == 0:
            key = "tied"
        elif nca != ncb or daca != dacb:
            key = "a multiplier moved (NC or DAC)"
        elif ttca != ttcb or ca != cb:
            key = "TTC or comfort moved, multipliers unchanged"
        elif epa != epb:
            key = "only ego progress moved"
        else:
            key = "other"
        c = cls.setdefault(key, {"n": 0, "better": 0, "worse": 0, "_sum": 0.0})
        c["n"] += 1
        c["better"] += int(d > 0)
        c["worse"] += int(d < 0)
        c["_sum"] += d
    for c in cls.values():
        c["contribution_to_mean_difference_x100"] = round(100 * c.pop("_sum") / n, 2)
    return cls


# the inference path's last COMMITTED versions (push mirror, read-only `git show`; D:'s own HEAD does not carry the
# package): both older than both points' runs, so a net diff against today's files bounds what changed between them
MIRROR_GIT = "C:/Users/Admin/tanitad-push/.git"
PKG_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
CODE_PATH_REVS = {"refe/model.py": "4ac00b4", "refe/planner.py": "7e9ccdd", "eval/refe_navtest_seam.py": "4ac00b4"}


def code_path_check():
    """REPORTED: apart from the repair, did 015's forward (10:45) and 016's (16:24) run one inference path? Net diff of
    today's files against their last committed versions; the changed lines are recorded for the reader to classify.
    A net diff cannot exclude a change made and reverted in between; a GPU re-forward of 015 is the direct test."""
    import difflib
    import subprocess
    out = {"what": code_path_check.__doc__.split("\n")[0].strip() + " -- net diff vs the last committed versions",
           "mirror_git_dir": MIRROR_GIT, "files": {}}
    for rel, rev in CODE_PATH_REVS.items():
        try:
            old = subprocess.run(["git", "--git-dir", MIRROR_GIT, "show", f"{rev}:{PKG_REL}/{rel}"], capture_output=True,
                                 check=True).stdout.decode("utf-8").replace("\r", "")
            new = open(os.path.join(PKG, rel), encoding="utf-8").read().replace("\r", "")
            dl = [ln for ln in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
                  if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
            out["files"][rel] = {"committed_rev": rev, "identical": old == new,
                                 "lines_added": sum(ln.startswith("+") for ln in dl),
                                 "lines_removed": sum(ln.startswith("-") for ln in dl),
                                 "changed_lines": [ln[:200] for ln in dl]}
        except Exception as e:                                          # absent, and said so
            out["files"][rel] = {"committed_rev": rev, "status": f"NOT AVAILABLE: {type(e).__name__}: {e}"}
    return out


def csv_to_seam_tie(label, seam_path):
    """the harness's own SeamAgentV1 call log names the seam file it served and when: tie each compared CSV to the
    exact seam file, unmodified since it was scored"""
    cl = f"{EC.DATA}/score/{label}/{label}.calls.jsonl"
    if not os.path.exists(cl):
        return {"call_log": cl, "status": "MISSING", "pass": False}
    ev = [json.loads(ln) for ln in open(cl, encoding="utf-8") if ln.strip()]
    init = next((e for e in ev if e.get("event") == "initialize"), {})
    calls = sum(1 for e in ev if e.get("event") == "call" and e.get("source") == "seam")
    t0 = init.get("t")
    unchanged = bool(t0 is not None and os.path.getmtime(seam_path) <= t0)
    return {"call_log": cl, "seam_file_served": init.get("seam_file"), "arm_served": init.get("arm"),
            "is_this_seam": same_path(init.get("seam_file"), seam_path), "seam_calls": calls,
            "scored_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)) if t0 else None,
            "seam_mtime_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(seam_path))),
            "seam_unmodified_since_scored": unchanged,
            "pass": bool(same_path(init.get("seam_file"), seam_path) and calls == init.get("n_rows") and unchanged)}


def heading_signature(poses):
    h = np.asarray(poses, dtype=np.float64)[:, 7, 2]
    return {"share_abs_heading_4.0s_above_pi": round(float((np.abs(h) > np.pi).mean()), 4),
            "median_abs_heading_4.0s_rad": round(float(np.median(np.abs(h))), 4)}


def copy_harness_artifacts():
    """the small per-label harness outputs into the repo (csv, counts, devkit log); everything else by sha256"""
    out = {}
    for k_, lb in LABELS.items():
        src = f"{EC.DATA}/score/{lb}"
        dst = os.path.join(HARNESS_COPY, lb)
        os.makedirs(dst, exist_ok=True)
        files = {}
        for fn in sorted(os.listdir(src)):
            p = os.path.join(src, fn)
            files[fn] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
            if fn in (f"{lb}.csv", f"{lb}.counts.json", f"{lb}.log"):
                shutil.copyfile(p, os.path.join(dst, fn))
                files[fn]["repo_copy"] = os.path.join(dst, fn)
                files[fn]["repo_copy_sha256_equal"] = sha256(os.path.join(dst, fn)) == files[fn]["sha256"]
        out[lb] = {"harness_dir": src, "files": files}
    return out


def analyze(a) -> int:
    if md5(TOK) != TOK_MD5:
        print("ZZLFL_FAIL token file md5"); return 1
    res, arr = build_all()                                          # gates A, R, B re-derived from the files
    toks = [str(t) for t in arr["token"]] if arr is not None else []
    out = {"_label": "like-for-like PDMS readout for REFe snapshot 016: 015 re-read with SPEC_NAVTEST Amendment 7's "
                     "repair on the same 200 tokens (W3's A1_sub200 subset), so 015 -> 016 is repaired-vs-repaired",
           "tier": "NAVSIM v1 PDMS = ego pseudo-simulation of an open-loop plan against logged agents (SPEC_NAVTEST "
                   "tier stamp); W3's 200-token paired subset, NOT the published split",
           "repair": "on the planner's native [20, 3] output, heading[19] := heading[18] for the EXECUTED plan, before "
                     "the NAVSIM conversion (refe/planner.py repair_last_heading); it acts after selection, so the pick "
                     "is unchanged",
           "analyzed_local": now(), "tokens": {"file": TOK, "md5": md5(TOK), "n": len(toks)},
           "points": {A: {"ckpt": CKPT_OF[A], "ckpt_md5": md5(CKPT_OF[A]), "seam": SEAM_OF[A], "csv": CSV_OF[A]},
                      B: {"ckpt": CKPT_OF[B], "ckpt_md5": md5(CKPT_OF[B]), "seam": SEAM_OF[B], "csv": CSV_OF[B]}},
           "structure": res["structure"], "rules_each_point_was_picked_with": res["rules"]}
    G = {"A_unique_exact_pick_recovery": res["gate_A"], "R_same_selection_rule": res["gate_R"]}
    if arr is not None:
        G["B_only_the_4s_heading_differs"] = res["gate_B"]
        out["repair_function"] = res["repair_function"]
        out["cross_checks_reported"] = res["cross_checks_reported"]
        for k_ in SEAMS:
            if not os.path.exists(SEAMS[k_]):
                G["seams_on_disk"] = {"pass": False, "missing": k_}
                break
        else:
            on_disk = {k_: np.load(SEAMS[k_]) for k_ in SEAMS}
            G["seams_on_disk_equal_rebuild"] = {
                "unrepaired": bits_equal(on_disk["unrepaired"]["poses"], arr["unrepaired"]),
                "repaired": bits_equal(on_disk["repaired"]["poses"], arr["repaired"]),
                "token_order": all([str(t) for t in on_disk[k_]["token"]] == toks for k_ in on_disk)}
            G["seams_on_disk_equal_rebuild"]["pass"] = all(G["seams_on_disk_equal_rebuild"].values())
    # ---------------------------------------------------------------- the harness runs
    runs, rows = {}, {}
    if arr is not None:
        for k_, lb in LABELS.items():
            log = os.path.join(LOGDIR, f"score_{k_}.log")
            st = status_of(open(log, encoding="utf-8", errors="replace").read()) if os.path.exists(log) else None
            rows[k_] = read_csv(csv_of(lb)) if os.path.exists(csv_of(lb)) else {}
            nvalid = sum(1 for t in toks if t in rows[k_] and rows[k_][t][0])
            runs[k_] = {"label": lb, "status": (st or {}).get("status"),
                        "csv_valid_rows": (st or {}).get("csv_valid_rows"), "tokens_valid": f"{nvalid}/{len(toks)}",
                        "C1_max_abs_delta": (st or {}).get("C1_max_abs_delta"), "failures": (st or {}).get("failures"),
                        "summary_x100_4dp": (st or {}).get("summary_x100_4dp"), "wall_s": (st or {}).get("wall_s"),
                        "stdout_log": log}
        rows["pipe_A"], rows["pipe_B"] = read_csv(CSV_OF[A]), read_csv(CSV_OF[B])
        G["C_rebuilt_unrepaired_reproduces_pipeline_ep015"] = gate_c(rows["unrepaired"], rows["pipe_A"],
                                                                     {"status": runs["unrepaired"]["status"]}, toks)
        pr = {A: pipeline_run(A, toks), B: pipeline_run(B, toks)}
        G["D_every_harness_run_PASS_every_token_valid"] = {
            "new_runs": runs, "pipeline_runs": pr,
            "pass": bool(all(r["status"] == "PASS" and r["tokens_valid"] == f"{len(toks)}/{len(toks)}"
                             for r in list(runs.values()) + list(pr.values()))
                         and all(p_["point_tokens_are_this_token_file"] and p_["point_ckpt_is_this_snapshot"]
                                 for p_ in pr.values()))}
    G["all_gates_pass"] = bool(arr is not None and all(v["pass"] for v in G.values() if isinstance(v, dict)))
    out["gates"] = G
    for k_, v in G.items():
        if isinstance(v, dict):
            print(f"  gate {k_}: pass={v['pass']}", flush=True)
    os.makedirs(RAW, exist_ok=True)
    if not G["all_gates_pass"]:
        like = G["R_same_selection_rule"]["pass"]
        out["statistic"] = ("NOT READ -- the two points were picked under DIFFERENT selection rules, so this readout is "
                            "NOT like-for-like" if not like else "NOT READ -- a validity gate failed")
        out["cpu_only"] = no_torch()
        json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
        print("ZZLFL_FAIL gates -- statistic not read"); return 1
    # ---------------------------------------------------------------- the statistic (reported, no decision)
    pd_ = {"015_shipped": np.array([rows["pipe_A"][t][1] for t in toks]),
           "015_repaired": np.array([rows["repaired"][t][1] for t in toks]),
           "016_repaired": np.array([rows["pipe_B"][t][1] for t in toks])}
    sub = {"015_shipped": np.array([rows["pipe_A"][t][2] for t in toks]),
           "015_repaired": np.array([rows["repaired"][t][2] for t in toks]),
           "016_repaired": np.array([rows["pipe_B"][t][2] for t in toks])}
    draws, nl = cluster_draws(toks, token_log(), a.boot)
    comps = {"016_repaired_minus_015_repaired": ("015_repaired", "016_repaired",
                                                 "LIKE-FOR-LIKE: both points executed with the repair; the epoch of "
                                                 "training, read against another draw of episodes"),
             "015_repaired_minus_015_shipped": ("015_shipped", "015_repaired",
                                                "the repair alone, snapshot 015, these 200 tokens"),
             "016_repaired_minus_015_shipped": ("015_shipped", "016_repaired",
                                                "the curve's step as published (repair + one epoch mixed)")}
    stat = {}
    for name, (lo_, hi_, what) in comps.items():
        d = pd_[hi_] - pd_[lo_]
        stat[name] = dict(paired(d, draws), what=what, a=lo_, b=hi_,
                          a_pdms=round(100 * float(pd_[lo_].mean()), 2), b_pdms=round(100 * float(pd_[hi_].mean()), 2),
                          subscore_deltas_x100=dict(zip(HEAD, [round(100 * float(x), 2)
                                                               for x in (sub[hi_] - sub[lo_]).mean(0)])))
    s1, s2, s3 = (stat["016_repaired_minus_015_repaired"], stat["015_repaired_minus_015_shipped"],
                  stat["016_repaired_minus_015_shipped"])
    s1["where_the_difference_comes_from_reported"] = where_the_difference_comes_from(rows["repaired"], rows["pipe_B"],
                                                                                     toks)
    out["statistic"] = {
        "headline": "016_repaired_minus_015_repaired",
        "estimator": f"paired log-cluster bootstrap over the {nl} logs, {a.boot} resamples, percentile 95 %, seed {SEED} "
                     "(eval/snapshot_pair_under_rule.py's estimator); all three comparisons on the SAME draws",
        "variance_answered": "EPISODES only (another draw of logs). One checkpoint per point, one deterministic "
                             "forward: NOT training variance (another run of epoch 16 -- no replicate exists) and NOT "
                             "inference variance",
        "n_tokens": len(toks), "n_logs": nl,
        "pdms_x100": {k_: round(100 * float(v.mean()), 2) for k_, v in pd_.items()},
        "subscores_x100": {k_: dict(zip(HEAD, [round(100 * float(x), 2) for x in v.mean(0)])) for k_, v in sub.items()},
        "comparisons": stat,
        "decomposition_identity_per_token": bool(np.allclose(s1["delta_unrounded"] + s2["delta_unrounded"],
                                                             s3["delta_unrounded"], atol=1e-9)),
        "a7_on_923_fresh_tokens_for_comparison": {"delta": None, "ci95": None, "source": A7_JSON}}
    try:
        a7 = json.load(open(A7_JSON, encoding="utf-8"))["statistic"]
        out["statistic"]["a7_on_923_fresh_tokens_for_comparison"].update(
            delta=a7["delta"], ci95=a7["ci95"], shipped_pdms=a7["shipped_pdms"], repaired_pdms=a7["repaired_pdms"])
    except Exception as e:
        out["statistic"]["a7_on_923_fresh_tokens_for_comparison"]["status"] = f"NOT AVAILABLE: {e}"
    out["estimator_control"] = estimator_control(a.boot)
    out["exploration_reproduction"] = exploration_reproduction(arr["repaired"], rows["repaired"], toks)
    out["code_path_check_reported"] = code_path_check()
    out["csv_to_seam_ties_reported"] = {
        "what": "each compared CSV was scored from exactly this seam file (the harness's call log), unmodified since",
        f"pipeline_{A}": csv_to_seam_tie(f"refe_{A}", SEAM_OF[A]),
        f"pipeline_{B}": csv_to_seam_tie(f"refe_{B}", SEAM_OF[B]),
        **{k_: csv_to_seam_tie(lb, SEAMS[k_]) for k_, lb in LABELS.items()}}
    out["heading_signature_reported"] = {
        "what": "the corrupted native heading[19] lands verbatim at NAVSIM t = 4.0 s (an exact grid point); a seam "
                "executed WITHOUT the repair shows |heading| > pi there on most tokens, one executed WITH it does not",
        "015_shipped_seam": heading_signature(np.load(SEAM_OF[A])["poses"]),
        "015_repaired_seam": heading_signature(arr["repaired"]),
        "016_shipped_seam": heading_signature(np.load(SEAM_OF[B])["poses"])}
    fam = {"what": "NOT recomputed. The four families read positions only (the NAVSIM family adapter drops the heading "
                   "column; its heading_mae_deg is the path tangent), and gate B proves 015-repaired has 015-shipped's "
                   "positions bit-for-bit, so the pipeline's existing per-snapshot family readouts for 015 (computed on "
                   "the unrepaired seam) and 016 (computed on the repaired seam) are already like-for-like. Strategic: "
                   "unavailable in NAVSIM.",
           "pipeline_family_readouts": {p: {"path": FAM_OF[p], "sha256": sha256(FAM_OF[p]) if os.path.exists(FAM_OF[p])
                                            else "MISSING"} for p in (A, B)}}
    try:
        f7 = json.load(open(A7_JSON, encoding="utf-8"))["four_families"]
        fam["a7_evidence_positions_only"] = {
            "headline_numbers_identical": f7.get("headline_numbers_identical"),
            "full_blocks_identical_apart_from_run_labels": f7.get("full_blocks_identical_apart_from_run_labels"),
            "source": A7_JSON + " -> four_families (families6.py run on a shipped and a repaired seam)"}
    except Exception as e:
        fam["a7_evidence_positions_only"] = f"NOT AVAILABLE: {e}"
    out["four_families"] = fam
    out["harness_artifacts"] = copy_harness_artifacts()
    out["provenance"] = {"script": os.path.abspath(__file__), "script_sha256": sha256(os.path.abspath(__file__)),
                         "dump_sha256": sha256(DUMP),
                         "seam_sha256": {**{p: sha256(SEAM_OF[p]) for p in (A, B)},
                                         **{k_: sha256(SEAMS[k_]) for k_ in SEAMS}},
                         "seam_md5": {k_: md5(SEAMS[k_]) for k_ in SEAMS},
                         "csv_sha256": {A: sha256(CSV_OF[A]), B: sha256(CSV_OF[B]),
                                        **{k_: sha256(csv_of(lb)) for k_, lb in LABELS.items()}},
                         "tokens_sha256": sha256(TOK)}
    out["cpu_only"] = no_torch()
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    if not a.no_docs:
        write_result(out, load64())
        write_manifest(out, load64())
    print(f"  015 shipped {s2['a_pdms']:.2f} | 015 repaired {s1['a_pdms']:.2f} | 016 repaired {s1['b_pdms']:.2f}", flush=True)
    for name, s in stat.items():
        print(f"  {name:34s} {s['delta']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]"
              f"{'  separated' if s['separated'] else '  not separated'}", flush=True)
    ec = out["estimator_control"]
    print(f"  estimator control (A7 re-read): pass={ec.get('pass')}  torch imported: {out['cpu_only']['torch_imported']}")
    print(f"ZZLFL_OK {s1['delta']:+.2f} {s1['ci95'][0]:+.2f} {s1['ci95'][1]:+.2f}")
    return 0


# ------------------------------------------------------------------------------------------------ docs
def load64():
    """the 64-proposal extension's record, only when its gates passed and the trend was read"""
    if not os.path.exists(OUT64_JSON):
        return None
    o64 = json.load(open(OUT64_JSON, encoding="utf-8"))
    ok = o64.get("gates", {}).get("all_gates_pass") and isinstance(o64.get("trend"), dict) and o64["trend"].get("result")
    return o64 if ok else None


def section64(o64):
    t, g, e16 = o64["trend"]["result"], o64["gates"], o64["ep016_table"]
    ga, gb, tie = (g["a64_unrepaired_rebuild_equals_pipeline_E6_poses"], g["b64_only_the_4s_heading_differs"],
                   g["tie_to_pick_readout"])
    g3, c64 = g["G3_every_run_PASS_every_token_valid"], g["c64_pick_column_reproduces_pick_readout"]
    ade = o64["oracle_ade_position_only"]["values"]
    names = {"015:016": "015 unrepaired -> 016 (as the two E-6 tables stand)",
             "015_repaired:016": "**015 repaired -> 016 (LIKE-FOR-LIKE)**",
             "015:015_repaired": "015 unrepaired -> 015 repaired (the repair on all 64)"}

    def cell(v):
        return (f"{v['a']:.2f} -> {v['b']:.2f}: **{v['b_minus_a']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]**"
                f"{' separated' if v['separated'] else ''}")
    lk = t["015_repaired:016"]
    lines = [
        "## Extension -- all 64 proposals: ep015 repaired vs ep016 (best-of-64, mean-of-64, share >= 80)", "",
        "*The PI's question: did the planner's BEST proposal degenerate or stabilise? All 64 of ep015's proposals are "
        "re-read with the repair (built from the same native dump, each of the 64 scored by `score_navtest_refe.py` "
        "UNCHANGED, 2 workers), so the comparison with ep016 -- whose 64 the pipeline already executed with the repair "
        "-- is like-for-like.*", "",
        "| pair (same 200 tokens) | best of 64 | mean of 64 | share of the 64 with PDMS >= 80 (%) |",
        "|---|---|---|---|"]
    for pr in TREND_PAIRS:
        r = t[pr]
        lines.append(f"| {names[pr]} | {cell(r['best_of_64'])} | {cell(r['mean_of_64'])} | {cell(r['share_ge_80'])} |")
    lines += [
        "", f"Like-for-like, best-of-64 moved {lk['best_of_64']['b_minus_a']:+.2f} [{lk['best_of_64']['ci95'][0]:+.2f}, "
        f"{lk['best_of_64']['ci95'][1]:+.2f}] ({'separated' if lk['best_of_64']['separated'] else 'not separated'}), "
        f"mean-of-64 {lk['mean_of_64']['b_minus_a']:+.2f} [{lk['mean_of_64']['ci95'][0]:+.2f}, "
        f"{lk['mean_of_64']['ci95'][1]:+.2f}] ({'separated' if lk['mean_of_64']['separated'] else 'not separated'}), "
        f"share >= 80 {lk['share_ge_80']['b_minus_a']:+.2f} [{lk['share_ge_80']['ci95'][0]:+.2f}, "
        f"{lk['share_ge_80']['ci95'][1]:+.2f}] points ({'separated' if lk['share_ge_80']['separated'] else 'not separated'}). "
        f"ep015 unrepaired, from its existing E-6 table: best-of-64 {t['015:016']['best_of_64']['a']:.2f}, mean-of-64 "
        f"{t['015:016']['mean_of_64']['a']:.2f}.", ""]
    sep = [k_ for k_ in ("best_of_64", "mean_of_64", "share_ge_80") if lk[k_]["separated"]]
    if sep:
        lines += [
            "**How far a separated like-for-like interval here reaches:** "
            + "; ".join(f"{k_} {lk[k_]['b_minus_a']:+.2f} [{lk[k_]['ci95'][0]:+.2f}, {lk[k_]['ci95'][1]:+.2f}] -- the "
                        f"bound nearest zero is {min(abs(lk[k_]['ci95'][0]), abs(lk[k_]['ci95'][1])):.2f} points from it"
                        for k_ in sep)
            + ". The interval answers another draw of EPISODES only; one training run and two consecutive snapshots "
            "cannot say whether another run of epoch 16 would show the same, so it is necessary, not sufficient, for "
            "calling this an effect of training (H-ESTIM-SEED-1: a same-flags replicate cleared 'separated' on 9.5 % of "
            "cells on the tiny rig).", ""]
    lines += [
        f"Estimator: `eval/oracle_trend.py` UNCHANGED (sha256 `{o64['trend']['script_sha256'][:12]}`), "
        f"`--pairs {' '.join(TREND_PAIRS)} --boot {o64['trend']['boot']}`, reading this script's table "
        f"`proptable/{NAME64}/table.npz` (proposal_table.py's E-6 format: ep015's logits, pick and rule, the repaired "
        "poses and their harness scores) beside the pipeline's ep015 and ep016 tables. "
        f"{o64['trend']['estimator']}. It answers {o64['trend']['variance_answered']}.", "",
        f"Gates: (a) {ga['proposals_bit_identical']} unrepaired rebuilds bit-identical to the pipeline's E-6 poses "
        f"(which equal its proposals.npz: {ga['table_poses_equal_pipeline_proposals_npz']}) -- "
        f"{'PASS' if ga['pass'] else 'FAIL'}; (b) x, y bit-identical {gb['x_y_bit_identical_all_8_poses']}, headings "
        f"0.5-3.5 s identical {gb['heading_bit_identical_0.5_to_3.5s']}, 4.0 s heading == native heading[18] "
        f"{gb['heading_4.0s_bit_identical_to_native_heading18']} (changed on {gb['proposals_whose_4.0s_heading_changed']}) "
        f"-- {'PASS' if gb['pass'] else 'FAIL'}; the pick column == the pick readout's repaired seam "
        f"{tie['pick_column_equals_repaired_pick_seam']}, table pick == recovered pick {tie['table_pick_equals_recovered_pick']} "
        f"-- {'PASS' if tie['pass'] else 'FAIL'}; {g3['runs_pass']} harness runs PASS with every token valid, each tied "
        f"to its seam file -- {'PASS' if g3['pass'] else 'FAIL'}; the table's pick column reproduces the pick readout's "
        f"repaired scores (max |diff| PDMS {c64['max_abs_pdms_diff']}, sub-scores {c64['max_abs_subscore_diff']}; PDMS "
        f"{c64['pick_column_pdms_x100']}) -- {'PASS' if c64['pass'] else 'FAIL'}. The reader oracle_trend.py uses "
        "(`snapshot_pair_under_rule.load`, which asserts each table's own rule reproduces its pick) accepts all three: "
        + ", ".join(f"{k_} {v['loads']}" for k_, v in o64["readers_snapshot_pair_under_rule_load"].items()) + ".", "",
        f"The ep016 table: its own gates {e16['own_gates']}; its dump records repair_last_heading = "
        f"{e16['dump_repair_last_heading']}; its pick column == the ep016 seam {e16['pick_column_equals_ep016_seam']}. "
        f"Share of the 64 proposals with |heading(4.0 s)| > pi: ep016 {100 * e16['share_abs_heading_4.0s_above_pi_all_64']:.1f} %, "
        f"ep015 unrepaired {100 * e16['ep015_unrepaired_table_same_share']:.1f} %, ep015 repaired "
        f"{100 * e16['ep015_repaired_table_same_share']:.1f} % -- ep016's 64 are the repaired ones.", "",
        f"**Position-only oracle ADE needs no re-read:** {ade[A]['ade_oracle_m']} m at 015 -> {ade[B]['ade_oracle_m']} m "
        f"at 016 (the seam reports' `proposals.ade_oracle_m`, best of {ade[A]['M']} vs the logged human future). It is "
        "computed from x, y only, and gate (b) proves the repair moves no position, so 015's value is 015-repaired's.",
        ""]
    return lines


def write_result(o, o64=None):
    s = o["statistic"]
    c = s["comparisons"]
    s1, s2, s3 = (c["016_repaired_minus_015_repaired"], c["015_repaired_minus_015_shipped"],
                  c["016_repaired_minus_015_shipped"])
    g = o["gates"]
    ga, gr, gb = g["A_unique_exact_pick_recovery"], g["R_same_selection_rule"], g["B_only_the_4s_heading_differs"]
    gc, gd = g["C_rebuilt_unrepaired_reproduces_pipeline_ep015"], g["D_every_harness_run_PASS_every_token_valid"]
    r = o["rules_each_point_was_picked_with"]
    ec, ex, a7 = o["estimator_control"], o["exploration_reproduction"], s["a7_on_923_fresh_tokens_for_comparison"]
    cc = o["cross_checks_reported"]
    hs = o["heading_signature_reported"]
    fam = o["four_families"]

    def row(label, st):
        return (f"| {label} | {st['a_pdms']:.2f} -> {st['b_pdms']:.2f} | **{st['delta']:+.2f} [{st['ci95'][0]:+.2f}, "
                f"{st['ci95'][1]:+.2f}]** | {'yes' if st['separated'] else 'no'} | {st['better']} / {st['worse']} / "
                f"{st['tied']} |")
    lines = [
        "# RESULT -- like-for-like PDMS readout for REFe snapshot 016 (015 re-read with Amendment 7's repair)", "",
        "*Every number is read from the unchanged harness's per-token scores; none is typed. Written by "
        "`eval/like_for_like_016.py` (CPU only; torch never imported). Full record: "
        "`eval/raw/like_for_like_016/like_for_like_016.json`.*", "",
        f"**Tokens:** W3's 200-token paired subset `A1_sub200_tokens.json` (md5 `{o['tokens']['md5']}`), "
        f"{s['n_logs']} logs -- the learning curve's own tokens, NOT the published split. **Tier:** NAVSIM v1 PDMS "
        "(ego pseudo-simulation of an open-loop plan against logged agents). **Repair:** on the planner's native "
        "[20, 3] output, heading[19] := heading[18] for the executed plan (`refe/planner.py repair_last_heading`); it "
        "acts after selection, so each token's pick is unchanged.", "",
        "## Headline", "",
        "| comparison (same 200 tokens) | PDMS | difference [95 %] | separated | better / worse / tied |",
        "|---|---|---|---|---|",
        row("**015 repaired -> 016 repaired (LIKE-FOR-LIKE: the epoch of training)**", s1),
        row("015 shipped -> 015 repaired (the repair alone, this subset)", s2),
        row("015 shipped -> 016 repaired (the curve's published step: repair + training)", s3), "",
        f"Differences are means of per-token differences at full precision (PDMS {s['pdms_x100']['015_shipped']}, "
        f"{s['pdms_x100']['015_repaired']}, {s['pdms_x100']['016_repaired']} are rounded), so a difference can sit "
        f"0.01 from the difference of the rounded PDMS: the like-for-like is {s1['delta_unrounded']:+.4f}.", "",
        "Where the like-for-like difference comes from (descriptive; NAVSIM v1 PDMS = NC x DAC x (5 EP + 5 TTC + 2 C) "
        "/ 12, DDC weight 0): " + "; ".join(
            f"{k_}: {v['n']} tokens ({v['better']} better / {v['worse']} worse), {v['contribution_to_mean_difference_x100']:+.2f}"
            for k_, v in s1["where_the_difference_comes_from_reported"].items())
        + ". The contributions sum to the mean difference.", "",
        f"The published {s3['delta']:+.2f} step decomposes exactly (same tokens, same draws) into the repair's "
        f"{s2['delta']:+.2f} and the like-for-like {s1['delta']:+.2f}. For comparison, "
        + (f"Amendment 7's confirmation read the repair at {a7['delta']:+.2f} [{a7['ci95'][0]:+.2f}, "
           f"{a7['ci95'][1]:+.2f}] on 923 fresh tokens (43 logs, none in these 200) at the same snapshot"
           if a7.get("delta") is not None else f"Amendment 7's confirmation value is {a7.get('status')}")
        + f"; here, on the curve's 200 tokens, the repair reads {s2['delta']:+.2f} [{s2['ci95'][0]:+.2f}, "
        f"{s2['ci95'][1]:+.2f}].", "",
        f"**What the interval answers:** {s['variance_answered']}. Estimator: {s['estimator']}. No decision rides on it.",
        "",
        "## The rule each point was picked with", "",
        "| point | selection rule (seam report / proposal dump) | select | last-heading repair (report / dump) |",
        "|---|---|---|---|"]
    for p in (A, B):
        v = r[p]
        lines.append(f"| {p} | {v['report_rule']} / {v['dump_rule']} | {v['report_select']} / {v['dump_select']} | "
                     f"{v['report_repair_last_heading']} / {v['dump_repair_last_heading']} |")
    lines += [
        "", f"One selection rule for both points (gate R: {'PASS' if gr['pass'] else 'FAIL'}), so the comparison is "
        "like-for-like in selection; the repair is the only executed-plan difference, and this readout removes it. "
        "015's shipped seam carries no repair flag because it predates Amendment 7; gate A proves it executed the "
        "UNREPAIRED plan (the unrepaired conversion matches it bit-for-bit on every token, the repaired one on "
        f"{cc['repaired_conversion_also_matches_shipped_seam_on_tokens']}).", ""]
    cp = o.get("code_path_check_reported", {}).get("files", {})
    if cp:
        lines += [
            "Apart from the repair, both points ran one inference path (reported; net diff of today's files against "
            "their last committed versions on the push mirror): " + "; ".join(
                f"`{rel}` vs {v.get('committed_rev')}: " + ("identical" if v.get("identical") else
                                                           (f"+{v['lines_added']}/-{v['lines_removed']} lines"
                                                            if "lines_added" in v else v.get("status", "?")))
                for rel, v in cp.items())
            + ". The changed lines (listed in the JSON) are Amendment 5's rule plumbing -- both points were picked with "
            "navsim_v1 -- and Amendment 7's repair with its metadata. A net diff cannot exclude a change made and "
            "reverted in between; a GPU re-forward of 015 under today's code with the repair off would be the direct "
            "test.", ""]
    lines += [
        "## Validity gates (all must pass before the statistic is read)", "",
        "| gate | result |", "|---|---|",
        f"| A: each token's shipped pick recovered from the ep015 native dump by a UNIQUE exact match, "
        f"to_navsim(traj[token, k]) == shipped seam bit-for-bit | {'PASS' if ga['pass'] else 'FAIL'}: "
        f"{ga['tokens_with_unique_exact_match']} unique; 0-match {ga['tokens_with_0_matches']}, >1-match "
        f"{ga['tokens_with_more_than_1_match']} |",
        f"| R: the same selection rule for both points | {'PASS' if gr['pass'] else 'FAIL'}: "
        f"{gr['distinct_rule_select_pairs']} |",
        f"| B: repaired x, y bit-identical on all 8 poses; headings 0.5-3.5 s identical; the 4.0 s heading == native "
        f"heading[18] | {'PASS' if gb['pass'] else 'FAIL'}: x/y {gb['x_y_bit_identical_all_8_poses']}, h 0.5-3.5 s "
        f"{gb['heading_bit_identical_0.5_to_3.5s']}, h 4.0 s {gb['heading_4.0s_bit_identical_to_native_heading18']}; "
        f"changed on {gb['tokens_whose_4.0s_heading_changed']}/{gb['n']} tokens |",
        f"| C: the rebuilt UNREPAIRED seam, scored today, reproduces the pipeline's ep015 scores token-for-token | "
        f"{'PASS' if gc['pass'] else 'FAIL'}: PDMS {gc['pdms_rebuilt_x100']} vs pipeline {gc['pdms_pipeline_x100']}; "
        f"max per-token |diff| PDMS {gc['max_abs_pdms_diff']}, sub-scores {gc['max_abs_subscore_diff']} |",
        f"| D: every harness run PASS, every token valid | {'PASS' if gd['pass'] else 'FAIL'}: "
        + "; ".join(f"{v['label']} {v['status']} {v['tokens_valid']}" for v in gd["new_runs"].values()) + "; "
        + "; ".join(f"pipeline {v['label']} {v['status']} {v['tokens_valid']}" for v in gd["pipeline_runs"].values())
        + " |", "",
        f"Cross-checks (reported): recovered pick == the pipeline's own recorded pick "
        f"{cc['recovered_k_equals_pipeline_proposal_dump_pick']}; == the dump's pick "
        f"{cc['recovered_k_equals_dump_pick_infer']}; the dump's second forward repeated its poses to "
        f"{cc['dump_second_forward_max_abs_pose_diff']} (a deterministic forward). The repair ran as planner.py's own "
        f"function (compiled from its source, sha256 `{o['repair_function']['planner_py_sha256'][:12]}`), equal "
        f"bit-for-bit to an independent literal: {o['repair_function']['planner_fn_equals_independent_literal_bitwise']}.",
        "",
        "CSV-to-seam ties (the harness's own call log names the seam file it served; the file must be unmodified "
        "since): " + "; ".join(f"{v.get('arm_served', k_)} {'PASS' if v['pass'] else 'FAIL'} "
                               f"({v.get('seam_calls')} seam calls, scored {v.get('scored_local')})"
                               for k_, v in o["csv_to_seam_ties_reported"].items() if isinstance(v, dict)) + ".", "",
        f"Estimator control: the same bootstrap on Amendment 7's own CSVs re-reads its banked "
        f"{ec['banked']['delta']:+.2f} [{ec['banked']['ci95'][0]:+.2f}, {ec['banked']['ci95'][1]:+.2f}] as "
        f"{ec['reread']['delta']:+.2f} [{ec['reread']['ci95'][0]:+.2f}, {ec['reread']['ci95'][1]:+.2f}] "
        f"({'exact' if ec['pass'] else 'MISMATCH'}).", "",
        "## Reported, not gating", "",
        "NAVSIM sub-scores x100: " + "; ".join(
            f"{k_} " + ", ".join(f"{h} {v:.2f}" for h, v in s["subscores_x100"][k_].items()) for k_ in s["subscores_x100"])
        + ".", "",
        "Sub-score deltas x100, like-for-like (016 repaired - 015 repaired): "
        + ", ".join(f"{h} {v:+.2f}" for h, v in s1["subscore_deltas_x100"].items()) + "; the repair alone: "
        + ", ".join(f"{h} {v:+.2f}" for h, v in s2["subscore_deltas_x100"].items()) + ".", "",
        f"Heading at t = 4.0 s, share |heading| > pi: 015 shipped "
        f"{100 * hs['015_shipped_seam']['share_abs_heading_4.0s_above_pi']:.1f} %, 015 repaired "
        f"{100 * hs['015_repaired_seam']['share_abs_heading_4.0s_above_pi']:.1f} %, 016 shipped "
        f"{100 * hs['016_shipped_seam']['share_abs_heading_4.0s_above_pi']:.1f} % -- consistent with 016's report "
        "that it executed the repaired plan.", ""]
    if "banked" in ex:
        lines += [f"The exploration that motivated Amendment 7 (same 200 tokens, `slow_copies.json` 'hold') banked "
                  f"{ex['banked']['pdms']} PDMS, {ex['banked']['minus_shipped']:+.2f} {ex['banked']['ci95_vs_shipped']}; "
                  f"its repaired seam is bit-identical to this one: {ex['poses_bit_identical_to_this_repaired_seam']}; "
                  f"per-token max |PDMS diff| vs this run {ex['per_token_max_abs_pdms_diff_vs_this_run']}.", ""]
    else:
        lines += [f"The motivating exploration's reproduction: {ex.get('status')}.", ""]
    fa = fam.get("a7_evidence_positions_only")
    lines += [
        "**The four metric families are NOT recomputed, and need not be.** They read positions only (the NAVSIM family "
        "adapter drops the heading column; its heading_mae_deg is the path tangent) -- Amendment 7 ran `families6.py` "
        "on a shipped and a repaired seam and found "
        + (f"headline numbers identical = {fa['headline_numbers_identical']}, full blocks identical apart from run "
           f"labels = {fa['full_blocks_identical_apart_from_run_labels']}" if isinstance(fa, dict) else str(fa))
        + " -- and gate B proves 015-repaired has 015-shipped's positions bit-for-bit. So the pipeline's existing "
        f"per-snapshot family readouts (`{FAM_OF[A]}`, on the unrepaired seam; `{FAM_OF[B]}`, on the repaired seam) "
        "are already like-for-like. Strategic: unavailable in NAVSIM.", ""]
    if o64:
        lines += section64(o64)
    lines += [
        "## What this does not answer", "",
        "One training run, two consecutive snapshots, one deterministic forward each: the interval is over episodes "
        "only. Whether another run of epoch 16 would land elsewhere (training variance) is not measured -- there is no "
        "replicate. The 200 tokens are the learning curve's paired subset, not the published navtest split.", "",
        "Artifacts and checksums: `eval/raw/like_for_like_016/MANIFEST.md`."]
    open(RESULT_MD, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


def write_manifest(o, o64=None):
    p = o["provenance"]
    lines = ["# like-for-like 016 -- inputs and outputs", "",
             "| artifact | where | sha256 |", "|---|---|---|",
             f"| like_for_like_016.py (this run's script) | repo: eval/like_for_like_016.py | `{p['script_sha256']}` |",
             "| like_for_like_016.json, build_record.json, selftest.json | repo: eval/raw/like_for_like_016/ | (written "
             "with this manifest) |",
             "| RESULT_like_for_like_016.md | repo: eval/ | (written with it) |",
             f"| repaired ep015 seam (md5 {p['seam_md5']['repaired']}) | data dir: {SEAMS['repaired']}; repo copy: "
             f"eval/raw/like_for_like_016/{os.path.basename(REPAIRED_COPY)} | `{p['seam_sha256']['repaired']}` |",
             f"| rebuilt unrepaired ep015 seam (md5 {p['seam_md5']['unrepaired']}) | data dir only: "
             f"{SEAMS['unrepaired']} (poses bit-identical to the shipped ep015 seam) | `{p['seam_sha256']['unrepaired']}` |",
             f"| shipped ep015 seam (the pipeline's) | data dir only: {SEAM_OF[A]} | `{p['seam_sha256'][A]}` |",
             f"| shipped ep016 seam (the pipeline's) | data dir only: {SEAM_OF[B]} | `{p['seam_sha256'][B]}` |",
             f"| ep015 native dump (stop_candidate_probe.py) | data dir only: {DUMP} | `{p['dump_sha256']}` |",
             f"| A1_sub200_tokens.json (md5 {TOK_MD5}) | repo: {TOK} | `{p['tokens_sha256']}` |"]
    for k_, h in p["csv_sha256"].items():
        lb = LABELS.get(k_)
        where = (f"data dir: {csv_of(lb)}; repo copy: eval/raw/like_for_like_016/harness/{lb}/" if lb
                 else f"data dir only: {CSV_OF[k_]}")
        lines.append(f"| harness csv, {k_} | {where} | `{h}` |")
    for lb, v in o["harness_artifacts"].items():
        for fn, f in v["files"].items():
            if "repo_copy" not in f:
                lines.append(f"| harness {fn} | data dir only: {v['harness_dir']}/{fn} | `{f['sha256']}` |")
    lines += ["", "Harness stdout logs: eval/raw/like_for_like_016/logs/. W3's devkit run directories: "
                  "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/runs/{" + ", ".join(LABELS.values()) + "} (Archive only).",
              f"Checkpoints (read for md5 only): {CKPT_OF[A]} md5 {o['points'][A]['ckpt_md5']}; {CKPT_OF[B]} md5 "
              f"{o['points'][B]['ckpt_md5']} (data dir only)."]
    if o64:
        lines += ["", "## Extension: all 64 proposals", "",
                  "| artifact | where | sha256 |", "|---|---|---|",
                  "| like_for_like_016_64.json, build64_record.json, oracle_trend_ep015rep_ep016.json, "
                  "logs/oracle_trend.log | repo: eval/raw/like_for_like_016/ | (written with this manifest) |",
                  f"| compact scores of the repaired table (token, pdms, sub, valid, pick) | repo: "
                  f"eval/raw/like_for_like_016/{os.path.basename(SCORES64_COPY)} | `{o64['scores_copy_sha256']}` |",
                  f"| the repaired E-6 table oracle_trend.py reads (+ gates.json) | data dir only: {TABLE64} | "
                  f"`{o64['table_sha256']}` |",
                  f"| the pipeline's ep015 / ep016 E-6 tables | data dir only: {TABLE_OF[A]} / {TABLE_OF[B]} | "
                  f"`{o64['gates']['a64_unrepaired_rebuild_equals_pipeline_E6_poses']['table_sha256']}` / "
                  f"`{o64['ep016_table']['table_sha256']}` |",
                  f"| 64 repaired proposal seams (md5 each in like_for_like_016_64.json) | data dir only: {SEAM64_DIR}/ | "
                  "-- |",
                  f"| 64 harness outputs (csv sha256 each in like_for_like_016_64.json) | data dir only: "
                  f"{EC.DATA}/score/{label64(0)} .. {label64(M64 - 1)}/; stdout logs {WD64}/logs/ | -- |"]
    open(MANIFEST, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------------------------ selftest
def selftest(a) -> int:
    """deliberate-regression arms: each mutation must turn its gate RED; the unmutated inputs must stay GREEN"""
    D, SA, SB, toks, struct = load_inputs()
    traj, shipped = np.array(D["traj"]), np.array(SA["poses"])
    n_small = 12                                                   # gate A's search is the slow part; a slice suffices
    cases = {}

    def rec(name, gate_pass, want_pass):
        cases[name] = {"gate_pass": bool(gate_pass), "expected_pass": want_pass, "ok": bool(gate_pass) == want_pass}
    m0 = recover_matches(traj[:n_small], shipped[:n_small])
    rec("A_none", gate_a(m0, toks[:n_small])["pass"], True)
    s1 = shipped[:n_small].copy()
    s1[3, 5, 0] = np.nextafter(s1[3, 5, 0], np.float32(np.inf))    # ONE float32 ulp on one pose of one token
    rec("A_seam_one_ulp", gate_a(recover_matches(traj[:n_small], s1), toks[:n_small])["pass"], False)
    t2 = traj[:n_small].copy()
    k3 = m0[4][0]
    t2[4, (k3 + 1) % t2.shape[1]] = t2[4, k3]                        # a duplicate proposal -> an ambiguous pick
    rec("A_duplicate_proposal", gate_a(recover_matches(t2, shipped[:n_small]), toks[:n_small])["pass"], False)
    s3 = shipped[:n_small].copy()
    s3[0, :, :] = nav(literal_repair(traj[:1, m0[0][0]]))[0]      # a REPAIRED row where an unrepaired one was shipped
    rec("A_shipped_row_is_repaired", gate_a(recover_matches(traj[:n_small], s3), toks[:n_small])["pass"], False)
    # gate B on all tokens (cheap)
    mall = recover_matches(traj, shipped)
    k = np.array([m[0] for m in mall])
    npk = traj[np.arange(len(toks)), k]
    fn, _ = planner_repair()
    good = nav(fn(npk))
    rec("B_none", gate_b(good, shipped, npk)["pass"], True)
    rec("B_identity_repair", gate_b(nav(npk), shipped, npk)["pass"], False)
    wrong = npk.copy()
    wrong[:, 18, 2] = wrong[:, 17, 2]                                 # repaired at the wrong index
    rec("B_wrong_index", gate_b(nav(wrong), shipped, npk)["pass"], False)
    xmut = fn(npk)
    xmut[5, 19, 0] += np.float32(1e-3)                                # the repair also moved a position by 1 mm
    rec("B_x_moved_1mm", gate_b(nav(xmut), shipped, npk)["pass"], False)
    hmut = fn(npk)
    hmut[7, 16, 2] += np.float32(1e-3)                                # an earlier heading (feeds t = 3.5 s) moved
    rec("B_heading_3.5s_moved", gate_b(nav(hmut), shipped, npk)["pass"], False)
    # the extension's gates (all 64 proposals) on a 6-token slice
    T6 = np.load(TABLE_OF[A])
    tp6 = T6["proposals"][table_rows(T6, toks)[:6]]
    tr6 = traj[:6]
    un6 = np.stack([nav(tr6[i]) for i in range(6)])
    rec("A64_none", gate_a64(un6, tp6, toks[:6])["pass"], True)
    tpm = tp6.copy()
    tpm[2, 17, 4, 1] = np.nextafter(tpm[2, 17, 4, 1], np.float32(np.inf))   # ONE ulp, one pose of one proposal
    rec("A64_one_ulp_one_proposal", gate_a64(un6, tpm, toks[:6])["pass"], False)
    rec("A64_only_63_proposals", gate_a64(un6[:, :63], tp6[:, :63], toks[:6])["pass"], False)
    rp6 = np.stack([nav(x) for x in fn(tr6)])
    rec("B64_none", gate_b64(rp6, tp6, tr6)["pass"], True)
    rec("B64_identity_repair", gate_b64(un6, tp6, tr6)["pass"], False)
    ym = fn(tr6)
    ym[1, 40, 19, 1] += np.float32(1e-3)                                   # one proposal's last y moved 1 mm
    rec("B64_y_moved_1mm_one_proposal", gate_b64(np.stack([nav(x) for x in ym]), tp6, tr6)["pass"], False)
    # gate C on the pipeline's own csv against itself and mutated copies
    pr = read_csv(CSV_OF[A])
    okst = {"status": "PASS"}
    rec("C_none", gate_c(pr, pr, okst, toks)["pass"], True)
    c1 = dict(pr)
    t0 = toks[0]
    c1[t0] = (c1[t0][0], c1[t0][1] + 1e-9, c1[t0][2])
    rec("C_one_score_1e-9", gate_c(c1, pr, okst, toks)["pass"], False)
    c2 = dict(pr)
    c2.pop(toks[1])
    rec("C_token_missing", gate_c(c2, pr, okst, toks)["pass"], False)
    c3 = dict(pr)
    c3[t0] = (False, c3[t0][1], c3[t0][2])
    rec("C_token_invalid", gate_c(c3, pr, okst, toks)["pass"], False)
    rec("C_harness_FAIL", gate_c(pr, pr, {"status": "FAIL"}, toks)["pass"], False)
    # gate R
    same = {p: {"report_rule": "navsim_v1", "report_select": "best", "dump_rule": "navsim_v1", "dump_select": "best"}
            for p in (A, B)}
    rec("R_none", gate_r(same)["pass"], True)
    diff = {A: dict(same[A], report_rule="v2_shape", dump_rule="v2_shape"), B: same[B]}
    rec("R_different_rules", gate_r(diff)["pass"], False)
    rec("R_real_files", gate_r(rules_record())["pass"], True)
    # the estimator: a constant difference must give exactly [c, c]; a zero difference exactly [0, 0]
    draws, _ = cluster_draws(toks, token_log(), 2000)
    pc = paired(np.full(len(toks), 0.0137), draws)
    cases["E_constant_difference"] = {"ci95": pc["ci95"], "delta": pc["delta"],
                                      "ok": pc["ci95"] == [1.37, 1.37] and pc["delta"] == 1.37}
    p0 = paired(np.zeros(len(toks)), draws)
    cases["E_zero_difference"] = {"ci95": p0["ci95"], "ok": p0["ci95"] == [0.0, 0.0] and not p0["separated"]}
    allok = all(v["ok"] for v in cases.values())
    os.makedirs(RAW, exist_ok=True)
    json.dump({"_label": "like_for_like_016.py deliberate-regression arms (in-memory mutations; no file is written but "
                         "this record)", "run_local": now(), "cases": cases, "all_ok": allok, "cpu_only": no_torch()},
              open(SELFTEST_JSON, "w", encoding="utf-8"), indent=1)
    for kk, v in cases.items():
        print(f"  [{'ok ' if v['ok'] else 'BAD'}] {kk}: {v}", flush=True)
    print("ZZLFL_SELFTEST_" + ("OK" if allok else "FAIL"))
    return 0 if allok else 1


# ------------------------------------------------------------------------------------------------ EXTENSION: all 64
def table_rows(T, toks):
    """an E-6 table re-indexed by token into `toks` order"""
    ti = {str(t): i for i, t in enumerate(T["token"])}
    if set(ti) != set(toks):
        raise RuntimeError("the table's tokens differ from the seam's")
    return np.array([ti[t] for t in toks], dtype=np.int64)


def gate_a64(unrep, tprop, toks, M_expected=M64):
    """unrep, tprop [n, M, 8, 3] float32: the unrepaired rebuild vs the pipeline's E-6 poses, per proposal, bitwise"""
    n, M = unrep.shape[:2]
    eq = np.array([[bits_equal(unrep[i, j], tprop[i, j]) for j in range(M)] for i in range(n)])
    return {"what": "the unrepaired rebuild of EVERY proposal, to_navsim(traj[token, j]) as float32, equals the "
                    "pipeline's own ep015 E-6 poses (proptable/sub200_ep015/table.npz `proposals` [200, 64, 8, 3], the "
                    "table proposal_table.py built from the landed eval's proposals.npz) bit-for-bit",
            "proposals_bit_identical": f"{int(eq.sum())}/{eq.size}",
            "tokens_with_any_mismatch": [toks[i] for i in range(n) if not eq[i].all()][:20],
            "M": int(M), "pass": bool(eq.all() and M == M_expected)}


def gate_b64(rep, tprop, traj):
    """rep, tprop [n, M, 8, 3] float32; traj [n, M, 20, 3] float32 native (UNREPAIRED)"""
    xy = bits_equal(rep[..., :2], tprop[..., :2])
    early = bits_equal(rep[..., :7, 2], tprop[..., :7, 2])
    h4 = bits_equal(rep[..., 7, 2], traj[..., 18, 2])
    changed = rep[..., 7, 2] != tprop[..., 7, 2]
    return {"what": "every repaired proposal vs the pipeline's E-6 pose: x, y bit-identical on all 8 poses; headings "
                    "0.5-3.5 s bit-identical; the 4.0 s heading bit-identical to the native heading[18] -- only it "
                    "differs",
            "x_y_bit_identical_all_8_poses": xy, "heading_bit_identical_0.5_to_3.5s": early,
            "heading_4.0s_bit_identical_to_native_heading18": h4,
            "proposals_whose_4.0s_heading_changed": f"{int(changed.sum())}/{changed.size}",
            "pass": bool(xy and early and h4)}


def build64_all():
    """pure: gates (a) and (b) over ALL 64 proposals, and the repaired [n, 64, 8, 3] poses"""
    D, SA, SB, toks, struct = load_inputs()
    traj = D["traj"]                                                      # [n, 64, 20, 3] float32, seam order
    T = np.load(TABLE_OF[A])
    rt = table_rows(T, toks)
    tprop = T["proposals"][rt]                                            # the pipeline's ep015 E-6 poses (unrepaired)
    Pd = np.load(PROP_OF[A], allow_pickle=False)
    rp = table_rows(Pd, toks)
    n = traj.shape[0]
    unrep = np.stack([nav(traj[i]) for i in range(n)])                   # [n, M, 8, 3]
    ga = gate_a64(unrep, tprop, toks)
    ga["table"], ga["table_sha256"] = TABLE_OF[A], sha256(TABLE_OF[A])
    ga["table_poses_equal_pipeline_proposals_npz"] = bits_equal(tprop, Pd["proposals"][rp])
    ga["pass"] = bool(ga["pass"] and ga["table_poses_equal_pipeline_proposals_npz"])
    fn, prov = planner_repair()
    rep_nat = fn(traj)
    rep = np.stack([nav(rep_nat[i]) for i in range(n)])
    gb = gate_b64(rep, tprop, traj)
    gb["planner_fn_equals_independent_literal_bitwise"] = bits_equal(rep_nat, literal_repair(traj))
    gb["pass"] = bool(gb["pass"] and gb["planner_fn_equals_independent_literal_bitwise"])
    # the tie to the pick readout: the pick column of this build IS the repaired pick seam, bit-for-bit
    pick = T["pick"][rt]
    Sr = np.load(SEAMS["repaired"]) if os.path.exists(SEAMS["repaired"]) else None
    tie = {"what": "this build's column at the pipeline's pick equals the pick readout's repaired seam bit-for-bit, and "
                   "the table's pick equals the pick recovered by exact match",
           "pick_column_equals_repaired_pick_seam": bool(Sr is not None and [str(t) for t in Sr["token"]] == toks
                                                         and bits_equal(rep[np.arange(n), pick], Sr["poses"])),
           "table_pick_equals_recovered_pick": None}
    if os.path.exists(BUILD_JSON):
        rk = json.load(open(BUILD_JSON, encoding="utf-8")).get("recovered_pick_k", {})
        tie["table_pick_equals_recovered_pick"] = f"{sum(int(rk.get(t, -1)) == int(p) for t, p in zip(toks, pick))}/{n}"
    tie["pass"] = bool(tie["pick_column_equals_repaired_pick_seam"] and tie["table_pick_equals_recovered_pick"] == f"{n}/{n}")
    return {"structure": struct, "gate_a64": ga, "gate_b64": gb, "tie_to_pick_readout": tie,
            "repair_function": prov}, {"rep": rep, "unrep": unrep, "toks": toks, "SA": SA, "T": T, "rt": rt,
                                       "pick": pick, "traj": traj}


def pick_gates_passed():
    if not os.path.exists(OUT_JSON):
        return False
    g = json.load(open(OUT_JSON, encoding="utf-8")).get("gates", {})
    return bool(g.get("all_gates_pass"))


def build64(a) -> int:
    if not pick_gates_passed():
        print("ZZLFL_FAIL build64: the pick readout's gates have not passed (run `all` first)"); return 1
    res, arr = build64_all()
    ok = bool(res["structure"]["pass"] and res["gate_a64"]["pass"] and res["gate_b64"]["pass"]
              and res["tie_to_pick_readout"]["pass"])
    res["_label"] = "like-for-like 016 EXTENSION: all 64 ep015 proposals repaired -- build record (gates a64, b64, tie)"
    res["built_local"] = now()
    if ok:
        os.makedirs(SEAM64_DIR, exist_ok=True)
        SA = arr["SA"]
        for k in range(M64):
            np.savez(seam64(k), token=SA["token"], fingerprint=SA["fingerprint"], poses=arr["rep"][:, k],
                     sampling=SA["sampling"], arm=np.array(f"REFe_{NAME64}_p{k:02d}"))
        res["seams"] = {label64(k): {"path": seam64(k), "md5": md5(seam64(k))} for k in range(M64)}
    res["cpu_only"] = no_torch()
    json.dump(res, open(BUILD64_JSON, "w", encoding="utf-8"), indent=1)
    for key in ("gate_a64", "gate_b64", "tie_to_pick_readout"):
        g = res[key]
        print(f"  {key}: pass={g['pass']}  " + "; ".join(f"{k_}={v}" for k_, v in g.items()
                                                         if k_ not in ("what", "pass", "table", "table_sha256")),
              flush=True)
    if not ok:
        print("ZZLFL_FAIL build64 -- STOP"); return 1
    print(f"  wrote {M64} repaired proposal seams under {SEAM64_DIR}; torch imported: {res['cpu_only']['torch_imported']}")
    print("ZZLFL_BUILD64_OK")
    return 0


def score64(a) -> int:
    if a.workers not in (1, 2):
        print("ZZLFL_FAIL at most 2 concurrent harness runs (the CPU is shared)"); return 1
    if not all(os.path.exists(seam64(k)) for k in range(M64)):
        print("ZZLFL_FAIL score64: seams missing -- run build64 first"); return 1
    os.makedirs(os.path.join(WD64, "logs"), exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor
    env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")

    def one(k):
        lb, log = label64(k), os.path.join(WD64, "logs", f"p{k:02d}.log")
        for attempt in range(a.tries):
            if (os.path.exists(log) and os.path.exists(csv_of(lb))
                    and os.path.getmtime(log) > os.path.getmtime(seam64(k))):
                st = status_of(open(log, encoding="utf-8", errors="replace").read())
                if st and st.get("status") == "PASS":
                    return k, st
            if attempt:
                time.sleep(60)
            EC.run([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", lb, "--seam", seam64(k), "--tokens", TOK,
                    "--out", f"{EC.DATA}/score"], HERE, env, log)
        return k, (status_of(open(log, encoding="utf-8", errors="replace").read()) if os.path.exists(log) else None)
    t1, status = time.time(), {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (k, st) in enumerate(ex.map(one, range(M64))):
            status[k] = st
            if (i + 1) % 4 == 0 or i + 1 == M64:
                npass = sum(1 for s in status.values() if s and s.get("status") == "PASS")
                print(f"    scored {i + 1}/{M64}  pass {npass}  {time.time() - t1:.0f} s  {now()}", flush=True)
    bad = [k for k in range(M64) if not (status.get(k) and status[k].get("status") == "PASS"
                                         and status[k].get("csv_valid_rows") == 200)]
    print(f"ZZLFL_SCORE64_{'OK' if not bad else 'FAIL'} {M64 - len(bad)}/{M64} {bad[:10]}")
    return 0 if not bad else 1


def share_h4_above_pi(poses):
    return round(float((np.abs(np.asarray(poses, dtype=np.float64)[..., 7, 2]) > np.pi).mean()), 4)


def analyze64(a) -> int:
    """gates (a64, b64, tie re-derived; G3; c64) -> the repaired E-6 table -> eval/oracle_trend.py UNCHANGED"""
    if not pick_gates_passed():
        print("ZZLFL_FAIL analyze64: the pick readout's gates have not passed"); return 1
    res, arr = build64_all()
    toks, rep, T, rt, pick = arr["toks"], arr["rep"], arr["T"], arr["rt"], arr["pick"]
    n = len(toks)
    G = {"a64_unrepaired_rebuild_equals_pipeline_E6_poses": res["gate_a64"],
         "b64_only_the_4s_heading_differs": res["gate_b64"], "tie_to_pick_readout": res["tie_to_pick_readout"]}
    disk = []
    for k in range(M64):
        z = np.load(seam64(k)) if os.path.exists(seam64(k)) else None
        disk.append(bool(z is not None and [str(t) for t in z["token"]] == toks and bits_equal(z["poses"], rep[:, k])))
    G["seams64_on_disk_equal_rebuild"] = {"seams_equal": f"{sum(disk)}/{M64}", "pass": all(disk)}
    runs, pdms = {}, np.full((n, M64), np.nan)
    sub, valid = np.full((n, M64, 6), np.nan), np.zeros((n, M64), dtype=bool)
    for k in range(M64):
        lb, log = label64(k), os.path.join(WD64, "logs", f"p{k:02d}.log")
        st = status_of(open(log, encoding="utf-8", errors="replace").read()) if os.path.exists(log) else None
        rows = read_csv(csv_of(lb)) if os.path.exists(csv_of(lb)) else {}
        for i, t in enumerate(toks):
            if t in rows:
                valid[i, k], pdms[i, k], sub[i, k] = rows[t]
        runs[lb] = {"status": (st or {}).get("status"), "csv_valid_rows": (st or {}).get("csv_valid_rows"),
                    "tokens_valid": f"{int(valid[:, k].sum())}/{n}", "C1_max_abs_delta": (st or {}).get("C1_max_abs_delta"),
                    "wall_s": (st or {}).get("wall_s"), "log": log,
                    "csv_sha256": sha256(csv_of(lb)) if os.path.exists(csv_of(lb)) else None,
                    "tie": csv_to_seam_tie(lb, seam64(k))}
    bad = [lb for lb, r in runs.items() if not (r["status"] == "PASS" and r["tokens_valid"] == f"{n}/{n}"
                                                and r["tie"]["pass"])]
    G["G3_every_run_PASS_every_token_valid"] = {
        "what": "all 64 harness runs PASS (guards C1-C3) with every token valid, each tied by the harness's call log to "
                "its seam file, unmodified since scored",
        "runs_pass": f"{M64 - len(bad)}/{M64}", "failed": bad[:20], "pass": not bad}
    rr = read_csv(csv_of(LABELS["repaired"]))
    have = all(t in rr for t in toks) and not np.isnan(pdms).any()
    d = max((abs(pdms[i, pick[i]] - rr[t][1]) for i, t in enumerate(toks)), default=float("inf")) if have else float("inf")
    ds = max((float(np.max(np.abs(sub[i, pick[i]] - np.array(rr[t][2])))) for i, t in enumerate(toks)),
             default=float("inf")) if have else float("inf")
    G["c64_pick_column_reproduces_pick_readout"] = {
        "what": "at the pick, the 64-run table must equal the pick readout's own repaired harness scores token-for-token "
                "(bit-identical poses through the same harness twice)",
        "max_abs_pdms_diff": d, "max_abs_subscore_diff": ds, "pick_column_pdms_x100": (
            round(100 * float(np.mean([pdms[i, pick[i]] for i in range(n)])), 4) if have else None),
        "pass": bool(have and d <= TOL and ds <= TOL)}
    G["all_gates_pass"] = bool(all(v["pass"] for v in G.values() if isinstance(v, dict)))
    out = {"_label": "like-for-like 016 EXTENSION: all 64 ep015 proposals re-read with Amendment 7's repair, as an E-6 "
                     "table, and the ceiling compared with ep016 (whose 64 were executed with the repair) by "
                     "eval/oracle_trend.py UNCHANGED",
           "analyzed_local": now(), "table": TABLE64, "gates": G, "harness_runs": runs}
    for k_, v in G.items():
        if isinstance(v, dict):
            print(f"  gate {k_}: pass={v['pass']}", flush=True)
    if not G["all_gates_pass"]:
        out["trend"] = "NOT READ -- a validity gate failed"
        out["cpu_only"] = no_torch()
        json.dump(out, open(OUT64_JSON, "w", encoding="utf-8"), indent=1)
        print("ZZLFL_FAIL analyze64 gates"); return 1
    # ---------------------------------------------------------------- the repaired E-6 table (proposal_table.py's format)
    os.makedirs(WD64, exist_ok=True)
    np.savez(TABLE64, token=np.array(toks), pdms=pdms, sub=sub, valid=valid, logits=T["logits"][rt], proposals=rep,
             pick=pick, sub_names=np.array(SUB), head_order=T["head_order"], rule=T["rule"])
    json.dump({"name": NAME64, "written_by": os.path.abspath(__file__), "source": {"native_dump": DUMP,
               "logits_pick_rule_from": TABLE_OF[A]}, "N": n, "M": M64,
               "gates": {k_: v["pass"] for k_, v in G.items() if isinstance(v, dict)}, "written_local": now()},
              open(os.path.join(WD64, "gates.json"), "w", encoding="utf-8"), indent=1)
    np.savez(SCORES64_COPY, token=np.array(toks), pdms=pdms, sub=sub, valid=valid, pick=pick, sub_names=np.array(SUB))
    import snapshot_pair_under_rule as SP                               # the reader oracle_trend.py uses
    readers = {}
    for nm in (A, NAME64, B):
        try:
            L_ = SP.load(nm)
            readers[nm] = {"loads": True, "shipped_rule": L_["shipped"], "n_tokens": len(L_["token"]),
                           "same_token_order_as_seams": L_["token"] == toks}
        except (AssertionError, KeyError, OSError) as e:
            readers[nm] = {"loads": False, "error": f"{type(e).__name__}: {e}"}
    TB, SB = np.load(TABLE_OF[B]), np.load(SEAM_OF[B])
    rtb = table_rows(TB, toks)
    gb_ = json.load(open(GATES_OF[B], encoding="utf-8"))
    PB = np.load(PROP_OF[B], allow_pickle=False)
    ep016 = {"what": "the ep016 E-6 table the comparison reads: complete (its own gates), and its 64 proposals are the "
                     "EXECUTED (repaired) ones -- its dump says so and its 4.0 s headings carry the repaired signature",
             "table": TABLE_OF[B], "table_sha256": sha256(TABLE_OF[B]),
             "table_written_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(TABLE_OF[B]))),
             # proposal_table.py's G3 keeps its verdict in "ok" ("pass" there is a COUNT of passing runs)
             "own_gates": {k_: (v["ok"] if "ok" in v else v.get("pass")) for k_, v in gb_.get("gates", {}).items()},
             "dump_repair_last_heading": bool(PB["repair_last_heading"]) if "repair_last_heading" in PB.files else None,
             "pick_column_equals_ep016_seam": bits_equal(TB["proposals"][rtb][np.arange(n), TB["pick"][rtb]],
                                                         SB["poses"]),
             "share_abs_heading_4.0s_above_pi_all_64": share_h4_above_pi(TB["proposals"]),
             "ep015_unrepaired_table_same_share": share_h4_above_pi(T["proposals"]),
             "ep015_repaired_table_same_share": share_h4_above_pi(rep)}
    ep016["pass"] = bool(all(v is True for v in ep016["own_gates"].values()) and ep016["dump_repair_last_heading"]
                         and ep016["pick_column_equals_ep016_seam"])
    # ---------------------------------------------------------------- eval/oracle_trend.py, UNCHANGED
    env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")
    if os.path.exists(TREND_JSON):
        os.remove(TREND_JSON)
    tlog = os.path.join(LOGDIR, "oracle_trend.log")
    rc, txt = EC.run([EC.DRIVERL_PY, "oracle_trend.py", "--pairs", *TREND_PAIRS, "--boot", str(a.boot),
                      "--out-json", TREND_JSON], HERE, env, tlog)
    trend = json.load(open(TREND_JSON, encoding="utf-8")) if os.path.exists(TREND_JSON) else None
    ade = {}
    for p in (A, B):
        pr = json.load(open(REPORT_OF[p], encoding="utf-8")).get("proposals", {})
        ade[p] = {k_: pr.get(k_) for k_ in ("ade_oracle_m", "ade_selected_m", "ade_random_m", "n", "M")}
    out.update({
        "readers_snapshot_pair_under_rule_load": readers, "ep016_table": ep016,
        "trend": {"script": os.path.join(HERE, "oracle_trend.py"), "script_sha256": sha256(os.path.join(HERE,
                                                                                                "oracle_trend.py")),
                  "argv_pairs": list(TREND_PAIRS), "boot": a.boot, "rc": rc, "log": tlog, "json": TREND_JSON,
                  "result": (trend or {}).get("pairs"),
                  "estimator": "paired log-cluster bootstrap over the logs, percentile 95 %, seed 20260927 "
                               "(oracle_trend.py, unchanged; snapshot_pair_under_rule.py's estimator)",
                  "variance_answered": "EPISODES only (another draw of logs); one checkpoint each, deterministic "
                                       "forward"},
        "oracle_ade_position_only": {
            "what": "the seam reports' open-loop ADE of the 64 proposals vs the logged human future, computed from x, y "
                    "ONLY (refe_navtest_seam.py: `to_navsim(traj[j])[:, :2]`). Gate b64 proves the repair leaves every "
                    "x, y bit-identical, so it needs no re-read: 015's value IS 015-repaired's.",
            "values": ade},
        "table_sha256": sha256(TABLE64), "scores_copy": SCORES64_COPY, "scores_copy_sha256": sha256(SCORES64_COPY),
        "seams_md5": {label64(k): md5(seam64(k)) for k in range(M64)},
        "cpu_only": no_torch()})
    json.dump(out, open(OUT64_JSON, "w", encoding="utf-8"), indent=1)
    ok = bool(rc == 0 and trend and all(v["loads"] for v in readers.values()) and ep016["pass"])
    if ok and not a.no_docs and os.path.exists(OUT_JSON):
        o = json.load(open(OUT_JSON, encoding="utf-8"))
        write_result(o, out)
        write_manifest(o, out)
    for pr, r in ((trend or {}).get("pairs") or {}).items():
        for k_, v in r.items():
            print(f"  {pr:17s} {k_:11s} {v['a']:6.2f} -> {v['b']:6.2f}  {v['b_minus_a']:+.2f} [{v['ci95'][0]:+.2f}, "
                  f"{v['ci95'][1]:+.2f}]{'  separated' if v['separated'] else ''}", flush=True)
    print(f"  readers {[(k_, v['loads']) for k_, v in readers.items()]}; ep016 table pass={ep016['pass']}; "
          f"torch imported: {out['cpu_only']['torch_imported']}")
    print(f"ZZLFL_{'ANALYZE64_OK' if ok else 'FAIL analyze64'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("build", "score", "analyze", "all", "selftest", "build64", "score64", "analyze64",
                                      "all64"))
    ap.add_argument("--workers", type=int, default=2, help="concurrent harness runs, 1 or 2 (the CPU is shared)")
    ap.add_argument("--tries", type=int, default=2)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--no-docs", action="store_true", help="do not write the RESULT doc and the MANIFEST")
    a = ap.parse_args()
    if a.stage == "selftest":
        return selftest(a)
    if a.stage in ("build64", "score64", "analyze64", "all64"):
        if a.stage in ("build64", "all64"):
            rc = build64(a)
            if rc or a.stage == "build64":
                return rc
        if a.stage in ("score64", "all64"):
            rc = score64(a)
            if rc or a.stage == "score64":
                return rc
        return analyze64(a)
    if a.stage in ("build", "all"):
        rc = build(a)
        if rc or a.stage == "build":
            return rc
    if a.stage in ("score", "all"):
        rc = score(a)
        if rc or a.stage == "score":
            return rc
    return analyze(a)


if __name__ == "__main__":
    sys.exit(main())
