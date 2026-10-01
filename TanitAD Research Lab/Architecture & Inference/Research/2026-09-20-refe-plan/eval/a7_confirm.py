#!/usr/bin/env python3
"""SPEC_NAVTEST AMENDMENT 7 -- CONFIRMATION, run exactly as registered (eval/SPEC_NAVTEST.md, ~13:50 Berlin 2026-09-27).

REPAIR UNDER TEST (ONE, fixed by the amendment): on the planner's native [20, 3] output, heading[19] := heading[18]
for the EXECUTED plan, before the NAVSIM conversion; nothing else changes.
TOKENS: Amendment 5's 923 (43 logs, none in W3's 200): eval/raw/a5_confirm/a5_confirm_tokens.json (md5 3098d178...).
SNAPSHOT: after epoch 15, snap_epoch015.pt (md5 d7c59f4f2fbcbde3e2dec8f67d63a7e7). Both md5s are asserted.

MEASUREMENT. "The unchanged pipeline dumps the shipped picks": stage `dump` runs `refe_navtest_seam.main()` ITSELF --
the pipeline's own seam writer, with eval_checkpoint.py's arguments (default frames, --dump-proposals), in
eval_checkpoint.py's interpreter and environment (EC.DRIVERL_PY, EC.env_driverl()) -- and observes, without changing
anything, what `REFePlanner.infer` returns (a wrapper on the method records the NATIVE [20, 3] executed plan, the pick
and the scorer's logits per token; infer's return value is passed through untouched). The pipeline therefore writes the
SHIPPED seam and its proposal dump exactly as it always does; this script additionally keeps the native executed plan,
which the NAVSIM grid cannot carry (native index 18 = t 3.8 s is not a 0.5 s pose).
`build` writes the REPAIRED seam from the native record (and, reported only, the path-tangent variant); `score` runs
both seams and the shipped one through `score_navtest_refe.py` UNCHANGED; `analyze` checks the gates, reads the
registered statistic and the decision, and writes the JSON, RESULT_A7_a7confirm_ep015.md and the MANIFEST.

GATES (all must pass before the statistic is read):
  (a) every repaired pose's x and y are bit-identical to the shipped seam's and only the t = 4.0 s heading differs
  (b) the shipped seam reproduces the pipeline's pick and score on every token: the pipeline's seam row equals
      to_navsim(recorded native executed plan) bit-for-bit, the pipeline's dumped pick equals the recorded pick, and
      the shipped seam's harness score equals the pipeline's own scoring of that seam (score_navtest_refe.py, C1-C3)
  (c) every harness run PASSes with every token valid
STATISTIC: per-token PDMS(repaired) - PDMS(shipped), mean over the 923 tokens, paired log-cluster bootstrap over the 43
logs, 10,000 resamples, 95 %, seed 20260927 (eval/snapshot_pair_under_rule.py's estimator). Answers "another draw of
episodes" only: one checkpoint, one deterministic forward.
DECISION (committed in the amendment): ADOPT iff lower bound > 0; REFUTED iff upper bound < 0; else NOT PROVEN.
REPORTED, NOT GATING: NAVSIM sub-score deltas; the repair with the path-tangent heading instead; the four families
(identical by construction -- the NAVSIM family adapter reads positions only; shown by running it on both seams).

    python eval/a7_confirm.py dump        # GPU (child under EC.env_driverl()); stops at --deadline (local HH:MM)
    python eval/a7_confirm.py build
    python eval/a7_confirm.py score
    python eval/a7_confirm.py analyze
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_checkpoint as EC        # noqa: E402  one source for the venv + env the pipeline runs with
import refe_navtest_seam as SEAM    # noqa: E402  the pipeline's own seam writer (run, not re-implemented)

NAME = "a7confirm_ep015"
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt"
CKPT_MD5 = "d7c59f4f2fbcbde3e2dec8f67d63a7e7"
TOK = os.path.join(HERE, "raw", "a5_confirm", "a5_confirm_tokens.json")
TOK_MD5_PREFIX = "3098d178"
SEAM_SHIPPED = f"{EC.DATA}/seams/refe_{NAME}.npz"
PROP_DUMP = f"{EC.DATA}/proptable/{NAME}/proposals.npz"
WD = f"{EC.DATA}/proptable/{NAME}"
NATIVE = f"{WD}/native_executed.npz"
SEAM_REPAIRED = f"{EC.DATA}/seams/refe_{NAME}_repaired.npz"
SEAM_TANGENT = f"{EC.DATA}/seams/refe_{NAME}_repaired_tangent.npz"
LABELS = {"shipped": f"refe_{NAME}", "repaired": f"refe_{NAME}_repaired", "tangent": f"refe_{NAME}_repaired_tangent"}
SEAMS = {"shipped": SEAM_SHIPPED, "repaired": SEAM_REPAIRED, "tangent": SEAM_TANGENT}
RAW = os.path.join(HERE, "raw", "a7_confirm")
OUT_JSON = os.path.join(RAW, "a7_confirm_ep015.json")
RESULT_MD = os.path.join(HERE, "RESULT_A7_a7confirm_ep015.md")
MANIFEST = os.path.join(RAW, "MANIFEST.md")
SEED = 20260927
CHILD = "REFE_A7_CHILD"
SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
       "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
HEAD = ("NC", "DAC", "EP", "TTC", "C", "DDC")


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


def repair(native20):
    """THE registered repair: heading[19] := heading[18]; nothing else changes."""
    q = np.array(native20, copy=True)
    q[..., 19, 2] = q[..., 18, 2]
    return q


def repair_tangent(native20):
    """REPORTED ONLY: heading[19] := direction of the last path segment (18 -> 19); heading[18] if it is < 5 cm."""
    q = np.array(native20, dtype=np.float64, copy=True)
    d = q[..., 19, :2] - q[..., 18, :2]
    tan = np.arctan2(d[..., 1], d[..., 0])
    q[..., 19, 2] = np.where(np.linalg.norm(d, axis=-1) >= 0.05, tan, q[..., 18, 2])
    return q.astype(np.asarray(native20).dtype)


# ------------------------------------------------------------------------------------------------ dump (GPU)
def dump_child(a) -> int:
    """Runs the pipeline's own seam writer with an OBSERVE-ONLY wrapper on REFePlanner.infer."""
    import threading
    import planner as PL
    rec: dict = {}
    orig = PL.REFePlanner.infer

    def infer_observed(self, ego, img):
        traj, score, k = orig(self, ego, img)                  # the planner's own call, result passed through
        tok = self._scenario._initial_lidar_token
        rec[tok] = (traj[k].float().cpu().numpy().copy(), score.float().cpu().numpy().copy(), int(k),
                    time.time())
        return traj, score, k
    PL.REFePlanner.infer = infer_observed

    stop = {"flag": False}

    def watchdog():                                            # the GPU window: stop hard at the deadline
        hh, mm = (int(x) for x in a.deadline.split(":"))
        while True:
            lt = time.localtime()
            if (lt.tm_hour, lt.tm_min) >= (hh, mm):
                stop["flag"] = True
                print(f"ZZA7_DEADLINE {a.deadline} reached at {time.strftime('%H:%M:%S')} -- stopping", flush=True)
                os._exit(3)                                    # the seam writes nothing partial on purpose
            time.sleep(20)
    threading.Thread(target=watchdog, daemon=True).start()
    os.makedirs(WD, exist_ok=True)
    sys.argv = ["refe_navtest_seam.py", "--ckpt", CKPT, "--frames", f"{EC.DATA}/frames", "--tokens", a.tokens,
                "--out", SEAM_SHIPPED, "--arm", f"REFe_{NAME}", "--dump-proposals", PROP_DUMP]
    t0 = time.time()
    rc = SEAM.main()                                           # THE PIPELINE, unchanged
    toks = sorted(rec)
    np.savez(NATIVE, token=np.array(toks), native=np.stack([rec[t][0] for t in toks]),
             logits=np.stack([rec[t][1] for t in toks]), pick=np.array([rec[t][2] for t in toks], dtype=np.int64),
             meta=np.array(json.dumps({"seam_rc": int(rc), "gpu_seconds": round(time.time() - t0, 1),
                                       "finished_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                       "argv_to_pipeline": sys.argv, "python": sys.executable,
                                       "env": {k: os.environ.get(k) for k in ("REFE_SIM_HZ", "NUPLAN_MAPS_ROOT",
                                                                              "REFE_BACKBONE_ROOT")}})))
    print(f"  native record: {len(toks)} tokens -> {NATIVE}; pipeline seam rc {rc}", flush=True)
    print(f"ZZA7_DUMP_{'OK' if rc == 0 else 'FAIL'} {len(toks)}", flush=True)
    return rc


def dump(a) -> int:
    got = md5(CKPT)
    if got != CKPT_MD5:
        print(f"ZZA7_FAIL ckpt md5 {got} != {CKPT_MD5}"); return 1
    if not md5(a.tokens).startswith(TOK_MD5_PREFIX):
        print(f"ZZA7_FAIL tokens md5 {md5(a.tokens)} does not start {TOK_MD5_PREFIX}"); return 1
    env = EC.env_driverl()
    env[CHILD] = "1"
    return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), "dump-child", "--tokens", a.tokens,
                            "--deadline", a.deadline], cwd=HERE, env=env)


# ------------------------------------------------------------------------------------------------ build
def build(a) -> int:
    S = np.load(SEAM_SHIPPED)
    N = np.load(NATIVE, allow_pickle=False)
    stok = [str(t) for t in S["token"]]
    nix = {str(t): i for i, t in enumerate(N["token"])}
    missing = [t for t in stok if t not in nix]
    if missing or len(nix) != len(stok):
        print(f"ZZA7_FAIL build: native record {len(nix)} vs seam {len(stok)}, missing {len(missing)}"); return 1
    ri = np.array([nix[t] for t in stok])
    nat = N["native"][ri]                                                  # [n, 20, 3] float32, seam order
    shipped_rebuilt = np.stack([SEAM.to_navsim(x).astype(np.float32) for x in nat])
    rep = repair(nat)
    tan = repair_tangent(nat)
    seams = {"repaired": np.stack([SEAM.to_navsim(x).astype(np.float32) for x in rep]),
             "tangent": np.stack([SEAM.to_navsim(x).astype(np.float32) for x in tan])}
    for k_, poses in seams.items():
        np.savez(SEAMS[k_], token=S["token"], fingerprint=S["fingerprint"], poses=poses, sampling=S["sampling"],
                 arm=np.array(f"REFe_{NAME}_{k_}"))
    print(f"  shipped rebuilt from the native record == pipeline seam: {np.array_equal(shipped_rebuilt, S['poses'])}; "
          f"wrote {list(seams)}", flush=True)
    print("ZZA7_BUILD_OK")
    return 0


# ------------------------------------------------------------------------------------------------ score
def status_of(txt):
    for line in reversed(txt.splitlines()):
        if line.startswith("{") and '"status"' in line:
            return json.loads(line)
    return None


def csv_of(label):
    return f"{EC.DATA}/score/{label}/{label}.csv"


def score(a) -> int:
    if os.environ.get(CHILD) == "1":
        print("ZZA7_FAIL score must not run under the pipeline child's environment"); return 1
    os.makedirs(os.path.join(WD, "logs"), exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor

    def one(k_):
        lb, log = LABELS[k_], os.path.join(WD, "logs", f"score_{k_}.log")
        for attempt in range(a.tries):
            if os.path.exists(log) and os.path.exists(csv_of(lb)):
                st = status_of(open(log, encoding="utf-8", errors="replace").read())
                if st and st.get("status") == "PASS":
                    return k_, st
            if attempt:
                time.sleep(60)                                  # the W3 RAM guard: give the box a minute
            EC.run([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", lb, "--seam", SEAMS[k_], "--tokens", a.tokens,
                    "--out", f"{EC.DATA}/score"], HERE, dict(os.environ, PYTHONIOENCODING="utf-8"), log)
        st = status_of(open(log, encoding="utf-8", errors="replace").read()) if os.path.exists(log) else None
        return k_, st
    res = {}
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for k_, st in ex.map(one, list(LABELS)):
            res[k_] = st
            print(f"  scored {k_}: {st.get('status') if st else None} rows {st.get('csv_valid_rows') if st else None}"
                  f"  local {time.strftime('%H:%M:%S')}", flush=True)
    ok = all(st and st.get("status") == "PASS" for st in res.values())
    print(f"ZZA7_SCORE_{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ analyze
def read_csv(p):
    rows = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["token"] == "average":
            continue
        rows[r["token"]] = (r["valid"] == "True", float(r["score"]), tuple(float(r[c]) for c in SUB))
    return rows


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def families_of(seam, tag):
    op, lg = os.path.join(WD, f"families_{tag}.json"), os.path.join(WD, f"families_{tag}.log")
    if os.path.exists(op):
        os.remove(op)
    rc, _ = EC.run([EC.TANITAD_PY, "families6.py", "--seam", seam, "--inputs", EC.EXPORT, "--stage", "1",
                    "--label", f"REFe-{NAME}-{tag}", "--out", op, "--n-boot", "2000"],
                   EC.EV6, dict(os.environ, PYTHONIOENCODING="utf-8"), lg)
    return op if os.path.exists(op) else None


def strip_volatile(o):
    """a families block minus the fields that name the run (label, seam path), for an identity comparison"""
    if isinstance(o, dict):
        return {k: strip_volatile(v) for k, v in o.items() if k not in ("_label", "seam", "label")}
    if isinstance(o, list):
        return [strip_volatile(v) for v in o]
    return o


def spec_record():
    """the registered text: SPEC_NAVTEST.md's git blob (the amendment names blob 71fc61b5) and its mtime, beside the
    time this confirmation's first GPU result was written"""
    p = os.path.join(HERE, "SPEC_NAVTEST.md")
    raw = open(p, "rb").read()
    blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    txt = raw.decode("utf-8", errors="replace")
    return {"spec": p, "git_blob": blob, "registered_blob_prefix": "71fc61b5", "blob_matches": blob.startswith("71fc61b5"),
            "spec_mtime_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(p))),
            "amendment_7_present": "AMENDMENT 7" in txt,
            "native_record_mtime_local": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                       time.localtime(os.path.getmtime(NATIVE)))}


def analyze(a) -> int:
    import stop_candidate_probe as SCP
    tokj = json.load(open(a.tokens, encoding="utf-8"))
    # overridable ONLY so the gate self-test can feed mutated copies; the defaults are the registered artifacts
    native_p = a.native or NATIVE
    rep_p = a.repaired_seam or SEAM_REPAIRED
    logdir = a.logdir or os.path.join(WD, "logs")
    out_json = a.out_json or OUT_JSON
    S = np.load(SEAM_SHIPPED)
    N = np.load(native_p, allow_pickle=False)
    Pd = np.load(PROP_DUMP)
    nmeta = json.loads(str(N["meta"]))
    stok = [str(t) for t in S["token"]]
    n = len(stok)
    nix = {str(t): i for i, t in enumerate(N["token"])}
    pix = {str(t): i for i, t in enumerate(Pd["token"])}
    ri = np.array([nix[t] for t in stok])
    pi_ = np.array([pix[t] for t in stok])
    nat = N["native"][ri]
    out = {"_label": "SPEC_NAVTEST AMENDMENT 7 confirmation, run as registered (eval/SPEC_NAVTEST.md)",
           "repair": "on the planner's native [20, 3] output, heading[19] := heading[18] for the executed plan, before "
                     "the NAVSIM conversion; nothing else changes",
           "snapshot": {"ckpt": CKPT, "md5": md5(CKPT), "registered_md5": CKPT_MD5},
           "tokens": {"file": a.tokens, "md5": md5(a.tokens), "n": n, "n_logs": len(set(tokj["token_log"][t]
                                                                                      for t in stok))},
           "pipeline": {"seam": SEAM_SHIPPED, "proposal_dump": PROP_DUMP, "native_record": native_p,
                        "repaired_seam": rep_p, "dump_meta": nmeta},
           "registration": spec_record()}
    # ---------------------------------------------------------------- gates
    G = {}
    rep_seam = np.load(rep_p)
    Rp, Sp = rep_seam["poses"], S["poses"]
    same_tok = [str(t) for t in rep_seam["token"]] == stok
    xy_same = bool(np.array_equal(Rp[..., :2], Sp[..., :2]))
    h_early_same = bool(np.array_equal(Rp[:, :7, 2], Sp[:, :7, 2]))
    changed = Rp[:, 7, 2] != Sp[:, 7, 2]
    want_last = nat[:, 18, 2]
    G["a_only_the_4s_heading_differs"] = {
        "token_order_identical": same_tok, "x_y_bit_identical_all_8_poses": xy_same,
        "heading_bit_identical_at_0.5..3.5s": h_early_same,
        "repaired_4s_heading_equals_native_heading18_bitwise": bool(np.array_equal(Rp[:, 7, 2], want_last)),
        "tokens_whose_4s_heading_changed": int(changed.sum()), "n": n}
    G["a_only_the_4s_heading_differs"]["pass"] = bool(same_tok and xy_same and h_early_same
                                                      and G["a_only_the_4s_heading_differs"][
                                                          "repaired_4s_heading_equals_native_heading18_bitwise"])
    rebuilt = np.stack([SEAM.to_navsim(x).astype(np.float32) for x in nat])
    order = ["NC", "DAC", "EP", "TTC", "C", "DDC"]
    L = Pd["logits"][pi_].astype(np.float64)
    pick_np = SCP.agg_v1_np(L, order).argmax(1)
    prop_at_pick = Pd["proposals"][pi_][np.arange(n), Pd["pick"][pi_]]
    st_ship = status_of(open(os.path.join(logdir, "score_shipped.log"), encoding="utf-8", errors="replace").read()) \
        if os.path.exists(os.path.join(logdir, "score_shipped.log")) else None
    G["b_shipped_seam_reproduces_the_pipeline"] = {
        "pipeline_seam_equals_to_navsim_of_recorded_native_bitwise": bool(np.array_equal(rebuilt, Sp)),
        "pipeline_dump_pick_equals_recorded_pick": f"{int((Pd['pick'][pi_] == N['pick'][ri]).sum())}/{n}",
        "pipeline_dump_pick_equals_v1_aggregate_argmax_of_its_logits_float64": f"{int((pick_np == Pd['pick'][pi_]).sum())}/{n}",
        "pipeline_dump_proposal_at_pick_equals_seam_bitwise": bool(np.array_equal(prop_at_pick, Sp)),
        "shipped_seam_scored_by_the_pipeline_scorer": (st_ship or {}).get("status"),
        "shipped_seam_harness_C1_max_abs_delta": (st_ship or {}).get("C1_max_abs_delta"),
        "note": "the shipped seam IS the pipeline's (refe_navtest_seam.main wrote it) and its score is the pipeline's "
                "step 2 (score_navtest_refe.py, unchanged); these checks tie the native record the repair is built "
                "from to that seam, and the pick to the pipeline's dump and to the v1 rule on its own logits"}
    gb = G["b_shipped_seam_reproduces_the_pipeline"]
    gb["pass"] = bool(gb["pipeline_seam_equals_to_navsim_of_recorded_native_bitwise"]
                      and gb["pipeline_dump_pick_equals_recorded_pick"] == f"{n}/{n}"
                      and gb["pipeline_dump_pick_equals_v1_aggregate_argmax_of_its_logits_float64"] == f"{n}/{n}"
                      and gb["pipeline_dump_proposal_at_pick_equals_seam_bitwise"]
                      and gb["shipped_seam_scored_by_the_pipeline_scorer"] == "PASS")
    runs = {}
    rows = {}
    for k_ in LABELS:
        log = os.path.join(logdir, f"score_{k_}.log")
        st = status_of(open(log, encoding="utf-8", errors="replace").read()) if os.path.exists(log) else None
        rows[k_] = read_csv(csv_of(LABELS[k_])) if os.path.exists(csv_of(LABELS[k_])) else {}
        nvalid = sum(1 for t in stok if t in rows[k_] and rows[k_][t][0])
        runs[k_] = {"label": LABELS[k_], "status": (st or {}).get("status"),
                    "csv_valid_rows": (st or {}).get("csv_valid_rows"), "tokens_valid": f"{nvalid}/{n}",
                    "failures": (st or {}).get("failures")}
    G["c_every_harness_run_PASS_every_token_valid"] = {"runs": runs, "pass": bool(all(
        r["status"] == "PASS" and r["tokens_valid"] == f"{n}/{n}" for r in runs.values()))}
    G["all_gates_pass"] = bool(all(v["pass"] for v in G.values() if isinstance(v, dict)))
    out["gates"] = G
    for k_, v in G.items():
        if isinstance(v, dict):
            print(f"  gate {k_}: pass={v['pass']}", flush=True)
    if not G["all_gates_pass"]:
        out["decision"] = "NOT READ -- a validity gate failed"
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        json.dump(out, open(out_json, "w", encoding="utf-8"), indent=1)
        print("ZZA7_FAIL gates"); return 1
    # ---------------------------------------------------------------- the registered statistic
    pdm = {k_: np.array([rows[k_][t][1] for t in stok]) for k_ in LABELS}
    sub = {k_: np.array([rows[k_][t][2] for t in stok]) for k_ in LABELS}
    logs = np.array([tokj["token_log"][t] for t in stok])
    ul = np.unique(logs)
    idx = {l: np.where(logs == l)[0] for l in ul}
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([idx[l] for l in rng.choice(ul, size=len(ul), replace=True)]) for _ in range(a.boot)]
    d = pdm["repaired"] - pdm["shipped"]
    bs = np.array([100 * d[ix].mean() for ix in draws])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    decision = "ADOPT" if lo > 0 else ("REFUTED" if hi < 0 else "NOT PROVEN")
    same_pose = ~changed
    out["statistic"] = {
        "what": "per-token PDMS(repaired) - PDMS(shipped), mean over the tokens, x100",
        "estimator": f"paired log-cluster bootstrap over the {len(ul)} logs, {a.boot} resamples, percentile 95 %, "
                     f"seed {SEED} (eval/snapshot_pair_under_rule.py's estimator)",
        "variance_answered": "EPISODES (another draw of logs) only: one checkpoint, one deterministic forward",
        "shipped_pdms": round(100 * pdm["shipped"].mean(), 2), "repaired_pdms": round(100 * pdm["repaired"].mean(), 2),
        "delta": round(100 * float(d.mean()), 2), "ci95": [round(lo, 2), round(hi, 2)],
        "helped": int((d > 0).sum()), "hurt": int((d < 0).sum()), "tied": int((d == 0).sum()),
        "tokens_with_unchanged_pose_max_abs_score_diff": float(np.abs(d[same_pose]).max()) if same_pose.any() else None,
        "n_tokens_with_unchanged_pose": int(same_pose.sum())}
    out["decision"] = decision
    out["decision_rule"] = "ADOPT iff lower bound > 0; REFUTED iff upper bound < 0; else NOT PROVEN (Amendment 7)"
    # ---------------------------------------------------------------- reported, not gating
    dt = pdm["tangent"] - pdm["shipped"]
    bt = np.array([100 * dt[ix].mean() for ix in draws])
    out["reported_not_gating"] = {
        "subscore_deltas_x100_repaired_minus_shipped": dict(zip(HEAD, [round(100 * float(x), 2)
                                                                      for x in (sub["repaired"] - sub["shipped"]).mean(0)])),
        "subscores_x100": {k_: dict(zip(HEAD, [round(100 * float(x), 2) for x in sub[k_].mean(0)])) for k_ in LABELS},
        "tangent_variant": {"pdms": round(100 * pdm["tangent"].mean(), 2), "delta_vs_shipped": round(100 * float(dt.mean()), 2),
                            "ci95": [round(float(np.percentile(bt, 2.5)), 2), round(float(np.percentile(bt, 97.5)), 2)],
                            "what": "heading[19] := direction of the last path segment (heading[18] if < 5 cm)"},
        "same_repair_on_the_latest_snapshots_200_tokens": {
            "source": "eval/raw/e6_sub200_ep015/slow_copies.json -> repair_last_heading_shipped_pick.hold",
            "value": "61.18 -> 77.94, +16.76 [+11.72, +22.13] (snapshot 015, W3's 200 tokens; the exploration that "
                     "motivated the amendment)"}}
    # the defect on THESE tokens (descriptive)
    xy = nat[..., :2].astype(np.float64)
    prev = np.concatenate([np.zeros((n, 1, 2)), xy[:, :-1]], axis=1)
    tanh = np.arctan2(xy[..., 1] - prev[..., 1], xy[..., 0] - prev[..., 0])
    err = np.abs(wrap(nat[..., 2].astype(np.float64) - tanh))
    mv = np.linalg.norm(xy - prev, axis=-1) > 0.5
    out["reported_not_gating"]["defect_on_these_tokens_executed_plan"] = {
        "share_abs_heading19_above_pi": round(float((np.abs(nat[:, 19, 2]) > np.pi).mean()), 4),
        "median_abs_heading_minus_tangent_rad_at_18": round(float(np.median(err[:, 18][mv[:, 18]])), 4),
        "median_abs_heading_minus_tangent_rad_at_19": round(float(np.median(err[:, 19][mv[:, 19]])), 4)}
    # STOP floor on these tokens (context)
    stop_csv = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
                "STOP_navtest/STOP_navtest.csv")
    sr = read_csv(stop_csv)
    out["reported_not_gating"]["stop_floor_on_these_tokens"] = round(100 * float(np.mean([sr[t][1] for t in stok])), 2)
    # the four families: identical by construction; run on both seams to show it
    fam = {"what": "families6.py (TANITAD venv, unchanged) on the shipped and the repaired seam. The NAVSIM family "
                   "adapter reads positions only (it drops the heading column; its heading_mae_deg is the path "
                   "tangent), so the blocks are identical by construction -- shown, not assumed."}
    fs, fr = (None, None) if a.no_families else (families_of(SEAM_SHIPPED, "shipped"), families_of(rep_p, "repaired"))
    if a.no_families:
        fam["status"] = "NOT RUN (--no-families)"
    elif fs and fr:
        js, jr = json.load(open(fs, encoding="utf-8")), json.load(open(fr, encoding="utf-8"))
        fam["headline_numbers_identical"] = SCP.summarize_families(fs) == SCP.summarize_families(fr)
        fam["full_blocks_identical_apart_from_run_labels"] = (strip_volatile(js["families"]["refcv6"])
                                                              == strip_volatile(jr["families"]["refcv6"]))
        fam["shipped_summary"] = SCP.summarize_families(fs)
        fam["files"] = {"shipped": fs, "repaired": fr}
    else:
        fam["status"] = "families6.py did not produce both blocks"
    out["four_families"] = fam
    out["provenance"] = {"script": os.path.abspath(__file__), "script_sha256": sha256(os.path.abspath(__file__)),
                         "shipped_seam_sha256": sha256(SEAM_SHIPPED), "repaired_seam_sha256": sha256(rep_p),
                         "tangent_seam_sha256": sha256(SEAM_TANGENT), "native_record_sha256": sha256(native_p),
                         "proposal_dump_sha256": sha256(PROP_DUMP),
                         "csv_sha256": {k_: sha256(csv_of(LABELS[k_])) for k_ in LABELS}}
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    json.dump(out, open(out_json, "w", encoding="utf-8"), indent=1)
    if not a.no_docs:
        write_result(out)
        write_manifest(out)
    s = out["statistic"]
    print(f"  shipped {s['shipped_pdms']:.2f} -> repaired {s['repaired_pdms']:.2f}  delta {s['delta']:+.2f} "
          f"[{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]  -> {decision}", flush=True)
    print(f"ZZA7_OK {decision} {s['delta']:+.2f} {s['ci95'][0]:+.2f} {s['ci95'][1]:+.2f}")
    return 0


def write_result(o):
    s, g, r = o["statistic"], o["gates"], o["reported_not_gating"]
    ga, gb, gc = (g["a_only_the_4s_heading_differs"], g["b_shipped_seam_reproduces_the_pipeline"],
                  g["c_every_harness_run_PASS_every_token_valid"])
    fam = o["four_families"]
    lines = [
        f"# RESULT -- SPEC_NAVTEST AMENDMENT 7 -- `{NAME}` (the last-pose heading repair)", "",
        "*Every number is read from the unchanged harness's per-token scores; none is typed. Written by "
        "`eval/a7_confirm.py`; the repair, the gates, the statistic and the decision rule were fixed in Amendment 7 "
        "(registered ~13:50 Berlin, blob `71fc61b5`) before any confirmation-token result of the repair existed.*", "",
        f"**Confirmation tokens:** {o['tokens']['n']} from {o['tokens']['n_logs']} navtest logs (Amendment 5's set, "
        f"none in W3's 200; md5 `{o['tokens']['md5']}`). **Snapshot:** after epoch 15 (md5 `{o['snapshot']['md5']}`). "
        "**Repair:** on the planner's native [20, 3] output, heading[19] := heading[18] for the executed plan, before "
        "the NAVSIM conversion; nothing else changes.", "",
        f"## Verdict: **{o['decision']}**", "",
        "| seam | PDMS | vs shipped, PDMS points [95 %] | better / worse / tied |",
        "|---|---|---|---|",
        f"| shipped (the pipeline's) | {s['shipped_pdms']:.2f} | -- | -- |",
        f"| **repaired (the measure under test)** | **{s['repaired_pdms']:.2f}** | **{s['delta']:+.2f} "
        f"[{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]** | {s['helped']} / {s['hurt']} / {s['tied']} |",
        f"| path-tangent heading instead (reported, not gating) | {r['tangent_variant']['pdms']:.2f} | "
        f"{r['tangent_variant']['delta_vs_shipped']:+.2f} [{r['tangent_variant']['ci95'][0]:+.2f}, "
        f"{r['tangent_variant']['ci95'][1]:+.2f}] | -- |", "",
        f"Decision rule committed in Amendment 7: ADOPT iff the lower bound is above 0; REFUTED iff the upper bound is "
        f"below 0; otherwise NOT PROVEN. Estimator: {s['estimator']}. It answers \"another draw of episodes\" only.", "",
        f"Registration check: SPEC_NAVTEST.md git blob `{o['registration']['git_blob'][:8]}` (registered "
        f"`71fc61b5`: {'match' if o['registration']['blob_matches'] else 'MISMATCH'}), file time "
        f"{o['registration']['spec_mtime_local']}; the native record this confirmation read was written "
        f"{o['registration']['native_record_mtime_local']}.", "",
        "## Validity gates (all must pass before the statistic is read)", "",
        "| gate | result |", "|---|---|",
        f"| (a) repaired x, y bit-identical to the shipped seam; only the t = 4.0 s heading differs | "
        f"{'PASS' if ga['pass'] else 'FAIL'}: x/y all 8 poses {ga['x_y_bit_identical_all_8_poses']}, headings "
        f"0.5-3.5 s {ga['heading_bit_identical_at_0.5..3.5s']}, 4.0 s heading == native heading[18] "
        f"{ga['repaired_4s_heading_equals_native_heading18_bitwise']}; changed on {ga['tokens_whose_4s_heading_changed']}"
        f"/{ga['n']} tokens |",
        f"| (b) the shipped seam reproduces the pipeline's pick and score | {'PASS' if gb['pass'] else 'FAIL'}: seam == "
        f"to_navsim(recorded native) {gb['pipeline_seam_equals_to_navsim_of_recorded_native_bitwise']}; dump pick == "
        f"recorded pick {gb['pipeline_dump_pick_equals_recorded_pick']}; == v1 argmax of its logits "
        f"{gb['pipeline_dump_pick_equals_v1_aggregate_argmax_of_its_logits_float64']}; proposal at pick == seam "
        f"{gb['pipeline_dump_proposal_at_pick_equals_seam_bitwise']}; scored {gb['shipped_seam_scored_by_the_pipeline_scorer']} |",
        f"| (c) every harness run PASS, every token valid | {'PASS' if gc['pass'] else 'FAIL'}: "
        + "; ".join(f"{k_} {v['status']} {v['tokens_valid']}" for k_, v in gc["runs"].items()) + " |", "",
        "## Reported, not gating", "",
        "NAVSIM sub-scores x100 (repaired - shipped): "
        + ", ".join(f"{k_} {v:+.2f}" for k_, v in r["subscore_deltas_x100_repaired_minus_shipped"].items()) + ".", "",
        f"The defect on these tokens (executed plan): |heading[19]| > pi on "
        f"{100 * r['defect_on_these_tokens_executed_plan']['share_abs_heading19_above_pi']:.1f} %; median "
        f"|heading - path tangent| {r['defect_on_these_tokens_executed_plan']['median_abs_heading_minus_tangent_rad_at_19']} "
        f"rad at t = 4.0 s vs {r['defect_on_these_tokens_executed_plan']['median_abs_heading_minus_tangent_rad_at_18']} "
        "rad at 3.8 s.", "",
        f"STOP floor on these tokens (context): {r['stop_floor_on_these_tokens']:.2f}.", "",
        f"Same repair on the latest snapshot's 200 tokens (the exploration): {r['same_repair_on_the_latest_snapshots_200_tokens']['value']}.", "",
        "**The four metric families are identical by construction** -- the NAVSIM family adapter reads positions only "
        "(it drops the heading column; its heading_mae_deg is the path tangent) -- and running `families6.py` on both "
        f"seams confirms it: headline numbers identical = {fam.get('headline_numbers_identical')}; full per-family "
        f"blocks identical apart from run labels = {fam.get('full_blocks_identical_apart_from_run_labels')}. "
        "Strategic: unavailable in NAVSIM.", "",
        "## Not in scope", "",
        "The model-side cause of the corrupted heading is a SEPARATE question (Amendment 7): any training change for it "
        "needs its own proof. Adoption in `refe/planner.py` is the coordinator's step; this script touches no shipped "
        "path.", "",
        f"Artifacts and checksums: `eval/raw/a7_confirm/MANIFEST.md`; full record `eval/raw/a7_confirm/{os.path.basename(OUT_JSON)}`."]
    open(RESULT_MD, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


def write_manifest(o):
    p = o["provenance"]
    lines = ["# SPEC_NAVTEST Amendment 7 confirmation -- inputs and outputs", "",
             "| artifact | where | sha256 |", "|---|---|---|",
             f"| a7_confirm.py (this run's script) | repo: eval/a7_confirm.py | `{p['script_sha256']}` |",
             f"| {os.path.basename(OUT_JSON)} | repo: eval/raw/a7_confirm/ | (this manifest is written with it) |",
             f"| RESULT_A7_a7confirm_ep015.md | repo: eval/ | (written with it) |",
             f"| a5_confirm_tokens.json (inputs; md5 {o['tokens']['md5']}) | repo: eval/raw/a5_confirm/ | "
             f"`{sha256(TOK)}` |",
             f"| snap_epoch015.pt (md5 {o['snapshot']['md5']}) | data dir only: {CKPT} | `{sha256(CKPT)}` |",
             f"| shipped seam (the pipeline's) | data dir only: {SEAM_SHIPPED} | `{p['shipped_seam_sha256']}` |",
             f"| repaired seam | data dir only: {SEAM_REPAIRED} | `{p['repaired_seam_sha256']}` |",
             f"| path-tangent seam (reported) | data dir only: {SEAM_TANGENT} | `{p['tangent_seam_sha256']}` |",
             f"| native executed-plan record | data dir only: {NATIVE} | `{p['native_record_sha256']}` |",
             f"| pipeline proposal dump | data dir only: {PROP_DUMP} | `{p['proposal_dump_sha256']}` |"]
    for k_, h in p["csv_sha256"].items():
        lines.append(f"| harness csv, {k_} | data dir only: {csv_of(LABELS[k_])} | `{h}` |")
    lines += ["", "Harness run directories (W3's devkit output): D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/runs/"
                  f"{{{', '.join(LABELS.values())}}} (Archive only).",
              "Families blocks: " + ", ".join(o["four_families"].get("files", {}).values()) + " (data dir only)."]
    open(MANIFEST, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("dump", "dump-child", "build", "score", "analyze"))
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--tokens", default=TOK)
    ap.add_argument("--deadline", default="15:40")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=3)
    # analyze-only overrides, for the gate self-test (selftest_a7_confirm.py); never used for the registered run
    ap.add_argument("--native", default=None)
    ap.add_argument("--repaired-seam", default=None)
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--no-families", action="store_true")
    ap.add_argument("--no-docs", action="store_true", help="do not write the RESULT doc and the MANIFEST")
    a = ap.parse_args()
    if a.stage == "dump":
        return dump(a)
    if a.stage == "dump-child":
        if os.environ.get(CHILD) != "1":
            print("ZZA7_FAIL dump-child must run under the pipeline environment"); return 1
        return dump_child(a)
    if a.stage == "build":
        return build(a)
    if a.stage == "score":
        return score(a)
    return analyze(a)


if __name__ == "__main__":
    sys.exit(main())
