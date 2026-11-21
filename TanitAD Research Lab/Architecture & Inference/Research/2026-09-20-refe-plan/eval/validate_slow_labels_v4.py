#!/usr/bin/env python3
"""VALIDITY of measure 5 (label_version 4: slowed copies in the on-policy sets) against the NAVSIM harness. Dev box.

QUESTION. Does `refe/onpolicy_label_v4.py` -- the labeller that would run on the pod -- give slowed copies of REFe's
proposals the labels NAVSIM's harness gives the SAME trajectories? Drivable area and comfort are NAVSIM's own code in
the labeller and must agree 100 %. The teacher's collision / TTC / progress / driving direction come from a different
simulator and are EXPECTED to differ: the question is by how much, and whether the disagreement is WORSE on slowed
copies than on the originals -- a label that is systematically wrong about slow plans would teach the scorer the wrong
thing about exactly what measure 5 exists to teach.

DATA. W3's 200 navtest tokens (A1_sub200_tokens.json), snapshot 015: REFe's native 20-pose proposals from
stop_candidate_probe.py's dump of the forward the E-6 table was built from (re-checked here: to_navsim of them == the
table's proposals, bit) and the table's stored logits. Harness scores of the SAME trajectories: the E-6 table (all 64
originals), eval/slow_copies.py's harness runs of the copies (per scorer rank `r00..` and per slot `p00..`, factors 0.75
and 0.5; its copies are gated here bit-for-bit against refe/slow_copies.py), and STOP = zeros re-scored through the REFe
path (stop_candidate_probe.py control C_f).
⚠️ DECLARED SUBSTITUTION: the labeller's EP is relative to the TEACHER's advance; navtest has no DriveRL teacher plan,
so the HUMAN logged future stands in as the `teacher` candidate (EP = advance / human advance). The props are written
at full float32 precision (the pod's dump rounds to 5 decimals) so the labelled copies ARE the harness-scored ones.

STAGES (each ends with a ZZ marker; data in <WD>, the banked result in eval/raw/e6_sub200_ep015/):
  prep     props chunks (4 tokens each) in onpolicy_dump.py's line format (+ `logits`, as --emit-logits writes) + gates
  label    --procs N copies of the REAL CLI `onpolicy_label_v4.py --slow-copies --slow-frac 1.0` on that queue
  ref      the first chunk through the REAL v3 worker (`onpolicy_label.py`) and the v4 CLI with the flag OFF
  mutate   --mutate label_source (the self-check must refuse every set) and label_source_nocheck (written)
  reverify the FINAL labeller file on chunk 0: flag OFF == the real v3 lines, flag ON == the main arm's lines
  wide     NAVSIM drivable area + comfort by the labeller's own call on copies of ALL 64 proposals x {0.75, 0.5} + STOP
  (label --arm <name> "--label-args=<flags>" runs another labelling arm on its own queue, e.g. sets_ncaf with
   --nc-at-fault; analyze --arm <name> reads it)
  analyze  every label read back THROUGH THE TRAINER'S OWN LOADER (train.OnPolicyBank) and joined to the harness

    python eval/validate_slow_labels_v4.py prep
    python eval/validate_slow_labels_v4.py label --procs 5
    python eval/validate_slow_labels_v4.py ref
    python eval/validate_slow_labels_v4.py mutate
    python eval/validate_slow_labels_v4.py wide
    python eval/validate_slow_labels_v4.py analyze
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REFE = os.path.join(PKG, "refe")
sys.path.insert(0, HERE)
sys.path.insert(0, REFE)
import eval_checkpoint as EC  # noqa: E402  the venv + env the seam and the harness ran with

NAME = "sub200_ep015"
TOKENS = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
          "A1_sub200_tokens.json")
TABLE = f"{EC.DATA}/proptable/{NAME}/table.npz"
STOP_DUMP = f"{EC.DATA}/proptable/{NAME}/stop_candidate_dump.npz"
SLOW_BUILD = f"{EC.DATA}/proptable/{NAME}/slow_copies/build.npz"
SLOW_RANKS = f"{EC.DATA}/proptable/{NAME}/slow_copies/ranks.npz"
STOP_CSV = f"{EC.DATA}/score/refe_{NAME}_stopzeros/refe_{NAME}_stopzeros.csv"
WD = f"{EC.DATA}/proptable/{NAME}/v4_labels"
Q = f"{WD}/queue"
SETS = f"{WD}/sets"
LOGS = f"{WD}/logs"
PREP_META = f"{WD}/prep_meta.json"
WIDE = f"{WD}/wide.npz"
OUT = os.path.join(HERE, "raw", f"e6_{NAME}", "v4_label_validity.json")
CKPT_STEP = 15                     # a stand-in id for the dump's checkpoint step: snapshot 015
FACTORS = (0.75, 0.5)
SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
       "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
HEAD = ("NC", "DAC", "EP", "TTC", "C", "DDC")        # == train.ScorerBank.COMPONENTS order == SUB order
V4 = os.path.join(REFE, "onpolicy_label_v4.py")
V3 = os.path.join(REFE, "onpolicy_label.py")


def ftag(f):
    return f"f{int(round(f * 100)):03d}"


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def worker_env(extra=None):
    """The seam's environment (EC.env_driverl) with the POD's thread discipline: one thread per labelling process."""
    e = EC.env_driverl()
    e.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": "0"})
    e.update(extra or {})
    return e


def run_cli(cmd, log, env):
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.Popen(cmd, cwd=REFE, env=env, stdout=f, stderr=subprocess.STDOUT)


def tail(log, n=3):
    try:
        return open(log, encoding="utf-8", errors="replace").read().strip().splitlines()[-n:]
    except OSError:
        return []


# ------------------------------------------------------------------------------------------------ prep
def human_future(sc, n=20, dt=0.2):
    """The logged ego future at dt..n*dt in the t0 rear-axle frame (x fwd, y left, heading) -- the EP reference."""
    e0 = sc.get_ego_state_at_iteration(0)
    fut = list(sc.get_ego_future_trajectory(iteration=0, time_horizon=n * dt, num_samples=n))
    if len(fut) != n:
        raise ValueError(f"{len(fut)} future samples, need {n}")
    x0, y0, h0 = e0.rear_axle.x, e0.rear_axle.y, e0.rear_axle.heading
    c, s = math.cos(h0), math.sin(h0)
    out = np.zeros((n, 3))
    ts = []
    for k, e in enumerate(fut):
        dx, dy = e.rear_axle.x - x0, e.rear_axle.y - y0
        out[k] = (c * dx + s * dy, -s * dx + c * dy, wrap(e.rear_axle.heading - h0))
        ts.append(e.time_point.time_s - e0.time_point.time_s)
    return out, ts


def prep(a) -> int:
    import navtrain_scenarios as NS
    import refe_navtest_seam as SEAM
    T = np.load(TABLE)
    D = np.load(STOP_DUMP, allow_pickle=False)
    tok = [str(t) for t in T["token"]]
    if tok != [str(t) for t in D["token"]]:
        print("ZZV4VAL_FAIL prep token order"); return 1
    traj = D["traj"].astype(np.float32)                                        # [200, 64, 20, 3]
    p8 = np.stack([[SEAM.to_navsim(traj[i, j]).astype(np.float32) for j in range(64)] for i in range(len(tok))])
    d_tab = float(np.abs(p8.astype(np.float64) - T["proposals"].astype(np.float64)).max())
    print(f"  gate: to_navsim(dump traj) vs table.proposals max |d| {d_tab}", flush=True)
    if d_tab != 0.0:
        print("ZZV4VAL_FAIL prep proposals"); return 1
    tl = json.load(open(TOKENS, encoding="utf-8"))["token_log"]
    dbs = NS.index_dbs()
    by_log: dict = {}
    for i, t in enumerate(tok):
        by_log.setdefault(tl[t], []).append(i)
    rows, miss, dts = {}, [], []
    t0 = time.time()
    for lg, idx in sorted(by_log.items()):
        scs = {sc.scenario_name: sc for sc in NS.build_scenarios_for_log(dbs[lg], [tok[i] for i in idx])}
        for i in idx:
            sc = scs.get(tok[i])
            if sc is None:
                miss.append(tok[i]); continue
            hum, ts = human_future(sc)
            dts.append(max(abs(t_ - 0.2 * (k + 1)) for k, t_ in enumerate(ts)))
            rows[i] = {"kind": "onpolicy_props", "log_name": lg, "token": tok[i], "step": 0, "rank": 0,
                       "ckpt_step": CKPT_STEP, "teacher": hum.tolist(), "aug": None,
                       "props": traj[i].astype(np.float64).tolist(),
                       "logits": T["logits"][i].astype(np.float64).tolist()}
    print(f"  scenarios: {len(rows)}/{len(tok)} ({len(miss)} missing) in {time.time() - t0:.0f} s; human-future "
          f"time grid max |t - 0.2k| {max(dts) if dts else float('nan'):.4f} s", flush=True)
    if miss or not dts or max(dts) > 0.02:
        print("ZZV4VAL_FAIL prep scenarios"); return 1
    if os.path.exists(Q):
        shutil.rmtree(Q)
    os.makedirs(Q)
    order = [i for _lg, idx in sorted(by_log.items()) for i in idx]
    n_chunks = 0
    for c in range(0, len(order), a.chunk):
        out = os.path.join(Q, f"props_r0_{CKPT_STEP:06d}_{c // a.chunk:010d}.jsonl")
        with open(out + ".tmp", "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(rows[i]) + "\n" for i in order[c:c + a.chunk]))
        os.replace(out + ".tmp", out)
        n_chunks += 1
    json.dump({"tokens": len(rows), "chunks": n_chunks, "chunk": a.chunk, "table_sha256": sha256(TABLE),
               "stop_dump_sha256": sha256(STOP_DUMP), "to_navsim_vs_table_max_abs": d_tab,
               "human_future_grid_max_abs_s": max(dts), "ckpt_step_standin": CKPT_STEP,
               "at_local": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(PREP_META, "w"), indent=1)
    print(f"ZZV4VAL_PREP_OK {len(rows)} tokens in {n_chunks} chunks -> {Q}")
    return 0


# ------------------------------------------------------------------------------------------------ label
def arm_dirs(arm: str) -> tuple:
    """(queue, sets) of a labelling arm: 'sets' is the main v4 arm; any other arm gets its own queue, re-made from the
    prepped chunks (in whatever state the main arm left them), and its own sets directory."""
    return (Q, SETS) if arm == "sets" else (f"{WD}/queue_{arm}", f"{WD}/{arm}")


def label(a) -> int:
    q, sets = arm_dirs(a.arm)
    if a.arm != "sets" and not os.path.exists(q):
        os.makedirs(q)
        n = 0
        while True:
            hit = sorted(glob.glob(os.path.join(Q, f"props_r0_{CKPT_STEP:06d}_{n:010d}.jsonl*")))
            if not hit:
                break
            shutil.copyfile(hit[0], os.path.join(q, f"props_r0_{CKPT_STEP:06d}_{n:010d}.jsonl"))
            n += 1
        print(f"  arm {a.arm}: fresh queue of {n} chunks -> {q}", flush=True)
    os.makedirs(sets, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    t0 = time.time()
    procs = []
    for w in range(a.procs):
        cmd = [EC.DRIVERL_PY, V4, "--queue", q, "--out", sets, "--rank", "0", "--worker", f"vw{w}", "--once",
               "--slow-copies", "--slow-factors", "0.75,0.5", "--slow-frac", "1.0"] + a.label_args.split()
        procs.append((w, run_cli(cmd, os.path.join(LOGS, f"label_{a.arm}_vw{w}.log"), worker_env())))
        time.sleep(7)                    # stagger the first claims: the exFAT rename race (see onpolicy_label_v4)
    rcs = {w: p.wait() for w, p in procs}
    left = glob.glob(os.path.join(q, "props_r0_*.jsonl")) + glob.glob(os.path.join(q, "props_r0_*.jsonl.vw*"))
    for w in rcs:
        print(f"  vw{w}: rc {rcs[w]} | " + " | ".join(tail(os.path.join(LOGS, f"label_{a.arm}_vw{w}.log"), 2)),
              flush=True)
    ok = all(v == 0 for v in rcs.values()) and not left
    print(f"ZZV4VAL_LABEL_{'OK' if ok else 'FAIL'} arm {a.arm}: {a.procs} procs in {time.time() - t0:.0f} s; "
          f"unclaimed/unfinished chunks {len(left)}")
    return 0 if ok else 1


def _first_chunks(n):
    """Chunks 0..n-1 of the prepped queue in WHATEVER state the label stage left them (pending, claimed, done)."""
    out = []
    for c in range(n):
        hit = sorted(glob.glob(os.path.join(Q, f"props_r0_{CKPT_STEP:06d}_{c:010d}.jsonl*")))
        if hit:
            out.append(hit[0])
    return out


def _arm(name, cmd_extra, chunks, env_extra=None, cli=V4):
    qd, od = f"{WD}/{name}/queue", f"{WD}/{name}/sets"
    for d in (qd, od):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    for c in chunks:
        base = os.path.basename(c).split(".jsonl")[0] + ".jsonl"
        shutil.copyfile(c, os.path.join(qd, base))
    cmd = [EC.DRIVERL_PY, cli, "--queue", qd, "--out", od, "--rank", "0", "--worker", name, "--once"] + cmd_extra
    return run_cli(cmd, os.path.join(LOGS, f"{name}.log"), worker_env(env_extra))


def ref(a) -> int:
    chunks = _first_chunks(a.ref_chunks)
    if not chunks:
        print("ZZV4VAL_FAIL ref: no prepped chunks"); return 1
    t0 = time.time()
    arms = {"ref_v3": _arm("ref_v3", [], chunks, cli=V3), "ref_v4off": _arm("ref_v4off", [], chunks)}
    rcs = {k: p.wait() for k, p in arms.items()}
    print(f"  {rcs} in {time.time() - t0:.0f} s", flush=True)
    ok = all(v == 0 for v in rcs.values())
    print(f"ZZV4VAL_REF_{'OK' if ok else 'FAIL'} {len(chunks)} chunks")
    return 0 if ok else 1


def mutate(a) -> int:
    chunks = _first_chunks(a.ref_chunks)
    t0 = time.time()
    env = {"REFE_V4_ALLOW_MUTATE": "1"}
    arms = {"mut_label_source": _arm("mut_label_source", ["--slow-copies", "--slow-factors", "0.75,0.5", "--slow-frac", "1.0", "--mutate",
                                                          "label_source"], chunks, env),
            "mut_nocheck": _arm("mut_nocheck", ["--slow-copies", "--slow-factors", "0.75,0.5", "--slow-frac", "1.0", "--mutate",
                                                "label_source_nocheck"], chunks, env)}
    rcs = {k: p.wait() for k, p in arms.items()}
    print(f"  {rcs} in {time.time() - t0:.0f} s", flush=True)
    ok = all(v == 0 for v in rcs.values())
    print(f"ZZV4VAL_MUTATE_{'OK' if ok else 'FAIL'} {len(chunks)} chunks")
    return 0 if ok else 1


def reverify(a) -> int:
    """The FINAL labeller file reproduces what was validated: on chunk 0, flag OFF must equal the REAL v3 worker's lines
    and flag ON must equal the main arm's lines (timing / labeller fields excluded) -- the main arm ran on an earlier
    revision of onpolicy_label_v4.py, so this is what licenses shipping the current one."""
    chunks = _first_chunks(a.ref_chunks)
    t0 = time.time()
    arms = {"final_v4off": _arm("final_v4off", [], chunks),
            "final_v4on": _arm("final_v4on", ["--slow-copies", "--slow-factors", "0.75,0.5", "--slow-frac", "1.0"],
                                 chunks)}
    rcs = {k: p.wait() for k, p in arms.items()}
    v3 = {r["token"]: r for r in _lines(f"{WD}/ref_v3/sets")}
    main_ = {r["token"]: r for r in _lines(SETS)}
    off = {r["token"]: r for r in _lines(f"{WD}/final_v4off/sets")}
    on = {r["token"]: r for r in _lines(f"{WD}/final_v4on/sets")}
    main_aux = {r["token"]: r for r in _lines(SETS, "onpolicy_slowaux")}
    on_aux = {r["token"]: r for r in _lines(f"{WD}/final_v4on/sets", "onpolicy_slowaux")}
    strip_aux = lambda r: {k: v for k, v in r.items() if k != "at"}                  # noqa: E731
    res = {"rc": rcs, "labeller_sha256": sha256(V4), "tokens": sorted(v3),
           "flag_off_equals_v3": bool(v3) and all(t in off and json.dumps(_strip(off[t])) == json.dumps(_strip(v3[t]))
                                                  for t in v3),
           "flag_on_equals_main_arm": bool(v3) and all(t in on and json.dumps(_strip(on[t])) == json.dumps(_strip(main_[t]))
                                                        for t in v3),
           "flag_on_aux_equals_main_arm": bool(v3) and all(
               t in on_aux and json.dumps(strip_aux(on_aux[t])) == json.dumps(strip_aux(main_aux[t])) for t in v3),
           "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(HERE, "raw", f"e6_{NAME}", "v4_final_reverify.json"), "w"), indent=1)
    ok = all(v == 0 for v in rcs.values()) and res["flag_off_equals_v3"] and res["flag_on_equals_main_arm"] \
        and res["flag_on_aux_equals_main_arm"]
    print(json.dumps(res, indent=1))
    print(f"ZZV4VAL_REVERIFY_{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ wide
def wide(a) -> int:
    import navsim_dac as ND
    import navtrain_scenarios as NS
    import onpolicy_label_v4 as L4
    import refe_navtest_seam as SEAM
    T = np.load(TABLE)
    D = np.load(STOP_DUMP, allow_pickle=False)
    B = np.load(SLOW_BUILD, allow_pickle=False)
    tok = [str(t) for t in T["token"]]
    if [str(t) for t in B["token"]] != tok:
        print("ZZV4VAL_FAIL wide build token order"); return 1
    tl = json.load(open(TOKENS, encoding="utf-8"))["token_log"]
    dbs = NS.index_dbs()
    n = len(tok)
    R = {k: np.full((n, 64), np.nan) for k in ("dac_orig", "c_orig") + tuple(f"{g}_{ftag(f)}" for f in FACTORS
                                                                           for g in ("dac", "c"))}
    R.update({"dac_stop": np.full(n, np.nan), "c_stop": np.full(n, np.nan)})
    gate = {"copy_vs_build_native_max_abs": 0.0, "copy_vs_build_navsim_max_abs": 0.0}
    by_log: dict = {}
    for i, t in enumerate(tok):
        by_log.setdefault(tl[t], []).append(i)
    t0 = time.time()
    for li, (lg, idx) in enumerate(sorted(by_log.items())):
        scs = {sc.scenario_name: sc for sc in NS.build_scenarios_for_log(dbs[lg], [tok[i] for i in idx])}
        for i in idx:
            sc = scs.get(tok[i])
            if sc is None:
                continue
            ego = sc.get_ego_state_at_iteration(0)
            P = D["traj"][i].astype(np.float64)
            dv, cf = ND.navsim_dac_and_comfort(P, ego, sc.map_api, grid="refe20")
            R["dac_orig"][i], R["c_orig"][i] = dv, cf
            for f in FACTORS:
                C = np.stack([L4.copy_traj(P, j, f) for j in range(64)])
                c32 = C.astype(np.float32)
                gate["copy_vs_build_native_max_abs"] = max(gate["copy_vs_build_native_max_abs"], float(
                    np.abs(c32.astype(np.float64) - B[f"nat_{ftag(f)}"][i].astype(np.float64)).max()))
                nav = np.stack([SEAM.to_navsim(x).astype(np.float32) for x in c32])
                gate["copy_vs_build_navsim_max_abs"] = max(gate["copy_vs_build_navsim_max_abs"], float(
                    np.abs(nav.astype(np.float64) - B[f"nav_{ftag(f)}"][i].astype(np.float64)).max()))
                dv, cf = ND.navsim_dac_and_comfort(C, ego, sc.map_api, grid="refe20")
                R[f"dac_{ftag(f)}"][i], R[f"c_{ftag(f)}"][i] = dv, cf
            stop = L4.copy_traj(P, -1, 0.0)[None]
            dv, cf = ND.navsim_dac_and_comfort(stop, ego, sc.map_api, grid="refe20")
            R["dac_stop"][i], R["c_stop"][i] = dv[0], cf[0]
        if (li + 1) % 10 == 0:
            print(f"    wide: {li + 1}/{len(by_log)} logs, {time.time() - t0:.0f} s", flush=True)
    np.savez(WIDE, token=np.array(tok), **R, gate=np.array(json.dumps(gate)))
    ok = gate["copy_vs_build_native_max_abs"] == 0.0 and gate["copy_vs_build_navsim_max_abs"] == 0.0
    print(f"  gate {gate}")
    print(f"ZZV4VAL_WIDE_{'OK' if ok else 'FAIL'} {int(np.isfinite(R['dac_orig']).all(1).sum())} tokens in "
          f"{time.time() - t0:.0f} s")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ analyze
def read_csv(p):
    rows = {}
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["token"] == "average":
                continue
            rows[r["token"]] = (r["valid"] == "True", tuple(float(r[c]) for c in SUB), float(r["score"]))
    return rows


def harness_copy_csvs():
    """{(factor, 'r'|'p', index): {token: (valid, subs, pdms)}} for every copy csv that exists (PASS or not is
    checked by `valid` per row)."""
    out = {}
    for f in FACTORS:
        for kind in ("r", "p"):
            for p in glob.glob(f"{EC.DATA}/score/refe_{NAME}_{ftag(f)}_{kind}[0-9][0-9]/refe_{NAME}_{ftag(f)}_{kind}*.csv"):
                idx = int(os.path.basename(p).split("_")[-1][1:3])
                out[(f, kind, idx)] = read_csv(p)
    return out


def trainer_targets(t: dict) -> list:
    """What train.OnPolicyBank feeds the loss for one target dict: ScorerBank.components + the NAVSIM drivable-area
    override of train.py:457-460 (every v3/v4 target carries `navsim_dac.violation`). Cross-checked below against the
    bank's own arrays for every served slot."""
    import train as TR
    c = TR.ScorerBank.components(t)
    if isinstance(t, dict) and "navsim_dac.violation" in t:
        c[1] = 1.0 - min(max(float(t["navsim_dac.violation"]), 0.0), 1.0)
    return c


def _lines(d, kind="onpolicy_set"):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "onpolicy_*.jsonl" if kind == "onpolicy_set" else "slowaux_*.jsonl"))):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == kind:
                    out.append(r)
    return out


def _strip(r):
    return {k: v for k, v in r.items() if k not in ("sec", "at", "labeller")}


def agree_table(lab, har, groups):
    """lab/har: dict group -> list of (label value, harness value). Binary components compared as pass/fail."""
    return {g: v for g, v in ((g, groups[g](lab.get(g, []), har.get(g, []))) for g in groups)}


def _bin_stats(pairs, lab_ok=lambda x: x >= 0.5, har_ok=lambda x: x >= 0.999):
    if not pairs:
        return {"n": 0}
    a = np.asarray([lab_ok(x) for x, _ in pairs])
    b = np.asarray([har_ok(y) for _, y in pairs])
    return {"n": len(pairs), "agree": round(float((a == b).mean()), 6),
            "label_pass": round(float(a.mean()), 4), "harness_pass": round(float(b.mean()), 4),
            "label_fail_harness_pass": int((~a & b).sum()), "label_pass_harness_fail": int((a & ~b).sum()),
            "label_mean": round(float(np.mean([x for x, _ in pairs])), 4),
            "harness_mean": round(float(np.mean([y for _, y in pairs])), 4)}


def _cont_stats(pairs):
    if not pairs:
        return {"n": 0}
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")
    return {"n": int(len(x)), "label_mean": round(float(x.mean()), 4), "harness_mean": round(float(y.mean()), 4),
            "mae": round(float(np.abs(x - y).mean()), 4), "pearson": round(r, 4)}


def _auc(scores, labels):
    s, l_ = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[l_], s[~l_]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def _extra(acc, t, lab, hs):
    """EP where NAVSIM does not zero it (its multiplier NC x DAC x DDC is 1), and -- in an arm that replaced the NC
    label -- the RAW teacher collision beside it, on the same slots."""
    if hs[0] >= 0.999 and hs[1] >= 0.999 and hs[5] >= 0.999:
        acc["EP_ok"].append((lab[2], hs[2]))
    raw = t.get("teacher_collision.NuPlanCollision.info")
    if raw is not None:
        acc["NC_raw"].append((1.0 - min(max(float(raw), 0.0), 1.0), hs[0]))


def v1_agg(v):
    """NAVSIM v1 PDMS on a [.., 6] vector (NC, DAC, EP, TTC, C, DDC): NC x DAC x (5 EP + 5 TTC + 2 C) / 12."""
    v = np.asarray(v, dtype=float)
    return v[..., 0] * v[..., 1] * (5 * v[..., 2] + 5 * v[..., 3] + 2 * v[..., 4]) / 12.0


def analyze(a) -> int:
    import train as TR
    global SETS, OUT
    if a.arm != "sets":
        SETS = arm_dirs(a.arm)[1]
        OUT = os.path.join(HERE, "raw", f"e6_{NAME}", f"v4_label_validity_{a.arm}.json")
    T = np.load(TABLE)
    tok = [str(t) for t in T["token"]]
    ix = {t: i for i, t in enumerate(tok)}
    RK = np.load(SLOW_RANKS)
    top = RK["top"]
    res: dict = {"_label": "VALIDITY of label_version 4 (measure 5) on W3's 200 navtest tokens, snapshot 015 -- "
                           "EXPLORATORY instrument check, dev box; no model was trained",
                 "script": os.path.abspath(__file__), "labeller": V4, "labeller_sha256": sha256(V4),
                 "v3_labeller_sha256": sha256(V3), "slow_copies_sha256": sha256(os.path.join(REFE, "slow_copies.py")),
                 "navsim_dac_sha256": sha256(os.path.join(REFE, "navsim_dac.py")),
                 "table": TABLE, "table_sha256": sha256(TABLE), "prep": json.load(open(PREP_META)),
                 "ep_reference": "HUMAN logged future (declared substitution for the DriveRL teacher plan)",
                 "arm": a.arm, "sets_dir": SETS, "at_local": time.strftime("%Y-%m-%dT%H:%M:%S")}
    # ---- 1. every label read back through the TRAINER'S OWN LOADER
    bank = TR.OnPolicyBank(SETS, 64, 20)
    lines = _lines(SETS)
    aux = {(r["token"]): r for r in _lines(SETS, "onpolicy_slowaux")}
    res["trainer_loader"] = {"sets_loaded": len(bank.by), "set_lines": bank.n_rows, "incomplete": bank.n_incomplete,
                             "navsim_dac_sets": bank.n_navsim_dac, "bad_lines": bank.n_bad_lines,
                             "label_versions": sorted({int(r["label_version"]) for r in lines})}
    Hc = harness_copy_csvs()
    stop_h = read_csv(STOP_CSV)
    res["harness_copy_csvs"] = sorted(f"{ftag(f)}_{k}{i:02d}" for f, k, i in Hc)
    groups = ("orig_src", "orig_kept", "orig_dropped", "copy_f075", "copy_f050", "stop")
    P: dict = {g: {h: [] for h in HEAD + ("PDMS", "NC_raw", "EP_ok")} for g in groups}   # (label, harness) pairs
    raw_adv: dict = {g: [] for g in groups}
    pair_dir: dict = {f: [] for f in FACTORS}                              # (d label agg, d harness pdms) copy - src
    nav_agree = {"DAC": [], "C": []}                                       # (group, label ok, harness ok)
    mism_loader = rank_mismatch = no_harness = 0
    mut = {"copy_slots": 0, "DAC_disagree": 0, "C_disagree": 0, "either_disagree": 0}
    per_token_src = {}
    for r in lines:
        key = (r["log_name"], r["token"], int(r["step"]), int(r["rank"]))
        e = bank.by.get(key)
        if e is None:
            continue
        tg_bank = e[2]                                                     # [64, 6] exactly what the loss reads
        i = ix[r["token"]]
        sl = r.get("slow") or {}
        slots = sl.get("slots", [])
        srcs, facs = sl.get("src", []), sl.get("factor", [])
        slot_of = {s: (sv, fv) for s, sv, fv in zip(slots, srcs, facs)}
        srcset = {s for s in srcs if s >= 0}
        per_token_src[r["token"]] = sorted(srcset)
        top_i = [int(x) for x in top[i]]
        if sorted(srcset) != sorted(top_i[:len(srcset)]):
            rank_mismatch += 1
        tvals = [trainer_targets(t) for t in r["targets"]]
        if not np.allclose(np.asarray(tvals, np.float32), tg_bank, atol=0, rtol=0):
            mism_loader += 1
        for slot in range(64):
            lab = tvals[slot]
            t = r["targets"][slot]
            if slot in slot_of:
                s, f = slot_of[slot]
                if f == 0.0:
                    h = stop_h.get(r["token"])
                    g = "stop"
                else:
                    g = f"copy_{ftag(f)}"
                    rr = top_i.index(s) if s in top_i else None
                    h = (Hc.get((f, "r", rr), {}).get(r["token"]) if rr is not None else None) or \
                        Hc.get((f, "p", s), {}).get(r["token"])
                if h is None or not h[0]:
                    no_harness += 1
                    continue
                hs, hp = h[1], h[2]
                # the MUTATION ARM (emulated exactly; mutate stage proves the emulation on real output): the slot
                # carries the SOURCE original's labels instead of the copy's
                if s >= 0:
                    ts = r["targets"][s]
                    mdac = (float(ts["navsim_dac.violation"]) == 0.0) == (hs[1] >= 0.999)
                    mc = (float(ts["navsim_comfort"]) >= 0.5) == (hs[4] >= 0.999)
                    mut["copy_slots"] += 1
                    mut["DAC_disagree"] += int(not mdac)
                    mut["C_disagree"] += int(not mc)
                    mut["either_disagree"] += int(not (mdac and mc))
                    pair_dir[f].append((v1_agg(lab) - v1_agg(tvals[s]), hp - T["pdms"][i, s],
                                        lab[2] - tvals[s][2], hs[2] - T["sub"][i, s, 2]))
            else:
                g = "orig_src" if slot in srcset else "orig_kept"
                if not T["valid"][i, slot]:
                    no_harness += 1
                    continue
                hs, hp = tuple(T["sub"][i, slot]), float(T["pdms"][i, slot])
            for k, hn in enumerate(HEAD):
                P[g][hn].append((lab[k], hs[k]))
            P[g]["PDMS"].append((float(v1_agg(lab)), hp))
            _extra(P[g], t, lab, hs)
            raw_adv[g].append((t.get("progress.advance_m"), hs[2]))
            nav_agree["DAC"].append((g, float(t["navsim_dac.violation"]) == 0.0, hs[1] >= 0.999))
            nav_agree["C"].append((g, float(t["navsim_comfort"]) >= 0.5, hs[4] >= 0.999))
        ax = aux.get(r["token"])
        if ax is not None:
            for slot, t in zip(ax["slots"], ax["dropped_targets"]):
                if not T["valid"][i, slot]:
                    continue
                lab = trainer_targets(t)
                hs, hp = tuple(T["sub"][i, slot]), float(T["pdms"][i, slot])
                for k, hn in enumerate(HEAD):
                    P["orig_dropped"][hn].append((lab[k], hs[k]))
                P["orig_dropped"]["PDMS"].append((float(v1_agg(lab)), hp))
                _extra(P["orig_dropped"], t, lab, hs)
                raw_adv["orig_dropped"].append((t.get("progress.advance_m"), hs[2]))
                nav_agree["DAC"].append(("orig_dropped", float(t["navsim_dac.violation"]) == 0.0, hs[1] >= 0.999))
                nav_agree["C"].append(("orig_dropped", float(t["navsim_comfort"]) >= 0.5, hs[4] >= 0.999))
    res["gates"] = {"trainer_loader_equals_line_components": {"sets_mismatched": mism_loader, "pass": mism_loader == 0},
                    "sources_are_the_harness_top_ranks": {"sets_mismatched": rank_mismatch, "pass": rank_mismatch == 0},
                    "labelled_slots_without_a_valid_harness_row": no_harness}
    # ---- 2. NAVSIM's own labels: must agree 100 %
    nav = {}
    for comp, rows in nav_agree.items():
        by = {}
        for g in groups:
            sel = [(x, y) for gg, x, y in rows if gg == g]
            if sel:
                a_ = np.asarray([x for x, _ in sel]); b_ = np.asarray([y for _, y in sel])
                by[g] = {"n": len(sel), "agree": round(float((a_ == b_).mean()), 6),
                         "label_fail_harness_pass": int((~a_ & b_).sum()), "label_pass_harness_fail": int((a_ & ~b_).sum())}
        tot = [(x, y) for _, x, y in rows]
        a_ = np.asarray([x for x, _ in tot]); b_ = np.asarray([y for _, y in tot])
        nav[comp] = {"all": {"n": len(tot), "agree": round(float((a_ == b_).mean()), 6) if tot else None,
                             "disagree": int((a_ != b_).sum())}, "by_group": by}
    res["navsim_components_production_worker"] = nav
    # ---- 3. the teacher's components vs NAVSIM, copies vs originals
    teach = {}
    for g in groups:
        if not P[g]["NC"]:
            continue
        teach[g] = {
            "NC": _bin_stats(P[g]["NC"]),
            "TTC": dict(_bin_stats(P[g]["TTC"]), auc_label_vs_harness=round(_auc([x for x, _ in P[g]["TTC"]],
                                                                                  [y >= 0.999 for _, y in P[g]["TTC"]]), 4)),
            "EP": _cont_stats(P[g]["EP"]),
            "EP_where_harness_NC_DAC_DDC_pass": _cont_stats(P[g]["EP_ok"]),
            "NC_raw_teacher_collision": _bin_stats(P[g]["NC_raw"]) if P[g]["NC_raw"] else None,
            "EP_raw_advance_m_vs_harness_EP": _cont_stats([(x, y) for x, y in raw_adv[g] if x is not None]),
            "DDC": _bin_stats(P[g]["DDC"]),
            "DAC_trainer_component": _bin_stats(P[g]["DAC"]),
            "C_trainer_component": _bin_stats(P[g]["C"]),
            "PDMS_v1_of_labels": _cont_stats(P[g]["PDMS"])}
    res["teacher_components_vs_harness"] = teach
    # ---- 4. does the label move the RIGHT WAY when a plan is slowed? (copy minus its own source, same token)
    dirn = {}
    for f in FACTORS:
        v = np.asarray(pair_dir[f], dtype=float) if pair_dir[f] else np.zeros((0, 4))
        if len(v):
            nz = np.abs(v[:, 1]) > 1e-9
            dirn[ftag(f)] = {"pairs": int(len(v)), "harness_pdms_changed": int(nz.sum()),
                             "sign_agree_where_harness_changed": round(float((np.sign(v[nz, 0]) == np.sign(v[nz, 1])).mean()), 4) if nz.any() else None,
                             "mean_delta_label_agg": round(float(v[:, 0].mean()), 4),
                             "mean_delta_harness_pdms": round(float(v[:, 1].mean()), 4),
                             "harness_says_copy_better": int((v[:, 1] > 1e-9).sum()),
                             "labels_say_copy_better": int((v[:, 0] > 1e-9).sum()),
                             "mean_delta_label_EP": round(float(v[:, 2].mean()), 4),
                             "mean_delta_harness_EP": round(float(v[:, 3].mean()), 4)}
    res["copy_minus_source_direction"] = dirn
    # ---- 4b. THE LABEL-RANKING CEILING: a scorer that learned these labels PERFECTLY ranks each token's candidates by
    # the labels' own v1 aggregate. Per token, the fraction of candidate pairs (harness PDMS differing) that this
    # ordering gets right -- over pairs involving an extra member (copy / STOP), per configuration, and over pairs of
    # originals only (what the v3 labels already teach). It bounds what measure 5 can teach from above.
    res["label_ranking_ceiling"] = label_ceiling(lines, aux, T, Hc, stop_h, top, ix)
    # ---- 5. the mutation arm: the copy slots labelled with their SOURCE's labels
    detected = mut["either_disagree"] > 0
    res["mutation_label_source"] = dict(mut, detected=detected,
                                        note="emulated from the written lines (the source's targets are in the same "
                                             "set); the mutate stage's label_source_nocheck output proves the "
                                             "emulation equals a real mutated labeller")
    # ---- 6. the reference arms (v3 worker, flag OFF), and the real mutation runs
    res["ref"] = ref_checks(lines)
    res["mutate_runs"] = mutate_checks(lines)
    # ---- 7. the wide NAVSIM arm (no teacher): copies of all 64 x 2 factors + STOP vs every harness csv present
    res["wide"] = wide_checks(Hc, stop_h, T, top) if os.path.exists(WIDE) else None
    ok_nav = all(v["all"]["disagree"] == 0 for v in nav.values()) and nav["DAC"]["all"]["n"] > 0
    ok_w = res["wide"] is None or (res["wide"]["DAC_disagree"] == 0 and res["wide"]["C_disagree"] == 0)
    res["verdict"] = {
        "navsim_components_100pct": ok_nav, "wide_navsim_100pct": ok_w if res["wide"] else None,
        "mutation_detected": detected,
        "trainer_loader_gate": mism_loader == 0,
        "ref_flag_off_identical_to_v3": res["ref"].get("flag_off_identical_to_v3"),
        "ref_originals_identical_with_copies": res["ref"].get("originals_identical_with_copies_present"),
        "selfcheck_refuses_label_source": res["mutate_runs"].get("selfcheck_refused_all"),
        "nocheck_run_equals_emulation": res["mutate_runs"].get("nocheck_equals_emulation")}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps(res["verdict"], indent=1))
    print(f"  NAVSIM DAC agree {nav['DAC']['all']}  comfort {nav['C']['all']}")
    for g, v in teach.items():
        print(f"  {g:13s} NC agree {v['NC'].get('agree')} (label {v['NC'].get('label_mean')} vs harness "
              f"{v['NC'].get('harness_mean')}) | TTC agree {v['TTC'].get('agree')} AUC {v['TTC'].get('auc_label_vs_harness')} "
              f"| EP mae {v['EP'].get('mae')} r {v['EP'].get('pearson')} | DDC agree {v['DDC'].get('agree')} "
              f"| n {v['NC'].get('n')}")
    print(f"  mutation (label the source): {mut}  -> {'DETECTED' if detected else 'NOT DETECTED'}")
    allok = ok_nav and detected and mism_loader == 0 and ok_w
    print(f"ZZV4VAL_{'OK' if allok else 'CHECK'} -> {OUT}")
    return 0


def label_ceiling(lines, aux, T, Hc, stop_h, top, ix) -> dict:
    """Per-token pairwise concordance of the LABELS' v1 aggregate with the harness PDMS (ties in the label count 0.5;
    pairs the harness ties are skipped), token-mean over tokens with >= 1 informative pair."""
    import train as TR  # noqa: F401  (trainer_targets imports it; keeps this callable alone)
    cfgs = {"originals_only": lambda g: g == "orig",
            "C1_extras_f075_f050_stop": lambda g: g in ("orig", "f075", "f050", "stop"),
            "C2_extras_f075_stop": lambda g: g in ("orig", "f075", "stop"),
            "extras_f050_only": lambda g: g in ("orig", "f050")}
    per = {c: [] for c in cfgs}
    for r in lines:
        i = ix[r["token"]]
        sl = r.get("slow") or {}
        slot_of = {s: (sv, fv) for s, sv, fv in zip(sl.get("slots", []), sl.get("src", []), sl.get("factor", []))}
        top_i = [int(x) for x in top[i]]
        cand = []                                           # (group, label aggregate, harness pdms)
        for slot in range(64):
            lab = v1_agg(trainer_targets(r["targets"][slot]))
            if slot in slot_of:
                s, f = slot_of[slot]
                if f == 0.0:
                    h, g = stop_h.get(r["token"]), "stop"
                else:
                    rr = top_i.index(s) if s in top_i else None
                    h = (Hc.get((f, "r", rr), {}).get(r["token"]) if rr is not None else None) or \
                        Hc.get((f, "p", s), {}).get(r["token"])
                    g = ftag(f)
                if h is None or not h[0]:
                    continue
                cand.append((g, float(lab), h[2]))
            elif T["valid"][i, slot]:
                cand.append(("orig", float(lab), float(T["pdms"][i, slot])))
        ax = aux.get(r["token"])
        if ax is not None:
            for slot, t in zip(ax["slots"], ax["dropped_targets"]):
                if T["valid"][i, slot]:
                    cand.append(("orig", float(v1_agg(trainer_targets(t))), float(T["pdms"][i, slot])))
        for c, keep in cfgs.items():
            sub = [x for x in cand if keep(x[0])]
            n = ok = 0.0
            for a_ in range(len(sub)):
                for b_ in range(a_ + 1, len(sub)):
                    if c != "originals_only" and sub[a_][0] == "orig" and sub[b_][0] == "orig":
                        continue                              # an extra member must be in the pair
                    dh = sub[a_][2] - sub[b_][2]
                    if abs(dh) <= 1e-9:
                        continue
                    dl = sub[a_][1] - sub[b_][1]
                    n += 1
                    ok += 1.0 if dl * dh > 0 else (0.5 if dl == 0 else 0.0)
            if n:
                per[c].append(ok / n)
    return {c: {"tokens": len(v), "mean_pair_concordance": round(float(np.mean(v)), 4) if v else None}
            for c, v in per.items()}


def as_v3(d: dict) -> dict:
    """A target dict with the --nc-at-fault change undone: the RAW teacher collision back in the NC key, the at-fault
    event and its copy removed. Identity on a dict the arm did not change."""
    if "teacher_collision.NuPlanCollision.info" not in d:
        return d
    e = dict(d)
    e["collision.NuPlanCollision.info"] = e.pop("teacher_collision.NuPlanCollision.info")
    e.pop("ttc.NuPlanTTC.at_fault.info", None)
    return e


def ref_checks(v4_lines) -> dict:
    out = {}
    v3 = {r["token"]: r for r in _lines(f"{WD}/ref_v3/sets")}
    off = {r["token"]: r for r in _lines(f"{WD}/ref_v4off/sets")}
    on = {r["token"]: r for r in v4_lines}
    aux = {r["token"]: r for r in _lines(SETS, "onpolicy_slowaux")}
    if not v3:
        return {"ran": False}
    same = [json.dumps(_strip(off[t])) == json.dumps(_strip(v3[t])) for t in v3 if t in off]
    out["flag_off_lines"] = len(same)
    out["flag_off_identical_to_v3"] = bool(same) and all(same) and len(off) == len(v3)
    n = ok = 0
    detail = []
    for t, r3 in v3.items():
        r4 = on.get(t)
        if r4 is None:
            continue
        n += 1
        sl = r4["slow"]
        kept = [j for j in range(64) if j not in set(sl["slots"])]
        c = {"ndiff": r4["ndiff"] == r3["ndiff"],
             "teacher_targets": as_v3(r4["teacher_targets"]) == r3["teacher_targets"],
             "kept_targets": all(as_v3(r4["targets"][j]) == r3["targets"][j] for j in kept),
             "kept_traj": all(r4["traj"][j] == r3["traj"][j] and r4["yaw"][j] == r3["yaw"][j] for j in kept),
             "dropped_targets": aux.get(t) is not None and all(
                 as_v3(d) == r3["targets"][j] for j, d in zip(aux[t]["slots"], aux[t]["dropped_targets"]))}
        ok += int(all(c.values()))
        if not all(c.values()):
            detail.append({"token": t, **c})
    out["compared_sets"] = n
    out["originals_identical_with_copies_present"] = n > 0 and ok == n
    out["mismatches"] = detail[:5]
    return out


def mutate_checks(v4_lines) -> dict:
    out = {}
    st_p = glob.glob(f"{WD}/mut_label_source/queue/status_r0_*.json")
    if st_p:
        st = json.load(open(st_p[0]))
        out["label_source"] = {k: st.get(k) for k in ("samples", "written", "skipped_ndiff", "selfcheck_failed", "failed")}
        out["selfcheck_refused_all"] = (st.get("written") == 0 and st.get("selfcheck_failed", 0) ==
                                        st.get("samples", -1) - st.get("skipped_ndiff", 0) and st.get("samples", 0) > 0)
    mut = {r["token"]: r for r in _lines(f"{WD}/mut_nocheck/sets")}
    on = {r["token"]: r for r in v4_lines}
    if mut:
        n = eq = 0
        for t, rm in mut.items():
            r4 = on.get(t)
            if r4 is None:
                continue
            for slot, s in zip(rm["slow"]["slots"], rm["slow"]["src"]):
                if s < 0:
                    continue
                n += 1
                eq += int(rm["targets"][slot] == r4["targets"][s])
        out["nocheck_copy_slots"] = n
        out["nocheck_copy_slots_equal_source_labels"] = eq
        out["nocheck_equals_emulation"] = n > 0 and eq == n
    return out


def wide_checks(Hc, stop_h, T, top) -> dict:
    W = np.load(WIDE, allow_pickle=False)
    tok = [str(t) for t in W["token"]]
    out = {"gate": json.loads(str(W["gate"])), "rows": 0, "DAC_disagree": 0, "C_disagree": 0, "by_source": {}}
    # originals vs the E-6 table (every slot)
    ok = np.isfinite(W["dac_orig"]) & T["valid"]
    dd = ((W["dac_orig"] == 0.0) != (T["sub"][:, :, 1] >= 0.999)) & ok
    cc = ((W["c_orig"] >= 0.5) != (T["sub"][:, :, 4] >= 0.999)) & ok
    out["by_source"]["originals_vs_table"] = {"n": int(ok.sum()), "DAC_disagree": int(dd.sum()), "C_disagree": int(cc.sum())}
    out["rows"] += int(ok.sum()); out["DAC_disagree"] += int(dd.sum()); out["C_disagree"] += int(cc.sum())
    for (f, kind, idx), rows in sorted(Hc.items()):
        n = d_ = c_ = 0
        for i, t in enumerate(tok):
            h = rows.get(t)
            if h is None or not h[0]:
                continue
            j = int(top[i, idx]) if kind == "r" else idx
            ld, lc = W[f"dac_{ftag(f)}"][i, j], W[f"c_{ftag(f)}"][i, j]
            if not (np.isfinite(ld) and np.isfinite(lc)):
                continue
            n += 1
            d_ += int((ld == 0.0) != (h[1][1] >= 0.999))
            c_ += int((lc >= 0.5) != (h[1][4] >= 0.999))
        out["by_source"][f"{ftag(f)}_{kind}{idx:02d}"] = {"n": n, "DAC_disagree": d_, "C_disagree": c_}
        out["rows"] += n; out["DAC_disagree"] += d_; out["C_disagree"] += c_
    n = d_ = c_ = 0
    for i, t in enumerate(tok):
        h = stop_h.get(t)
        if h is None or not h[0] or not np.isfinite(W["dac_stop"][i]):
            continue
        n += 1
        d_ += int((W["dac_stop"][i] == 0.0) != (h[1][1] >= 0.999))
        c_ += int((W["c_stop"][i] >= 0.5) != (h[1][4] >= 0.999))
    out["by_source"]["stop"] = {"n": n, "DAC_disagree": d_, "C_disagree": c_}
    out["rows"] += n; out["DAC_disagree"] += d_; out["C_disagree"] += c_
    # unique trajectories behind those rows, and the harness's own consistency where it scored one trajectory twice
    uniq, twice, same = set(), 0, 0
    for f in FACTORS:
        for i, t in enumerate(tok):
            for r_ in range(16):
                hr = Hc.get((f, "r", r_), {}).get(t)
                hp_ = Hc.get((f, "p", int(top[i, r_])), {}).get(t)
                if hr is not None and hp_ is not None:
                    twice += 1
                    same += int(hr[1] == hp_[1] and hr[2] == hp_[2])
            for j in range(64):
                if Hc.get((f, "p", j), {}).get(t) is not None or any(
                        int(top[i, r_]) == j and Hc.get((f, "r", r_), {}).get(t) is not None for r_ in range(16)):
                    uniq.add((f, t, j))
    out["unique_copy_trajectories"] = len(uniq)
    out["unique_rows_total"] = len(uniq) + out["by_source"]["originals_vs_table"]["n"] + out["by_source"]["stop"]["n"]
    out["harness_scored_twice"] = twice
    out["harness_twice_identical"] = same
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prep", "label", "ref", "mutate", "reverify", "wide", "analyze"))
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--ref-chunks", type=int, default=1)
    ap.add_argument("--arm", default="sets", help="labelling arm: 'sets' = the main v4 arm; another name = its own "
                                                  "queue + sets dir (e.g. sets_ncaf with --label-args --nc-at-fault)")
    ap.add_argument("--label-args", default="", help="extra onpolicy_label_v4.py flags for this arm")
    a = ap.parse_args()
    os.makedirs(WD, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    return {"prep": prep, "label": label, "ref": ref, "mutate": mutate, "reverify": reverify, "wide": wide,
            "analyze": analyze}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
