#!/usr/bin/env python3
"""COMPLETE a milestone whose ``run_navsim_refcv7.py`` died part-way -- with ZERO re-computation of
anything already banked (TANITAD VENV). Written 2026-10-04 for step 30,000, whose runner was killed
by the host restart of 2026-10-02 16:55:03 Berlin (System log: User32 1074, a user-initiated restart
from the Start menu) and whose navtest main bridge had died earlier, at 12:50 Berlin, inside a burst
of USB-storage resets on the drive that held D: (UASPStor 129 x14 / disk 153 x5, 12:46-12:50), at
row 11,074 / 12,146 of R7_A1 -- so no navtest main seam was ever written (the runner's ``NO_SEAM``).

    python code/complete_milestone7.py --milestone raw/milestones/step30000 \
        --ckpt D:/refcv7_eval_kit/ckpt/ckpt_30000.pt --md5 ac4e4fab87e35b20de24d8d94910d910 \
        --compare-to raw/milestones/step5000

Everything is gated on ARTIFACTS, and everything it runs is the RUNNER'S OWN code path, imported
(``run_navsim_refcv7`` as ``R``: ``SPLITS``, ``DERIVED``, ``bridge_cmd``, ``seams_ready``,
``ScorePool``, ``run``):

  1. SEAMS. A seam that exists and reads OK is NEVER rewritten (``np.savez`` stamps zip times, so a
     rewrite changes the bytes and orphans the ``seam_sha256`` a PASS score recorded). A split whose
     main seams are missing is re-bridged by ``run_bridge7.py`` -- which RESUMES its rows (banked
     tokens are skipped, never recomputed) -- on the DEVICE ITS BANKED ROWS USED, read from the rows
     AND from the runner's own ``BRIDGE <split> ... device=`` line (the two must agree; one device
     per arm and per split, refcv6 amendment A3). Only CPU resumption is implemented: a CUDA split
     is REFUSED loudly (its lock/dedup protocol belongs to the runner). Stage order: R7_A1 with
     ``--derived`` (its seam + R7_CEILDECL_d + PRIOR_ha0p), then every other missing arm, one
     process each. The runner's CPU-bridge RAM gate (>= 6 GB available, sustained >= 60 s) holds
     before every launch; a stage that exits without its seams is relaunched (resumable) up to 3
     times, then reported FAILED -- never scored partial.
  2. SCORES. Every OK seam whose counts do not read PASS goes into the runner's ``ScorePool`` (its
     8 GB RAM gate, its RAM-guard retries, ``--max-par``); PASS counts are skipped, never re-scored.
  3. POST-PROCESSING per split as soon as all its seams are OK and scored PASS: the runner's own
     steps with the runner's arguments (``plan_deltas7``, ``parse7`` / ``parse_navtest7``,
     ``decompose7``, ``families7``) -- ``postprocess_split`` below is a line-for-line copy of
     ``run_navsim_refcv7.main`` lines 355-410 (the runner keeps it inline; it is not importable).
  4. ``MILESTONE_SUMMARY.json`` rebuilt FROM ARTIFACTS (device / precision / dedup / lock token from
     the seam manifests; the CUDA controls from ``cuda_controls_<split>.log``, read exactly as the
     runner reads them), stamped ``reconstructed``; then ``bars7.py``; then ``step_compare7.py``
     against ``--compare-to`` on every split.

Every log line goes to ``<milestone>/complete.log`` AND to a local-disk mirror
(``C:/Users/Admin/qland/work/refcv7/complete_step<N>.log``): the 2026-10-02 bridge death was silent
because its only log lived on the drive that was resetting.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_navsim_refcv7 as R  # noqa: E402

MIRROR_DIR = "C:/Users/Admin/qland/work/refcv7"
BRIDGE_RAM_GB = 6.0          # = run_navsim_refcv7.main's ram_wait(6.0) for a CPU bridge
BRIDGE_RAM_SUSTAIN_S = 60    # = its 3 samples x 30 s
MAX_BRIDGE_TRIES = 3
_ORIG_LOG = R.log
_MIRROR = {"path": None}


def tee_log(msg: str, path: str) -> None:
    """R.log, plus a local-disk mirror (survives a D: outage)."""
    try:
        _ORIG_LOG(msg, path)
    except OSError as e:                                                  # D: gone: keep going
        print(f"[complete7] primary log write failed: {e!r}", flush=True)
    mp = _MIRROR["path"]
    if mp:
        try:
            with open(mp, "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")
        except OSError:
            pass


R.log = tee_log              # ScorePool resolves ``log`` in R's globals at call time


def main_arms(sk: str) -> list:
    return R.SPLITS[sk]["arms"].split(",")


def dirs_of(out: str, sk: str) -> dict:
    """The runner's ``dirs`` for a split (main bridge + derived; navtest's 200-token diagnostics)."""
    d = {"main": (os.path.join(out, f"bridge_{sk}"), main_arms(sk) + list(R.DERIVED))}
    if sk == "navtest":
        d["sub200"] = (os.path.join(out, "bridge_navtest_sub200"),
                       R.SPLITS["navtest"]["diag_arms"].split(","))
    return d


def rows_devices(bdir: str) -> dict:
    """{arm: (set(devices), n_model_rows)} over every rows_<arm>.jsonl in a bridge dir."""
    out = {}
    if not os.path.isdir(bdir):
        return out
    for fn in sorted(os.listdir(bdir)):
        m = re.match(r"^rows_(.+)\.jsonl$", fn)
        if not m:
            continue
        devs, n = set(), 0
        with open(os.path.join(bdir, fn), encoding="utf-8") as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if r.get("source") == "refcv7":
                    devs.add(r.get("device"))
                    n += 1
        out[m.group(1)] = (devs, n)
    return out


def runner_device(runner_log: str, sk: str):
    """The device the runner's ``BRIDGE <sk> arms=... device=X`` line names (last one wins)."""
    dev = None
    if os.path.exists(runner_log):
        for ln in open(runner_log, encoding="utf-8", errors="replace"):
            m = re.search(rf"\bBRIDGE {sk} arms=\S+ device=(\w+)", ln)
            if m:
                dev = m.group(1)
    return dev


def bridge_log_md5(log_path: str):
    """The checkpoint md5 a bridge log's ``[bridge7] ... md5=`` lines name (set)."""
    out = set()
    if os.path.exists(log_path):
        for ln in open(log_path, encoding="utf-8", errors="replace"):
            m = re.search(r"\[bridge7\] step=\d+ .* md5=([0-9a-f]{32})", ln)
            if m:
                out.add(m.group(1))
    return out


def score_jobs(sk: str, kind: str, d: str, arms: list, step: int, out: str) -> list:
    """Copy of ``run_navsim_refcv7.main`` lines 329-349: the scorer job for each seam."""
    sp = R.SPLITS[sk]
    sdir = os.path.join(out, f"scores_{sk}")
    os.makedirs(sdir, exist_ok=True)
    jobs = []
    for arm in arms:
        seam = os.path.join(d, f"seam_{arm}.npz")
        if sk == "navtest":
            lab = f"r7s{step}_{arm}" + ("_sub" if kind == "sub200" else "")
            cmd = [R.PY, os.path.join(R.HERE, "score_navtest7.py"), "--label", lab, "--seam",
                   seam, "--out", sdir] + (["--tokens", R.SUB200] if kind == "sub200" else [])
            jobs.append({"name": f"{sk}:{lab}", "cmd": cmd,
                         "counts": os.path.join(sdir, lab, f"{lab}.counts.json"),
                         "scorer_log": os.path.join(sdir, lab, f"{lab}.log"),
                         "driver_log": os.path.join(sdir, f"{lab}.driver.txt")})
        else:
            tag = arm if sp["split"] == "warmup_two_stage" else f"{arm}__{sp['split']}"
            cmd = [R.NPY, os.path.join(R.HERE, "score_arm7.py"), "--arm", arm, "--seam", seam,
                   "--split", sp["split"], "--out", sdir, "--exp-tag", f"e7s{step}"]
            jobs.append({"name": f"{sk}:{arm}", "cmd": cmd,
                         "counts": os.path.join(sdir, f"score_{tag}.counts.json"),
                         "scorer_log": os.path.join(sdir, f"score_{tag}.log"),
                         "driver_log": os.path.join(sdir, f"{arm}.driver.txt")})
    return jobs


def split_scored(sk: str, out: str, step: int) -> bool:
    """Every expected seam OK AND every one of them scored PASS."""
    for kind, (d, arms) in dirs_of(out, sk).items():
        st = R.seams_ready(d, arms)
        if any(v != "OK" for v in st.values()):
            return False
        for j in score_jobs(sk, kind, d, arms, step, out):
            if R.jget(j["counts"], "status") != "PASS":
                return False
    return True


def postprocess_split(sk: str, out: str, step: int, label: str, env: dict, qlog: str) -> dict:
    """Line-for-line copy of ``run_navsim_refcv7.main`` lines 355-410 (one split)."""
    dirs = dirs_of(out, sk)
    sp = R.SPLITS[sk]
    rec: dict = {}
    bdir = dirs["main"][0]
    sdir = os.path.join(out, f"scores_{sk}")
    summ = os.path.join(out, f"summary_{sk}.json")
    dec = os.path.join(out, f"decomposition_{sk}.json")
    R.run([R.PY, os.path.join(R.HERE, "plan_deltas7.py"), "--bridge", bdir, "--label", label,
           "--out", os.path.join(out, f"plan_deltas_{sk}.json")], env,
          os.path.join(out, f"plan_deltas_{sk}.log"))
    if sk == "navtest":
        def csv_of(arm, sub=False):
            lab = f"r7s{step}_{arm}" + ("_sub" if sub else "")
            c = os.path.join(sdir, lab, f"{lab}.csv")
            ok = R.jget(os.path.join(sdir, lab, f"{lab}.counts.json"), "status") == "PASS"
            return c if ok and os.path.exists(c) else None
        a1 = csv_of("R7_A1")
        if a1:
            extra = [f"{arm}={csv_of(arm)}" for arm in ("R7_A1_s1",) + R.DERIVED if csv_of(arm)]
            R.run([R.PY, os.path.join(R.HERE, "parse_navtest7.py"), "--arm-csv", a1, "--label", label,
                   "--out", summ, "--bridge", bdir] + (["--extra"] + extra if extra else []),
                  env, os.path.join(out, "parse_navtest.log"))
            R.run([R.PY, os.path.join(R.HERE, "decompose7.py"), "--split", "navtest", "--arm-csv", a1,
                   "--inputs", sp["inputs"], "--label", label, "--out", dec], env,
                  os.path.join(out, "decompose_navtest.log"))
            if "sub200" in dirs:
                sub_extra = [f"{arm}={csv_of(arm, True)}" for arm in dirs["sub200"][1]
                             if csv_of(arm, True)]
                R.run([R.PY, os.path.join(R.HERE, "parse_navtest7.py"), "--arm-csv", a1,
                       "--tokens", R.SUB200, "--label", label + " -- 200-token DIAGNOSTIC",
                       "--out", os.path.join(out, "summary_navtest_sub200.json"),
                       "--bridge", dirs["sub200"][0], "--census", sp["speed_oracle"],
                       "--map", sp["speed"]] + (["--extra"] + sub_extra if sub_extra else []),
                      env, os.path.join(out, "parse_navtest_sub200.log"))
        else:
            R.log("navtest R7_A1 has no PASSING score -- no summary (never a partial CSV)", qlog)
    else:
        floors = os.path.join(PKG, "raw", "floors", sp["split"])
        suffix = [] if sp["split"] == "warmup_two_stage" else ["--csv-suffix", f"__{sp['split']}"]
        R.run([R.PY, os.path.join(R.HERE, "parse7.py"), "--split", sp["split"], "--scores", sdir,
               "--floors", floors, "--bridge", bdir, "--inputs", sp["inputs"], "--label", label,
               "--out", summ] + suffix, env, os.path.join(out, f"parse_{sk}.log"))
        R.run([R.PY, os.path.join(R.HERE, "decompose7.py"), "--split", sp["split"], "--scores", sdir,
               "--floors", floors, "--bridge", bdir, "--inputs", sp["inputs"], "--label", label,
               "--out", dec] + suffix, env, os.path.join(out, f"decompose_{sk}.log"))
    if sk in ("navhard", "navtest"):
        for arm in ("R7_A1", "PRIOR_ha0p"):
            sm = os.path.join(bdir, f"seam_{arm}.npz")
            if os.path.exists(sm):
                R.run([R.PY, os.path.join(R.HERE, "families7.py"), "--seam", sm, "--inputs",
                       sp["inputs"], "--stage", "1", "--label", label, "--out",
                       os.path.join(out, f"families_{sk}_{arm}.json")], env,
                      os.path.join(out, f"families_{sk}.log"))
    rec["summary"] = summ if os.path.exists(summ) else None
    rec["decomposition"] = dec if os.path.exists(dec) else None
    R.log(f"SPLIT {sk} parsed: summary={'present' if rec['summary'] else 'ABSENT'}", qlog)
    return rec


def split_record(out: str, sk: str, notes: dict) -> dict:
    """The runner's per-split record, rebuilt from ARTIFACTS."""
    bdir = os.path.join(out, f"bridge_{sk}")
    man = R.jget(os.path.join(bdir, "seam_R7_A1.manifest.json")) or {}
    devs = man.get("device") or []
    dev = devs[0] if len(devs) == 1 else (devs or None)
    rec = {"device": dev, "precision": man.get("precision"),
           "gpu_lock_token": man.get("gpu_lock_token"),
           "exact_dedup": (man.get("model") or {}).get("exact_dedup"),
           "reconstructed_from": [os.path.relpath(os.path.join(bdir, "seam_R7_A1.manifest.json"), PKG)]}
    clog = os.path.join(out, f"cuda_controls_{sk}.log")
    if dev == "cuda" and os.path.exists(clog):
        txt = open(clog, encoding="utf-8", errors="replace").read().replace("\\", "/")
        rec["cuda_controls"] = {"pytest_rc": None,
                                "pytest_rc_note": "not recorded (the runner died before writing it)",
                                "K0_pass": "PASSED tests/test_model_seam7.py::test_K0" in txt,
                                "KD_pass": "PASSED tests/test_model_seam7.py::test_KD" in txt,
                                "KD": R.jget(os.path.join(PKG, "raw", "controls",
                                                          "KD_exact_dedup_cuda.json"))}
        rec["reconstructed_from"].append(os.path.relpath(clog, PKG))
    rec["seams"] = {k: R.seams_ready(d, arms) for k, (d, arms) in dirs_of(out, sk).items()}
    rec.update(notes.get(sk, {}))
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", required=True, help="raw/milestones/step<N> (absolute or PKG-relative)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--md5", required=True)
    ap.add_argument("--splits", default="warmup,navtest,navhard")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-par", type=int, default=4)
    ap.add_argument("--compare-to", default="", help="earlier milestone dir for step_compare7.py")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    a = ap.parse_args(argv)
    if os.environ.get("R7_MUTATION"):
        sys.exit("⛔ R7_MUTATION is set -- a mutation-test environment never scores a checkpoint")
    out = a.milestone if os.path.isabs(a.milestone) else os.path.join(PKG, a.milestone)
    out = os.path.normpath(out)
    if not os.path.isdir(out):
        sys.exit(f"⛔ no milestone dir {out}")
    qlog = os.path.join(out, "complete.log")
    md5 = R.md5_file(a.ckpt)
    if md5 != a.md5:
        sys.exit(f"⛔ md5 {md5} != --md5 {a.md5}")
    import torch
    step = int(torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True).get("step", -1))
    if os.path.basename(out) != f"step{step}":
        sys.exit(f"⛔ checkpoint step {step} does not name this milestone dir {out}")
    label = "RESULT" if step >= 5000 else "VALIDATION ONLY (SPEC §6)"
    os.makedirs(MIRROR_DIR, exist_ok=True)
    _MIRROR["path"] = os.path.join(MIRROR_DIR, f"complete_step{step}.log")
    cfg_md5 = R.md5_file(R.CONFIG)
    rlog = os.path.join(out, "runner.log")
    m0 = re.search(r"START .* md5=([0-9a-f]{32}) .* config_md5=([0-9a-f]{32})",
                   open(rlog, encoding="utf-8", errors="replace").read()) if os.path.exists(rlog) else None
    if not m0 or m0.group(1) != md5 or m0.group(2) != cfg_md5:
        sys.exit(f"⛔ runner.log START does not name this checkpoint/config: "
                 f"{m0.groups() if m0 else None} vs ({md5}, {cfg_md5})")
    R.log(f"COMPLETE7 START step={step} md5={md5} label={label} config_md5={cfg_md5} pid={os.getpid()} "
          f"mirror={_MIRROR['path']} dry_run={a.dry_run}", qlog)
    if not a.dry_run:
        try:
            R.log(f"RESUMED by complete_milestone7.py (pid {os.getpid()}) -> {qlog}", rlog)
        except OSError:
            pass
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{R.TREE}/stack;{R.TREE}/taniteval"
    env["TANITAD_REPO"] = R.TREE
    env["OMP_NUM_THREADS"] = str(a.threads)
    env["PYTHONIOENCODING"] = "utf-8"
    splits = [s for s in a.splits.split(",") if s]
    # ---- 0. guard: every OK seam already on disk was made from THIS checkpoint
    for sk in splits:
        for kind, (d, arms) in dirs_of(out, sk).items():
            for arm, st in R.seams_ready(d, arms).items():
                if st == "OK":
                    mm = R.jget(os.path.join(d, f"seam_{arm}.manifest.json"), "model", "ckpt_md5")
                    if mm != md5:
                        sys.exit(f"⛔ {d}/seam_{arm} names checkpoint md5 {mm}, not {md5} -- refused")
    # ---- 1. plan the bridge stages (never rewriting an OK seam)
    stages = []
    notes: dict = {}
    for sk in splits:
        sp = R.SPLITS[sk]
        for kind, (d, arms) in dirs_of(out, sk).items():
            st = R.seams_ready(d, arms)
            missing = [x for x, v in st.items() if v != "OK"]
            if not missing:
                continue
            if kind != "main":
                sys.exit(f"⛔ {sk}/{kind} seams missing {missing}: resuming a diagnostic bridge is not "
                         "implemented here -- refused")
            rd = rows_devices(d)
            devs = set().union(*[v[0] for v in rd.values()]) if rd else set()
            rdev = runner_device(rlog, sk)
            if devs != {"cpu"} or rdev != "cpu":
                sys.exit(f"⛔ {sk}: banked rows on {devs or 'NO ROWS'}, runner line device={rdev} -- only a "
                         "CPU resumption is implemented (CUDA splits belong to the runner's lock "
                         "protocol) -- refused")
            bm = bridge_log_md5(d + ".log")
            if bm != {md5}:
                sys.exit(f"⛔ {sk}: the banked rows' bridge log names checkpoint md5 {bm}, not {{{md5}}}")
            protected = {x for x, v in st.items() if v == "OK"}
            plan = []
            if any(x in missing for x in ("R7_A1",) + R.DERIVED):
                plan.append({"arms": "R7_A1", "derived": True, "expect": ["R7_A1", *R.DERIVED]})
            for arm in main_arms(sk):
                if arm != "R7_A1" and arm in missing:
                    plan.append({"arms": arm, "derived": False, "expect": [arm]})
            for p in plan:
                clash = protected & set(p["expect"])
                if clash:
                    sys.exit(f"⛔ {sk}: stage {p['arms']} would REWRITE OK seam(s) {sorted(clash)} -- refused")
                p.update({"split": sk, "bdir": d, "tries": 0})
                stages.append(p)
            notes[sk] = {"resumed_by": "complete_milestone7.py",
                         "banked_rows_at_resume": {k: v[1] for k, v in rd.items()},
                         "resume_device": "cpu"}
    R.log(f"PLAN stages={[(s['split'], s['arms'], s['derived']) for s in stages]} "
          f"banked={ {s: notes[s]['banked_rows_at_resume'] for s in notes} }", qlog)
    pool = R.ScorePool(a.max_par, env, qlog)
    # ---- 2. score every OK seam not yet PASS (PASS ones are skipped by ScorePool.add)
    for sk in splits:
        for kind, (d, arms) in dirs_of(out, sk).items():
            ok = [x for x, v in R.seams_ready(d, arms).items() if v == "OK"]
            for j in score_jobs(sk, kind, d, ok, step, out):
                if a.dry_run:
                    R.log(f"DRY {j['name']} counts={R.jget(j['counts'], 'status')}", qlog)
                else:
                    pool.add(j)
    if a.dry_run:
        R.log("DRY RUN -- nothing launched", qlog)
        return 0
    summary_recs: dict = {}
    done_splits: set = set()
    bridge = None          # (Popen, file handle, stage)
    ram_since = None
    while True:
        # -- the CPU bridge (one stage at a time, RAM-gated, resumable, retried)
        if bridge is None and stages:
            f = R.free_gb()
            ram_since = (ram_since or time.time()) if f >= BRIDGE_RAM_GB else None
            if ram_since and time.time() - ram_since >= BRIDGE_RAM_SUSTAIN_S:
                s = stages[0]
                s["tries"] += 1
                sp = R.SPLITS[s["split"]]
                cmd = R.bridge_cmd(sp, s["split"], s["arms"], a.ckpt, md5, "cpu", None, label, s["bdir"],
                                   a.threads, derived=s["derived"], dedup=True)
                e = dict(env, CUDA_VISIBLE_DEVICES="-1")
                fh = open(s["bdir"] + ".log", "a", encoding="utf-8")
                fh.write("CMD: " + " ".join(cmd) + "\n")
                fh.flush()
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=e)
                bridge = (proc, fh, s)
                ram_since = None
                R.log(f"BRIDGE {s['split']} arms={s['arms']} derived={s['derived']} device=cpu "
                      f"try={s['tries']} pid={proc.pid} (RAM gate passed: {f:.1f} GB)", qlog)
                time.sleep(90)            # let the model load before any scorer reads free RAM
                continue
        if bridge is not None and bridge[0].poll() is not None:
            proc, fh, s = bridge
            fh.close()
            bridge = None
            st = R.seams_ready(s["bdir"], s["expect"])
            R.log(f"BRIDGE {s['split']} arms={s['arms']} exited rc={proc.returncode} seams={st}", qlog)
            if all(v == "OK" for v in st.values()):
                stages.pop(0)
                for j in score_jobs(s["split"], "main", s["bdir"], s["expect"], step, out):
                    pool.add(j)
            elif s["tries"] >= MAX_BRIDGE_TRIES:
                R.log(f"BRIDGE {s['split']} arms={s['arms']} FAILED after {s['tries']} tries -- its seams "
                      "are NOT scored", qlog)
                notes.setdefault(s["split"], {}).setdefault("bridge_failed", []).append(s["arms"])
                stages.pop(0)
        pool.poll()
        for sk in splits:
            if sk not in done_splits and split_scored(sk, out, step):
                summary_recs[sk] = postprocess_split(sk, out, step, label, env, qlog)
                done_splits.add(sk)
        if bridge is None and not stages and not pool.queue and not pool.live:
            break
        time.sleep(20)
    for sk in splits:                       # whatever could not complete is still parsed (as the runner)
        if sk not in done_splits:
            summary_recs[sk] = postprocess_split(sk, out, step, label, env, qlog)
    # ---- 4. MILESTONE_SUMMARY from artifacts, BARS, step comparisons
    msp = os.path.join(out, "MILESTONE_SUMMARY.json")
    summary = {"ckpt": a.ckpt, "md5": md5, "step": step, "label": label, "config_md5": cfg_md5,
               "splits": {}, "reconstructed": {
                   "by": "code/complete_milestone7.py", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "why": ("run_navsim_refcv7.py died part-way (host restart 2026-10-02 14:55:03Z); the per-split "
                           "records are rebuilt from the seam manifests and logs, never from memory")}}
    for sk in splits:
        summary["splits"][sk] = dict(split_record(out, sk, notes), **summary_recs.get(sk, {}))
    if os.path.exists(msp):
        prev = json.load(open(msp, encoding="utf-8"))
        if prev.get("md5") != summary["md5"]:
            sys.exit(f"⛔ {msp} belongs to md5 {prev.get('md5')}, not {summary['md5']} -- refused")
        merged = dict(prev.get("splits", {}))
        merged.update(summary["splits"])
        summary["splits"] = merged
    with open(msp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    R.run([R.PY, os.path.join(R.HERE, "bars7.py"), "--milestone", out], env, os.path.join(out, "bars.log"))
    R.log(f"BARS {'present' if os.path.exists(os.path.join(out, 'BARS.json')) else 'ABSENT'}", qlog)
    if a.compare_to:
        ca = a.compare_to if os.path.isabs(a.compare_to) else os.path.join(PKG, a.compare_to)
        tag = os.path.basename(os.path.normpath(ca))
        for sk in splits:
            co = os.path.join(out, f"compare_vs_{tag}_{sk}.json")
            R.run([R.PY, os.path.join(R.HERE, "step_compare7.py"), "--split", sk, "--a", ca, "--b", out,
                   "--out", co], env, os.path.join(out, f"compare_vs_{tag}_{sk}.log"))
            R.log(f"COMPARE {sk} vs {tag}: {'present' if os.path.exists(co) else 'ABSENT'}", qlog)
    R.log(f"ZZCOMPLETE7DONEZZ step={step} bars={'present' if os.path.exists(os.path.join(out, 'BARS.json')) else 'ABSENT'}",
          qlog)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:                                                 # noqa: BLE001
        tb = traceback.format_exc()
        for p in (os.path.join(MIRROR_DIR, "complete_crash.txt"),):
            try:
                with open(p, "a", encoding="utf-8") as fh:
                    fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + tb + "\n")
            except OSError:
                pass
        print(tb, flush=True)
        raise
