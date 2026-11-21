#!/usr/bin/env python3
"""A fake box 2 for the chain's end-to-end self-test (section E). The chain talks to box 2 only through `ssh tanitad-box2 "<powershell>"` and `scp`; two shims on PATH hand those calls to this file, which answers them from a local tree:

    D:/tanitad/...  <->  $FAKEBOX_ROOT/tanitad/...

  ssh   boot time, Get-FileHash, (Get-Item).Length, free space, telemetry tail, launch.ps1 (the job runs HERE, synchronously, and writes its own log with the ZZJOB-END line the chain waits for), process listing (empty)
  scp   copies in either direction through the path map
  jobs  `refresh` (tar -xf of the shipped code archive into tanitad/repo), `telemetry_long` (a no-op), and `queue` jobs written by the fake mk_v8f_job.py: the stage is EMULATED -- gatedag writes a gate selfcheck.json (DAG2-0's REAL
        gate record, with ORC's REAL ego probe spliced in for A / B, argv_sha of the queue's printed gate line) and the pin; traindag writes a run directory (DAG2-0's REAL summary.json relabelled); evaldag writes eval files.
        The lines come from the REAL queue under test (FAKEBOX_QUEUE, run dry through cmd.exe), so the chain's gate asserter, run record and K-REACH judge exactly what the queue prints.
Fault injection: FAKE_FAULT = comma list of kind:arm:seed
  gate_init (the gate record's init sha256 is another checkpoint) | gate_probe (the ego probe moves no gradient) | rep_applied (REP's run applied the perturbation: ego_perturb_n > 0) | reach0 (A / B: ego_perturb_n 0) |
  argv_mismatch (argv.json is not the planned argv) | job_rc1 (the training job exits 1) | bad_sha (any hash of that box-2 path comes back wrong: kind = bad_sha, arm field = a path substring)
Every ssh / scp / job is appended to $FAKEBOX_ROOT/calls.log.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FB = Path(os.environ["FAKEBOX_ROOT"])
LOG = FB / "calls.log"
TREE = Path(os.environ.get("DAGORC_TREE") or r"F:\TanitAD-scratch\dagorc_tree")
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
FAULTS = [tuple(x.split(":")) for x in os.environ.get("FAKE_FAULT", "").split(",") if x]


def note(*a):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(" ".join(str(x) for x in a) + "\n")


BANKED = {"dag": Path(os.environ.get("DAGORC_DAG_DIR") or "E:/Projects/TanitAD-artifacts/v8f-dagger"), "v9": Path("C:/Users/Admin/box2_ship/v9ship")}


def mp(p: str) -> Path:
    """box-2 path -> local path. A file that is not in the fake box but IS a banked artifact (DAG_s<S>'s run, the v9 releases) is READ from the banked copy (read-only; the fake box never writes there)."""
    p = p.replace("\\", "/").strip("'\"")
    if re.match(r"^[A-Za-z]:/tanitad", p):
        q = FB / p[3:]
        if not q.exists():
            m = re.search(r"/runs/v8f_mini_full/(M2b_s[01]_dag/.+)$", p)
            if m and (BANKED["dag"] / m.group(1)).exists():
                return BANKED["dag"] / m.group(1)
            m = re.search(r"/data/v9/(v9_labels_\w+\.npz)$", p)
            if m and (BANKED["v9"] / m.group(1)).exists():
                return BANKED["v9"] / m.group(1)
        return q
    return Path(p)


def fault(kind, arm, seed) -> bool:
    return any(k == kind and a == arm and str(s) == str(seed) for k, a, s in FAULTS)


def bad_sha_for(path) -> bool:
    return any(k == "bad_sha" and a in str(path).replace("\\", "/") for k, a, s in FAULTS)


def hexof(path: Path, algo: str) -> str:
    h = hashlib.new(algo.lower())
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------------------ the queue stages (emulated)
def queue_lines(arm, s):
    sys.path.insert(0, str(HERE))
    os.environ["DAGORC_QUEUE"] = os.environ["FAKEBOX_QUEUE"]
    import dagorc_lib as L
    g = L.trainer_line(L.dry(f"gatedag {arm} {s}"))
    t = L.trainer_line(L.dry(f"traindag {arm} {s}"))
    e = [x for x in L.dry(f"evaldag {arm} {s}") if "v8f_mini_eval.py" in x][0]
    return g, t, e, L


def sha16(toks):
    return hashlib.sha256(" ".join(toks).encode()).hexdigest()[:16]


def gate_record(arm, s, gl_toks, L):
    sc = json.loads((L.DAG2_DIR / f"M2b_s{s}_dag20" / "selfcheck.json").read_text(encoding="utf-8"))
    if arm != "dagorcrep":
        orc = json.loads((TREE / PK / "raw" / "2026-10-26-orecover-run" / "runs" / f"M2b_s{s}_orc" / "selfcheck.json").read_text(encoding="utf-8"))
        sc["ego_perturb_probe"] = copy.deepcopy(orc["ego_perturb_probe"])
        sc["levers"]["ego_perturb"] = copy.deepcopy(orc["levers"]["ego_perturb"])
        if arm == "dagorc01":
            sc["ego_perturb_probe"]["p_declared"] = 0.1
            sc["levers"]["ego_perturb"]["ego_perturb_p"] = 0.1
    sc["argv_sha"] = sha16(gl_toks)
    if fault("gate_init", arm, s):
        sc["levers"]["init_ckpt"]["sha256"] = "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8"
        sc["info"]["init_ckpt"] = sc["levers"]["init_ckpt"]
    if fault("gate_probe", arm, s) and arm != "dagorcrep":
        sc["ego_perturb_probe"]["grad_diff_all"] = 0.0
    return sc


def stage(args, log):
    st, arm, s = args[0], args[1], int(args[2])
    g, t, e, L = queue_lines(arm, s)
    runs = FB / "tanitad" / "runs" / "v8f_mini_full"
    gtok, ttok = L.train_argv(g), L.train_argv(t)
    ppath = runs / "dagpin" / f"M2b_s{s}_{arm}.train.argv.txt"
    if st == "gatedag":
        gd = runs / f"gate_M2b-{arm}_s{s}"
        gd.mkdir(parents=True, exist_ok=True)
        (gd / "selfcheck.json").write_text(json.dumps(gate_record(arm, s, gtok, L)), encoding="utf-8")
        ppath.parent.mkdir(parents=True, exist_ok=True)
        ppath.write_bytes((t + "\r\n").encode("utf-8"))
        (runs / f"gate_M2b-{arm}_s{s}.log").write_text("gate ok\r\n", encoding="utf-8")
        return 0
    if st == "traindag":
        if not ppath.exists():
            log.append("REFUSED: no pinned training argv")
            return 2
        if fault("job_rc1", arm, s):
            log.append("[FAIL] train - injected")
            return 1
        d = runs / f"M2b_s{s}_{arm}"
        d.mkdir(parents=True, exist_ok=True)
        p = {"dagorc": 0.3, "dagorc01": 0.1, "dagorcrep": 0.0}[arm]
        tseed = 10 + s if arm == "dagorcrep" else s
        s16 = sha16(ttok)
        sm = json.loads((L.DAG2_DIR / f"M2b_s{s}_dag20" / "summary.json").read_text(encoding="utf-8"))
        sm["seed"], sm["argv_sha"], sm["steps"] = tseed, s16, 3000
        sm["levers"]["ego_perturb"] = {"ego_perturb_p": p, **({"ego_perturb_vscale": [0.5, 1.0], "ego_perturb_vmax": 8.0, "ego_perturb_adrop": 1.5, "ego_perturb_yawrate_std": 2.0, "ego_perturb_seed": s} if p > 0 else
                                                              {"ego_perturb_vscale": [0.5, 1.0], "ego_perturb_vmax": 8.0, "ego_perturb_adrop": 1.5, "ego_perturb_yawrate_std": 0.0, "ego_perturb_seed": 0}), "kin_mode": "yaw"}
        n = int(round(p * 82457))
        if fault("reach0", arm, s):
            n = 0
        if fault("rep_applied", arm, s):
            n = 123
        sm["ego_perturb_n"], sm["ego_perturb_eligible"] = n, 82457
        argv = {"argv": ttok, "sha": s16}
        if fault("argv_mismatch", arm, s):
            argv = {"argv": [("2e-4" if x == "1e-4" else x) for x in ttok], "sha": s16}
        (d / "argv.json").write_text(json.dumps(argv), encoding="utf-8")
        (d / "summary.json").write_text(json.dumps(sm), encoding="utf-8")
        (d / "selfcheck.json").write_text(json.dumps({"argv_sha": s16, "declared_vs_built": "PASS"}), encoding="utf-8")
        (d / "train_log.jsonl").write_text('{"step": 3000}\n', encoding="utf-8")
        (d / "train.log").write_text("done\n", encoding="utf-8")
        (d / "ckpt_last.pt").write_bytes(f"ckpt {arm} {s}".encode() * 100)
        return 0
    if st == "evaldag":
        d = runs / f"M2b_s{s}_{arm}"
        if not (d / "summary.json").exists():
            log.append("REFUSED: not finished training")
            return 2
        ev = runs / "eval"
        ev.mkdir(parents=True, exist_ok=True)
        tag = f"M2b-{arm}_s{s}"
        (ev / f"eval_{tag}.json").write_text("{}", encoding="utf-8")
        (ev / f"eval_{tag}.npz").write_bytes(b"npz")
        (ev / f"eval_{tag}.log").write_text("eval\n", encoding="utf-8")
        if fault("eval_touch", arm, s):      # an eval that rewrites the checkpoint it evaluated (the chain hashes it after training and again after the evals)
            with open(d / "ckpt_last.pt", "ab") as f:
                f.write(b"x")
        return 0
    log.append(f"REFUSED: unknown stage {st}")
    return 2


def run_job(name: str):
    jobf = FB / "tanitad" / "jobs" / f"{name}.cmd"
    logf = FB / "tanitad" / "logs" / f"{name}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    note("job", name)
    log, rc = [f"===== job {name}"], 0
    if name.endswith("_refresh"):
        tar = FB / "tanitad" / "repo_v8fdagorc_tip.tar"
        dst = FB / "tanitad" / "repo"
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar) as tf:
            tf.extractall(dst)
        tipf = [ln for ln in jobf.read_text().splitlines() if ln.startswith("echo ") and "TIP_V8FDAGORC" in ln]
        m = re.match(r"echo ([0-9a-f]{40})>", tipf[0]) if tipf else None
        (dst / "TIP_V8FDAGORC.txt").write_text((m.group(1) if m else "?") + "\n")
    elif jobf.exists():
        txt = jobf.read_text()
        m = re.search(r"v8f_mini_box2_queue\.cmd (.+)$", txt, re.M)
        if m:
            note("stage", m.group(1).strip())
            rc = stage(m.group(1).split(), log)
    log.append(f"===== ZZJOB-END {time.strftime('%a %d.%m.%Y %H:%M:%S')} rc={rc}")
    logf.write_text("\r\n".join(log) + "\r\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------------------------------------ ssh / scp
def ssh(cmd: str) -> int:
    note("ssh", cmd[:160])
    if "LastBootUpTime" in cmd:
        print("2026-10-27 08:00:00")
        return 0
    m = re.search(r"Get-FileHash -Algorithm (\w+) -LiteralPath '([^']+)'", cmd)
    if m:
        p = mp(m.group(2))
        if not p.exists():
            return 1
        h = hexof(p, m.group(1))
        if bad_sha_for(m.group(2)):
            h = ("0" if h[0] != "0" else "1") + h[1:]
        print(h)
        return 0
    m = re.search(r"\(Get-Item -LiteralPath '([^']+)'\)\.Length", cmd)
    if m:
        p = mp(m.group(1))
        if not p.exists():
            return 1
        print(p.stat().st_size)
        return 0
    if "Get-PSDrive" in cmd:
        print("200")
        return 0
    if "telemetry_full.csv" in cmd and "-Tail" in cmd:
        c = FB / "telemetry.counter"
        n = int(c.read_text()) + 1 if c.exists() else 1
        c.write_text(str(n))
        print(f"2026-10-27 08:{n // 60 % 60:02d}:{n % 60:02d},0,0,0,0,0,0,0,0,0,40,0")
        return 0
    m = re.search(r"launch\.ps1 -Job (\S+) -Log (\S+)", cmd)
    if m:
        name = Path(m.group(1).replace("\\", "/")).stem
        run_job(name)
        print("launched " + name)
        return 0
    m = re.search(r"Get-Content '([^']+)' -Tail", cmd)
    if m:
        p = mp(m.group(1))
        if p.exists():
            print("\n".join(p.read_text(errors="replace").splitlines()[-6:]))
        return 0
    return 0      # Test-Path / Remove-Item / New-Item / Get-CimInstance: nothing to do


def scp(args):
    # drop the options and the values of -o
    clean, skip = [], False
    for x in args:
        if skip:
            skip = False
            continue
        if x == "-o":
            skip = True
            continue
        if x.startswith("-"):
            continue
        clean.append(x)
    src, dst = clean[-2], clean[-1]
    note("scp", src, dst)

    def side(p):
        if p.startswith("tanitad-box2:"):
            return mp(p[len("tanitad-box2:"):]), True
        m = re.match(r"^/([A-Za-z])/(.*)$", p)      # the chain passes MSYS paths (/f/...): the shim is a native Windows program
        return Path(f"{m.group(1).upper()}:/{m.group(2)}" if m else p), False

    (s, s_remote), (d, d_remote) = side(src), side(dst)
    if not s.exists():
        return 1
    if d_remote and (d.name == "" or str(dst).endswith("/")):
        d = d / s.name
    d.parent.mkdir(parents=True, exist_ok=True)
    if d.exists() and d.is_dir():
        d = d / s.name
    shutil.copyfile(s, d)
    if s_remote and any(k == "pull_corrupt" and a in str(src).replace("\\", "/") for k, a, _ in FAULTS) and d.stat().st_size > 0:      # a pull that lands the right NUMBER of bytes with one of them wrong
        b = bytearray(d.read_bytes())
        b[0] ^= 0xFF
        d.write_bytes(bytes(b))
    return 0


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "ssh":
        sys.exit(ssh(sys.argv[-1]))
    if kind == "scp":
        sys.exit(scp(sys.argv[2:]))
    if kind == "job":
        run_job(sys.argv[2])
        sys.exit(0)
    sys.exit(2)
