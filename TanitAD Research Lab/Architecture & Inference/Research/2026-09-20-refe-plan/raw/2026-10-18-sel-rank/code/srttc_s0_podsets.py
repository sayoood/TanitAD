#!/usr/bin/env python3
"""SR-TTC S0 -- fetch the 15,808 fit SETS (64 proposals per set) and the PDM LABELS P2 trained on, for the registered 10 % subsample, from the idle pod, and build
the per-log set stores the dump reads.  READS NO LABEL VALUE (counts / shapes / key checks only).

WHAT P2 TRAINED ON (refe/p2r_pod.sh:228, the pod's own command line):  --onpolicy-targets refe_navtrain/onpolicy/sets   --pdm-labels refe_sft2/train_pdm   --pdm-only
  * a set = one line `kind: onpolicy_set` (log_name, token, step, rank, ckpt_step, traj [64,20,2], yaw [64,20], targets ...), the student's OWN 64 proposals of an EARLIER
    checkpoint (1,281 distinct (step, rank, ckpt_step) combinations among the served keys: the fan of a fit set is NOT P2's own fan unless ckpt_step == 10075);
  * its labels = the pod's `train_pdm/pdm_onpolicy_*.jsonl` row with the SAME key AND ckpt_step (NC, DAC, EP, TTC, C, DDC), clipped to [0, 1].
This script reads those two stores with `train.OnPolicyBank` ITSELF (the class P2's trainer used), on the lines of the selected logs only.  Nothing is re-implemented, so the
bank rules (newest ckpt_step wins; a PDM row counts only for the set it was computed on; pdm_only) are the training run's.

Pod access: the ssh prefix is read at run time from the file named in pod_cfg.local; it lives only in this process' environment and is NEVER written anywhere.
The pod must be IDLE (podcopy.guard: no GPU compute app, no p6_pod.sh); otherwise this waits.

    python srttc_s0_podsets.py --plan-dir <run/plan> --work <work> [--wait-idle-min 240]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.environ.get("P2HO_TREE", "C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
POD_CFG = "C:/Users/Admin/srttc_s0/pod_cfg.local"      # one line: the path of the file that holds the ssh prefix (kept out of the repo on purpose)
POD_SETS = "/workspace/data/refe_navtrain/onpolicy/sets"
POD_PDM = "/workspace/data/refe_sft2/train_pdm"
MIN_FREE_GB = 6.0


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_podcopy():
    os.environ["PODSSH"] = open(open(POD_CFG, encoding="utf-8").read().strip(), encoding="utf-8").read().strip()      # in-process only
    sys.path.insert(0, HERE)
    import podcopy
    return podcopy


def wait_idle(pc, minutes):
    t0 = time.time()
    while True:
        ok, why = pc.guard()
        if ok:
            return
        if time.time() - t0 > minutes * 60:
            raise SystemExit(f"rc3 pod not idle for {minutes} min ({why})")
        print(f"[{now()}] pod not idle ({why}); waiting", flush=True)
        time.sleep(60)


def remote_grep(pc, pod_dir, glob, patterns, out_path):
    """grep -F -f <patterns> <pod_dir>/<glob> on the pod, streamed to out_path; sha256 computed on both sides and required equal.  Returns (bytes, sha256, n_lines)."""
    cmd = (f"cd {pod_dir} && bash -c 'set -o pipefail; grep -h -F -f /dev/stdin {glob} | tee >(sha256sum 1>&2)'")
    p = subprocess.Popen(pc.PODSSH + [cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdin.write(("\n".join(patterns) + "\n").encode("utf-8"))
    p.stdin.close()
    h, n, nl = hashlib.sha256(), 0, 0
    with open(out_path + ".part", "wb") as fo:
        for b in iter(lambda: p.stdout.read(1 << 22), b""):
            fo.write(b)
            h.update(b)
            n += len(b)
            nl += b.count(b"\n")
    err = p.stderr.read().decode("utf-8", "replace")
    p.wait()
    import re
    m = re.search(r"^([0-9a-f]{64})\s+-\s*$", err, re.M)
    rsha = m.group(1) if m else ""
    if p.returncode not in (0,) or rsha != h.hexdigest():
        raise SystemExit(f"rc4 remote grep failed: rc={p.returncode} remote sha {rsha[:12]!r} local {h.hexdigest()[:12]!r} stderr {err[:200]!r}")
    os.replace(out_path + ".part", out_path)
    return n, h.hexdigest(), nl


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--wait-idle-min", type=float, default=240.0)
    ap.add_argument("--skip-fetch", action="store_true", help="reuse work/sets_pod/ already fetched (rebuild the per-log stores only)")
    a = ap.parse_args(argv)
    plan = json.load(open(f"{a.plan_dir}/master_plan.json", encoding="utf-8"))
    logs = [r["log"] for r in plan["logs"]]
    sp = f"{a.work}/sets_pod"
    os.makedirs(f"{sp}/sets", exist_ok=True)
    os.makedirs(f"{sp}/pdm", exist_ok=True)
    os.makedirs(f"{a.work}/sets", exist_ok=True)
    prov = {"what": "pod-side fetch of the SR-TTC S0 fit sets + P2's PDM labels (selected logs only)", "written": now(), "pod_sets_dir": POD_SETS.rsplit("/", 3)[-3:], "n_logs": len(logs)}
    if not a.skip_fetch:
        free = __import__("shutil").disk_usage(a.work).free / 2**30
        if free < MIN_FREE_GB:
            raise SystemExit(f"rc5 disk guard: {free:.1f} GiB free < {MIN_FREE_GB}")
        pc = load_podcopy()
        wait_idle(pc, a.wait_idle_min)
        t0 = time.time()
        n, sha, nl = remote_grep(pc, POD_SETS, "onpolicy_*.jsonl", [f'"log_name": "{lg}"' for lg in logs], f"{sp}/sets/onpolicy_selected.jsonl")
        prov["sets_file"] = {"bytes": n, "sha256": sha, "lines": nl, "seconds": round(time.time() - t0)}
        print(f"[{now()}] sets: {nl:,} lines {n / 2**20:.0f} MiB sha256 equal on both sides ({prov['sets_file']['seconds']} s)", flush=True)
        wait_idle(pc, a.wait_idle_min)
        t0 = time.time()
        n, sha, nl = remote_grep(pc, POD_PDM, "pdm_onpolicy_*.jsonl", [f'"key": ["{lg}", ' for lg in logs], f"{sp}/pdm/pdm_selected.jsonl")
        prov["pdm_file"] = {"bytes": n, "sha256": sha, "lines": nl, "seconds": round(time.time() - t0)}
        print(f"[{now()}] pdm labels: {nl:,} lines {n / 2**20:.0f} MiB sha256 equal on both sides ({prov['pdm_file']['seconds']} s)", flush=True)
    else:
        for k, p in (("sets_file", f"{sp}/sets/onpolicy_selected.jsonl"), ("pdm_file", f"{sp}/pdm/pdm_selected.jsonl")):
            prov[k] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
    # ---- the bank, by the training run's own class
    sys.path.insert(0, f"{TREE}/refe")
    import train as T
    t0 = time.time()
    bank = T.OnPolicyBank(f"{sp}/sets", 64, 20, pdm_labels=f"{sp}/pdm", pdm_only=True)
    served = {(l, t, s, r): c for (l, t, s, r, c) in plan["sets"]}
    prov["bank"] = {"rows": bank.n_rows, "incomplete": bank.n_incomplete, "superseded": bank.n_superseded, "bad_lines": bank.n_bad_lines, "keys_in_bank_after_pdm_only": len(bank.by),
                    "n_pdm": bank.n_pdm, "n_pdm_stale": bank.n_pdm_stale, "n_pdm_superseded": bank.n_pdm_superseded, "dropped_non_pdm": bank.n_dropped_non_pdm, "seconds": round(time.time() - t0)}
    missing = sorted(k for k in served if k not in bank.by)
    extra = sorted(k for k in bank.by if k not in served)
    ck_bad = sorted(k for k in served if k in bank.by and int(bank.by[k][0]) != int(served[k]))
    prov["checks"] = {"served_keys": len(served), "served_missing_from_bank": len(missing), "bank_keys_not_served": len(extra), "ckpt_step_differs_from_served_keys": len(ck_bad),
                      "missing_examples": [list(k) for k in missing[:3]], "extra_examples": [list(k) for k in extra[:3]]}
    print(json.dumps(prov["bank"]), flush=True)
    print(json.dumps(prov["checks"]), flush=True)
    if missing or ck_bad or bank.n_bad_lines:
        with open(f"{a.plan_dir}/podsets_provenance.json", "w", encoding="utf-8") as w:
            json.dump(prov, w, indent=1)
        raise SystemExit("rc2 the fit sets do not reproduce: missing / ckpt_step differs / bad lines (podsets_provenance.json)")
    # ---- per-log stores (keys in served order): traj3 [S,64,20,3] (x, y, yaw as the trainer's `ot`), labels6 [S,64,6] (NC, DAC, EP, TTC, C, DDC; clipped), ckpt_step
    by_log: dict = {}
    for k in sorted(served):
        by_log.setdefault(k[0], []).append(k)
    n_sets = 0
    for lg, ks in sorted(by_log.items()):
        tr3 = np.stack([bank.by[k][1] for k in ks]).astype(np.float32)
        lab = np.stack([bank.by[k][2] for k in ks]).astype(np.float32)
        assert tr3.shape[1:] == (64, 20, 3) and lab.shape[1:] == (64, 6) and np.isfinite(tr3).all() and np.isfinite(lab).all(), lg
        np.savez(f"{a.work}/sets/{lg}.npz", token=np.array([k[1] for k in ks]), step=np.array([k[2] for k in ks], np.int32), rank=np.array([k[3] for k in ks], np.int32),
                 ckpt_step=np.array([int(bank.by[k][0]) for k in ks], np.int64), traj3=tr3, labels6=lab, label_order=np.array(list(T.OnPolicyBank.PDM_ORDER)))
        n_sets += len(ks)
    prov["per_log_stores"] = {"logs": len(by_log), "sets": n_sets}
    assert n_sets == len(served) == 15808
    with open(f"{a.plan_dir}/podsets_provenance.json", "w", encoding="utf-8") as w:
        json.dump(prov, w, indent=1)
    print(f"[{now()}] PODSETS_OK logs={len(by_log)} sets={n_sets}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
