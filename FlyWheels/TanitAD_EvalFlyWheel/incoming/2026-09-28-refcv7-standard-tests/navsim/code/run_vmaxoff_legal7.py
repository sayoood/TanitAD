#!/usr/bin/env python3
"""refcv7 step-50,400 LEGAL-input NavSim row = ``R7_VMAXOFF`` on FULL navtest and FULL navhard (TANITAD VENV).

    python code/run_vmaxoff_legal7.py --splits navtest,navhard

WHY (SPEC_REFCV8_DRAFT sec. 6.2, BAR-R8-N4; PI Q3): NavSim's ``AgentInput`` has no map, so a posted speed
limit is PRIVILEGED. ``R7_A1`` (the bar arm) feeds the map limit and is therefore NOT a leaderboard-legal
baseline. ``R7_VMAXOFF`` feeds, per ``refcv7_bridge.py:68-69`` + ``refcv6_bridge.py:83-86,181-186``:
  frames (static-t0 history) + ego pose/velocity at t0-1.0, t0-0.5, t0 + the BARE ``driving_command[3]``
  + ``v_max_ms = 0.0, v_max_valid = 0.0`` on EVERY scene (the all-zero one-hot = the UNKNOWN row,
  ``refcv6_max_speed.py:188-211``; the ceiling filter reads +inf there, ``refc_v3.py:1782-1786``);
  ``lan = ego_state = nav_args = agent_gt = None`` (``refcv7_bridge.py:282``: no route checkpoint, no turn
  distance). This script re-derives that from the BANKED rows after the bridge ran (``verify_legal``) and
  refuses to score a seam whose rows say anything else.

IT ROLLS THE SAME WAY R7_A1 WAS ROLLED, or the pair is not a pair: CUDA, ``precision as_trained``,
exact-dedup = whatever R7_A1's seam manifest recorded, the same tree commit and checkpoint md5; K0 is
re-measured on CUDA first and a K0 failure STOPS the split (no silent CPU fallback: a different device
would contaminate every paired delta against R7_A1). Bridges run behind the dev-box GPU lock
(``gpu_lock.py``) and release it BEFORE scoring; scoring runs through ``score_queue_gov7.py`` (one scorer
at a time, ``ram_governor7``), in the background while the next split bridges.

Outputs (never under the runner's directories): ``<out>/bridge_<split>/``, ``<out>/scores_<split>/``,
``<out>/LEGAL_ROW_MANIFEST.json``, ``<out>/vmaxoff_legal.log``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gpu_lock  # noqa: E402
import run_navsim_refcv7 as R  # noqa: E402
from pytest_control7 import control_passed  # noqa: E402

ARM = "R7_VMAXOFF"


def verify_legal(rows_jsonl: str, n_expected: int) -> dict:
    """Re-derive the LEGAL claim from the banked rows. -> {'ok': bool, 'why': [...], 'counts': {...}}."""
    why, n, valid0, ceil_inf, declared, src = [], 0, 0, 0, set(), {}
    with open(rows_jsonl, encoding="utf-8") as fh:
        for ln in fh:
            r = json.loads(ln)
            n += 1
            src[r.get("source")] = src.get(r.get("source"), 0) + 1
            if r.get("source") != "refcv7":
                continue                                         # a declared stand-in carries no model input
            v = r["vmax"]
            if v["v_max_valid"] == 0.0 and v["v_max_ms"] == 0.0 and v["why"] == "withheld by the arm":
                valid0 += 1
            if r["diag"]["ceiling_read_ms"] == float("inf"):
                ceil_inf += 1
            declared.add(tuple(sorted(k for k in r["declared_values"] if not k.startswith("_"))))
    n_model = src.get("refcv7", 0)
    want = tuple(sorted(["ego_pose[1]", "ego_pose[2]", "ego_pose[3]", "ego_velocity[1]",
                         "ego_velocity[2]", "ego_velocity[3]", "driving_command[3]"]))
    if n != n_expected:
        why.append(f"{n} rows != {n_expected}")
    if valid0 != n_model:
        why.append(f"only {valid0}/{n_model} model rows carry v_max_valid=0 / v_max_ms=0 / 'withheld by the arm'")
    if ceil_inf != n_model:
        why.append(f"only {ceil_inf}/{n_model} model rows read an infinite ceiling in the filter")
    if declared != {want}:
        why.append(f"declared ego/route fields {sorted(declared)} != {want}")
    return {"ok": not why, "why": why,
            "counts": {"rows": n, "model_rows": n_model, "sources": src, "vmax_unknown_row": valid0,
                       "filter_ceiling_inf": ceil_inf, "declared_sets": [list(d) for d in declared]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="D:/refcv7_eval_kit/ckpt/ckpt_50400.pt")
    ap.add_argument("--md5", default="b418d0fc4a92a6848c246a6a7c50207b")
    ap.add_argument("--splits", default="navtest,navhard")
    ap.add_argument("--out", default=os.path.join(PKG, "raw", "milestones", "step50400", "vmaxoff_legal"))
    ap.add_argument("--ref-bridge-root", default=os.path.join(PKG, "raw", "milestones", "step50400"),
                    help="where R7_A1's seam manifests live (the device protocol is copied from them)")
    ap.add_argument("--gpu-wait-s", type=int, default=43200)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args(argv)
    if os.environ.get("R7_MUTATION"):
        sys.exit("R7_MUTATION is set -- a mutation-test environment never scores a checkpoint")
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    qlog = os.path.join(out, "vmaxoff_legal.log")
    md5 = R.md5_file(a.ckpt)
    if md5 != a.md5:
        sys.exit(f"checkpoint md5 {md5} != {a.md5}")
    # the step is read in a CHILD process: importing torch here would keep ~0.5 GB resident for the many hours
    # this process spends waiting for the GPU lock on a box whose free RAM is the scarce resource
    r = subprocess.run([R.PY, "-c", "import sys,torch;print(int(torch.load(sys.argv[1],map_location='cpu',"
                        "weights_only=False,mmap=True).get('step',-1)))", a.ckpt],
                       capture_output=True, text=True, timeout=600)
    step = int(r.stdout.strip().splitlines()[-1])
    label = "RESULT" if step >= 5000 else "VALIDATION ONLY (SPEC 6)"
    R.log(f"START LEGAL ROW arm={ARM} step={step} md5={md5} splits={a.splits}", qlog)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{R.TREE}/stack;{R.TREE}/taniteval"
    env["TANITAD_REPO"] = R.TREE
    env["OMP_NUM_THREADS"] = str(a.threads)
    env["PYTHONIOENCODING"] = "utf-8"
    manifest = {"arm": ARM, "step": step, "ckpt_md5": md5, "label": label, "splits": {}}
    scorers = []
    for sk in [s for s in a.splits.split(",") if s]:
        sp = R.SPLITS[sk]
        bdir = os.path.join(out, f"bridge_{sk}")
        sdir = os.path.join(out, f"scores_{sk}")
        # --- the protocol R7_A1 was rolled under (read from ITS manifest, never retyped) --------
        ref = R.jget(os.path.join(a.ref_bridge_root, f"bridge_{sk}", "seam_R7_A1.manifest.json"))
        if not ref:
            sys.exit(f"no R7_A1 seam manifest for {sk} under {a.ref_bridge_root} -- cannot copy its protocol")
        ref_dev, ref_prec = ref["device"][0], ref["precision"][0]
        ref_dedup = bool((ref.get("model") or {}).get("exact_dedup"))
        ref_commit = (ref.get("model") or {}).get("tree", {}).get("tree_commit")
        rec = {"ref_R7_A1": {"device": ref_dev, "precision": ref_prec, "exact_dedup": ref_dedup,
                             "tree_commit": ref_commit}}
        manifest["splits"][sk] = rec
        if ref_dev != "cuda":
            sys.exit(f"R7_A1 on {sk} ran on {ref_dev}; this script only reproduces a CUDA roll")
        seam_state = R.seams_ready(bdir, [ARM]) if os.path.isdir(bdir) else {ARM: "NO_SEAM"}
        if seam_state[ARM] != "OK":
            tok = gpu_lock.acquire(f"refcv7-navsim-vmaxoff-{sk}", a.gpu_wait_s, poll_s=20.0,
                                   log=lambda m: None)
            if not tok:
                R.log(f"BLOCKED {sk}: no GPU lock within {a.gpu_wait_s} s (held by "
                      f"{gpu_lock.read()}) -- nothing was bridged", qlog)
                rec["blocked"] = "GPU lock not acquired"
                continue
            R.log(f"GPU lock ACQUIRED for {sk} ({tok})", qlog)
            try:
                ce = dict(env, R7_TEST_DEVICE="cuda", R7_TEST_CKPT=a.ckpt, R7_TEST_CONFIG=R.CONFIG)
                clog = os.path.join(out, f"cuda_controls_{sk}.log")
                rc = R.run([R.PY, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider",
                            os.path.join(PKG, "tests", "test_model_seam7.py"), "-k", "K0 or KD"], ce, clog)
                txt = open(clog, encoding="utf-8", errors="replace").read().replace("\\", "/")
                k0, kd = control_passed(txt, "test_K0"), control_passed(txt, "test_KD")
                rec["cuda_controls"] = {"pytest_rc": rc, "K0_pass": k0, "KD_pass": kd}
                R.log(f"CUDA controls {sk}: K0={k0} KD={kd} (R7_A1 used exact_dedup={ref_dedup})", qlog)
                if not k0:
                    R.log(f"STOP {sk}: K0 failed on CUDA -- not falling back to CPU (device parity with "
                          f"R7_A1)", qlog)
                    rec["blocked"] = "K0 failed on CUDA"
                    continue
                if ref_dedup and not kd:
                    R.log(f"STOP {sk}: R7_A1 used the exact dedup but KD fails now -- cannot match", qlog)
                    rec["blocked"] = "KD mismatch"
                    continue
                R.log(f"BRIDGE {sk} arm={ARM} device=cuda dedup={ref_dedup}", qlog)
                R.run(R.bridge_cmd(sp, sk, ARM, a.ckpt, md5, "cuda", tok, label, bdir, a.threads,
                                   derived=False, dedup=ref_dedup), env, bdir + ".log")
            finally:
                R.log(f"GPU lock RELEASED={gpu_lock.release(tok)} after {sk}", qlog)
        seam_state = R.seams_ready(bdir, [ARM])
        R.log(f"SEAM {sk} {seam_state}", qlog)
        if seam_state[ARM] != "OK":
            rec["blocked"] = f"seam {seam_state[ARM]}"
            continue
        man = R.jget(os.path.join(bdir, f"seam_{ARM}.manifest.json"))
        same = {"device": man["device"][0] == ref_dev, "precision": man["precision"][0] == ref_prec,
                "exact_dedup": bool((man.get("model") or {}).get("exact_dedup")) == ref_dedup,
                "ckpt_md5": (man.get("model") or {}).get("ckpt_md5") == md5,
                "tree_commit": (man.get("model") or {}).get("tree", {}).get("tree_commit") == ref_commit,
                "KPR": (man.get("KPR") or {}).get("verdict") == "PASS",
                "n_cv_standin_equal": man.get("n_cv_standin_rows") ==
                R.jget(os.path.join(a.ref_bridge_root, f"bridge_{sk}", "seam_R7_A1.manifest.json"),
                       "n_cv_standin_rows")}
        legal = verify_legal(os.path.join(bdir, f"rows_{ARM}.jsonl"), int(man["n_expected"]))
        rec.update({"seam_manifest": os.path.join(bdir, f"seam_{ARM}.manifest.json"),
                    "protocol_matches_R7_A1": same, "legal_verification": legal})
        R.log(f"PROTOCOL {sk} == R7_A1: {same}", qlog)
        R.log(f"LEGAL VERIFICATION {sk}: ok={legal['ok']} {legal['counts']} {legal['why']}", qlog)
        if not all(same.values()) or not legal["ok"]:
            R.log(f"REFUSED {sk}: not scoring a seam that is not the LEGAL R7_A1-protocol roll", qlog)
            rec["blocked"] = "protocol / legality check failed"
            continue
        if a.no_score:
            continue
        cmd = [R.PY, os.path.join(HERE, "score_queue_gov7.py"), "--split", sk, "--bridge", bdir,
               "--scores", sdir, "--step", str(step), "--arms", ARM]
        if sk == "navtest":
            cmd += ["--label-prefix", f"r7s{step}_"]
        os.makedirs(sdir, exist_ok=True)
        lf = open(os.path.join(sdir, "queue.stdout.txt"), "a", encoding="utf-8")
        scorers.append((sk, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env), lf))
        R.log(f"SCORER queued for {sk} (pid {scorers[-1][1].pid})", qlog)
        with open(os.path.join(out, "LEGAL_ROW_MANIFEST.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1, default=str)
    for sk, p, lf in scorers:
        p.wait()
        lf.close()
        R.log(f"SCORER {sk} exited", qlog)
    with open(os.path.join(out, "LEGAL_ROW_MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    R.log("DONE (parse with code/vmaxoff_legal_summary7.py)", qlog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
