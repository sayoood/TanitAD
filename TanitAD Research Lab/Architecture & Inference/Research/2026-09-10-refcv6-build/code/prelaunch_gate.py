#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv6 -- ⛔ THE PRE-LAUNCH GATE. Nothing starts until this writes PASS.

    PYTHONPATH=/workspace/TanitAD/stack MSYS_NO_PATHCONV=1 \
    python3 prelaunch_gate.py \
        --trainer /workspace/TanitAD/stack/scripts/refc_v3_train.py \
        --labels  /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz \
        --agent-join /root/data/joins/train2400_agents.jsonl.xz \
        --anchors /root/data/refcv6/anchors.pt \
        --w-agent <W> \
        --json /workspace/refcv6/raw/prelaunch_gate.json

⛔⛔ THE VERDICT IS THE JSON FILE, NOT THE EXIT CODE.

    *"The admissible evidence that this gate did not run is the MISSING JSON,
    never the exit code."*

MEASURED 2026-09-07, twice in two days: a 25-minute `timeout` killed
`pod_currency_audit.py` with ZERO lines of output and no `--json` file written,
and its wrapper printed `GATE_EXIT=0`. A timed-out, output-less PRE-LAUNCH GATE
reported clean. `$?` after a pipeline is the LAST element's status, so
`gate.py | tail` reports tail's success. ⇒ the caller MUST read this file and
require `"verdict": "PASS"`; a missing file is a FAILED gate, never a passed one.

# The five checks, and the failure each exists to prevent

C1  BASE PROVENANCE     `arms.BASE_V5V2` equals the banked run's own
                        `config.json['argv']`. A hand-transcribed baseline is a
                        different experiment wearing the same name.

C2  ONE VARIABLE        every arm pair moves exactly ONE lever key, on PARSED
                        namespaces, through the trainer's OWN parser. The `--v2`
                        conflation failure (ten levers on two axes) is the
                        precedent.

C3  IMPORT CLOSURE      `launch_closure_audit.py` over the trainer's real import
                        closure, `MISSING_REMOTE == 0` AND `DRIFT == 0`.
                        ⛔ md5 agreement proves TRANSFER, not FUNCTION -- C99:
                        three green md5s on a 2.6x-stale dependency that was
                        never listed because it had not been edited.
                        ⚠️ Set `MSYS_NO_PATHCONV=1`, or MSYS rewrites the remote
                        root and the tool reports a plausible 120/120
                        MISSING_REMOTE that is pure artifact.

C4  LABEL n JOIN        |labels ∩ join| / |labels| >= --min-coverage (default
                        0.90). ⛔ THE FAILURE THIS EXISTS FOR: a join sharing
                        only 182 clips with v7.2 is 182/4,572 = 3.98 % coverage,
                        and the arm would have trained on ~4 % of its intended
                        supervision and read as *"the lever does not help"* -- a
                        MANUFACTURED NEGATIVE, which is worse than a crash
                        because it looks like a result.
                        Reference: the B1 TRAIN join is 4,427 / 4,572 = 96.83 %.

C5  20-STEP SMOKE       run the real trainer for 20 steps and assert on CONTENT
                        of `metrics.json`: finite loss, loss actually MOVED
                        (a constant loss is a disconnected graph), and the
                        lever's own term present and non-zero where the arm
                        declares it.
                        ⛔ Never the exit code. And never `ls`: a decode that
                        raises into a pre-allocated memmap leaves a FULL-SIZE
                        FILE OF ZEROS and the job can still exit 0.

C6  ANCHOR PARITY       one `anchors.pt`, one md5, shared by every arm. A per-arm
                        anchor vocabulary is a hidden second lever.

Any check that could not be RUN reports `INCONCLUSIVE`, never `PASS`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
import os
import subprocess
import sys
import time

import arms as A
import check_one_variable as C

DEFAULT_MIN_COVERAGE = 0.90

#: MEASURED (`D-B1TRAIN-JOIN-1`, and reproduced by WP-D §1 fact 4): the B1 TRAIN
#: join covers 4,427 of the corpus's 4,572 clips. Quoted as the REFERENCE the
#: gate's own number should land near -- ⛔ never used AS the measurement.
REFERENCE_COVERAGE = {"join_clips": 4427, "label_clips": 4572, "ratio": 0.9683}


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def _open_any(path: str):
    if path.endswith(".xz"):
        return lzma.open(path, "rt", encoding="utf-8")
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _md5(path: str) -> str | None:
    try:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _clip_ids(path: str, keys=("clip_id", "clip", "uid", "episode_id", "id"),
              limit: int | None = None) -> tuple[set[str], int, str | None]:
    """Collect distinct clip ids from a jsonl(.gz|.xz). Returns (ids, n_rows, err)."""
    ids: set[str] = set()
    n = 0
    try:
        with _open_any(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                for k in keys:
                    if k in rec and rec[k] is not None:
                        ids.add(str(rec[k]))
                        break
                if limit and n >= limit:
                    break
    except Exception as exc:
        return ids, n, f"{exc.__class__.__name__}: {exc}"
    return ids, n, None


# --------------------------------------------------------------------------- #
# C1 / C2                                                                      #
# --------------------------------------------------------------------------- #

def check_base_provenance() -> dict:
    v = A.verify_base_against_banked_config()
    return {
        "check": "C1_base_provenance", "detail": v,
        "verdict": "PASS" if v.get("ok") else
                   ("INCONCLUSIVE" if v.get("verdict") == "INCONCLUSIVE" else "FAIL"),
        "why": ("BASE_V5V2 equals the banked refcv5-v2 config.json argv"
                if v.get("ok") else
                "⛔ the baseline in arms.py is NOT the banked run's argv"),
    }


def check_one_variable(trainer: str, *, steps: int, w_agent, w_tac_goal,
                       anchors: str, agent_join: str) -> dict:
    mod, why = C.load_trainer(trainer)
    if mod is None:
        return {"check": "C2_one_variable", "verdict": "INCONCLUSIVE",
                "why": f"could not import the trainer: {why}"}
    ns: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for arm in sorted(A.ARMS):
        try:
            argv = A.arm_argv(arm, steps=steps, anchors=anchors,
                              agent_join=agent_join, w_agent=w_agent,
                              w_tac_goal=w_tac_goal)
        except A.PIDecisionRequired as exc:
            skipped[arm] = f"PI_DECISION_REQUIRED: {exc}"
            continue
        parsed, err = C.parse_arm(mod.build_parser, argv)
        if parsed is None:
            skipped[arm] = f"PARSE_FAILED: {err}"
            continue
        ns[arm] = parsed
    pairs = [C.check_pair(a, A.PAIRING[a], ns[a], ns[A.PAIRING[a]])
             for a in sorted(ns) if A.PAIRING.get(a) in ns]
    bad = [p for p in pairs if p["verdict"] == "REFUSE"]
    return {
        "check": "C2_one_variable",
        "n_pairs": len(pairs), "n_refused": len(bad),
        "arms_skipped": skipped,
        "pairs": pairs,
        "verdict": "PASS" if (pairs and not bad) else
                   ("INCONCLUSIVE" if not pairs else "FAIL"),
        "why": ("every checked pair moves exactly one lever key"
                if (pairs and not bad) else
                "⛔ a pair moves more than its declared lever"),
    }


# --------------------------------------------------------------------------- #
# C3                                                                           #
# --------------------------------------------------------------------------- #

def check_import_closure(*, host: str | None, trainer: str, remote_root: str,
                         audit_tool: str | None, out_json: str,
                         timeout_s: int) -> dict:
    if not host:
        return {"check": "C3_import_closure", "verdict": "INCONCLUSIVE",
                "why": "no --host given; the closure is a property of the BOX and "
                       "cannot be audited from here. ⛔ This is not a pass."}
    if not audit_tool or not os.path.isfile(audit_tool):
        return {"check": "C3_import_closure", "verdict": "INCONCLUSIVE",
                "why": f"launch_closure_audit.py not found at {audit_tool!r}"}
    env = dict(os.environ)
    env["MSYS_NO_PATHCONV"] = "1"   # ⚠️ or the remote root is rewritten -> 120/120 artifact
    cmd = [sys.executable, audit_tool, "--host", host,
           "--remote-root", remote_root, "--entry", trainer,
           "--verify-import", "--json", out_json]
    started = time.time()
    try:
        # ⛔ No pipe: `$?` through a pipe is the pipe's last element. Output is
        #    captured directly so the status belongs to the tool.
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=timeout_s)
        rc, tail = proc.returncode, (proc.stdout or "")[-1500:]
    except subprocess.TimeoutExpired:
        return {"check": "C3_import_closure", "verdict": "INCONCLUSIVE",
                "why": f"⛔ audit TIMED OUT after {timeout_s}s and wrote no "
                       f"verdict. A timed-out gate is a FAILED gate.",
                "elapsed_s": round(time.time() - started, 1)}
    except Exception as exc:
        return {"check": "C3_import_closure", "verdict": "INCONCLUSIVE",
                "why": f"{exc.__class__.__name__}: {exc}"}

    # ⭐ ASSERT ON THE ARTIFACT, not on rc.
    if not os.path.isfile(out_json):
        return {"check": "C3_import_closure", "verdict": "INCONCLUSIVE",
                "why": f"⛔ the audit wrote NO json at {out_json} (rc={rc}). "
                       f"The missing artifact is the evidence, not the rc.",
                "stdout_tail": tail}
    try:
        with open(out_json, "r", encoding="utf-8") as fh:
            rep = json.load(fh)
    except Exception as exc:
        return {"check": "C3_import_closure", "verdict": "INCONCLUSIVE",
                "why": f"audit json unreadable: {exc}"}

    rows = rep.get("rows") or rep.get("files") or []
    counts: dict[str, int] = rep.get("counts") or {}
    if not counts and rows:
        for r in rows:
            v = r.get("verdict", "?")
            counts[v] = counts.get(v, 0) + 1
    missing = int(counts.get("MISSING_REMOTE", 0))
    drift = int(counts.get("DRIFT", 0))
    n_rows = sum(counts.values()) if counts else len(rows)

    # ⚠️ The 120/120 artifact: a total wipe-out is far more likely a path bug
    #    than a genuinely empty box. Refuse to read it as a real finding.
    suspicious = bool(n_rows and missing == n_rows)
    ok = (missing == 0 and drift == 0 and n_rows > 0)
    return {
        "check": "C3_import_closure",
        "host": host, "remote_root": remote_root,
        "counts": counts, "n_rows": n_rows,
        "MISSING_REMOTE": missing, "DRIFT": drift,
        "audit_rc": rc, "artifact": out_json,
        "verdict": ("PASS" if ok else
                    ("INCONCLUSIVE" if (suspicious or n_rows == 0) else "FAIL")),
        "why": ("MISSING_REMOTE=0 and DRIFT=0 over the real import closure" if ok
                else ("⛔ EVERY row is MISSING_REMOTE -- almost certainly a remote-root "
                      "path artifact (set MSYS_NO_PATHCONV=1), not an empty box"
                      if suspicious else
                      ("⛔ the closure audit produced no rows" if n_rows == 0 else
                       f"⛔ MISSING_REMOTE={missing}, DRIFT={drift} -- the box would "
                       f"run different code than the repo"))),
    }


# --------------------------------------------------------------------------- #
# C4                                                                           #
# --------------------------------------------------------------------------- #

def check_label_join_coverage(labels: str, agent_join: str,
                              min_cov: float) -> dict:
    if not labels or not os.path.isfile(labels):
        return {"check": "C4_label_join_coverage", "verdict": "INCONCLUSIVE",
                "why": f"labels not found at {labels!r}"}
    if not agent_join or not os.path.isfile(agent_join):
        return {"check": "C4_label_join_coverage", "verdict": "INCONCLUSIVE",
                "why": f"agent join not found at {agent_join!r}"}

    lab_ids, n_lab, lab_err = _clip_ids(labels)
    join_ids, n_join, join_err = _clip_ids(agent_join)

    # ⛔ A read error must never look like an empty set. An empty intersection
    #    from a file that could not be READ is indistinguishable from a genuine
    #    absence -- so a same-breath positive control is required: both sides
    #    must have produced a non-empty id set.
    if lab_err or join_err or not lab_ids or not join_ids:
        return {"check": "C4_label_join_coverage", "verdict": "INCONCLUSIVE",
                "n_label_rows": n_lab, "n_join_rows": n_join,
                "n_label_clips": len(lab_ids), "n_join_clips": len(join_ids),
                "label_error": lab_err, "join_error": join_err,
                "why": "⛔ one side produced NO clip ids. A zero here is a claim "
                       "about the READ, not about the corpus -- refusing to "
                       "report it as coverage."}

    inter = lab_ids & join_ids
    cov = len(inter) / len(lab_ids)
    ok = cov >= min_cov
    return {
        "check": "C4_label_join_coverage",
        "labels": labels, "agent_join": agent_join,
        "n_label_rows": n_lab, "n_join_rows": n_join,
        "n_label_clips": len(lab_ids), "n_join_clips": len(join_ids),
        "n_intersection": len(inter),
        "coverage": round(cov, 6),
        "min_coverage": min_cov,
        "reference": REFERENCE_COVERAGE,
        "verdict": "PASS" if ok else "FAIL",
        "why": (f"{len(inter)}/{len(lab_ids)} = {cov:.4f} >= {min_cov}" if ok else
                f"⛔ {len(inter)}/{len(lab_ids)} = {cov:.4f} < {min_cov}. An arm on "
                f"this join would train on {cov:.1%} of its intended supervision "
                f"and read as 'the lever does not help' -- a MANUFACTURED "
                f"NEGATIVE. (The 182-clip incident is 182/4572 = 0.0398.)"),
    }


# --------------------------------------------------------------------------- #
# C5                                                                           #
# --------------------------------------------------------------------------- #

def check_smoke(*, trainer: str, arm: str, steps: int, out_dir: str,
                anchors: str, agent_join: str, w_agent, w_tac_goal,
                python_bin: str, timeout_s: int) -> dict:
    """Run the real trainer for `steps` steps; assert on metrics.json CONTENT."""
    try:
        argv = A.arm_argv(arm, steps=steps, out_root=os.path.dirname(out_dir) or ".",
                          anchors=anchors, agent_join=agent_join,
                          w_agent=w_agent, w_tac_goal=w_tac_goal,
                          run_tag=os.path.basename(out_dir).rsplit("-", 1)[0] or "smoke")
    except A.PIDecisionRequired as exc:
        return {"check": "C5_smoke", "arm": arm, "verdict": "INCONCLUSIVE",
                "why": f"PI_DECISION_REQUIRED: {exc}"}
    argv = A._replace_flag(argv, "--out", [out_dir])
    argv = A._replace_flag(argv, "--steps", [str(steps)])
    argv = A._replace_flag(argv, "--warmup", ["1"])
    # ⛔⛔ THIS USED TO PUSH BOTH BEYOND THE SMOKE'S OWN HORIZON (steps + 10), which
    #     made the smoke fast and STRUCTURALLY BLIND to every eval-time and
    #     save-time failure. MEASURED 2026-09-11: refcv6 arm C dies on its FIRST
    #     EVAL -- `tac_goal_targets` is wired train-only, the held-out split hits
    #     the refuse-do-not-skip guard, and `SystemExit` is not an `Exception` so
    #     the eval block's handler cannot catch it. refcv5-v2's argv carries
    #     `--eval-every 500`, so arm C would have died at STEP 500 OF 40,284 --
    #     ~35 GPU-minutes into a ~47 GPU-hour run -- and THIS GATE WOULD HAVE
    #     PASSED IT, because the eval it dies on never ran here.
    # ⭐ That is "a check that shares the defect it checks for": a smoke that skips
    #     the eval can never fail on the eval. Both now fire INSIDE the horizon.
    _every = max(1, steps // 2)
    argv = A._replace_flag(argv, "--eval-every", [str(_every)])
    argv = A._replace_flag(argv, "--save-every", [str(_every)])

    os.makedirs(out_dir, exist_ok=True)
    metrics = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics):
        os.remove(metrics)  # ⛔ a stale artifact would pass this gate

    started = time.time()
    rc = None
    try:
        proc = subprocess.run([python_bin, trainer] + argv,
                              capture_output=True, text=True, timeout=timeout_s)
        rc = proc.returncode
        tail = ((proc.stdout or "")[-1200:], (proc.stderr or "")[-1200:])
    except subprocess.TimeoutExpired:
        tail = ("", f"TIMEOUT after {timeout_s}s")
    except Exception as exc:
        tail = ("", f"{exc.__class__.__name__}: {exc}")

    # ⭐ CONTENT, not rc. And not `ls` either: a full-size file of zeros has the
    #    right size and the wrong contents.
    if not os.path.isfile(metrics):
        return {"check": "C5_smoke", "arm": arm, "verdict": "FAIL",
                "trainer_rc": rc, "elapsed_s": round(time.time() - started, 1),
                "stdout_tail": tail[0], "stderr_tail": tail[1],
                "why": "⛔ no metrics.json. The MISSING ARTIFACT is the evidence; "
                       "the rc is not."}
    try:
        with open(metrics, "r", encoding="utf-8") as fh:
            m = json.load(fh)
    except Exception as exc:
        return {"check": "C5_smoke", "arm": arm, "verdict": "FAIL",
                "why": f"metrics.json unreadable/!json: {exc}"}

    rows = m if isinstance(m, list) else m.get("rows") or m.get("history") or []
    losses = [r.get("loss") for r in rows
              if isinstance(r, dict) and isinstance(r.get("loss"), (int, float))]
    finite = [x for x in losses if x == x and abs(x) != float("inf")]
    moved = len(set(round(float(x), 12) for x in finite)) > 1
    n_steps = len([r for r in rows if isinstance(r, dict) and "step" in r])

    # ⭐ THE EVAL MUST HAVE RUN, and the evidence is the ROW, never the rc.
    #    Arm C exits 1 AFTER metrics.json is written, so a status-blind verdict
    #    passes it while an artifact-based one cannot: a crashed eval leaves no
    #    eval row behind. This keeps the doctrine (assert on the artifact) and
    #    closes the hole (assert on the RIGHT artifact).
    def _is_eval_row(r):
        return isinstance(r, dict) and any(
            str(k).startswith(("eval", "val_")) or str(k) in ("split", "phase")
            and str(r.get(k, "")).startswith(("eval", "val"))
            for k in r)
    n_eval_rows = len([r for r in rows if _is_eval_row(r)])

    ok = (bool(finite) and len(finite) == len(losses) and moved
          and n_steps >= 1 and n_eval_rows >= 1)
    return {
        "check": "C5_smoke", "arm": arm, "steps_requested": steps,
        "trainer_rc": rc, "elapsed_s": round(time.time() - started, 1),
        "n_metric_rows": len(rows), "n_eval_rows": n_eval_rows,
        "eval_every_used": _every, "n_loss_values": len(losses),
        "n_finite": len(finite), "loss_first": finite[0] if finite else None,
        "loss_last": finite[-1] if finite else None,
        "loss_moved": moved, "metrics_path": metrics,
        "stderr_tail": tail[1] if not ok else "",
        "verdict": "PASS" if ok else "FAIL",
        "why": ("metrics.json carries finite, MOVING loss values AND at least "
                "one EVAL row (the eval actually ran)" if ok else
                ("⛔ NO EVAL ROW -- the eval did not run or it crashed. This is "
                 "the arm-C failure: a SystemExit in the eval is not an "
                 "Exception, so the trainer's handler cannot catch it, and the "
                 "process dies AFTER metrics.json exists."
                 if n_eval_rows == 0 else
                 "⛔ loss absent, non-finite, or CONSTANT. A constant loss is a "
                 "disconnected graph, not a converged one.")),
    }


# --------------------------------------------------------------------------- #
# C6                                                                           #
# --------------------------------------------------------------------------- #

def check_anchor_parity(anchors: str) -> dict:
    if not anchors or not os.path.isfile(anchors):
        return {"check": "C6_anchor_parity", "verdict": "INCONCLUSIVE",
                "why": f"anchors not found at {anchors!r}"}
    md5 = _md5(anchors)
    size = os.path.getsize(anchors)
    ok = bool(md5) and size > 0
    return {
        "check": "C6_anchor_parity", "anchors": anchors,
        "md5": md5, "size_bytes": size,
        "verdict": "PASS" if ok else "INCONCLUSIVE",
        "why": ("one anchors file, md5 recorded; every arm's argv points HERE "
                "(arms.py DEFAULT_ANCHORS) and check_one_variable classifies any "
                "per-arm anchors path as VIOLATION_PARITY" if ok else
                "⛔ anchors unreadable or empty"),
    }


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="refcv6 pre-launch gate")
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--agent-join", default=A.DEFAULT_AGENT_JOIN)
    ap.add_argument("--anchors", default=A.DEFAULT_ANCHORS)
    ap.add_argument("--budget", default="full", choices=sorted(A.STEP_BUDGETS))
    ap.add_argument("--w-agent", type=float, default=None)
    ap.add_argument("--w-tac-goal", type=float, default=None)
    ap.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    ap.add_argument("--host", default=None, help="box to audit for C3")
    ap.add_argument("--remote-root", default="/workspace/TanitAD")
    ap.add_argument("--closure-tool", default=None)
    ap.add_argument("--closure-timeout", type=int, default=1800)
    ap.add_argument("--smoke-arm", default="V0")
    ap.add_argument("--smoke-steps", type=int, default=20)
    ap.add_argument("--smoke-out", default=None)
    ap.add_argument("--smoke-timeout", type=int, default=3600)
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--skip-smoke", action="store_true",
                    help="⛔ records C5 as INCONCLUSIVE, which is NOT a pass")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    steps = A.STEP_BUDGETS[a.budget]
    closure_tool = a.closure_tool
    if closure_tool is None:
        for cand in (os.path.join(os.environ.get("TANITAD_STACK", ""), "scripts",
                                  "launch_closure_audit.py"),
                     "/workspace/TanitAD/stack/scripts/launch_closure_audit.py",
                     os.path.join(os.path.dirname(os.path.abspath(a.trainer)),
                                  "launch_closure_audit.py")):
            if cand and os.path.isfile(cand):
                closure_tool = cand
                break

    checks = [
        check_base_provenance(),
        check_one_variable(a.trainer, steps=steps, w_agent=a.w_agent,
                           w_tac_goal=a.w_tac_goal, anchors=a.anchors,
                           agent_join=a.agent_join),
        check_import_closure(host=a.host, trainer=a.trainer,
                             remote_root=a.remote_root, audit_tool=closure_tool,
                             out_json=os.path.join(
                                 os.path.dirname(os.path.abspath(a.json)),
                                 "closure_audit.json"),
                             timeout_s=a.closure_timeout),
        check_label_join_coverage(a.labels, a.agent_join, a.min_coverage),
        check_anchor_parity(a.anchors),
    ]
    if a.skip_smoke:
        checks.append({"check": "C5_smoke", "verdict": "INCONCLUSIVE",
                       "why": "--skip-smoke was passed. ⛔ NOT a pass."})
    else:
        smoke_out = a.smoke_out or os.path.join(
            os.path.dirname(os.path.abspath(a.json)), "smoke-V0")
        checks.append(check_smoke(
            trainer=a.trainer, arm=a.smoke_arm, steps=a.smoke_steps,
            out_dir=smoke_out, anchors=a.anchors, agent_join=a.agent_join,
            w_agent=a.w_agent, w_tac_goal=a.w_tac_goal,
            python_bin=a.python_bin, timeout_s=a.smoke_timeout))

    n_fail = sum(1 for c in checks if c["verdict"] == "FAIL")
    n_inc = sum(1 for c in checks if c["verdict"] == "INCONCLUSIVE")
    verdict = "PASS" if (n_fail == 0 and n_inc == 0) else \
              ("FAIL" if n_fail else "INCONCLUSIVE")

    report = {
        "tool": "prelaunch_gate.py",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trainer": a.trainer, "budget": a.budget, "steps": steps,
        "w_agent": a.w_agent, "w_tac_goal": a.w_tac_goal,
        "checks": checks,
        "n_pass": sum(1 for c in checks if c["verdict"] == "PASS"),
        "n_fail": n_fail, "n_inconclusive": n_inc,
        "verdict": verdict,
        "READ_THIS": ("⛔ The verdict is THIS FILE's 'verdict' field. A missing "
                      "file is a FAILED gate. Never read the exit code through "
                      "a pipe -- `cmd | tail` reports tail's status."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    print(f"[gate] wrote {a.json}")
    for c in checks:
        print(f"  [{c['verdict']:>12}] {c['check']}: {str(c.get('why',''))[:150]}")
    print(f"[gate] verdict = {verdict} "
          f"(pass {report['n_pass']} / fail {n_fail} / inconclusive {n_inc})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
