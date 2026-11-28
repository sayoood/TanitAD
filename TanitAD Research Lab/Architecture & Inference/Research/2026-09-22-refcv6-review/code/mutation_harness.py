#!/usr/bin/env python
"""Q5 — the mutation harness: construct the REAL regression each guard claims
to catch, and record which guards stay GREEN.

⛔ THE RULES THIS HARNESS ENFORCES ON ITSELF (advisory class F applies to
instruments too — the advisory's own package shipped an inert arm):

  1. **Anchors are WHOLE LINES matched by EQUALITY**, after stripping the line
     terminator. These files are CRLF; a substring anchor collides across
     indentation levels and silently patches the wrong site.
  2. **An arm whose anchor does not apply ABORTS THE RUN AS INVALID.** It is
     never skipped, never counted as "not caught". An anchor that matches 0 or
     >1 lines is a harness defect, and a harness that hides one manufactures
     findings.
  3. **The mutation is applied to the SOURCE the guard reads**, never to the
     test. A test edited to fail proves nothing.
  4. **Every arm carries a BASELINE**: the same test set must be GREEN before
     the mutation. An arm whose baseline is already red is INVALID.
  5. **Subprocesses are decoded utf-8/replace.** `text=True` uses the parent's
     cp1252 here and returns EMPTY streams on one stray byte, which scores as
     "not caught".
  6. **Exit status is read off the CompletedProcess**, never through a pipe,
     and the VERDICT is read from pytest's own summary line in the captured
     output — the artifact, not the status code alone. A run with neither is
     INCONCLUSIVE.
  7. The tree under mutation is a **COPY** (`--tree`), never the repo.

Usage:
  python mutation_harness.py --tree C:/.../mt/stack --json out.json [--only A1,A2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time

PY = r"C:/Users/Admin/venvs/tanitad/Scripts/python.exe"

# --------------------------------------------------------------------------- #
# THE ARMS. `anchor` is a whole line WITHOUT its terminator; `repl` replaces it.
# `claim` is what the guard layer asserts; `expect` is what SHOULD happen if the
# claim is true. A disagreement between `expect` and the measurement is the
# finding.
# --------------------------------------------------------------------------- #
ARMS: list[dict] = [
    dict(
        id="A1-lr-constructor",
        file="tanitad/models/timm_trunk.py",
        anchor='        {"params": enc, "lr": lr * float(encoder_lr_mult),',
        repl='        {"params": enc, "lr": lr * 1.0,',
        tests=["tests/test_refcv6_trunk.py"],
        claim="SPEC §2: the encoder group runs at 0.5x the head lr "
              "(param_groups_dd builds it)",
        expect="RED",
        note="the CONSTRUCTOR half of the recipe."),
    dict(
        id="A2-lr-consumer-ZEROED",
        file="scripts/refc_v3_train.py",
        anchor='            g["lr"] = args.lr * sched(step)',
        repl='            g["lr"] = 0.0',
        tests=["tests/test_refcv6_trunk.py", "tests/test_refc_v3.py",
               "tests/test_refcv3_arm.py", "tests/test_refcv6_diffusion.py",
               "tests/test_refcv6_perception_training.py",
               "tests/test_refcv6_tactical_training.py"],
        claim="the learning rate the optimiser actually steps with is the one "
              "the recipe specifies",
        expect="RED",
        note="EVERY parameter group at lr 0.0 — the model cannot learn at all. "
             "If this is GREEN, no guard reads the trained learning rate."),
    dict(
        id="A3-lr-consumer-NO-DECAY",
        file="scripts/refc_v3_train.py",
        anchor='            g["lr"] = args.lr * sched(step)',
        repl='            g["lr"] = args.lr',
        tests=["tests/test_refcv6_trunk.py", "tests/test_refc_v3.py",
               "tests/test_refcv3_arm.py"],
        claim="SPEC §2: warm-up then cosine",
        expect="RED",
        note="the schedule is computed and thrown away — flat lr for 30k steps."),
    dict(
        id="A4-weight-decay-torch-default",
        file="scripts/refc_v3_train.py",
        anchor='    ap.add_argument("--weight-decay", type=float, default=1e-4,',
        repl='    ap.add_argument("--weight-decay", type=float, default=1e-2,',
        tests=["tests/test_refcv6_trunk.py", "tests/test_refc_v3.py"],
        claim="SPEC §2: weight decay 1e-4, NOT torch AdamW's 1e-2",
        expect="RED",
        note="exactly the trap the flag's own help text names."),
    dict(
        id="A5-encoder-lr-mult-default",
        file="scripts/refc_v3_train.py",
        anchor='    ap.add_argument("--encoder-lr-mult", type=float, default=0.5,',
        repl='    ap.add_argument("--encoder-lr-mult", type=float, default=1.0,',
        tests=["tests/test_refcv6_trunk.py", "tests/test_refc_v3.py"],
        claim="SPEC §2: encoder lr x0.5 of the heads",
        expect="RED",
        note="the launch default that decides the shipped arm."),
    dict(
        id="A6-il-grad-clip",
        file="scripts/refc_v3_train.py",
        anchor='        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)',
        repl='        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-08)',
        tests=["tests/test_refcv6_trunk.py", "tests/test_refc_v3.py",
               "tests/test_refcv6_grad_conflict.py",
               "tests/test_grad_budget_honesty.py"],
        claim="the IL trainer's gradient clip is a chosen value",
        expect="RED",
        note="max-norm 1e-8 annihilates every gradient. SPEC §10.7 measured "
             "that a too-small clip is a real, shipped failure mode."),
    dict(
        id="A7-knob-derivation-M18",
        file="scripts/refc_v3_train.py",
        anchor='        and any(o.startswith("--agent") or o.startswith("--w-")',
        repl='        and any(o.startswith("--agent") or o.startswith("--NOPE-")',
        tests=["tests/test_refc_v3_agent_provenance.py",
               "tests/test_built_heads_receive_gradient.py",
               "tests/test_occ_knob_is_stamped.py",
               "tests/test_cls_weight_stamp.py",
               "tests/test_goal_point_trainer_flags.py"],
        claim="assert_knobs_stamped: every --agent-*/--w-* knob reaches "
              "config.json (mm-decisions M18)",
        expect="RED",
        note="THE ACTUAL M18 DEFECT: w_agent/w_u0 absent from config.json. "
             "Here every --w-* knob stops being derived, so none is stamped."),
    dict(
        id="A7b-knob-derivation-M18-valid",
        file="scripts/refc_v3_train.py",
        anchor='                or o.startswith("--bev-aux") or o.startswith("--wp-index")',
        repl='                or o.startswith("--NOPE-aux") or o.startswith("--NOPE-index")',
        tests=["tests/test_refc_v3_agent_provenance.py",
               "tests/test_built_heads_receive_gradient.py",
               "tests/test_occ_knob_is_stamped.py",
               "tests/test_cls_weight_stamp.py"],
        claim="assert_knobs_stamped: --bev-aux*/--wp-index knobs reach "
              "config.json (the 2026-09-07 widening)",
        expect="RED",
        note="the WP-D half of M18: --bev-aux-occlusion is what separates the "
             "pre-registered arm from its deliberate-regression twin."),
    dict(
        id="A8-knob-stamp-EMPTIED",
        file="scripts/refc_v3_train.py",
        anchor='        "agent_knobs": agent_knob_stamp(args),',
        repl='        "agent_knobs": {},',
        tests=["tests/test_refc_v3_agent_provenance.py",
               "tests/test_built_heads_receive_gradient.py",
               "tests/test_occ_knob_is_stamped.py",
               "tests/test_cls_weight_stamp.py"],
        claim="assert_knobs_stamped catches a stamp that does not carry the knobs",
        expect="RED",
        note="the CONTROL for A7: this half of the guard should work."),
    dict(
        id="A9-maneuver-budget-doubled",
        file="scripts/refc_v3_train.py",
        anchor='            + (LAT_WEIGHT / 2.0) * (loss_lat + loss_lat_tac)',
        repl='            + LAT_WEIGHT * (loss_lat + loss_lat_tac)',
        tests=["tests/test_refc_v3.py", "tests/test_refcv3_arm.py",
               "tests/test_refc_tactical.py", "tests/test_goal_tac.py",
               "tests/test_grad_budget_honesty.py"],
        claim="the tactical aux pressure is held at EXACTLY MANEUVER_WEIGHT "
              "(the /2 that refc_v3_train.py:3636-3643 documents as 'the fix, "
              "not a typo')",
        expect="RED",
        note="restores the DOUBLE loss budget refcv3 shipped with."),
    dict(
        id="A10-rl-grad-clip-off",
        file="tanitad/rl/config.py",
        anchor="    grad_clip: float = 1.0",
        repl="    grad_clip: float = 0.0",
        tests=["tests/test_rl_posttrain.py", "tests/test_ddv2_il.py",
               "tests/test_rl_config.py"],
        claim="the GRPO post-train library's gradient clip is a declared value",
        expect="RED",
        note="0.0 is falsy, so posttrain.py:454's `if cfg.grad_clip:` skips "
             "clipping entirely — the release behaviour, silently."),
    dict(
        id="A11-f6-w-u0-default",
        file="scripts/refc_v3_train.py",
        anchor="U0_WEIGHT_DEFAULT = 0.0           # WP-4: the x0 loss, in CONTROL space",
        repl="U0_WEIGHT_DEFAULT = 1.0           # WP-4: the x0 loss, in CONTROL space",
        tests=["tests/test_refcv6_diffusion.py", "tests/test_refc_v3.py",
               "tests/test_refc_v3_refcv5_wiring.py",
               "tests/test_v6_effective_weights.py"],
        claim="SPEC §3 F6: `--w-u0 0` — DD has ONE reconstruction loss and "
              "ours duplicated it",
        expect="RED",
        note="flips the F6 decision by changing a module constant."),
    dict(
        id="A12-warmup-default-zero",
        file="scripts/refc_v3_train.py",
        anchor='    ap.add_argument("--warmup", type=int, default=2000)',
        repl='    ap.add_argument("--warmup", type=int, default=0)',
        tests=["tests/test_refcv6_trunk.py", "tests/test_refc_v3.py",
               "tests/test_refcv3_arm.py"],
        claim="SPEC §2: warm-up then cosine",
        expect="RED",
        note="no warm-up at all — the ImageNet prior meets a full-rate step 0."),
]


def _split_lines(raw: bytes) -> tuple[list[bytes], list[bytes]]:
    """-> (line bodies without terminator, terminators). CRLF-safe."""
    out_b, out_t, i, n = [], [], 0, len(raw)
    start = 0
    while i < n:
        if raw[i:i + 2] == b"\r\n":
            out_b.append(raw[start:i]); out_t.append(b"\r\n"); i += 2; start = i
        elif raw[i:i + 1] in (b"\n", b"\r"):
            out_b.append(raw[start:i]); out_t.append(raw[i:i + 1]); i += 1; start = i
        else:
            i += 1
    if start < n:
        out_b.append(raw[start:]); out_t.append(b"")
    return out_b, out_t


def apply_mutation(path: pathlib.Path, anchor: str, repl: str) -> tuple[bytes, dict]:
    """Replace the UNIQUE whole line equal to `anchor`. -> (original bytes, info).

    ⛔ Raises on 0 or >1 matches. That is an INVALID arm, never a skip.
    """
    raw = path.read_bytes()
    bodies, terms = _split_lines(raw)
    a = anchor.encode("utf-8")
    hits = [i for i, b in enumerate(bodies) if b == a]
    if len(hits) != 1:
        raise SystemExit(
            f"INVALID ARM: anchor matched {len(hits)} whole lines in {path} "
            f"(need exactly 1). anchor={anchor!r}")
    i = hits[0]
    bodies[i] = repl.encode("utf-8")
    new = b"".join(b + t for b, t in zip(bodies, terms))
    path.write_bytes(new)
    return raw, {"line": i + 1, "crlf": terms[i] == b"\r\n",
                 "sha_before": hashlib.sha256(raw).hexdigest()[:16],
                 "sha_after": hashlib.sha256(new).hexdigest()[:16]}


def run_tests(tree: pathlib.Path, tests: list[str], tag: str,
              logdir: pathlib.Path) -> dict:
    """-> {'rc', 'passed', 'failed', 'errors', 'summary', 'log'} — decoded utf-8."""
    present = [t for t in tests if (tree / t).exists()]
    absent = [t for t in tests if not (tree / t).exists()]
    if not present:
        return {"rc": None, "verdict": "INVALID", "absent": absent,
                "summary": "no named test file exists"}
    cp = subprocess.run(
        [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", *present],
        cwd=str(tree), capture_output=True, encoding="utf-8",
        errors="replace",
        env={**__import__("os").environ,
             "PYTHONPATH": str(tree), "OMP_NUM_THREADS": "6"},
        timeout=3600)
    out = (cp.stdout or "") + (cp.stderr or "")
    log = logdir / f"{tag}.log"
    log.write_text(out, encoding="utf-8", errors="replace")
    # ⭐ ASSERT ON THE ARTIFACT: pytest's own summary line, not the exit code.
    m = re.findall(r"^(\d+) (passed|failed|error|errors)", out, re.M)
    tail = [ln for ln in out.splitlines() if re.search(
        r"\d+ (passed|failed|error|skipped|deselected)", ln)]
    summary = tail[-1] if tail else ""
    n_pass = int(re.search(r"(\d+) passed", summary).group(1)) if "passed" in summary else 0
    n_fail = int(re.search(r"(\d+) failed", summary).group(1)) if "failed" in summary else 0
    n_err = int(re.search(r"(\d+) error", summary).group(1)) if "error" in summary else 0
    if not summary:
        verdict = "INCONCLUSIVE"     # empty output is never a pass
    elif n_fail or n_err:
        verdict = "RED"
    elif n_pass:
        verdict = "GREEN"
    else:
        verdict = "INCONCLUSIVE"
    return {"rc": cp.returncode, "verdict": verdict, "summary": summary,
            "passed": n_pass, "failed": n_fail, "errors": n_err,
            "absent": absent, "log": str(log), "n_files": len(present),
            "raw_counts": m}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--only", default=None, help="comma-separated arm ids")
    a = ap.parse_args()
    tree = pathlib.Path(a.tree)
    logdir = pathlib.Path(a.logdir); logdir.mkdir(parents=True, exist_ok=True)
    if not (tree / "tanitad" / "__init__.py").exists():
        raise SystemExit(f"INVALID: {tree} is not a stack tree")
    if "Projects" in str(tree):
        raise SystemExit("INVALID: refusing to mutate the repo — pass a COPY")

    arms = ARMS if not a.only else [x for x in ARMS
                                    if x["id"] in a.only.split(",")]
    results = []
    for arm in arms:
        path = tree / arm["file"]
        t0 = time.time()
        print(f"\n=== {arm['id']} === {arm['file']}")
        base = run_tests(tree, arm["tests"], f"{arm['id']}__baseline", logdir)
        print(f"  baseline : {base['verdict']:<12} {base.get('summary','')}"
              + (f"   absent={base['absent']}" if base.get("absent") else ""))
        if base["verdict"] != "GREEN":
            results.append({**{k: arm[k] for k in
                               ("id", "file", "claim", "expect", "note")},
                            "baseline": base, "mutated": None,
                            "RESULT": "INVALID_BASELINE"})
            print("  ⛔ INVALID: baseline not green; arm cannot testify.")
            continue
        orig, info = apply_mutation(path, arm["anchor"], arm["repl"])
        try:
            mut = run_tests(tree, arm["tests"], f"{arm['id']}__mutated", logdir)
        finally:
            path.write_bytes(orig)
            back = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            if back != info["sha_before"]:
                raise SystemExit(f"⛔ RESTORE FAILED for {path}: {back} != "
                                 f"{info['sha_before']}")
        print(f"  mutated  : {mut['verdict']:<12} {mut.get('summary','')}")
        caught = mut["verdict"] == "RED"
        res = "CAUGHT" if caught else (
            "NOT_CAUGHT" if mut["verdict"] == "GREEN" else "INCONCLUSIVE")
        print(f"  -> {res}   (expected {arm['expect']})   "
              f"line {info['line']} crlf={info['crlf']}  "
              f"{time.time() - t0:.0f}s")
        results.append({**{k: arm[k] for k in
                           ("id", "file", "claim", "expect", "note")},
                        "anchor": arm["anchor"], "repl": arm["repl"],
                        "site": info, "baseline": base, "mutated": mut,
                        "RESULT": res})

    pathlib.Path(a.json).write_text(json.dumps(results, indent=2),
                                    encoding="utf-8")
    print("\n================ SUMMARY ================")
    for r in results:
        print(f"  {r['RESULT']:<18} {r['id']:<28} {r['file']}")
    n_green = sum(1 for r in results if r["RESULT"] == "NOT_CAUGHT")
    print(f"\n  arms: {len(results)}   NOT_CAUGHT (the finding): {n_green}")
    print(f"[artifact] {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
