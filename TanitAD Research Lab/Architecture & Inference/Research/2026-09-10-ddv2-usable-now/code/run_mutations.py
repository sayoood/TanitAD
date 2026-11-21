#!/usr/bin/env python3
"""Deliberate-regression runner for the E-DDA-2b package.

⛔ A GUARD WHOSE FAILURE BRANCH IS UNREACHABLE IS GREEN FOREVER. Each mutation
below reintroduces a defect that has really happened in this programme (or that
the DDv2 analysis names as the faithful-port failure), applies it to the working
tree, runs the suite, records the FAILING test ids, and restores the file from
the authored copy. A mutation that does NOT go RED is itself a finding and is
reported as such.

Usage:  python run_mutations.py <authored_dir> <stack_dir> <out_dir>
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import time

PY = r"C:/Users/Admin/venvs/tanitad/Scripts/python.exe"

#: (id, target file, what it reintroduces, old -> new)
MUTATIONS = [
    ("M1-per-coordinate-noise",
     "tanitad/refs/refc_selector_aug.py",
     "the V1-shaped PER-COORDINATE multiplicative noise instead of V2's TWO "
     "SCALARS (DDv2 analysis section 3.2 names exactly this as the gap)",
     "        eps = torch.randn((b, n, 1, 2), generator=generator,\n"
     "                          device=cand.device, dtype=cand.dtype)",
     "        eps = torch.randn((b, n, s, 2), generator=generator,\n"
     "                          device=cand.device, dtype=cand.dtype)"),

    ("M2-off-path-draws-randomness",
     "tanitad/refs/refc_selector_aug.py",
     "an OFF path that still advances the global RNG, so every downstream arm "
     "silently re-rolls",
     "    if cfg.n_aug == 0:\n"
     "        return cand, torch.zeros((b, n), dtype=torch.long, device=cand.device)",
     "    if cfg.n_aug == 0:\n"
     "        torch.randn((b, n, 1, 2))\n"
     "        return cand, torch.zeros((b, n), dtype=torch.long, device=cand.device)"),

    ("M3-foreign-marked-emittable",
     "tanitad/refs/refc_selector_aug.py",
     "a FOREIGN candidate marked emittable, so the selector may emit a "
     "trajectory the generator cannot produce",
     "    emittable = origin >= NATIVE",
     "    emittable = origin >= FOREIGN"),

    ("M4-dropped-knob-from-the-stamp",
     "scripts/refc_selector_e2b.py",
     "a knob parsed and NOT stamped -- the --wp-index failure (3 of 6 knobs "
     "parsed, stamped and inert)",
     "    for a in group._group_actions:\n        v = getattr(args, a.dest)",
     "    for a in group._group_actions:\n"
     "        if a.dest == 'top_k':\n            continue\n"
     "        v = getattr(args, a.dest)"),

    # ⚠️ MEASURED 2026-09-10: the FIRST version of this mutation replaced the
    # tie-group credit with a `for _k in range(tp_g): ap += (cum_tp/cum_n)`
    # loop -- ARITHMETICALLY THE SAME NUMBER, because cum_tp and cum_n are
    # already advanced to the END of the group before the credit is taken. It
    # read STILL GREEN and it was right to: the mutation shared the property it
    # was meant to break, which is the "check that shares the defect" class
    # appearing inside a deliberate regression. The real defect is TIE
    # DETECTION, so that is what this disables.
    ("M5-naive-AP-ties-by-index",
     "scripts/refc_selector_e2b.py",
     "AP with ties broken by ARRAY ORDER -- the form that scores a CONSTANT arm "
     "off its own base rate and already forced a retraction here",
     "        while j < n and s_sorted[j] == s_sorted[i]:",
     "        while False and j < n and s_sorted[j] == s_sorted[i]:"),

    ("M6-bank-horizon-silently-accepted",
     "tanitad/refs/refc_selector_aug.py",
     "a foreign bank at a DIFFERENT horizon accepted instead of refused -- the "
     "derived-constant trap (HORIZON 7 -> 8 made a reproduction a different "
     "experiment)",
     "    if bank.shape[-2] != s:",
     "    if False and bank.shape[-2] != s:"),
    ("M7-emittable-guard-never-called",
     "scripts/refc_selector_e2b.py",
     "a guard that EXISTS and is never CALLED -- the eval pick could then be a "
     "foreign trajectory the generator cannot emit",
     "            AUG.assert_pick_emittable(out[\"sel_idx_selector\"], ae[\"origin\"],",
     "            _ = (lambda *a, **k: None)(out[\"sel_idx_selector\"], ae[\"origin\"],"),

    ("M8-eval-aug-inert-knob-accepted",
     "scripts/refc_selector_e2b.py",
     "the --foreign-at-eval INERT-knob refusal removed -- the --wp-index "
     "failure (parsed, stamped, does nothing)",
     "        if a.dest == \"foreign_at_eval\" and v:",
     "        if False and a.dest == \"foreign_at_eval\" and v:"),
]

TESTS = ["tests/test_refc_selector_aug.py", "tests/test_refc_selector_e2b.py"]


def run(stack: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (r"C:/Users/Admin/tanitad-wt/stack;"
                         r"C:/Users/Admin/tanitad-wt/taniteval")
    env["PYTHONIOENCODING"] = "utf-8"
    env["OMP_NUM_THREADS"] = "6"
    p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        *TESTS], cwd=stack, env=env,
                       capture_output=True, text=True, errors="replace")
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    authored, stack, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    lines = [f"### DELIBERATE-REGRESSION (mutation) RUN "
             f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} UTC",
             f"### suite: {' '.join(TESTS)}",
             "### a mutation that does NOT go RED is a FINDING, not a pass",
             ""]
    verdicts = []
    for mid, rel, why, old, new in MUTATIONS:
        target = os.path.join(stack, rel)
        src = os.path.join(authored, os.path.basename(rel))
        original = io.open(target, encoding="utf-8").read()
        if old not in original:
            lines.append(f"[{mid}] ⛔ ANCHOR NOT FOUND in {rel} -- the mutation "
                         f"could not be applied, so it proves NOTHING")
            verdicts.append((mid, "ANCHOR-MISSING", []))
            continue
        io.open(target, "w", encoding="utf-8").write(
            original.replace(old, new, 1))
        rc, txt = run(stack)
        failed = sorted(set(re.findall(r"^FAILED (\S+)", txt, re.M)))
        errored = sorted(set(re.findall(r"^ERROR (\S+)", txt, re.M)))
        tail = [l for l in txt.splitlines() if re.search(r"\d+ (passed|failed)", l)]
        io.open(target, "w", encoding="utf-8").write(
            io.open(src, encoding="utf-8").read())
        state = "RED" if (rc != 0) else "⛔ STILL GREEN"
        verdicts.append((mid, state, failed + errored))
        lines += [f"[{mid}] {state}  (exit {rc})",
                  f"    reintroduces: {why}",
                  f"    file: {rel}",
                  f"    summary: {tail[-1] if tail else '(no summary line)'}"]
        for f in (failed + errored):
            lines.append(f"    RED: {f}")
        lines.append("")

    # restore-verification: the tree must be back to the authored bytes
    rc, txt = run(stack)
    tail = [l for l in txt.splitlines() if re.search(r"\d+ (passed|failed)", l)]
    lines += ["### RESTORE CHECK -- the suite must be GREEN again",
              f"    exit {rc}: {tail[-1] if tail else '(no summary)'}", ""]
    lines.append("### VERDICTS")
    for mid, state, failed in verdicts:
        lines.append(f"    {mid:38s} {state:14s} {len(failed)} test(s) RED")
    body = "\n".join(lines)
    io.open(os.path.join(out, "RED_mutations.log"), "w",
            encoding="utf-8").write(body + "\n")
    print(body)
    return 0 if all(v[1] == "RED" for v in verdicts) and rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
