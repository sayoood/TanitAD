#!/usr/bin/env python
"""THE DELIBERATE-REGRESSION HARNESS for the 2026-09-23 refcv6 fixes.

For each of the five findings it re-introduces the EXACT historical defect in
the SOURCE and records whether the new guard goes RED. A guard that cannot fail
is worse than none.

⛔ THE RULES THIS HARNESS ENFORCES ON ITSELF:

  1. **Anchors are WHOLE LINES matched by EQUALITY**, after stripping the line
     terminator. These files are CRLF and a substring anchor collides across
     indentation levels — MEASURED here: ``if rv6.f8_flat_waypoint_noise:``
     occurs at TWO indentation levels and only equality tells them apart.
  2. **An arm whose anchor matches 0 or >1 lines ABORTS THE RUN AS INVALID.**
     Never skipped, never scored as "not caught".
  3. **The mutation is applied to the SOURCE the guard reads**, never to the
     test — with ONE declared exception, arm ``M7``, whose defect *lived in the
     test module* (a baseline revision pinned to ``HEAD``). It is labelled.
  4. **Every arm carries a BASELINE**: the same test file must be GREEN before
     any mutation. A baseline that is already red makes every arm meaningless.
  5. **Subprocesses are decoded utf-8/replace.** ``text=True`` uses this box's
     cp1252 and returns EMPTY streams on one stray byte, which scores as
     "not caught".
  6. **The verdict is read from pytest's own summary ARTIFACT** (the
     ``FAILED <nodeid>`` lines), never from an exit code through a pipe.
     A run with neither a summary line nor a pass count is INCONCLUSIVE.
  7. **The tree under mutation is a COPY**, never the repo. The copy carries
     only ``*.py`` (``stack/tanitad`` is 4.8 MB of source inside 437 MB of
     ``__pycache__`` and artifacts).
  8. **The restore is VERIFIED by sha256** after every arm, and the run aborts
     if a file did not come back.

Usage:
  python mutation_harness.py --json ../raw/mutation_results.json [--only M1,M5]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

PY = r"C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO = pathlib.Path(r"D:/Projects/TanitAD")
TESTFILE = "tests/test_refcv6_diffusion.py"

# --------------------------------------------------------------------------- #
# THE ARMS. `anchor` is a whole line WITHOUT its terminator; `repl` replaces it.
# `expect_red` is the set of tests that MUST fail once the defect is back.
# --------------------------------------------------------------------------- #
ARMS: list[dict] = [
    dict(
        id="M1-coupling1-sampler-arity",
        finding="1 — coupling (1) built and never called",
        file="tanitad/refs/refc.py",
        anchor="                agent_pos, bev)",
        repl="                agent_pos)",
        expect_red=["test_coupling1_REACHES_the_sampler_and_the_classifier",
                    "test_coupling1_fires_on_the_CLASSIFIER_ONLY_pass",
                    "test_coupling1_is_GATED_not_DEAD"],
        note="THE defect: `_sample` got EIGHT positionals and `bev` is its "
             "NINTH parameter, so coupling (1) was None on the whole "
             "diffusion path."),
    dict(
        id="M2-coupling1-classifier-arity",
        finding="1 — the DEFAULT classifier pass omitted it too",
        file="tanitad/refs/refc.py",
        anchor="                                         agent_pos, bev)   # classifier",
        repl="                                         agent_pos)   # classifier",
        expect_red=["test_coupling1_fires_on_the_CLASSIFIER_ONLY_pass",
                    "test_coupling1_REACHES_the_sampler_and_the_classifier"],
        note="the second of the three breaks; independent of M1."),
    dict(
        id="M10-break-B-no-caller-supplies-the-map",
        finding="1 — Break B: `RefCModel.forward` declared `bev` and NO caller "
                "in `stack/` ever supplied it",
        file="tanitad/refs/refc.py",
        anchor="                if bev is None:",
        repl="                if False:",
        expect_red=["test_BREAK_B_the_perception_branch_FEEDS_the_coupling"],
        note="with the feed removed the built coupling gets no map and the "
             "model's own guard refuses — which is the CORRECT failure, and "
             "it is still a RED test."),
    dict(
        id="M11-built-coupling-with-no-map-allowed",
        finding="1 — the self-enforcing guard: BUILT + no map == the measured "
                "0-fires state, and must not be runnable",
        file="tanitad/refs/refc.py",
        anchor='        if bev is None and int(getattr(self.decoder, "bev_coupling_params",',
        repl="        if False and int(getattr(self.decoder, \"bev_coupling_params\",",
        expect_red=["test_a_BUILT_coupling_with_NO_map_is_REFUSED_before_the_compute"],
        note="a coupling that is built, counted and stamped while never "
             "called is the state that would publish 'the BEV coupling does "
             "not help'."),
    dict(
        id="M3-blind-rank",
        finding="2 — the ranked score is blind to the emitted trajectory",
        file="tanitad/refs/refc.py",
        anchor="        base = refined if (sel.refined or self.rv6.f5_emitting_conf) else conf",
        repl="        base = refined if sel.refined else conf",
        expect_red=["test_F5_makes_the_ranked_score_SEE_the_emitted_trajectory"],
        note="drops the F5 term, so even an F5 arm ranks with the classifier "
             "surface. The DEFAULT-is-blind test must STAY GREEN — it pins the "
             "documented behaviour, not the bug."),
    dict(
        id="M4-telemetry-lies",
        finding="2 — `sampler_ranks_the_fan` contradicted the line it describes",
        file="tanitad/refs/refc.py",
        anchor="                                              or rv6.f5_emitting_conf)}",
        repl="                                              or False)}",
        expect_red=["test_sampler_ranks_the_fan_TELEMETRY_no_longer_lies_on_an_F5_arm"],
        note="`bool(self.sel.refined or False)` is exactly the historical "
             "`bool(self.sel.refined)`."),
    dict(
        id="M5-v0-guard-reads-the-flag",
        finding="3 — the v0 refusal tested the DECLARATION, not the tensor",
        file="tanitad/refs/refc.py",
        anchor="                if float(self.anchor_controls.abs().sum()) == 0.0:",
        repl="                if False:",
        expect_red=["test_v0_refusal_READS_THE_TENSOR_not_the_flag"],
        note="⭐ MEASURED, and it is DEFENCE IN DEPTH rather than redundancy: "
             "with ONLY the decoder refusal disarmed, "
             "`test_F9_reaches_the_tensor_THROUGH_THE_FORWARD` STAYS GREEN, "
             "because `assert_f9_vocabulary`'s own tensor check still fires "
             "on an F9 build. My first `expect_red` listed it and this "
             "harness scored the arm NOT CAUGHT — the harness caught MY "
             "error, which is what rule 2 is for. Both guards must go to "
             "restore the full defect: arm M5b."),
    dict(
        id="M5b-both-v0-guards-disarmed",
        finding="3 — the FULL historical state: neither guard reads a tensor",
        file="tanitad/refs/refc.py",
        anchor="                if float(self.anchor_controls.abs().sum()) == 0.0:",
        repl="                if False:",
        also=[dict(file="tanitad/models/refcv6_diffusion.py",
                   anchor="    if anchor_controls is not None and bool(v0_conditioned):",
                   repl="    if False:")],
        expect_red=["test_v0_refusal_READS_THE_TENSOR_not_the_flag",
                    "test_F9_reaches_the_tensor_THROUGH_THE_FORWARD",
                    "test_assert_f9_vocabulary_READS_the_controls_tensor"],
        note="the exactly-degenerate bank (spread 0.000000000 m) then runs "
             "silently, which is what was MEASURED on 2026-09-22."),
    dict(
        id="M6-f9-assert-has-no-tensor",
        finding="3 — `assert_f9_vocabulary(n, v0, expect_n)` had no tensor",
        file="tanitad/models/refcv6_diffusion.py",
        anchor="    if anchor_controls is not None and bool(v0_conditioned):",
        repl="    if False:",
        expect_red=["test_assert_f9_vocabulary_READS_the_controls_tensor"],
        note="the declaration-only signature, restored."),
    dict(
        id="M7-bitidentity-baseline-is-HEAD",
        finding="4 — the bit-identity baseline had rotted to a tautology",
        file=TESTFILE,
        anchor='_BASELINE_REV = "cbadba5844a2db70a968172ab0b0bbe3a0140af0"   # == 8c7d215^',
        repl='_BASELINE_REV = "HEAD"',
        expect_red=["test_bitidentity_baseline_is_PINNED_and_really_pre_refcv6"],
        in_test_by_design=True,
        note="⚠️ THE ONE ARM THAT MUTATES THE TEST MODULE, and it is legitimate "
             "because the defect LIVED there: the guard materialised its "
             "'pre-refcv6' baseline from `HEAD`, whose blob IS the worktree "
             "file. The 64-window comparison then passes trivially, which is "
             "the point — only the PINNED guard can see it."),
    dict(
        id="M8-f8-clamp-outside-the-ladder",
        finding="5 — F8 clamped once, outside the ladder",
        file="tanitad/refs/refc.py",
        anchor="            if rv6.f8_flat_waypoint_noise:",
        repl="            if False:",
        expect_red=["test_F8_clamps_INSIDE_the_ladder_like_DiffusionDrive"],
        note="⚠️ THE ANCHOR IS WHY RULE 1 EXISTS: the identical text occurs at "
             "8 spaces (the pre-loop / DD-training clamp, which STAYS) and at "
             "12 spaces (the in-ladder / DD-test clamp, which is the fix). A "
             "substring anchor would have hit the wrong one."),
    dict(
        id="M9-blind-rank-guard-disarmed",
        finding="2 — the opt-in refusal itself",
        file="tanitad/refs/refc.py",
        anchor="            if (self.rv6.f5_refuse_blind_rank",
        repl="            if (False",
        expect_red=["test_f5_refuse_blind_rank_REFUSES_and_is_OPT_IN"],
        note="a guard that cannot refuse is a stamp."),
]


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_tree(dest: pathlib.Path) -> dict:
    """Copy the *.py surface of `stack/` into `dest/stack`. -> a census."""
    if dest.exists():
        shutil.rmtree(dest)
    src = REPO / "stack"
    out = dest / "stack"
    n = 0
    for p in src.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(src)
        q = out / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
        n += 1
    for extra in ("pyproject.toml", "setup.cfg", "pytest.ini"):
        f = src / extra
        if f.exists():
            shutil.copyfile(f, out / extra)
    census = {"files_copied": n,
              "refc_sha256": sha256(out / "tanitad/refs/refc.py"),
              "repo_refc_sha256": sha256(src / "tanitad/refs/refc.py")}
    if census["refc_sha256"] != census["repo_refc_sha256"]:
        raise SystemExit("INVALID: the copied tree does not match the repo")
    return census


_SUMMARY = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.M)
_COUNTS = re.compile(r"(\d+) (passed|failed|error|errors)")


def run_pytest(tree: pathlib.Path, testfile: str, timeout=1800) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree / "stack")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # ⭐ the bit-identity guard reads blobs from the REAL object store; the copy
    # is not a git repo. `git show <rev>:<path>` needs only GIT_DIR.
    env["GIT_DIR"] = str(REPO / ".git")
    t0 = time.time()
    p = subprocess.run([PY, "-m", "pytest", testfile, "-q", "--tb=no", "-rf",
                        "-p", "no:cacheprovider"],
                       cwd=str(tree / "stack"), capture_output=True,
                       timeout=timeout, env=env)
    out = (p.stdout or b"").decode("utf-8", errors="replace")
    err = (p.stderr or b"").decode("utf-8", errors="replace")
    failed = sorted({m.split("::")[-1] for m in _SUMMARY.findall(out)})
    counts = {k: int(v) for v, k in _COUNTS.findall(out)}
    # ⛔ THE ARTIFACT, not the status code: a run with no counts at all told us
    # nothing, whatever it exited with.
    if not counts:
        return dict(verdict="INCONCLUSIVE", failed=failed, counts=counts,
                    rc=p.returncode, secs=round(time.time() - t0, 1),
                    tail=(out + err)[-2000:])
    return dict(verdict="RAN", failed=failed, counts=counts, rc=p.returncode,
                secs=round(time.time() - t0, 1), tail=out[-1200:])


def apply_arm(tree: pathlib.Path, arm: dict) -> list[tuple[pathlib.Path, bytes]]:
    """Apply every edit of `arm`. -> [(path, original_bytes)] for the restore.

    ⛔ Whole-line EQUALITY, and an anchor that does not match EXACTLY ONE line
    aborts the whole run.
    """
    edits = [dict(file=arm["file"], anchor=arm["anchor"], repl=arm["repl"])]
    edits += list(arm.get("also", []))
    saved = []
    for e in edits:
        path = tree / "stack" / e["file"]
        raw = path.read_bytes()
        saved.append((path, raw))
        text = raw.decode("utf-8")
        nl = "\r\n" if "\r\n" in text else "\n"
        lines = text.split(nl)
        hits = [i for i, ln in enumerate(lines) if ln == e["anchor"]]
        if len(hits) != 1:
            for p, b in saved:
                p.write_bytes(b)
            raise SystemExit(
                f"INVALID ARM {arm['id']}: the anchor matched {len(hits)} "
                f"lines in {e['file']} (must be exactly 1). Anchor: "
                f"{e['anchor']!r}")
        lines[hits[0]] = e["repl"]
        path.write_bytes(nl.join(lines).encode("utf-8"))
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=None)
    ap.add_argument("--json", required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    tree = pathlib.Path(a.tree) if a.tree else pathlib.Path(
        os.environ.get("TEMP", "/tmp")) / "refcv6_mut_tree"
    res: dict = {"repo": str(REPO), "tree": str(tree),
                 "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "testfile": TESTFILE}
    res["census"] = build_tree(tree)

    base = run_pytest(tree, TESTFILE)
    res["baseline"] = base
    if base["verdict"] != "RAN" or base["failed"]:
        res["VERDICT"] = ("INVALID: the baseline is not green — every arm "
                          "below would be meaningless")
        pathlib.Path(a.json).write_text(json.dumps(res, indent=2),
                                        encoding="utf-8")
        print(json.dumps(res, indent=2))
        return 2

    only = set(a.only.split(",")) if a.only else None
    arms = []
    for arm in ARMS:
        if only and not any(arm["id"].startswith(o) for o in only):
            continue
        saved = apply_arm(tree, arm)
        try:
            r = run_pytest(tree, TESTFILE)
        finally:
            for p, b in saved:
                p.write_bytes(b)
            for p, b in saved:                      # restore VERIFIED
                if hashlib.sha256(p.read_bytes()).hexdigest() != \
                        hashlib.sha256(b).hexdigest():
                    raise SystemExit(f"INVALID: {p} did not restore")
        got = set(r["failed"])
        want = set(arm["expect_red"])
        missed = sorted(want - got)
        extra = sorted(got - want)
        caught = r["verdict"] == "RAN" and not missed
        arms.append(dict(id=arm["id"], finding=arm["finding"],
                         file=arm["file"], anchor=arm["anchor"],
                         repl=arm["repl"],
                         also=[x["file"] for x in arm.get("also", [])],
                         in_test_by_design=arm.get("in_test_by_design", False),
                         note=arm["note"], expect_red=sorted(want),
                         went_red=sorted(got), missed=missed,
                         unexpected_also_red=extra, counts=r["counts"],
                         secs=r["secs"],
                         VERDICT=("CAUGHT" if caught else
                                  ("INCONCLUSIVE" if r["verdict"] != "RAN"
                                   else "NOT CAUGHT"))))
        print(f"{arm['id']:38s} {arms[-1]['VERDICT']:14s} "
              f"red={len(got):2d} missed={missed}")
    res["arms"] = arms
    n_caught = sum(x["VERDICT"] == "CAUGHT" for x in arms)
    res["VERDICT"] = (f"{n_caught}/{len(arms)} arms CAUGHT"
                      + ("" if n_caught == len(arms)
                         else " — read the NOT CAUGHT rows"))
    pathlib.Path(a.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(res["VERDICT"])
    return 0 if n_caught == len(arms) else 1


if __name__ == "__main__":
    sys.exit(main())
