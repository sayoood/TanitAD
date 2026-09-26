"""The SUBMISSION REFUSAL — nothing leaves this box for an official leaderboard without the PI.

PI 2026-09-19, verbatim: *"dont submit now until I approve"*. BUILD_PLAN.md §1: submission
commands are REFUSED unless ``--pi-approval <decision-id>`` names a RECORDED PI decision.

What counts as a recorded PI decision approving submission (strict, machine-checkable, and
deliberately impossible for an agent to satisfy by editing a file):

1. a line, in a PI decision record that is COMMITTED IN ``HEAD`` (``git show HEAD:<file>`` — a
   worktree edit or a staged file does not count), of the exact form::

       SUBMISSION-APPROVED: <decision-id> target=<target> benchmark=<benchmark>

2. the record files searched are the PI's decision records only:
   ``Project Steering/PI_DECISION_QUEUE.md`` and ``Project Steering/Decisions/*.md``;
3. ``<decision-id>``, ``target`` and ``benchmark`` must ALL equal the request.

MEASURED 2026-09-19: no such line exists anywhere, so today every submission is refused. Even
with an approval the upload itself is NOT BUILT here (exit 4, "approval verified, upload not
implemented") — building it is a separate, PI-visible work package.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .contract import REPO

EXIT_REFUSED = 3
EXIT_APPROVED_NOT_BUILT = 4
PI_RECORD_FILES = ("Project Steering/PI_DECISION_QUEUE.md",)
PI_RECORD_GLOBS = ("Project Steering/Decisions/*.md",)
TARGETS = ("hf_navsim_warmup", "hf_navsim_navhard", "hf_navsim_private_test_hard", "navsim_v1_navtest_eval_server")
MARKER_RE = re.compile(r"^\s*SUBMISSION-APPROVED:\s*(?P<id>\S+)\s+target=(?P<target>\S+)\s+benchmark=(?P<bench>\S+)\s*$",
                       re.MULTILINE)
PI_QUOTE = "PI 2026-09-19: 'dont submit now until I approve'"


def _git(args, repo: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", f"safe.directory={str(repo).replace(chr(92), '/')}", "-C", str(repo)] + args,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)


def read_committed_records(repo: Path = REPO) -> dict:
    """``{relpath: text}`` of every PI decision record AS COMMITTED IN HEAD (never the worktree)."""
    rels = list(PI_RECORD_FILES)
    for g in PI_RECORD_GLOBS:
        r = _git(["ls-tree", "--name-only", "HEAD", str(Path(g).parent).replace("\\", "/") + "/"], repo)
        rels += [ln.strip() for ln in r.stdout.splitlines() if ln.strip().endswith(".md")]
    out = {}
    for rel in rels:
        r = _git(["show", f"HEAD:{rel}"], repo)
        if r.returncode == 0:
            out[rel] = r.stdout
    return out


def check_approval(decision_id: str | None, *, target: str, benchmark: str, records=None) -> dict:
    """``{"approved": bool, "reason": str, ...}``. ``records`` is injectable for tests; production
    reads the committed PI records."""
    if not decision_id:
        return {"approved": False, "reason": (f"no --pi-approval given. Submission is REFUSED until the PI approves "
                                              f"({PI_QUOTE}); none has been recorded.")}
    if target not in TARGETS:
        return {"approved": False, "reason": f"unknown target {target!r}; known {TARGETS}"}
    recs = read_committed_records() if records is None else records
    if not recs:
        return {"approved": False, "reason": "no PI decision record could be read from HEAD — refusing"}
    hits = []
    for rel, text in recs.items():
        for m in MARKER_RE.finditer(text):
            if m.group("id") == decision_id:
                hits.append({"file": rel, "target": m.group("target"), "benchmark": m.group("bench")})
    if not hits:
        return {"approved": False, "n_records_searched": len(recs),
                "reason": (f"decision {decision_id!r} is not a recorded PI submission approval: no line "
                           f"'SUBMISSION-APPROVED: {decision_id} target=… benchmark=…' in any committed PI record "
                           f"({', '.join(sorted(recs))}). {PI_QUOTE}.")}
    ok = [h for h in hits if h["target"] == target and h["benchmark"] == benchmark]
    if not ok:
        return {"approved": False, "hits": hits,
                "reason": f"decision {decision_id!r} approves {hits}, not target={target} benchmark={benchmark}"}
    return {"approved": True, "hits": ok, "reason": f"recorded PI approval {decision_id} in {ok[0]['file']}"}


def submit(run_dir: str, *, target: str, benchmark: str | None, decision_id: str | None, records=None,
           log=print) -> int:
    rd = Path(run_dir)
    if benchmark is None:
        try:
            import json
            benchmark = json.loads((rd / "bench_run.json").read_text(encoding="utf-8"))["benchmark"]
        except Exception:                                              # noqa: BLE001
            benchmark = "UNKNOWN"
    v = check_approval(decision_id, target=target, benchmark=benchmark, records=records)
    if not v["approved"]:
        log(f"⛔ SUBMISSION REFUSED — {v['reason']}")
        return EXIT_REFUSED
    log(f"APPROVAL VERIFIED ({v['reason']}). ⛔ The upload step is NOT BUILT in the suite (W1): "
        f"nothing was sent. Building it is a separate, PI-visible work package.")
    return EXIT_APPROVED_NOT_BUILT
