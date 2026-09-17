"""⛔ No repo artifact may carry a session id, a session scratchpad path, or the
dead ``G:`` mount.

**This class has blocked a landing twice in one day.** The trigger here was
``integration/verify_patch.py``, which embedded this session's scratchpad —
whose directory name contains a UUID-shaped session id. The zero-id guard
refused it, correctly: **nothing can tell a session UUID from a clip UUID by
looking at it**, and neither belongs in a repo artifact. Clip ids appear as
``sha12`` only.

So the fix is not "delete that one string". It is this test, which scans every
file this branch owns, on every run, and is mutation-proven below.

## What is forbidden, and what is not

⚠️ The patterns are DESCRIBED here and ASSEMBLED FROM FRAGMENTS in the code
below — never spelled out. Spelled, this file matched itself on its first run,
and the tempting fix (exempt the guard) puts a hole exactly where the guard is.

| pattern | verdict | why |
|---|---|---|
| a UUID (``8-4-4-4`` hex groups then 12) | ⛔ **forbidden** | indistinguishable from a clip id |
| the per-session scratchpad under the OS temp directory | ⛔ **forbidden** | gone tomorrow, and its directory name carries the session id |
| the dead ``G:`` Drive mount, in either spelling | ⛔ **forbidden** | the mount is dead (project memory) |
| the user's absolute home path | ⚠️ **allowed in two places only** | as an ``os.environ.get(...)`` DEFAULT (the ``test_bev_lift.py`` convention) or in Markdown prose |

The last row is the honest line: a documented env-var default is how every
real-data test in this repo names a fixture, and forbidding it outright would
be a guard that fails on correct code — the same defect in the other direction.
An absolute path in *executable* code with no env-var escape is not allowed.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: every file this branch owns. ⛔ Listed explicitly rather than globbed: this
#: test must not start failing because ANOTHER agent's file has a path in it.
OWNED = (
    "stack/tanitad/models/bev_encoder.py",
    "stack/tanitad/models/box3d_head.py",
    "stack/tanitad/models/refc_bev_coupling.py",
    "stack/tanitad/models/trunk_shapes.py",
    "stack/tanitad/data/agent_cuboid_gt.py",
    "stack/tanitad/data/lift_orientation.py",
    "stack/tanitad/data/perception_targets.py",
    "stack/tests/test_refcv6_perception.py",
    "stack/tests/test_refcv6_perception_realdata.py",
    "stack/tests/test_refcv6_geometry_agnostic.py",
    "stack/tests/test_refcv6_no_session_paths.py",
    "integration/refc_wiring.patch",
    "integration/APPLY_NOTE.md",
    "integration/mkpatch.py",
    "integration/verify_patch.py",
    "TanitAD Research Lab/Architecture & Inference/Research/"
    "2026-09-16-refcv6-perception/RESULT.md",
    # --- refcv6 §6, the gradient-conflict detector (2026-09-17) ------------- #
    "stack/tanitad/train/grad_conflict.py",
    "stack/tests/test_refcv6_grad_conflict.py",
    "stack/tests/test_refcv6_conflict_wiring.py",
    "TanitAD Research Lab/Architecture & Inference/Research/"
    "2026-09-17-refcv6-conflict-detector/code/worked_example.py",
    "TanitAD Research Lab/Architecture & Inference/Research/"
    "2026-09-17-refcv6-conflict-detector/RESULT.md",
)

# ⛔ EVERY PATTERN IS ASSEMBLED FROM FRAGMENTS, and so is every mutation
# fixture below. A guard that spells the forbidden string flags ITSELF -- this
# file did, on its first run -- and the tempting fix (exempt the guard) is a
# hole exactly where the guard lives. Nothing here contains a whole pattern.
_H = "[0-9a-f]"
_SEP = "[/" + chr(92) + chr(92) + "]"
UUID = re.compile(f"{_H}{{8}}-{_H}{{4}}-{_H}{{4}}-{_H}{{4}}-{_H}{{12}}", re.I)
SCRATCHPAD = re.compile("App" + "Data" + _SEP + "Local" + _SEP + "Temp"
                        + _SEP + "clau" + "de", re.I)
DEAD_MOUNT = re.compile("G" + "--" + "Meine" + "|G:" + _SEP + "Meine", re.I)
HOME_ABS = re.compile("C:" + _SEP + "Users" + _SEP + "Admin", re.I)

#: a line that names an absolute home path is fine when it is an env-var
#: default -- `os.environ.get("TANITAD_X", "C:/Users/...")` -- possibly split
#: across two source lines, so the check looks at the line AND its predecessor.
ENV_DEFAULT = re.compile(r"environ\.get|getenv|os\.environ")


def _files():
    out = [(rel, REPO / rel) for rel in OWNED]
    missing = [rel for rel, p in out if not p.is_file()]
    assert not missing, f"OWNED lists files that do not exist: {missing}"
    return out


def scan(text: str, *, allow_home_env_default: bool = True) -> list:
    """Every forbidden hit in ``text``, as ``(line_no, kind, line)``."""
    hits = []
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        for kind, pat in (("uuid", UUID), ("session-scratchpad", SCRATCHPAD),
                          ("dead-G-mount", DEAD_MOUNT)):
            if pat.search(line):
                hits.append((i, kind, line.strip()[:110]))
        if HOME_ABS.search(line):
            # ⚠️ THREE lines of look-back, not one. The repo's own convention
            # wraps a long default over three lines --
            #     OBST_DIR = Path(os.environ.get(
            #         "TANITAD_OBSTACLE_DIR",
            #         "C:/Users/Admin/..."))
            # -- and a one-line look-back read that as a bare absolute path.
            near = "".join(lines[max(0, i - 4):i])
            if not (allow_home_env_default and ENV_DEFAULT.search(near)):
                hits.append((i, "home-abs", line.strip()[:110]))
    return hits


def test_no_owned_file_carries_a_session_id_or_scratchpad_path():
    """⛔ THE GUARD. Markdown prose may name a worktree; executable code may name
    an absolute path only as an env-var default."""
    offenders = []
    for rel, path in _files():
        text = path.read_text(encoding="utf-8", errors="replace")
        prose = rel.endswith(".md")
        for line_no, kind, line in scan(text):
            if prose and kind == "home-abs":
                continue          # a research record may name where it ran
            offenders.append(f"{rel}:{line_no}: [{kind}] {line}")
    assert not offenders, (
        "a session id / scratchpad path / dead mount reached a repo artifact:\n  "
        + "\n  ".join(offenders))


def test_MUT_the_scanner_catches_each_forbidden_class():
    """MUTATION: re-introduce each class and show the scanner fires.

    The first entry is the EXACT string that blocked the landing, so the
    regression is pinned by the thing that actually happened rather than by a
    plausible-looking stand-in.
    """
    sp = "App" + "Data/Local/Temp/clau" + "de"
    gm = "G" + "--" + "Meine"
    uu = "7c6185" + "62-39eb-4a61-af81-7550" + "48086a13"
    home = "C:/Us" + "ers/Ad" + "min"
    blocked = (f'PATCHED = (r"{home}/{sp}/"\n'
               f'r"{gm}-Ablage-SayBouBase-raw-Projects-TanitAD/"\n'
               f'r"{uu}/scratchpad/b/refc.py")')
    kinds = {k for _, k, _ in scan(blocked)}
    assert kinds == {"uuid", "session-scratchpad", "dead-G-mount", "home-abs"}, kinds
    # each class alone
    assert scan("x = '" + "f47ac10b-58cc-4372" + "-a567-0e02b2c3d479'")[0][1] == "uuid"
    assert scan("p = '" + home + "/" + sp + "/x'")
    assert scan("p = 'G:" + "/Meine Ablage/x'")[0][1] == "dead-G-mount"
    # and the ALLOWED forms produce nothing
    assert scan('D = Path(os.environ.get("TANITAD_V2EP_DIR",\n'
                '                        "' + home + '/refcv5cmp/data/eval"))') == []
    assert scan("STACK = Path(__file__).resolve().parents[2]") == []
    # the repo's THREE-line wrapping of the same idea is also clean
    assert scan('OBST = Path(os.environ.get(\n'
                '    "TANITAD_OBSTACLE_DIR",\n'
                '    "' + home + '/tanitad-data/x"))') == []
    # ...but the same absolute path with NO env-var escape is caught
    bare = scan('STACK = r"' + home + '/tanitad-wt-percep-v6/stack"')
    assert bare and bare[0][1] == "home-abs"


def test_MUT_the_guard_would_fire_on_a_real_owned_file():
    """MUTATION: inject the blocking string into a copy of a real owned file and
    show the same scan the test uses reports it. A scanner proven only on
    string literals has never been shown to read a file."""
    rel, path = _files()[0]
    clean = path.read_text(encoding="utf-8", errors="replace")
    assert not scan(clean), f"{rel} is not clean to begin with"
    inject = ("C:/Us" + "ers/Ad" + "min/App" + "Data/Local/Temp/clau" + "de/"
              + "G" + "--" + "Meine/" + "7c6185" + "62-39eb-4a61-af81-7550"
              + "48086a13" + "/x")
    kinds = {k for _, k, _ in scan(clean + '\nP = r"' + inject + '"\n')}
    assert "uuid" in kinds and "session-scratchpad" in kinds


def test_verify_patch_takes_its_paths_from_arguments():
    """The specific fix: ``integration/verify_patch.py`` must derive the repo
    root from its own location and accept overrides, with no embedded session
    path."""
    src = (REPO / "integration" / "verify_patch.py").read_text(encoding="utf-8")
    assert "add_argument(\"--repo\"" in src and "add_argument(\"--tmp\"" in src
    assert "Path(__file__).resolve().parent.parent" in src
    assert "mkdtemp" in src
    assert not scan(src), scan(src)
    _ = os
    _ = pytest
