#!/usr/bin/env python3
"""E1 patch 2/2 — the navhard blocker, applied to the C: COPY only (never the D: devkit).

WHAT FAILS (MEASURED 2026-09-20, raw/navhard/diag/idm_assert_diag.json)
----------------------------------------------------------------------
`navsim_idm_agent_manager.py:84`

    intersecting_agents = self.agent_occupancy.intersects(
        agent_path.buffer((agent.width / 2), cap_style=CAP_STYLE.flat))
    assert intersecting_agents.contains(agent_token), "Agent's baseline does not intersect the agent itself"

An IDM agent that has CONSUMED its whole baseline has `get_path_to_go()` of **zero length**
(diagnosed offender: `path_to_go_length_m = 0.0`, path start == agent position). MEASURED with the
pinned shapely 2.0.7: a zero-length LineString buffered with **flat** caps is an **EMPTY** polygon
(round caps would give a disc), and `distance(box, empty) = nan`. So `intersecting_agents` is empty,
the agent's own token can NEVER be in it, and the assert fires — for 166 of 5,462 navhard stage-2
tokens (3.04 %), 0 of 450 stage-1. Each failure makes its row invalid/NaN, and one NaN row then kills
the whole run downstream (`scene_aggregator.py:62` -> caught at `run_pdm_score.py:382` ->
uncaught TypeError in the summary aggregation -> exit 1, NO CSV: 68 minutes of rollout lost).

THE PATCH, and why it is faithful
---------------------------------
Only the ASSERT CONDITION changes. An agent with nothing ahead of it is precisely the case the
devkit's own `else` branch already handles — *"Free road case: no leading vehicle"*, with
`relative_distance = agent.get_progress_to_go()` (0 here, so IDM brings it to a stop). With an empty
intersection set `intersecting_agents.size > 1` is already False, so that branch is ALREADY the one
the code would take; the assert is what prevents reaching it. The patch therefore changes behaviour
in exactly one situation — the one where the unpatched devkit produces no result at all — and is
inert everywhere else (pinned by the inertness control in verify_idm_patch.py: the same tokens score
bit-identically with and without it).

⛔ It is a PATCH ON TOP OF THE PINNED SHA and part of the column: quote it with every navhard number.
"""
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

F = Path("C:/Users/Admin/navsim-crun/devkit/navsim/planning/simulation/observation/navsim_idm/navsim_idm_agent_manager.py")
OLD = ('                assert intersecting_agents.contains(agent_token), '
       '"Agent\'s baseline does not intersect the agent itself"\n')
NEW = ('                # E1 2026-09-20: a fully consumed baseline gives a ZERO-LENGTH path_to_go, whose\n'
       '                # flat-cap buffer is an EMPTY polygon (shapely 2.0.7, MEASURED), so the agent can\n'
       '                # never intersect its own path and this assert fires. Such an agent has nothing\n'
       '                # ahead of it: fall through to the devkit\'s own free-road branch below\n'
       '                # (intersecting_agents.size == 0, so the `> 1` test is already False).\n'
       '                assert agent_path.length == 0.0 or intersecting_agents.contains(agent_token), \\\n'
       '                    "Agent\'s baseline does not intersect the agent itself"\n')


def blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def main() -> int:
    out = Path(sys.argv[1])
    raw = F.read_bytes()
    crlf = b"\r\n" in raw
    txt = raw.decode("utf-8").replace("\r\n", "\n")
    if "E1 2026-09-20" in txt:
        print("already applied")
        return 0
    if txt.count(OLD) != 1:
        print(f"⛔ anchor found {txt.count(OLD)} times — refusing")
        return 2
    new_txt = txt.replace(OLD, NEW)
    new_raw = (new_txt.replace("\n", "\r\n") if crlf else new_txt).encode("utf-8")
    F.write_bytes(new_raw)
    diff = "".join(difflib.unified_diff(txt.splitlines(True), new_txt.splitlines(True),
                                        "a/navsim/planning/simulation/observation/navsim_idm/navsim_idm_agent_manager.py",
                                        "b/navsim/planning/simulation/observation/navsim_idm/navsim_idm_agent_manager.py"))
    out.with_suffix(".diff").write_text(diff, encoding="utf-8")
    rec = {"file": str(F), "blob_before": blob(raw), "blob_after": blob(new_raw), "crlf": crlf,
           "d_devkit_untouched_sha256": subprocess.run(
               ["sha256sum", "D:/Archive/devbox-C/navsim/devkit/navsim/planning/simulation/observation/navsim_idm/navsim_idm_agent_manager.py"],
               capture_output=True, text=True).stdout.split()[0] if sys.platform != "win32" or True else None,
           "mechanism": "zero-length path_to_go -> flat-cap buffer is empty -> own token cannot intersect -> assert",
           "scope": "assert condition only; the free-road branch it falls through to is the devkit's own"}
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
