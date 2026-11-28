#!/usr/bin/env python3
"""RED -> GREEN for the stale-`tanitad` shadowing, on a DELIBERATELY STALE TREE.

⛔ A guard that cannot fail is worse than none (class C13). This script builds a
miniature pre-v5 `stack` — no `tanitad/train/heldout_gate.py`, no
`heldout_goal`, no `register_v2_geometry_sibling`, no `resolve_v2_frames`,
exactly the shape MEASURED at pod2:/root/TanitAD/stack — and runs six scenarios
in six fresh interpreters.

⭐ The one that matters is S1: with the guard OFF the stale tree answers
**HFOV 120.0** where the pinned tree says **117.0**, with no exception and no
warning. That is the plausible-wrong-number failure, reproduced, and every GREEN
below is only meaningful because S1 succeeds.

    python3 stale_import_demo.py --repo <repo> --out raw/stale_import_demo.json

⛔ Touches no pod, no GPU, no corpus. Pure interpreter plumbing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

GOOD_HFOV, STALE_HFOV = 117.0, 120.0     # v5's real frame vs the parent cache


def _w(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(s), encoding="utf-8")


def make_tree(root: Path, *, tag: str, hfov: float, post_v5: bool) -> Path:
    _w(root / "tanitad" / "__init__.py", f'STACK_TAG = "{tag}"\n')
    _w(root / "tanitad" / "geometry.py",
       f'STACK_TAG = "{tag}"\nHFOV_DEG = {hfov}\n'
       f'def frame_from_args(*a, **k):\n    return HFOV_DEG\n')
    _w(root / "tanitad" / "train" / "__init__.py", "")
    _w(root / "tanitad" / "data" / "__init__.py", "")
    _w(root / "scripts" / "train_flagship_v4.py", f'STACK_TAG = "{tag}"\n')
    if post_v5:
        _w(root / "tanitad" / "train" / "heldout_gate.py",
           'PRIMARY_NAME = "pseudosim_composite_PSS_recovery_progress@twosided_v2"\n')
        _w(root / "tanitad" / "train" / "heldout_goal.py",
           "def make_goal_kwargs_fn(*a, **k):\n    return None\n")
        _w(root / "tanitad" / "data" / "parity.py",
           "def register_v2_geometry_sibling(*a, **k):\n    return None\n")
        with open(root / "scripts" / "train_flagship_v4.py", "a",
                  encoding="utf-8") as fh:
            fh.write("def resolve_v2_frames(*a, **k):\n    return None\n")
    else:
        _w(root / "tanitad" / "data" / "parity.py", "PRE_V5 = True\n")
    return root


def run(repo: Path, body: str, *, env_extra=None, path_extra=()) -> dict:
    lines = ["import sys, json",
             f"sys.path.insert(0, {str(repo / 'taniteval')!r})"]
    lines += [f"sys.path.insert(0, {str(p)!r})" for p in path_extra]
    env = dict(os.environ)
    for k in ("TANITEVAL_STACK_OVERRIDE", "TANITEVAL_STACK_GUARD", "PYTHONPATH"):
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(env_extra or {})
    r = subprocess.run([sys.executable, "-c",
                        "\n".join(lines) + "\n" + textwrap.dedent(body)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=300)
    payload = None
    for ln in reversed(r.stdout.strip().splitlines()):
        try:
            payload = json.loads(ln)
            break
        except Exception:                                     # noqa: BLE001,S112
            continue
    err = [x for x in r.stderr.splitlines()
           if "STACK SHADOWING" in x or "STACK GUARD REFUSED" in x
           or x.startswith(("StackShadowError", "taniteval.stack_guard"))]
    return {"exit": r.returncode, "json": payload,
            "guard_lines": err[:2],
            "raised_StackShadowError": "StackShadowError" in r.stderr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="stale_stack_"))
    try:
        good = make_tree(tmp / "workspace_stack", tag="GOOD", hfov=GOOD_HFOV,
                         post_v5=True)
        stale = make_tree(tmp / "root_stack", tag="STALE", hfov=STALE_HFOV,
                          post_v5=False)
        R = {}

        # S1 ⭐ RED — the wrong number, silently.
        R["S1_RED_wrong_number_guard_off"] = run(repo, f"""
            import taniteval
            sys.path.insert(0, {str(stale)!r})    # the old hardcoded line
            import tanitad.geometry as g
            print(json.dumps({{"tag": g.STACK_TAG, "hfov": g.HFOV_DEG}}))
            """, env_extra={"TANITEVAL_STACK_GUARD": "off"}, path_extra=[good])

        # S2 ⚠️ RED — the sync proof that says everything is fine.
        R["S2_RED_bare_import_tanitad_says_GOOD"] = run(repo, """
            import tanitad
            print(json.dumps({"tag": tanitad.STACK_TAG}))
            """, path_extra=[good])
        R["S2b_RED_same_import_after_the_stale_insert"] = run(repo, f"""
            sys.path.insert(0, {str(stale)!r})
            import tanitad
            print(json.dumps({{"tag": tanitad.STACK_TAG}}))
            """, env_extra={"TANITEVAL_STACK_GUARD": "off"}, path_extra=[good])

        # S3 ⭐ GREEN — no env var at all; PYTHONPATH is the intent.
        R["S3_GREEN_late_stale_insert_refused"] = run(repo, f"""
            import taniteval
            sys.path.insert(0, {str(stale)!r})
            import tanitad.geometry as g
            print(json.dumps({{"hfov": g.HFOV_DEG}}))
            """, path_extra=[good])

        # S4 GREEN — an override that is SET BUT INEFFECTIVE.
        empty = tmp / "not-a-stack"
        empty.mkdir()
        R["S4_GREEN_override_set_but_ineffective_refused"] = run(repo, f"""
            sys.path.insert(0, {str(stale)!r})
            import taniteval
            import tanitad.geometry as g
            print(json.dumps({{"hfov": g.HFOV_DEG}}))
            """, env_extra={"TANITEVAL_STACK_OVERRIDE": str(empty)})

        # S5 GREEN — capability probe: right path, wrong vintage.
        R["S5_GREEN_capability_probe_refuses_pre_v5"] = run(repo, """
            import taniteval
            from taniteval import stack_guard as sg
            sys.exit(sg.main(["--require", "v5"]))
            """, env_extra={"TANITEVAL_STACK_OVERRIDE": str(stale)})
        R["S6_GREEN_capability_probe_accepts_post_v5"] = run(repo, """
            import taniteval
            from taniteval import stack_guard as sg
            sys.exit(sg.main(["--require", "v5"]))
            """, env_extra={"TANITEVAL_STACK_OVERRIDE": str(good)})

        R["_summary"] = {
            "RED S1 produced the wrong number with no error":
                R["S1_RED_wrong_number_guard_off"]["exit"] == 0
                and (R["S1_RED_wrong_number_guard_off"]["json"] or {}).get("hfov")
                == STALE_HFOV,
            "RED S2 the bare `import tanitad` sync proof PASSES":
                (R["S2_RED_bare_import_tanitad_says_GOOD"]["json"] or {}).get("tag")
                == "GOOD",
            "RED S2b the SAME import after the stale insert is STALE":
                (R["S2b_RED_same_import_after_the_stale_insert"]["json"] or {})
                .get("tag") == "STALE",
            "GREEN S3 refuses with NO env var":
                R["S3_GREEN_late_stale_insert_refused"]["exit"] != 0,
            "GREEN S4 refuses a set-but-ineffective override":
                R["S4_GREEN_override_set_but_ineffective_refused"]["exit"] != 0,
            "GREEN S5 capability probe exits 2 on a pre-v5 tree":
                R["S5_GREEN_capability_probe_refuses_pre_v5"]["exit"] == 2,
            "GREEN S6 capability probe exits 0 on a post-v5 tree":
                R["S6_GREEN_capability_probe_accepts_post_v5"]["exit"] == 0,
        }
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(R, indent=2), encoding="utf-8")
        print(json.dumps(R["_summary"], indent=2))
        return 0 if all(R["_summary"].values()) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
