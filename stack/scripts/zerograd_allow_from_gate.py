"""Derive the supervisor's ``ZEROGRAD_ALLOW`` from the launch gate's pre-registered dead groups, 2026-10-26.

Why this exists (MEASURED, ``FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-26-nt-a1-zerograd/``): NT A1 trained to
completion and FAILED the supervisor's post-run zero-gradient gate (``stack/ops/sup_refcv7.sh`` G5) on 21 parameters, 20 of
which (106,952 / 106,953 values) sit in groups its OWN launch gate had admitted dead BEFORE launch
(``launch_gate.py`` profile ``refcv8-nt``, ``live_dead_admitted``, each admitted only while its flag is in the argv). The two
instruments carry two allow-lists and nobody copied one into the other, so ``ZEROGRAD_ALLOW`` was empty and the FAIL was
final.

This tool is the copy, done mechanically: for a profile and a launch argv it emits one ``re.fullmatch`` pattern per admitted
group whose flag is in the argv (the group itself or any parameter below it, never a sibling that merely shares the prefix),
in the ``ZEROGRAD_ALLOW=(...)`` form the supervisor sources. Anything NOT pre-registered must be passed as ``--extra
REGEX::REASON`` and is recorded as such, so a post-hoc exemption is never mixed into the pre-registered ones silently.

``--check-offenders <zerograd_gate.json>`` replays ``zerograd_gate._allowed_by`` (fail closed: every candidate name of a row
must match) over a gate record's offenders and reports which the derived list exempts. It changes no gate and no run.

    python stack/scripts/zerograd_allow_from_gate.py --profile refcv8-nt --argv run/config.json --emit bash
    python stack/scripts/zerograd_allow_from_gate.py --profile refcv8-nt --argv run/config.json \\
        --extra 'core\\.decoder\\.r8_rc_gate::same rc-is-None guard (refc.py r8_rc_gate)' --check-offenders run/zerograd_gate.json
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_gate  # noqa: E402
from zerograd_gate import _allowed_by  # noqa: E402


def read_argv(path: str) -> List[str]:
    """A launch argv: a JSON list, or a dict carrying ``argv`` (a trainer ``config.json`` / an argv record)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    argv = d.get("argv") if isinstance(d, dict) else d
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        raise SystemExit(f"REFUSED: {path} holds no argv list")
    return argv


def group_pattern(group: str) -> str:
    return re.escape(group) + r"(\..+)?"


def derive(profile: str, argv: Sequence[str],
           extra: Sequence[Tuple[str, str]] = ()) -> List[Dict[str, Any]]:
    if profile not in launch_gate.PROFILES:
        raise SystemExit(f"REFUSED: unknown launch-gate profile {profile!r}")
    rows: List[Dict[str, Any]] = []
    for group, flag, why in launch_gate.PROFILES[profile].get("live_dead_admitted", ()):
        if launch_gate.has_flag(list(argv), flag):
            rows.append({"pattern": group_pattern(group), "group": group, "flag": flag, "why": why,
                         "source": "pre-registered"})
    for pat, why in extra:
        re.compile(pat)
        rows.append({"pattern": pat, "group": None, "flag": None, "why": why, "source": "extra (NOT pre-registered)"})
    return rows


def check(rows: Sequence[Dict[str, Any]], gate: Dict[str, Any]) -> Dict[str, Any]:
    pats = [r["pattern"] for r in rows]
    out = []
    for o in gate.get("offenders", []):
        hit = _allowed_by(pats, o.get("names") or [])
        out.append({"names": o.get("names"), "case": o.get("case"), "numel": o.get("numel_lo"), "allowed_by": hit})
    left = [r for r in out if r["allowed_by"] is None]
    return {"n_offenders": len(out), "n_exempted": len(out) - len(left), "not_exempted": left, "rows": out}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--argv", required=True, help="the launch argv (JSON list, or a JSON dict with 'argv')")
    ap.add_argument("--extra", action="append", default=[], metavar="REGEX::REASON",
                    help="a pattern that is NOT pre-registered; recorded as such")
    ap.add_argument("--emit", choices=("bash", "json"), default="json")
    ap.add_argument("--check-offenders", metavar="ZEROGRAD_GATE_JSON")
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)
    extra = []
    for e in a.extra:
        if "::" not in e:
            raise SystemExit(f"REFUSED: --extra needs REGEX::REASON, got {e!r}")
        p, why = e.split("::", 1)
        extra.append((p, why))
    rows = derive(a.profile, read_argv(a.argv), extra)
    rec: Dict[str, Any] = {"tool": "zerograd_allow_from_gate", "profile": a.profile, "argv_file": a.argv, "allow": rows}
    rc = 0
    if a.check_offenders:
        rec["check"] = check(rows, json.loads(Path(a.check_offenders).read_text(encoding="utf-8")))
        rc = 0 if not rec["check"]["not_exempted"] else 1
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if a.emit == "bash":
        # LF only: the line is sourced by bash, and a CRLF from a Windows console makes it ONE garbled element
        # (MEASURED 2026-10-26 on Thor: n=1 instead of 23)
        line = "ZEROGRAD_ALLOW=(" + " ".join(shlex.quote(r["pattern"]) for r in rows) + ")\n"
        sys.stdout.flush()
        if hasattr(sys.stdout, "buffer"):
            sys.stdout.buffer.write(line.encode("utf-8"))
        else:
            sys.stdout.write(line)
        sys.stdout.flush()
    else:
        print(json.dumps(rec, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
