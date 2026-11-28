"""refcv9 E2 -- the gate3 kit hands an arm's COMMITTED PI cost approval to the gate (SPEC 6.2 item 5), and K0's approval is D1 as costed (MM 2026-11-05).

MEASURED 2026-11-05 00:44Z: K0's explore G-LIVE smoke read 9.4036 s/step = 1.47x refcv6's 6.4 (> 1.25x) -> PI-DECISION, and the e2 chain STOPPED
EXPLORE-NOT-PASS. The gate has had ``--pi-cost-approval`` since refcv8; ``r9gate3_thor_gate.sh`` had no way to pass one. The PI's D1 ("~300 Thor-h core,
<=337 with conditionals", REFCV9_DESIGN 4.5 approved AS COSTED) prices E2 K0 at 8.94 and the refcv9 full config at 9.6-10.5 s/step.

Pinned here, through the REAL script in test_r9gate3_study's Thor sandbox:
 (a) with ``runs.d/<prefix>_<ARM>.pi_cost_approval.json`` committed, BOTH modes hand it to the gate; without one, neither does (the rule applies unchanged);
 (b) the committed approvals are well formed; K0's carries the PI's answer ("Train K0", 2026-11-05) to a question naming 10.5 s/step, which covers 9.4036;
 (c) K0's reading trips the profile's own cost constants (6.4 x 1.25) and sits under the approval; the lift is proved by K0's Thor explore re-run.
 RED: the hand-over line deleted, the K0 explore command line carries no approval.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
import test_r9gate3_study as S  # noqa: E402

RUNS = ROOT / "ops" / "runs.d"
K0_APPROVAL = RUNS / "r9e2_K0.pi_cost_approval.json"
LINE = '[ -f "$PICOST" ] && A+=(--pi-cost-approval "$PICOST")'
D1_COSTED_MAX = 10.5          # REFCV9_DESIGN 4.5: the refcv9 full config 9.6-10.5 s/step, central 10.1 (D1: "approved as costed")
K0_MEASURED = 9.4036          # /home/nvidia/r8ns/r9e2/gate3_explore/384e79f2b162_K0/evidence/G-LIVE.json (2026-11-05 00:44Z)


def _gate(tmp, mode, *, approval: bool, script_text: str | None = None):
    sb = S._sandbox(tmp)
    r = S._arm_files(sb, "r9e2", "r9e2", "K0", "23fbf8f7e2b1")       # e2 has no default record commit: the restart stamps REC_C12
    if script_text is not None:
        (sb / "kit" / "r9gate3_thor_gate.sh").write_text(script_text.replace("/home/nvidia", (sb / "home/nvidia").as_posix()), encoding="utf-8", newline="\n")
    ap = r / "launch_tree/stack/ops/runs.d/r9e2_K0.pi_cost_approval.json"
    if approval:
        ap.write_text(K0_APPROVAL.read_text(encoding="utf-8"), encoding="utf-8")
    rec = sb / "recorded_args.txt"
    e = dict(R9_STUDY="e2", PATH=(sb / "bin").as_posix() + os.pathsep + os.environ.get("PATH", ""), REC_OUT=rec.as_posix(),
             REAL_PY=Path(sys.executable).as_posix())
    args = ["explore", "K0"] if mode == "explore" else ["final", "K0", (r / "suite_bound.json").as_posix(), (r / "ev").as_posix()]
    rc, out, err = S._run([(sb / "kit" / "r9gate3_thor_gate.sh").as_posix(), *args], env=e, cwd=sb.as_posix(), timeout=120)
    got = rec.read_text(encoding="utf-8").replace("\r", "").splitlines() if rec.exists() else None
    return rc, out + err, got, ap.as_posix()


@S.needs_bash
@pytest.mark.parametrize("mode", ["explore", "final"])
def test_a_committed_approval_reaches_the_gate_in_both_modes_and_its_absence_changes_nothing(tmp_path, mode):
    rc, out, got, ap = _gate(tmp_path / "with", mode, approval=True)
    assert got is not None and "REFUSED" not in out, out
    i = got.index("--pi-cost-approval")
    assert got[i + 1] == ap and got.count("--pi-cost-approval") == 1, got
    rc, out, got, _ = _gate(tmp_path / "without", mode, approval=False)
    assert got is not None and "--pi-cost-approval" not in got, (got, out)


@S.needs_bash
def test_RED_the_hand_over_line_deleted_leaves_K0_without_its_approval(tmp_path):
    src = (S.GATE3 / "r9gate3_thor_gate.sh").read_text(encoding="utf-8")
    assert src.count(LINE) == 1
    rc, out, got, _ = _gate(tmp_path, "explore", approval=True, script_text=src.replace(LINE, ":"))
    assert got is not None and "--pi-cost-approval" not in got, (got, out)


def test_every_committed_approval_is_well_formed_and_K0_s_carries_the_PI_s_words():
    files = sorted(RUNS.glob("*.pi_cost_approval.json"))
    assert K0_APPROVAL in files
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        assert isinstance(d.get("max_s_per_step"), (int, float)) and d["max_s_per_step"] > 0, f.name
        assert str(d.get("pi_verbatim", "")).strip() and str(d.get("date", "")).strip() and d.get("recorded_by"), f.name
        assert (RUNS / f.name.replace(".pi_cost_approval.json", ".argv.json")).is_file(), f.name     # an approval belongs to a committed arm
    k0 = json.loads(K0_APPROVAL.read_text(encoding="utf-8"))
    assert k0["max_s_per_step"] == D1_COSTED_MAX and K0_MEASURED <= k0["max_s_per_step"]
    assert k0["pi_verbatim"].startswith("Train K0  [PI, chat, 2026-11-05") and k0["date"] == "2026-11-05"
    assert k0["status"].startswith("PI-APPROVED.") and "10.5" in k0["pi_verbatim"]                 # the ceiling the PI was asked about


def test_K0_s_reading_trips_the_profile_s_own_cost_rule_and_sits_under_the_approval():
    """the constants judge_new2_live reads (profile refcv9-e2, map_hires): without an approval K0's 9.4036 is PI-DECISION; the approval covers it.
    The lift itself is proved on Thor by K0's explore re-run (G-LIVE "within the PI's recorded approval"), never re-implemented here."""
    h = LG.PROFILES["refcv9-e2"]["map_hires"]
    ref, mx = float(h["cost_ref_s_per_step"]), float(h["cost_max_ratio"])
    assert (ref, mx) == (6.4, 1.25) and K0_MEASURED / ref > mx                        # SPEC 6.2 item 5: the rule the K0 reading tripped
    assert K0_MEASURED <= json.loads(K0_APPROVAL.read_text(encoding="utf-8"))["max_s_per_step"] == D1_COSTED_MAX
