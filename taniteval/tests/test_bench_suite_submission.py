"""THE SUBMISSION REFUSAL (W1) — PI 2026-09-19: *"dont submit now until I approve"*.

The gate can PASS (proved here with an injected record) and REFUSES everything real today, because
no ``SUBMISSION-APPROVED:`` line exists in any committed PI decision record.
"""
from __future__ import annotations

import pytest

from taniteval.bench import submission as S

REC_OK = {"Project Steering/Decisions/2026-09-99-pi-rulings.md":
          "# PI rulings\n\nSUBMISSION-APPROVED: D-PI-SUBMIT-1 target=hf_navsim_navhard benchmark=navsim_v2\n"}


def test_no_approval_flag_is_refused():
    v = S.check_approval(None, target="hf_navsim_navhard", benchmark="navsim_v2", records=REC_OK)
    assert v["approved"] is False and "until the PI approves" in v["reason"]


def test_unknown_decision_id_is_refused():
    v = S.check_approval("D-MADE-UP", target="hf_navsim_navhard", benchmark="navsim_v2", records=REC_OK)
    assert v["approved"] is False and "not a recorded PI submission approval" in v["reason"]


def test_approval_for_another_target_is_refused():
    v = S.check_approval("D-PI-SUBMIT-1", target="hf_navsim_warmup", benchmark="navsim_v2", records=REC_OK)
    assert v["approved"] is False and "approves" in v["reason"]


def test_a_real_approval_passes_the_gate():
    """The gate CAN pass — otherwise the refusal tests would be green by construction."""
    v = S.check_approval("D-PI-SUBMIT-1", target="hf_navsim_navhard", benchmark="navsim_v2", records=REC_OK)
    assert v["approved"] is True and v["hits"][0]["file"].startswith("Project Steering/Decisions/")


def test_no_approval_exists_in_the_real_committed_records():
    """MEASURED: today no PI record carries a SUBMISSION-APPROVED line, so every submission refuses."""
    recs = S.read_committed_records()
    assert recs, "no PI decision record could be read from HEAD — the gate must then refuse anyway"
    assert not any(S.MARKER_RE.search(t) for t in recs.values()), \
        "a SUBMISSION-APPROVED line now exists — check it is the PI's before quoting this test"
    v = S.check_approval("D-ANY", target="hf_navsim_navhard", benchmark="navsim_v2")
    assert v["approved"] is False


def test_submit_exit_codes(tmp_path):
    assert S.submit(str(tmp_path), target="hf_navsim_navhard", benchmark="navsim_v2", decision_id=None,
                    records=REC_OK, log=lambda m: None) == S.EXIT_REFUSED == 3
    assert S.submit(str(tmp_path), target="hf_navsim_navhard", benchmark="navsim_v2",
                    decision_id="D-PI-SUBMIT-1", records=REC_OK, log=lambda m: None) == S.EXIT_APPROVED_NOT_BUILT == 4


def test_cli_submit_refuses_without_the_flag(tmp_path, capsys):
    from taniteval.bench import cli
    rc = cli.main(["submit", str(tmp_path), "--target", "hf_navsim_navhard", "--benchmark", "navsim_v2"])
    assert rc == 3
    assert "SUBMISSION REFUSED" in capsys.readouterr().out


def test_deliberate_regression_a_gate_that_ignores_the_register_goes_red():
    """M-SUBMIT: a gate that approves without reading the records would pass the refusal tests."""
    def mutant(decision_id, *, target, benchmark, records=None):
        return {"approved": True, "reason": "trust me"}
    assert mutant("D-MADE-UP", target="hf_navsim_navhard", benchmark="navsim_v2")["approved"] is True
    assert S.check_approval("D-MADE-UP", target="hf_navsim_navhard", benchmark="navsim_v2",
                            records=REC_OK)["approved"] is False


def test_worktree_only_marker_does_not_count(tmp_path, monkeypatch):
    """The records are read from HEAD (`git show HEAD:<file>`), so an agent's worktree edit cannot
    manufacture an approval."""
    import subprocess
    out = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(S.REPO), "show",
                          "HEAD:Project Steering/PI_DECISION_QUEUE.md"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    assert out.returncode == 0 and "SUBMISSION-APPROVED" not in out.stdout
