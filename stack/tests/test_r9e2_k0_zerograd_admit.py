"""refcv9 E2 K0 -- the refcv9-full family admits core.decoder.r8_rc_gate under --r8-no-rc, so the e2 chain's zero-gradient controls can be written (MM 2026-11-05).

MEASURED 2026-11-05 00:45Z on Thor: ``r9e2_zg_auto.sh`` ran ``zerograd_gate.py`` on K0's explore smoke checkpoint (384e79f) -> 3 untrained parameters
(core.decoder.r8_rc_gate, core.decoder.r8_rc_to_cond.weight / .bias); ``zerograd_allow_from_gate.py --profile refcv9-e2 --check-offenders`` exempted the two
r8_rc_to_cond rows (refcv8's pre-registered dead group) but NOT r8_rc_gate, rc 1 -> no env -> the chain would STOP NO-ZEROGRAD-CONTROLS. refcv9-warm has
admitted that exact parameter since NT A1 (2026-10-26): refc.py applies it only when ``r8['rc'] is not None`` and ``--r8-no-rc`` feeds none.

Pinned on Thor's REAL record (``data/r9e2_K0_zerograd_smoke_384e79f.json``) and the COMMITTED K0 argv, through the tool the chain runs:
 (a) refcv9-e2 (and refcv9-full) now exempt all three offenders: --check-offenders rc 0;
 (b) RED: refcv8's list (the row absent) leaves r8_rc_gate unexplained, rc 1 -- what Thor read;
 (c) the row is refcv9-warm's (same reason text), admitted only while --r8-no-rc is in the argv, and only that one parameter is added.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402

FIXTURE = ROOT / "tests" / "data" / "r9e2_K0_zerograd_smoke_384e79f.json"
K0_ARGV = ROOT / "ops" / "runs.d" / "r9e2_K0.argv.json"
TOOL = ROOT / "scripts" / "zerograd_allow_from_gate.py"
ROW = "core.decoder.r8_rc_gate"


def _check(profile: str, argv: Path):
    r = subprocess.run([sys.executable, str(TOOL), "--profile", profile, "--argv", str(argv), "--emit", "bash", "--check-offenders", str(FIXTURE)],
                       capture_output=True, text=True, encoding="utf-8", timeout=300, cwd=str(ROOT))
    return r.returncode, r.stdout + r.stderr


def test_the_fixture_is_Thor_s_K0_record_with_its_three_offenders():
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    names = sorted(n for o in d["offenders"] for n in o["names"])
    assert names == ["core.decoder.r8_rc_gate", "core.decoder.r8_rc_to_cond.bias", "core.decoder.r8_rc_to_cond.weight"]
    assert d["verdict"] == "FAIL" and d["untrained_numel"]["offending_lo"] == 1921
    assert "/384e79f2b162_K0/" in d["ckpt"]


@pytest.mark.parametrize("profile", ["refcv9-e2", "refcv9-full"])
def test_the_refcv9_full_family_exempts_every_K0_offender(profile):
    rc, out = _check(profile, K0_ARGV)
    assert rc == 0 and "r8_rc_gate" in out, out                               # the emitted ZEROGRAD_ALLOW carries the row


def test_RED_without_the_row_r8_rc_gate_stays_unexplained_as_Thor_read():
    rc, out = _check("refcv8", K0_ARGV)                                       # refcv8's list: r8_rc_to_cond only
    assert rc != 0 and "r8_rc_to_cond" in out and "r8_rc_gate" not in out, out   # Thor's exact read: rc 1, the gate absent from the list


def test_the_row_is_refcv9_warm_s_admitted_only_with_its_flag_and_it_is_the_only_addition():
    warm = {r[0]: r for r in LG.PROFILES["refcv9-warm"]["live_dead_admitted"]}
    for name in ("refcv9-e2", "refcv9-full"):
        rows = LG.PROFILES[name]["live_dead_admitted"]
        assert tuple(rows[:len(LG.PROFILES["refcv8"]["live_dead_admitted"])]) == tuple(LG.PROFILES["refcv8"]["live_dead_admitted"])
        extra = rows[len(LG.PROFILES["refcv8"]["live_dead_admitted"]):]
        assert [r[0] for r in extra] == [ROW] and extra[0] == warm[ROW] and extra[0][1] == "--r8-no-rc", name
    assert '"--r8-no-rc"' in K0_ARGV.read_text(encoding="utf-8")                # K0 carries the flag the row is admitted under
