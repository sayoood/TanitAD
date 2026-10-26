"""``stack/scripts/zerograd_allow_from_gate.py``: the supervisor allow-list derived from the launch gate, 2026-10-26.

Pinned on NT A1's REAL gate record (21 offenders, names only; ``2026-10-26-nt-a1-zerograd/zerograd_gate.json``): the
refcv8-nt profile with that run's two flags exempts exactly 20, and ``core.decoder.r8_rc_gate`` stays an offender until it
is passed as a disclosed ``--extra``. Expected values are LITERALS, never expressions over the code under test.
Mutations: drop ``--r8-no-rc`` from the argv -> the rc projection is no longer exempt (18); a sibling that only shares a
group's prefix is never exempt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import zerograd_allow_from_gate as Z  # noqa: E402

#: NT A1's 21 offenders as the supervisor's gate recorded them (case, numel), names verbatim
OFFENDERS = [
    ("core.decoder.r8_rc_gate", "a_no_state", 1),
    ("core.decoder.r8_rc_to_cond.weight", "a_no_state", 1536),
    ("core.decoder.r8_rc_to_cond.bias", "a_no_state", 384),
    ("_map_hires.refine.inp.0.weight", "b_zero_state", 2048),
    ("_map_hires.refine.inp.1.weight", "b_zero_state", 32),
    ("_map_hires.refine.inp.1.bias", "b_zero_state", 32),
    ("_map_hires.refine.conv1.weight", "b_zero_state", 9216),
    ("_map_hires.refine.norm1.weight", "b_zero_state", 32),
    ("_map_hires.refine.norm1.bias", "b_zero_state", 32),
    ("_map_hires.refine.conv2.weight", "b_zero_state", 9216),
    ("_map_hires.refine.norm2.weight", "b_zero_state", 32),
    ("_map_hires.refine.norm2.bias", "b_zero_state", 32),
    ("_map_hires.refine.cls.weight", "b_zero_state", 256),
    ("_map_hires.refine.cls.bias", "b_zero_state", 8),
    ("_map_hires.near.lift.unobserved", "b_zero_state", 32),
    ("_map_hires.near.lift.proj.weight", "b_zero_state", 65536),
    ("_map_hires.near.lift.proj.bias", "b_zero_state", 32),
    ("_map_hires.near_refine.0.c1.weight", "b_zero_state", 9216),
    ("_map_hires.near_refine.0.n1.weight", "b_zero_state", 32),
    ("_map_hires.near_refine.0.n1.bias", "b_zero_state", 32),
    ("_map_hires.near_refine.0.c2.weight", "b_zero_state", 9216),
]
GATE = {"offenders": [{"names": [n], "case": c, "numel_lo": k} for n, c, k in OFFENDERS]}
ARGV = ["--refcv8", "--r8-no-rc", "--map-min-coverage", "0.0", "--steps", "12478"]


def test_nt_a1_exactly_twenty_exempt_and_the_rc_gate_is_not():
    rows = Z.derive("refcv8-nt", ARGV)
    c = Z.check(rows, GATE)
    assert c["n_offenders"] == 21
    assert c["n_exempted"] == 20
    assert [r["names"] for r in c["not_exempted"]] == [["core.decoder.r8_rc_gate"]]
    assert sum(r["numel"] for r in c["rows"] if r["allowed_by"]) == 106952


def test_the_rc_gate_needs_a_disclosed_extra():
    rows = Z.derive("refcv8-nt", ARGV, extra=[(r"core\.decoder\.r8_rc_gate", "same rc guard")])
    assert Z.check(rows, GATE)["n_exempted"] == 21
    assert [r["source"] for r in rows if r["group"] is None] == ["extra (NOT pre-registered)"]


def test_mutation_without_the_flag_the_rc_projection_is_not_exempt():
    argv = [a for a in ARGV if a != "--r8-no-rc"]
    c = Z.check(Z.derive("refcv8-nt", argv), GATE)
    assert c["n_exempted"] == 18
    assert "core.decoder.r8_rc_to_cond.weight" in [r["names"][0] for r in c["not_exempted"]]


def test_a_prefix_sibling_is_never_exempt():
    rows = Z.derive("refcv8-nt", ARGV)
    g = {"offenders": [{"names": ["core.decoder.r8_rc_to_cond_extra.weight"], "case": "a_no_state", "numel_lo": 1},
                       {"names": ["_map_hires.refine.cls2.weight"], "case": "b_zero_state", "numel_lo": 1}]}
    assert Z.check(rows, g)["n_exempted"] == 0


def test_bash_emit_and_config_json_argv(tmp_path, capsys):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"argv": ARGV}), encoding="utf-8")
    gj = tmp_path / "zg.json"
    gj.write_text(json.dumps(GATE), encoding="utf-8")
    assert Z.main(["--profile", "refcv8-nt", "--argv", str(cfg), "--emit", "bash"]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("ZEROGRAD_ALLOW=(") and r"'core\.decoder\.r8_rc_to_cond(\..+)?'" in line
    assert Z.main(["--profile", "refcv8-nt", "--argv", str(cfg), "--check-offenders", str(gj)]) == 1


def test_unknown_profile_refuses():
    with pytest.raises(SystemExit):
        Z.derive("no-such-profile", ARGV)
