"""PREREG_V8F_ROLLHIST section 4.1: the chain checker's G-NAME / G-PAIR (stack/scripts/rollhist_pair_check.py). Each rule has a deliberate-regression arm that must read RED; the control reads PASS."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "stack" / "scripts"))
import rollhist_pair_check as P  # noqa: E402

SHA = "ab" * 32
COMMON = ["--arm", "M2b", "--seed", "0", "--grid", "pool4x10", "--kin-mode", "yaw", "--batch", "64", "--init-ckpt", "/x/init.pt", "--steps", "3000", "--lr", "1e-4", "--warmup", "100",
          "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", "0", "--w-acc", "49.6969", "--detach-upper", "--fit-join"]
RH = [*COMMON, "--out", "/r/RH", "--dagger-p", "0.1", "--dagger-bank", "/b/rh", "--dagger-bank-sha256", SHA, "--dagger-seed", "0", "--dagger-bank-readout", "causal_g1x", "--dagger-bank-kappa", "0.2"]
B0 = [*COMMON, "--out", "/r/B0", "--dagger-p", "0"]


def test_control_the_pair_passes():
    assert P.pair_problems(RH, B0, SHA) == []
    assert P.pair_problems(RH, [*B0, "--dagger-seed", "0"], SHA) == [], "a B0 that names the same seed is the same arm"


def _without(argv, flag, n=1):
    i = argv.index(flag)
    return argv[:i] + argv[i + 1 + n:]


@pytest.mark.parametrize("flag", ["--dagger-bank-readout", "--dagger-bank-kappa", "--dagger-bank-sha256", "--dagger-bank"])
def test_g_name_an_unnamed_declaration_or_bank_is_refused(flag):
    bad = P.pair_problems(_without(RH, flag), B0, SHA)
    assert any(flag in m for m in bad), bad


@pytest.mark.parametrize("tok", ["causal_g", "causal_g1", "causal_g0.5x", "causal_g1.0x"])
def test_g_name_the_whole_token_is_required_not_the_base_name(tok):
    r = copy.deepcopy(RH)
    r[r.index("--dagger-bank-readout") + 1] = tok
    assert any("--dagger-bank-readout" in m and "causal_g1x" in m for m in P.pair_problems(r, B0, SHA))


def test_the_expected_token_is_the_one_function_s_spelling():
    from tanitad.data.v8f_dagger import readout_token
    assert P.RH_READOUT == readout_token("causal_g", 1.0, True) == "causal_g1x"


def test_g_name_a_wrong_value_or_a_wrong_sha_is_refused():
    r = copy.deepcopy(RH)
    r[r.index("--dagger-bank-readout") + 1] = "pinned"
    assert any("--dagger-bank-readout" in m for m in P.pair_problems(r, B0, SHA))
    assert any("not the bank's" in m for m in P.pair_problems(RH, B0, "cd" * 32))
    r = copy.deepcopy(RH)
    r[r.index("--dagger-bank-sha256") + 1] = "abc123"
    assert any("64 hex" in m for m in P.pair_problems(r, B0))


def test_g_pair_an_extra_difference_is_refused_and_so_is_a_missing_one():
    moved = copy.deepcopy(B0)
    moved[moved.index("--lr") + 1] = "2e-4"
    assert any("--lr" in m and "beyond" in m for m in P.pair_problems(RH, moved, SHA))
    same_out = copy.deepcopy(B0)
    same_out[same_out.index("--out") + 1] = "/r/RH"
    bad = P.pair_problems(RH, same_out, SHA)
    assert any("--out" in m for m in bad)
    named = [*B0, "--dagger-bank-readout", "causal_g1x"]
    assert any("declaration at --dagger-p 0" in m for m in P.pair_problems(RH, named, SHA))
    assert any("NAME --dagger-p" in m for m in P.pair_problems(RH, _without(B0, "--dagger-p"), SHA))
    assert any("must name --dagger-p 0" in m for m in P.pair_problems(RH, [*_without(B0, "--dagger-p"), "--dagger-p", "0.1"], SHA))


def test_cli_exit_codes(tmp_path, capsys):
    (tmp_path / "rh.json").write_text(json.dumps({"argv": RH}), encoding="utf-8")
    (tmp_path / "b0.json").write_text(json.dumps(B0), encoding="utf-8")
    assert P.main(["--rh", str(tmp_path / "rh.json"), "--b0", str(tmp_path / "b0.json"), "--bank-sha256", SHA]) == 0
    (tmp_path / "bad.json").write_text(json.dumps(_without(RH, "--dagger-bank-kappa")), encoding="utf-8")
    assert P.main(["--rh", str(tmp_path / "bad.json"), "--b0", str(tmp_path / "b0.json")]) == 1
    assert "ZZ ROLLHIST-PAIR FAIL" in capsys.readouterr().out
