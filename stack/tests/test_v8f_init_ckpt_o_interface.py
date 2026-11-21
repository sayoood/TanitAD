"""PREREG_V8F_FLATDAG launch blocker, the FLAT half: ``--init-ckpt`` must agree with ``--o-interface``.

MEASURED (oaccflat_check/RESULT.md F1): the launch gate ACCEPTED a FLAT-DAG argv that forgot ``--o-interface cut``. ``INIT_CKPT_MATCH`` did not contain ``o_interface``, so the gate built a FULL model, loaded the
cut-trained FLAT weights into it, and every probe passed (declared full == built full; O reads e_ck / e_code through weights that were never trained). Only the post-hoc C-F3 would have refused the run, after the
GPU hours. The fix is one key in ``INIT_CKPT_MATCH``; it is read only when the checkpoint HAS the key, so every checkpoint written before ``--o-interface`` existed (all M2b) loads exactly as before.

Each case runs the REAL trainer (``v8f_mini_train.main``) on its synthetic world with tiny dimensions. ``raw/2026-10-27-oaccflat-check`` runs this file against the tip trainer (RED) and the fixed one (GREEN).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(1)
COMMON = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]
GATE = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]


@pytest.fixture(scope="module")
def cks(tmp_path_factory):
    root = tmp_path_factory.mktemp("oi")
    out = {}
    for name, extra in (("cut", ["--o-interface", "cut"]), ("full", [])):
        assert TR.main(["--out", str(root / name), *COMMON, *extra, "--steps", "2"]) == 0
        out[name] = root / name / "ckpt_last.pt"
    blob = torch.load(out["full"], map_location="cpu", weights_only=False)
    assert blob["cfg"]["o_interface"] == "full"
    del blob["cfg"]["o_interface"]                                                         # a checkpoint written before the flag existed (every banked M2b)
    out["old"] = root / "old_no_key.pt"
    torch.save(blob, out["old"])
    assert torch.load(out["cut"], map_location="cpu", weights_only=False)["cfg"]["o_interface"] == "cut"
    return out, root


def gate(root, name, ck, extra=()):
    try:
        rc = TR.main(["--out", str(root / name), *GATE, "--init-ckpt", str(ck), *extra, "--selfcheck-only"])
    except SystemExit as e:
        rc = e.code
    return rc


def test_a_cut_checkpoint_with_an_argv_that_forgets_the_cut_is_refused(cks):
    out, root = cks
    rc = gate(root, "g1", out["cut"])
    assert isinstance(rc, str) and "ZZDVB-REFUSEDZZ init-ckpt" in rc and "o_interface: 'cut' != 'full'" in rc, rc


def test_a_cut_checkpoint_with_the_cut_argv_passes_and_builds_the_cut(cks):
    out, root = cks
    assert gate(root, "g2", out["cut"], ["--o-interface", "cut"]) == 0
    import json
    sc = json.loads((root / "g2" / "selfcheck.json").read_text(encoding="utf-8"))
    assert sc["levers"]["o_interface"] == "cut" and sc["o_interface_probe"]["o_interface"] == "cut" and sc["levers"]["init_ckpt"]["n_mismatch_after_load"] == 0


def test_a_full_checkpoint_with_a_cut_argv_is_refused_too(cks):
    out, root = cks
    rc = gate(root, "g3", out["full"], ["--o-interface", "cut"])
    assert isinstance(rc, str) and "ZZDVB-REFUSEDZZ init-ckpt" in rc and "o_interface: 'full' != 'cut'" in rc, rc


def test_a_full_checkpoint_with_the_default_argv_passes(cks):
    out, root = cks
    assert gate(root, "g4", out["full"]) == 0


@pytest.mark.parametrize("extra", [[], ["--o-interface", "cut"]])
def test_a_checkpoint_without_the_key_loads_as_before_under_either_argv(cks, extra):
    out, root = cks
    assert gate(root, "g5" + ("c" if extra else "f"), out["old"], extra) == 0


def test_the_match_tuple_names_o_interface_exactly_once_and_keeps_every_architecture_key():
    keys = TR.INIT_CKPT_MATCH
    assert keys.count("o_interface") == 1
    for k in ("d", "heads", "ff", "d_feat", "n_tok", "hist", "hmax", "n_q", "depth_s", "depth_t", "depth_o", "nb", "res_scale_m", "navonly", "wm", "wm_dim", "wm_strides", "wm_beta", "wm_gamma", "kin_mode", "max_age_t", "max_age_s"):
        assert k in keys
    assert "w_acc" not in keys, "w_acc is a loss weight, not an architecture key: DAG-OACC fine-tunes M2b (w_acc absent / 0.0) with --w-acc > 0"
