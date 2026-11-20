"""PREREG_V8F_N1DH control K-LIM (``stack/scripts/n1dh_klim.py``): the route head's accuracy with the logged / the causal / no speed bucket. CPU, one thread. REPORT ONLY: these tests pin the MECHANICS (the bucket variants, the
exact-0.0 control when the model cannot read ``limit``, the non-zero reading when it can, the CLI), not a measurement."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)
REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "stack" / "scripts", REPO / "taniteval", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import n1dh_klim as K  # noqa: E402
import v8f_mini_train as TR  # noqa: E402
from test_v8f_n1dh import build, routed, world  # noqa: E402,F401


def _model_and_batch(world, n=16):
    b, J = routed(world, n=n)
    m = build(J, route=True, detach_upper=True)
    return m, b, J


def test_causal_limit_is_a_valid_one_hot_from_the_speed_at_now_only(world):
    m, b, J = _model_and_batch(world)
    oh, ok = K.causal_limit(b)
    assert tuple(oh.shape) == (16, 4) and bool(ok.all()) and bool((oh.sum(1) == 1).all())
    b2 = dict(b)
    b2["gt"] = b["gt"] * 7.0                                                              # the future is not read
    b2["lab_t"] = {k: v for k, v in b["lab_t"].items()}
    oh2, _ = K.causal_limit(b2)
    assert torch.equal(oh, oh2)
    b3 = dict(b); kin = b["kin3"].clone(); kin[:, 0] = float("nan"); b3["kin3"] = kin
    oh3, ok3 = K.causal_limit(b3)
    assert not bool(ok3.any()) and float(oh3.abs().sum()) == 0.0, "no finite speed -> the off format"


def test_variants_logged_is_the_batch_off_is_zeros(world):
    m, b, J = _model_and_batch(world)
    assert torch.equal(K.with_limit(b, "logged")["limit"], b["limit"])
    o = K.with_limit(b, "off")
    assert float(o["limit"].abs().sum()) == 0.0 and not bool(o["limit_ok"].any())
    assert torch.equal(K.with_limit(b, "causal")["limit"], K.causal_limit(b)[0])
    with pytest.raises(ValueError):
        K.with_limit(b, "future")


def test_klim_reads_exactly_zero_gap_when_the_model_cannot_read_limit_and_nonzero_when_it_can(world):
    m, b, J = _model_and_batch(world)
    with torch.no_grad():
        m.e_lim.weight.zero_(); m.e_lim.bias.zero_()
    r0 = K.klim(m, b)
    assert r0["causal"]["plan_l1_to_logged"] == 0.0 and r0["off"]["plan_l1_to_logged"] == 0.0 and r0["gap_logged_minus_causal"] == 0.0 and r0["gap_logged_minus_off"] == 0.0
    m2, b2, _ = _model_and_batch(world)
    with torch.no_grad():
        m2.e_lim.weight.copy_(torch.randn(m2.e_lim.weight.shape, generator=torch.Generator().manual_seed(5)) * 2.0)
        set = torch.randn(m2.route_head.weight.shape, generator=torch.Generator().manual_seed(6)) * 0.5
        m2.route_head.weight.copy_(set)
    r1 = K.klim(m2, b2)
    assert r1["off"]["plan_l1_to_logged"] >= 0.0 and r1["n_labelled_rows"] > 0 and 0.0 < r1["majority_share"] <= 1.0
    assert sum(r1["logged"]["argmax_class_counts"]) == r1["n_labelled_rows"]
    assert (r1["logged"]["mean_max_posterior"] != r0["logged"]["mean_max_posterior"]) or (r1["off"]["mean_max_posterior"] != r1["logged"]["mean_max_posterior"]), "a model that reads limit moves with it"


def test_klim_refuses_a_model_without_the_head(world):
    b, J = routed(world)
    with pytest.raises(ValueError, match="aux-route cond"):
        K.klim(build(J, route=False), b)


def test_cli_loads_a_checkpoint_and_a_saved_batch_and_writes_the_record(world, tmp_path):
    m, b, J = _model_and_batch(world)
    ck = {"model": {k: v.clone() for k, v in m.state_dict().items()}, "cfg": dict(m.cfg.__dict__)}
    torch.save(ck, tmp_path / "ck.pt")
    torch.save({k: v for k, v in b.items() if k not in ("route_y", "route_ok")}, tmp_path / "b.pt")             # the CLI runs route_stage itself when the batch has no labels
    assert K.main(["--ckpt", str(tmp_path / "ck.pt"), "--batch-pt", str(tmp_path / "b.pt"), "--out", str(tmp_path / "o.json")]) == 0
    rec = json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
    assert set(rec) >= {"logged", "causal", "off", "n_labelled_rows", "majority_share", "gap_logged_minus_causal", "note"}
    ref = K.klim(m, b)
    assert rec["logged"]["acc"] == ref["logged"]["acc"] and rec["causal"]["plan_l1_to_logged"] == ref["causal"]["plan_l1_to_logged"]
