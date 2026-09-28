"""I3 (refcv7 box head BUILD.md; restart package 2026-09-28): the vendored refcv6 battery loader
rebuilds an OLD record at the query counts it STAMPED.

refcv7 A9 R4 moved the ONE spelling of the slot-query count from 100 to 300. A pre-A9 record (its
argv silent on `--agent-queries`, its stamps `seams.agents.queries: 100` and
`refcv6_perception.n_queries: 100`) rebuilt through `tanitad.eval.refcv6_loader.build_model` on a
post-A9 tree died on its STRICT load on BOTH heads (MEASURED by the I3 agent,
`…/2026-09-27-loader-stamped-queries/raw/refcv6_loader_exposure.log`). `refcv3_arm.load_model` already
carried the stamped rule; the loader now carries the same two lines.

Records are synthesised through the trainer's OWN `build_parser` + `_pin_trainer_cfg` +
`RefCV3Model` + `build_perception_branch` (the I3 agent's recipe; resnet18, no pretrained weights,
CPU). Every expected count is a LITERAL read off the loaded query tables.
RED on the unpatched loader: test 1 raises the strict-load RuntimeError (both size mismatches).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

STACK = Path(__file__).resolve().parents[1]
ROOT = STACK.parent
for _p in (str(STACK), str(STACK / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ARGV = ["--arm", "hier", "--smoke", "--trunk", "timm", "--trunk-name", "resnet18.a1_in1k",
        "--no-trunk-pretrained", "--trunk-in-channels", "9", "--image-hw", "256", "640",
        "--agents", "head", "--agent-join", "join.jsonl", "--w-agent", "1.0", "--w-box3d", "1.0",
        "--out", "x"]
A9_AGENT_KEYS = ("presence_loss", "presence_prior", "deep_supervision", "vis1")
A9_PERCEPTION_KEYS = A9_AGENT_KEYS + ("query_select", "bev_source", "planner_crop_m")


@pytest.fixture(scope="module")
def T():
    spec = importlib.util.spec_from_file_location("refc_v3_train_i3_rq", str(STACK / "scripts" /
                                                                            "refc_v3_train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def L(monkeypatch):
    """The vendored loader resolves ITS tree from REFCV6_REPO at import: point it at THIS tree."""
    monkeypatch.setenv("REFCV6_REPO", str(ROOT))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    sys.modules.pop("tanitad.eval.refcv6_loader", None)
    mod = importlib.import_module("tanitad.eval.refcv6_loader")
    yield mod
    sys.modules.pop("tanitad.eval.refcv6_loader", None)


def _record(T, run: Path, queries: int, pre_a9: bool) -> tuple[dict, Path]:
    from tanitad.models import refcv6_perception_branch as PB
    from tanitad.refs import refc_v3 as v3
    args = T.build_parser().parse_args(ARGV)
    args.agent_queries = queries
    cfg = T._pin_trainer_cfg(v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = v3.RefCV3Model(cfg)
    pcfg = PB.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, n_queries=queries)
    model._perception = PB.build_perception_branch(model, pcfg)
    agents, perc = dict(cfg.core.agents.as_dict()), dict(pcfg.as_dict())
    if pre_a9:
        for k in A9_AGENT_KEYS:
            agents.pop(k, None)
        for k in A9_PERCEPTION_KEYS:
            perc.pop(k, None)
        agents["n_queries_default_upstream"] = queries
    perc["branch_params"] = model._perception.param_breakdown()
    config = {"argv": list(ARGV), "arm": "hier", "image_hw": list(cfg.core.encoder.image_hw()),
              "horizons": list(cfg.core.trajectory.horizons), "seams": {"agents": agents},
              "refcv6_perception": perc,
              "param_breakdown": {k: int(v) for k, v in v3.param_breakdown_v3(model).items()}}
    run.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 7, "model": model.state_dict(), "opt": {}}, run / "ckpt.pt")
    (run / "config.json").write_text(json.dumps(config, default=str), encoding="utf-8")
    return config, run / "ckpt.pt"


def _counts(model) -> list[int]:
    return [int(model.core.agent_head.queries.shape[1]),
            int(model._perception.box_dec.queries.shape[1])]


def test_1_a_PRE_A9_record_rebuilds_at_100_100_STRICT(T, L, tmp_path):
    config, ck = _record(T, tmp_path / "pre", 100, pre_a9=True)
    model, _cfg, _args, rec = L.build_model(config, str(ck), device="cpu", strict=True)
    assert _counts(model) == [100, 100]
    assert rec["agent_queries"]["as_trained"] == 100 and rec["agent_queries"]["parser"] == 300
    assert rec["box_queries"]["stamp"] == 100


def test_2_a_POST_A9_record_rebuilds_at_300_300_STRICT(T, L, tmp_path):
    config, ck = _record(T, tmp_path / "post", 300, pre_a9=False)
    model, _cfg, _args, _rec = L.build_model(config, str(ck), device="cpu", strict=True)
    assert _counts(model) == [300, 300]


def test_3_RED_a_record_whose_box_stamp_LIES_is_refused_on_the_box_head(T, L, tmp_path):
    """The stamp is READ, not bypassed: a 100-query record whose box stamp claims 300 fails on the
    box table and ONLY there (the agent head still rebuilds from its own stamp)."""
    config, ck = _record(T, tmp_path / "lie", 100, pre_a9=True)
    config["refcv6_perception"]["n_queries"] = 300
    with pytest.raises(RuntimeError) as ei:
        L.build_model(config, str(ck), device="cpu", strict=True)
    msg = str(ei.value)
    assert "_perception.box_dec.queries" in msg and "core.agent_head.queries" not in msg


def test_4_the_vendored_body_still_matches_its_pin():
    """A11 kept: the vendored body re-hashes to the (re-pinned) harness constant."""
    spec = importlib.util.spec_from_file_location("g_box_overfit_i3_rq",
                                                  str(STACK / "scripts" / "g_box_overfit.py"))
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    src = (STACK / "tanitad" / "eval" / "refcv6_loader.py").read_bytes().decode("utf-8")
    assert G.vendored_audit_blob(src) == G.AUDIT_LOADER_BLOB == \
        "4e823c7373d1c31aa6240306196fefcc23d7860e"
