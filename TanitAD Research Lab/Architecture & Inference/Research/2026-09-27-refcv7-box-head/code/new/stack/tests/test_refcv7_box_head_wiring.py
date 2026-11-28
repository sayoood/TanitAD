"""refcv7 A9/A10 -- the TRAINER wiring of the refined slot heads: flags, pins, refusals, G-DVB entries, the loss path
through ``compute_losses_v3``, the LOGGING_SPEC keys, and the as-trained query count of a pre-A9 record.

Each guard is shown to go RED under its defect (the refusal fires / the mismatch is named / the key disappears).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_T = None


def _trainer():
    global _T
    if _T is None:
        spec = importlib.util.spec_from_file_location("refc_v3_train_for_boxhead",
                                                      str(ROOT / "scripts" / "refc_v3_train.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _T = mod
    return _T


BASE = ["--arm", "hier", "--size", "tiny", "--out", "X"]
AGENT = ["--agents", "head", "--agent-join", "join.jsonl", "--w-agent", "1.0"]
REFINE = ["--slot-presence-loss", "focal", "--slot-presence-prior", "0.01", "--slot-deep-supervision"]


def _args(*extra):
    return _trainer().build_parser().parse_args(BASE + list(extra))


# --------------------------------------------------------------------------------------------------------- #
# flags: declared, legacy by default                                                                        #
# --------------------------------------------------------------------------------------------------------- #
def test_the_five_flags_exist_and_default_to_the_pre_A9_arm():
    a = _args()
    assert (a.slot_presence_loss, a.slot_presence_prior, a.slot_deep_supervision, a.slot_vis1,
            a.vis1_sidecar) == ("bce", 0.05, False, False, None)
    assert a.agent_queries == 300


def test_the_config_fields_carry_the_flags_for_BOTH_heads():
    T = _trainer()
    k = T._slot_refine_kwargs(_args(*REFINE))
    assert k == {"presence_loss": "focal", "presence_prior": 0.01, "deep_supervision": True, "vis1": False}
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), _args(*AGENT, *REFINE))
    ag = cfg.core.agents
    assert (ag.presence_loss, ag.presence_prior, ag.deep_supervision, ag.vis1) == ("focal", 0.01, True, False)
    st = ag.as_dict()
    assert st["presence_loss"] == "focal" and st["deep_supervision"] is True and st["queries"] == 300


@pytest.mark.parametrize("extra,why", [
    (REFINE, "NO learned slot head"),                                                # no head at all
    (AGENT + ["--slot-vis1"], "without --vis1-sidecar"),
    (AGENT + ["--vis1-sidecar", "s.npz"], "without --slot-vis1"),
    (AGENT + ["--slot-vis1", "--vis1-sidecar", "s.npz"], "--agent-join AND --join3d"),
    (AGENT + ["--slot-presence-prior", "1.5"], r"\(0, 1\)"),
])
def test_every_dead_refinement_REFUSES_before_the_gpu(extra, why):
    T = _trainer()
    with pytest.raises(SystemExit, match=why):
        T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), _args(*extra))


# --------------------------------------------------------------------------------------------------------- #
# the model: built as declared, G-DVB reads it back                                                         #
# --------------------------------------------------------------------------------------------------------- #
def _model(*extra):
    T = _trainer()
    args = _args(*extra)
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    m = T.v3.RefCV3Model(cfg)
    m._w_agent = float(args.w_agent)
    m._vis1 = bool(args.slot_vis1)
    m._cls_class_weight = None
    return m, args


def test_the_built_agent_head_carries_the_declared_refinement():
    m, _a = _model(*AGENT, *REFINE)
    h = m.core.agent_head
    assert h.n_queries == 300 and h.deep_supervision is True and h.presence_prior == 0.01
    assert float(h.head.bias[0].detach()) == pytest.approx(math.log(0.01 / 0.99), abs=1e-6)


def test_GDVB_the_new_levers_are_clean_when_built_and_named_when_not():
    from tanitad.train import declared_vs_built as dvb
    m, a = _model(*AGENT, *REFINE)
    lev = ("slot_presence_loss", "slot_presence_prior", "slot_deep_supervision", "slot_vis1")
    for d in lev:
        assert dvb.REGISTRY[d].kind == "built"
        assert dvb.REGISTRY[d].check(m, a) == [], d
    assert dvb.REGISTRY["vis1_sidecar"].kind == "data"
    # RED arms: unplumb each built value in turn
    m.core.agent_head.deep_supervision = False
    assert [x.lever for x in dvb.REGISTRY["slot_deep_supervision"].check(m, a)] == ["--slot-deep-supervision"]
    m.core.agent_head.deep_supervision = True
    m.core.agent_head.presence_prior = 0.05
    assert [x.lever for x in dvb.REGISTRY["slot_presence_prior"].check(m, a)] == ["--slot-presence-prior"]
    m.core.cfg.agents.presence_loss = "bce"
    assert [x.lever for x in dvb.REGISTRY["slot_presence_loss"].check(m, a)] == ["--slot-presence-loss"]
    m._vis1 = True
    assert "--slot-vis1" in [x.lever for x in dvb.REGISTRY["slot_vis1"].check(m, a)]


def test_GDVB_a_refinement_on_a_model_with_no_slot_head_is_a_mismatch():
    from tanitad.train import declared_vs_built as dvb
    m, a = _model()
    a.slot_presence_loss = "focal"                     # declared, but nothing built reads it
    assert [x.lever for x in dvb.REGISTRY["slot_presence_loss"].check(m, a)] == ["--slot-presence-loss"]


def test_GDVB_every_trainer_flag_is_still_covered():
    from tanitad.train import declared_vs_built as dvb
    assert dvb.coverage(_trainer().build_parser()) == []


# --------------------------------------------------------------------------------------------------------- #
# the loss path through compute_losses_v3 (a G-LIVE miniature on the smoke model)                           #
# --------------------------------------------------------------------------------------------------------- #
def _batch(T, cfg, *, vis1: bool):
    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    b = torch.utils.data.default_collate([ds[0], ds[1]])
    P = 8
    box = torch.zeros(2, P, 4)
    box[:, :4] = torch.tensor([[10.0, 1.0, 4.5, 1.9], [22.0, -3.0, 4.5, 1.9], [35.0, 4.0, 4.5, 1.9],
                               [80.0, 0.0, 4.5, 1.9]])
    valid = torch.zeros(2, P, dtype=torch.bool)
    valid[:, :4] = True
    b.update(agent_box=box, agent_yaw=torch.zeros(2, P), agent_cls=torch.zeros(2, P, dtype=torch.long),
             agent_valid=valid, agent_occ=torch.full((2, P), -1.0), agent_rates=torch.zeros(2, P, 3),
             agent_rates_mask=torch.zeros(2, P, dtype=torch.bool), agent_label=torch.tensor([True, True]),
             agent_ep=torch.tensor([11, 12]))
    if vis1:
        b["agent_vis_full"] = torch.full((2, P), 1000, dtype=torch.int32)
        b["agent_vis_px"] = torch.tensor([[900, 30, 800, 900, 0, 0, 0, 0]] * 2, dtype=torch.int32)
        b["agent_vis_known"] = valid.clone()
    return b


def _losses(vis1: bool, *, train: bool = True, drop_vis_keys: bool = False):
    T = _trainer()
    m, _a = _model(*AGENT, *REFINE)
    if vis1:                                          # the pin needs --join3d (box3d); the loss path does not
        m.core.cfg.agents.vis1 = True
        m._vis1 = True
    _ac = m.core.decoder.anchor_controls
    with torch.no_grad():
        _ac.copy_(torch.stack([torch.linspace(-2.0, 2.0, _ac.shape[0]),
                               torch.linspace(-1.5, 1.5, _ac.shape[0])], -1).to(_ac.dtype))
    b = _batch(T, m.cfg, vis1=vis1)
    if drop_vis_keys:
        for k in ("agent_vis_full", "agent_vis_px", "agent_vis_known"):
            b.pop(k)
    m.train(train)
    out = T.compute_losses_v3(m, b, "cpu", mode="diffusion")
    return m, out


def test_the_refined_agent_loss_runs_in_the_trainer_with_every_new_term_finite_and_live():
    m, out = _losses(vis1=True)
    for k in ("agent_presence", "agent_layer0", "agent_layer1", "agent_layer2", "agent_presence_layer0"):
        assert k in out and torch.isfinite(torch.as_tensor(out[k])).all(), k
    assert out["agent_n_layers"] == 3.0
    # LOGGING_SPEC §1 census, per row
    assert out["agent_n_pos"] == 4.0 and out["agent_n_ignore"] == 2.0      # rows 0, 2 POS x2; row 1 IGN x2
    for k in ("agent_n_dropped_hidden", "agent_n_ignore_masked_slots", "agent_n_conf", "agent_conf_ratio",
              "agent_tp@gate", "agent_presence_frac_confident"):
        assert k in out, k
    assert out["agent_presence_frac_confident"] < 0.5                        # G-LIVE-PRES at init (prior 0.01)
    out["loss"].backward()
    h = m.core.agent_head
    for name, p in (("queries", h.queries), ("head", h.head.weight), ("layer0", h.blocks.layers[0].linear1.weight)):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, name
    assert "_det_pack_agent" not in out                                     # packs only in EVAL mode


def test_eval_mode_emits_the_pooled_pack():
    _m, out = _losses(vis1=True, train=False)
    pk = out["_det_pack_agent"]
    assert len(pk) == 2 and pk[0]["pos"].tolist() == [True, False, True, False]
    assert pk[0]["ign"].tolist() == [False, True, False, False]            # row 3 (80 m) is outside the filter


def test_RED_ARM_vis1_on_without_the_batch_block_REFUSES():
    with pytest.raises(SystemExit, match="no \\['agent_vis_full'"):
        _losses(vis1=True, drop_vis_keys=True)


def test_the_refined_path_without_vis1_emits_no_census_and_still_trains():
    _m, out = _losses(vis1=False)
    assert "agent_layer2" in out and "agent_n_pos" not in out


# --------------------------------------------------------------------------------------------------------- #
# R4: a PRE-A9 record rebuilds at ITS OWN query count                                                        #
# --------------------------------------------------------------------------------------------------------- #
def test_a_pre_A9_record_rebuilds_at_its_stamped_query_count():
    T = _trainer()
    refcv6_like = {"argv": ["--arm", "hier", "--agents", "head", "--w-agent", "1.0"],
                   "seams": {"agents": {"queries": 100}}}
    args = T.build_parser().parse_args(refcv6_like["argv"] + ["--out", "X"])
    assert args.agent_queries == 300                           # the RED arm: the parser alone says 300
    assert T.agent_queries_as_trained(refcv6_like, args) == (100, "config.json seams.agents.queries (argv silent)")
    explicit = {"argv": ["--agent-queries", "64"], "seams": {"agents": {"queries": 64}}}
    a2 = T.build_parser().parse_args(["--arm", "hier", "--agent-queries", "64", "--out", "X"])
    assert T.agent_queries_as_trained(explicit, a2)[0] == 64
    assert T.agent_queries_as_trained({"argv": []}, args)[0] is None


def test_the_arm_loader_uses_the_as_trained_count():
    tv = ROOT.parent / "taniteval" / "tools" / "refcv3_arm.py"
    if not tv.exists():
        pytest.skip("taniteval not in this tree")
    src = tv.read_text(encoding="utf-8")
    assert "agent_queries_as_trained" in src and "args.agent_queries = int(_n_q)" in src
    for k in ("presence_loss", "presence_prior", "deep_supervision", "vis1"):
        assert f'"{k}"' in src.split("def rebuild_perception_branch")[1].split("pcfg = ")[0], k


def test_the_config_json_block_is_absent_for_a_pre_A9_arm():
    T = _trainer()
    m, a = _model()
    assert T._slot_refine_block(a, m) is None
    m2, a2 = _model(*AGENT, *REFINE)
    blk = T._slot_refine_block(a2, m2)
    assert blk["presence_loss"] == "focal" and blk["heads_built"]["agent"]["n_queries"] == 300
    assert blk["heads_built"]["agent"]["n_decoder_layers"] == 3 and blk["decision_rule"].startswith("sigmoid")


# --------------------------------------------------------------------------------------------------------- #
# the refcv7 launch requirement (box_head_guard): ON in argv AND built, on BOTH heads                        #
# --------------------------------------------------------------------------------------------------------- #
def _refcv7_like():
    import types
    from tanitad.models import box3d_head as B3
    from tanitad.models.refcv6_perception_branch import PerceptionBranchConfig
    m, a = _model(*AGENT, *REFINE)
    a.slot_vis1, a.vis1_sidecar = True, "vis1_sidecar.npz"
    m._vis1 = True
    m.core.cfg.agents.vis1 = True
    bd = B3.Box3DSlotDecoder(16, 8, n_queries=300, d_model=32, depth=3, n_heads=4, enforce_band=False,
                             presence_prior=0.01)
    bd.deep_supervision = True
    m._perception = types.SimpleNamespace(
        box_dec=bd, cfg=PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, presence_loss="focal", presence_prior=0.01,
                                               deep_supervision=True, vis1=True, enforce_param_band=False))
    return m, a


def test_the_refcv7_box_requirement_passes_on_a_refcv7_build():
    from tanitad.train.box_head_guard import REFCV7_BOX_REQUIRED, check_refcv7_box_required
    assert REFCV7_BOX_REQUIRED == {"slot_presence_loss": "focal", "slot_presence_prior": 0.01,
                                   "slot_deep_supervision": True, "slot_vis1": True}
    m, a = _refcv7_like()
    assert check_refcv7_box_required(m, a) == []


@pytest.mark.parametrize("break_it,lever", [
    (lambda m, a: setattr(a, "slot_presence_loss", "bce"), "--slot-presence-loss"),
    (lambda m, a: setattr(a, "slot_deep_supervision", False), "--slot-deep-supervision"),
    (lambda m, a: setattr(m._perception.box_dec, "deep_supervision", False), "--slot-deep-supervision"),
    (lambda m, a: setattr(m.core.agent_head, "presence_prior", 0.05), "--slot-presence-prior"),
    (lambda m, a: setattr(a, "vis1_sidecar", None), "--vis1-sidecar"),
    (lambda m, a: setattr(m, "_perception", None), "box3d slot head"),
])
def test_RED_ARMS_each_missing_A9_lever_is_named(break_it, lever):
    from tanitad.train.box_head_guard import check_refcv7_box_required
    m, a = _refcv7_like()
    break_it(m, a)
    assert lever in [x.lever for x in check_refcv7_box_required(m, a)]


def test_the_pre_A9_argv_fails_the_requirement_everywhere():
    from tanitad.train.box_head_guard import check_refcv7_box_required
    m, a = _model(*AGENT)
    levers = {x.lever for x in check_refcv7_box_required(m, a)}
    assert {"--slot-presence-loss", "--slot-deep-supervision", "--slot-vis1", "--vis1-sidecar"} <= levers
