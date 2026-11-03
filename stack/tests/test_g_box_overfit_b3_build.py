"""G-BOX-OVERFIT builds -- and trains, and scores -- the B3 own-head model the argv DECLARES (refcv9 E1 arms B3 / B3p; BRIEF_BOX_B3).

WHY (MEASURED on Thor 2026-11-03 18:09Z, ``/home/nvidia/r8ns/r9e1/ovf_B3_23fbf8f7e2b1/box-b3.out``): ``r9e1_bind.sh box-b3 B3`` runs
``g_box_overfit.py`` on ``stack/ops/runs.d/r9e1_B3.argv.json`` (``--r9-box-quality --r9-box-dn --r9-box-class-sqrt``) and exited 1 at the harness's own
build check, after ``C3 PASS``:

    [gbo] the replayed build is NOT the declared launch model:
      - r9_box_quality: declared True but BUILT False (read from model._r9_box_b3.quality) -- argv and the attribute the LOSS reads disagree
      - r9_box_dn: declared True but BUILT False (read from model._r9_box_b3.dn) -- argv and the attribute the LOSS reads disagree
      - r9_box_dn: declared True but BUILT False (read from model._perception.box_dec.dn_embed) -- argv and the BUILT DN parameters disagree
      - r9_box_class_sqrt: declared True but BUILT False (read from model._r9_box_b3.class_sqrt) -- argv and the attribute the LOSS reads disagree
      - r9_box_class_sqrt: declared True but BUILT False (read from model._r9_box_cls_weight) -- argv and the BUILT sqrt class weight disagree

``train()`` builds the B3 decoder state with the trainer's own ``_bbf.attach(model, args)`` right after ``build_perception_branch``
(``refc_v3_train.py:10092``); ``TrainerAdapter._build_model`` replays that build and never called it (its header comment, "the trainer's own
``compute_losses_v3`` already builds the B3 decoder", was wrong). So ``_bbf.prepare`` / ``_bbf.box3d_row`` would have run B3 as OFF.

The REFERENCE is the trainer's own ``train()`` on a tiny rig, captured as the launch gate captures it (``launch_gate.run_trainer_until(..., "model")``,
before any data is read); every expectation is that model or a LITERAL.

* the harness replay of a B3 / B3p rig argv IS train()'s model: G-DVB clean, same state_dict keys, same fresh-init digests, the attribute the loss reads,
  the sqrt class weight, the DN parameters and the NMS tau equal the trainer's;
* the OFF path is BIT-IDENTICAL: on an argv with no ``--r9-box-*`` flag the harness and a COPY with the attach call removed (= Y4's replay) build the same
  keys and digests, both equal train()'s model, the attribute is ABSENT, and ``--identity-only`` gives the same ``identity_sha256`` (C0 C1 M1 M2a B1 T1 + B3 B3p);
* RED: remove the attach call from a COPY => the five measured mismatches return BY NAME; attach without DN => only the DN pair;
* (a) the LOSS row of a loss-changing lever is ``box_b3.box3d_loss_row_b3`` (NOT ``PB.box3d_loss_row``): the one-frame capture wraps both and is
  non-empty; with the B3 wrapper removed the harness REFUSES (and without the refusal the same defect is a silent ``keep = None`` on every row);
* (b) DN is ON in a training step (a pack is set, ``box3d_dn`` is in the row) and OFF in eval (no pack set, no DN output); the batch carries the DN inputs;
* (c) the replay's ``m._dn_gen`` is read by nothing; B3's DN draws from the GLOBAL torch generator: the two do not interact;
* the TRAINER-SIDE fix (``box_b3_flag.box3d_row`` reads a DN config with DN off in EVAL; MEASURED on the tiny path: ``refined_box3d_losses_b3`` raised
  "--r9-box-dn is on and the loss ran with no DN output" in eval, so every in-training / EMA / final eval of a DN arm was an ``eval_error`` row): (a) the
  trainer's own eval row of a DN arm is a row (no raise); (b) the TRAINING row and its gradients are BIT-IDENTICAL to the pre-fix function for B3, B3p and an
  OFF model, an OFF model's eval row too; (c) RED: with the guard removed (= Y4) the eval row raises again, in the trainer's call and in the harness's
  ``evaluate()``. The harness has no eval override of its own any more: it scores through the trainer's row.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

STACK = Path(__file__).resolve().parents[1]
for _p in (str(STACK), str(STACK / "scripts"), str(STACK / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG                                           # noqa: E402
import test_g_box_overfit_near_lift as NL                          # noqa: E402  (its tiny rig, inputs, adapter helpers)
import test_g_box_overfit_rig as RG                                # noqa: E402  (the E1 rig argv at tiny size)

GBO_PATH = STACK / "scripts" / "g_box_overfit.py"
TRAINER_PATH = STACK / "scripts" / "refc_v3_train.py"
RUNS = STACK / "ops" / "runs.d"
AUDIT = STACK / "ops" / "refcv8_box_audit"
B3_FLAGS = ["--r9-box-quality", "--r9-box-dn", "--r9-box-class-sqrt"]
B3P_FLAGS = B3_FLAGS + ["--r9-box-inloss-nms"]
ATTACH_LINE = "        tr._bbf.attach(m, a)\n"
#: the five lines Thor measured (and the tiny rig reproduces) for the replay WITHOUT the attach call -- in this order
MEASURED_MISMATCHES = [
    "  - r9_box_quality: declared True but BUILT False (read from model._r9_box_b3.quality) -- argv and the attribute the LOSS reads disagree",
    "  - r9_box_dn: declared True but BUILT False (read from model._r9_box_b3.dn) -- argv and the attribute the LOSS reads disagree",
    "  - r9_box_dn: declared True but BUILT False (read from model._perception.box_dec.dn_embed) -- argv and the BUILT DN parameters disagree",
    "  - r9_box_class_sqrt: declared True but BUILT False (read from model._r9_box_b3.class_sqrt) -- argv and the attribute the LOSS reads disagree",
    "  - r9_box_class_sqrt: declared True but BUILT False (read from model._r9_box_cls_weight) -- argv and the BUILT sqrt class weight disagree",
]


@pytest.fixture(scope="module")
def rigs(tmp_path_factory):
    d = tmp_path_factory.mktemp("gbo_b3_build")
    NL._synth_inputs(d)
    argvs = {"b3": RG._rig_argv(d) + B3_FLAGS, "b3p": RG._rig_argv(d) + B3P_FLAGS, "off": RG._rig_argv(d)}
    T = NL._load("refc_v3_train_gbo_b3_ref", TRAINER_PATH)
    refs = {k: LG.run_trainer_until(T, v, "model")["model"] for k, v in argvs.items()}      # the REFERENCES: the real train()
    G = NL._load("g_box_overfit_b3_build_t", GBO_PATH)
    return types.SimpleNamespace(d=d, T=T, G=G, argvs=argvs, ref=refs)


def _mutated(tmp_path: Path, name: str, edits: list[tuple[str, str]]):
    src = GBO_PATH.read_text(encoding="utf-8")
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    p = tmp_path / f"{name}.py"
    p.write_text(src, encoding="utf-8", newline="\n")
    return NL._load(f"{name}_t", p)


def _y4_edits() -> list[tuple[str, str]]:
    """The edit that turns the harness back into Y4's replay: the attach call removed."""
    return [(ATTACH_LINE, "")]


def _digests(m):
    return LG.state_digests(m.state_dict())


def _items(T, ref, n=4):
    """``n`` synthetic harness items (a dataset window + the agent / VIS-1 / map blocks the rig loss reads), collated by the adapter like the real ones."""
    from tanitad.data.v2_dataset import stable_episode_id
    cfg = ref.cfg
    eps = T._synth_episodes(2, cfg.core, seed=0, clip_ids=list(NL.CLIPS))
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    items, P = [], 8
    for i in range(n):
        it = dict(ds[i % 2])
        box = torch.zeros(P, 4)
        box[:3] = torch.tensor([[10.0, 1.0, 4.5, 1.9], [22.0 + i, -3.0, 4.5, 1.9], [35.0, 4.0, 4.5, 1.9]])
        valid = torch.zeros(P, dtype=torch.bool)
        valid[:3] = True
        it.update(agent_box=box, agent_yaw=torch.zeros(P), agent_cls=torch.zeros(P, dtype=torch.long), agent_valid=valid,
                  agent_occ=torch.full((P,), -1.0), agent_rates=torch.zeros(P, 3), agent_rates_mask=torch.zeros(P, dtype=torch.bool),
                  agent_label=torch.tensor(True), agent_ep=torch.tensor(11 + i),
                  agent_vis_full=torch.full((P,), 1000, dtype=torch.int32),
                  agent_vis_px=torch.tensor([900, 30, 800, 0, 0, 0, 0, 0], dtype=torch.int32), agent_vis_known=valid.clone())
        codes = torch.randint(0, 8, (600, 320), dtype=torch.uint8)
        codes[:40] = 255
        it.update(map_fine=codes, map_fine_label=torch.tensor(True), map_ep=torch.tensor(int(stable_episode_id(NL.CLIPS[i % 2]))))
        items.append(it)
    return items


def _adapter(G, T, argv, ref):
    ad = NL._adapter(G, T, argv)
    ad.items, ad._hooks, ad._patched = _items(T, ref), [], []
    ad.per_frame = [(f"{i:012x}", i, 2, 1) for i in range(len(ad.items))]
    return ad


# ---------------------------------------------------------------------------------------------------------------------- #
# 0. the rig carries what B3 / B3p declare                                                                                #
# ---------------------------------------------------------------------------------------------------------------------- #
def test_the_tiny_rig_carries_exactly_the_B3_and_B3p_levers():
    b3 = json.loads((RUNS / "r9e1_B3.argv.json").read_text(encoding="utf-8"))["argv"]
    b3p = json.loads((RUNS / "r9e1_B3p.argv.json").read_text(encoding="utf-8"))["argv"]
    assert [t for t in b3 if t.startswith("--r9-box-")] == B3_FLAGS
    assert [t for t in b3p if t.startswith("--r9-box-")] == B3P_FLAGS
    for av in (b3, b3p):
        assert "--r9-perc-rig" in av and "--slot-vis1" in av and av[av.index("--slot-presence-loss") + 1] == "focal"   # the rig loss path B3 rides


# ---------------------------------------------------------------------------------------------------------------------- #
# 1. the build                                                                                                            #
# ---------------------------------------------------------------------------------------------------------------------- #
def test_MEASURED_the_Y4_replay_of_a_B3_argv_is_refused_with_the_five_Thor_mismatches(rigs, tmp_path):
    """The defect as it stood on tip Y4 (= the harness with the attach call removed): exactly the text Thor printed."""
    G0 = _mutated(tmp_path, "gbo_y4_replay", _y4_edits())
    with pytest.raises(SystemExit) as e:
        NL._harness_model(G0, rigs.T, rigs.argvs["b3"])
    lines = str(e.value).splitlines()
    assert lines[0] == "[gbo] the replayed build is NOT the declared launch model:"
    assert lines[1:] == MEASURED_MISMATCHES


@pytest.mark.parametrize("kind,flags", [("b3", B3_FLAGS), ("b3p", B3P_FLAGS)])
def test_the_harness_replay_IS_the_B3_model_train_builds(rigs, kind, flags):
    m, ad = NL._harness_model(rigs.G, rigs.T, rigs.argvs[kind])
    ref, T = rigs.ref[kind], rigs.T
    assert ad.dvb_mismatches == []                                                       # 0 mismatches (the five are gone)
    assert m._r9_box_b3 == ref._r9_box_b3
    assert m._r9_box_b3.as_dict() == {"quality": True, "dn": True, "class_sqrt": True, "inloss_nms": kind == "b3p"}
    dec, dec_t = m._perception.box_dec, ref._perception.box_dec
    assert dec.dn_embed is not None and dec.dn_cls_embed is not None                     # the DN parameters
    assert dec.b3_nms_tau == dec_t.b3_nms_tau == (0.1 if kind == "b3p" else None)
    assert torch.equal(m._r9_box_cls_weight.cpu(), ref._r9_box_cls_weight.cpu())         # the sqrt-tempered vector
    from tanitad.models.box_b3 import sqrt_temper
    assert torch.allclose(m._r9_box_cls_weight.cpu(), sqrt_temper(m._cls_class_weight).cpu(), atol=1e-6)
    kt, kh = set(ref.state_dict()), set(m.state_dict())
    assert sorted(kt ^ kh) == [] and any(k.startswith("_perception.box_dec.dn_") for k in kh)
    dt, dh = _digests(ref), _digests(m)
    assert [k for k in dt if dt[k] != dh[k]] == []                                       # same seed, same build order: the harness's init IS the launch init
    assert ad.b3_stamp == T._bbf.stamp(ref, T.build_parser().parse_args(rigs.argvs[kind]))
    assert ad.b3_stamp["flags"] == {"--r9-box-quality": True, "--r9-box-dn": True, "--r9-box-class-sqrt": True, "--r9-box-inloss-nms": kind == "b3p"}
    assert ad.b3_stamp["dn"]["new_params"] == 4352                                       # Linear(6 -> 256) + Embedding(10 -> 256) on the tiny rig


def test_OFF_path_is_bit_identical_to_the_Y4_replay_and_to_train(rigs, tmp_path):
    G0 = _mutated(tmp_path, "gbo_y4_off", _y4_edits())
    m_new, ad_new = NL._harness_model(rigs.G, rigs.T, rigs.argvs["off"])
    m_old, ad_old = NL._harness_model(G0, rigs.T, rigs.argvs["off"])
    ref = rigs.ref["off"]
    assert ad_new.dvb_mismatches == ad_old.dvb_mismatches == []
    dn, do_, dt = _digests(m_new), _digests(m_old), _digests(ref)
    assert sorted(dn) == sorted(do_) == sorted(dt)
    assert [k for k in dn if dn[k] != do_[k]] == [] and [k for k in dn if dn[k] != dt[k]] == []
    for m in (m_new, m_old, ref):
        assert not hasattr(m, "_r9_box_b3") and not hasattr(m, "_r9_box_cls_weight")               # ABSENT (not False), as in train()
        assert getattr(m._perception.box_dec, "dn_embed", None) is None and m._perception.box_dec.b3_nms_tau is None
    assert ad_new.b3_stamp is None and rigs.G.b3_build_fields(ad_new.b3_stamp) == {}                # the record is untouched


def test_RED_without_the_attach_call_the_mismatches_return_by_name_and_each_lever_has_its_own_check(rigs, tmp_path):
    G0 = _mutated(tmp_path, "gbo_unfixed", _y4_edits())
    with pytest.raises(SystemExit) as e:                                                 # B3p: the same five, plus its own lever (the NMS tau) named
        NL._harness_model(G0, rigs.T, rigs.argvs["b3p"])
    lines = str(e.value).splitlines()
    assert lines[0] == "[gbo] the replayed build is NOT the declared launch model:" and lines[1:6] == MEASURED_MISMATCHES
    assert len(lines) == 8 and lines[6].startswith("  - r9_box_inloss_nms: declared True but BUILT False (read from model._r9_box_b3.inloss_nms)")
    assert lines[7].startswith("  - r9_box_inloss_nms: declared True but BUILT None (read from model._perception.box_dec.b3_nms_tau)")
    # attach called WITHOUT the DN flag: only the DN pair is named (the DN check does not hide behind the others)
    G1 = _mutated(tmp_path, "gbo_attach_no_dn",
                  [(ATTACH_LINE, "        tr._bbf.attach(m, argparse.Namespace(**{**vars(a), 'r9_box_dn': False}))\n")])
    with pytest.raises(SystemExit) as e1:
        NL._harness_model(G1, rigs.T, rigs.argvs["b3"])
    lines = str(e1.value).splitlines()[1:]
    assert lines == [MEASURED_MISMATCHES[1], MEASURED_MISMATCHES[2]]


def test_the_replay_goes_through_the_trainers_own_attach_at_its_position_not_a_reimplementation():
    src = GBO_PATH.read_text(encoding="utf-8")
    assert src.count(ATTACH_LINE) == 1
    tree = ast.parse(src)                                                                 # identifiers, not comments: the header comment names them
    used = ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            | {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names})
    for banned in ("enable_b3", "sqrt_temper", "BoxB3Config", "build_dn_pack", "set_dn", "dn_embed", "_r9_box_cls_weight"):   # the trainer's modules own these
        assert banned not in used, banned
    i_build = src.index("m._perception = _perc.build_perception_branch(m, _pcfg).to(dev)")
    i_attach = src.index(ATTACH_LINE)
    i_next = src.index("        if m._w_map > 0.0:\n            _pe, _ptable")
    assert i_build < i_attach < i_next                                                    # right after the branch, before the lift bank (as in train())
    tsrc = TRAINER_PATH.read_text(encoding="utf-8")
    assert tsrc.count("_bbf.attach(model, args)") == 1
    assert tsrc.index("model._perception = _perc.build_perception_branch(model, _pcfg).to(device)") < tsrc.index("_bbf.attach(model, args)") \
        < tsrc.index("_pwf.attach(model, args)")                                          # the trainer line the replay mirrors


# ---------------------------------------------------------------------------------------------------------------------- #
# 2. --identity-only is untouched                                                                                         #
# ---------------------------------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", ["C0", "C1", "M1", "M2a", "B1", "T1", "B3", "B3p"])
def test_identity_only_sha_is_unchanged_by_the_change(arm, tmp_path):
    """``--identity-only`` exits BEFORE any build: nothing here can move ``identity_sha256``. Measured against a COPY with the attach call removed."""
    G1 = NL._load("g_box_overfit_b3_id_t", GBO_PATH)
    G0 = _mutated(tmp_path, "gbo_id_y4", _y4_edits())
    blocks = []
    for tag, G in (("hold", G1), ("y4", G0)):
        out = tmp_path / f"{arm}.{tag}.json"
        toks = ["--launch-argv", str(RUNS / f"r9e1_{arm}.argv.json"), "--audit-dir", str(AUDIT), "--out", str(tmp_path / "rec.json"),
                "--device", "cuda", "--binding", "--commit", "ab" * 20, "--candidate", "x", "--arms", "main", "--identity-only", str(out)]
        assert G.main(toks) == 0
        blocks.append(json.loads(out.read_text(encoding="utf-8")))
    assert blocks[0] == blocks[1] and len(blocks[0]["identity_sha256"]) == 64


# ---------------------------------------------------------------------------------------------------------------------- #
# 3. (b) + (c): DN on in a training step, off in eval; the replay's DN generator is not B3's                              #
# ---------------------------------------------------------------------------------------------------------------------- #
class _DnSpy:
    """Counts the non-null DN packs the decoder is handed, by mode, and the DN outputs an EVAL forward produced."""

    def __init__(self, monkeypatch):
        from tanitad.models import box3d_head as BH
        self.n = {"set_train": 0, "set_eval": 0, "eval_dn_out": 0}
        o_set, o_fwd = BH.Box3DSlotDecoder.set_dn, BH.Box3DSlotDecoder.forward

        def set_dn(me, pack):
            if pack is not None:
                self.n["set_train" if me.training else "set_eval"] += 1
            return o_set(me, pack)

        def fwd(me, memory):
            out = o_fwd(me, memory)
            if not me.training and "_dn_out" in out:
                self.n["eval_dn_out"] += 1
            return out
        monkeypatch.setattr(BH.Box3DSlotDecoder, "set_dn", set_dn)
        monkeypatch.setattr(BH.Box3DSlotDecoder, "forward", fwd)


def test_b_DN_is_on_in_a_training_step_and_off_in_eval_and_the_batch_carries_its_inputs(rigs, monkeypatch):
    spy = _DnSpy(monkeypatch)
    ad = _adapter(rigs.G, rigs.T, rigs.argvs["b3"], rigs.ref["b3"])
    ad.setup("main", 0)
    m = ad.model
    batch = ad._batch([0, 1])
    for k in ("agent_box", "agent_yaw", "agent_cls", "agent_valid", "agent_vis_full", "agent_vis_px", "agent_vis_known"):
        assert k in batch, k                                                              # what `_bbf.prepare` reads (the VIS-1 keys: the arm is --slot-vis1)
    assert m.training and m._r9_box_b3.dn
    row = ad._loss_row(batch)                                                             # a TRAINING step's row through the trainer's own loss
    assert spy.n == {"set_train": 1, "set_eval": 0, "eval_dn_out": 0}
    assert torch.is_tensor(row["box3d_dn"]) and row["box3d_n_dn_pos"] == 12.0 and row["box3d_n_dn_neg"] == 24.0   # 2 windows x 3 groups x (2 pos | 4 neg)
    assert "_det_pack_box3d" not in row                                                   # packs are an EVAL-mode product
    row["box3d"].backward()
    g = lambda p: 0.0 if p.grad is None else float(p.grad.abs().sum())                  # noqa: E731
    assert g(m._perception.box_dec.dn_embed.weight) > 0.0 and g(m._perception.box_dec.dn_cls_embed.weight) > 0.0   # DN reaches its parameters
    packs, pres = ad.evaluate()                                                           # EVAL: packs from the trainer's row, DN off
    assert len(packs) == len(ad.items) and pres == pres and spy.n == {"set_train": 1, "set_eval": 0, "eval_dn_out": 0}
    assert m.training and m._r9_box_b3.dn is True                                         # train mode and the declared cfg are restored after evaluate()
    ad.teardown()


def test_b_the_trainers_own_DN_prepare_refuses_a_batch_without_its_inputs(rigs):
    ad = _adapter(rigs.G, rigs.T, rigs.argvs["b3"], rigs.ref["b3"])
    ad.setup("main", 0)
    batch = ad._batch([0, 1])
    del batch["agent_box"]
    with pytest.raises(SystemExit, match="no `agent_box`"):
        rigs.T._bbf.prepare(ad.model, batch, "cpu")                                       # REFUSE, do not skip
    ad.model.eval()
    rigs.T._bbf.prepare(ad.model, batch, "cpu")                                           # eval: a no-op, whatever the batch carries
    ad.teardown()


def test_c_the_replays_dn_generator_is_read_by_nothing_and_B3_draws_from_the_global_stream(rigs):
    n_hits = 0
    for p in list((STACK / "tanitad").rglob("*.py")) + list((STACK / "scripts").rglob("*.py")):
        n_hits += len(re.findall(r"\b_dn_gen\b", p.read_text(encoding="utf-8", errors="replace")))
    assert n_hits == 1                                                                    # the harness's own assignment, and no reader anywhere
    assert "generator=" not in (STACK / "tanitad" / "train" / "box_b3_flag.py").read_text(encoding="utf-8")    # prepare() passes none: B3's DN is the global stream
    ad = _adapter(rigs.G, rigs.T, rigs.argvs["b3"], rigs.ref["b3"])
    ad.setup("main", 0)
    m = ad.model
    batch = ad._batch([0, 1])
    before = m._dn_gen.get_state().clone()
    packs = []
    for seed in (5, 5, 6):
        torch.manual_seed(seed)
        rigs.T._bbf.prepare(m, batch, "cpu")
        packs.append(m._perception.box_dec._dn_pending["feat"].clone())
    assert torch.equal(packs[0], packs[1]) and not torch.equal(packs[0], packs[2])        # B3's DN follows torch.manual_seed (setup(arm, seed) seeds it)
    assert torch.equal(m._dn_gen.get_state(), before)                                     # the replay's generator was never advanced by B3
    ad.teardown()


FLAG_PATH = STACK / "tanitad" / "train" / "box_b3_flag.py"
GUARD_HEAD = '    if cfg is not None and cfg.dn and not getattr(model, "training", True):\n'
GUARD_TAIL = "        cfg = dataclasses.replace(cfg, dn=False)\n"


def _flag_y4(tmp_path: Path):
    """``box_b3_flag`` WITHOUT the eval guard (= tip Y4's ``box3d_row`` body; only the docstring sentence differs), loaded as its own module."""
    src = FLAG_PATH.read_text(encoding="utf-8")
    assert src.count(GUARD_HEAD) == 1 and src.count(GUARD_TAIL) == 1
    i = src.index(GUARD_HEAD)
    j = src.index(GUARD_TAIL) + len(GUARD_TAIL)
    assert j > i and "dataclasses.replace" in src[i:j] and src[i:j].count("\n") >= 4, src[i:j]       # the `if` line, its comment, the import, the replace
    p = tmp_path / "box_b3_flag_y4.py"
    p.write_text(src[:i] + src[j:], encoding="utf-8", newline="\n")
    return NL._load("box_b3_flag_y4_t", p)


def _eq(a, b, path="row"):
    """Bit equality of two loss rows (tensors, numbers, the eval packs' numpy arrays, nested lists / dicts)."""
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and sorted(map(str, a)) == sorted(map(str, b)), path
        for k in a:
            _eq(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for n, (x, y) in enumerate(zip(a, b)):
            _eq(x, y, f"{path}[{n}]")
    elif isinstance(a, float) and a != a:
        assert isinstance(b, float) and b != b, path
    else:
        assert a == b, path


def _run_row(ad, fn, monkeypatch, batch, snapshot, *, train: bool, backward: bool = False, seed: int = 123):
    """One loss row through the trainer's own ``compute_losses_*`` with ``box_b3_flag.box3d_row`` = ``fn``; the model state and the RNG are reset so two
    calls differ ONLY in ``fn``. -> (row, grads of the box decoder when ``backward``)."""
    monkeypatch.setattr(ad.tr._bbf, "box3d_row", fn)
    ad.model.load_state_dict(snapshot)
    ad.model.zero_grad(set_to_none=True)
    ad.model.train(train)
    torch.manual_seed(seed)
    with torch.set_grad_enabled(train):
        row = ad._loss_row(batch)
    grads = None
    if backward:
        row["box3d"].backward()
        grads = {n: p.grad.clone() for n, p in ad.model._perception.box_dec.named_parameters() if p.grad is not None}
    ad.model.train()
    return row, grads


@pytest.mark.parametrize("kind", ["b3", "b3p"])
def test_a_the_trainers_own_row_in_eval_returns_a_row_for_a_DN_arm(rigs, kind):
    """No harness override, nothing wrapped: ``compute_losses_perc_rig`` under ``model.eval()`` -> ``_bbf.box3d_row`` -> a finite row with the eval pack."""
    ad = _adapter(rigs.G, rigs.T, rigs.argvs[kind], rigs.ref[kind])
    ad.setup("main", 0)
    assert ad.model._r9_box_b3.dn is True
    ad.model.eval()
    with torch.no_grad():
        row = ad._loss_row(ad._batch([0, 1]))
    assert torch.isfinite(row["box3d"]) and torch.isfinite(torch.as_tensor(row["box3d_presence"]))
    assert len(row["_det_pack_box3d"]) == 2
    assert "box3d_dn" not in row and not any(k.startswith("box3d_n_dn") for k in row)       # an eval row carries no DN term
    assert ad.model._r9_box_b3.dn is True                                                    # the declared config is not touched
    ad.model.train()
    packs, pres = ad.evaluate()                                                              # and the harness scores through exactly that row
    assert len(packs) == len(ad.items) and pres == pres
    ad.teardown()


def test_a_in_eval_only_DN_is_dropped_from_the_config_the_row_reads_and_the_other_levers_stay(rigs, monkeypatch):
    from tanitad.models import box_b3 as BB3
    from tanitad.models import refcv6_perception_branch as PB
    seen = {"b3": [], "plain": []}
    o_b3, o_pb = BB3.box3d_loss_row_b3, PB.box3d_loss_row
    monkeypatch.setattr(BB3, "box3d_loss_row_b3", lambda slots, tgt, cfg, **kw: (seen["b3"].append((cfg, kw["cls_class_weight"])), o_b3(slots, tgt, cfg, **kw))[1])
    monkeypatch.setattr(PB, "box3d_loss_row", lambda *a, **kw: (seen["plain"].append(kw["cls_class_weight"]), o_pb(*a, **kw))[1])
    ad = _adapter(rigs.G, rigs.T, rigs.argvs["b3"], rigs.ref["b3"])
    ad.setup("main", 0)
    m = ad.model
    batch = ad._batch([0, 1])
    ad._loss_row(batch)                                                                      # TRAINING: the full declared config
    assert [c for c, _ in seen["b3"]] == [m._r9_box_b3] and seen["b3"][0][0].dn is True and seen["plain"] == []
    seen["b3"].clear()
    m.eval()
    with torch.no_grad():
        ad._loss_row(batch)                                                                  # EVAL: DN off, quality + class-sqrt unchanged
    (cfg_e, cw_e), = seen["b3"]
    assert (cfg_e.quality, cfg_e.dn, cfg_e.class_sqrt, cfg_e.inloss_nms) == (True, False, True, False) and seen["plain"] == []
    assert torch.equal(cw_e, m._r9_box_cls_weight)                                           # the sqrt vector still replaces the class weight
    ad.teardown()
    # a DN-ONLY arm (no loss-changing lever left once DN is off): the plain row, with the ORIGINAL class weight
    seen["b3"].clear()
    ad1 = _adapter(rigs.G, rigs.T, RG._rig_argv(rigs.d) + ["--r9-box-dn"], rigs.ref["b3"])
    ad1.setup("main", 0)
    batch1 = ad1._batch([0, 1])
    ad1._loss_row(batch1)
    assert len(seen["b3"]) == 1 and seen["b3"][0][0].dn is True and seen["plain"] == []
    seen["b3"].clear()
    ad1.model.eval()
    with torch.no_grad():
        ad1._loss_row(batch1)
    assert seen["b3"] == [] and len(seen["plain"]) == 1 and seen["plain"][0] is ad1.model._cls_class_weight
    ad1.teardown()


def test_a_a_model_without_a_training_attribute_takes_the_training_path_as_before(monkeypatch):
    """The guard reads ``getattr(model, "training", True)``: a stub model (no ``.training``) keeps the pre-fix behaviour, DN config untouched."""
    from tanitad.models import box_b3 as BB3
    from tanitad.models import refcv6_perception_branch as PB
    got = []
    monkeypatch.setattr(BB3, "box3d_loss_row_b3", lambda slots, tgt, cfg, **kw: (got.append(cfg), {"loss": 0.0})[1])
    stub = types.SimpleNamespace(_r9_box_b3=BB3.BoxB3Config(dn=True, quality=True))
    from tanitad.train import box_b3_flag as BF
    BF.box3d_row(stub, PB, {}, {}, cls_class_weight=None, visible_filter=True, presence_loss="focal", vis1=False, vis=None)
    assert got == [BB3.BoxB3Config(dn=True, quality=True)]
    stub.training = False
    BF.box3d_row(stub, PB, {}, {}, cls_class_weight=None, visible_filter=True, presence_loss="focal", vis1=False, vis=None)
    assert got[1] == BB3.BoxB3Config(dn=False, quality=True) and stub._r9_box_b3.dn is True


@pytest.mark.parametrize("kind", ["b3", "b3p", "off"])
def test_b_the_TRAINING_row_and_its_gradients_are_bit_identical_to_Y4_and_an_OFF_model_is_identical_in_eval_too(rigs, tmp_path, monkeypatch, kind):
    Y4 = _flag_y4(tmp_path)
    T = rigs.T
    new_fn = T._bbf.box3d_row                                                                # the fixed function, captured before any patch
    y4_body = (tmp_path / "box_b3_flag_y4.py").read_text(encoding="utf-8").split("def box3d_row")[1].split("def stamp")[0]
    assert new_fn is not Y4.box3d_row and "dataclasses" not in y4_body and 'getattr(model, "training"' not in y4_body
    ad = _adapter(rigs.G, T, rigs.argvs[kind], rigs.ref[kind])
    ad.setup("main", 0)
    batch = ad._batch([0, 1])
    snap = copy.deepcopy(ad.model.state_dict())
    r_new, g_new = _run_row(ad, new_fn, monkeypatch, batch, snap, train=True, backward=True)
    r_old, g_old = _run_row(ad, Y4.box3d_row, monkeypatch, batch, snap, train=True, backward=True)
    _eq(r_old, r_new)
    assert sorted(g_old) == sorted(g_new) and len(g_new) > 0
    for n in g_new:
        assert torch.equal(g_old[n], g_new[n]), n
    if kind != "off":
        assert "box3d_dn" in r_new and r_new["box3d_n_dn_pos"] == 12.0                       # DN is ON in the training row (the fix never touches it)
        assert any(n.startswith("dn_") and float(g_new[n].abs().sum()) > 0.0 for n in g_new)
    else:
        assert not hasattr(ad.model, "_r9_box_b3")                                           # cfg None: byte-identical behaviour in eval too
        e_new, _ = _run_row(ad, new_fn, monkeypatch, batch, snap, train=False)
        e_old, _ = _run_row(ad, Y4.box3d_row, monkeypatch, batch, snap, train=False)
        _eq(e_old, e_new)
        assert len(e_new["_det_pack_box3d"]) == 2
    ad.teardown()


@pytest.mark.parametrize("kind", ["b3", "b3p"])
def test_c_RED_without_the_eval_guard_the_eval_row_raises_again(rigs, tmp_path, monkeypatch, kind):
    """Remove the guard (= Y4): the trainer's own eval row of a DN arm raises (the MEASURED defect), through ``compute_losses_*`` and through the
    harness's ``evaluate()``; the TRAINING row still works (the guard is eval-only)."""
    Y4 = _flag_y4(tmp_path)
    ad = _adapter(rigs.G, rigs.T, rigs.argvs[kind], rigs.ref[kind])
    ad.setup("main", 0)
    monkeypatch.setattr(ad.tr._bbf, "box3d_row", Y4.box3d_row)
    batch = ad._batch([0, 1])
    assert torch.isfinite(ad._loss_row(batch)["box3d"])                                      # training: fine either way
    ad.model.eval()
    with torch.no_grad(), pytest.raises(ValueError, match="--r9-box-dn is on and the loss ran with no DN output"):
        ad._loss_row(batch)
    ad.model.train()
    with pytest.raises(ValueError, match="--r9-box-dn is on and the loss ran with no DN output"):
        ad.evaluate()
    assert ad.model.training                                                                 # evaluate() put the model back in train mode on the raise
    ad.teardown()


def test_the_guard_is_in_the_trainers_function_and_the_harness_has_no_eval_override():
    gsrc = GBO_PATH.read_text(encoding="utf-8")
    assert "dataclasses.replace(b3_cfg" not in gsrc and "b3_restore" not in gsrc and "B3_EVAL_NOTE" not in gsrc and "harness_eval" not in gsrc
    fsrc = FLAG_PATH.read_text(encoding="utf-8")
    assert fsrc.count(GUARD_HEAD) == 1 and fsrc.index(GUARD_HEAD) < fsrc.index("if cfg is None or not cfg.any_loss_change:")


# ---------------------------------------------------------------------------------------------------------------------- #
# 4. (a) the loss row of a loss-changing lever is box_b3.box3d_loss_row_b3: the one-frame capture must see it              #
# ---------------------------------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["b3", "b3p"])
def test_a_the_one_frame_capture_is_non_empty_under_B3_and_everything_is_restored(rigs, kind):
    from tanitad.models import box_b3 as BB3
    from tanitad.models import refcv6_perception_branch as PB
    o_b3, o_pb = BB3.box3d_loss_row_b3, PB.box3d_loss_row
    ad = _adapter(rigs.G, rigs.T, rigs.argvs[kind], rigs.ref[kind])
    r = rigs.G.run_one_frame(ad, "main", steps=4, every=2)
    assert [x["step"] for x in r["rows"]] == [2, 4]
    assert r["rows"][0]["keep"] is None                                                   # no previous assignment at the first capture ...
    assert r["rows"][1]["keep"] is not None and 0.0 <= r["rows"][1]["keep"] <= 1.0       # ... a CAPTURED assignment at the second (empty capture reads None)
    assert BB3.box3d_loss_row_b3 is o_b3 and PB.box3d_loss_row is o_pb                    # both wrappers are removed
    assert ad.items and len(ad.items) == 4                                                # the frame list is restored


def test_a_the_trainer_hands_the_box_row_to_b3_through_the_module_attribute_the_wrapper_patches():
    """The wrapper patches ``box_b3.box3d_loss_row_b3``; it only works while ``box_b3_flag.box3d_row`` looks it up at CALL time (a function-level import)."""
    src = (STACK / "tanitad" / "train" / "box_b3_flag.py").read_text(encoding="utf-8")
    assert "    from tanitad.models.box_b3 import box3d_loss_row_b3\n" in src and "\nfrom tanitad.models.box_b3 import box3d_loss_row_b3" not in src
    b3 = (STACK / "tanitad" / "models" / "box_b3.py").read_text(encoding="utf-8")
    assert "    from tanitad.models.agent_slots import match_slots\n" in b3                    # the match the capture wraps is also resolved at call time


def test_RED_a_capture_that_misses_the_B3_row_REFUSES_loudly_instead_of_scoring_an_empty_capture(rigs, tmp_path):
    wrap = "    PB.box3d_loss_row, _AS.match_slots, _B3.match_slots, _BB3.box3d_loss_row_b3 = row, ms, ms, row_b3\n"
    unwrapped = "    PB.box3d_loss_row, _AS.match_slots, _B3.match_slots = row, ms, ms\n"
    G0 = _mutated(tmp_path, "gbo_capture_pb_only", [(wrap, unwrapped)])                  # Y4's capture: only PB.box3d_loss_row is wrapped
    ad = _adapter(G0, rigs.T, rigs.argvs["b3"], rigs.ref["b3"])
    with pytest.raises(SystemExit) as e:
        G0.run_one_frame(ad, "main", steps=4, every=2)
    msg = str(e.value)
    assert msg.startswith("[gbo] one-frame main step 2: --r9-box-* is on") and "NO assignment was captured" in msg
    # ... and WITHOUT the refusal the same defect is silent: every `keep` is None, (ii) can never pass, and nothing says why
    G1 = _mutated(tmp_path, "gbo_capture_silent", [(wrap, unwrapped), ("            if b3_cfg is not None and not cap[\"calls\"]:\n", "            if False:\n")])
    ad1 = _adapter(G1, rigs.T, rigs.argvs["b3"], rigs.ref["b3"])
    r = G1.run_one_frame(ad1, "main", steps=4, every=2)
    assert all(x["keep"] is None for x in r["rows"]) and not G1.one_frame_verdict(r["rows"], 0.9, steps=4, every=2)["ii_pass"]
    from tanitad.models import box_b3 as BB3
    assert BB3.box3d_loss_row_b3.__name__ == "box3d_loss_row_b3"                          # nothing leaked out of either failed run


def test_a_a_toy_adapter_without_a_B3_model_keeps_the_old_silent_behaviour():
    """The refusal is B3's: a non-B3 adapter whose loss bypasses ``box3d_loss_row`` still reads ``keep = None`` (test_g_box_overfit pins it)."""
    import test_g_box_overfit as TG
    r = TG.G.run_one_frame(TG._ToyOneFrame(route_through_row=False), "main", steps=50, every=25)
    assert all(x["keep"] is None for x in r["rows"])


# ---------------------------------------------------------------------------------------------------------------------- #
# 5. the record                                                                                                           #
# ---------------------------------------------------------------------------------------------------------------------- #
def test_the_record_fields_are_untouched_off_and_say_what_was_built_for_B3(rigs):
    G = rigs.G
    assert G.b3_build_fields(None) == {} and G.b3_record_fields(["--w-box3d", "1.0"]) == {}
    st = rigs.T._bbf.stamp(rigs.ref["b3"], rigs.T.build_parser().parse_args(rigs.argvs["b3"]))
    f = G.b3_build_fields(st)
    assert f == {"built": st}                                                             # what was BUILT, read off the model; nothing else
    sq = {"flags": {"--r9-box-quality": False, "--r9-box-dn": False, "--r9-box-class-sqrt": True, "--r9-box-inloss-nms": False}, "dn": None}
    assert G.b3_build_fields(sq) == {"built": sq}
    json.dumps(f, default=str)                                                            # JSON-able (the record writer's own call)
    assert G.b3_record_fields(B3_FLAGS)["b3"]["flags"] == B3_FLAGS                       # the stamp the launch gate matches is unchanged


def test_the_header_comment_no_longer_claims_the_trainer_builds_the_decoder():
    src = GBO_PATH.read_text(encoding="utf-8")
    assert "already builds the B3 decoder and loss from the argv" not in src
    assert "WHO BUILDS WHAT" in src and "``_bbf.attach(model, args)``" in src
