"""refcv9 C5-INTEG (MM 207ebd): B4's two integration items.

1. ``r9_e1_dump.py`` STAGES the per-row rig camera on the per-frame-query seed module BEFORE every forward (``stage_perframe_queries``) -- and only for a model
   that carries ``box_pfq``: a pfq model dumps; an unstaged forward still RAISES (fail loud); a batch with no ``agent_ep`` REFUSES; a default (C0-style) dump is
   byte-for-byte what it is with the hook absent.
2. ``launch_gate.box_pfq_stamp_reasons`` -- a line-for-line mirror of ``box_aux_stamp_reasons`` for ``perframe_seeds_not_certified`` -- at both box judges; plus
   the G-LIVE rule that lists the perspective head's loss keys on the real B4 argv.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STACK / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import launch_gate as LG  # noqa: E402
import r9_e1_dump as D  # noqa: E402
import test_launch_gate_e1rig_e1rec2 as R2  # noqa: E402
import test_launch_gate_e1rig_e1rec3 as R3  # noqa: E402
import test_r9_c5_pfq as PF  # noqa: E402
import test_r9_e1_dump as TD  # noqa: E402

G = pytest.importorskip("g_box_overfit")
PFQ = "--r9-box-perframe-queries"
B4 = list(R3.B2) + ["--r9-box-layers", "6", "--r9-box-s8-horizon", PFQ]          # B2's persp aux + the three own-head levers (the template's shape)


# =========================================================================================================== 1. the dump ==
class _PfqDumpModel:
    """A model-shaped object for the dump's stub loader: the REAL perception branch with the REAL ``box_pfq`` (``test_r9_c5_pfq._model``) behind a
    ``forward`` the dump can capture. A forward that nobody staged raises inside ``PerFrameQueries.forward``."""

    def __init__(self, m, br):
        self.m, self._perception, self._rig_camera = m, br, m._rig_camera
        self.core = types.SimpleNamespace(_perc_rig=True)
        br.eval()

    def forward(self, frames, future=None):
        g = torch.Generator().manual_seed(1)
        b = int(frames.shape[0])
        s16 = torch.randn(b, PF.D16, *PF.IMG, generator=g)
        s8 = torch.randn(b, PF.D8, PF.H8, PF.W8, generator=g)
        return {"perception": self._perception(s16, fmap_s8=s8)}

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


def _install_pfq(monkeypatch, tmp_path, *, with_agent_ep=True, trainer_stages=False):
    """The dump's existing stub (``test_r9_e1_dump._install_stub``, a RIG) with the model swapped for a real pfq model, the dataset emitting ``agent_ep`` and the
    trainer stub standing in for ``compute_losses_v3`` WITHOUT staging (the real one stages too; here only the dump's own call can)."""
    st = TD._install_stub(monkeypatch, tmp_path, rig=True)
    L = sys.modules["tanitad.eval.refcv7_loader"]
    tr = L.trainer()
    m, br, _ = PF._model()
    m._rig_camera = PF.RigCameraBank(by_episode={1000 + k: PF._cam(0.02 * k) for k in range(4)})
    model = _PfqDumpModel(m, br)
    log = {"ep": [], "R": []}
    pfq = br.box_pfq
    orig_stage = pfq.stage

    def spy(R, t, frame):
        log["R"].append(R.detach().clone())
        orig_stage(R, t, frame)
    pfq.stage = spy
    orig_build = L.build_model

    def build_model(config, ckpt, device):
        _, cfg, args, mrec = orig_build(config, ckpt, device)
        return model, cfg, args, mrec
    L.build_model = build_model
    if with_agent_ep:
        orig_get = tr.V3Dataset.__getitem__

        def get(self, i):
            item = orig_get(self, i)
            item["agent_ep"] = torch.tensor(self.episodes[self.index[i][0]].episode_id, dtype=torch.long)
            return item
        monkeypatch.setattr(tr.V3Dataset, "__getitem__", get)
    pack = TD._pack(TD.GT, [0, 1, 0, 2, 0, 1])

    def compute_losses_v3(model_, eb, device, mode="diffusion", ablate_frames=False):
        log["ep"].append(int(eb["agent_ep"].reshape(-1)[0]) if "agent_ep" in eb else None)
        if trainer_stages:
            from tanitad.train import box_c5_flag as C5
            C5.stage(model_, eb, device)
        model_(tr.frames_to_device(eb["frames"], device))
        return {"loss": torch.zeros(()), "_det_pack_box3d": [pack], "_det_pack_agent": [pack]}
    tr.compute_losses_v3 = compute_losses_v3
    return st, model, log


def test_a_pfq_model_dumps_with_the_clips_own_camera_staged_before_every_forward(monkeypatch, tmp_path):
    st, model, log = _install_pfq(monkeypatch, tmp_path)
    out = tmp_path / "pfq"
    assert D.main(TD._argv(st, out, "--no-rasters")) == 0
    meta = json.loads((out / "dump.json").read_text(encoding="utf-8"))
    n = meta["n_windows"] + 2 * len(meta["future_frame_control"]["windows"])                    # 8 windows + 2 control windows x 2 forwards
    assert meta["future_frame_control"]["pass"] is True and n == 12 == len(log["ep"]) == len(log["R"])
    assert meta["perframe_queries"]["staged_forwards"] == n and meta["perframe_queries"]["n_seeds"] == 200 and meta["perframe_queries"]["n_global"] == 100
    bank = model._rig_camera
    for ep, R in zip(log["ep"], log["R"]):                                                       # the camera of THAT window's clip, per forward
        assert torch.allclose(R[0], bank.for_episodes(torch.tensor([ep]))[0].R_cam_to_rig.float(), atol=1e-6), ep
    assert len({ep for ep in log["ep"]}) > 1 and len({tuple(R[0].flatten().tolist()) for R in log["R"]}) > 1       # the cameras DIFFER across clips
    assert model._perception.box_pfq._staged is None                                             # consumed by the last forward (one-shot)
    assert len(pickle.loads((out / "perc_det.pkl").read_bytes())) == 8


def test_RED_an_unstaged_forward_still_raises_and_a_batch_without_agent_ep_refuses(monkeypatch, tmp_path):
    st, model, log = _install_pfq(monkeypatch, tmp_path)
    with monkeypatch.context() as mp:
        mp.setattr(D, "stage_perframe_queries", lambda *a, **k: False)                          # RED: the dump's staging hop deleted
        with pytest.raises(RuntimeError, match="no camera staged"):
            D.main(TD._argv(st, tmp_path / "unstaged", "--no-rasters"))
    assert not (tmp_path / "unstaged" / "dump.json").exists()                                    # nothing was banked from a forward that raised
    (tmp_path / "b").mkdir()
    st2, _, _ = _install_pfq(monkeypatch, tmp_path / "b", with_agent_ep=False)
    with pytest.raises(SystemExit, match="agent_ep"):                                            # the batch carries no camera key: stage REFUSES
        D.main(TD._argv(st2, tmp_path / "noep", "--no-rasters"))


def test_the_hook_is_one_call_at_the_one_forward_site_both_loops_share():
    src = inspect.getsource(D.run_dump)
    assert src.count("stage_perframe_queries(model, eb, device)") == 1 and src.count("tr.compute_losses_v3(") == 1       # both loops go through fwd_losses
    assert src.index("stage_perframe_queries(model, eb, device)") < src.index("tr.compute_losses_v3(")
    assert "box_c5_flag" in inspect.getsource(D.stage_perframe_queries)


def _tree_digest(out: Path) -> dict:
    d = {"packs": hashlib.sha256((out / "perc_det.pkl").read_bytes()).hexdigest()}
    for f in sorted((out / "rast").iterdir()):
        z = np.load(f)
        d[f.name] = hashlib.sha256(b"".join(z[k].tobytes() for k in sorted(z.files))).hexdigest()
    return d


@pytest.mark.parametrize("rig", [False, True])
def test_a_default_C0_style_dump_is_unchanged_by_the_hook_and_never_touches_box_c5_flag(monkeypatch, tmp_path, rig):
    from tanitad.train import box_c5_flag as C5
    monkeypatch.setattr(C5, "stage", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a default dump must not stage")))
    res = []
    for name, hook in (("on", D.stage_perframe_queries), ("off", lambda *a, **k: False)):
        (tmp_path / name).mkdir()
        torch.manual_seed(0)                                                                     # the stub's episode frames are drawn unseeded
        st = TD._install_stub(monkeypatch, tmp_path / name, rig)
        monkeypatch.setattr(D, "stage_perframe_queries", hook)
        out = tmp_path / name / "out"
        assert D.main(TD._argv(st, out)) == 0
        meta = json.loads((out / "dump.json").read_text(encoding="utf-8"))
        res.append((_tree_digest(out), set(meta)))
    assert res[0][0] == res[1][0] and len(res[0][0]) == 9                                       # packs + 8 rasters, byte-identical
    assert res[0][1] == res[1][1] and "perframe_queries" not in res[0][1]                       # the record has exactly the tip's keys
    assert D.stage_perframe_queries(types.SimpleNamespace(), {}, "cpu") is False


# ====================================================================================================== 2. the gate judge ==
def _own_closes(tmp_path, rec, argv):
    return R2._box_closes(tmp_path, rec, argv)


def test_the_pfq_stamp_names_are_ONE_fact_across_gate_and_harness():
    assert (LG.E1_PFQ_FLAG, LG.E1_PFQ_STAMP, LG.E1_PFQ_STAMP_WHY) == (G.PFQ_FLAG, G.PFQ_STAMP, G.PFQ_STAMP_WHY) == (PFQ, "perframe_seeds_not_certified",
                                                                                                                 "perframe_seeds_not_certified_why")
    assert PFQ in LG.E1_BOX_OWN_FLAGS and PFQ not in LG.E1_BOX_AUX_FLAGS and PFQ in G.B3_FLAGS
    rec = G.b3_record_fields(B4)                                                                 # the harness's OWN function writes the stamp the gate reads
    assert rec[G.PFQ_STAMP] == [PFQ] and rec[G.PFQ_STAMP_WHY].strip() and rec[G.AUX_STAMP] == ["--r9-box-persp-aux"]
    assert G.PFQ_STAMP not in G.b3_record_fields(R3.B3) and G.PFQ_STAMP not in G.b3_record_fields(R3.C0)
    item = next(it for it in LG.PROFILES["refcv9-e1rig"]["open_items"] if it["id"] == "E1-BOX-OWN-RECORD")
    assert "perframe_seeds_not_certified" in item["closes_when"] and "box_pfq_stamp_reasons" in item["closes_when"]        # the open-item row names it


def test_GREEN_the_harness_stamped_own_record_for_a_pfq_argv_closes_the_item(tmp_path):
    rec = R2._box_record(B4)
    assert rec["schema"] == G.B3_SCHEMA and PFQ in rec["b3"]["flags"] and rec[G.PFQ_STAMP] == [PFQ]
    assert _own_closes(tmp_path, rec, B4) == (True, [])
    reasons, det = LG.box_pfq_stamp_reasons("T", rec, B4)
    assert reasons == [] and det == {"argv_pfq_set": [PFQ], "record_stamp": [PFQ]}
    assert LG.box_pfq_stamp_reasons("T", R2._box_record(R3.B3), R3.B3) == ([], {"argv_pfq_set": [], "record_stamp": None})     # B3: unchanged


def test_RED_arms_each_pfq_stamp_defect_is_refused_by_name(tmp_path):
    def why(rec, argv=B4):
        ok, reasons = _own_closes(tmp_path, rec, argv)
        assert ok is False
        return " | ".join(reasons)
    base = R2._box_record(B4)
    no_stamp = {k: v for k, v in base.items() if k not in (G.PFQ_STAMP, G.PFQ_STAMP_WHY)}
    assert f"stamps no `{LG.E1_PFQ_STAMP}`" in why(no_stamp)                                    # a pfq argv whose record lacks the stamp
    assert "must EQUAL" in why(dict(base, **{G.PFQ_STAMP: []})) and "must EQUAL" in why(dict(base, **{G.PFQ_STAMP: [PFQ, PFQ]}))
    assert "must EQUAL" in why(dict(base, **{G.PFQ_STAMP: [PFQ, "--r9-box-dn"]}))
    assert "not a list of flags" in why(dict(base, **{G.PFQ_STAMP: PFQ}))
    assert f"no one-line `{LG.E1_PFQ_STAMP_WHY}`" in why(dict(base, **{G.PFQ_STAMP_WHY: "  "}))
    # an argv WITHOUT pfq whose record carries the stamp: the own judge (B3) AND the inherited /1 judge (C0)
    b3 = R2._box_record(R3.B3)
    assert "must EQUAL" in why(dict(b3, **{G.PFQ_STAMP: [PFQ], G.PFQ_STAMP_WHY: "x"}), R3.B3)
    rec_c0 = R3._box1(R3.C0, **{G.PFQ_STAMP: [PFQ], G.PFQ_STAMP_WHY: "x"})
    reasons, _ = LG.judge_box_overfit(R3.RIG, R3._write(tmp_path, rec_c0), R3.COMMIT, argv_sha=LG.argv_sha256(R3.C0), argv=R3.C0)
    assert any(LG.E1_PFQ_STAMP in r and "must EQUAL" in r for r in reasons), reasons
    assert not [r for r in LG.judge_box_overfit(R3.RIG, R3._write(tmp_path, R3._box1(R3.C0), "c0.json"), R3.COMMIT,
                                                argv_sha=LG.argv_sha256(R3.C0), argv=R3.C0)[0] if LG.E1_PFQ_STAMP in r]                # C0: unchanged
    # the inherited /1 judge on a pfq argv with no stamp refuses too (it is also OPEN on the own record: the item stays INCOMPLETE)
    rec_b4 = R3._box1(R3.B2)
    r4, _ = LG.judge_box_overfit(R3.RIG, R3._write(tmp_path, rec_b4, "b4.json"), R3.COMMIT, argv_sha=LG.argv_sha256(B4), argv=B4)
    assert any(f"stamps no `{LG.E1_PFQ_STAMP}`" in r for r in r4), r4
    assert [x.split(":")[0] for x in LG.open_item_reasons(R3.RIG, B4)] == ["OPEN ITEM E1-BOX-OWN-RECORD"]


@pytest.mark.parametrize("fn", ["judge_box_own_record", "judge_box_overfit"])
def test_RED_deleting_the_judge_line_lets_a_stampless_pfq_record_through(monkeypatch, tmp_path, fn):
    """The mutation: the single call is deleted -> the defect this suite refuses is accepted. Proves each refusal above is THAT line's."""
    src = inspect.getsource(getattr(LG, fn))
    line = "r_pfq, d_pfq = box_pfq_stamp_reasons(tag, rec, argv)"
    assert src.count(line) == 1
    ns = dict(LG.__dict__)
    exec(compile(src.replace(line, "r_pfq, d_pfq = [], {'argv_pfq_set': [], 'record_stamp': None}"), "<mutant>", "exec"), ns)
    monkeypatch.setattr(LG, fn, ns[fn])
    if fn == "judge_box_own_record":
        rec = {k: v for k, v in R2._box_record(B4).items() if k not in (G.PFQ_STAMP, G.PFQ_STAMP_WHY)}
        assert _own_closes(tmp_path, rec, B4) == (True, [])                                       # the mutant accepts it ...
    else:
        rec = R3._box1(R3.B2)
        r, _ = LG.judge_box_overfit(R3.RIG, R3._write(tmp_path, rec), R3.COMMIT, argv_sha=LG.argv_sha256(B4), argv=B4)
        assert not [x for x in r if LG.E1_PFQ_STAMP in x]
    monkeypatch.undo()
    assert LG.box_pfq_stamp_reasons("T", rec, B4)[0] and "box_pfq_stamp_reasons(tag, rec, argv)" in inspect.getsource(getattr(LG, fn))       # ... the real gate refuses


# ========================================================================================================= 3. G-LIVE (B4) ==
PERSP_KEYS = ["box_persp", "box_persp_heat", "box_persp_depth", "box_persp_size", "box_persp_yaw", "box_persp_ctr"]


def test_G_LIVE_on_the_B4_argv_lists_the_persp_heads_loss_keys_and_adds_no_term_for_pfq():
    T, args = R3._args(B4)
    terms, problems = LG.declared_terms(T, args)
    assert problems == [] and getattr(args, "r9_box_perframe_queries") is True
    mine = [t for t in terms if t["id"] == "w_box_persp"]
    assert len(mine) == 1 and mine[0]["keys"] == PERSP_KEYS and mine[0]["min_frac"] == 1.0         # the existing rule covers the real B4 argv
    _, b2 = R3._args(R3.B2)
    assert [t["id"] for t in terms] == [t["id"] for t in LG.declared_terms(T, b2)[0]]              # pfq / layers / s8 add no loss key of their own
    flag_src = (STACK / "tanitad" / "train" / "persp3d_aux_flag.py").read_text(encoding="utf-8")
    assert all(f'"{k}"' in flag_src or f"'{k}'" in flag_src or k.replace("box_persp_", "") in flag_src for k in PERSP_KEYS)
    # a healthy smoke passes; RED: a step that drops the persp loss key, a non-finite persp loss, and a dead new group each fail by name
    keys = sorted({k for t in terms for k in t["keys"]})
    steps = [R3._step(keys, 9.0 - 0.01 * i) for i in range(100)]
    assert R3._live(terms, steps, args)[0] == []
    drop = [R3._step([k for k in keys if k != "box_persp"], 9.0 - 0.01 * i) for i in range(100)]
    assert any("`box_persp`" in r and "ABSENT" in r for r in R3._live(terms, drop, args)[0])
    bad = [R3._step(keys, 9.0 - 0.01 * i, nonfinite=["box_persp_depth"]) for i in range(100)]
    assert any("box_persp_depth" in r for r in R3._live(terms, bad, args)[0])
    row = {"step": 100, **{k: round(steps[-1]["vals"][k], 5) for k in keys}}
    dead = {"n_trainable_params": 5, "nonfinite_params": [], "dead_groups": [{"group": "_perception.box_pfq"}]}
    r, _ = LG.judge_live(terms, [], steps, expect_steps=100, summary={"done": True, "step": 100}, train_rows=[row], grads=dead, prior=None, exit_msg=None,
                         absent_terms=LG.rig_absent_terms(args))
    assert any("box_pfq" in x and "ZERO gradient" in x for x in r), r                               # the anchors' group at zero gradient fails G-LIVE
