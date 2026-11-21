"""`--resume` x `--ema-targets` for `refa_v1_train.py`
(Architecture & Inference FlyWheel, register row D-REFAV1-EMA-RESUME,
2026-09-03).

MEASURED, E-ARCH-TSC-2 R7 (`TanitAD Research Lab/Architecture & Inference/
Research/2026-09-02-refav1-ema-inflation/raw/R7_resume.log`): `--resume
--ema-targets` on a checkpoint written WITHOUT `--ema-targets` died in 2.8 s on
`RuntimeError: … Missing key(s) in state_dict: "ema.tac_queries",
"ema.adapter.pos", … "ema.str_read.1.bias"` — the 20 `ema.*` keys, strict
load, no teacher-from-student init. A run therefore could not be switched to
EMA targets at a checkpoint (C-REFAV1-TAC-INFLATION option (b) was blocked by
this; the clean-epoch restart only routed around it by starting fresh).

What is pinned, and why:

(a) OFF checkpoint -> resume ON: succeeds; at load every teacher tensor is
    `torch.equal` to the live student's (a COPY, not a lerp) and equal, name
    by name, to the CHECKPOINTED student; the one info line is printed
    verbatim; the run continues one step with a finite `tgt_std_tac` and a
    non-None `ema_decay`. The optimizer state loads for the student EXACTLY
    as saved, and the teacher is frozen and inside no param group.
(b) ON checkpoint -> resume ON: strict; the teacher is the SAVED teacher, not
    the student (the two already differ after one `ema_update`).
(c) A missing NON-EMA key refuses with the key named — from an OFF and from
    an ON checkpoint — and a PARTIAL teacher (one `ema.*` key removed) is
    refused, never silently re-initialised.
(d) An extra unexpected key refuses with the key named, from both kinds of
    checkpoint, and an unexpected `ema.*` key too.
(e) The OFF path (`--resume` without `--ema-targets` from an OFF checkpoint)
    is unchanged: no teacher, no EMA line, the step-2 checkpoint has the same
    key set as the step-1 one. The step-1 NUMBERS are pinned elsewhere
    (`test_refa_v1_precision.py`, `test_refa_v1_speed_channel.py`).
(f) THE PINNED DECISION: an ON checkpoint under a launch with `--ema-targets`
    OFF is REFUSED (SystemExit naming the flag), not silently stripped —
    dropping the teacher would continue the run as a different experiment
    under a launch line that says nothing about it (the stale-manifest
    relaunch class), and nothing is trained or overwritten by the refusal.

Rig: the trainer's own `--smoke` path on CPU (`--bs 2`, 0.9 s per step on the
dev box) — the rig of `test_refa_v1_precision.py`. The model and the optimizer
are read AT LOAD by wrapping the trainer's own resume helpers
(`load_resume_state`, `verify_resume_optimizer`): after the resumed step the
teacher has moved again, so an end-of-run read could not tell (a) from (b).
"""
import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from tanitad.refs.refa_v1 import RefAV1, RefAV1Config

COMMON = ["--smoke", "--bs", "2", "--log-every", "1", "--seed", "0",
          "--device", "cpu", "--save-every", "1"]

#: teacher key -> student key, stated INDEPENDENTLY of `_EmaTargetPath.pairs`
#: (the implementation derives the pairing from the model; the test states
#: the contract in its own words so the two can disagree).
_TEACHER_PREFIX = {"ema.adapter.": "adapter.", "ema.tac_pool.": "tac_pool.",
                   "ema.str_read.": "strategic.read.",
                   "ema.tac_queries": "tac_queries"}
N_EMA_KEYS = 20          # the 20 keys R7 named; same count in the smoke config


def _student_key(ema_key: str) -> str:
    for pre, stu in _TEACHER_PREFIX.items():
        if ema_key.startswith(pre):
            return stu + ema_key[len(pre):]
    raise KeyError(ema_key)


# ------------------------------------------------------------------ helpers --
def _trainer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "refa_v1_train.py"
    spec = importlib.util.spec_from_file_location(
        "refa_v1_train_ema_resume_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _argv(out: Path, *extra: str, steps: int) -> list:
    return COMMON + ["--steps", str(steps), "--out", str(out)] + list(extra)


def _rows(out: Path) -> list:
    return [json.loads(l) for l in
            (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _run(tr, out: Path, *extra: str, steps: int) -> list:
    """One trainer run; returns EVERY row of the (append-mode) log."""
    assert tr.main(_argv(out, *extra, steps=steps)) == 0
    return _rows(out)


def _ckpt(out: Path) -> dict:
    return torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)


def _save(out: Path, ck: dict) -> None:
    torch.save(ck, out / "ckpt.pt")


def _capture(tr, monkeypatch) -> dict:
    """Wrap the trainer's resume helpers to read model + optimizer AT LOAD."""
    rec = {}
    load, verify = tr.load_resume_state, tr.verify_resume_optimizer

    def _load(model, state, step):
        rec["record"] = load(model, state, step)
        rec["model"] = model
        rec["state_at_load"] = {k: v.detach().clone()
                                for k, v in model.state_dict().items()}
        if model.ema is not None:
            rec["teacher_equals_student"] = all(
                torch.equal(p_e, p_s) for p_e, p_s in model.ema.pairs(model))
        return rec["record"]

    def _verify(model, opt):
        verify(model, opt)
        rec["opt"] = opt
        rec["opt_state_at_load"] = copy.deepcopy(opt.state_dict())

    monkeypatch.setattr(tr, "load_resume_state", _load)
    monkeypatch.setattr(tr, "verify_resume_optimizer", _verify)
    return rec


def _tensors_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def _opt_state_equal(a: dict, b: dict) -> bool:
    """`optimizer.state_dict()["state"]` equality, tensor by tensor."""
    if set(a) != set(b):
        return False
    for idx in a:
        sa, sb = a[idx], b[idx]
        if set(sa) != set(sb):
            return False
        for k in sa:
            va, vb = sa[k], sb[k]
            if torch.is_tensor(va):
                if not (torch.is_tensor(vb) and torch.equal(va, vb)):
                    return False
            elif va != vb:
                return False
    return True


# ------------------------------------------------ (a) OFF ckpt -> resume ON --
def test_a_OFF_checkpoint_resumes_under_ema_targets_with_the_teacher_copied_from_the_student(
        tmp_path, monkeypatch, capsys):
    tr = _trainer()
    out = tmp_path / "run"
    _run(tr, out, steps=1)                                # the OFF checkpoint
    ck = _ckpt(out)
    assert ck["step"] == 1
    assert not any(k.startswith("ema.") for k in ck["model"])
    capsys.readouterr()
    rec = _capture(tr, monkeypatch)
    rows = _run(tr, out, "--ema-targets", "--resume", steps=2)
    text = capsys.readouterr().out
    # the ONE info line, verbatim
    assert ("[refav1] resume: EMA targets initialised from the student "
            "(checkpoint had no ema.* keys; step 1)") in text
    assert text.count("[refav1] resume:") == 1
    assert "resumed from step 1" in text
    assert rec["record"] == {"ema_init": "student", "n_ema_keys": N_EMA_KEYS,
                             "step": 1}
    # at load: teacher == live student on every pair (a copy, not a lerp) ...
    assert rec["teacher_equals_student"] is True
    m = rec["model"]
    assert m.ema is not None
    sd = rec["state_at_load"]
    ema_keys = sorted(k for k in sd if k.startswith("ema."))
    assert len(ema_keys) == N_EMA_KEYS
    # ... and == the CHECKPOINTED student, name by name (independent mapping)
    for k in ema_keys:
        assert torch.equal(sd[k], ck["model"][_student_key(k)]), k
    # the student itself loaded strictly: every non-EMA tensor is the ckpt's
    for k, v in ck["model"].items():
        assert torch.equal(sd[k], v), k
    assert set(sd) == set(ck["model"]) | set(ema_keys)
    # the run continued ONE step under the teacher
    assert [r["step"] for r in rows] == [1, 2]
    row = rows[-1]
    assert math.isfinite(row["loss"])
    assert math.isfinite(row["tgt_std_tac"]) and row["tgt_std_tac"] > 0.0
    assert math.isfinite(row["tgt_std_str"])
    assert row["ema_decay"] is not None and 0.0 < row["ema_decay"] < 1.0
    # the step-2 checkpoint now carries the teacher
    ck2 = _ckpt(out)
    assert ck2["step"] == 2
    assert sorted(k for k in ck2["model"] if k.startswith("ema.")) == ema_keys


def test_a2_the_optimizer_loads_for_the_student_and_the_teacher_stays_outside_it(
        tmp_path, monkeypatch):
    tr = _trainer()
    out = tmp_path / "run"
    _run(tr, out, steps=1)
    ck = _ckpt(out)
    rec = _capture(tr, monkeypatch)
    _run(tr, out, "--ema-targets", "--resume", steps=2)
    m, opt = rec["model"], rec["opt"]                   # verify DID run
    ema_ids = {id(p) for p in m.ema.parameters()}
    assert ema_ids
    assert all(p.requires_grad is False for p in m.ema.parameters())
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    assert not (in_opt & ema_ids)
    # the same two groups, same sizes, as the OFF run that wrote the ckpt
    assert ([len(g["params"]) for g in opt.param_groups]
            == [len(g["params"]) for g in ck["opt"]["param_groups"]])
    # and the moments loaded EXACTLY as saved (read at load, before step 2)
    assert ck["opt"]["state"], "the OFF step left no optimizer state to load"
    assert _opt_state_equal(rec["opt_state_at_load"]["state"],
                            ck["opt"]["state"])
    assert (rec["opt_state_at_load"]["param_groups"]
            == ck["opt"]["param_groups"])


# ------------------------------------------------- (b) ON ckpt -> resume ON --
def test_b_ON_checkpoint_resumes_strictly_with_the_SAVED_teacher_not_the_student(
        tmp_path, monkeypatch, capsys):
    tr = _trainer()
    out = tmp_path / "run"
    _run(tr, out, "--ema-targets", steps=1)               # the ON checkpoint
    ck = _ckpt(out)
    ema_keys = sorted(k for k in ck["model"] if k.startswith("ema."))
    assert len(ema_keys) == N_EMA_KEYS
    # after ONE ema_update the saved teacher already differs from the saved
    # student — so this test CAN tell "restored" from "re-copied"
    assert any(not torch.equal(ck["model"][k], ck["model"][_student_key(k)])
               for k in ema_keys)
    capsys.readouterr()
    rec = _capture(tr, monkeypatch)
    rows = _run(tr, out, "--ema-targets", "--resume", steps=2)
    text = capsys.readouterr().out
    assert "initialised from the student" not in text
    assert ("[refav1] resume: EMA teacher restored from the checkpoint "
            f"({N_EMA_KEYS} ema.* keys; step 1)") in text
    assert rec["record"] == {"ema_init": "checkpoint",
                             "n_ema_keys": N_EMA_KEYS, "step": 1}
    assert rec["teacher_equals_student"] is False
    sd = rec["state_at_load"]
    for k in ema_keys:
        assert torch.equal(sd[k], ck["model"][k]), k      # the SAVED teacher
    for k, v in ck["model"].items():
        assert torch.equal(sd[k], v), k                   # everything, strictly
    assert set(sd) == set(ck["model"])
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[-1]["ema_decay"] is not None
    assert math.isfinite(rows[-1]["tgt_std_tac"])


# ------------------------- (c) a missing NON-EMA key / a PARTIAL teacher --
@pytest.mark.parametrize("ckpt_flags", [(), ("--ema-targets",)],
                         ids=["OFF-ckpt", "ON-ckpt"])
def test_c_a_missing_non_ema_key_is_refused_with_the_key_named(tmp_path,
                                                                ckpt_flags):
    tr = _trainer()
    out = tmp_path / "run"
    _run(tr, out, *ckpt_flags, steps=1)
    ck = _ckpt(out)
    del ck["model"]["adapter.out.bias"]
    _save(out, ck)
    with pytest.raises(RuntimeError, match=r"adapter\.out\.bias"):
        tr.main(_argv(out, "--ema-targets", "--resume", steps=2))
    assert [r["step"] for r in _rows(out)] == [1]        # nothing trained


def test_c2_a_partial_teacher_is_refused_not_silently_reinitialised(tmp_path,
                                                                     capsys):
    tr = _trainer()
    out = tmp_path / "run"
    _run(tr, out, "--ema-targets", steps=1)
    ck = _ckpt(out)
    del ck["model"]["ema.tac_queries"]                    # 19 of 20 remain
    _save(out, ck)
    capsys.readouterr()
    with pytest.raises(RuntimeError, match=r"ema\.tac_queries"):
        tr.main(_argv(out, "--ema-targets", "--resume", steps=2))
    assert "[refav1] resume:" not in capsys.readouterr().out


# ------------------------------------------ (d) an extra unexpected key --
@pytest.mark.parametrize("ckpt_flags", [(), ("--ema-targets",)],
                         ids=["OFF-ckpt", "ON-ckpt"])
def test_d_an_unexpected_key_is_refused_with_the_key_named(tmp_path,
                                                            ckpt_flags):
    tr = _trainer()
    out = tmp_path / "run"
    _run(tr, out, *ckpt_flags, steps=1)
    ck = _ckpt(out)
    ck["model"]["adapter.bogus"] = torch.zeros(3)
    _save(out, ck)
    with pytest.raises(RuntimeError, match=r"adapter\.bogus"):
        tr.main(_argv(out, "--ema-targets", "--resume", steps=2))
    assert [r["step"] for r in _rows(out)] == [1]


def test_d2_an_unexpected_ema_key_is_refused_too(tmp_path):
    tr = _trainer()
    out = tmp_path / "run"
    _run(tr, out, "--ema-targets", steps=1)
    ck = _ckpt(out)
    ck["model"]["ema.bogus"] = torch.zeros(3)
    _save(out, ck)
    with pytest.raises(RuntimeError, match=r"ema\.bogus"):
        tr.main(_argv(out, "--ema-targets", "--resume", steps=2))


# ----------------------------------------------- (e) the OFF path unchanged --
def test_e_OFF_resume_from_an_OFF_checkpoint_is_unchanged_no_teacher_no_line(
        tmp_path, monkeypatch, capsys):
    tr = _trainer()
    out = tmp_path / "run"
    _run(tr, out, steps=1)
    ck = _ckpt(out)
    capsys.readouterr()
    rec = _capture(tr, monkeypatch)
    rows = _run(tr, out, "--resume", steps=2)
    text = capsys.readouterr().out
    assert "[refav1] resume:" not in text
    assert "resumed from step 1" in text
    assert rec["record"] == {"ema_init": None, "n_ema_keys": 0, "step": 1}
    assert rec["model"].ema is None
    assert "teacher_equals_student" not in rec
    assert _tensors_equal(rec["state_at_load"], ck["model"])
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[-1]["ema_decay"] is None
    assert math.isfinite(rows[-1]["loss"])
    ck2 = _ckpt(out)
    assert set(ck2["model"]) == set(ck["model"])        # no ema.* key appeared
    # the optimizer loaded as before (verify is a no-op without a teacher)
    assert _opt_state_equal(rec["opt_state_at_load"]["state"],
                            ck["opt"]["state"])


# ---------------------------------------------- (f) THE PINNED DECISION --
def test_f_ON_checkpoint_under_ema_targets_OFF_is_REFUSED_not_silently_stripped(
        tmp_path, capsys):
    tr = _trainer()
    out = tmp_path / "run"
    _run(tr, out, "--ema-targets", steps=1)
    before = _ckpt(out)
    capsys.readouterr()
    with pytest.raises(SystemExit, match=r"--ema-targets"):
        tr.main(_argv(out, "--resume", steps=2))
    # nothing trained, nothing overwritten: the checkpoint is the ON one
    after = _ckpt(out)
    assert after["step"] == 1
    assert _tensors_equal(after["model"], before["model"])
    assert [r["step"] for r in _rows(out)] == [1]
    assert "[refav1] resume:" not in capsys.readouterr().out


# ------------------------------------ the helpers alone, tiny no-hierarchy --
def _cfg(**kw) -> RefAV1Config:
    """`test_refa_v1_ema_targets._cfg` — the small no-hierarchy config."""
    base = dict(d_enc=16, d_state=16, n_tokens=8,
                op_dt=0.2, op_steps=30, op_layers=1, op_heads=2, op_window=2,
                tac_dt=0.6, tac_steps=10, tac_queries=4, tac_layers=1,
                str_dt=3.0, str_steps=2, str_dim=8, str_layers=1)
    base.update(kw)
    return RefAV1Config(**base)


def test_helper_copies_the_teacher_from_the_student_and_loads_the_rest_strictly(
        capsys):
    tr = _trainer()
    torch.manual_seed(1)
    src = RefAV1(_cfg())                 # an OFF model = the checkpointed student
    torch.manual_seed(2)
    dst = RefAV1(_cfg(ema_targets=True))  # a fresh ON model, a different init
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    given = src.state_dict()
    rec = tr.load_resume_state(dst, given, step=7)
    assert rec == {"ema_init": "student", "n_ema_keys": N_EMA_KEYS, "step": 7}
    sd = dst.state_dict()
    for k, v in given.items():
        assert torch.equal(sd[k], v), k
    ema_keys = [k for k in sd if k.startswith("ema.")]
    assert len(ema_keys) == N_EMA_KEYS
    for k in ema_keys:
        assert torch.equal(sd[k], given[_student_key(k)]), k
    assert any(not torch.equal(sd[k], before[k]) for k in ema_keys)  # it moved
    assert all(torch.equal(p_e, p_s) for p_e, p_s in dst.ema.pairs(dst))
    assert all(p.requires_grad is False for p in dst.ema.parameters())
    assert "(checkpoint had no ema.* keys; step 7)" in capsys.readouterr().out
    assert not any(k.startswith("ema.") for k in given)   # caller's dict intact


def test_helper_refuses_a_teacher_the_launch_did_not_ask_for():
    tr = _trainer()
    src = RefAV1(_cfg(ema_targets=True))
    dst = RefAV1(_cfg())
    with pytest.raises(SystemExit, match=r"--ema-targets"):
        tr.load_resume_state(dst, src.state_dict(), step=3)


def test_verify_resume_optimizer_catches_a_teacher_in_the_optimizer_or_unfrozen():
    tr = _trainer()
    m = RefAV1(_cfg(ema_targets=True))
    good = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad])
    tr.verify_resume_optimizer(m, good)                  # silent when right
    bad = torch.optim.AdamW(list(m.parameters()))        # the teacher inside
    with pytest.raises(RuntimeError, match=r"inside the optimizer"):
        tr.verify_resume_optimizer(m, bad)
    m.ema.tac_queries.requires_grad_(True)
    with pytest.raises(RuntimeError, match=r"requires_grad=True"):
        tr.verify_resume_optimizer(m, good)
    tr.verify_resume_optimizer(RefAV1(_cfg()), good)     # no teacher: no-op
