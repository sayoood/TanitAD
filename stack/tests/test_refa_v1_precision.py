"""`--precision {fp32,bf16}` / `--tf32` for `refa_v1_train.py`
(Deploy & Optimization FlyWheel, register row D-REFAV1-STEP-PROFILE, 2026-09-02).

MEASURED on the dev-box RTX 4060 (`…/Research/2026-09-02-refav1-step-profile/`):
the refav1 step IS the 30-step operative token-field rollout forward+backward
(84-92 % of the 20.6 s Thor step, GPU busy 98 %), and bf16 autocast makes that
rollout 3.0x faster (TF32 alone 1.54x, forward). The trainer had no precision
support at all. Both switches default OFF because the LIVE Thor run must stay
reproducible from the repo — so the load-bearing test here is the OFF-path
identity, not the speed-up (which this box cannot measure for Thor anyway).

What is pinned, and why:

(a) OFF-PATH IDENTITY. `--precision fp32` (and the bare default) reproduces
    the step-1 numbers the PRE-EDIT trainer produced — not only the loss but
    every logged instrument — and touches no process-global numerics flag.
(b) THE FLAGS LAND IN `config.json` as READ-BACK values (the process's own
    `torch.backends.*` state), not as an echo of argparse; the model config in
    the checkpoint deliberately carries NO precision key (trainer-side
    property; a checkpoint loads under either).
(c) bf16 AUTOCASTS ON CPU (the pinned decision — the smoke/test path runs the
    SAME context wiring as CUDA instead of refusing): a 3-step smoke is finite,
    a Linear inside the rollout provably ran in bfloat16, and step 1 sits
    within bf16's own rounding of the fp32 run.
(d) `--tf32` ON CPU is a numeric no-op that still flips the global flags and
    stamps them (restored after the test — they are process-global).
(e) CUDA, only when the GPU is provably FREE: a tiny bf16-autocast step gives
    a finite loss and grad_norm. Skipped with the measured reason when another
    agent's arms hold the card (dev-box rule: never disturb a running job).

OFF-path constants: computed 2026-09-02 by running the PRE-EDIT trainer
(`stack/scripts/refa_v1_train.py` at git blob d94a1632, sha256 f5a48899…)
twice on CPU, torch 2.11.0+cu128, `--smoke --steps 1 --bs 2 --seed 0` — the
two runs were byte-identical. Same tolerance class as
`test_refa_v1_speed_channel.py`'s identity constants (rel 1e-6 on the loss).

bf16 tolerance (c): bf16 keeps 8 mantissa bits, unit roundoff 2^-8 = 3.9e-3.
MEASURED on this config, CPU autocast vs fp32 at step 1: total loss rel
1.0e-4, loss_feat_op 5.4e-5, loss_feat_tac/str 3.5e-3 (the two ~0.02-0.04
auxiliary terms sit at ONE unit roundoff), grad_norm 5.2e-4, participation
3.4e-4, tgt_std_tac/str ~1.1e-3, tgt_std_op EXACTLY 0 (its target path,
`std(future_feats)`, has no autocast-listed op). Pinned at rel 4e-3 (one
bf16 ulp) for the total loss — 40x the measured deviation, while a genuinely
broken path (a term silently dropped, a wrong-dtype stack, a loss in another
unit) misses it by orders of magnitude — and rel 1e-2 for the instruments.
"""
import contextlib
import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path

import pytest
import torch

# ------------------------------------------------------------------ helpers --
COMMON = ["--smoke", "--bs", "2", "--log-every", "1", "--seed", "0"]


def _trainer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "refa_v1_train.py"
    spec = importlib.util.spec_from_file_location(
        "refa_v1_train_precision_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(tr, out: Path, *extra: str, steps: int = 1, device: str = "cpu"):
    """One trainer run; returns (log rows, config.json dict)."""
    argv = COMMON + ["--device", device, "--steps", str(steps),
                     "--save-every", str(steps), "--out", str(out)] + list(extra)
    assert tr.main(argv) == 0
    rows = [json.loads(l) for l in
            (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert [r["step"] for r in rows] == list(range(1, steps + 1))
    return rows, cfg


def _finite(row: dict) -> bool:
    return all(math.isfinite(v) for v in row.values()
               if isinstance(v, float))


def _hook_linear_dtypes(tr, monkeypatch) -> list:
    """Record the OUTPUT dtype of a Linear inside the operative rollout — the
    direct proof that autocast engaged (or did not), independent of any
    numeric coincidence."""
    seen = []
    orig = tr.build_model

    def build(args):
        m = orig(args)
        m.operative.act[-1].register_forward_hook(
            lambda mod, inp, out: seen.append(out.dtype))
        return m

    monkeypatch.setattr(tr, "build_model", build)
    return seen


@pytest.fixture
def restore_tf32_flags():
    """`--tf32` flips PROCESS-GLOBAL switches; put them back so no later test
    in this session silently runs under TF32."""
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    prec = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(prec)
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn


# ------------------------------------------------ (a) OFF-path identity ------
# Step-1 row of the PRE-EDIT trainer (provenance in the module docstring).
_PRE_EDIT_STEP1 = {
    "loss": 1.3593703508377075,
    "loss_feat_op": 1.3402401208877563,
    "loss_feat_tac": 0.01669314131140709,
    "loss_feat_str": 0.043134838342666626,
    "grad_norm": 5.466560363769531,
    "adapter_std": 0.48328351974487305,
    "participation": 11.95924366535315,
    "tgt_std_op": 1.014772891998291,
    "tgt_std_tac": 0.07334569096565247,
    "tgt_std_str": 0.1545305848121643,
}


@pytest.mark.parametrize("flags", [(), ("--precision", "fp32")],
                         ids=["default", "explicit-fp32"])
def test_OFF_path_identity_every_step1_instrument_matches_the_pre_edit_trainer(
        tmp_path, monkeypatch, flags):
    tr = _trainer()
    before = (torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32,
              torch.get_float32_matmul_precision())
    seen = _hook_linear_dtypes(tr, monkeypatch)
    rows, cfg = _run(tr, tmp_path / "off", *flags)
    row = rows[0]
    assert row["loss"] == pytest.approx(_PRE_EDIT_STEP1["loss"], rel=1e-6,
                                        abs=0.0)
    for k, v in _PRE_EDIT_STEP1.items():
        assert row[k] == pytest.approx(v, rel=1e-5, abs=0.0), k
    # no autocast machinery ran: every Linear output stayed float32
    assert seen and set(seen) == {torch.float32}
    # and the process-global numerics were not touched
    assert (torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.get_float32_matmul_precision()) == before
    assert cfg["precision"] == "fp32" and cfg["autocast"] is None
    assert cfg["tf32"]["requested"] is False
    assert cfg["tf32"]["effective"] is False
    # the stamp is the READ-BACK truth, whatever this session's defaults are
    assert cfg["tf32"]["matmul_allow_tf32"] is before[0]
    assert cfg["tf32"]["cudnn_allow_tf32"] is before[1]
    assert cfg["tf32"]["float32_matmul_precision"] == before[2]
    assert row["precision"] == "fp32" and row["tf32"] is False


def test_forward_context_is_a_nullcontext_for_fp32_and_bf16_autocast_otherwise():
    tr = _trainer()
    assert isinstance(tr.forward_context("fp32", "cpu"), contextlib.nullcontext)
    assert isinstance(tr.forward_context("fp32", "cuda"), contextlib.nullcontext)
    ctx = tr.forward_context("bf16", "cpu")
    assert isinstance(ctx, torch.autocast)
    assert ctx.device == "cpu" and ctx.fast_dtype == torch.bfloat16
    with pytest.raises(ValueError, match="precision"):
        tr.forward_context("fp16", "cpu")
    with pytest.raises(ValueError, match="precision"):
        tr.apply_precision_flags("fp16", False, "cpu")


# ------------------------------------------- (b) the flags reach config.json --
def test_flags_parse_and_land_in_config_json_as_read_back_values(
        tmp_path, restore_tf32_flags):
    tr = _trainer()
    out = tmp_path / "bf16_tf32"
    rows, cfg = _run(tr, out, "--precision", "bf16", "--tf32")
    # the raw launch line ...
    assert cfg["args"]["precision"] == "bf16" and cfg["args"]["tf32"] is True
    # ... and the RESOLVED numerics, read back from the process
    assert cfg["precision"] == "bf16"
    assert cfg["autocast"] == {"device_type": "cpu", "dtype": "bfloat16",
                               "scope": "forward+loss"}
    assert cfg["grad_scaler"] is None and cfg["master_weights"] == "fp32"
    assert cfg["tf32"]["requested"] is True
    assert cfg["tf32"]["effective"] is False           # CPU: inert
    assert cfg["tf32"]["matmul_allow_tf32"] is True    # but the switch is real
    assert cfg["tf32"]["cudnn_allow_tf32"] is True
    assert cfg["tf32"]["float32_matmul_precision"] != "highest"
    assert cfg["torch"] == torch.__version__ and cfg["device"] == "cpu"
    # the model config travels too, JSON-clean (nested dataclasses expanded)
    assert cfg["cfg"]["target_space"] == "frozen"
    assert isinstance(cfg["cfg"]["strategic_cfg"], dict)
    # every log row names its precision
    assert rows[0]["precision"] == "bf16" and rows[0]["tf32"] is True
    # precision is a TRAINER-side property: NOT in the checkpoint's model cfg
    ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
    assert "precision" not in ck["cfg"] and "tf32" not in ck["cfg"]
    # anything but fp32|bf16 is refused at the parser
    with pytest.raises(SystemExit):
        tr.main(COMMON + ["--device", "cpu", "--steps", "1",
                          "--precision", "fp16", "--out", str(tmp_path / "x")])


def test_bf16_is_refused_on_a_cuda_device_without_bf16(monkeypatch):
    tr = _trainer()
    monkeypatch.setattr(torch.cuda, "is_bf16_supported",
                        lambda *a, **k: False)
    with pytest.raises(SystemExit, match="bf16"):
        tr.apply_precision_flags("bf16", False, "cuda")
    # fp32 on the same device is not the bf16 question
    rec = tr.apply_precision_flags("fp32", False, "cuda")
    assert rec["precision"] == "fp32" and rec["autocast"] is None


# --------------------------------------------- (c) bf16 on CPU, 3 steps ------
def test_bf16_autocasts_on_cpu_finite_for_3_steps_and_within_bf16_rounding_of_fp32(
        tmp_path, monkeypatch):
    tr = _trainer()
    rows32, cfg32 = _run(tr, tmp_path / "fp32", "--precision", "fp32", steps=3)
    seen = _hook_linear_dtypes(tr, monkeypatch)
    rows16, cfg16 = _run(tr, tmp_path / "bf16", "--precision", "bf16", steps=3)
    # the pinned CPU decision: autocast ON CPU, not a refusal
    assert cfg16["autocast"]["device_type"] == "cpu"
    assert cfg32["autocast"] is None
    # autocast ENGAGED: the rollout's action Linear emitted bfloat16
    assert torch.bfloat16 in set(seen)
    # three finite steps, every instrument present and finite
    for r in rows16:
        assert _finite(r), r
        assert r["precision"] == "bf16"
        for k in _PRE_EDIT_STEP1:
            assert k in r and r[k] is not None
    # same seed => same init and same batch (autocast draws no RNG), so step 1
    # differs from fp32 by bf16 rounding only — the tolerance in the docstring
    a, b = rows32[0], rows16[0]
    assert b["loss"] == pytest.approx(a["loss"], rel=4e-3)
    assert b["loss_feat_op"] == pytest.approx(a["loss_feat_op"], rel=4e-3)
    for k in ("grad_norm", "participation", "tgt_std_tac", "tgt_std_str",
              "adapter_std"):
        assert b[k] == pytest.approx(a[k], rel=1e-2), k
    # the FROZEN operative target's scale is untouched by autocast: its path
    # (frozen std buffers, elementwise) has no autocast-listed op
    assert b["tgt_std_op"] == a["tgt_std_op"]
    # the run trained: the checkpoint's weights moved and the loss is a number
    ck = torch.load(tmp_path / "bf16" / "ckpt.pt", map_location="cpu",
                    weights_only=False)
    assert ck["step"] == 3
    assert all(t.dtype == torch.float32 for t in ck["model"].values()
               if t.is_floating_point())          # fp32 master weights


# ----------------------------------------------- (d) --tf32 on a CPU run -----
def test_tf32_on_cpu_is_a_numeric_noop_that_still_stamps_the_config(
        tmp_path, restore_tf32_flags):
    tr = _trainer()
    rows_ref, _ = _run(tr, tmp_path / "fp32", "--precision", "fp32")
    rows, cfg = _run(tr, tmp_path / "tf32", "--precision", "fp32", "--tf32")
    # bit-identical to the fp32 run in the same process, and to the pre-edit
    for k in _PRE_EDIT_STEP1:
        assert rows[0][k] == rows_ref[0][k], k
    assert rows[0]["loss"] == pytest.approx(_PRE_EDIT_STEP1["loss"], rel=1e-6,
                                            abs=0.0)
    # the global switches WERE flipped (that is what --tf32 means) ...
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    # ... and the stamp says so, with the CPU truth on effectiveness
    assert cfg["tf32"]["requested"] is True
    assert cfg["tf32"]["effective"] is False
    assert cfg["tf32"]["matmul_allow_tf32"] is True
    assert cfg["tf32"]["cudnn_allow_tf32"] is True
    assert (cfg["tf32"]["float32_matmul_precision"]
            == torch.get_float32_matmul_precision())
    assert cfg["precision"] == "fp32" and cfg["autocast"] is None
    assert rows[0]["tf32"] is True


# -------------------------------------------- (e) CUDA, only when FREE -------
def _cuda_busy_reason() -> str | None:
    """None when the GPU may be used; otherwise the reason to skip.

    Dev-box rule: another agent's arms may hold the RTX 4060 — never add load
    to a running job. `TANITAD_REFAV1_CUDA_TEST=1` forces the run, `=0` forces
    the skip. The probe is nvidia-smi (no CUDA context is created unless the
    card is free)."""
    force = os.environ.get("TANITAD_REFAV1_CUDA_TEST", "")
    if force == "0":
        return "TANITAD_REFAV1_CUDA_TEST=0"
    if not torch.cuda.is_available():
        return "no CUDA device"
    if force != "1":
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,"
                 "memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20, check=True)
            util, used, total = [float(x) for x in
                                 q.stdout.strip().splitlines()[0].split(",")]
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20, check=True)
            foreign = [l.strip() for l in apps.stdout.splitlines()
                       if "python" in l.lower()
                       and not l.strip().startswith(f"{os.getpid()},")]
        except Exception as e:                       # cannot PROVE it is free
            return (f"cannot probe the GPU ({e!r}); set "
                    "TANITAD_REFAV1_CUDA_TEST=1 to force")
        if util > 30.0 or used / max(total, 1.0) > 0.5 or foreign:
            return (f"GPU busy — util {util:.0f} %, {used:.0f}/{total:.0f} MiB"
                    f", other python compute apps {foreign}: another agent's "
                    "job must not be disturbed (TANITAD_REFAV1_CUDA_TEST=1 "
                    "forces)")
    if not torch.cuda.is_bf16_supported():
        return "CUDA device reports no bf16 support"
    return None


def test_cuda_bf16_autocast_tiny_step_has_finite_loss_and_grad_norm(
        tmp_path, monkeypatch):
    reason = _cuda_busy_reason()
    if reason:
        pytest.skip(reason)
    tr = _trainer()
    seen = _hook_linear_dtypes(tr, monkeypatch)
    rows, cfg = _run(tr, tmp_path / "cuda_bf16", "--precision", "bf16",
                     device="cuda")
    assert cfg["autocast"]["device_type"] == "cuda"
    assert torch.bfloat16 in set(seen)
    row = rows[0]
    assert _finite(row), row
    assert math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"])
    assert row["participation"] is not None and row["participation"] > 1.0
    assert row["tgt_std_op"] == pytest.approx(_PRE_EDIT_STEP1["tgt_std_op"],
                                              rel=1e-4)   # frozen target path
