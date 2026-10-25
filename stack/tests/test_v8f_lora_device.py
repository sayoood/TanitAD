"""PREREG_V8F_LORA AR-2 (2026-10-25): the LoRA live check ran on cuda with a CPU ``torch.Generator`` and crashed box 2's smoke
(``RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'``, ``v8f_mini.py:611``). CPU-only tests cannot see a
device mismatch, so this pins it statically: every ``torch.Generator`` built inside ``live_check_lora`` (which runs on ``dev``)
passes ``device=``. The CPU live check (``live_check(model.cpu(), ...)``) is the same-breath control: the probe must find its
device-less generator, or a zero reading would only mean the probe is blind."""
import ast
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1] / "scripts" / "v8f_mini_train.py"


def _generator_calls(src: str, fn: str) -> list:
    f = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == fn]
    assert len(f) == 1, fn
    return [c for c in ast.walk(f[0]) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and c.func.attr == "Generator" and isinstance(c.func.value, ast.Name) and c.func.value.id == "torch"]


def check(src: str) -> list:
    calls = _generator_calls(src, "live_check_lora")
    if not calls:
        return ["live_check_lora builds no torch.Generator: the probe found nothing to check"]
    return [f"live_check_lora line {c.lineno}: torch.Generator without device= (the model runs on dev)"
            for c in calls if not any(k.arg == "device" for k in c.keywords)]


def test_lora_live_check_generator_is_on_the_model_device():
    assert check(TRAIN.read_text(encoding="utf-8")) == []


def test_control_the_probe_sees_the_cpu_live_checks_generator():
    calls = _generator_calls(TRAIN.read_text(encoding="utf-8"), "live_check")
    assert len(calls) >= 1 and not any(k.arg == "device" for c in calls for k in c.keywords)


def test_mutant_the_crashing_line_is_caught():
    src = TRAIN.read_text(encoding="utf-8")
    assert src.count("torch.Generator(device=dev)") == 1
    bad = check(src.replace("torch.Generator(device=dev)", "torch.Generator()"))
    assert len(bad) == 1 and "without device=" in bad[0]
