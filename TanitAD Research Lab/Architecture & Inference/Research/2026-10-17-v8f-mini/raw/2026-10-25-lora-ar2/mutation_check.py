"""Deliberate-regression check of test_v8f_lora_ar2.py: break one thing at a time in the instruments, require the named tests to go RED, restore the bytes (always)."""
import json
import os
import subprocess
import sys
from pathlib import Path

W = Path(r"C:\Users\Admin\AppData\Local\Temp\claude\E--Projects-TanitAD\bfe78918-15e8-439b-8566-593ca3297bd8\scratchpad\lora")
PK = W / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "code"
PY = r"C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=f"{W / 'stack'};{W / 'taniteval'}")

MUTANTS = [
    ("driver: tau 1e-3 -> 1e-2", PK / "v8f_lora_identity_dev.py", b"TAU_DEVICE = 1e-3", b"TAU_DEVICE = 1e-2", "driver_constants or driver_decide"),
    ("driver: allow any manifest source", PK / "v8f_lora_identity_dev.py", None, None, "production_row or refuses_a_row"),
    ("queue: edit an old stage (gatedscale)", PK / "v8f_mini_box2_queue.cmd", None, None, "byte_identical"),
    ("queue: dispatch to a missing label", PK / "v8f_mini_box2_queue.cmd", b'goto :traindscale', b'goto :traindscalX', "queue_ar2_every or queue_ar2_the_queue"),
    ("queue: LF line endings", PK / "v8f_mini_box2_queue.cmd", None, None, "crlf"),
    ("queue: --amp in the smokelora line", PK / "v8f_mini_box2_queue.cmd", b'--max-clips %LORA_NTRAIN% --frames-root', b'--amp --max-clips %LORA_NTRAIN% --frames-root', "runs_the_command or stages_dry"),
    ("chain: identity does not stop the chain", PK / "box2_ops" / "chain_v8f_lora_ar2.sh", b'|| stop "the same-device identity control did not PASS', b'|| true # "the same-device identity control did not PASS', "chain_ar2_order or load_bearing"),
    ("chain: no sha check of the shipped frames", PK / "box2_ops" / "chain_v8f_lora_ar2.sh", b'stop "box 2\'s train frames $c != the local build (sha256)"', b'true', "load_bearing or order"),
    ("smoke: --amp unconditional again", PK / "v8f_lora_smoke.py", b'*(["--amp"] if a.amp else [])', b'"--amp"', "smoke_amp"),
]


def mutate(path, old, new, label):
    raw = path.read_bytes()
    if label.startswith("driver: allow any"):
        i = raw.index(b"def production_problems")
        j = raw.index(b'"source"', i)
        # neutralise the first mention of the source check by renaming the key it compares
        old, new = raw[j:j + 8], b'"sourcX"'
    elif label.startswith("queue: edit an old stage"):
        i = raw.index(b"\r\n:gatedscale\r\n") + 2
        j = i + len(b":gatedscale" + bytes([13, 10]))
        return raw, raw[:j] + b"rem touched" + bytes([13, 10]) + raw[j:]
    elif label.startswith("queue: LF"):
        return raw, raw.replace(b"\r\n", b"\n", 5)
    assert old in raw, (label, old)
    return raw, raw.replace(old, new, 1)


bad = 0
for label, path, old, new, k in MUTANTS:
    raw, mut = mutate(path, old, new, label)
    path.write_bytes(mut)
    try:
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", "C:/Users/Admin/pt_ar2/m", "stack/tests/test_v8f_lora_ar2.py", "-x", "-q", "--tb=no", "-k", k.replace(" or ", " or ")],
                           cwd=W, env=ENV, capture_output=True, text=True, timeout=800)
    finally:
        path.write_bytes(raw)
    red = r.returncode != 0
    last = [x for x in r.stdout.splitlines() if x.strip()][-1] if r.stdout.strip() else ""
    print(("RED  (good)" if red else "GREEN (BLIND)"), "|", label, "|", last)
    bad += 0 if red else 1
print("blind mutants:", bad)
sys.exit(bad)
