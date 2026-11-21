#!/usr/bin/env python3
"""HB memory fallback (PREREG_VDAC_P6_A5 line 31 / section 9.2): HB arms ONLY run --batch 4 --accum 2 (effective batch 8).  CPU only (CUDA_VISIBLE_DEVICES=-1), no pod, no data.
Run:  CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python refe/test_vdac_hb_accum.py   ->   ZZTEST_PASS <n>

  * ARGV layer: vdac_argv / the pod script's MODE=argv put 4 / 2 on HB-s0 and HB-SHUF and 8 / 1 on every other arm; the trainer's parser resolves them (a.batch == 4, a.accum == 2); the schedule the
    trainer builds (ceil(sets / batch x epochs / accum), scorer_finetune.py:2986) is IDENTICAL for 8 x 1 and 4 x 2.
  * MODULE layer (the real tiny REFe + the real HBTrunk LoRA copy + the real head, test_vdac_hb.Rig): the aux loss of 8 rows backward at div 1 gives the SAME copy-LoRA and head gradients as two
    micro-batches of 4 backward at div = accum 2 (the aux loss is a mean over frames); ONE HB.lora_step per accum micro-batches.
  * MUTATIONS (each must go RED): drop the divisor; step the optimiser on every micro-batch; accum 1 at batch 4.
  * TRAINER-TEXT layer: the loop's three lines (`aux_backward(dl.tail, a.accum)`, `if micro % a.accum: continue`, `HB.lora_step(...)` AFTER the continue) are present once, in that order; each deletion is RED.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import aux_hb as HB  # noqa: E402
import gate_checks as C  # noqa: E402
import launch_gate as LG  # noqa: E402
import test_aux_s1_wiring as W  # noqa: E402
import test_vdac_hb as T  # noqa: E402

N = [0]
TABLE: list = []


def ok(cond, msg):
    N[0] += 1
    assert cond, msg


def red(cond_caught, what, how):
    N[0] += 1
    assert cond_caught, f"MUTATION NOT CAUGHT: {what}"
    TABLE.append((what, "RED", how))


def rows8():
    g = torch.Generator().manual_seed(11)
    img = torch.rand(8, 1, 3, 32, 32, generator=g)
    label = torch.stack([torch.from_numpy(W.synth_label("T0", str(i))) for i in range(8)])
    return img, label, torch.ones(8, dtype=torch.bool)


def snap_of(rig):
    return ([p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in rig.hb.lora_parameters()],
            [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in rig.aux.head.parameters()])


def update(rig, img, label, valid, *, batch, accum, div=None, step_every_micro=False):
    """one optimiser update as scorer_finetune.py:2991-3016 does it: micro-batches of `batch` rows, aux backward at div (= accum), ONE lora_step per `accum` micro-batches.
    -> (grads of the copy's LoRA tensors and the head just BEFORE the step, number of lora_step calls)"""
    div = accum if div is None else div
    steps = 0
    for m in range(accum):
        sl = slice(m * batch, (m + 1) * batch)
        actx = rig.hb.ctx(rig.model, img[sl], None)
        rig.cap["aux"] = rig.aux.predict(actx)
        rig.aux_backward((label[sl], valid[sl]), div)
        if step_every_micro:
            HB.lora_step(rig.hb, rig.optL, 1e-3)
            steps += 1
    snap = snap_of(rig)
    if not step_every_micro:
        HB.lora_step(rig.hb, rig.optL, 1e-3)
        steps += 1
    return snap, steps


def close(a, b):
    return all(torch.allclose(x, y, rtol=1e-4, atol=1e-8) for x, y in zip(a, b))


def params_close(r1, r2):
    return all(torch.allclose(a, b, rtol=1e-3, atol=1e-7) for a, b in zip(r1.hb.lora_parameters(), r2.hb.lora_parameters()))


def argv_layer():
    parser = __import__("scorer_finetune").build_parser()
    for arm in LG.VDAC_ARMS:
        ns = C.parse_scorer_argv(parser, LG.vdac_argv(arm))[0]
        want = (4, 2) if arm in ("HB-s0", "HB-SHUF", "HBC-s0", "HBC-SHUF") else (8, 1)
        ok((ns.batch, ns.accum) == want, f"{arm}: the trainer's parser resolves --batch/--accum to {(ns.batch, ns.accum)}, want {want}")
        ok(ns.batch * ns.accum == 8, f"{arm}: effective batch stays 8")
        rec = LG.PROFILES["refe_vdac_p6"]["arm_recipe"][LG.VDAC_ARM_CLASS[arm]]
        ok((rec["batch"], rec["accum"]) == want, f"{arm}: the profile's arm-class recipe carries {want}")
    for n in (60000, 59999):                  # the schedule the trainer builds (scorer_finetune.py:2986): the same for 8 x 1 and 4 x 2, so --max-updates still binds
        t81, t42 = int(math.ceil(n / 8 * 2 / 1)), int(math.ceil(n / 4 * 2 / 2))
        ok(t81 == t42 and min(t42, 10000) == 10000, f"schedule at {n} sets: 8x1 -> {t81}, 4x2 -> {t42}, --max-updates 10000 binds")
    bash = shutil.which("bash")
    if bash:
        for arm, want in (("HB-s0", "--batch 4 --accum 2"), ("HB-SHUF", "--batch 4 --accum 2"), ("HBC-s0", "--batch 4 --accum 2"), ("HBC-SHUF", "--batch 4 --accum 2"), ("HC-s0", "--batch 8 --accum 1"), ("HA-s0", "--batch 8 --accum 1")):
            r = subprocess.run([bash, str(HERE / "vdac_pod.sh")], env={**os.environ, "ARM": arm, "MODE": "argv"}, capture_output=True, encoding="utf-8")
            ok(r.returncode == 0 and want in r.stdout, f"vdac_pod.sh MODE=argv ARM={arm} prints '{want}' (rc {r.returncode})")
    TABLE.append(("argv: every arm's (batch, accum): parser + recipe + schedule + pod script", "OK", "HB and HBC 4x2, the rest 8x1"))


def module_layer():
    img, label, valid = rows8()
    ref = T.Rig()
    g_ref, s_ref = update(ref, img, label, valid, batch=8, accum=1)
    ok(s_ref == 1 and all(float(x.abs().sum()) > 0.0 for x in g_ref[0]), "reference 1 x 8: one step, every copy LoRA tensor has a non-zero gradient")
    acc = T.Rig()
    g_acc, s_acc = update(acc, img, label, valid, batch=4, accum=2)
    ok(s_acc == 1, "2 x 4: ONE optimiser step per 2 micro-batches")
    ok(close(g_acc[0], g_ref[0]) and close(g_acc[1], g_ref[1]), "2 x 4 accumulated gradient == 1 x 8 gradient (copy LoRA and head), rtol 1e-4")
    ok(params_close(acc, ref), "after the one step the copy's LoRA tensors agree with the 1 x 8 update")
    ok(any(not torch.equal(p, q) for p, q in zip(acc.hb.lora_parameters(), T.Rig().hb.lora_parameters())), "the step MOVED the copy (control: not equal to a fresh copy)")
    m1 = T.Rig()
    g1, _ = update(m1, img, label, valid, batch=4, accum=2, div=1)
    red(not close(g1[0], g_ref[0]), "divisor dropped (div 1 at accum 2)", "accumulated grad is 2x the 1 x 8 grad")
    m2 = T.Rig()
    _g2, s2 = update(m2, img, label, valid, batch=4, accum=2, step_every_micro=True)
    red(s2 == 2 and not params_close(m2, ref), "optimiser stepped on every micro-batch", "two steps, params differ from the 1 x 8 update")
    m3 = T.Rig()
    g3, s3 = update(m3, img, label, valid, batch=4, accum=1)
    red(s3 == 1 and not close(g3[0], g_ref[0]), "accum 1 at batch 4 (a 4-row update, not the 8-row one)", "gradient of 4 rows != gradient of 8 rows")


def trainer_text_layer():
    src = (HERE / "scorer_finetune.py").read_text(encoding="utf-8")
    a_line = "a_bce, a_hub = aux_backward(dl.tail, a.accum)"
    c_line = "            if micro % a.accum:\n                continue\n"
    s_line = "HB.lora_step(hb, optL, lr_at(upd) * (a.aux_drv_lora_lr / a.lr))"
    ok(all(src.count(x) == 1 for x in (a_line, c_line, s_line)), "the three loop lines each appear once in the real trainer")

    def judge(s):
        return s.count(a_line) == 1 and s.count(c_line) == 1 and s.count(s_line) == 1 and s.index(a_line) < s.index(c_line) < s.index(s_line)
    ok(judge(src), "order: aux backward (/ accum) < the accum `continue` < HB.lora_step, in the real trainer")
    red(not judge(src.replace(a_line, "a_bce, a_hub = aux_backward(dl.tail)")), "trainer: the aux backward divisor deleted", "text check")
    red(not judge(src.replace(c_line, "")), "trainer: the accum `continue` deleted (step every micro-batch)", "text check")
    red(not judge(src.replace(s_line, "pass")), "trainer: HB.lora_step deleted", "text check")


def main() -> int:
    argv_layer()
    module_layer()
    trainer_text_layer()
    print(f"{'mutation / check':72s} result  how")
    for what, res, how in TABLE:
        print(f"{what[:72]:72s} {res:7s} {how}")
    print(f"ZZTEST_PASS {N[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
