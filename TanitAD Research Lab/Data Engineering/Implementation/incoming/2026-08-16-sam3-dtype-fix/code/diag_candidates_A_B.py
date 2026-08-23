"""STEP 1i ? TWO candidate fixes, MEASURED head-to-head on the same frame.

A) scoped autocast bf16 around set_image/set_text_prompt  ? the context every
   other SAM3 entry point enters (sam3_multiplex_base.py:171); makes fc2's
   fp32 weight agree with the fused bf16 GEMM of vitdet.py:71.
B) fp32 restore ? bypass sam3.perflib.fused.addmm_act in Mlp.forward, so the
   whole trunk stays fp32 (the precision pod4 MEASURED on 2026-08-12).

* LIVENESS CONTROL: `road` and `sky` cannot legitimately be 0 on a
forward-facing driving frame ? the fix is proven by a NON-ZERO detection on
those, never by the absence of a traceback."""
import json
import os
import subprocess
import sys
import time

os.chdir("/content")
for p in ("/content/repo/colab", "/content/repo/stack",
          "/content/repo/stack/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch                                                     # noqa: E402
import ph0_sam3                                                  # noqa: E402
import ph0_pilot                                                 # noqa: E402
import s2_lab_lib as L2                                          # noqa: E402
from PIL import Image                                            # noqa: E402

# --- does USE_PERFLIB=0 help? (fresh interpreter, source-level check) --------
r = subprocess.run(
    [sys.executable, "-c",
     "import sam3.perflib as p, inspect, sam3.model.vitdet as v;"
     "print('is_enabled', p.is_enabled);"
     "print('Mlp.forward uses addmm_act:',"
     " 'addmm_act' in inspect.getsource(v.Mlp.forward))"],
    env={**os.environ, "USE_PERFLIB": "0"}, capture_output=True, text=True)
print("[perflib=0]", r.stdout.strip(), r.stderr.strip()[-200:])

CONCEPTS = ["road", "sky", "car", "truck", "bus", "pedestrian", "cyclist",
            "traffic light", "traffic sign", "tree"]

fx = json.load(open("/content/repo/colab/fixtures/"
                    "sam3_backfill_expected.json"))
CID = fx["clips"][0]
mp4 = L2.hf_download(L2.DS_LABELS,
                     f"bridged_w120train_2400/videos/{CID}.mp4")
frames, _t, _n = ph0_pilot.sample_clip_frames(mp4, t0_s=8.0)
IMG = Image.fromarray(frames[len(frames) // 2])
print("[frame]", CID[:8], "of", len(frames), "frames", frames[0].shape)

PROC, meta = ph0_sam3.build_processor(None)
model = PROC.model
import sam3.model.vitdet as vitdet                               # noqa: E402
ORIG_FWD = vitdet.Mlp.forward


def fp32_forward(self, x):
    """Vendor Mlp.forward with the fused bf16 GEMM replaced by the plain
    fp32 fc1+act (vitdet.py:70-76, minus perflib)."""
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.norm(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x


def run(tag, ctx_factory, patch_fp32):
    vitdet.Mlp.forward = fp32_forward if patch_fp32 else ORIG_FWD
    torch.cuda.reset_peak_memory_stats()
    res, t0 = {}, time.time()
    try:
        with ctx_factory():
            state = PROC.set_image(IMG)
            for c in CONCEPTS:
                out = PROC.set_text_prompt(state=state, prompt=c)
                sc = out.get("scores")
                sc = None if sc is None else sc.detach().float().cpu()
                n = 0 if sc is None else int(sc.reshape(-1).shape[0])
                res[c] = (n, [round(float(v), 4)
                              for v in (sc.reshape(-1)[:4] if n else [])])
    except Exception as e:
        res["__error__"] = f"{type(e).__name__}: {e}"[:160]
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"\n=== {tag} === {dt:.1f}s  peak {peak:.0f} MiB")
    for c in CONCEPTS:
        if c in res:
            print(f"    {c:<14} n={res[c][0]:<3} scores={res[c][1]}")
    if "__error__" in res:
        print("    ERROR:", res["__error__"])
    return res, dt, peak


import contextlib                                                # noqa: E402
A, dtA, pkA = run("A: autocast bf16 (scoped)",
                  lambda: torch.autocast(device_type="cuda",
                                         dtype=torch.bfloat16), False)
B, dtB, pkB = run("B: fp32 (perflib fused bypassed)",
                  contextlib.nullcontext, True)
vitdet.Mlp.forward = ORIG_FWD

print("\n=== LIVENESS CONTROL (road/sky MUST be non-zero) ===")
for tag, R in (("A", A), ("B", B)):
    live = (R.get("road", (0,))[0] > 0) and (R.get("sky", (0,))[0] > 0)
    print(f"  {tag}: road n={R.get('road', ('-',))[0]} "
          f"sky n={R.get('sky', ('-',))[0]} -> LIVE={live}")
print("\n=== A vs B score agreement ===")
for c in CONCEPTS:
    if c in A and c in B and A[c][0] and B[c][0]:
        d = [round(abs(x - y), 4) for x, y in zip(A[c][1], B[c][1])]
        print(f"    {c:<14} nA={A[c][0]} nB={B[c][0]} |dscore|={d}")
    elif c in A and c in B:
        print(f"    {c:<14} nA={A[c][0]} nB={B[c][0]}")
print(f"\n[wall] A {dtA:.1f}s  B {dtB:.1f}s | [peak MiB] A {pkA:.0f} B {pkB:.0f}")
print("DIAG9_DONE")
