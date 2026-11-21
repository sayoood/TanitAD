"""STEP 1j - candidate C: keep the vendor's FUSED kernel, drop only its three
bf16 casts, so the fused addmm+GELU runs in the tensors' own dtype.

Patch target is `sam3.model.vitdet.addmm_act` (vitdet.py:31 binds the name
into its own namespace, so patching sam3.perflib.fused would NOT take).
Compared against B (plain fp32 fc1+act) on the same frame, same processor."""
import json
import os
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
import sam3.model.vitdet as vitdet                               # noqa: E402

CONCEPTS = ["road", "sky", "car", "truck", "bus", "pedestrian", "cyclist",
            "traffic light", "traffic sign", "tree"]
fx = json.load(open("/content/repo/colab/fixtures/"
                    "sam3_backfill_expected.json"))
CID = fx["clips"][0]
mp4 = L2.hf_download(L2.DS_LABELS,
                     f"bridged_w120train_2400/videos/{CID}.mp4")
frames, _t, _n = ph0_pilot.sample_clip_frames(mp4, t0_s=8.0)
IMG = Image.fromarray(frames[len(frames) // 2])
PROC, meta = ph0_sam3.build_processor(None)

ORIG_ADDMM = vitdet.addmm_act
ORIG_FWD = vitdet.Mlp.forward
_ACT_OP = torch.ops.aten._addmm_activation


def addmm_act_same_dtype(activation, linear, mat1):
    """sam3/perflib/fused.py::addmm_act with the three `.to(torch.bfloat16)`
    casts removed - everything stays in mat1's dtype."""
    bias = linear.bias.detach().to(mat1.dtype)
    w = linear.weight.detach().to(mat1.dtype)
    flat = mat1.reshape(-1, mat1.shape[-1])
    if activation in (torch.nn.functional.relu, torch.nn.ReLU):
        y = _ACT_OP(bias, flat, w.t(), beta=1, alpha=1, use_gelu=False)
    elif activation in (torch.nn.functional.gelu, torch.nn.GELU):
        y = _ACT_OP(bias, flat, w.t(), beta=1, alpha=1, use_gelu=True)
    else:
        raise ValueError(f"Unexpected activation {activation}")
    return y.view(mat1.shape[:-1] + (y.shape[-1],))


def fp32_forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.norm(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x


def run(tag):
    torch.cuda.reset_peak_memory_stats()
    res, t0 = {}, time.time()
    try:
        state = PROC.set_image(IMG)
        for c in CONCEPTS:
            out = PROC.set_text_prompt(state=state, prompt=c)
            sc = out.get("scores")
            sc = None if sc is None else sc.detach().float().cpu()
            n = 0 if sc is None else int(sc.reshape(-1).shape[0])
            res[c] = (n, [round(float(v), 6)
                          for v in (sc.reshape(-1)[:4] if n else [])])
    except Exception as e:
        res["__error__"] = f"{type(e).__name__}: {e}"[:160]
    dt, peak = time.time() - t0, torch.cuda.max_memory_allocated() / 2**20
    print(f"\n=== {tag} === {dt:.1f}s peak {peak:.0f} MiB")
    for c in CONCEPTS:
        if c in res:
            print(f"    {c:<14} n={res[c][0]:<3} {res[c][1]}")
    if "__error__" in res:
        print("    ERROR:", res["__error__"])
    return res, dt, peak


vitdet.Mlp.forward, vitdet.addmm_act = fp32_forward, ORIG_ADDMM
B, dtB, pkB = run("B: plain fp32 fc1+act (no fused kernel)")
vitdet.Mlp.forward, vitdet.addmm_act = ORIG_FWD, addmm_act_same_dtype
C, dtC, pkC = run("C: FUSED kernel, bf16 casts removed")
# second C pass to separate warm-up from steady state
C2, dtC2, pkC2 = run("C (2nd pass, warm)")
vitdet.addmm_act = ORIG_ADDMM

print("\n=== B vs C ===")
for c in CONCEPTS:
    if B.get(c) and C.get(c):
        same = B[c][0] == C[c][0]
        d = max([abs(x - y) for x, y in zip(B[c][1], C[c][1])], default=0.0)
        print(f"    {c:<14} nB={B[c][0]} nC={C[c][0]} same_n={same} "
              f"max|dscore|={d:.2e}")
print("\n=== LIVENESS CONTROL (road/sky must be non-zero) ===")
for t, R in (("B", B), ("C", C)):
    print(f"  {t}: road={R.get('road',(0,))[0]} sky={R.get('sky',(0,))[0]} "
          f"LIVE={R.get('road',(0,))[0] > 0 and R.get('sky',(0,))[0] > 0}")
print(f"\n[wall] B {dtB:.1f}s  C {dtC:.1f}s  C-warm {dtC2:.1f}s")
print(f"[peak MiB] B {pkB:.0f}  C {pkC:.0f}  C-warm {pkC2:.0f}")
print("DIAG10_DONE")
