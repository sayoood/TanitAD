"""The FINAL recommended REFe configuration, module by module, with the arithmetic.

Formulas verified against a real instantiation of refe/model.py (refe_sizing_params.py,
"ROUTES AGREE"). Head counts do not appear in any formula because nn.MultiheadAttention's
parameter count is 4d^2+4d regardless of num_heads -- that is why every head recommendation
in this study is parameter-free.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def ln(d):                 return 2 * d
def lin(a, b, bias=True):  return a * b + (b if bias else 0)
def mha(d, ctx=None):
    if ctx is None or ctx == d: return 3 * d * d + 3 * d + lin(d, d)
    return d * d + 2 * d * ctx + 3 * d + lin(d, d)
def crossblock(d, ctx, r=4.0):
    hid = int(d * r); return 3 * ln(d) + mha(d) + mha(d, ctx) + lin(d, hid) + lin(hid, d)
def regcompress(d, r, blocks=1):
    hid = int(d * r)
    per = 3 * ln(d) + mha(d) + (lin(d, hid) + lin(hid, d) if r > 0 else -ln(d))
    return blocks * per

D, DD, NTOK, NREG, NPROP, HZ = 1024, 256, 1920, 16, 64, 20
LORA = 24 * 2 * (2 * 32 * D)
FROZEN = 303_079_424
L_FOURIER = 8            # sin+cos at 8 frequencies per goal coordinate

CUR = dict(
    registers = NREG * D,
    pos3d     = NTOK * D,
    reg_compress = regcompress(D, 4.0),
    scene_proj = lin(D, DD),
    ego_enc   = lin(8 + 4, DD) + lin(DD, DD),
    queries   = NPROP * DD,
    dec       = 4 * crossblock(DD, DD, 4.0),
    traj_head = lin(DD, DD) + lin(DD, HZ * 3),
    score_q_mlp = 0,
    score_dec = 4 * crossblock(DD, DD, 4.0),
    score_head = lin(DD, 6),
)
REC = dict(
    registers = NREG * D,                                   # unchanged, 16 stays
    pos3d     = lin(2 * 3, D // 2) + lin(D // 2, D),        # PETR-style, ND=2, hidden C/2
    reg_compress = regcompress(D, 1.0),                     # MLP ratio 4 -> 1 (Perceiver)
    scene_proj = lin(D, DD),
    ego_enc   = lin(8 + 4 + 4 * 2 * L_FOURIER, DD) + lin(DD, DD),   # sinusoidal goal encoding
    queries   = NPROP * DD,
    dec       = 4 * crossblock(DD, DD, 4.0),                # unchanged, PUBLISHED
    traj_head = lin(DD, DD) + lin(DD, HZ * 3),
    score_q_mlp = lin(HZ * 3, DD) + lin(DD, DD),            # DrivoR 3.4 / DriveZero "score query"
    score_dec = 4 * crossblock(DD, DD, 4.0),                # unchanged, PUBLISHED via DrivoR
    score_head = 6 * (lin(DD, DD) + lin(DD, 1)),            # DrivoR "a dedicated MLP for each score"
)

print(f"{'module':<14} {'CURRENT':>12} {'RECOMMENDED':>13} {'delta':>13}   note")
NOTES = {
 "registers":   "unchanged -- 16 is DrivoR's measured per-camera optimum",
 "pos3d":       "learned 1920x1024 table -> PETR-style MLP over analytic 3D coords",
 "reg_compress":"MLP ratio 4 -> 1 (Perceiver/Perceiver IO); heads 8 -> 16, free",
 "scene_proj":  "unchanged",
 "ego_enc":     "+ sinusoidal goal encoding, as their own teacher does",
 "queries":     "unchanged",
 "dec":         "unchanged -- 4 layers @256, FFN dilation 4, PUBLISHED",
 "traj_head":   "unchanged",
 "score_q_mlp": "NEW -- the paper encodes the TRAJECTORY as the score query",
 "score_dec":   "unchanged -- 4 layers, PUBLISHED via DrivoR's 'all decoders'",
 "score_head":  "single Linear -> six per-component MLPs (DrivoR 3.4)",
}
for k in CUR:
    a, b = CUR[k], REC[k]
    print(f"{k:<14} {a:>12,} {b:>13,} {b-a:>+13,}   {NOTES[k]}")
ca, cb = sum(CUR.values()) + LORA, sum(REC.values()) + LORA
print(f"{'LoRA(trunk)':<14} {LORA:>12,} {LORA:>13,} {0:>+13,}   unchanged (rank-32 Q/V, PUBLISHED)")
print(f"{'TRAINABLE':<14} {ca:>12,} {cb:>13,} {cb-ca:>+13,}")
print(f"{'TOTAL':<14} {FROZEN+ca:>12,} {FROZEN+cb:>13,} {cb-ca:>+13,}")
print()
print(f"trainable fraction   {100*ca/(FROZEN+ca):.2f} %  ->  {100*cb/(FROZEN+cb):.2f} %")
print(f"DriveZero published (4 cameras)  18,580,000 trainable / 338,460,000 total / 5.49 %")
print(f"recommended vs published trainable: {cb-18_580_000:+,} ({100*(cb-18_580_000)/18_580_000:+.1f} %)")
print(f"recommended vs published total    : {FROZEN+cb-338_460_000:+,}")
print()
print("IF E-SIZE-1 RETURNS 'in-backbone' (DrivoR's actual method), reg_compress goes to 0:")
rec2 = dict(REC); rec2["reg_compress"] = 0
c2 = sum(rec2.values()) + LORA
print(f"   trainable {c2:,}   total {FROZEN+c2:,}   ({100*c2/(FROZEN+c2):.2f} %)"
      f"   -- and the model is then UNDER 300 M? {FROZEN+c2 < 300e6}")
