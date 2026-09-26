"""CROSS-CHECK: rebuild DriveZero's FOUR-camera student from the two sources it CITES,
and compare against a number I did not fit -- their published 18.58 M trainable (Table A12).

⭐ The discriminator that makes this worth anything: the module choices below come from
DrivoR (ref [34]) and PETR (ref [50]), read from the papers, NOT reverse-engineered from
18.58 M. If they land near it, the reading is corroborated; if the CURRENT REFe reading
lands far above it, that reading is refuted.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def ln(d):                    return 2 * d
def lin(a, b, bias=True):     return a * b + (b if bias else 0)
def mha(d, ctx=None):
    if ctx is None or ctx == d: return 3 * d * d + 3 * d + lin(d, d)
    return d * d + 2 * d * ctx + 3 * d + lin(d, d)
def crossblock(d, ctx, r=4.0):
    hid = int(d * r)
    return 3 * ln(d) + mha(d) + mha(d, ctx) + lin(d, hid) + lin(hid, d)
def regcompress(d, r=4.0, blocks=1):
    hid = int(d * r)
    return blocks * (3 * ln(d) + mha(d) + lin(d, hid) + lin(hid, d))

D, DD, NCAM, NREG, NPROP = 1024, 256, 4, 16, 64
LORA = 24 * 2 * (2 * 32 * D)
PUB = 18.58e6

def theirs(compress, pos3d_kind, score_head, score_q_mlp, score_depth=4):
    p = {}
    p["registers"] = NCAM * NREG * D                              # DrivoR: N x R at D_ViT
    p["reg_compress"] = {"inbackbone": 0,
                         "xattn_mlp4": regcompress(D, 4.0, 1),
                         "xattn_mlp1": regcompress(D, 1.0, 1)}[compress]
    p["scene_proj"] = lin(D, DD)
    p["pos3d"] = {"petr_4C": lin(64 * 3, 4 * D) + lin(4 * D, D),  # PETR official: 192->4096->1024
                  "petr_C":  lin(64 * 3, D) + lin(D, D),
                  "table":   NCAM * 1920 * D,
                  "none":    0}[pos3d_kind]
    p["ego_enc"] = lin(8 + 4, DD) + lin(DD, DD)
    p["queries"] = NPROP * DD
    p["dec"] = 4 * crossblock(DD, DD, 4.0)                        # DrivoR: 4 layers, 256, dilation 4
    p["traj_head"] = lin(DD, DD) + lin(DD, 60)
    # DrivoR 3.4: "Each decoded trajectory is turned into a D_score-dimensional query using an MLP"
    p["score_q_mlp"] = (lin(60, DD) + lin(DD, DD)) if score_q_mlp else 0
    p["score_dec"] = score_depth * crossblock(DD, DD, 4.0)        # DrivoR: "mirroring", 4 layers
    # DrivoR 3.4: "we predict the six score components using a dedicated MLP for each score"
    p["score_head"] = 6 * (lin(DD, DD) + lin(DD, 1)) if score_head == "per_comp" else lin(DD, 6)
    return p, sum(p.values()) + LORA

CASES = [
 ("A  DrivoR-faithful + PETR literal (4C) + per-comp MLPs", "inbackbone", "petr_4C", "per_comp", True),
 ("B  DrivoR-faithful + PETR (hidden=C)  + per-comp MLPs",  "inbackbone", "petr_C",  "per_comp", True),
 ("C  DrivoR-faithful + PETR literal, single linear score", "inbackbone", "petr_4C", "linear",   False),
 ("D  DrivoR-faithful + LEARNED TABLE (4 cams)",            "inbackbone", "table",   "per_comp", True),
 ("E  REFe's CURRENT reading: x-attn @1024 MLP 4x + table", "xattn_mlp4", "table",   "linear",   False),
 ("F  x-attn @1024 MLP 4x + PETR literal",                  "xattn_mlp4", "petr_4C", "linear",   False),
 ("G  x-attn @1024 MLP 1x + PETR literal",                  "xattn_mlp1", "petr_4C", "linear",   False),
]

print("=" * 104)
print("RECONSTRUCTING DriveZero's FOUR-camera student -- target is their published 18,580,000")
print("=" * 104)
print(f"{'reading':<56} {'trainable':>12} {'gap vs 18.58M':>15} {'rel':>8}")
for name, c, pk, sh, sq in CASES:
    p, tr = theirs(c, pk, sh, sq)
    print(f"{name:<56} {tr:>12,} {tr - PUB:>+15,.0f} {100*(tr-PUB)/PUB:>+7.1f} %")
print()
print("  ⇒ the readings that put a MULTI-MILLION-PARAMETER module in the compression path")
print("    (E, F) OVERSHOOT a number they were not fitted to; the DrivoR-faithful readings")
print("    (A, B, C) land inside it.  Row A is the closest.")
print()
p, tr = theirs("inbackbone", "petr_4C", "per_comp", True)
print("PER-MODULE for reading A (the closest reconstruction):")
for k, v in sorted(p.items(), key=lambda kv: -kv[1]):
    if v: print(f"   {k:<16} {v:>12,}")
print(f"   {'LoRA(trunk)':<16} {LORA:>12,}")
print(f"   {'TOTAL':<16} {tr:>12,}   unexplained residual vs published: {PUB-tr:>+11,.0f}")
