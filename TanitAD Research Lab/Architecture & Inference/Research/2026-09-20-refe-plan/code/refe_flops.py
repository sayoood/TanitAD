"""Where does REFe's forward compute actually go? Analytic MAC counts, one camera, ViT-L.

⭐ WHY THIS MATTERS FOR SIZING: if the frozen trunk dominates by two orders of magnitude,
then head sizing is a PARAMETER / OVERFITTING question and NOT a compute question, and any
"we cannot afford a bigger head" argument is false. DrivoR's own decomposition
(Table 11, supp.): 350 GFLOPs backbone || 1 GFLOP rest. This checks whether ours agrees.

MAC = multiply-accumulate. FLOPs ~= 2 x MAC.
"""
G = 1e9

# --- our one-camera production geometry -------------------------------------
H, W, P = 512, 960, 16
NTOK = (H // P) * (W // P)          # 1920 visual tokens, ONE camera
D = 1024                            # ViT-L width
L = 24                              # ViT-L depth
DD = 256                            # planning width
NREG = 16
NPROP = 64

def attn_macs(nq, nk, d, dctx=None):
    """q/k/v projections + the two score/aggregate matmuls + output projection."""
    dctx = dctx or d
    return (nq * d * d                 # q proj
            + 2 * nk * dctx * d        # k, v proj
            + nq * nk * d              # QK^T
            + nq * nk * d              # attn @ V
            + nq * d * d)              # out proj

def mlp_macs(n, d, ratio):
    return 2 * n * d * int(d * ratio)

print("=" * 88)
print("REFe forward MACs, ONE camera, ViT-L, batch 1 -- ANALYTIC")
print("=" * 88)
trunk_tokens = 1 + 4 + NTOK          # cls + 4 DINOv3 registers + patches
trunk = L * (attn_macs(trunk_tokens, trunk_tokens, D) + mlp_macs(trunk_tokens, D, 4.0))
trunk += 3 * P * P * D * NTOK        # patch_embed conv
print(f"  frozen trunk (ViT-L, {trunk_tokens} tokens)        {trunk/G:10.2f} GMAC")

print("\n  -- CURRENT head stack --")
rc_attn = attn_macs(NREG, NTOK, D)
rc_mlp = mlp_macs(NREG, D, 4.0)
print(f"  reg_compress  cross-attn (16 q over {NTOK} kv @1024) {rc_attn/G:10.2f} GMAC")
print(f"  reg_compress  MLP 4x on 16 tokens                 {rc_mlp/G:10.2f} GMAC")
sp_scene = NREG * D * DD
sp_vis = NTOK * D * DD
print(f"  scene_proj(scene)  16 x 1024 x 256                {sp_scene/G:10.4f} GMAC")
print(f"  scene_proj(visual) {NTOK} x 1024 x 256             {sp_vis/G:10.4f} GMAC"
      "   <- only needed because score_dec reads VISUAL tokens")
dec = 4 * (attn_macs(NPROP, NPROP, DD) + attn_macs(NPROP, NREG, DD) + mlp_macs(NPROP, DD, 4.0))
print(f"  dec        4 x CrossBlock, ctx = 16 scene tokens   {dec/G:10.4f} GMAC")
sdec_vis = 4 * (attn_macs(NPROP, NPROP, DD) + attn_macs(NPROP, NTOK, DD) + mlp_macs(NPROP, DD, 4.0))
sdec_scene = dec
print(f"  score_dec  4 x CrossBlock, ctx = {NTOK} VISUAL tok  {sdec_vis/G:10.4f} GMAC   (DriveZero's wording)")
print(f"  score_dec  4 x CrossBlock, ctx = 16 SCENE  tok     {sdec_scene/G:10.4f} GMAC   (DrivoR's wording)")

head_now = rc_attn + rc_mlp + sp_scene + sp_vis + dec + sdec_vis
print(f"\n  HEAD TOTAL (current)  {head_now/G:8.2f} GMAC  = {100*head_now/(trunk+head_now):5.2f} % of the forward")
print(f"  TRUNK                 {trunk/G:8.2f} GMAC  = {100*trunk/(trunk+head_now):5.2f} %")

print("\n  -- IF the compression were IN-BACKBONE (DrivoR's actual method) --")
trunk_reg = L * (attn_macs(trunk_tokens + NREG, trunk_tokens + NREG, D)
                 + mlp_macs(trunk_tokens + NREG, D, 4.0)) + 3 * P * P * D * NTOK
head_drivor = sp_scene + dec + sdec_scene
print(f"  trunk with +16 task registers ({trunk_tokens+NREG} tokens)  {trunk_reg/G:10.2f} GMAC"
      f"   (+{100*(trunk_reg-trunk)/trunk:.2f} % over the plain trunk)")
print(f"  HEAD TOTAL (DrivoR route)  {head_drivor/G:8.4f} GMAC"
      f"  = {100*head_drivor/(trunk_reg+head_drivor):5.3f} % of the forward")
print(f"  ratio head_now / head_drivor = {head_now/head_drivor:,.1f} x")

print("\n" + "=" * 88)
print("SENSITIVITY -- what each sizing knob costs in GMAC (head only)")
print("=" * 88)
for lab, macs in (("reg_compress MLP ratio 4 -> 1", mlp_macs(NREG, D, 4.0) - mlp_macs(NREG, D, 1.0)),
                  ("reg_compress 1 block -> 2 blocks", rc_attn + rc_mlp),
                  ("score_dec context VISUAL -> SCENE", sdec_vis - sdec_scene),
                  ("drop scene_proj(visual) (follows the line above)", sp_vis),
                  ("score_dec depth 4 -> 2 (visual ctx)", sdec_vis / 2),
                  ("registers 16 -> 32 (cross-attn route)",
                   attn_macs(32, NTOK, D) + mlp_macs(32, D, 4.0) - rc_attn - rc_mlp)):
    print(f"  {lab:<50} {macs/G:>9.3f} GMAC  ({100*macs/trunk:5.3f} % of the trunk)")

print("\n" + "=" * 88)
print("ACTIVATION MEMORY -- the dev-box ceiling is 8.00 GB and LoRA forces backprop through")
print("the WHOLE trunk, so trunk activations, not head activations, set the batch size.")
print("=" * 88)
f32 = 4
tr_act = L * trunk_tokens * D * f32 * 6      # ~6 saved tensors per block, order-of-magnitude
hd_act = (NTOK * D + NTOK * DD + NREG * D + NPROP * DD * 8) * f32
print(f"  trunk activations  ~{tr_act/2**30:6.3f} GiB / sample   (order-of-magnitude, 6 tensors/block)")
print(f"  head activations   ~{hd_act/2**30:6.3f} GiB / sample")
print(f"  ratio {tr_act/hd_act:,.0f} x  ->  head sizing does NOT move the batch-size ceiling")
