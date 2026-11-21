"""What can, and cannot, be recovered about DriveZero's ENCODER sizes.

The paper states exactly three parameter counts (teacher 5.7 M; student 338.46 M full /
18.58 M trainable) and NEVER states the DriveVFM backbone's own size, its patch size, its width
or its depth -- only the names ViT-S / ViT-B / ViT-L. So the backbone size is DERIVED by
subtraction and BOUNDED by the standard ViT ladder, and the residual is reported as a residual,
not silently assigned.
"""

# ---- stated by the paper ----------------------------------------------------
STUDENT_FULL = 338.46      # Table A12
STUDENT_TRAIN = 18.58      # Table A12
TEACHER = 5.7              # section 2.1.1
VIT_L_LORA_RANK = 32       # Q/V LoRA

print("STATED BY THE PAPER (the only three param counts in 32 pages)")
print("  DriveRL teacher (privileged, structured)   %8.2f M" % TEACHER)
print("  DriveZero student, FULL                    %8.2f M" % STUDENT_FULL)
print("  DriveZero student, TRAINABLE               %8.2f M  (%.2f %%)"
      % (STUDENT_TRAIN, 100 * STUDENT_TRAIN / STUDENT_FULL))
print()

frozen = STUDENT_FULL - STUDENT_TRAIN
print("DERIVED by subtraction")
print("  FROZEN part of the student                 %8.2f M" % frozen)
print("  => this is DriveVFM ViT-L plus any frozen glue; the paper does not split it further")
print()


def vit_params(depth, width, mlp_ratio=4, patch=16, img=512, in_ch=3, extra_tokens=5):
    """Standard pre-LN ViT parameter count, in millions. Rough but transparent."""
    per_block = (4 * width * width + 4 * width) + (2 * mlp_ratio * width * width + (mlp_ratio + 1) * width)
    blocks = depth * per_block
    patch_embed = in_ch * patch * patch * width + width
    n_tok = (img // patch) ** 2 + extra_tokens
    pos = n_tok * width
    norms = 2 * width * depth * 2 + 2 * width
    return (blocks + patch_embed + pos + norms) / 1e6


LADDER = {"ViT-S": (12, 384), "ViT-B": (12, 768), "ViT-L": (24, 1024)}
print("STANDARD ViT LADDER (computed here, patch 16 @ 512x512, 5 extra tokens = 4 summary + cls)")
for name, (d, w) in LADDER.items():
    print("  %-7s depth %2d width %4d   %8.2f M" % (name, d, w, vit_params(d, w)))
print()

vit_l = vit_params(*LADDER["ViT-L"])
residual = frozen - vit_l
print("CONSISTENCY CHECK")
print("  frozen student part      %8.2f M" % frozen)
print("  standard ViT-L           %8.2f M" % vit_l)
print("  RESIDUAL (frozen glue)   %8.2f M   <- NOT explained by the paper" % residual)
print()

# LoRA budget: rank r on Q and V of every block
d, w = LADDER["ViT-L"]
lora = d * 2 * (w * VIT_L_LORA_RANK * 2) / 1e6
print("TRAINABLE BUDGET, DERIVED")
print("  Q/V LoRA rank %d on %d blocks of width %d   %8.2f M" % (VIT_L_LORA_RANK, d, w, lora))
print("  stated trainable                            %8.2f M" % STUDENT_TRAIN)
print("  => heads (2 decoders, registers, queries)   %8.2f M" % (STUDENT_TRAIN - lora))
print()

print("WHAT THE PAPER NEVER STATES ABOUT THE ENCODER")
for s in ("DriveVFM parameter count (any of S/B/L)",
          "patch size  -- no ViT-L/14 or /16 notation appears anywhere",
          "width / depth / heads of DriveVFM",
          "GPU count or GPU-hours for DriveVFM pretraining",
          "the mixture RATIOS across the six data sources ('identical mixture ratios' is asserted, never given)",
          "per-source image counts, or any figure in hours",
          "student inference latency or FLOPs"):
    print("  - " + s)
