"""GPU cost for the pre-registered sizing experiments, on ONE A40.

BASIS, and why this one:
  MEASURED (raw/refe_wta_batch_starvation.txt, REFE_MODEL.md 7): ViT-L, batch 2, 512x960
  on the dev-box RTX 4060 = 1.56 s/step = 0.780 s/sample.
  ESTIMATED: an A40 is ~3x the 4060 on this workload -> 0.260 s/sample.
  CROSS-CHECK that makes the basis quotable: at their 337 K samples x 25 epochs this gives
  608.5 A40-h, and their paper reports 16 x H20 x 38 h = 608 GPU-hours. The basis reproduces
  a number it was not fitted to.
  ⛔ THE ~3x MULTIPLIER IS UNMEASURED (POD_HANDOFF.md item 3 says so). Every figure below is
  ESTIMATED and must be replaced by a real A40 timing in the first pod hour.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

S_PER_SAMPLE_4060_VITL = 0.780
A40_MULT = 3.0
S_VITL = S_PER_SAMPLE_4060_VITL / A40_MULT
# ViT-S / ViT-L per-token cost ratio, depth x width^2 (the attention-length term is common)
VITS_RATIO = (12 * 384 ** 2) / (24 * 1024 ** 2)
S_VITS = S_VITL * VITS_RATIO

BANKS = {"rehearsal (1,964 tuples, MEASURED bank)": 1964,
         "arm B mini-scale (~14,000 tuples)": 14000,
         "arm C val14-scale (~123,000 tuples)": 123000}

def hours(n, ep, s): return n * ep * s / 3600.0

print("BASIS CHECK")
print(f"  their scale 337,000 x 25 epochs at {S_VITL:.3f} s/sample = "
      f"{hours(337000,25,S_VITL):.1f} A40-h  (their paper: 608 GPU-h)")
print(f"  ViT-S is {1/VITS_RATIO:.1f}x cheaper than ViT-L -> {S_VITS:.4f} s/sample\n")

print("COST OF ONE RUN")
print(f"{'bank':<42} {'epochs':>6} {'ViT-L A40-h':>12} {'ViT-S A40-h':>12}")
for name, n in BANKS.items():
    for ep in (25, 50):
        print(f"{name:<42} {ep:>6} {hours(n,ep,S_VITL):>12.2f} {hours(n,ep,S_VITS):>12.2f}")
print()

EXPTS = [
    ("E-SIZE-1  reg_compress MECHANISM (in-backbone vs bolt-on) + replicate", 3, "L", 14000, 25),
    ("E-SIZE-2  reg_compress MLP ratio at 1024 (4/1/0) + replicate",          4, "L", 14000, 25),
    ("E-SIZE-3  pos3d FORM (table/PETR/none/constant-ctrl) + replicate",      5, "L", 14000, 25),
    ("E-SIZE-4  dec_heads (8/4/16), zero parameter change",                   3, "S", 14000, 25),
    ("E-SIZE-5  register count (8/16/32)",                                    3, "S", 14000, 25),
    ("E-SIZE-7  LoRA rank ladder (8/32/64) + frozen floor + replicate",       5, "L", 14000, 25),
]
print("EXPERIMENT TOTALS (arm-B bank, 14,000 tuples, 25 epochs)")
print(f"{'experiment':<70} {'runs':>5} {'bb':>3} {'A40-h':>9}")
tot = 0.0
for name, runs, bb, n, ep in EXPTS:
    s = S_VITL if bb == "L" else S_VITS
    h = runs * hours(n, ep, s); tot += h
    print(f"{name:<70} {runs:>5} {bb:>3} {h:>9.1f}")
print(f"{'TOTAL':<70} {'':>5} {'':>3} {tot:>9.1f}")
print()
print("SAME LADDER RUN FIRST ON THE REHEARSAL BANK (1,964 tuples, 25 epochs) AS A SMOKE PASS")
tot2 = 0.0
for name, runs, bb, n, ep in EXPTS:
    s = S_VITL if bb == "L" else S_VITS
    h = runs * hours(1964, ep, s); tot2 += h
    print(f"  {name[:60]:<60} {h:>7.2f}")
print(f"  {'TOTAL':<60} {tot2:>7.2f}")
print()
print("POD_HANDOFF.md ARM TABLE -- CONSISTENCY CHECK AGAINST THE SAME BASIS")
for lab, n, ep, claimed in (("A rehearsal", 1746, 50, 0.7), ("B mini-scale", 14000, 25, 2.8),
                            ("D full", 337000, 25, 608.0)):
    got = hours(n, ep, S_VITL)
    print(f"  arm {lab:<14} claimed {claimed:>6.1f} A40-h   this basis {got:>7.1f}   "
          f"ratio {got/claimed:>5.1f}x")
