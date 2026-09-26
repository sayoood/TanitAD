"""Independent recomputation of TRAINING_TIME.md from raw/2026-09-21-camera-scaling/cam_scaling.txt.
Every input here is TRANSCRIBED FROM THE ARTIFACT; none is taken from the document under review."""
M = {("vits16",1):0.127, ("vits16",2):0.243, ("vits16",4):0.492,
     ("vitb16",1):0.279, ("vitb16",2):0.554, ("vitb16",4):1.139,
     ("vitl16",1):0.788, ("vitl16",2):1.630}                    # vitl16/4 refused (9.49 GiB on 8 GiB)
print("== camera-scaling ratios, recomputed from the artifact ==")
for b in ("vits16","vitb16","vitl16"):
    r2 = M[(b,2)]/M[(b,1)]; r4 = M.get((b,4),0)/M[(b,1)] if (b,4) in M else None
    print(f"  {b}: t(2)/t(1)={r2:.4f}   t(4)/t(1)={f'{r4:.4f}' if r4 else '--'}")
print()
lo, hi = M[("vits16",4)]/M[("vits16",1)], M[("vitb16",4)]/M[("vitb16",1)]
print(f"MEASURED 4-cam multiplier band: {lo:.4f} .. {hi:.4f}")
base = M[("vitl16",1)]
print(f"ViT-L 4-cam extrapolation from the MEASURED band: {base*lo:.4f} .. {base*hi:.4f} s/sample")
print("   document says 3.05 - 3.22   ->", f"{base*lo:.2f} - {base*hi:.2f}")
print()
print("== the document's stated third route: 'the ViT-L per-doubling ratio (2.067) instead gives 3.26' ==")
rl = M[("vitl16",2)]/M[("vitl16",1)]
print(f"  ViT-L per-doubling ratio, recomputed  = {rl:.4f}")
print(f"  t(1) * ratio^2                        = {base*rl*rl:.4f}")
print(f"  t(2) * ratio                          = {M[('vitl16',2)]*rl:.4f}")
print(f"  t(2) * 2.000 (the ANALYTIC doubling)  = {M[('vitl16',2)]*2.0:.4f}   <- this is 3.26")
print(f"  => 3.26 comes from the ANALYTIC x2, NOT from the measured 2.067 the sentence names.")
print(f"  => honest upper bound on the stated method: {max(base*rl*rl, M[('vitl16',2)]*rl):.4f} s/sample")
print()
PASSES = 337_000*25
print(f"== sample-passes: 337,000 x 25 = {PASSES:,} ==")
def hrs(s): return PASSES*s/3600
for lbl, s in (("doc low 3.05",3.05), ("doc high 3.26",3.26), ("honest high 3.372",base*rl*rl)):
    print(f"  {lbl:16s} -> {hrs(s):9,.0f} 4060-hours")
print()
print("== the A40 multiplier: what do the document's OWN three cited ratios give? ==")
for nm, a40, r4060 in (("FP32 TFLOPS",37.4,15.11), ("bandwidth GB/s",696,272), ("TF32 tensor TFLOPS",74.8,30.2)):
    print(f"  {nm:20s} {a40}/{r4060} = {a40/r4060:.3f}x")
print("  document quotes the band 2.48 - 3.0 ; max of its own three ratios = 2.559")
print()
for nm, mlo, mhi in (("document band", 2.48, 3.0), ("band its OWN evidence supports", 2.48, 2.559)):
    for arm, passes, sps_lo, sps_hi in (("D full ViT-L", 337_000*25, 3.05, 3.26),):
        h_lo = passes*sps_lo/3600/mhi; h_hi = passes*sps_hi/3600/mlo
        print(f"  {nm:32s} arm {arm}: {h_lo:7,.0f} - {h_hi:7,.0f} A40-h = {h_lo/24:5.1f} - {h_hi/24:5.1f} days")
print()
print("== §5 cross-check, recomputed ==")
for lbl, h20 in (("148 TFLOPS (doc; NVIDIA lists this as the SPARSE TF32 figure)",148.0),
                 ("74 TFLOPS (dense TF32)",74.0), ("44 TFLOPS (FP32 non-tensor; the paper trains FP32)",44.0)):
    r = h20/30.2
    pred = 608*r
    print(f"  H20 {lbl}")
    print(f"     ratio vs 4060 = {r:.2f}x  -> 608 GPU-h predicts {pred:,.0f} 4060-h;"
          f"  our 7,140-7,630 is {7140/pred:.1f}-{7630/pred:.1f}x slower")
