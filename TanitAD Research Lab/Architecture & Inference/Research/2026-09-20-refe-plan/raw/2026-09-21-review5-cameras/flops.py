"""Analytic FLOPs for REFe at 1 vs 4 cameras -- independent of any timing measurement.
Counts multiply-adds x2, forward only; backward is ~2x forward and scales identically."""
W,D,H = 1024,24,16            # ViT-L
P = (512//16)*(960//16)       # 1920 patch tokens
T = P + 1 + 4                 # + cls + 4 DINOv3 registers
R_PER_CAM, DW, DD, M, HZ = 16, 256, 4, 64, 20
def vit(n_img):
    per = 0
    per += 2*(3*16*16)*W*P                      # patch embed
    for _ in range(D):
        per += 2*T*W*W*4                        # q,k,v,proj
        per += 2*T*T*W*2                        # attn scores + AV
        per += 2*T*W*(4*W)*2                    # mlp
    return per*n_img
def head(ncam):
    Ntok = ncam*P; Rg = ncam*R_PER_CAM
    f = 0
    f += 2*(Rg*W*W + Ntok*W*W*2 + Rg*W*W)       # reg_compress q,k,v,out proj
    f += 2*Rg*Ntok*W*2                          # reg_compress attention
    f += 2*Rg*W*W*2                             # reg_compress mlp (ratio 1)
    f += 2*Ntok*W*DW + 2*Rg*W*DW                # scene_proj over visual + scene
    f += DD*(2*(M*DW*DW*4 + M*M*DW*2 + Rg*DW*DW*2 + M*Rg*DW*2 + M*DW*DW*8))   # traj decoder
    f += DD*(2*(M*DW*DW*4 + M*M*DW*2 + Ntok*DW*DW*2 + M*Ntok*DW*2 + M*DW*DW*8)) # score decoder
    f += 2*M*(HZ*3)*DW*2                        # score_q_mlp
    return f
print(f"{'cams':>4s} {'imgs':>5s} {'ctx tokens':>11s} {'trunk GFLOP':>12s} {'head GFLOP':>11s} {'total GFLOP':>12s}")
for nc in (1,4):
    t,h = vit(nc)/1e9, head(nc)/1e9
    print(f"{nc:4d} {nc:5d} {nc*P:11d} {t:12.1f} {h:11.2f} {t+h:12.1f}")
a=vit(1)+head(1); b=vit(4)+head(4)
print(f"\n1 -> 4 cameras at batch 1, FORWARD FLOPs: x{b/a:.3f}")
print(f"   the frozen trunk is {100*vit(4)/b:.2f} % of the 4-camera total, and it scales EXACTLY x4")
print(f"   the head grows x{head(4)/head(1):.2f} but is only {100*head(4)/b:.2f} % of the work")
print(f"\n=> a correctly-resident 4-camera step should cost about x{b/a:.2f} of a 1-camera step,")
print(f"   NOT x15.25.  Any excess is the machine, not the model.")
