"""Stage 2b: recover the radiometric transform by QUANTILE MATCHING.

Why not paired pixels: conditioning ref on our render value is destroyed at high
radiance by sub-pixel misregistration of small bright sources (street lamps) -- it
returned a NON-MONOTONE curve, which no CRF can be. Quantile matching compares the
DISTRIBUTIONS over the same covered pixel set, so it is invariant to how the bright
pixels are arranged, and a monotone radiometric transform is recovered exactly.

FALSIFIER, pre-registered with both outcomes:
  The PPISP CRF is PER-CAMERA and FRAME-INDEPENDENT (exposure/colour are per-frame and
  are identity here). So the curve recovered on frame 0 MUST also be the curve on frames
  150/300/450.
    * SUPPORTED  -> the 4 curves agree (max spread small vs the correction they apply)
    * REFUTED    -> they disagree; the residual is per-frame, not a fixed camera ISP.
"""
import numpy as np, msgpack, itertools, json

D = np.load("/home/nvidia/nurec_work/linear_dump.npz")
frames, render, alpha, ref = D["frames"], D["render"], D["alpha"], D["ref"]
obj = msgpack.unpackb(open("/home/nvidia/nurec_work/x/volume.msgpack","rb").read(), raw=False, strict_map_key=False)
sd = obj["nre_data"]["state_dict"]
CRF = np.frombuffer(sd[".post_processings.0.ppisp.crf_params"], np.float16).astype(np.float64).reshape(6,3,7)
CAM = 3

softplus = lambda z: np.logaddexp(0.0, z)
sigmoid  = lambda z: 1.0/(1.0+np.exp(-z))

def crf_forward(x, toe, shoulder, gamma, center):
    x = np.clip(x, 0.0, 1.0)
    lerp = toe + center*(shoulder - toe)
    a = (shoulder*center)/lerp; b = 1.0 - a; eps = 1e-6
    lo = a*np.power(np.clip(x/center, eps, None), toe)
    hi = 1.0 - b*np.power(np.clip((1.0-x)/(1.0-center), eps, None), shoulder)
    return np.power(np.clip(np.where(x <= center, lo, hi), eps, None), gamma)

QS = np.linspace(0.001, 0.999, 199)
XG = np.linspace(0.0, 1.0, 101)

def quantile_curve(fi, ch, amin=0.995):
    m = alpha[fi] >= amin
    x = np.clip(render[fi][...,ch][m], 0, 1)
    y = ref[fi][...,ch][m]
    qx, qy = np.quantile(x, QS), np.quantile(y, QS)
    # monotone interpolant on a common grid; extrapolate flat
    return np.interp(XG, qx, qy, left=qy[0], right=qy[-1]), qx, qy, int(m.sum())

print("=== quantile-matched transfer curves, per frame, per channel ===")
curves = np.zeros((len(frames), 3, XG.size))
for fi in range(len(frames)):
    for c in range(3):
        curves[fi,c], qx, qy, npx = quantile_curve(fi, c)
    print(f"  frame {frames[fi]}  n_px={npx}")
    for c,n in enumerate("RGB"):
        s = [f"{XG[i]:.1f}->{curves[fi,c,i]:.3f}" for i in (10,20,30,40,50,60,80,100)]
        print(f"    {n}: " + "  ".join(s))

print("\n=== MONOTONICITY (a CRF must be non-decreasing) ===")
for fi in range(len(frames)):
    for c,n in enumerate("RGB"):
        d = np.diff(curves[fi,c])
        print(f"  f{frames[fi]} {n}: min_diff={d.min():+.5f}  "
              f"{'MONOTONE' if d.min() >= -1e-9 else 'NON-MONOTONE'}")

print("\n=== FALSIFIER: does the frame-0 curve TRANSFER to unseen frames? ===")
print("(a per-camera CRF is frame-independent; a per-frame fudge is not)")
for c,n in enumerate("RGB"):
    ref0 = curves[0,c]
    spread = np.abs(curves[:,c,:] - ref0[None,:]).max(axis=1)
    # how big is the correction the curve itself applies (vs identity)?
    corr = np.abs(ref0 - XG).max()
    print(f"  {n}: |curve_f - curve_0|_max per frame = "
          f"{np.round(spread,4)}   correction size |curve_0 - identity|_max = {corr:.4f}")
    print(f"      cross-frame spread / correction = "
          f"{np.round(spread[1:]/max(corr,1e-9),3)}  (small => frame-independent)")

print("\n=== identity check: how far from identity is the transform? ===")
for c,n in enumerate("RGB"):
    print(f"  {n}: mean|curve_0 - x| = {np.abs(curves[0,c]-XG).mean():.4f}, "
          f"curve(0.1)={curves[0,c,10]:.3f} curve(0.3)={curves[0,c,30]:.3f} "
          f"curve(0.5)={curves[0,c,50]:.3f} curve(0.8)={curves[0,c,80]:.3f}")

print("\n=== score all 840 ordered 4-of-7 assignments against the QUANTILE curve ===")
res = []
for perm in itertools.permutations(range(7), 4):
    tot = 0.0
    for c in range(3):
        p = CRF[CAM,c]
        y = crf_forward(XG, 0.3+softplus(p[perm[0]]), 0.3+softplus(p[perm[1]]),
                        0.1+softplus(p[perm[2]]), sigmoid(p[perm[3]]))
        tot += float(np.mean((y - curves[0,c])**2))
    res.append((tot/3.0, perm))
res.sort()
ident = np.mean([(XG - curves[0,c])**2 for c in range(3)])
print(f"  IDENTITY baseline (apply no CRF at all): RMSE={np.sqrt(ident):.5f}")
for rank,(e,perm) in enumerate(res[:6]):
    print(f"  #{rank+1} (toe,sh,gamma,center)=idx{perm}: RMSE={np.sqrt(e):.5f}")
print(f"  worst of {len(res)}: RMSE={np.sqrt(res[-1][0]):.5f}")
print(f"  -> best stored-param decode {'BEATS' if res[0][0] < ident else 'DOES NOT BEAT'} identity")

np.savez("/home/nvidia/nurec_work/quantile_curves.npz", XG=XG, curves=curves, frames=frames)
print("\nsaved /home/nvidia/nurec_work/quantile_curves.npz")
