"""Probe: curvature at t0 from the STEER channel (ha0_ext's source) vs from the POSE yaw
backward difference (the ego-history channel), on the TRAIN-a6 manifest (139 train clips).
CPU, metadata only (no frames)."""
import math, torch
m = torch.load('D:/refcv6_eval_kit/data/refcv6-b1-416x1024-train-a6/_v2manifest.pt',
               map_location='cpu', weights_only=False)
dt = 0.1
ks, kp, vs, a_fd, a_ax, om = [], [], [], [], [], []
for P, A in zip(m['poses'], m['actions']):
    v = P[:, 3]; yaw = P[:, 2]
    dyaw = ((yaw[1:] - yaw[:-1]) + math.pi) % (2 * math.pi) - math.pi
    w = dyaw / dt                                  # omega at t (backward), t = 1..T-1
    k_st = torch.tan(A[1:, 0]) / 2.9               # kappa from steer at t
    vs.append(v[1:]); ks.append(k_st); om.append(w)
    a_fd.append((v[1:] - v[:-1]) / dt); a_ax.append(A[1:, 1])
v = torch.cat(vs); ks = torch.cat(ks); om = torch.cat(om); a_fd = torch.cat(a_fd); a_ax = torch.cat(a_ax)
print('n frames', v.numel())
print('|kappa_steer| quantiles 50/90/99/99.9/max:',
      [round(float(x), 4) for x in torch.quantile(ks.abs(), torch.tensor([.5, .9, .99, .999]))], round(float(ks.abs().max()), 4))
print('frac |kappa_steer| > 0.12:', round(float((ks.abs() > 0.12).float().mean()), 5),
      ' > 0.2:', round(float((ks.abs() > 0.2).float().mean()), 5))
for lo, hi in [(0, .5), (.5, 1), (1, 2), (2, 4), (4, 8), (8, 15), (15, 40)]:
    s = (v >= lo) & (v < hi)
    if s.sum() == 0: continue
    kpose = om[s] / v[s].clamp_min(1e-6)
    r0 = v[s] * ks[s]                               # yaw rate implied by steer
    print(f'v in [{lo},{hi}) n={int(s.sum())}: |omega_pose - v*kappa_steer| med {float((om[s]-r0).abs().median()):.4f} '
          f'p90 {float(torch.quantile((om[s]-r0).abs(), .9)):.4f} rad/s | |kpose-ksteer| med {float((kpose-ks[s]).abs().median()):.4f} '
          f'p90 {float(torch.quantile((kpose-ks[s]).abs(), .9)):.4f} | corr(omega, v*k) {float(torch.corrcoef(torch.stack([om[s], r0]))[0,1]):.4f}')
print('a_fd vs ax: corr', round(float(torch.corrcoef(torch.stack([a_fd, a_ax]))[0, 1]), 4),
      ' med|diff|', round(float((a_fd - a_ax).abs().median()), 4))
