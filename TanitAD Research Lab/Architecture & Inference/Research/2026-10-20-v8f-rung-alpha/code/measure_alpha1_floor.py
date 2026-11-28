"""v8F alpha-1: MEASURE what the swap puts under the residual -- on REAL eval139 windows (the S1 package's own S2 / S6 window
lists, 4,754 / 3,668 windows, 139 episodes), CPU only.

  (1) FIDELITY  ADE(P10u, P10): the path the decoder rolls at Delta == 0 (P10's knots -> per-tick controls ->
      rollout_unicycle from the measured v0) against P10's own 30 knots. Brief escalation bar: <= 0.05 m.
      Computed through `kinematic_prior.p10_tick_prior`, i.e. the function the decoder calls, with the COMMITTED
      constants (P10_KAPPA_CAP, 4-point subdivision).
  (2) WHAT THE PRIOR IS AS A PLAN, vs ground truth `g` (T0, open loop, eval139): ADE of P10u, of S1's P10, of the
      ha0_ext_pose prior (alpha-0's), of ha0_ext and of `os` (refcv7-50,400, the banked T1 dump) -- paired
      episode-cluster bootstrap (taniteval.ci, B = 2000, seed 0). NOTE the two P10s are different artifacts: v8F-mini's
      (600 train clips, stride 8, a0 = 2-frame difference) vs S1's refcv-grid P10 (4,572-clip view, a0 = 1-frame).
  (3) HOW DIFFERENT the two priors are as controls (slot-mean, the exported `u0_hat`): the distribution alpha-0's
      warm-started residual head would suddenly sit on.
Estimator for every interval: paired episode-cluster bootstrap over the 139 eval139 episodes (models fixed): it answers
'another draw of EPISODES', not 'another training run' -- there is no trained arm in this file.
"""
import json
import sys
import time

import numpy as np
import torch

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/Admin/v8fa1/tree'
OUT = sys.argv[2] if len(sys.argv) > 2 else None
sys.path.insert(0, f'{ROOT}/stack')
sys.path.insert(0, f'{ROOT}/taniteval')
from taniteval import ci as CI                                    # noqa: E402
from tanitad.models import kinematic_prior as KP                  # noqa: E402
from tanitad.models import p10_prior as P                         # noqa: E402
from tanitad.models.kinematic import rollout_unicycle             # noqa: E402

S1 = f'{ROOT}/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-p10-floors-refcv/raw'
P10F = f'{ROOT}/stack/tanitad/configs/v8f_p10/p10_prior.pt'
SHA = 'af5a29ddadb4c7d98f1db0d8116bddd0634e83a7e687fc601989d3fda984d7b3'
MAN = 'C:/Users/Admin/tanitad-data/add203/in/eval139_v2manifest.pt'
W = 8
HZ = (5, 10, 15, 20, 30, 40, 50, 60)
TICKS = {'S2': (5, 10, 15, 20), 'S6': (10, 20, 30, 40, 50, 60)}      # S2: 4 x 0.5 s, S6: 6 x 1.0 s (S1 RESULT, Arms)

m = P.load_frozen_p10(P10F, expect_file_sha256=SHA)
man = torch.load(MAN, map_location='cpu', weights_only=False)
poses = [p.numpy().astype(np.float64) for p in man['poses']]
acts = [a.numpy().astype(np.float64) for a in man['actions']]
print('P10 file sha256', m.file_sha, '| content', m.content_sha()[:16], '| eval139 clips', len(poses))


def roll(ctrl, v0):
    st = torch.zeros(ctrl.shape[0], 4)
    st[:, 3] = v0
    return rollout_unicycle(st, ctrl, dt=0.1)[..., :2]


def ade(path, g):
    return np.linalg.norm(path - g, axis=-1).mean(axis=1)


def boot(a, b, eid):
    r = CI.paired_episode_cluster_bootstrap(a, b, eid, n_boot=2000, seed=0)
    return {k: r[k] for k in ('delta', 'lo', 'hi', 'separated', 'estimator', 'n_windows', 'n_episodes', 'n_boot')}


res = {'p10_file_sha256': m.file_sha, 'p10_content_sha256': m.content_sha(), 'kappa_cap': KP.P10_KAPPA_CAP,
       'bar_ade6_m': 0.05}
for tag in ('S2', 'S6'):
    z = np.load(f'{S1}/arms_{tag}_s0.npz', allow_pickle=True)
    ci_, ws = z['clip_index'].astype(int), z['ws'].astype(int)
    full = ws >= W - 1
    n_all = int(len(ws))
    sel = np.where(full)[0]
    print(tag, 'windows', n_all, 'with a full 8-frame observed window', int(full.sum()))
    po = np.stack([poses[c][t - W + 1:t + 1] for c, t in zip(ci_[sel], ws[sel])])
    ac = np.stack([acts[c][t - W + 1:t + 1] for c, t in zip(ci_[sel], ws[sel])])
    row = {'n_windows_all': n_all, 'n_windows_full_window': int(full.sum()), 'n_dropped_short_history': int((~full).sum())}
    t0 = time.time()
    fid6, fid2, p10u_h, ha_pose_h, a_slot, k_slot, a_hp, k_hp, kmax_tick, cap_tick = ([] for _ in range(10))
    for s in range(0, len(sel), 512):
        pb, ab = torch.from_numpy(po[s:s + 512]), torch.from_numpy(ac[s:s + 512])
        a, k, p30 = KP.p10_tick_prior(m, pb, W, ab)                       # the decoder's own function
        v0 = pb[:, -1, 3].float()
        rolled = roll(torch.stack([a, k], -1), v0)                        # 60 ticks
        d = (rolled[:, 1::2] - p30).norm(dim=-1)                         # vs P10's own knots
        fid6.append(d.mean(1).numpy())
        fid2.append(d[:, :10].mean(1).numpy())
        p10u_h.append(rolled[:, [t - 1 for t in TICKS[tag]]].numpy())
        ahp, khp = KP.prior_controls('ha0_ext_pose', pb.float(), W, None, dt=0.1)
        hp = roll(torch.stack([ahp[:, None].expand(-1, 60), khp[:, None].expand(-1, 60)], -1), v0)
        ha_pose_h.append(hp[:, [t - 1 for t in TICKS[tag]]].numpy())
        a_slot.append(KP.slot_prior(a, HZ).numpy())
        k_slot.append(KP.slot_prior(k, HZ).numpy())
        a_hp.append(ahp.numpy()); k_hp.append(khp.numpy())
        kmax_tick.append(k.abs().amax(1).numpy()); cap_tick.append((k.abs() >= KP.P10_KAPPA_CAP - 1e-6).float().mean(1).numpy())
    fid6, fid2, kmax_tick, cap_tick = map(np.concatenate, (fid6, fid2, kmax_tick, cap_tick))
    p10u_h, ha_pose_h = np.concatenate(p10u_h), np.concatenate(ha_pose_h)
    a_slot, k_slot, a_hp, k_hp = map(np.concatenate, (a_slot, k_slot, a_hp, k_hp))
    eid = ci_[sel]
    row['fidelity_ADE6_P10u_vs_P10_m'] = {
        'mean': float(fid6.mean()), 'median': float(np.median(fid6)), 'p95': float(np.percentile(fid6, 95)),
        'p99': float(np.percentile(fid6, 99)), 'max': float(fid6.max()), 'frac_gt_0p05': float((fid6 > 0.05).mean()),
        'ade2_mean': float(fid2.mean()), 'cluster_bootstrap_mean_ci95': CI.episode_cluster_bootstrap(
            fid6, eid, n_boot=2000, seed=0)}
    row['fidelity_ADE6_P10u_vs_P10_m']['cluster_bootstrap_mean_ci95'] = {
        k: v for k, v in row['fidelity_ADE6_P10u_vs_P10_m']['cluster_bootstrap_mean_ci95'].items()
        if k in ('mean', 'lo', 'hi', 'estimator', 'n_episodes', 'n_boot')}
    row['fidelity_bar_met'] = bool(fid6.mean() <= 0.05)
    row['kappa_tick'] = {'abs_max_p99': float(np.percentile(kmax_tick, 99)), 'frac_windows_with_a_cap_bound_tick':
                         float((cap_tick > 0).mean())}
    # (2) vs ground truth, tier T0 (open loop, the window's observed state), at this grid's horizons
    g = z['g'][sel].astype(np.float64)
    arms = {'P10u(v8F-mini, alpha-1 Delta=0)': ade(p10u_h, g),
            'P10(S1 artifact)': ade(z['P10'][sel].astype(np.float64), g),
            'ha0_ext_pose prior (alpha-0 Delta=0)': ade(ha_pose_h, g),
            'ha0_ext': ade(z['ha0_ext'][sel].astype(np.float64), g),
            'os (refcv7-50,400, T1 dump s0)': ade(z['os'][sel].astype(np.float64), g)}
    row['ade_vs_gt_m'] = {k: float(v.mean()) for k, v in arms.items()}
    base = arms['P10u(v8F-mini, alpha-1 Delta=0)']
    row['paired_vs_P10u'] = {k: boot(v, base, eid) for k, v in arms.items() if not k.startswith('P10u')}
    row['paired_vs_P10u_note'] = 'delta = arm - P10u (positive = P10u scores LOWER ADE = better)'
    row['prior_vs_prior_ADE_m'] = float(np.linalg.norm(p10u_h - ha_pose_h, axis=-1).mean())
    # (3) slot-mean controls, the exported u0_hat of a Delta = 0 plan
    row['slot_mean_controls'] = {
        'a_P10u': {'p05': float(np.percentile(a_slot, 5)), 'median': float(np.median(a_slot)), 'p95': float(np.percentile(a_slot, 95)),
                   'abs_max': float(np.abs(a_slot).max())},
        'kappa_P10u': {'p05': float(np.percentile(k_slot, 5)), 'median': float(np.median(k_slot)),
                       'p95': float(np.percentile(k_slot, 95)), 'abs_max': float(np.abs(k_slot).max())},
        'a_ha0_ext_pose': {'p05': float(np.percentile(a_hp, 5)), 'median': float(np.median(a_hp)),
                           'p95': float(np.percentile(a_hp, 95)), 'abs_max': float(np.abs(a_hp).max())},
        'kappa_ha0_ext_pose': {'p05': float(np.percentile(k_hp, 5)), 'median': float(np.median(k_hp)),
                               'p95': float(np.percentile(k_hp, 95)), 'abs_max': float(np.abs(k_hp).max())},
        'mean_abs_diff_a_slotmean_vs_ha0_ext_pose': float(np.abs(a_slot - a_hp[:, None]).mean()),
        'mean_abs_diff_kappa_slotmean_vs_ha0_ext_pose': float(np.abs(k_slot - k_hp[:, None]).mean())}
    row['seconds'] = round(time.time() - t0, 1)
    res[tag] = row
    print(tag, json.dumps({k: row[k] for k in ('fidelity_ADE6_P10u_vs_P10_m', 'ade_vs_gt_m', 'paired_vs_P10u')}, indent=1)[:2600])
if OUT:
    json.dump(res, open(OUT, 'w'), indent=1)
    print('wrote', OUT)
