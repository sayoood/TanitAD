'''SECOND PROBE on the clips the pre-registered admission rule dropped.

Pearson correlation is UNDEFINED on a constant series, so the admission statistic degenerates on
steady-speed clips. This probe re-aligns the dropped clips with a statistic that is defined there:
the minimum RMSE between the two ego-speed series over the same integer-lag search. A small RMSE
means the clip was aligned correctly and the ADMISSION STATISTIC failed, not the alignment.
CPU/IO only - no GPU, no model.
'''
import json, os, sys
import numpy as np, torch
CACHES = {'train': '/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894',
          'val': '/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11'}
A = json.load(open('/workspace/h2clf/feats/align_summary.json'))
meta = {m['k']: m for m in json.load(open('/workspace/h2clf/bundle/h2c_meta.json'))}
Z = np.load('/workspace/h2clf/bundle/h2c_labels.npz')
out = []
for a in A['per_clip']:
    if a['admitted']:
        continue
    k = a['k']; m = meta[k]
    gi = Z[f'c{k}_gi'].astype(np.int64); vr = Z[f'c{k}_ego_v'].astype(np.float64)
    nl = int(gi.max()) + 1; v_lab = np.full(nl, np.nan); v_lab[gi] = vr
    v_lab = np.nan_to_num(v_lab, nan=float(np.nanmean(v_lab)))
    ep = torch.load(os.path.join(CACHES[m['cache']], m['file']), map_location='cpu',
                    weights_only=True, mmap=True)
    v_ep = ep['poses'][:, 3].to(torch.float64).numpy()
    best = (0, 1e9)
    for L in range(-80, 81):
        g0, g1 = max(0, L), min(nl, len(v_ep) + L)
        if g1 - g0 < 60: continue
        r = float(np.sqrt(np.mean((v_lab[g0:g1] - v_ep[g0-L:g1-L])**2)))
        if r < best[1]: best = (L, r)
    out.append({'k': k, 'side': m['side'], 'corr': a['corr'], 'lag_corr': a['lag'],
                'lag_rmse': best[0], 'rmse_mps': round(best[1], 5),
                'v_std': round(float(vr.std()), 4), 'v_mean': round(float(vr.mean()), 3)})
r = np.array([o['rmse_mps'] for o in out])
print(json.dumps({'n_dropped_probed': len(out),
                  'rmse_quantiles': {q: round(float(np.quantile(r, float(q))), 5)
                                     for q in ('0.0','0.25','0.5','0.75','0.95','1.0')},
                  'n_rmse_le_0.10': int((r <= 0.10).sum()),
                  'n_rmse_le_0.25': int((r <= 0.25).sum()),
                  'n_rmse_gt_1.0': int((r > 1.0).sum()),
                  'per_clip': out}, indent=2))
