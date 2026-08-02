"""RR-20 vs RR-CTL — does training on longer recursive rollouts reduce compounding?

E-CR returned H-COMPOUND (CR 3.50 -> 80.77, teacher-forced arm FLAT), which prescribes training on
prediction-corrupted histories. RR-20 fine-tuned v1 for 2000 steps at --rollout-k 20; RR-CTL is the
SAME 2000 steps at --rollout-k 4. Same pod, same seed, one flag apart.

⛔ The comparison is RR-20 vs RR-CTL, NEVER RR-20 vs v1 — against v1 you would be measuring
'rollout-k 20' PLUS '2000 extra fine-tune steps' with no way to separate them.

⛔ BINDING (PI 2026-08-02): report the FOUR FAMILIES, not ADE alone.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, '/workspace/TanitAD/taniteval')
sys.path.insert(0, '/workspace/TanitAD/stack')
import taniteval.runner as R
from taniteval import registry, rollout, four_families as FF

R.VAL = Path('/workspace/val40cache')
BASE = dict(family='TanitAD', arch='flagship-worldmodel', config='flagship4b',
            encoder='trained ViT-12 (9ch, 256px)', encoder_frozen=False,
            speed_input=True, action_dim=3, anti_collapse='SigReg-64')
registry.MODELS.append(dict(BASE, key='rr20', name='RR-20 (rollout-k 20 FT)',
                            ckpt='/workspace/rrft/ckpt.pt'))
registry.MODELS.append(dict(BASE, key='rrctl', name='RR-CTL (rollout-k 4 FT)',
                            ckpt='/workspace/rrctl/ckpt.pt'))

RES = Path('/root/taniteval/results')
for k in ('rrctl', 'rr20'):
    print(f'=== scoring {k} ===', flush=True)
    R.run_one(k, episodes=40)

print('=== paired RR-20 vs RR-CTL ===', flush=True)
R.run_ab('rrctl', 'rr20')

print('=== FOUR FAMILIES ===', flush=True)
for k in ('rrctl', 'rr20'):
    w = rollout.load_windows(RES / f'windows_{k}.pt')
    fam = FF.all_families(w)
    (RES / f'fourfam_{k}.json').write_text(json.dumps(fam, indent=2))
    lo, la = fam['longitudinal'], fam['lateral']
    print(f'[{k}] LON speed_mae={lo["speed_mae_mps"]} bias={lo["speed_bias_mps"]} '
          f'along_final_bias={lo["along_final_bias_m"]}', flush=True)
    print(f'[{k}] LAT head={la["heading_mae_deg"]}deg curv={la["curvature_mae_1pm"]} '
          f'yaw={la["yaw_rate_mae_degps"]}degps cross={la["cross_mae_m"]}m', flush=True)
    print(f'[{k}] unavailable={fam["_families_unavailable"]}', flush=True)
