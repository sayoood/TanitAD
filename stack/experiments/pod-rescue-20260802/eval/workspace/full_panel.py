"""FULL BINDING PANEL — all four families, for every available arm.

PI 2026-08-02: "I'm missing the tactical and strategic metrics and its evaluations."
The hierarchy pass DOES traverse the brains, so tactical+strategic come from hierarchy.run
instead of reporting UNAVAILABLE.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, '/workspace/TanitAD/taniteval')
sys.path.insert(0, '/workspace/TanitAD/stack')
import taniteval.runner as R
from taniteval import registry, rollout, four_families as FF

R.VAL = Path('/root/valdata/val19_leakfree')
RES = Path('/root/taniteval/results')
BASE = dict(family='TanitAD', config='flagship4b', encoder='trained ViT-12 (9ch, 256px)',
            encoder_frozen=False, speed_input=True, action_dim=3, anti_collapse='SigReg-64')
registry.MODELS.append(dict(BASE, arch='flagship-worldmodel', key='v1-lf19',
                            ckpt='/workspace/v1_modelonly.pt', name='v1 30k'))
registry.MODELS.append(dict(BASE, arch='flagship-worldmodel-v2', key='v2corpus-lf19',
                            ckpt='/workspace/v2corpus_modelonly.pt', name='v2corpus 30k',
                            run_config='/workspace/v2corpus_config.json'))

for k in ('v1-lf19', 'v2corpus-lf19'):
    print(f'######## {k} ########', flush=True)
    hier = None
    try:
        hier = R.run_hierarchy(k, episodes=19)
    except Exception as e:
        print(f'[hier] {k} FAILED: {type(e).__name__}: {e}', flush=True)
    w = rollout.load_windows(RES / f'windows_{k}.pt')
    fam = FF.all_families(w, hier)
    (RES / f'fourfam_{k}.json').write_text(json.dumps(fam, indent=2, default=str))
    t, s = fam['tactical'], fam['strategic']
    print(f'[TACTICAL ] status={t.get("status")} kappa={t.get("maneuver_vs_trajectory_kappa")} '
          f'verdict={t.get("maneuver_consistency_verdict")} seams={t.get("seams_beneficial_of_3")}/3', flush=True)
    print(f'[STRATEGIC] status={s.get("status")} route_nav={s.get("route_acc_nav")} '
          f'route_follow={s.get("route_acc_follow")} majority={s.get("majority_straight_rate")} '
          f'beats_majority={s.get("beats_majority_baseline")}', flush=True)
    print(f'[UNAVAILABLE] {fam["_families_unavailable"]}', flush=True)
