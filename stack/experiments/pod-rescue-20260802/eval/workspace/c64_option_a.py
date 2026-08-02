"""C64 option A — v2corpus vs v1 on the 19 LEAK-FREE canonical val episodes.

21 of the 40 canonical val episodes are inside v2corpus's training corpus (C64), so the full-40
surface is VOID for this arm. Both arms are scored on the SAME 19 survivors — which is why v1 is
RE-SCORED here rather than compared against its published full-40 0.4271.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, '/workspace/TanitAD/taniteval')
sys.path.insert(0, '/workspace/TanitAD/stack')

import taniteval.runner as R
from taniteval import registry

R.VAL = Path('/root/valdata/val19_leakfree')

BASE = dict(family='TanitAD', arch='flagship-worldmodel', config='flagship4b',
            encoder='trained ViT-12 (9ch, 256px)', encoder_frozen=False,
            speed_input=True, action_dim=3, anti_collapse='SigReg-64')
registry.MODELS.append(dict(BASE, key='v1-lf19', name='v1 speedjerk 30k (leak-free 19)',
                            ckpt='/workspace/v1_modelonly.pt',
                            note='RE-SCORED on the 19; NOT the published full-40 0.4271'))
registry.MODELS.append(dict(BASE, arch='flagship-worldmodel-v2', key='v2corpus-lf19', run_config='/workspace/v2corpus_config.json', name='v2corpus 30k (leak-free 19)',
                            ckpt='/workspace/v2corpus_modelonly.pt',
                            note='trained on physicalai-v2bal; --v2 lever pack, rollout_k 12'))

for k in ('v1-lf19', 'v2corpus-lf19'):
    print(f'=== scoring {k} on 19 leak-free episodes ===', flush=True)
    r = R.run_one(k, episodes=19)
    if r is None:
        print(f'{k}: NO TRAJECTORY SURFACE', flush=True)

print('=== paired comparison ===', flush=True)
R.run_ab('v1-lf19', 'v2corpus-lf19')
