"""Long, high-quality v1-vs-v2corpus overlays on the canonical val, for VISUAL assessment.

⚠️ EVERY CLIP IS TAGGED LEAKFREE or INTRAIN. 21 of the 40 canonical val episodes are inside
v2corpus's TRAINING corpus (C64). An untagged clip invites exactly the wrong read — v2corpus
looking good on an episode it memorised is not generalisation.
"""
import sys
sys.path.insert(0, '/workspace/TanitAD/taniteval')
sys.path.insert(0, '/workspace/TanitAD/stack')
from taniteval import registry

BASE = dict(family='TanitAD', config='flagship4b', encoder='trained ViT-12 (9ch, 256px)',
            encoder_frozen=False, speed_input=True, action_dim=3, anti_collapse='SigReg-64')
registry.MODELS.append(dict(BASE, arch='flagship-worldmodel', key='v1',
                            name='v1 speedjerk 30k', ckpt='/workspace/v1_modelonly.pt'))
registry.MODELS.append(dict(BASE, arch='flagship-worldmodel-v2', key='v2corpus',
                            name='v2corpus 30k', ckpt='/workspace/v2corpus_modelonly.pt',
                            run_config='/workspace/v2corpus_config.json'))

LEAKFREE = {0,2,3,4,6,8,12,13,15,16,17,22,24,29,31,32,33,35,36}
PICK = [0, 3, 8, 31, 1, 11]          # 4 leak-free + 2 in-train, deliberately mixed
clips = ','.join(f"{i}:{'LEAKFREE' if i in LEAKFREE else 'INTRAIN'}-ep{i:02d}" for i in PICK)
print('[clips]', clips, flush=True)

sys.argv = ['corpus_overlay', '--corpus', 'physicalai', '--models', 'v1,v2corpus',
            '--clips', clips, '--max-frames', '600', '--fps', '10']
from taniteval import corpus_overlay
corpus_overlay.main()
