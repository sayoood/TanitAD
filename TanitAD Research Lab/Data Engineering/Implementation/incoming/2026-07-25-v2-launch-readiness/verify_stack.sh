set -u
cd /workspace/TanitAD/stack
echo "=== $(hostname) ==="
PY=python3
echo "--- 1. --v2-cache flag present ---"
grep -c 'v2-cache' scripts/train_flagship4b.py
echo "--- 2. v2_dataset.py present ---"
ls -l tanitad/data/v2_dataset.py
echo "--- 3. refb_labels line count + md5 ---"
wc -l < scripts/refb_labels.py; md5sum scripts/refb_labels.py
echo "--- 4. FULL import of the trainer module ---"
PYTHONPATH=/workspace/TanitAD/stack $PY -c "
import sys; sys.path.insert(0,'scripts')
import importlib.util, torch, torchvision
spec = importlib.util.spec_from_file_location('train_flagship4b','scripts/train_flagship4b.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('IMPORT_TRAINER_OK torch',torch.__version__,'tv',torchvision.__version__)
p = m.build_argparser() if hasattr(m,'build_argparser') else None
import subprocess
" 2>&1 | tail -5
echo "--- 5. --v2-cache accepted by the real CLI parser (--help) ---"
PYTHONPATH=/workspace/TanitAD/stack $PY scripts/train_flagship4b.py --help 2>&1 | grep -A2 -- '--v2-cache' | head -6
echo "--- 6. v2_dataset imports + decode path ---"
PYTHONPATH=/workspace/TanitAD/stack $PY -c "
from tanitad.data.v2_dataset import build_v2_providers, decode_full_episode, V2CompressedCache
import torchvision.io as tvio
print('V2_DATASET_IMPORT_OK; decode_jpeg present:', hasattr(tvio,'decode_jpeg'))
" 2>&1 | tail -3
echo "--- 7. refb_labels route_from_future_v21 present ---"
PYTHONPATH=/workspace/TanitAD/stack $PY -c "
import sys; sys.path.insert(0,'scripts')
import refb_labels as R
print('HAS route_from_future_v21:', hasattr(R,'route_from_future_v21'))
print('HAS maneuver_labels:', hasattr(R,'maneuver_labels'), '| maneuver_labels_v2:', hasattr(R,'maneuver_labels_v2'))
" 2>&1 | tail -3
echo "--- 8. taniteval.ci import ---"
PYTHONPATH=/workspace/TanitAD/taniteval $PY -c "import taniteval.ci as c; print('TANITEVAL_CI_OK', [f for f in dir(c) if 'boot' in f.lower()][:5])" 2>&1 | tail -2
