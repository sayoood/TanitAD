#!/bin/bash
# Wait for the Alpamayo batch, then: reconstruct+validate GT -> compare -> video.
export PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts:/workspace/TanitAD/taniteval:/workspace
export TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack
export MPLCONFIGDIR=/tmp/a2_mpl
O=/workspace/a2_batch_out
while pgrep -f a2_batch.py > /dev/null; do sleep 30; done
cd /workspace
python3 -u a2_gt_from_ego.py --jsonl $O/alpamayo_oodval.jsonl --traj-dir $O \
  --ego-dir /workspace/pai_build/labels/egomotion --out $O/alpamayo_gt.json
python3 -u a2_compare.py --alpamayo-jsonl $O/alpamayo_oodval.jsonl --traj-dir $O \
  --flagship-json $O/flagship_at_t0.json --alpamayo-gt $O/alpamayo_gt.json \
  --out $O/comparison.json
/workspace/a2venv/bin/python -u -c "import imageio_ffmpeg,os;os.environ.setdefault('X','1')" 2>/dev/null
python3 -u a2_compare_video.py --compare-json $O/comparison.json --traj-dir $O \
  --flagship-json $O/flagship_at_t0.json \
  --corpus /workspace/pai_epcache/physicalai-oodval-6f4b94e4c7ce-q90 \
  --out $O/alpamayo_vs_flagship.mp4 --seconds-per-clip 2.5
echo "A2_FINISH_DONE"
