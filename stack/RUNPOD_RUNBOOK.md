# RunPod Runbook — real-data training of TanitAD-4B-M (261 M)

**Run ID:** `p0-sB01-base250cam-comma` · **Budget:** planned $25 (ledger) · **Owner:** Sayed
*(Updated per D-009: trains on comma2k19 real camera data — the toy pipeline command was retired.)*

## 1. Create the pod (web UI, ~3 min)

1. runpod.io → *Pods* → *Deploy*.
2. GPU: **1× A40 48 GB** (Secure Cloud ~$0.85/h or Community ~$0.40/h — Community is fine, we
   checkpoint every 30 min). Fallback if no A40: A6000 48 GB; A100 80 GB is overkill for Stage A.
3. Template: **RunPod PyTorch 2.x** (any image with torch ≥ 2.4 + CUDA 12.x).
4. Volume: **100 GB** persistent at `/workspace` (checkpoints + data live here, survive pod stop).
5. Deploy → open **Web Terminal** (or SSH).

## 2. Setup (~5 min, copy-paste block)

```bash
cd /workspace
git clone https://github.com/sayoood/TanitAD.git
cd TanitAD/stack
pip install -e .[dev]
pytest -q                      # must be green (25 tests) before spending GPU time
```

## 3. Get real data onto the pod (D-009: comma2k19, ungated — no token needed)

```bash
pip install -e .[real]
python - <<'EOF'
from huggingface_hub import hf_hub_download
for i in (1, 2, 3):        # 3 chunks ~ 27 GB ~ 10 h of driving; add more later
    p = hf_hub_download('commaai/comma2k19', f'raw_data/Chunk_{i}.zip',
                        repo_type='dataset', local_dir='/workspace/data')
    print('got', p)
EOF
for z in /workspace/data/raw_data/Chunk_*.zip; do unzip -q "$z" -d /workspace/data/comma2k19; done
```

## 4. Launch the real-data Stage run (in tmux so the browser can disconnect)

```bash
tmux new -s train
cd /workspace/TanitAD/stack
python -m tanitad.train.train_worldmodel \
    --config base250cam \
    --data comma2k19 --data-root /workspace/data/comma2k19 \
    --episodes 600 \
    --out /workspace/experiments/p0-sB01-base250cam-comma \
    2>&1 | tee /workspace/experiments/p0-sB01.log
```
Detach: `Ctrl-b d`. Re-attach: `tmux attach -t train`.

- Expected: ~18–30 h for 60 k steps at batch 64 on 256 px inputs (fp32 today; bf16 autocast lands
  this week and roughly halves it). Budget accordingly or cut `--steps 30000` for the first pass.

## 4. Monitor

```bash
tail -f /workspace/experiments/p0-sA02.log        # loss lines are JSON
```
Healthy run: `loss` falling, `sigreg` falling toward ~O(1), `h15` falling, no NaN. The final
`metrics.json` must contain the I2/I4 instrument rows and the D9 rows — a run without them does not
exist for decision-making (D-004).

## 5. Retrieve results (metrics + checkpoint)

```bash
# metrics + config back into the repo (small, committed):
cp -r /workspace/experiments/p0-sA02-base250-stageA /workspace/TanitAD/stack/experiments/
cd /workspace/TanitAD && git add stack/experiments && \
  git -c user.name=sayoood -c user.email=sayedbouzouraa@googlemail.com \
  commit -m "exp(p0-sA02): Stage-A base250 run — metrics" && git push
# checkpoint (large): keep on the volume, or push to HF hub:
# huggingface-cli upload Sayood/tanitad-checkpoints /workspace/experiments/p0-sA02-base250-stageA/model.pt
```
(`model.pt` is gitignored by design — never push checkpoints to GitHub.)

## 6. Shut down

**Stop the pod** when the run finishes (volume persists; billing for GPU stops). Enter the actual $
into `Project Steering/RESOURCE_LEDGER.md`.
