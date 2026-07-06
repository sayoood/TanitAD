# RunPod Runbook v3 — real-data training of TanitAD-4B-M (261 M, D-015 input)

**Run ID:** `p0-sB01-realmix` · **Budget:** planned $25–35 (ledger) · **Owner:** Sayed
Data tags: `data:comma2k19, data:physicalai` (D-012 — PhysicalAI exposure auditable, no public
claims from it until license resolution).

## 0. What this run is

First learning-valid training of the 261 M stack: batch 64 (SigReg n=1024 — the F-2 lesson),
**D-015 input** (3 RGB frames @ 100 ms = 9 channels + aligned actions), real mix of comma2k19
highway (~40 %) + PhysicalAI-AV urban R0 (~60 %). Watch the live collapse-health rows.

## 1. Create the pod (~3 min)

1. runpod.io → Pods → Deploy → **1× A40 48 GB** (Community ~$0.40/h fine; A6000 fallback).
2. Template: RunPod PyTorch 2.x (torch ≥ 2.4, CUDA 12.x). Volume: **150 GB** at `/workspace`.
3. Open the web terminal.

## 2. Setup (~5 min)

```bash
cd /workspace
git clone https://github.com/sayoood/TanitAD.git
cd TanitAD/stack
pip install -e .[dev,real]
pytest -q                                  # must be green before GPU spend
export HF_TOKEN=hf_XXXX                    # your (rotated) token — PhysicalAI is gated
export TANITAD_PHYSICALAI_ROOT=/workspace/data/physicalai
```

## 3. Data — comma2k19 (~15 min on DC bandwidth)

```bash
mkdir -p /workspace/data && python - <<'EOF'
from huggingface_hub import hf_hub_download
for i in (1, 2, 3):
    print(hf_hub_download('commaai/comma2k19', f'raw_data/Chunk_{i}.zip',
                          repo_type='dataset', local_dir='/workspace/data'))
EOF
# python -m zipfile: RunPod containers ship without the `unzip` binary
for z in /workspace/data/raw_data/Chunk_*.zip; do python -m zipfile -e "$z" /workspace/data/comma2k19/ && rm "$z"; done
```

## 4. Data — PhysicalAI-AV R0 (urban 500 clips; ~30–60 min on DC bandwidth)

```bash
cd /workspace/TanitAD/stack
python - <<'EOF'
from huggingface_hub import hf_hub_download
for f in ('clip_index.parquet', 'metadata/data_collection.parquet'):
    hf_hub_download('nvidia/PhysicalAI-Autonomous-Vehicles', f,
                    repo_type='dataset', local_dir='/workspace/data/physicalai')
EOF
python scripts/physicalai_r0.py select --chunks 30 --target 500
python scripts/physicalai_r0.py fetch-camera --max-chunks 30   # ~60 GB transient, zips auto-deleted
```

## 5. Sanity: visualize frames + actions before training (2 min — always)

```bash
SEG=$(find /workspace/data/comma2k19 -name video.hevc | head -1 | xargs dirname)
python scripts/visualize_episode.py --source comma2k19 --path "$SEG" --out /workspace/viz_comma.mp4
python scripts/visualize_episode.py --source physicalai --path /workspace/data/physicalai --out /workspace/viz_phys.mp4
# download both MP4s via the RunPod file browser and EYEBALL them: steering dial,
# accel bar and trajectory inset must match what the video shows (F-3 lesson).
```

## 6. Launch training (survives disconnects; RAM-sized)

```bash
apt-get update -qq && apt-get install -y -qq tmux   # containers ship without it
mkdir -p /workspace/experiments                     # tee needs the dir to exist
free -g                                             # note the available RAM
```
**RAM sizing (episodes are held in memory as uint8):** ≈ 0.18 GB per comma segment +
0.12 GB per PhysicalAI clip. `--episodes N` caps EACH corpus. Rule of thumb:
~50 GB RAM pod → `--episodes 120`; ~100 GB → `--episodes 250`; ≥180 GB → `--episodes 500`.
(Disk-backed episode cache is on the DataEng backlog to lift this cap.)

**GPU sizing (F-5, measured):** naive batch 64 OOMs the 48 GB A40 (the first encoder pass holds
512 frames of activations). Use **micro-batch 16 × accumulation 4** (effective 64) with
`--grad-checkpoint`. SigReg runs per micro-batch: micro 16 × window 8 × 2 = 256 samples/step —
at the healthy floor; do not go below micro 16 without raising `--accum` awareness.

```bash
EPISODES=500 bash scripts/pod_launch.sh      # does all of the below in tmux
# equivalent manual command:
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   python -u -m tanitad.train.train_worldmodel --config base250cam --data realmix \
#     --data-root /workspace/data/comma2k19 --sim-root /workspace/data/physicalai \
#     --sim-frac 0.6 --episodes 500 --batch-size 16 --accum 4 --grad-checkpoint \
#     --out /workspace/experiments/p0-sB01-realmix 2>&1 | tee /workspace/experiments/p0-sB01-realmix.log
```
Attach `tmux attach -t train`; detach `Ctrl-b d`. Expected wall-clock rises ~30 % from
checkpointing; `--steps 30000` for a cheaper first pass. Dataset build (video decode) prints
`building episodes i/N` progress and takes ~10–45 min depending on `--episodes`.

## 7. What healthy looks like (check after ~30 min)

```bash
tail -f /workspace/experiments/p0-sB01.log
```
- `erank` RISING over the first thousands of steps (batch-2 collapse was 23/2048 — F-2)
- `dim_std` moving toward ~1.0 (LeJEPA isotropic Gaussian target)
- `step_ratio` well above 0.007 (the collapsed value)
- `sigreg` falling, `pred`/`tac`/`h15` falling, no NaN
Final `metrics.json` must contain I2 (pass), I4, and the D9 imagination rows — no rows, no run.

## 8. Results back / shutdown

```bash
cp -r /workspace/experiments/p0-sB01-realmix /workspace/TanitAD/stack/experiments/
cd /workspace/TanitAD && git add stack/experiments && \
  git -c user.name=sayoood -c user.email=sayedbouzouraa@googlemail.com \
  commit -m "exp(p0-sB01): realmix base250cam run — metrics + health rows" && git push
```
(`model.pt` stays on the volume — gitignored by design.) **Stop the pod**; enter actual $ into
`Project Steering/RESOURCE_LEDGER.md`.
