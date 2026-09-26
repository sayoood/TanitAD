#!/bin/bash
# REFe pod bootstrap + preflight. Run this FIRST on a fresh A40 pod, before anything else.
#
# It does four things, in this order, and REFUSES to continue when one of them fails:
#   1. preflight the environment (the traps below have each cost this programme hours)
#   2. re-time the camera scaling, which REPLACES the only ESTIMATED term in TRAINING_TIME.md
#   3. report the largest batch that is genuinely RESIDENT on this card
#   4. pull the four camera channels for the chosen split, into CONTAINERS
#
# ⛔ MARKERS ARE OPAQUE ON PURPOSE. A monitor grepping a PTY for the same words its own command
# contains matches the ECHOED COMMAND LINE and reports a failure that never happened -- MEASURED
# three times in this programme. Every marker here is of the form ZZ<name>|<payload>ZZ, and a
# watcher should parse ZZ...ZZ rather than grep for English.
#
# Usage:  bash pod_bootstrap.sh [--split mini|trainval|test] [--skip-data]
set -u

REPO="${REPO:-/workspace/TanitAD}"
PKG="$REPO/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
PY="${PY:-python3}"
SPLIT="mini"
SKIP_DATA=0
while [ $# -gt 0 ]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2 ;;
    --skip-data) SKIP_DATA=1; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
fail() { echo "ZZPREFLIGHT|FAIL:$1ZZ"; exit 1; }

echo "=== 1. preflight ==================================================="

# ⛔ PYTHONPATH IS REQUIRED, cd IS NOT ENOUGH. Trainers die with ModuleNotFound: tanitad otherwise,
# and the death looks like a broken suite rather than a broken path.
export PYTHONPATH="$REPO/stack:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"   # torch spawns ~113 threads PER PROCESS; concurrent
                                                 # arms then sit at 0-6 % sm and look exactly like a hang

# ⛔ VERIFY CUDA WITH A REAL conv2d, NOT WITH `import torch`. cuBLAS/matmul can succeed while
# cuDNN/conv is broken -- which is what a torch/driver mismatch actually looks like.
"$PY" - <<'EOF' || fail "cuda_conv2d"
import sys, torch
print(f"  torch {torch.__version__}  cuda {torch.version.cuda}  available {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit(1)
d = torch.cuda.get_device_properties(0)
print(f"  device {d.name}  {d.total_memory/2**30:.2f} GiB  {d.multi_processor_count} SMs")
x = torch.randn(1, 3, 64, 64, device="cuda")
w = torch.randn(8, 3, 3, 3, device="cuda")
y = torch.nn.functional.conv2d(x, w)              # the real test: cuDNN, not just cuBLAS
torch.cuda.synchronize()
print(f"  conv2d OK, out {tuple(y.shape)}")
EOF
echo "ZZCUDA|okZZ"

# ⛔ NEVER JUDGE POD DISK WITH df -- it reports the whole cluster and hides the per-pod quota.
# A full quota killed a flagship mid-checkpoint. Use a real write test.
DD_DIR="${DD_DIR:-/workspace/_ddtest}"
mkdir -p "$DD_DIR" || fail "mkdir_$DD_DIR"
if dd if=/dev/zero of="$DD_DIR/probe.bin" bs=1M count=2048 conv=fsync 2>/dev/null; then
  echo "ZZDISK|wrote_2GiBZZ"
else
  rm -f "$DD_DIR/probe.bin"; fail "disk_quota_under_2GiB"
fi
rm -f "$DD_DIR/probe.bin"

# ⚠️ CONTAINERS ARE MANDATORY ONLY WHERE THE ALLOCATION UNIT IS LARGE. On D: (exFAT, 1 MiB clusters)
# loose JPEGs inflate 5.0x MEASURED. CHECK, do not assume it is a D:-only problem -- and do not
# assume it binds here either.
BS=$(stat -f -c %s "$DD_DIR" 2>/dev/null || echo "unknown")
echo "ZZALLOCUNIT|${BS}ZZ"
[ "$BS" != "unknown" ] && [ "$BS" -gt 65536 ] 2>/dev/null && \
  echo "  allocation unit ${BS} B > 64 KiB -- USE --container for every camera fetch"

# the model must import and build before anything expensive is started
cd "$PKG/refe" || fail "no_package_at_$PKG"
"$PY" - <<'EOF' || fail "model_import"
import sys; sys.path.insert(0, ".")
import model as M
cfg = M.REFeConfig()
m = M.REFe(cfg)
r = M.param_report(m)
print(f"  REFe {r['total']:,} total / {r['trainable']:,} trainable ({r['pct']:.2f} %)")
print(f"  ego_dim {cfg.ego_dim}  n_cameras {cfg.n_cameras}  undistort {cfg.undistort}")
EOF
echo "ZZMODEL|okZZ"

# ⭐ THE CONSUMER-CONFORMANCE ARM IS A PRE-LAUNCH GATE, NOT A NICETY. It is the one check that can
# see the model, the bank, the trainer and the planner disagreeing -- the failure that wasted a
# whole evening on the dev box while every other guard stayed green.
"$PY" diag_consumer_conformance.py --self-test >/tmp/conf.txt 2>&1
CONF=$?
tail -3 /tmp/conf.txt
# ⛔ READ THE ARTIFACT, NOT THE STATUS. $? through a pipe is the LAST element's status, and a
# timed-out gate with no output has reported GATE_EXIT=0 in this programme before.
grep -q "SELF_TEST_OK" /tmp/conf.txt || fail "conformance_selftest_did_not_run"
[ $CONF -eq 0 ] || echo "  ZZCONFORM|diverged_rebuild_the_bankZZ"
[ $CONF -eq 0 ] && echo "ZZCONFORM|okZZ"

echo ""
echo "=== 2. re-time the camera scaling (replaces the ESTIMATED A40 multiplier) ==="
# TRAINING_TIME.md's only estimated term is the A40/4060 ratio. This is the three-minute job that
# removes it. Its own guard refuses to quote any config that is not resident.
"$PY" "$PKG/raw/2026-09-21-camera-scaling/cam_scaling.py" 2>&1 | tee /tmp/scaling.txt
grep -q "vitl16" /tmp/scaling.txt && echo "ZZSCALING|okZZ" || echo "ZZSCALING|incompleteZZ"

echo ""
echo "=== 3. largest RESIDENT batch on this card ==========================="
"$PY" - <<'EOF'
import sys, torch
sys.path.insert(0, ".")
import model as M
if not torch.cuda.is_available():
    print("  no CUDA"); raise SystemExit(0)
total = torch.cuda.get_device_properties(0).total_memory / 2**30
cfg = M.REFeConfig()
net = M.REFe(cfg).cuda(); net.train()
opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4)
best = 0
for B in (1, 2, 4, 6, 8, 12, 16):
    try:
        torch.cuda.reset_peak_memory_stats()
        img = torch.randn(B, cfg.n_cameras, 3, cfg.img_h, cfg.img_w, device="cuda")
        ego = torch.randn(B, cfg.ego_dim, device="cuda")
        goal = torch.randn(B, 2 * cfg.n_goal_points, device="cuda")
        tgt = torch.randn(B, cfg.horizon_steps, cfg.traj_dim, device="cuda")
        opt.zero_grad(set_to_none=True)
        traj, score = net(img, ego, goal)
        loss, _ = M.wta_loss(traj, tgt)
        (loss + score.sum() * 0.0).backward(); opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        # ⛔ "it ran" IS NOT "it fits". Size the batch by MEASURED peak against the card, never by
        # absence of an OOM -- a batch that spills to host RAM runs and is ~34x slower.
        ok = peak < total - 2.0
        print(f"  batch {B:3d}  peak {peak:6.2f} GiB / {total:.2f}  {'RESIDENT' if ok else 'TOO CLOSE -- do not use'}")
        if ok:
            best = B
        else:
            break
        del img, ego, goal, tgt, traj, score, loss
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(f"  batch {B:3d}  OOM"); torch.cuda.empty_cache(); break
print(f"ZZBATCH|{best}ZZ")
# ⭐ ACCUMULATION IS NOT OPTIONAL. WTA gives gradient to ONE proposal per SAMPLE, so batch B touches
# at most B of 64 proposals per step. Their effective batch was 256.
if best:
    print(f"  => train with --batch {best} --accum {max(1, round(256/best))}  (effective 256, the paper's)")
EOF

if [ "$SKIP_DATA" -eq 1 ]; then
  echo ""; echo "ZZBOOTSTRAP|done_no_dataZZ"; exit 0
fi

echo ""
echo "=== 4. pull the FOUR camera channels for split '$SPLIT' ============="
# ⚠️ The ranged fetcher pulls only the named channels: MEASURED 23.39 GB = 48.1 % of a 48.63 GB
# archive for four cameras, against 12.2 % for one. Whole-zip downloads waste the rest.
#
# ⛔ IF A TARGET BANK EXISTS, DERIVE THE LOG LIST FROM IT -- NEVER HAND-WRITE ONE. MEASURED
# 2026-09-21 on the dev box: a hand-written six-log list covered 6 of the **7** logs the bank
# actually referenced, leaving 109 of 1,091 tuples (10.0 %) with no side/rear cameras. The missed
# log was a different SEGMENT of the same recording session as one that WAS listed, so the two read
# almost identically and the omission survived. A list is a copy of a fact and drifts from it; the
# bank IS the fact. `--logs-from-bank` REFUSES on an empty bank (exit 4) rather than falling back
# to fetching everything.
B_URL="https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/${SPLIT}_set"
OUT="${CAM_OUT:-/workspace/data/nuplan-camera/$SPLIT}"
BANK="${BANK:-/workspace/data/refe_targets_4cam}"
mkdir -p "$OUT"
SEL=()
if [ -d "$BANK" ] && [ -n "$(ls "$BANK"/*.jsonl 2>/dev/null)" ]; then
  SEL=(--logs-from-bank "$BANK")
  echo "  restricting the fetch to the logs in $BANK"
else
  echo "  no bank at $BANK -- fetching the WHOLE split (this is the expensive path)"
fi
n_ok=0
for i in 0 1 2 3 4 5 6 7 8; do
  U="$B_URL/nuplan-v1.1_${SPLIT}_camera_${i}.zip"
  if "$PY" "$PKG/code/fetch_front_camera.py" --container \
       --cameras CAM_F0,CAM_L0,CAM_R0,CAM_B0 "${SEL[@]}" "$U" "$OUT"; then
    n_ok=$((n_ok+1)); echo "ZZARCHIVE|${i}:okZZ"
  else
    echo "ZZARCHIVE|${i}:failZZ"
  fi
done
# ⛔ AND VERIFY BY CONTENT, NOT BY COUNT: every tuple must resolve ALL FOUR cameras and decode.
# A container that exists and a frame that decodes are different claims.
"$PY" - <<'EOF'
import glob, json, os, sys
sys.path.insert(0, os.environ.get("REFE_DIR", "."))
bank = os.environ.get("BANK", "/workspace/data/refe_targets_4cam")
have = {os.path.basename(p).rsplit("_CAM_", 1)[0]
        for p in glob.glob(os.path.join(os.environ.get("CAM_OUT", ""), "*.zip"))}
need = set()
for p in glob.glob(os.path.join(bank, "*.jsonl")):
    if "stats" in os.path.basename(p):
        continue
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                need.add(json.loads(line)["log_name"])
miss = sorted(need - have) if have else sorted(need)
print(f"ZZLOGS|need={len(need)}_have={len(have)}_missing={len(miss)}ZZ")
for m in miss[:10]:
    print("  MISSING", m)
EOF
echo "ZZFETCH|${n_ok}_archives_${SPLIT}ZZ"
echo "ZZBOOTSTRAP|doneZZ"
