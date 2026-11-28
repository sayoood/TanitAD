#!/bin/bash
# REFe TRAINING on the pod -- run AFTER code/pod_dataprep.sh has finished (its ZZDONE line), on the
# same pod (PI decision 2026-09-23: ONE pod, data prep then training).
#
#   STAGES (select with STAGES="...", default all, in order)
#     weights    DINOv3 ViT-L + ViT-S from timm's UNGATED mirror, at PINNED commits, sha256-verified
#                (ViT-S only feeds the CPU resume test below)
#     preflight  the four training-bank files + the camera-rig table and its guard (R22) + 32/32 pixel
#                shards + host RAM + a real disk write
#     resume     diag_train_resume.py on CPU: halt + resume must be BIT-IDENTICAL to an uninterrupted
#                run, and its two mutation arms must go RED (~6 min). The run is only as safe as this.
#     batch      the largest RESIDENT batch (budget = free memory, next size predicted before it is
#                tried, spill detected by time) -> ACCUM = 256 / BATCH; FIXED for the run's lifetime
#     smoke      ONE real step on the FULL bank: every tuple's 4 frames resolve + decode, every tuple
#                carries its own camera rig, and it trains
#     train      the run, under a self-healing supervisor that stops on summary.json {"done": true}
#
# ⭐ GROW=1 (PI 2026-09-24: "parallelize training and data prep"): train WHILE the augmentation search
#   and the scorer still run. An `assemble` stage builds $BANK/train_grow APPEND-ONLY from the shards
#   (code/grow_assemble.py, then a 30-min loop), and train.py --grow re-reads it at every epoch
#   boundary up to a byte snapshot kept in the checkpoint (refe/diag_grow.py proves resume stays
#   exact). Preflight then needs only rank 0; the resume stage runs diag_grow.py. The schedule is
#   sized for GROW_SCENES, the FINAL scene count, so early epochs are shorter than late ones.
#
# ⛔ LAUNCH IT DETACHED -- an SSH drop must not end a two-month run:
#     nohup setsid bash $WORK/refe-plan/code/pod_train.sh > $WORK/pod_train.out 2>&1 < /dev/null &
#   After ANY pod restart, run the SAME command again: every stage is idempotent and the train stage
#   resumes from ckpt_last.pt (at most --ckpt-every-min of work is lost).
#   Status:  tail -2 $RUN/metrics.jsonl ; cat $RUN/restarts.log ; ls $RUN/snap_epoch*.pt
#
# ⛔ NEVER `sed -i` THIS FILE WHILE IT RUNS -- bash reads scripts lazily by byte offset, and an in-place
#   edit makes a live supervisor execute garbage. Stop it, edit, relaunch (it resumes).
set -u
WORK="${WORK:-/workspace}"
PKG="${PKG:-$WORK/refe-plan}"
DATA="${DATA:-$WORK/data}"
BANK="$DATA/refe_navtrain"; PIX="$DATA/navtrain_pixels"; TRAIN_BANK="$BANK/train"
BACKBONE="${BACKBONE:-vitl16}"          # PI DECISION 2026-09-23: ViT-L (Table A13, 94.55 PDMS)
EPOCHS="${EPOCHS:-25}"                  # PAPER: 25 epochs
EPOCH_UNIT="${EPOCH_UNIT:-scenes}"      # PI 2026-09-23 (R24): an epoch = one pass over SCENES, the paper's
                                        # unit (Table A12) -- not over rows, which is 1.71x its budget
EFF="${EFF:-256}"                       # PAPER: batch 256 (theirs: 16 x H20 data parallel)
RUN="${RUN:-$DATA/refe_runs/${BACKBONE}_navtrain_aug_tau0.3}"
MAXR="${MAXR:-50}"
GROW="${GROW:-0}"
GROW_SCENES="${GROW_SCENES:-103039}"    # the final navtrain scene count the schedule is sized for
if [ "$GROW" = 1 ]; then
  TRAIN_BANK="$BANK/train_grow"
  STAGES="${STAGES:-weights assemble preflight resume batch smoke train}"
else
  STAGES="${STAGES:-weights preflight resume batch smoke train}"
fi
MIN_RAM_GB="${MIN_RAM_GB:-24}"          # MEASURED banks ~7.8 GiB at full scale + model/CUDA context
# SPEED (MEASURED on the first pod, A40, ViT-L, 4 cams, real step): FP32 1.282 s/sample -> bf16 0.366
# -> bf16 + compiled backbone 0.220 (5.8x). bf16 is a DECLARED departure from the paper's FP32,
# numerics-checked by refe/diag_amp.py; set AMP=none COMPILE=0 for the strict-FP32 arm.
AMP="${AMP:-bf16}"; TF32="${TF32:-1}"; COMPILE="${COMPILE:-1}"; WORKERS="${WORKERS:-2}"

# DINOv3 weights: timm's mirror is UNGATED (facebook/* is gated). Pinned to the commit AND the
# LFS sha256 that HuggingFace itself reports (X-Linked-ETag), which equal our dev-box copies.
declare -A W_REPO=([vitl16]=timm/vit_large_patch16_dinov3.lvd1689m [vits16]=timm/vit_small_patch16_dinov3.lvd1689m)
declare -A W_REV=([vitl16]=30c1109559f65dea34316b0d4842d35c5771fe11 [vits16]=3bf4720a82ec2066db88137180ff1f83a675cef0)
declare -A W_SHA=([vitl16]=45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941 [vits16]=2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040)

fail () { echo "ZZFAIL $* ZZ"; exit 1; }
has () { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }

[ -f "$WORK/teacher_env.sh" ] || fail "no teacher_env.sh -- run code/pod_teacher_env.sh first"
# shellcheck disable=SC1091
source "$WORK/teacher_env.sh"
PY="$DRIVERL_EVAL_PYTHON"
export REFE_BACKBONE_ROOT="$DATA/backbones"
export OMP_NUM_THREADS="${THREADS:-8}" MKL_NUM_THREADS="${THREADS:-8}" OPENBLAS_NUM_THREADS="${THREADS:-8}"
export PYTHONIOENCODING=utf-8
# a redirected Python stdout is BLOCK-buffered: MEASURED in the rehearsal, train.log stayed 0 B
# while the run was live. The run itself is unaffected (metrics.jsonl writes per row); a human
# tailing the log is not.
export PYTHONUNBUFFERED=1
mkdir -p "$RUN" "$REFE_BACKBONE_ROOT"
LOGS="$RUN/preflight"; mkdir -p "$LOGS"
echo "ZZSTART pod_train stages=[$STAGES] backbone=$BACKBONE epochs=$EPOCHS($EPOCH_UNIT) eff=$EFF grow=$GROW run=$RUN $(date -u +%FT%TZ) ZZ"
[ -n "${W_SHA[$BACKBONE]:-}" ] || fail "no pinned weights for backbone=$BACKBONE"

if has weights; then
  echo "=== weights ==="
  for bb in "$BACKBONE" vits16; do
    d="$REFE_BACKBONE_ROOT/dinov3-$bb"; mkdir -p "$d"
    url="https://huggingface.co/${W_REPO[$bb]}/resolve/${W_REV[$bb]}"
    if [ "$(sha256sum "$d/model.safetensors" 2>/dev/null | cut -d' ' -f1)" != "${W_SHA[$bb]}" ]; then
      curl -sSL --retry 10 --retry-all-errors -C - -o "$d/model.safetensors.part" "$url/model.safetensors" \
        || fail "weights download $bb"
      mv -f "$d/model.safetensors.part" "$d/model.safetensors"
      curl -sSL --retry 5 -o "$d/config.json" "$url/config.json" || true      # provenance only
    fi
    got="$(sha256sum "$d/model.safetensors" | cut -d' ' -f1)"
    # ⛔ assert on the ARTIFACT: a 64-char hash that equals the pin -- never on curl's exit code
    [ ${#got} -eq 64 ] && [ "$got" = "${W_SHA[$bb]}" ] || fail "weights $bb sha256 $got != ${W_SHA[$bb]}"
    echo "  $bb: sha256 ${got:0:12} == pin, $(stat -c %s "$d/model.safetensors") bytes"
  done
  echo "ZZOK weights ZZ"
fi

if has assemble; then
  echo "=== assemble: the GROWING bank, append-only ==="
  mkdir -p "$TRAIN_BANK"
  "$PY" "$PKG/code/grow_assemble.py" --bank "$BANK" --out "$TRAIN_BANK" > "$LOGS/assemble_first.log" 2>&1
  grep "ZZASSEMBLE_OK" "$LOGS/assemble_first.log" | cut -c1-300 || fail "assembler pass failed -- see $LOGS/assemble_first.log"
  # ONE assembler loop per bank: it keeps appending while data prep runs (closes fd 200, see train)
  if ! { [ -s "$TRAIN_BANK/.assembler.pid" ] && kill -0 "$(cat "$TRAIN_BANK/.assembler.pid")" 2>/dev/null; }; then
    nohup setsid "$PY" "$PKG/code/grow_assemble.py" --bank "$BANK" --out "$TRAIN_BANK" --loop-min 30 \
        >> "$TRAIN_BANK/assembler.log" 2>&1 < /dev/null 200>&- &
    echo $! > "$TRAIN_BANK/.assembler.pid"
  fi
  echo "ZZOK assemble (loop pid $(cat "$TRAIN_BANK/.assembler.pid")) ZZ"
fi

if has preflight; then
  echo "=== preflight ==="
  if [ "$GROW" = 1 ]; then
    # a growing bank needs rank 0 now; the twins and the scorer rows arrive while it trains
    [ -s "$TRAIN_BANK/targets_rank0.jsonl" ] || fail "missing or empty $TRAIN_BANK/targets_rank0.jsonl -- run the assemble stage"
    for f in targets_rank0.jsonl targets_rank1.jsonl scorer_targets.jsonl scorer_targets_rank1.jsonl; do
      echo "  $f: $([ -f "$TRAIN_BANK/$f" ] && wc -l < "$TRAIN_BANK/$f" || echo 0) rows (growing)"
    done
  else
  for f in targets_rank0.jsonl targets_rank1.jsonl scorer_targets.jsonl scorer_targets_rank1.jsonl; do
    [ -s "$TRAIN_BANK/$f" ] || fail "missing or empty $TRAIN_BANK/$f -- run pod_dataprep.sh (stage score)"
    echo "  $f: $(wc -l < "$TRAIN_BANK/$f") rows"
  done
  fi
  stray="$(ls "$TRAIN_BANK"/targets_*.jsonl | grep -v '/targets_rank[0-9]*\.jsonl$' || true)"
  [ -z "$stray" ] || fail "stray target files the trainer would refuse: $stray"
  # R22: each tuple's OWN camera rig (PETR). The table comes from pod_dataprep.sh's `calib` stage.
  [ -s "$TRAIN_BANK/calib_table.json" ] || fail "missing $TRAIN_BANK/calib_table.json -- run pod_dataprep.sh (stage calib, then score)"
  ( cd "$PKG/refe" && "$PY" diag_calib.py --table "$TRAIN_BANK/calib_table.json" ) > "$LOGS/diag_calib.log" 2>&1
  grep -E '^\s+\[' "$LOGS/diag_calib.log"
  grep -q "CALIB_OK" "$LOGS/diag_calib.log" || fail "calibration guard failed -- see $LOGS/diag_calib.log"
  ndone="$(ls "$PIX"/.done_* 2>/dev/null | wc -l)"
  [ "$ndone" -eq 32 ] || fail "pixels: $ndone/32 OpenScene shards extracted -- run pod_dataprep.sh (stage pix)"
  # host RAM from the CGROUP limit (a pod's `free` shows the host, not the container)
  lim="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)"
  if [ "$lim" != "max" ] && [ "$lim" -lt $((MIN_RAM_GB * 1024 * 1024 * 1024)) ] 2>/dev/null; then
    fail "container RAM limit $((lim / 1073741824)) GiB < $MIN_RAM_GB GiB"
  fi
  echo "  container RAM limit: $([ "$lim" = max ] && echo unlimited || echo "$((lim / 1073741824)) GiB")"
  # ⛔ NEVER JUDGE POD DISK WITH df (it shows the cluster, not the quota). A real 4 GiB write where
  # the checkpoints will live: ckpt_last ~0.23 GB x2 during an atomic swap, 25 snapshots ~1.9 GB,
  # model_final ~1.3 GB.
  dd if=/dev/zero of="$RUN/.ddtest" bs=64M count=64 conv=fsync status=none || { rm -f "$RUN/.ddtest"; fail "disk: 4 GiB write failed in $RUN"; }
  [ "$(stat -c %s "$RUN/.ddtest")" -eq 4294967296 ] || { rm -f "$RUN/.ddtest"; fail "disk: short write"; }
  rm -f "$RUN/.ddtest"
  echo "ZZOK preflight ZZ"
fi

cd "$PKG/refe" || fail "no $PKG/refe"

if has resume && [ "$GROW" = 1 ]; then
  echo "=== resume (GROW): a mid-epoch bank append + resume must equal the static run (CPU) ==="
  rm -rf "$DATA/refe_grow_test"
  "$PY" diag_grow.py --work "$DATA/refe_grow_test" > "$LOGS/grow_test.log" 2>&1
  grep -E '^\s+\[' "$LOGS/grow_test.log"
  grep -q "ZZGROW_EXACT_AND_GUARDS_LIVEZZ" "$LOGS/grow_test.log" || fail "grow test did not pass -- see $LOGS/grow_test.log"
  echo "ZZOK resume (grow) ZZ"
elif has resume; then
  echo "=== resume: halt + resume must be bit-identical (CPU, ~6 min) ==="
  rm -rf "$DATA/refe_resume_test"
  "$PY" diag_train_resume.py --bank "$TRAIN_BANK" --work "$DATA/refe_resume_test" --keep > "$LOGS/resume_test.log" 2>&1
  grep -E '^\s+\[' "$LOGS/resume_test.log"
  # ⛔ the artifact decides, not the exit status
  grep -q "RESUME_EXACT_AND_GUARDS_LIVE" "$LOGS/resume_test.log" || fail "resume test did not pass -- see $LOGS/resume_test.log"
  echo "ZZOK resume ZZ"
fi

if has batch; then
  echo "=== batch: largest RESIDENT batch, powers of two ==="
  if [ -n "${BATCH:-}" ]; then
    echo "  BATCH=$BATCH given -- probe skipped"
  elif [ -s "$RUN/batch_accum.txt" ]; then
    # ⛔ A RUN'S BATCH IS FIXED FOR ITS LIFETIME. batch and accum are part of the resume identity, so
    # a re-probe after a pod restart that lands on a different batch would make the trainer REFUSE
    # to resume. The first probe's answer is the run's answer.
    read -r BATCH ACCUM < "$RUN/batch_accum.txt"
    echo "  reusing this run's batch $BATCH x accum $ACCUM ($RUN/batch_accum.txt)"
  else
    "$PY" - "$BACKBONE" "$AMP" > "$LOGS/batch_probe.log" 2>&1 <<'EOF'
import sys, time, torch
sys.path.insert(0, ".")
import model as M
cfg = M.REFeConfig.for_backbone(sys.argv[1])
total = torch.cuda.get_device_properties(0).total_memory / 2**30
# ⛔ THE BUDGET IS WHAT IS FREE NOW, NOT THE CARD'S SIZE -- anything else already on the GPU counts.
free0 = torch.cuda.mem_get_info()[0] / 2**30
budget = min(total, free0) - 2.0
print(f"  card {total:.2f} GiB, free at start {free0:.2f} GiB -> budget {budget:.2f} GiB", flush=True)
net = M.REFe(cfg).cuda().train()
opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4)
best, prev_B, prev_peak, best_dt = 0, None, None, None
for B in (1, 2, 4, 8, 16):
    # ⛔ PREDICT BEFORE TRYING. A batch that does not fit may not raise: under a driver that can
    # spill past VRAM (MEASURED on the Windows dev box, 2026-09-23) it runs ~30x slower instead,
    # and the probe hangs for many minutes on the one batch it should never have attempted.
    if prev_peak is not None and prev_peak * B / prev_B > budget:
        print(f"  batch {B:3d}  predicted {prev_peak * B / prev_B:6.2f} GiB > budget -- not tried", flush=True)
        break
    try:
        torch.cuda.reset_peak_memory_stats()
        img = torch.randn(B, cfg.n_cameras, 3, cfg.img_h, cfg.img_w, device="cuda")
        ego = torch.randn(B, cfg.ego_dim, device="cuda")
        goal = torch.randn(B, 2 * cfg.n_goal_points, device="cuda")
        tgt = torch.randn(B, cfg.horizon_steps, cfg.traj_dim, device="cuda")
        # a DISTINCT rig per sample -- the per-sample embedding's worst case (R22), as in training
        rig = [list(map(list, net.baked_calib())) for _ in range(B)]
        for i in range(B):
            rig[i][0][9] += 1e-6 * i
        calib = torch.tensor(rig, dtype=torch.float64)
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(sys.argv[2] == "bf16")):
            traj, score = net(img, ego, goal, calib=calib)
        loss, _ = M.wta_loss(traj.float(), tgt)
        (loss + score.float().sum() * 0.0).backward()
        opt.step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / B
        # ⛔ "it ran" is not "it fits": size by MEASURED peak against the budget, AND refuse a batch
        # whose per-sample time jumps -- the signature of memory that is not resident
        peak = torch.cuda.max_memory_allocated() / 2**30
        ok = peak < budget and (best_dt is None or B == 1 or dt < 3.0 * best_dt)
        why = "RESIDENT" if ok else ("TOO CLOSE" if peak >= budget else f"SPILLING ({dt / best_dt:.1f}x per sample)")
        print(f"  batch {B:3d}  peak {peak:6.2f} GiB / budget {budget:.2f}  {dt:.3f} s/sample  {why}", flush=True)
        if not ok:
            break
        best, prev_B, prev_peak = B, B, peak
        best_dt = dt if best_dt is None else min(best_dt, dt)
        del img, ego, goal, tgt, traj, score, loss
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(f"  batch {B:3d}  OOM")
        break
print(f"ZZBATCH_{best}ZZ")
EOF
    cat "$LOGS/batch_probe.log"
    BATCH="$(grep -oE 'ZZBATCH_[0-9]+ZZ' "$LOGS/batch_probe.log" | tail -1 | tr -dc 0-9)"
  fi
  [ -n "$BATCH" ] && [ "$BATCH" -ge 1 ] || fail "no resident batch"
  ACCUM=$(( EFF / BATCH ))
  [ $(( BATCH * ACCUM )) -eq "$EFF" ] || fail "BATCH $BATCH does not divide EFF $EFF"
  [ -s "$RUN/batch_accum.txt" ] || echo "$BATCH $ACCUM" > "$RUN/batch_accum.txt"
  echo "ZZOK batch $BATCH x accum $ACCUM = $EFF ZZ"
fi
# a later relaunch (e.g. STAGES=train after a pod restart) must reuse the SAME batch/accum, or the
# resume identity check refuses the run -- by design
NEED_BATCH=0; has smoke && NEED_BATCH=1; has train && NEED_BATCH=1
if [ "$NEED_BATCH" = 1 ] && [ -z "${ACCUM:-}" ]; then
  if [ -n "${BATCH:-}" ]; then
    ACCUM=$(( EFF / BATCH ))
    [ $(( BATCH * ACCUM )) -eq "$EFF" ] || fail "BATCH $BATCH does not divide EFF $EFF"
  else
    [ -s "$RUN/batch_accum.txt" ] || fail "no batch_accum.txt -- run the batch stage first"
    read -r BATCH ACCUM < "$RUN/batch_accum.txt"
  fi
fi
SPEED_FLAGS="--amp $AMP"; [ "$TF32" = 1 ] && SPEED_FLAGS="$SPEED_FLAGS --tf32"
[ "$COMPILE" = 1 ] && SPEED_FLAGS="$SPEED_FLAGS --compile"; SPEED_FLAGS="$SPEED_FLAGS --workers $WORKERS"
GROW_FLAGS=""; [ "$GROW" = 1 ] && GROW_FLAGS="--grow --grow-scenes $GROW_SCENES"
[ "$NEED_BATCH" = 1 ] && echo "  batch $BATCH x accum $ACCUM = $(( BATCH * ACCUM )) (paper: $EFF)   speed: $SPEED_FLAGS"

if has smoke; then
  echo "=== smoke: one real step on the full bank ==="
  "$PY" train.py --backbone "$BACKBONE" --targets "$TRAIN_BANK" --scorer-targets "$TRAIN_BANK" \
      --images "$PIX" --calib "$TRAIN_BANK/calib_table.json" \
      --steps 1 --batch "$BATCH" --log-every 1 $SPEED_FLAGS $GROW_FLAGS > "$LOGS/smoke.log" 2>&1
  grep -E "trunk:|targets:|scorer bank:|images:|calib:|DROPPED|bank:|step +0|TRAIN_DONE|REFUS|Error" "$LOGS/smoke.log" | cut -c1-200
  grep -q "calib: .* tuples carry their own rig" "$LOGS/smoke.log" || fail "smoke did not attach per-sample rigs"
  grep -q "TRAIN_DONE" "$LOGS/smoke.log" || fail "smoke step failed -- see $LOGS/smoke.log"
  grep -q "images: .* tuples resolve ALL cameras" "$LOGS/smoke.log" || fail "smoke did not resolve the pixels"
  echo "ZZOK smoke ZZ"
fi

if has train; then
  echo "=== train: $EPOCHS epochs over $EPOCH_UNIT, batch $BATCH x accum $ACCUM, run dir $RUN ==="
  # ⛔ ONE supervisor per run dir. And EVERY child closes fd 200 (`200>&-`): an inherited lock fd
  # outlives the supervisor inside the trainer or a `sleep`, and then no replacement can ever start
  # (MEASURED twice on 2026-09-02, the second time on a `sleep`).
  # a MISSING flock must not read as "another supervisor holds the lock" -- two different faults
  command -v flock >/dev/null 2>&1 || fail "flock not installed (util-linux) -- apt-get install -y util-linux"
  exec 200>"$RUN/.supervisor.lock"
  flock -n 200 || fail "another supervisor holds $RUN/.supervisor.lock"
  done_flag () {
    "$PY" -c 'import json,sys; print(1 if json.load(open(sys.argv[1])).get("done") else 0)' \
      "$RUN/summary.json" 2>/dev/null 200>&- || echo 0
  }
  r=0
  while [ "$(done_flag)" != "1" ]; do
    "$PY" train.py --backbone "$BACKBONE" --targets "$TRAIN_BANK" --scorer-targets "$TRAIN_BANK" \
        --images "$PIX" --calib "$TRAIN_BANK/calib_table.json" \
        --epochs "$EPOCHS" --epoch-unit "$EPOCH_UNIT" --batch "$BATCH" --accum "$ACCUM" \
        --out "$RUN" --resume --log-every 1 --ckpt-every-min 20 $SPEED_FLAGS $GROW_FLAGS \
        >> "$RUN/train.log" 2>> "$RUN/train.stderr.log" 200>&-
    rc=$?
    [ "$(done_flag)" = "1" ] && break
    # 3 = trunk refused, 4 = identity/--out refusal: DETERMINISTIC, a restart cannot fix them
    case "$rc" in 3|4) fail "trainer refused (rc=$rc) -- see $RUN/train.log; restarting cannot help";; esac
    r=$((r + 1))
    echo "ZZRESTART train rc=$rc n=$r $(date -u +%FT%TZ)" >> "$RUN/restarts.log"
    [ "$r" -ge "$MAXR" ] && fail "gave up after $r restarts -- see $RUN/restarts.log"
    sleep 60 200>&-
  done
  echo "ZZOK train done: $(cat "$RUN/summary.json" 2>/dev/null | tr -d '\n' | cut -c1-240) ZZ"
fi
echo "ZZDONE $(date -u +%FT%TZ) ZZ"
