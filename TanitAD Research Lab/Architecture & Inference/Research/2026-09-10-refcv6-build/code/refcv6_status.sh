#!/usr/bin/env bash
# refcv6 -- ONE-SHOT HONEST STATUS of the campaign on the box it runs on.
#
#   ssh -n <box> 'bash /home/nvidia/refcv6/code/refcv6_status.sh'
#
# Every line here exists because the obvious version of it has lied:
#
# ⛔ NO `grep -c` ON `ps`, AND NO `pgrep -f`. Both SELF-MATCH the command line
#    that carried them -- the PTY echoes it back -- and have three times
#    produced a count that was pure artifact: once inventing a `Traceback CUDA
#    out of memory` on a run that was healthy and 3 minutes in, once reporting
#    "1 trainer, 1 supervisor" on a box with nothing on it. Processes are read
#    from `/proc/<pid>/cmdline`, and the search tokens are ASSEMBLED AT RUNTIME
#    so this script's own argv cannot contain them.
#
# ⛔ PROGRESS IS READ FROM `metrics.jsonl`, NOT FROM A LOG AND NOT FROM AN EXIT
#    CODE. `refc_v3_train.py:4956` appends JSON-LINES there. ⚠️ It is
#    `metrics.jsonl`, NOT `metrics.json` -- a gate and a supervisor both read the
#    wrong name here on 2026-09-11, one blocking every launch and the other
#    unable to ever write a done-marker.
#
# ⛔ `df` IS NOT READ FOR A QUOTA, only for this box's own single disk. On a pod
#    it reports the cluster and hides the per-pod quota.
#
# ⚠️ On Thor `free` / `tegrastats` / `mem_get_info` are ALL unreliable for device
#    memory -- only in-process `torch.cuda.max_memory_allocated()` is admissible.
#    Host RSS below is host RSS and is labelled as such; `nvidia-smi` utilisation
#    is a coarse liveness signal, not a memory claim.

set -u
ROOT="${REFCV6_ROOT:-/home/nvidia/refcv6}"
OUT_ROOT="${REFCV6_OUT_ROOT:-/home/nvidia/experiments}"
ARMS="${*:-V0 V0b D}"

tok_trainer="$(printf 'refc_v3')_$(printf 'train.py')"
# ⛔ THE TOKENS CARRY `.sh`/`.py`, AND THAT IS NOT COSMETIC. MEASURED
#    2026-09-11: an operator ssh command that merely MENTIONED
#    `chain_refcv6.log` was counted as a second CHAIN by this very script. The
#    runtime assembly protects THIS script's own argv; it cannot protect a
#    CALLER whose command line names the file. Anchoring on the executable's
#    extension is what makes the match about the process rather than about any
#    string that happens to contain the stem.
tok_sup="$(printf 'sup_')$(printf 'refcv6.sh')"
tok_chain="$(printf 'chain_')$(printf 'refcv6.sh')"

echo "ZZSTATUS_BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"

# ---- processes, from /proc ------------------------------------------------- #
n_tr=0; n_tr_main=0; n_sup=0; n_sup_main=0; n_chain=0; n_chain_main=0; n_proc=0
tr_pids=" "; sup_pids=" "; chain_pids=" "
for p in /proc/[0-9]*/cmdline; do
  pid="${p#/proc/}"; pid="${pid%/cmdline}"
  cmd="$(tr '\0' ' ' < "$p" 2>/dev/null)" || continue
  [ -n "$cmd" ] || continue
  n_proc=$((n_proc + 1))
  case "$cmd" in *prelaunch_gate*) continue ;; esac
  case "$cmd" in *"$tok_trainer"*) n_tr=$((n_tr+1)); tr_pids="${tr_pids}${pid} " ;; esac
  case "$cmd" in *"$tok_sup"*)     n_sup=$((n_sup+1)); sup_pids="${sup_pids}${pid} " ;; esac
  case "$cmd" in *"$tok_chain"*)   n_chain=$((n_chain+1)); chain_pids="${chain_pids}${pid} " ;; esac
done

# ⛔⛔ A RAW TRAINER COUNT IS A TRAP ON THIS BOX, AND IT MISREADS AS THE EXACT
#    CATASTROPHE THE CAMPAIGN IS ORGANISED AROUND. `--workers 6` means one arm
#    shows SEVEN processes sharing the trainer's argv (1 main + 6 dataloader
#    workers). Printed bare, "trainers=7" reads as *"seven concurrent arms"* --
#    the MEASURED failure where 7 arms sat at 0-6 % GPU for fifty minutes.
#    ⇒ a process is a MAIN arm only if its PARENT is not also a trainer. That is
#    what "how many arms are running" actually means, and it is the number the
#    one-arm-at-a-time rule is about.
# ⚠️ The SAME misread hits supervisors and chains: a forked bash child carries
#    its PARENT'S cmdline in /proc until it execs, so one chain can read as two.
count_mains() {                         # $1 = label   $2 = space-padded pid list
  local label="$1" list="$2" pid ppid n=0
  for pid in $list; do
    [ -n "$pid" ] || continue
    ppid="$(awk '{print $4}' "/proc/${pid}/stat" 2>/dev/null)" || continue
    case "$list" in
      *" ${ppid} "*) continue ;;        # a child of its own kind: worker / fork
    esac
    n=$((n+1))
    echo "  ${label} pid=$pid  $(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | cut -c1-110)"
  done
  printf '%s' "$n" > "/tmp/.refcv6_count_$$"
}
count_mains "ARM(main) " "$tr_pids";    n_tr_main=$(cat "/tmp/.refcv6_count_$$")
count_mains "SUPERVISOR" "$sup_pids";   n_sup_main=$(cat "/tmp/.refcv6_count_$$")
count_mains "CHAIN     " "$chain_pids"; n_chain_main=$(cat "/tmp/.refcv6_count_$$")
rm -f "/tmp/.refcv6_count_$$"
# ⭐ n_proc is the SAME-BREATH CONTROL: a zero count from a /proc that could not
#    be read is indistinguishable from an empty box, so the total must be > 0
#    before any of the zeros above is quotable.
echo "ZZPROC arms_main=${n_tr_main} trainer_procs_incl_workers=${n_tr} supervisors=${n_sup_main} supervisor_procs=${n_sup} chains=${n_chain_main} chain_procs=${n_chain} CONTROL_total_readable=${n_proc}"
[ "$n_tr_main" -le 1 ] || echo "  ⛔ ${n_tr_main} ARMS ARE RUNNING AT ONCE. Thor saturates at batch 8; MEASURED: 7 concurrent arms = 0-6 % GPU for 50 min with zero progress. Kill by EXPLICIT pid."
[ "$n_proc" -gt 0 ] || echo "  ⛔ CONTROL FAILED: /proc read nothing. The zeros above are claims about the READ, not the box."

# ---- per-arm artifacts ----------------------------------------------------- #
for ARM in $ARMS; do
  d="${OUT_ROOT}/refcv6-${ARM}"
  mj="${d}/metrics.jsonl"
  done_m="${d}/summary.json"
  if [ ! -d "$d" ]; then echo "ZZARM ${ARM} state=NOT_STARTED"; continue; fi
  step=-1; nrows=0; rate="?"
  if [ -s "$mj" ]; then
    read -r step nrows rate <<EOF
$(python3 - "$mj" <<'PYEOF'
import json, sys
steps, el = [], []
try:
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue          # a torn final line is skipped, never fatal
            if isinstance(r, dict) and isinstance(r.get("step"), int):
                steps.append(r["step"])
                if isinstance(r.get("elapsed_s"), (int, float)) and "loss" in r:
                    el.append((r["step"], float(r["elapsed_s"])))
except Exception:
    print("-1 0 ?"); raise SystemExit(0)
# ⭐ MARGINAL rate over the LAST HALF of the rows, never the whole run: an
#    average from step 0 is dominated by startup and by --warmup, which is
#    exactly how a "~1.2 s/step" figure got published and retracted.
rate = "?"
if len(el) >= 4:
    a = el[len(el) // 2]
    b = el[-1]
    if b[0] > a[0]:
        rate = "%.3f" % ((b[1] - a[1]) / (b[0] - a[0]))
print(f"{max(steps) if steps else -1} {len(steps)} {rate}")
PYEOF
)
EOF
  fi
  st="RUNNING"
  if [ -s "$done_m" ] && grep -q '"done"[[:space:]]*:[[:space:]]*true' "$done_m" 2>/dev/null; then
    st="DONE"
  elif [ "$n_tr" -eq 0 ]; then
    st="STOPPED_NO_MARKER"     # ⛔ the ABSENCE of the marker is the evidence
  fi
  echo "ZZARM ${ARM} state=${st} step=${step} rows=${nrows} marginal_s_per_step=${rate}"
done

# ---- box ------------------------------------------------------------------- #
echo "ZZDISK $(df -h "$OUT_ROOT" | tail -1 | awk '{print "size="$2" used="$3" avail="$4" pct="$5}')"
echo "ZZGPU util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' ')"
echo "ZZHOSTMEM $(free -g | awk '/^Mem:/{print "total_gb="$2" used_gb="$3" available_gb="$7}')  (host RSS only -- NOT admissible for device memory on Thor)"
echo "ZZSTATUS_END"
