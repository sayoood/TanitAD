#!/bin/bash
# E-THOR-CLK — does pinning Thor's clocks buy back any of the 34.7 % deficit?
#
# WHY THIS IS A REAL QUESTION AND NOT A TWEAK. MEASURED 2026-08-16 while the
# v6F S-W resume was training:
#
#     nvpmodel        : mode 1 = "120W"   (mode 0 = MAXN exists and is unused)
#     gpu-gpc-0       : cur 1386 MHz  max 1386 MHz   <- ALREADY MAXED
#     bwmgr (EMC)     : cur 3200 MHz  max 4266 MHz   <- 25 % LEFT ON THE TABLE
#     power.draw      : 30.75 W of a 120 W budget
#     tj-thermal      : 58.5 C  (nowhere near throttling)
#     EMC sampled 5x over 5 s: 3200000000 every time — pinned, not oscillating
#
# The GPU core is already at its ceiling, so the only headroom is MEMORY
# BANDWIDTH — and a 336 M transformer at batch 8 on 20 SMs, with
# --grad-checkpoint ON (which trades compute for MORE memory traffic:
# recompute + re-read activations), is exactly the workload where EMC binds.
#
# ⛔ THE MEASUREMENT HAZARD THIS SCRIPT EXISTS TO AVOID (RETRACTION_LOG C68).
# `train_log.jsonl` is one append-only file across the whole run. If the clock
# change lands in the MIDDLE of a 50-step logging interval, that interval is a
# rate across TWO CONFIGURATIONS and is not attributable to either. So this
# script records the EXACT step at which it fired, and the analysis must DISCARD
# the straddling interval rather than average through it.
#
# ⚠️ AND THE SCOPE CAVEAT THIS ESTABLISHES (C14's family): until this runs, the
# 27.21 s/step figure is "Thor AS CONFIGURED", not "Thor". A number that reports
# our own configuration and gets read as a capability is the grid-terminus error.
#
# SAFETY. `jetson_clocks` pins clocks to max WITHIN the current nvpmodel and
# disables downward DVFS. It does not signal, restart or touch any process, and
# it is reversible via the stored config. It deliberately does NOT change
# nvpmodel: `nvpmodel -m 0` can prompt for a reboot on some Jetson platforms,
# and a reboot would kill a 336 M training run. MAXN stays a PI decision.
set -u
OUT=~/experiments/v6F-SW-30k
MARK=$OUT/E_THOR_CLK.json
LOG=~/logs/thor_clocks.log
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

snapshot() {   # $1 = label
  python3 - "$1" <<'PY'
import glob, json, os, subprocess, sys
def rd(p):
    try:
        return open(p).read().strip()
    except OSError:
        return None
d = {"label": sys.argv[1], "devfreq": {}}
for f in sorted(glob.glob("/sys/class/devfreq/*/")):
    d["devfreq"][os.path.basename(f.rstrip("/"))] = {
        "cur": rd(f + "cur_freq"), "min": rd(f + "min_freq"),
        "max": rd(f + "max_freq"), "gov": rd(f + "governor")}
try:
    d["nvpmodel"] = subprocess.run(["nvpmodel", "-q"], capture_output=True,
                                   text=True).stdout.strip()
except Exception as e:
    d["nvpmodel"] = f"{type(e).__name__}"
try:
    d["smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except Exception as e:
    d["smi"] = f"{type(e).__name__}"
d["thermal"] = {rd(z + "/type"): rd(z + "/temp")
                for z in sorted(glob.glob("/sys/devices/virtual/thermal/thermal_zone*"))}
print(json.dumps(d))
PY
}

# The step the change lands at — read from the LOG, which is what the analysis
# will segment on. Reading it from anywhere else would not be the same clock.
cur_step() {
  python3 - <<'PY'
import json
best = 0
try:
    for ln in open("/home/nvidia/experiments/v6F-SW-30k/train_log.jsonl",
                   errors="ignore"):
        if ln.startswith("{"):
            try:
                s = json.loads(ln).get("step")
            except ValueError:
                continue
            if isinstance(s, int) and s > best:
                best = s
except OSError:
    pass
print(best)
PY
}

STEP_BEFORE=$(cur_step)
say "E-THOR-CLK: last logged step BEFORE the change = $STEP_BEFORE"
BEFORE=$(snapshot before)

say "storing the restore point"
sudo jetson_clocks --store ~/jetson_clocks.before.conf >> "$LOG" 2>&1
say "store rc=$?"

say "applying jetson_clocks (pins clocks to max within nvpmodel 1 = 120W)"
sudo jetson_clocks >> "$LOG" 2>&1
say "apply rc=$?"
sleep 3
AFTER=$(snapshot after)
STEP_AFTER=$(cur_step)

python3 - "$STEP_BEFORE" "$STEP_AFTER" "$BEFORE" "$AFTER" > "$MARK" <<'PY'
import json, sys
sb, sa, b, a = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4])
print(json.dumps({
    "experiment": "E-THOR-CLK",
    "what": "jetson_clocks applied to a LIVE v6F S-W run; nvpmodel UNCHANGED",
    "step_last_logged_before": int(sb),
    "step_last_logged_after": int(sa),
    "straddle_interval_note":
        "⛔ The logging interval containing this change spans TWO "
        "configurations and MUST BE DISCARDED, not averaged through "
        "(RETRACTION_LOG C68). The first admissible 'after' interval is the "
        "one whose BOTH endpoints are logged after step "
        + str(max(int(sb), int(sa))) + ".",
    "before": b, "after": a,
    "restore": "sudo jetson_clocks --restore ~/jetson_clocks.before.conf",
    "_evidence_class": "MEASURED (ours; /sys/class/devfreq + nvpmodel -q)",
}, indent=1))
PY
say "marker written -> $MARK"
python3 -c "
import json
d=json.load(open('$MARK'))
for k in ('bwmgr','gpu-gpc-0','gpu-nvd-0'):
    b=d['before']['devfreq'].get(k,{}); a=d['after']['devfreq'].get(k,{})
    print(f\"  {k:12s} cur {b.get('cur')} -> {a.get('cur')}   (max {a.get('max')})\")
print('  nvpmodel:', d['after']['nvpmodel'].replace(chr(10),' / '))
print('  smi     :', d['after']['smi'])
"
