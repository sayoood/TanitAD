#!/bin/bash
# Test of run_l7cap_devbox.sh's marker / lock discipline with a STUB gpu_lock and a scratch queue dir (no GPU, no real lock, no real marker).
# Expectations are literals: the marker must exist while the chain runs and be gone after normal exit, after TERM, and a marker owned by someone
# else must NEVER be removed (the script times out waiting instead).
S=C:/Users/Admin/refcv8_pf/l7cap/launch/run_l7cap_devbox.sh
T=$(mktemp -d); Q=$T/q; mkdir -p $Q
cat > $T/gl_stub.py <<'PYEOF'
import json, sys
a = sys.argv
out = a[a.index("--out") + 1] if "--out" in a else None
r = {"ok": True} if a[1] == "acquire" else {"released": True}
if out: json.dump(r, open(out, "w"))
PYEOF
GL=$T/gl_stub.py; export L7C_Q=$Q L7C_GL=$GL L7C_OUT=$T/out L7C_DRY=1
PYBIN=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
fail=0
# the stub is run by the venv python through $PY in the script; GL points at the stub
# 1. normal exit
L7C_DRY_HOLD_S=0 bash $S 0 > /dev/null 2>&1
[ ! -e $Q/devbox_gpu.chain_active ] && echo "T1 PASS marker cleared after normal exit" || { echo "T1 FAIL marker left"; fail=1; }
grep -q "ZZL7CDFLAGSETZZ" $T/out/devbox.log && grep -q "ZZL7CDLOCKEDZZ" $T/out/devbox.log && grep -q "ZZL7CDRELEASEDZZ" $T/out/devbox.log && echo "T1b PASS set+locked+released" || { echo "T1b FAIL markers"; fail=1; }
# 2. marker present WHILE running, cleared after TERM
rm -f $T/out/devbox.log
L7C_DRY_HOLD_S=30 bash $S 0 > /dev/null 2>&1 &
pid=$!
for i in $(seq 1 30); do [ -e $Q/devbox_gpu.chain_active ] && break; sleep 0.5; done
[ -e $Q/devbox_gpu.chain_active ] && echo "T2a PASS marker present while running" || { echo "T2a FAIL no marker"; fail=1; }
kill -TERM $pid; wait $pid 2>/dev/null
for i in $(seq 1 20); do [ ! -e $Q/devbox_gpu.chain_active ] && break; sleep 0.5; done
[ ! -e $Q/devbox_gpu.chain_active ] && echo "T2b PASS marker cleared after TERM" || { echo "T2b FAIL marker left after TERM"; fail=1; }
# 3. somebody else's marker is never removed
echo '{"chain": "other-chain", "pid": "1", "started": "x"}' > $Q/devbox_gpu.chain_active
L7C_QUEUE_WAIT_S=2 L7C_DRY_HOLD_S=0 timeout 60 bash $S 0 > /dev/null 2>&1
grep -q "other-chain" $Q/devbox_gpu.chain_active && echo "T3 PASS a foreign marker survives (script timed out waiting)" || { echo "T3 FAIL foreign marker removed"; fail=1; }
# 4. deliberate regression: a copy WITHOUT the EXIT trap must leave the marker behind (so tests 1-2 can actually discriminate)
sed 's/^trap cleanup EXIT$//' $S > $T/mutant.sh
rm -f $Q/devbox_gpu.chain_active
L7C_DRY_HOLD_S=0 bash $T/mutant.sh 0 > /dev/null 2>&1
[ -e $Q/devbox_gpu.chain_active ] && echo "T4 PASS the trap-less mutant leaves the marker (the test discriminates)" || { echo "T4 FAIL the mutant did not leave the marker: tests 1-2 are blind"; fail=1; }
rm -rf $T
echo "ZZL7CDTESTZZ fail=$fail"
exit $fail
