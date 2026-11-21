import re
p = "run_bpeval.sh"
s = open(p, encoding="utf-8", newline="").read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)
# python.exe count without nested quotes
rep("npy=$(rsh \"(Get-CimInstance Win32_Process -Filter \\\\\\\"Name='python.exe'\\\\\\\" | Measure-Object).Count\" | head -1)", "npy=$(rsh \"(Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count\" | head -1)")
# directory put/get semantics: the remote parent receives the local directory under its own name
rep("""rput()  { if [ $DRY = 1 ]; then mkdir -p "$(dirname "$(fk "$2")")"; cp -r "$1" "$(fk "$2")"; else""", """rput()  { if [ $DRY = 1 ]; then mkdir -p "$(dirname "$(fk "$2")")"; cp -r "$1" "$(fk "$2")"; else""")
rep("""rget()  { if [ $DRY = 1 ]; then mkdir -p "$2"; cp -r "$(fk "$1")"/. "$2"/ 2>/dev/null || cp "$(fk "$1")" "$2"/; else""", """rget()  { if [ $DRY = 1 ]; then mkdir -p "$2"; cp -r "$(fk "$1")" "$2"/; else""")
rep("""rput "$CACHE" "$RR/cache_pod/eval"
""", """rputdir "$CACHE" "$RR/cache_pod"          # the pod's eval split directory goes in under its own name: cache_pod/eval/{clips,manifest.jsonl}
""")
rep("""rget()  {""", """rputdir() { if [ $DRY = 1 ]; then cp -r "$1" "$(fk "$2")"/; else printf 'put -r "%s" "/%s"\n' "$1" "$2" | sftp -b - tanitad-box2 2>&1 | grep -vE "$NOISE" | tail -2; fi; }
rget()  {""")
rep("""[ -d "$CACHE/clips" ] &&""", """[ "$(basename "$CACHE")" = eval ] && [ -d "$CACHE/clips" ] &&""")
rep("""if [ $DRY = 1 ]; then mkdir -p "$(fk $RR/code)"; tar -xf "$WORK/bp_code.tar" -C "$(fk $RR/code)"; else""", """if [ $DRY = 1 ]; then mkdir -p "$(fk $RR/code)"; (cd "$(fk $RR/code)" && tar --force-local -xf "$WORK/bp_code.tar"); else""")
# a timeout keeps the box CLAIMED (a job may still run); every other wait failure frees it
rep("""rwait "$RR/logs/bp_hash.log" 60 || die HASH-WAIT "rc=$? (2 = box rebooted, 3 = unreachable, 4 = timeout)\"""", """rwait "$RR/logs/bp_hash.log" 60; wrc=$?; [ $wrc = 4 ] && { say "ZZ BPEVAL HASH-WAIT-TIMEOUT: the box stays CLAIMED, a job may still run ZZ"; exit 9; }; [ $wrc = 0 ] || die HASH-WAIT "rc=$wrc (2 = box rebooted, 3 = unreachable)\"""")
rep("""rwait "$RR/logs/bp_job.log" 120 || die EVAL-WAIT "rc=$? (2 = box rebooted, 3 = unreachable, 4 = timeout; the box stays CLAIMED only in case 4)" 4""", """rwait "$RR/logs/bp_job.log" 120; wrc=$?; [ $wrc = 4 ] && { say "ZZ BPEVAL EVAL-WAIT-TIMEOUT: the box stays CLAIMED, the job may still run ZZ"; exit 9; }; [ $wrc = 0 ] || die EVAL-WAIT "rc=$wrc (2 = box rebooted, 3 = unreachable)" 4""")
rep("""rget "$RR/out" "$OUT/failed_run_$(date +%H%M%S)\"""", """rget "$RR/out" "$WORK/failed_run_$(date +%H%M%S)\"""")
rep("""logs pulled to $OUT/failed_run_*" 5""", """logs pulled to $WORK/failed_run_*" 5""")
rep("""rget "$RR/out" "$OUT"
""", """PULL="$WORK/pull_$(date +%H%M%S)"; rget "$RR/out" "$PULL"; [ -d "$PULL/out" ] || die PULL-FAILED "$PULL/out" 6; mv "$PULL"/out/* "$OUT"/
""")
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")
