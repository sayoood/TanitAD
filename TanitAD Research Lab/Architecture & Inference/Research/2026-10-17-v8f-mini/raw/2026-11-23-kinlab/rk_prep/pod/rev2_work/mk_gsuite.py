s = open("rk_gsuite.sh.rev1", encoding="utf-8", newline="").read()
def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a)); s = s.replace(a, b)
sub("# rk_gsuite.sh -- G-SUITE (full stack/tests, CPU ONLY) for the KINLAB code of record (the RK commit 0ca6f49d) on the pod.",
    "# rk_gsuite.sh -- REVISION 2. G-SUITE (full stack/tests, CPU ONLY) for the KINLAB code of record (commit 1a8b11a1, with --dagger-label-cache) on the pod.")
sub("(stack + taniteval of the RK commit, blob-verified)", "(stack + taniteval of commit 1a8b11a1, blob-verified)")
sub("pinned to cores 48-95, nice 10, OMP/MKL 2 threads,", "pinned to cores 48-95, nice 19 (revision 2: was 10), OMP/MKL 2 threads,")
sub("(< 20 GB; the watchdog kills the shards by PID)", "(< 20 GB; the watchdog kills the shards by PID; revision 2: a wave is not started while the container's total_rss is >= 40 GB, was 45 GB)")
sub("taskset -c 48-95 nice -n 10 timeout 4h", "taskset -c 48-95 nice -n 19 timeout 4h")
sub('-lt 45000000000 ] || fail "container anon memory $R0 B >= 45 GB: wave $wave not started"', '-lt 42949672960 ] || fail "container anon memory $R0 B >= 40 GB: wave $wave not started"')
sub("$PY $R/blobcheck.py $TREE $BLOBS", "$PY $R/blobcheck.py $TREE $BLOBS")
open("rk_gsuite.sh", "w", encoding="utf-8", newline="").write(s)
