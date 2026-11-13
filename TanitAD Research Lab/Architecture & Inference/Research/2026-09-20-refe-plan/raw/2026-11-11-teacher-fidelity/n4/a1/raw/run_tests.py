import sys, subprocess
RUN = "F:/tanitad_scratch/n4nt/run"
sys.path.insert(0, RUN + "/n4")
import n4_nt_chain as C
d, args = sys.argv[1], sys.argv[2:]
r = subprocess.run([C.PY] + args, capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL, env={**C.env(RUN), "N4_TMP": "F:/tanitad_scratch/n4a1/tmp"}, cwd=d)
print(r.returncode); print(r.stdout[-3000:]); print(r.stderr[-3000:])
