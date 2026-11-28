# VM side (exec -f, launches DETACHED): the G-H lever after a non-zero G-H. Re-scores the dev box's ep016 seam with the
# dev box's cache (a) again with the default env -- run-to-run determinism on this VM -- and (b) with the CPU kernels
# pinned to what the dev box (i9-12900F: AVX2, no AVX-512) can execute: OpenBLAS forced to Haswell and NumPy's AVX-512
# dispatch disabled. Writes /content/p0/out/ghx/cpu.json (this VM's CPU flags + NumPy's dispatch in the navsim venv)
# and one harness output dir per arm. ASCII only.
import json
import os
import subprocess

OUT = "/content/p0/out/ghx"
os.makedirs(OUT, exist_ok=True)
NVPY = "/content/p0/venv_navsim/bin/python"
flags = open("/proc/cpuinfo").read().split("flags", 2)[1].splitlines()[0]
model = [l for l in open("/proc/cpuinfo").read().splitlines() if l.startswith("model name")][0]
AVX512_OFF = "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
probe = ("import json, numpy; from numpy.core._multiarray_umath import __cpu_features__ as f, __cpu_dispatch__ as d; "
         "print(json.dumps({'numpy': numpy.__version__, 'enabled': sorted(k for k, v in f.items() if v), 'dispatch': d}))")
cpu = {"model": model, "avx512f": " avx512f" in flags, "avx2": " avx2" in flags, "fma": " fma" in flags}
for tag, extra in (("default", {}), ("avx2", {"NPY_DISABLE_CPU_FEATURES": AVX512_OFF})):
    p = subprocess.run([NVPY, "-c", probe], capture_output=True, text=True, env=dict(os.environ, **extra))
    cpu["numpy_" + tag] = p.stdout.strip()[-1500:] or p.stderr[-800:]
json.dump(cpu, open(f"{OUT}/cpu.json", "w"), indent=1)
script = f"""
cd /content/p0
run() {{ env $2 MPLBACKEND=Agg {NVPY} /content/p0/code/navtest/linux_run_v1.py score --label $1 \
  --arm SEAM:/content/p0/bundle/ref/refe_sub200_ep016.npz --tokens /content/p0/bundle/tokens/A1_sub200_tokens.json \
  --out {OUT} --cache-name navtest --paths /content/p0/out/paths.json --windowspath-shim > {OUT}/$1.driver.log 2>&1; }}
run refe_ghx_default ""
run refe_ghx_avx2 "OPENBLAS_CORETYPE=Haswell NPY_DISABLE_CPU_FEATURES='{AVX512_OFF}'"
echo done > {OUT}/DONE
"""
open(f"{OUT}/run.sh", "w").write(script)
p = subprocess.Popen(["bash", f"{OUT}/run.sh"], stdout=open(f"{OUT}/run.log", "w"), stderr=subprocess.STDOUT,
                     start_new_session=True)
print("ZZGHX_LAUNCHED", p.pid, json.dumps({k: cpu[k] for k in ("model", "avx512f", "avx2", "fma")}))
