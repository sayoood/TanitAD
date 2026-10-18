import subprocess, sys, os, time
sys.path.insert(0, "C:/Users/Admin/freept/raw/2026-10-17-free-point-navtest/code")
import fp_capture as C
env = C.env_for_children()
log = sys.argv[1]
t0 = time.time()
rc = subprocess.call([C.PY, "-I", C.__file__, "one", "--out-dir", "C:/Users/Admin/freept/work", "--log", log], env=env)
print("rc", rc, f"{time.time()-t0:.0f}s")
