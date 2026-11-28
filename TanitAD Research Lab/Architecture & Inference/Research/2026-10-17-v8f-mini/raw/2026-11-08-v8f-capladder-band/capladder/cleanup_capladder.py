"""CAPLADDER cleanup (CLAUDE.md auto-delete rule): after RESULT_CAPLADDER.json and .md exist, copies the few small control files into raw/, then deletes THIS run's own temporary data by literal absolute path and
logs path + size in DELETIONS_CAPLADDER.log. KEEPS F:/TanitAD-scratch/imag1_ctx_eval141.pkl and imag1_real_eval141.pkl (the Master Mind deletes them) and everything in this folder (rungs/*.npz are the evidence)."""
import os, shutil, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
assert os.path.exists("RESULT_CAPLADDER.json") and os.path.exists("RESULT_CAPLADDER.md"), "nothing is deleted before the result and the report exist"
FEAT = "F:/TanitAD-scratch/capladder_feat"
os.makedirs("raw", exist_ok=True)
for f in ("control4_capladder.json", "control4_arrays.npz", "R0_STATUS.json", "r0_progress.jsonl", "feats_report.json"):
    if os.path.exists(f"{FEAT}/{f}"):
        shutil.copy2(f"{FEAT}/{f}", f"raw/{f}")


def size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)


with open("DELETIONS_CAPLADDER.log", "a", encoding="utf-8") as lg:
    for p in ("F:/TanitAD-scratch/capladder_frames", "F:/TanitAD-scratch/capladder_feat", "F:/TanitAD-scratch/capladder_K", "F:/TanitAD-scratch/capladder_smoke", "F:/TanitAD-scratch/capladder_smoke2"):
        if os.path.exists(p):
            n = size(p)
            shutil.rmtree(p, ignore_errors=True)
            lg.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} deleted {p} {n} bytes (own temporary data of the CAPLADDER run; re-creatable: box 2 frames, the cache, the models)\n")
    for p in ("F:/TanitAD-scratch/imag1_ctx_eval141.pkl", "F:/TanitAD-scratch/imag1_real_eval141.pkl"):
        lg.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} KEPT {p} ({os.path.getsize(p) if os.path.exists(p) else 'absent'} bytes; the Master Mind deletes it)\n")
print(open("DELETIONS_CAPLADDER.log", encoding="utf-8").read())
