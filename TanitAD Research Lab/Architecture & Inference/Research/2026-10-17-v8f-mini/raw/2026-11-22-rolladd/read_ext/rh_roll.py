"""rh_roll.py -- ONE rollout of the ROLLHIST read: python rh_roll.py KIND FORM SEED   (KIND RH|B0|UEA|RHp3|RHK|RL|RA, FORM in rh_common.FORMS, SEED 0|1)
Renders the tier-of-record argv (rh_common.argv_record), refuses unless the checkpoint sha256 equals the pod's sha256sum record (ckpts/SHA256_PODS.txt; UEA: the card's pinned sha256),
runs it on the dev-box CPU (2 threads), writes ticks/run_<name>.txt (the checkpoint sha256 and the executor sha256 the rollout used), checks 141 dump files and 1 tick file, touches T1_DONE.
Resumable: T1_DONE present = skipped.  A failed rollout stops with rc 1.  `--dry` prints the argv and runs nothing.  `--max-windows N` is a code check, never a result (writes under out/smoke)."""
import argparse
import glob
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rh_common as C  # noqa: E402


def expected_sha(kind, k):
    if kind == "UEA":
        e = C.CKPT_SHA_UEA[k]
        return e
    for ln in open(f"{C.CK}/SHA256_PODS.txt", encoding="utf-8").read().splitlines():
        p = ln.split()
        if len(p) == 2 and p[1].replace("\\", "/").endswith(f"{kind}_s{k}/ckpt_last.pt") and len(p[0]) == 64:
            return p[0]
    raise SystemExit(f"REFUSED: no pod sha256 for {kind}_s{k} in {C.CK}/SHA256_PODS.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("RH", "B0", "UEA", "RHp3", "RHK", "RL", "RA"))
    ap.add_argument("form", choices=list(C.FORMS))
    ap.add_argument("seed", type=int, choices=(0, 1))
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()
    k = a.seed
    out = C.folder(a.kind, a.form, k, w=(C.OUT + "/smoke") if a.max_windows else None)
    argv = C.argv_record(a.kind, a.form, k, ckpt=a.ckpt or None, out=out, nwin=a.max_windows)
    if a.dry:
        print(" ".join(argv))
        return 0
    if os.path.exists(out + "/T1_DONE"):
        print("skip (done)", out, flush=True)
        return 0
    ck = a.ckpt or C.ckpt_of(a.kind, k)
    if not os.path.isfile(ck):
        raise SystemExit(f"REFUSED: checkpoint missing {ck}")
    got, want = C.sha256_file(ck), expected_sha(a.kind, k)
    if not got.startswith(want[:16]) or (len(want) == 64 and got != want):
        raise SystemExit(f"REFUSED: checkpoint sha256 {got[:12]} differs from the record {want[:12]}")
    os.makedirs(out + "/ticks", exist_ok=True)
    name = f"I10{C.run_name(a.kind, a.form)}_s{k}"
    open(f"{out}/ticks/run_{name}.txt", "w", encoding="utf-8").write(
        f"arm {a.kind}_s{k} form {a.form} readout {C.FORMS[a.form]} kappa {C.KAPPA} ckpt sha256 {got} executor sha256 {C.sha256_file(C.TREE + '/stack/scripts/v8f_t1_eval_hb.py')} "
        f"wrapper sha256 {C.sha256_file(C.CODE + '/integ10k/' + C.WRAPPER + '.py')} started {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    open(out + "/argv.txt", "w", encoding="utf-8").write("\n".join(argv) + "\n")
    t = time.time()
    with open(out + "/t1.log", "a", encoding="utf-8") as lf:
        rc = subprocess.run(argv, env=C.env(), stdout=lf, stderr=subprocess.STDOUT, encoding="utf-8").returncode
    wall = round(time.time() - t, 1)
    nd = len(glob.glob(f"{out}/dump_{C.tag(a.kind, a.form, k)}/ep*.npz"))
    nt = len(glob.glob(f"{out}/ticks/ticks_{name}_m10_causal_heads.npz"))
    open(f"{out}/ticks/run_{name}.txt", "a", encoding="utf-8").write(f"rc {rc} dump_files {nd} tick_files {nt} wall_s {wall}\n")
    print(f"{a.kind} {a.form} s{k} rc {rc} dump_files {nd} tick_files {nt} wall_s {wall}", flush=True)
    if rc != 0 or (not a.max_windows and (nd != 141 or nt != 1)):
        raise SystemExit(f"ROLLOUT FAILED rc {rc} dump_files {nd} tick_files {nt}")
    open(out + "/T1_DONE", "w").write(f"wall_s {wall}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
