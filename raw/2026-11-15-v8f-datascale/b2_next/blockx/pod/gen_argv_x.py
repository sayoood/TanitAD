"""gen_argv_x.py -- DSCALE10K conditional arm X_s0 (card 2.3 row X_s0, section 6 DATA-HURTS row): the 10,000 data-scale clips ONLY, stage A 19,164 steps, stage B 3,000 steps, seed 0 / dagger-seed 0.
Written the way gen_argv2.py / gen_argv_b3.py write the others (same setf, same placeholders).  X_s0.A = U_s0.A with the seven data flags pointed at the 10k-only files; X_s0.B.tmpl = U_s0.B.tmpl likewise.
The data flags are the 10k files of the merge inputs (in/labels, md5 TENK_V9_MD5), the X cache view (xview.py), and --join-train merged/join_X.npz (written by the trainer's --fit-join on the dry build).
Nothing else differs from U_s0 (G-DVB: argv_diff_problems(B_s0 base, X_s0) allows exactly the seven data flags, --fit-join, --steps 19164 and --out).
DS_ROOT / DS_ARGV_OUT override the root and the output folder.  Idempotent.  CPU, JSON only."""
import json
import os

D = os.environ.get("DS_ROOT", "/root/dscale10k")
OUT = os.environ.get("DS_ARGV_OUT", D + "/argv")
XCACHE = os.environ.get("DS_XCACHE", "/root/data/cache_x_dscale")
TENK_V9_MD5 = "395221440e3a3c20f17dd381297af4f0"
L = lambda n: json.load(open(f"{D}/argv/{n}.json"))["argv"]


def setf(t, f, vals):
    t = list(t)
    i = t.index(f)
    j = i + 1
    while j < len(t) and not (t[j].startswith("--") and len(t[j]) > 2):
        j += 1
    return t[:i + 1] + list(vals) + t[j:]


def to_x(t, out):
    t = setf(t, "--out", [out])
    t = setf(t, "--cache-dir", [XCACHE])
    t = setf(t, "--v9-train", [f"{D}/in/labels/v9_labels_datascale10k.npz"])
    t = setf(t, "--v9-train-md5", [TENK_V9_MD5])
    t = setf(t, "--lead-train", [f"{D}/in/labels/lead_datascale10k.npz"])
    t = setf(t, "--speed-train", [f"{D}/in/labels/speed_sidecar_datascale10k.jsonl"])
    t = setf(t, "--tracks", [f"{D}/in/labels/native_tracks_datascale10k.pt"])
    t = setf(t, "--join-train", [f"{D}/merged/join_X.npz"])
    return t


a = to_x(L("U_s0.A"), f"{D}/runs/X_s0/A")
b = to_x(L("U_s0.B.tmpl"), f"{D}/runs/X_s0/B")
assert a[a.index("--steps") + 1] == "19164" and "--fit-join" in a and "--exclude" not in a
assert b[b.index("--steps") + 1] == "3000" and b[b.index("--dagger-seed") + 1] == "0" and "@INIT@" in b and "@BANK@" in b and "@BANKSHA@" in b
os.makedirs(OUT, exist_ok=True)
for n, t in (("X_s0.A", a), ("X_s0.B.tmpl", b)):
    json.dump({"argv": t}, open(f"{OUT}/{n}.json", "w"), indent=1)
    print(n, " ".join(t))
    print()
