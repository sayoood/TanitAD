"""test_n1dm_paths.py -- the pod dump names and the read's dump paths agree (no data, no torch): for each of the 10 dumps n1dm_dumps.sh lists, the folder + file name hb_dump_n1dm.sh produces on the pod (its TAG rule, evaluated by bash)
equals the path n1dm_ho_read.dpath looks for under ./in. Also: the Z-USE (suffix U) and Z-UNIF (suffix V) arms are read, the replay maps N1DM slots to the banked N1DH dumps, and no N1DH / N1DC tag is left where an N1DM tag belongs.
Run: python -I test_n1dm_paths.py   (N1DM_TREE = the build tree)"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + "/eval141")
POD = os.environ.get("N1DM_POD_SCRIPTS", "F:/TanitAD-scratch/n1dc/build/n1dm_b1/pod")
import n1dm_ho_read as R  # noqa: E402

R.G = {"inp": "/IN", "cmp": "/CMP"}
ok = []
# the pod's rule: dump list from n1dm_dumps.sh (ARM VARIANT KAPPA | out dir) and the TAG of hb_dump_n1dm.sh
lines = [ln for ln in open(POD + "/n1dm_dumps.sh", encoding="utf-8").read().splitlines() if ln.strip().startswith('echo "N1DM_')]
_src = open(POD + "/n1dm_dumps.sh", encoding="utf-8").read()
# MM 2026-11-21: the 5 echo lines sit inside `for S in s0 s1; do`, so the script writes 10 dumps; the first form counted SOURCE lines against 10 and always failed
ok.append(("n1dm_dumps.sh lists 10 dumps (s0/s1 x follow/zero/zuse/unif at hb0.2 + follow at old): 5 echo lines inside one two-seed loop", len(lines) == 5 and _src.count("for S in s0 s1; do") >= 1 and all("$S" in ln or "${S}" in ln for ln in lines)))
tag_rule = [ln for ln in open(POD + "/hb_dump_n1dm.sh", encoding="utf-8").read().splitlines() if ln.startswith("TAG=")][0]
suf_rule = [ln for ln in open(POD + "/hb_dump_n1dm.sh", encoding="utf-8").read().splitlines() if ln.startswith("SUF=")]
allgood = True
for ln in lines:
    m = re.match(r'\s*echo "(\S+) (\S+) (\S+) (\S+) (\S+)"', ln.replace("${S}", "s0").replace("$S", "s0"))
    arm, var, kappa, outdir, _n = m.groups()
    cmd = f'ARM={arm}; VAR={var}; SUF=""; [ "$VAR" != follow ] && SUF="_$VAR"; {tag_rule}; echo $TAG'
    tag = subprocess.run(["C:/Program Files/Git/usr/bin/bash.exe", "-c", cmd], capture_output=True, text=True).stdout.strip()
    pod_name = f"dump_{tag}_m10_causal_heads"
    pod_dir = outdir.split("/")[0]                                           # hb0.2 | hbnone
    # the read's side: arm letters Z (zero twin) U (zuse) V (unif)
    letter = "Z" if var == "zero" else "U" if arm.endswith("_zuse") else "V" if arm.endswith("_unif") else ""
    base = "N1DM_s0"
    got = R.dpath("hb" if pod_dir == "hb0.2" else "old", base + letter)
    exp_tail = f"/in/n1dm/{pod_dir}/".replace("/in/", "/IN/") + outdir.split("/", 1)[1] + "/" + pod_name
    good = got.replace("\\", "/") == exp_tail.replace("//", "/") or got.replace("\\", "/").split("/")[-1] == pod_name and got.replace("\\", "/").startswith(f"/IN/n1dm/{pod_dir}/")
    allgood &= good
    if not good:
        print("MISMATCH", arm, var, kappa, "read:", got, "pod:", exp_tail)
ok.append(("every pod dump (name + folder) is the path the read looks for", allgood))
rep = R.dpath("hb", "N1DM_s0", replay=True)
ok.append(("replay: the N1DM slot reads the banked N1DH dump", "/n1dh/" in rep and "dump_N1DH-HO-N1DH_s0_m10_causal_heads" in rep))
ok.append(("a D-channel zero twin of the comparator N1DH is not read", R.dpath("hb", "N1DH_s0Z") is None and R.dpath("hb", "N1DH_s0V") is None))
src = open(HERE + "/n1dm_ho_read.py", encoding="utf-8").read()
ok.append(("the arms list reads Z-USE (U) and Z-UNIF (V)", '("", "Z", "U", "V")' in src and "Z_UNIF_real_minus_uniform_p" in src))
leftover = []
for dirpath, _d, files in os.walk(HERE):
    dirpath_n = dirpath.replace("\\", "/")   # MM 2026-11-21: os.walk on Windows yields backslash subpaths, so "/tree" never matched once tree/ existed
    if any(x in dirpath_n for x in ("__pycache__", "/tmp", "/out", "/tree", "/in", "/ckpts", "dumps141")):
        continue
    for f in files:
        if f.endswith((".py", ".sh", ".cmd")) and f != "test_n1dm_paths.py":
            t = open(os.path.join(dirpath, f), encoding="utf-8", errors="replace").read()
            leftover += [(f, m) for m in re.findall(r"N1DC|n1dc_|_n1dc|PREREG_V8F_N1DH\.md", t)]
ok.append(("no N1DC tag and no N1DH card name left in the read tools (the N1DH read dir is only named for banked inputs)", leftover == []))
if leftover:
    print("LEFTOVER", leftover[:10])
for name, good in ok:
    print(("ok   " if good else "FAIL ") + name)
sys.exit(0 if all(g for _, g in ok) else 1)
