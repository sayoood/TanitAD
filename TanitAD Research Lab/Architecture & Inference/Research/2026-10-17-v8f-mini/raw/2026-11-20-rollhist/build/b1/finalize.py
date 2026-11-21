"""Finalize ROLLHIST B1 (no pytest): the patch, the sha256 list, the file:line hops. Run after the mutation and regression drivers. Cleans __pycache__ in the two trees first."""
import hashlib
import json
import os
import re
import shutil
import subprocess

ROOT = "F:/TanitAD-scratch/rollhist/build"
B1 = ROOT + "/b1"
for t in ("tree", "orig"):
    for d, ds, fs in os.walk(f"{ROOT}/{t}"):
        for x in list(ds):
            if x in ("__pycache__", ".pytest_cache"):
                shutil.rmtree(os.path.join(d, x), ignore_errors=True)
                ds.remove(x)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# the patch: stack/ only, against the exported tip, LF, binary-safe
r = subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "diff", "--no-index", "--binary", "--no-color", "orig/stack", "tree/stack"], cwd=ROOT, capture_output=True)
patch = r.stdout.replace(b"a/orig/stack/", b"a/stack/").replace(b"b/tree/stack/", b"b/stack/").replace(b"orig/stack/", b"stack/").replace(b"tree/stack/", b"stack/")
open(B1 + "/rollhist.patch", "wb").write(patch)
files = sorted(set(re.findall(rb"^diff --git a/(\S+) b/", patch, re.M)))
files = [f.decode() for f in files]
open(B1 + "/TREE_FILES_SHA256.txt", "w", encoding="utf-8", newline="\n").write("".join(f"{sha(ROOT + '/tree/' + f)}  {f}\n" for f in files))


def hop(path, pat, nth=0):
    ls = open(f"{ROOT}/tree/{path}", encoding="utf-8").read().split("\n")
    hits = [i + 1 for i, l in enumerate(ls) if re.search(pat, l)]
    return f"{path}:{hits[nth]}" if len(hits) > nth else f"{path}:NOT-FOUND({pat})"


hops = {
    "flag_to_header": [hop("stack/scripts/v8f_dagger_bank.py", r'g\.add_argument\("--readout"'), hop("stack/scripts/v8f_dagger_bank.py", r"^def generator_record"),
                       hop("stack/scripts/v8f_dagger_bank.py", r"with executor_installed\(T1, gen\)"), hop("stack/scripts/v8f_dagger_bank.py", r'"rollout": G.ROLLOUT, \*\*\(\{\} if gen_rec is None'),
                       hop("stack/scripts/v8f_dagger_bank.py", r'"plan": plan, \*\*\(\{\} if gen_rec is None'), hop("stack/scripts/v8f_dagger_bank.py", r'"controls": controls, \*\*\(\{"generator": common')],
    "install": [hop("stack/scripts/v8f_dagger_bank.py", r"frames = RO.readout_frames"), hop("stack/scripts/v8f_dagger_bank.py", r"HB.bounded_frames\(frames"),
                hop("stack/scripts/v8f_t1_readout.py", r"^def causal_g_frames"), hop("stack/scripts/v8f_t1_eval_hb.py", r"^def bounded_frames"), hop("stack/scripts/v8f_t1_eval.py", r"hermite_frames\(self.track")],
    "header_to_built_levers": [hop("stack/tanitad/data/v8f_dagger.py", r"gen = \(\{\} if self.bank is None"), hop("stack/tanitad/data/v8f_dagger.py", r'"dagger_bank_readout": readout_token')],
    "built_levers_to_gdvb": [hop("stack/scripts/v8f_mini_train.py", r'"dagger_bank_readout": "dagger:dagger_bank_readout"'), hop("stack/scripts/v8f_mini_train.py", r'"--dagger-bank-readout"'),
                             hop("stack/scripts/v8f_mini_train.py", r'"--dagger-bank-kappa"'), hop("stack/scripts/v8f_mini_train.py", r"^def bad_dagger"), hop("stack/scripts/v8f_mini_train.py", r"for k in \(\"dagger_bank_readout\", \"dagger_bank_kappa\"\):"),
                             hop("stack/scripts/v8f_mini_train.py", r"bad \+= bad_dagger\(a, dag")],
}
json.dump(hops, open(B1 + "/hops.json", "w"), indent=1)
print(len(files), "files in the patch;", len(patch), "bytes")
for f in files:
    print(f)
