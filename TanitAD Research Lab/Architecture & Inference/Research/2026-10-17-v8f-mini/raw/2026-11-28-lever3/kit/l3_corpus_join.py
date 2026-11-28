#!/usr/bin/env python3
"""l3_corpus_join.py -- re-creates the corpus join (the pod's /root/strat1/inputs/join_train.npz, which C-JOIN-EQ reads, md5 prefix 9ebc7a48) on the dev box with the trainer's own build_data under --fit-join,
from the corpus inputs (cache manifest, v9 corpus, lead corpus, speed corpus, --exclude 540:600), exactly as the pod's B-set argv names them. CPU. Writes OUT (own data) and prints its md5.
usage: l3_corpus_join.py OUT.npz"""
import json, os, subprocess, sys
KIT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, KIT)
import l3_chk as K
out = sys.argv[1]
t = K.load(K.REC + "/dscale10k/argv/B_s0.B.tmpl.json")
L = "F:/TanitAD-scratch/rollhist/lever3/labels/in/corpus"
sub = {"--cache-dir": "F:/TanitAD-scratch/speedread/cache", "--v9-train": L + "/v9_labels_train.npz", "--lead-train": L + "/lead_train4572.npz", "--speed-train": L + "/refcv6_speed_max_v8_train.jsonl",
       "--join-train": out, "--p10": K.A.P10, "--tracks": "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt", "--out": out + ".unused", "--device": "cpu"}
for k, v in sub.items():
    t[t.index(k) + 1] = v
t += ["--fit-join"]
p = out + ".argv.json"
json.dump({"argv": t}, open(p, "w"))
tree = "C:/Users/Admin/lever3/tree_cdb"
code = ("import sys,json;import v8f_mini_train as TR;ns=TR.build_parser().parse_args(json.load(open(sys.argv[1]))['argv']);cfg=TR.make_config(ns);"
        "store,J,wins,clips,info=TR.build_data(ns,cfg);print('join written',ns.join_train,len(clips),'clips',len(wins),'windows')")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=";".join([tree + "/stack", tree + "/stack/scripts", tree]))
rc = subprocess.run([K.PY, "-c", code, p], env=env, cwd=tree).returncode
if rc == 0 and os.path.isfile(out):
    import hashlib
    print("md5", hashlib.md5(open(out, "rb").read()).hexdigest())
sys.exit(rc)
