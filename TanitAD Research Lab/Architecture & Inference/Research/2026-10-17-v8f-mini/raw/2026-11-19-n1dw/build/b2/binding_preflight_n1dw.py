"""binding_preflight_n1dw.py -- the CPU part of `strat1_check.py mint` (token binding recomputed now) for N1DW4_s0/s1 and N1DW2_s0/s1 on the pod. No GPU, no token written.
Runs: render argv (as mint does), binding() (data sha256s, G-SUITE result for 134f5da1, nav + nav_rb(d) builder reports, tree manifest, trainer sha), the N1DW pairing rule against the BANKED N1D argv of the seed, G-HYG + hyg_rb + hyg_w.
Does NOT run the trainer --selfcheck-only (G-DVB, G-LIVE) or the 4-step crash/resume (G-CKPT, G-EVAL): those need the GPU (mint_n1dw.sh)."""
import json
import os
import sys

TREE = "/root/strat1/n1dw/tree"
INP = "/root/strat1/inputs"
COMMIT = "134f5da1ee50a3c2fa242bb340a26dd6e4072ade"
SUITE = "/root/strat1/n1dw/gsuite_result_n1dw.json"
OUT = "/root/strat1/n1dw/b2/binding_preflight.json"
sys.path.insert(0, os.path.join(TREE, "stack", "scripts"))
sys.path.insert(0, os.path.join(TREE, "stack"))
import strat1_check as S  # noqa: E402

assert os.path.realpath(S.__file__).startswith(TREE + "/stack/"), S.__file__
res = {}
for kind, seed in (("N1DW4", "s0"), ("N1DW4", "s1"), ("N1DW2", "s0"), ("N1DW2", "s1")):
    tag = f"{kind}_{seed}"
    toks = S.render(kind, seed, INP, f"/root/strat1/n1dw/runs/{tag}/train", os.path.join(INP, "strat1", "navtext_nav.npz"), os.path.join(INP, "strat1", "strat_labels_table.npz"),
                    base_argvs=os.path.join(INP, "frozen", "argvs_obd.json"), cache_dir="/root/data/cache_pod", rb_npz=os.path.join(INP, "strat1", S.rb_names(kind)[0]))
    b, bad = S.binding(kind, toks, TREE, COMMIT, SUITE, nav_report=os.path.join(INP, "strat1", "navtext_nav_report.json"), rb_report=os.path.join(INP, "strat1", S.rb_names(kind)[1]))
    ref = json.load(open(f"/root/strat1/navrbd/runs/N1D_{seed}/argv.json", encoding="utf-8"))
    pair = S.w_pairing_problems(kind, toks, ref)
    res[tag] = {"binding_problems": bad, "pair_problems": pair, "argv_sha256": b["argv_sha256"], "tree_manifest_sha256": b["tree_manifest_sha256"], "trainer_sha256": b["trainer_sha256"],
                "gsuite_result_sha256": b["gsuite_result_sha256"], "nav_rb_report_sha256": b.get("nav_rb_report_sha256"), "n_data_files_bound": len(b["data"])}
h = S.hyg() + S.hyg_rb() + S.hyg_w()
res["G-HYG+hyg_rb+hyg_w"] = {"problems": h}
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(1 if any(v.get("binding_problems") or v.get("pair_problems") or v.get("problems") for v in res.values()) else 0)
