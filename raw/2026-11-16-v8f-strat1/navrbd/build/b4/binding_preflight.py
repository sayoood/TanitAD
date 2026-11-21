"""binding_preflight.py -- the CPU part of `strat1_check.py mint` (the token binding, recomputed now) for N1D_s0, N1D_s1, N1Dx_s0 on the pod. No GPU, no token written.
Runs: render argv (as mint does), binding() (data sha256s, G-SUITE result for the commit, nav + nav_rb(d) builder reports, tree manifest, trainer sha), pairing_problems() against the banked N1R argv, G-HYG + hyg_rb.
Does NOT run: the trainer --selfcheck-only (G-DVB, G-LIVE) and the short crash/resume run (G-CKPT, G-EVAL): those need the GPU."""
import json
import os
import sys

TREE = "/root/strat1/navrbd/tree"
INP = "/root/strat1/inputs"
COMMIT = "0c3b374fb1036b0ca4ca7620f98137dd6c59cca4"
SUITE = "/root/strat1/gsuite_result_navrbd.json"
sys.path.insert(0, os.path.join(TREE, "stack", "scripts"))
sys.path.insert(0, os.path.join(TREE, "stack"))
import strat1_check as S  # noqa: E402

assert os.path.realpath(S.__file__).startswith(TREE + "/stack/"), S.__file__
res = {}
for kind, seed in (("N1D", "s0"), ("N1D", "s1"), ("N1Dx", "s0")):
    tag = f"{kind}_{seed}"
    toks = S.render(kind, seed, INP, f"/root/strat1/navrbd/runs/{tag}/train", os.path.join(INP, "strat1", "navtext_nav.npz"), os.path.join(INP, "strat1", "strat_labels_table.npz"),
                    base_argvs=os.path.join(INP, "frozen", "argvs_obd.json"), cache_dir="/root/data/cache_pod", rb_npz=os.path.join(INP, "strat1", S.rb_names(kind)[0]))
    b, bad = S.binding(kind, toks, TREE, COMMIT, SUITE, nav_report=os.path.join(INP, "strat1", "navtext_nav_report.json"), rb_report=os.path.join(INP, "strat1", S.rb_names(kind)[1]))
    ref = json.load(open(f"/root/strat1/navrb2/runs/N1R_{seed}/argv.json", encoding="utf-8"))
    pair = S.pairing_problems(kind, toks, ref)
    res[tag] = {"binding_problems": bad, "pair_problems": pair, "argv_sha256": b["argv_sha256"], "tree_manifest_sha256": b["tree_manifest_sha256"], "trainer_sha256": b["trainer_sha256"],
                "gsuite_result_sha256": b["gsuite_result_sha256"], "nav_rb_report_sha256": b.get("nav_rb_report_sha256"), "n_data_files_bound": len(b["data"])}
h = S.hyg() + S.hyg_rb()
res["G-HYG+hyg_rb"] = {"problems": h}
json.dump(res, open("/root/strat1/navrbd/b4/binding_preflight.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(1 if any(v.get("binding_problems") or v.get("pair_problems") or v.get("problems") for v in res.values()) else 0)
