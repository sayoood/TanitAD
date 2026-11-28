"""PRE-GATE (NON-BINDING) hand ctx builder -- refcv8 A2 PROVISIONAL argv, tree d014ff2, Thor CPU only.
No key is read or created: `check` children sign nothing. The ctx is a HAND ctx (not `run`'s)."""
import json, sys
from pathlib import Path
T = "/home/nvidia/refcv6_run/d014ff2fd6"
W = "/home/nvidia/pregate_d014ff2"
COMMIT = "d014ff2fd695c306abf769af0572a17e8d713991"
sys.path[:0] = [f"{T}/stack/scripts", f"{T}/stack", f"{T}/taniteval"]
import launch_gate as G
tree = Path(T)
prof = G.PROFILES["refcv8"]
argv = G.load_argv_file(f"{W}/work/argv_PROVISIONAL.json")
tman = G.tree_manifest(tree)
tsha = G.tree_digest(tman)
dflt = G._resolve_repo_rel(tree, prof["nav_tau_record_default"])
options = {"cpu_only": True, "omp": 4, "min_free_gb": 0.0, "ram_wait_s": 0, "git_dir": None, "baseline": None,
           "eval_loader": str(G._resolve_repo_rel(tree, prof["eval_loader"])),
           "eval_stamps": f"{W}/work/stamps/config_refcv8_smoke_measure.json",
           "clock_reference": [str(G._resolve_repo_rel(tree, r)) for r in prof["clock_reference"]],
           "clock_manifest": {}, "nav_tau_record": str(dflt) if dflt.is_file() else None,
           "suite_verdict": None, "abort_free_gb": 0.0}
print("tree_sha256", tsha, "n_files", len(tman), "argv_sha256", G.argv_sha256(argv))
print("options", json.dumps(options)[:600])
for job in ("model", "clock", "suite", "pinned"):
    out = f"{W}/gate_out_{job}"
    Path(out).mkdir(parents=True, exist_ok=True)
    ctx = G.Ctx(profile="refcv8", tree=str(tree), commit=COMMIT, argv=argv, out_dir=out, path_map=[],
                tree_sha256=tsha, argv_sha256=G.argv_sha256(argv), options=options, arm=None)
    ctx.dump(Path(f"{W}/work/ctx_{job}.json"))
print("CTX_DONE")
