"""REHEARSAL of G-SUITE-PINNED on a commit that does not yet contain the gate (the gate files are
unlanded), through the gate's own `run_job` (the RAM rule: start at >= 7.5 GB free on 3 samples,
abort below 4.0 GB) and `check` child. `run` would refuse -- correctly -- because the tree carrying
the gate is not the commit; this skips ONLY that refusal, and the evidence says so (its tree-digest
reason is expected here, never at a launch).

    python rehearse_pinned.py <tree with the gate> <commit> <out dir>
"""
import sys
from pathlib import Path

tree, commit, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(tree / "stack" / "scripts"))
import launch_gate as LG  # noqa: E402

argv = LG.load_argv_file(tree / "stack" / "ops" / "runs.d" / "refcv7-r101-s0.argv.json")
out.mkdir(parents=True, exist_ok=True)
ctx = LG.Ctx(profile="refcv7", tree=str(tree), commit=commit, argv=argv, out_dir=str(out),
             path_map=[], tree_sha256=LG.tree_digest(LG.tree_manifest(tree)),
             argv_sha256=LG.argv_sha256(argv),
             options={"git_dir": "C:/Users/Admin/tanitad-push/.git", "suite_work": "C:/lgs",
                      "cpu_only": True, "omp": 4, "min_free_gb": 7.5, "abort_free_gb": 4.0,
                      "ram_wait_s": 21600, "rehearsal": "the gate is not in this commit"},
             arm=None)
p = ctx.dump(out / "ctx.json")
LG.run_job(ctx, p, "pinned", ["G-SUITE-PINNED"], sys.executable)
ev = LG.read_json(out / "evidence" / "G-SUITE-PINNED.json")
print("ZZPINNED-" + ev["status"] + "ZZ")
for r in ev["reasons"]:
    print(" -", r[:400])
