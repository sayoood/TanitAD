"""PLUMBING SMOKE: the real M2b_s0 checkpoint through the CLI on the SYNTHETIC world (random features -> numbers are meaningless)."""
import sys, json, tempfile, time
from pathlib import Path
import v8f_t1_eval as T1
from tanitad.data import v8f_mini_data as D
root = Path(tempfile.mkdtemp())
D.make_synthetic_world(root, n_train=2, n_eval=3, T=80, seed=3)
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for m in (1, 5, 30):
    t = time.time()
    T1.main(["run", "--ckpt", "C:/Users/Admin/box2_ship/logs/v8f_mini/runs/M2b_s0/ckpt_last.pt", "--tag", f"synth_m{m}",
             "--tracks", str(root / "native_tracks.pt"), "--eval-cache", str(root), "--eval-grid", str(root / "eval_grid_windows.json"),
             "--eval-join", str(root / "join_eval.npz"), "--out-dir", str(out), "--replan-steps", str(m), "--batch", "12",
             "--max-windows", "24", "--n-boot", "50"])
    r = json.load(open(out / f"t1v8f_synth_m{m}.json"))
    h = r["t1v8f"]
    ade = {a: r["arms"][a]["intervals"]["metrics"]["ade_dense_m"]["mean"] if isinstance(r["arms"][a]["intervals"]["metrics"]["ade_dense_m"], dict) else r["arms"][a]["intervals"]["metrics"]["ade_dense_m"] for a in ("cl", "ol", "fl", "ha0")}
    print(f"m={m}: {time.time()-t:.1f}s  ADE(dense) {ade}  envelope-clip windows {h['windows_with_envelope_clip']}  tiers {r['tiers']}")
