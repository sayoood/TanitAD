"""CPU-only check that the TRAIN-DIAG remap builds the model + eval dataset on Thor (no forward, no GPU)."""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_repoint
L = r8_repoint.apply()
from pathlib import Path as P
t = time.time()
config = L.load_config(P(sys.argv[1]) / "config.json")
model, cfg, args, mrec = L.build_model(config, sys.argv[2], "cpu")
e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True)
print("[check] windows", len(e_ds), "episodes", len(e_eps), "map_fine", drec.get("map_fine", {}).get("verdict"), "pose_sync", drec.get("pose_sync", {}).get("n_covered"), "build_s", round(time.time() - t))
by = {}
for e_i, tt in e_ds.index:
    by[int(e_i)] = by.get(int(e_i), 0) + 1
print("[check] windows per episode min/max", min(by.values()), max(by.values()))
print("[check] OK")
