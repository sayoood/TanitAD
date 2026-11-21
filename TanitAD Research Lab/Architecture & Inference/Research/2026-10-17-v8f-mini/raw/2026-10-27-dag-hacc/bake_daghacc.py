#!/usr/bin/env python3
"""bake_daghacc.py <seed 0|1>: bake_hacc.py's MECHANISM, UNCHANGED, applied to DAG (DECLARATIONS D4):
source ckpt -> V8FMini -> load_hacc_weights(JSON) -> torch.save({**ck, "model": state_dict incl. hacc_w, "hacc": prov}).
Only the M2b-specific PINS of bake_hacc.py are replaced by DAG's (source sha256, the DAG weight table and its hash); its step 5 (agreement with the banked
M2b HACC plan) has no DAG analogue and is replaced by C-H3 / C-H4 below.
Controls run here (all must hold, else rc 2):
  C-H4  the baked checkpoint differs from DAG's ONLY in the post-map buffer: state-dict keys == source keys + {hacc_w}; EVERY source tensor (model AND
        optimizer state) bit-identical; hacc_w == the JSON table in float32; non-model top-level fields equal except the added 'hacc' provenance.
  C-H3  a bake with w == 1 (in memory AND through a saved-and-reloaded checkpoint) reproduces plain DAG's plan exactly: max|d| = 0.0 m on traj30 AND
        traj60, on the first 24 eval windows (and the first 64).
  plus: the baked file through the harness's strict load == runtime-set weights, bit for bit on every output tensor; the post-map has power
        (baked plan != plain); the OTHER seed's table gives a different plan.
"""
import hashlib
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/stack")
sys.dont_write_bytecode = True
TREE = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"
PK = "E:/Projects/TanitAD-artifacts/daghacc"
sys.path.insert(0, f"{TREE}/stack")
sys.path.insert(0, f"{TREE}/taniteval")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

torch.set_num_threads(4)
RAW = f"{TREE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
S = sys.argv[1]
SRC = f"E:/Projects/TanitAD-artifacts/v8f-dagger/M2b_s{S}_dag/ckpt_last.pt"
SRC_SHA = [ln.split()[0] for ln in open(f"E:/Projects/TanitAD-artifacts/v8f-dagger/M2b_s{S}_dag/SHA256SUMS.txt") if ln.strip().endswith("ckpt_last.pt")][0]
WJ = f"{PK}/fit/weights_HACC_DAG_s{S}.json"
OTHER_WJ = f"{PK}/fit/weights_HACC_DAG_s{1 - int(S)}.json"
OUT = f"{PK}/ckpt/M2b_s{S}_dag_hacc.pt"
TMP = f"{PK}/pt_tmp"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def canon(w):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(w, np.float64)).tobytes()).hexdigest()


def build(ck):
    m = M.V8FMini(M.V8FMiniConfig(**ck["cfg"]), M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    m.load_state_dict(ck["model"])
    return m.eval()


def infer(m, b):
    with torch.no_grad():
        return {k: v for k, v in m.forward_infer(b).items() if torch.is_tensor(v)}


def same(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def tree_equal(a, b, path=""):
    """recursive bit-equality over nested dict / list / tuple / tensor / scalar; returns the list of differing paths"""
    bad = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: key sets differ"]
        for k in a:
            bad += tree_equal(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: len differs"]
        for i, (x, y) in enumerate(zip(a, b)):
            bad += tree_equal(x, y, f"{path}[{i}]")
    elif torch.is_tensor(a) and torch.is_tensor(b):
        if not torch.equal(a, b):
            bad.append(path)
    else:
        try:
            ok = (a == b)
            ok = bool(ok.all()) if hasattr(ok, "all") else bool(ok)
        except Exception:
            ok = False
        if not ok:
            bad.append(path)
    return bad


def eval_batch(ck, cfg, n):
    tracks = D.load_native_tracks("E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt")
    ej = D.load_join(f"{RAW}/join_eval141.npz")
    store = D.ClipStore("E:/Projects/TanitAD-artifacts/sfresh_inputs/cache", "eval", lru=32, grid=ck.get("grid", "pool4x10"), kin_mode=cfg.kin_mode, tracks=tracks)
    wins = D.eval_grid_windows(f"{RAW}/eval_grid_windows.json", ej)
    ds = D.V8FMiniWindows(store, ej, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    return D.collate([ds.item(wins[i][0], wins[i][1], 0, 0) for i in range(n)])          # ages fresh (a_t = a_s = 0), DAG's decisive T0 setting


os.makedirs(f"{PK}/ckpt", exist_ok=True)
os.makedirs(TMP, exist_ok=True)
assert sha256(SRC) == SRC_SHA, f"seed {S}: not the DAG checkpoint"
assert not os.path.exists(OUT), f"{OUT} exists"
w = M.read_hacc_weights(WJ)
ck = torch.load(SRC, map_location="cpu", weights_only=False)
cfg = M.V8FMiniConfig(**ck["cfg"])
m_src = build(ck)
assert m_src.hacc_w is None
m_bake = build(ck)
m_bake.load_hacc_weights(WJ)
prov = {"prereg": "Project Steering/PREREG_V8F_DAG_HACC.md", "weights_json_sha256": sha256(WJ), "weights_canonical_sha256": canon(w), "source_ckpt_sha256": SRC_SHA,
        "tier_of_the_fit": "T0 plans on HBLEND's W_fit (141 train clips / 9,582 windows, cycle ages), train-fitted; no eval window enters",
        "what": "forward_infer returns P + w (O - P) with this [30, 2] table (column 0 along, column 1 cross); training (_forward) is untouched; bake mechanism = bake_hacc.py (2026-10-26) with DAG's pins"}
torch.save({**ck, "model": m_bake.state_dict(), "hacc": prov}, OUT)
rep = {"seed": f"DAG_s{S}", "out": OUT, "out_sha256": sha256(OUT), "out_bytes": os.path.getsize(OUT), "source_ckpt_sha256": SRC_SHA, "weights_json_sha256": prov["weights_json_sha256"]}
# ---- C-H4: bake scope
ck2 = torch.load(OUT, map_location="cpu", weights_only=False)
extra, missing = set(ck2["model"]) - set(ck["model"]), set(ck["model"]) - set(ck2["model"])
model_bad = [k for k in ck["model"] if k in ck2["model"] and not torch.equal(ck2["model"][k], ck["model"][k])]
other_bad = tree_equal({k: v for k, v in ck2.items() if k not in ("model", "hacc")}, {k: v for k, v in ck.items() if k != "model"})
hw_ok = bool(torch.equal(ck2["model"]["hacc_w"], torch.as_tensor(w, dtype=torch.float32)))
rep["C_H4"] = {"extra_state_dict_keys": sorted(extra), "missing_state_dict_keys": sorted(missing), "n_source_model_tensors": len(ck["model"]),
               "model_tensors_not_bit_identical": model_bad, "non_model_fields_not_identical (opt, step, cfg, ...)": other_bad,
               "toplevel_added": sorted(set(ck2) - set(ck)), "hacc_w_equals_json_table_float32": hw_ok,
               "pass": bool(extra == {"hacc_w"} and not missing and not model_bad and not other_bad and hw_ok and sorted(set(ck2) - set(ck)) == ["hacc"])}
# ---- the baked file through the harness's strict load == runtime-set weights
b64 = eval_batch(ck, cfg, 64)
m_file = build(ck2)
assert m_file.hacc_active
y_file, y_rt, y_src = infer(m_file, b64), infer(m_bake, b64), infer(m_src, b64)
rep["file_vs_runtime_bit_identical_all_outputs_64win"] = same(y_file, y_rt)
rep["traj_o_equals_plain_traj"] = bool(torch.equal(y_file["traj30_o"], y_src["traj30"]) and torch.equal(y_file["traj60_o"], y_src["traj60"]))
rep["power_max_abs_baked_vs_plain_traj30_m"] = float((y_file["traj30"] - y_src["traj30"]).abs().max())
m_oth = build(ck)
m_oth.load_hacc_weights(OTHER_WJ)
rep["other_seed_table_differs_max_abs_m"] = float((infer(m_oth, b64)["traj30"] - y_file["traj30"]).abs().max())
# ---- C-H3: w == 1 reproduces plain DAG exactly (in memory, and through a saved + strictly reloaded checkpoint)
m_one = build(ck)
m_one.set_hacc_weights(np.ones((30, 2)))
tmp1 = f"{TMP}/w1_seed{S}.pt"
torch.save({**ck, "model": m_one.state_dict()}, tmp1)
m_one_f = build(torch.load(tmp1, map_location="cpu", weights_only=False))
y_one, y_one_f = infer(m_one, b64), infer(m_one_f, b64)
c3 = {}
for nm, n in (("first_24", 24), ("first_64", 64)):
    c3[nm] = {k: float((y_one[k][:n] - y_src[k][:n]).abs().max()) for k in ("traj30", "traj60")}
    c3[nm + "_through_saved_checkpoint"] = {k: float((y_one_f[k][:n] - y_src[k][:n]).abs().max()) for k in ("traj30", "traj60")}
c3["every_other_output_equal_plain_64win"] = bool(all(torch.equal(y_one[k], y_src[k]) for k in y_src if k not in ("traj30", "traj60", "traj30_o", "traj60_o")))
c3["pass"] = bool(all(v == 0.0 for nm in ("first_24", "first_64", "first_24_through_saved_checkpoint", "first_64_through_saved_checkpoint") for v in c3[nm].values())
                  and c3["every_other_output_equal_plain_64win"])
rep["C_H3"] = c3
if os.path.exists(tmp1):
    os.remove(tmp1)          # created above by this script
rep["all_pass"] = bool(rep["C_H4"]["pass"] and c3["pass"] and rep["file_vs_runtime_bit_identical_all_outputs_64win"] and rep["traj_o_equals_plain_traj"]
                       and rep["power_max_abs_baked_vs_plain_traj30_m"] > 1e-2 and rep["other_seed_table_differs_max_abs_m"] > 1e-3)
json.dump(rep, open(f"{PK}/ckpt/BAKE_REPORT_seed{S}.json", "w"), indent=1)
print("ZZBAKE", S, json.dumps({k: rep[k] for k in ("out_sha256", "all_pass")}), "C_H4", rep["C_H4"]["pass"], "C_H3", c3["pass"], flush=True)
sys.exit(0 if rep["all_pass"] else 2)
