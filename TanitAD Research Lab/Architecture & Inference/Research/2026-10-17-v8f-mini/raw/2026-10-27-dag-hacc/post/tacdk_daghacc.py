"""Beside: (a) arm means for TAC-LAT / TAC-LON / DK-VIOL / DK-HW / DK-TG (decisive config) for DAG-HACC, DAG and P10, and (b) the paired rows DAG-HACC - DAG on those five rows,
with t1kf_read_a1's OWN prep() and rows() (imported, not re-implemented), as dagreread/code/tacdk_means.py did for DAG. Decisive config only."""
import importlib.util
import json
import os
import sys

os.environ.setdefault("RR_READERS", "E:/Projects/TanitAD-artifacts/dagreread/reader_src")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagreread/code")
import numpy as np  # noqa: E402
import rr_lib as R  # noqa: E402

PK = "E:/Projects/TanitAD-artifacts/daghacc"
DAGP = "E:/Projects/TanitAD-artifacts/dagreread/packed"
G = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/gitdumps"
argv0 = sys.argv[:]
sys.argv = ["t1kf", "--tree", R.TREE, "--t1k-read", f"{R.RD}/t1k_read.py", "--lead-block", f"{G}/b1_eval_lead_block_k30.npz", "--out", "unused.json"]
spec = importlib.util.spec_from_file_location("t1kf", f"{R.RD}/t1kf_read_a1.py")
T = importlib.util.module_from_spec(spec)
sys.modules["t1kf"] = T
spec.loader.exec_module(T)
sys.argv = argv0
DK, LM = T.DK, T.LM
out = {"tier": "T1-kin decisive (m=10, causal bucket)", "seeds": {}}
for s in R.SEEDS:
    dh = R.load_npz(f"{PK}/packed/dumps_DAGHACC_s{s}_m10_causal.npz")
    dg = R.load_npz(f"{DAGP}/dumps_DAG_s{s}_m10_causal.npz")
    join = np.load(f"{R.TREE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz", allow_pickle=False)
    clip = join["clip"].astype(str)[dh["clip_index"]]
    blk = DK.load_block(f"{G}/b1_eval_lead_block_k30.npz")
    Jd = DK.join_block(blk, clip, 2 * dh["ws"].astype(np.int64), dh["v0"], k=T.K, dt=T.DT)
    hdk = LM.distance_keeping(dh["g"][:, :T.K].astype(np.float64), Jd["leads"], Jd["lead_lens"], Jd["speeds"], T.DT)
    cx = T.Ctx(dh["g"].astype(np.float64), dh["eid"], Jd, hdk)
    Ps = {"DAGHACC": T.prep("DAGHACC", dh["cl"], cx), "DAG": T.prep("DAG", dg["cl"], cx), "P10": T.prep("P10", dh["fl"], cx)}
    o = {"means": {}}
    for nm, P_ in Ps.items():
        o["means"][nm] = {"TAC-LAT": float(P_["ind"]["TAC-LAT"].mean()), "TAC-LON": float(P_["ind"]["TAC-LON"].mean()), "DK-VIOL_share_of_at_risk": P_["lv"].get("share_of_at_risk"),
                          "DK-HW_mean_abs_err_m_own_finite": float(np.nanmean(P_["abs"]["DK-HW"])), "DK-TG_mean_abs_err_s_own_finite": float(np.nanmean(P_["abs"]["DK-TG"]))}
    o["rows_DAGHACC_minus_DAG"] = T.rows(Ps["DAGHACC"], Ps["DAG"], dh["eid"])
    o["identity_DAGHACC_DAGHACC_exactly_zero"] = bool(all(T.is_zero(v) for v in T.rows(Ps["DAGHACC"], Ps["DAGHACC"], dh["eid"]).values()))
    out["seeds"][f"s{s}"] = o
    print(f"s{s}", {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in o["means"].items()}, flush=True)
    print(f"s{s} DAGHACC - DAG", {k: (round(v.get("delta", float('nan')), 4), v.get("separated")) for k, v in o["rows_DAGHACC_minus_DAG"].items()}, flush=True)
json.dump(out, open(f"{PK}/raw/tacdk_daghacc.json", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
print("ZZTACDK-DONE")
