"""Extract the HUMAN GT 2 s prefix for the 240 banked windows. 0 GPU: a data read.

The bank (`fan_bank_base_240w.npz`) was built with `with_gt=False`, so the SPEC's
ADE guard has no target. GT comes from the SAME corpus + the SAME window indices
(`wi`), through the driver's own `waypoint_targets`, so it is the identical map the
rerank probe uses -- never a second convention.
"""
import os, sys, json, numpy as np, torch

REPO = os.environ["TANITAD_REPO"]
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, REPO)
import importlib.util
spec = importlib.util.spec_from_file_location("rl_refcv3_min", os.path.join(REPO, "stack", "scripts", "rl_refcv3_min.py"))
D = importlib.util.module_from_spec(spec)
sys.modules["rl_refcv3_min"] = D
spec.loader.exec_module(D)

BANK = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Lab\Deployment & Optimization\Research\2026-09-05-veto-only-fan-safety\raw\fan_bank_base_240w.npz"
CKPT = "C:/Users/Admin/rl_refcv3_min/base/ckpt_step40284_frozen.pt"
CONFIG = "C:/Users/Admin/rl_refcv3_min/base/config.json"
EPS = "C:/Users/Admin/run_refcv3_ol/data/eval"
LAB = "C:/Users/Admin/run_refcv3_ol/data/s2_labels_v7.2_eval.jsonl.gz"
LEAD = "C:/Users/Admin/tanitad-wt/_lead_b1/b1_eval_lead_block.npz"
OUT = "C:/Users/Admin/collproj/out/gt_240w.npz"

z = np.load(BANK, allow_pickle=True)
wis = np.asarray(z["wi"]).astype(int).ravel().tolist()
eid_b = np.asarray(z["eid"]).astype(int).ravel()
v0_b = np.asarray(z["v0"], dtype=np.float64).ravel()
lead5_b = np.asarray(z["lead5"], dtype=np.float64)
has_b = np.asarray(z["has_lead"]).astype(bool)
print(f"[gt] {len(wis)} windows from the bank", flush=True)

D.LEAD_MODE = "track"
class _A:
    ckpt, config, expect_step = CKPT, CONFIG, 40284
model, cfg, _targs, prov = D.load(_A, "cpu")
D.ARM_HORIZONS[:] = [int(h) for h in cfg.core.trajectory.horizons]
print(f"[gt] horizons {D.ARM_HORIZONS[:4]} step={prov.get('step')}", flush=True)
del model
corp = D.open_corpus(EPS, LAB, cfg, prov, 6)
lead, _m, _i = D.load_lead_block(LEAD)

gt, v0, eids, lead5, has = [], [], [], [], []
B = 4
for i in range(0, len(wis), B):
    b = D.build_batch(corp, lead, wis[i:i+B], "cpu", with_gt=True)
    gt.append(b["gt_traj"].detach().cpu().numpy().astype(np.float64))
    v0.append(b["v0"].detach().cpu().numpy().astype(np.float64))
    lead5.append(b["lead_track"].detach().cpu().numpy().astype(np.float64))
    has.extend([bool(x) for x in b["has_lead"]])
    for wi in b["wis"]:
        eids.append(int(corp.ds.index[wi][0]))
    if (i // B) % 10 == 0:
        print(f"  [{i}/{len(wis)}]", flush=True)
gt = np.concatenate(gt, 0); v0 = np.concatenate(v0, 0); lead5 = np.concatenate(lead5, 0)
has = np.asarray(has, dtype=bool); eids = np.asarray(eids, dtype=np.int64)

# ---- IDENTITY CONTROLS: this GT read must line up with the BANK it will be joined to
d_v0 = float(np.abs(v0 - v0_b).max())
d_eid = int(np.abs(eids - eid_b).max())
d_lead = float(np.abs(lead5 - lead5_b).max())
d_has = int((has != has_b).sum())
print(f"[gt] IDENTITY vs bank: max|v0 diff|={d_v0:.3e}  max|eid diff|={d_eid}  "
      f"max|lead5 diff|={d_lead:.3e}  has_lead mismatches={d_has}", flush=True)
ok = (d_v0 < 1e-6) and (d_eid == 0) and (d_lead < 1e-4) and (d_has == 0)
print(f"[gt] JOIN-CONTROL {'PASS' if ok else 'FAIL'}", flush=True)
np.savez_compressed(OUT, gt=gt, v0=v0, eid=eids, wi=np.asarray(wis, dtype=np.int64),
                    lead5=lead5, has_lead=has,
                    join_ok=np.asarray([ok]), d_v0=np.asarray([d_v0]),
                    d_lead=np.asarray([d_lead]))
print(f"[gt] -> {OUT}  gt shape {gt.shape}", flush=True)
