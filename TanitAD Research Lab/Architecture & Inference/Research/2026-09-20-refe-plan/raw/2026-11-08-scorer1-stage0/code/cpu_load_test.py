"""CPU only (CUDA_VISIBLE_DEVICES=-1): the trainer's argv parsing + its own bank construction (scorer_finetune.py lines ~1396-1434), no model forward."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("REFE_BACKBONE_ROOT", "E:/Projects/TanitAD/data/backbones")
sys.path.insert(0, "F:/tanitad_scratch/stage0_20261108")
import common_args as C
sys.path.insert(0, C.PK + "/refe")
import scorer_finetune as sf
import train as T
import ckpt_io
from model import REFeConfig
ap = sf.build_parser()
for pf in (False, True):
    a = ap.parse_args(C.argv(pf))
    print("parsed ok preflight=%s refusals: %s" % (pf, sf.epo_refusals(a, ap) + sf.epo_con_refusals(a, ap) + sf.pair_refusals(a) + sf.hn_refusals(a, ap)))
a = ap.parse_args(C.argv(False))
cfg = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone(a.backbone), a.ckpt)
op = T.OnPolicyBank(a.onpolicy, cfg.n_proposals, cfg.horizon_steps, lane_labels=None, pdm_labels=a.pdm_labels_train, teacher_lane_labels=a.teacher_lane_train)
ho = T.OnPolicyBank(a.heldout, cfg.n_proposals, cfg.horizon_steps, lane_labels=a.lane_labels_heldout, pdm_labels=a.pdm_labels_heldout, teacher_lane_labels=a.teacher_lane_heldout)
print("train bank: sets", len(op.by), "pdm", op.n_pdm, "stale", op.n_pdm_stale, "superseded", op.n_pdm_superseded, "tlane", op.n_tlane, "n_cols", op.n_cols)
print("heldout bank: sets", len(ho.by), "pdm", ho.n_pdm, "stale", ho.n_pdm_stale, "superseded", ho.n_pdm_superseded, "tlane", ho.n_tlane, "tlane_stale", ho.n_tlane_stale, "lane", ho.n_lane, "lane_stale", ho.n_lane_stale, "n_cols", ho.n_cols)
ds = T.TargetBank(a.targets, a.images, cfg, calib=a.calib, onpolicy=op)
ho_logs = set(open(a.heldout_logs, encoding="utf-8").read().split())
tr_idx = [i for i, r in enumerate(ds.rows) if sf.key(r) in op.by and r.get("log_name") not in ho_logs and key_ok(r) ] if False else [i for i, r in enumerate(ds.rows) if sf.key(r) in op.by and r.get("log_name") not in ho_logs and sf.key(r) in op.pdm_keys]
ho_idx = [i for i, r in enumerate(ds.rows) if sf.key(r) in ho.by and sf.key(r) in ho.pdm_keys]
print("ds rows", len(ds.rows), "tr_idx", len(tr_idx), "ho_idx", len(ho_idx))
print("CPU_LOAD_TEST_DONE")
