"""BAR-A step 0 -- REPRODUCTION CHECK, run BEFORE any new number is quoted.

The brief's rule: reproduce v4's committed 30k numbers first. If they do not
reproduce, STOP and report that instead -- it is the more important finding.

Committed targets (MEASURED, `.../2026-07-26-v4-30k-gate/` diagnostics JSON,
re-read from `rescued_perwindow/*_v4_diagnostics.json`):

    goal-ORACLE   ade_0_2s (4wp)      0.6423     oracle_in_fan (4wp)  0.2330
    goal-PRODUCED ade_0_2s (4wp)      0.8563     oracle_in_fan (4wp)  0.2505

This re-runs the FORWARD PASS (not just a re-reduction of the persisted
windows_*.pt), so it is an independent path: encoder -> imagination ->
decoder -> select, on the same checkpoint, same 40 episodes, same stride 8.

TREE PROVENANCE is stamped into the output: the eval pod holds SIX copies of
`eval_flagship_v4.py` and two sizes of `flagship_v15.py`. `/root/v4eval/stack`
is the verified-current tree. Every imported module's md5 + path is recorded.
"""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch

STACK = "/root/v4eval/stack"
sys.path.insert(0, STACK + "/scripts")
sys.path.insert(0, STACK)
sys.path.insert(0, "/root/taniteval")

import eval_flagship_v4 as E  # noqa: E402
import driving_diagnostic as dd  # noqa: E402

CKPT = "/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt"
HCFG = "/workspace/_v4gate/flagship-v4-fromscratch-30k/config.json"
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
ANCH = "/root/models/flagship-v4-fromscratch-15k/flagship_v4_anchors_dense.pt"
DEV = "cuda"
OUT = "/root/bara/repro_v4_30k.json"

TARGETS = {
    "oracle": {"ade_0_2s": 0.6423, "oracle_in_fan": 0.2330},
    "produced": {"ade_0_2s": 0.8563, "oracle_in_fan": 0.2505},
}
TOL = 0.001


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    Path("/root/bara").mkdir(exist_ok=True)
    res = {
        "_experiment": "BAR-A step 0: reproduction of the committed v4-30k numbers",
        "_evidence_class": "MEASURED (ours)",
        "_rule": "no new number is quotable unless these reproduce to <= %.4f m" % TOL,
        "_host": platform.node(),
        "_python": sys.version.split()[0],
        "_torch": torch.__version__,
        "_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    # ---- TREE PROVENANCE: which tree did we actually import? --------------
    mods = {}
    for name in ("eval_flagship_v4", "driving_diagnostic", "goal_modes",
                 "train_flagship_v4", "refb_labels", "flagship_v4_data",
                 "tanitad.models.flagship_v15", "tanitad.models.flagship_v4",
                 "tanitad.refs.refc", "taniteval.ci"):
        try:
            m = __import__(name, fromlist=["__file__"])
            f = getattr(m, "__file__", None)
            if f:
                mods[name] = {"path": f, "md5": md5(f)}
        except Exception as e:  # pragma: no cover
            mods[name] = {"error": repr(e)}
    res["_module_provenance"] = mods
    res["_ckpt"] = {"path": CKPT, "bytes": Path(CKPT).stat().st_size,
                    "md5": md5(CKPT)}
    res["_stack_root"] = STACK
    print("[repro] tree provenance stamped", flush=True)

    cfg = E._eval_cfg()
    plan = E._plan(cfg)
    ds_val = E.build_val_dataset_v4(VAL, cfg, plan)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    world, grounding, head, step, hcfg, goal_head = E.load_v4_from_ck(
        ck, DEV, head_config_path=HCFG, anchors_dense_path=ANCH)
    del ck
    res["_ckpt_step"] = int(step)
    res["_head_params"] = int(sum(p.numel() for p in head.parameters()))

    res["arms"] = {}
    ok_all = True
    for gm in ("oracle", "produced"):
        t = time.time()
        data, diag = E.collect_planner(
            world, grounding, head, ds_val, DEV, dd, episodes=40, stride=8,
            batch=16, goal_mode=gm, goal_head=goal_head)
        ade = float((data["pred"] - data["gt"]).norm(dim=-1).mean(1).mean())
        oif = float(diag["wp4_oracle_ade_0_2s"])
        miss = float(((data["pred"] - data["gt"]).norm(dim=-1)[:, -1] > 2.0)
                     .float().mean())
        tg = TARGETS[gm]
        row = {
            "n_windows": int(data["pred"].shape[0]),
            "n_episodes": len(set(int(x) for x in data["eid"])),
            "wallclock_s": round(time.time() - t, 1),
            "ade_0_2s_recomputed": round(ade, 4),
            "ade_0_2s_committed": tg["ade_0_2s"],
            "ade_0_2s_abs_diff": round(abs(ade - tg["ade_0_2s"]), 5),
            "oracle_in_fan_recomputed": round(oif, 4),
            "oracle_in_fan_committed": tg["oracle_in_fan"],
            "oracle_in_fan_abs_diff": round(abs(oif - tg["oracle_in_fan"]), 5),
            "miss_at_2m_recomputed": round(miss, 4),
            "selector_waste_recomputed": round(ade - oif, 4),
            "wm_seam_norm_ratio_max": diag["seam_norm_ratio_max"],
        }
        row["PASS"] = (row["ade_0_2s_abs_diff"] <= TOL
                       and row["oracle_in_fan_abs_diff"] <= TOL)
        ok_all &= row["PASS"]
        res["arms"][gm] = row
        torch.save(data, f"/root/bara/repro_windows_30k_{gm}.pt")
        print(f"[repro] {gm}: ade={ade:.4f} (committed {tg['ade_0_2s']}) "
              f"oif={oif:.4f} (committed {tg['oracle_in_fan']}) "
              f"PASS={row['PASS']}", flush=True)

    res["REPRODUCES"] = bool(ok_all)
    res["_deployable_waste_recomputed"] = res["arms"]["produced"][
        "selector_waste_recomputed"]
    res["_v1_reference_ade_0_2s_full_set"] = 0.4271
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("_module_provenance",)}, indent=2))


if __name__ == "__main__":
    main()
