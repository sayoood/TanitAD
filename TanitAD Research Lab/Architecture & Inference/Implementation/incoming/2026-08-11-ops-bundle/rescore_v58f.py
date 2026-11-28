"""v5.8f windows rescore: decision-grade episode-cluster bootstrap CIs (pod4).

Pulls the banked v58f windows from HF /v58f/, computes selected/oracle ADE with
episode-cluster bootstrap CIs (taniteval.ci) + the selgap block (taniteval.selgap
when fan windows carry per-candidate errors), per arm. Output JSON for the
registry §1.14 completion. CPU-only.
"""
import json

import numpy as np
import torch


def cluster_ci(vals, eids, n_boot=2000, seed=0, alpha=0.05):
    from taniteval.ci import episode_cluster_bootstrap
    return episode_cluster_bootstrap(np.asarray(vals, np.float64),
                                     np.asarray(eids), n_boot=n_boot,
                                     seed=seed, alpha=alpha)


def main():
    from huggingface_hub import hf_hub_download
    tok = open("/root/.cache/huggingface/token").read().strip()
    R = "Sayood/tanitad-flagship-v5f-w120"
    out = {"_tier": "T0", "_estimator": "episode-cluster bootstrap (taniteval.ci), 2000 draws",
           "_evidence_class": "MEASURED (ours; artifact = this JSON)", "arms": {}}
    for arm in ("v58f-rescorer-top8-kincost", "v58f-frozen-argmax"):
        wp = hf_hub_download(R, f"v58f/windows_{arm}.pt", token=tok)
        d = torch.load(wp, map_location="cpu", weights_only=False)
        pred = d["pred_dense"] if "pred_dense" in d else d["pred"]
        gt = d["gt_dense"] if "gt_dense" in d else d["gt"]
        eid = np.asarray(d["eid"])
        pred = torch.as_tensor(pred).float().numpy()
        gt = torch.as_tensor(gt).float().numpy()
        per_win = np.linalg.norm(pred - gt, axis=-1).mean(-1)      # [N]
        ci = cluster_ci(per_win, eid)
        blk = {"n_windows": int(len(per_win)), "n_episodes": int(len(set(eid.tolist()))),
               "selected_ade_mean": round(float(per_win.mean()), 4),
               "selected_ade_ci": ci}
        try:
            fp = hf_hub_download(R, f"v58f/v58f_fan_windows_{arm}.pt", token=tok)
            f = torch.load(fp, map_location="cpu", weights_only=False)
            if "fan_err" in f and "sel_idx" in f:
                from taniteval.selgap import selgap
                sg = selgap(np.asarray(f["fan_err"], np.float64),
                            np.asarray(f["sel_idx"]), np.asarray(f["eid"]))
                blk["selgap"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in sg.items() if not k.startswith("_")}
        except Exception as e:
            blk["selgap"] = f"n/a ({type(e).__name__})"
        out["arms"][arm] = blk
        print(arm, "->", json.dumps(blk)[:300], flush=True)
    json.dump(out, open("/workspace/v58f_rescore_ci.json", "w"), indent=1, default=str)
    print("RESCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
