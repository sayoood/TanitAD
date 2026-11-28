"""q5_fill_controls.py (MM 2026-11-16): fill read_cfg.json["controls"] from MEASURED artifacts only, with a provenance record per key.
A key whose artifact is missing stays null (q5_read refuses). C2 (planted-direction fits) is filled only from --c2 <json> written by its own run.
usage: python q5_fill_controls.py --base F:/TanitAD-scratch/q5_416/run [--c7-pytest-log <log with 'N passed'>] [--c2 <c2 result json>]"""
import argparse
import hashlib
import json
import os
import re

import numpy as np


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--c7-pytest-log")
    ap.add_argument("--c2")
    a = ap.parse_args()
    B = a.base.rstrip("/") + "/"
    cfg_p = B + "read_cfg.json"
    cfg = json.load(open(cfg_p, encoding="utf-8"))
    c, prov = cfg["controls"], {}
    c0 = json.load(open(B + "c0/C0_RESULT.json", encoding="utf-8"))
    rf = json.load(open(B + "c0/refit/RESULT_C0_REFIT.json", encoding="utf-8"))
    ok0 = str(c0.get("overall", c0.get("PASS", ""))).upper() in ("PASS", "TRUE") and rf.get("C0_refit_bitwise") is True
    c["c0_differing_bytes"] = 0 if ok0 else None
    prov["c0_differing_bytes"] = {"from": ["c0/C0_RESULT.json", "c0/refit/RESULT_C0_REFIT.json"], "c0_overall": c0.get("overall"), "refit_bitwise": rf.get("C0_refit_bitwise")}
    eq = {}
    for s in cfg["samples"] if isinstance(cfg.get("samples"), list) else ("S1", "S2"):
        for t in ("416", "256p"):
            pa, pb = B + f"assembled/{s}_{t}/E.npy", B + f"c0/{s}/E.npy"
            eq[f"{s}_{t}"] = os.path.exists(pa) and os.path.exists(pb) and open(pa, "rb").read() == open(pb, "rb").read()
    c["c4_E_bitwise_equal_C0"] = all(eq.values()) if eq else None
    prov["c4_E_bitwise_equal_C0"] = {"pairs": eq}
    r6a = json.load(open(B + "c6a/RESULT_C6A.json", encoding="utf-8"))
    c["c6a_max_abs_diff"] = r6a["max_abs_diff"] if r6a.get("PASS") is not None else None
    prov["c6a_max_abs_diff"] = {"from": "c6a/RESULT_C6A.json", "n_clips": r6a.get("n_clips"), "n_frames": r6a.get("n_frames_compared")}
    r6 = json.load(open(B + "podlogs/controls_c6.json", encoding="utf-8"))
    c["c6b_mae_255"] = r6.get("c6b_mae_255")
    c["c6c_bitwise"] = r6.get("c6c_bitwise")
    prov["c6b_mae_255"] = prov["c6c_bitwise"] = {"from": "podlogs/controls_c6.json", "sha256": sha(B + "podlogs/controls_c6.json"), "n_clips": r6.get("n_clips")}
    if a.c7_pytest_log:
        t = open(a.c7_pytest_log, encoding="utf-8", errors="replace").read()
        m = re.search(r"(\d+) passed", t)
        failed = re.search(r"(\d+) (failed|error)", t)
        c["c7_ok"] = bool(m) and not failed and int(m.group(1)) >= 5
        prov["c7_ok"] = {"from": a.c7_pytest_log, "summary": t.strip().splitlines()[-1] if t.strip() else ""}
    s8p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_c8.json")
    s8 = json.load(open(s8p, encoding="utf-8"))
    c["c8_bitwise"] = s8.get("all_bitwise") is True and sum(1 for v in s8.values() if isinstance(v, dict) and v.get("bitwise_equal")) >= 3
    prov["c8_bitwise"] = {"from": "build/smoke_c8.json", "sha256": sha(s8p), "n_clips_bitwise": sum(1 for v in s8.values() if isinstance(v, dict) and v.get("bitwise_equal"))}
    ev = np.load(B + "c0/s0p_eval.npz", allow_pickle=False)
    cnt = {"eval_n": int(len(ev["y"])), "eval_pos": int(np.asarray(ev["y"], bool).sum()), "eval_episodes": int(len(np.unique(ev["eid"])))}
    tr = {}
    for s in ("S1", "S2"):
        y = np.load(B + f"assembled/{s}_416/y.npy")
        tr[s] = (int(len(y)), int(np.asarray(y, bool).sum()))
    same = len(set(tr.values())) == 1
    if same:
        cnt["train_n"], cnt["train_pos"] = list(tr.values())[0]
    c["c9_counts"] = cnt if same else None
    prov["c9_counts"] = {"per_sample_train": tr, "eval_npz_sha256": sha(B + "c0/s0p_eval.npz")}
    if a.c2:
        r2 = json.load(open(a.c2, encoding="utf-8"))
        c["c2_auroc"], c["c2_outcome"] = r2.get("c2_auroc"), r2.get("c2_outcome")
        prov["c2"] = {"from": a.c2, "sha256": sha(a.c2)}
    cfg["controls"] = c
    json.dump(cfg, open(cfg_p, "w", encoding="utf-8"), indent=1)
    json.dump(prov, open(B + "controls_provenance.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
