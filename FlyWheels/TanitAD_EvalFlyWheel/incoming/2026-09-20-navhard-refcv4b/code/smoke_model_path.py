#!/usr/bin/env python3
"""END-TO-END REACHABILITY SMOKE for the navhard model path — ckpt -> declared seam -> poses.

⛔ WHY (CLAUDE.md, "built, tested, and unreachable from its caller" is a CLASS):
the navhard run costs ~9 h. A model path that raises at scene 1 must raise HERE, in
90 s, on a 12-scene scratch bank — not at hour 4 after the floors have been paid for.
It exercises the REAL caller (``taniteval.bench.navsim.model_arms.run_model_arms``),
not a re-implementation, so what passes here is what the run will execute.

⭐ THE CONTROLS, and each must read a value that is known in advance:
  * A1 vs A4(BLIND) on the SAME tokens must DIFFER — a blind arm that matched A1
    would prove the pixels never reached the model (the frames-blind arm is the
    registered deliberate regression; if it is indistinguishable, the probe is
    broken, not the model).
  * every stage-1 row must be ``cv_standin`` and every stage-2 row ``refcv4b`` —
    the stand-in count is read from the SEAM ARTIFACT, never from a report.
  * poses must be finite and [8,3].

    python smoke_model_path.py --bank <scratch bank> --ckpt <path> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "taniteval"))
sys.path.insert(0, str(REPO / "stack"))


class _NullGap:
    """A gap launcher that answers CPU and never probes — the smoke must not touch the GPU."""

    def acquire(self):
        return "cpu"

    def should_back_off(self):
        return False

    def mark_own_usage(self, *a, **k):
        pass

    def backed_off(self):
        pass

    def stop(self):
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-stage2", type=int, default=6)
    ap.add_argument("--n-stage1", type=int, default=2)
    ap.add_argument("--ckpt-md5", default=None)
    ap.add_argument("--threads", type=int, default=6, help="torch CPU threads (run_model_arms' own knob)")
    ap.add_argument("--cache-max", type=int, default=None,
                    help="override FRAME_CACHE_MAX — used as the EVICTION control: a run with the "
                         "bound set below the working set must EVICT and must still produce "
                         "bit-identical poses, or the bound is not inert")
    a = ap.parse_args(argv)
    from taniteval.bench.navsim import model_arms as MA
    if a.cache_max is not None:
        MA.FRAME_CACHE_MAX = int(a.cache_max)

    prov_keys = set(__import__("pandas").read_parquet(
        os.path.join(a.bank, "frames_provenance.parquet")).scene_token)
    doc = json.load(open(a.inputs, encoding="utf-8"))
    toks = doc["tokens"]
    s1 = sorted(t for t, r in toks.items() if r["stage"] == 1)[:a.n_stage1]
    s2 = sorted((t for t, r in toks.items()
                 if r["stage"] == 2 and r["scene_token"] in prov_keys))[:a.n_stage2]
    if not s2:
        print("REFUSED: no stage-2 token of the export is present in this bank")
        return 2
    sub = {"tokens": {t: toks[t] for t in s1 + s2}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = MA.run_model_arms(arms={"A1": "A1_ego_cmd", "A4": "A4_blind_ego_cmd"}, doc=sub,
                            ckpt=a.ckpt, bank=a.bank, out_dir=out, gpu=_NullGap(),
                            threads=int(a.threads), ckpt_md5=a.ckpt_md5, log=print)
    rep = {"n_stage1": len(s1), "n_stage2": len(s2), "seconds": round(time.time() - t0, 1),
           "model": res["model"], "arms": res["arms"], "checks": {}}
    z1 = np.load(out / "A1.npz", allow_pickle=False)
    z4 = np.load(out / "A4.npz", allow_pickle=False)
    src1 = [str(x) for x in z1["source"]]
    st = {t: toks[t]["stage"] for t in s1 + s2}
    order = [str(x) for x in z1["token"]]
    rep["checks"]["stage1_rows_are_cv_standin"] = {
        "n_stage1_rows": sum(1 for t in order if st[t] == 1),
        "n_cv_standin": sum(1 for t, s in zip(order, src1) if st[t] == 1 and s == "cv_standin"),
        "expect": "equal — a camera arm cannot answer stage 1 on this corpus"}
    rep["checks"]["stage2_rows_are_model"] = {
        "n_stage2_rows": sum(1 for t in order if st[t] == 2),
        "n_refcv4b": sum(1 for t, s in zip(order, src1) if st[t] == 2 and s == "refcv4b"),
        "expect": "equal"}
    m2 = np.array([st[t] == 2 for t in order])
    p1, p4 = z1["poses"][m2], z4["poses"][m2]
    diff = np.abs(p1 - p4)
    rep["checks"]["A1_vs_A4_blind_DISCRIMINATING_CONTROL"] = {
        "n_scenes": int(m2.sum()),
        "n_scenes_differing": int((diff.reshape(len(p1), -1).max(1) > 1e-9).sum()),
        "max_abs_pose_diff_m": float(diff.max()),
        "expect": ("EVERY stage-2 scene differs; identical poses would prove the PIXELS NEVER "
                   "REACHED THE MODEL, i.e. a broken probe, not a blind arm")}
    rep["frame_cache"] = res.get("frame_cache")
    rep["checks"]["poses_finite"] = {"shape": list(p1.shape),
                                     "n_nonfinite_A1": int((~np.isfinite(p1)).sum()),
                                     "n_nonfinite_A4": int((~np.isfinite(p4)).sum()), "expect": "0 / 0"}
    c = rep["checks"]
    rep["verdict"] = ("PASS" if (c["stage1_rows_are_cv_standin"]["n_stage1_rows"] ==
                                 c["stage1_rows_are_cv_standin"]["n_cv_standin"]
                                 and c["stage2_rows_are_model"]["n_stage2_rows"] ==
                                 c["stage2_rows_are_model"]["n_refcv4b"]
                                 and c["A1_vs_A4_blind_DISCRIMINATING_CONTROL"]["n_scenes_differing"] ==
                                 c["A1_vs_A4_blind_DISCRIMINATING_CONTROL"]["n_scenes"]
                                 and c["poses_finite"]["n_nonfinite_A1"] == 0
                                 and c["poses_finite"]["n_nonfinite_A4"] == 0) else "FAIL")
    (out / "SMOKE.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({"verdict": rep["verdict"], "seconds": rep["seconds"], "checks": rep["checks"]}, indent=1))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
