#!/usr/bin/env python3
"""WRAPPER (new file; the frozen s1_read_v2.py / s1_read.py are not touched).  HC's S1-F on the SFT-held-out frames only (PREREG_VDAC_P6_A5 7.5 pre-registered fallback).
Why it is needed: s1_read_v2.load_real REFUSES any arm that does not cover all of the export's keys (`if not got.all()` -> "does not cover the export's N keys"), and HC has no head_init0_b
(the extra-log init head is not built), so HC's initial head cannot be an arm of the frozen manifest.  Here: the frozen load_real reads HC WITHOUT an init arm (manifest_a5_hc.json), and this
wrapper then runs the frozen run_read on the frame subset covered by `hc_init_a`, with the trained arm and its initial head, and returns the S1-F block.
DESIGN CHOICE (UNVERIFIED against the card's wording, MM to confirm): the log-cluster bootstrap is redrawn over the logs present in the subset (same seed, same n_boot), since the frozen
code resamples the groups of the frames it is given.  Core = s1f_on_subset (tested on synthetic fixtures); main() reads real files and is NOT tested without data.
usage: S1_READ_V1=<frozen s1_read.py> python a5_hc_s1f_subset.py --manifest manifest_a5_hc.json --init-key hc_init_a --trained HC-s0 --out <dir>"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("S1_READ_V2_DIR", HERE))
import s1_read_v2 as R  # noqa: E402  (refuses a v1 whose sha256 is not the frozen one)


def subset(d, keep):
    keep = np.asarray(keep, bool)
    out = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v) for k, v in d.items() if k != "arms"}
    out["arms"] = {k: {kk: np.asarray(vv)[keep] for kk, vv in a.items()} for k, a in d["arms"].items()}
    out["all_logs"] = sorted(set(out["frame_log"].astype(str)))
    return out


def s1f_on_subset(d, keep, trained_key, init_key, cell, n_boot=R.R.N_BOOT, seed=R.R.SEED):
    """S1-F (trained minus its OWN initial head, both 95 % lower bounds > 0) on the frames `keep`; arms = trained + init only (no shuf: the verdict of this sub-read is not used)."""
    ds = subset(d, keep)
    spec = [{"key": trained_key, "name": "TRAINED", "role": "trained", "cell": cell, "init": init_key},
            {"key": init_key, "name": "INIT", "role": "init", "cell": cell}]
    r = R.run_read(ds, n_boot=n_boot, seed=seed, arms=spec)
    s1f = r["per_arm"]["TRAINED"]["S1_F"]
    cities = sorted({ds["city_of_log"][l] for l in set(ds["frame_log"].astype(str))})
    absent = sorted(set(R.R.CITY_WEIGHT) - set(cities))
    return {"S1_F": s1f, "cities_in_subset": cities, "cities_absent": absent,
            "p1_defined": bool(absent == []) and bool(np.isfinite(s1f["p1"]["lower95"])),
            "p1_note": "the frozen S1-P1 is CITY-REWEIGHTED: a city with no frame in the subset leaves it NaN (hold False); then S1-F is NOT READABLE on this subset, say so, never read it as a fail" if absent else "", "n_frames": int(len(ds["frame_log"])), "n_groups": r["n_groups"], "estimator": "paired log-cluster bootstrap redrawn over the subset's logs",
            "scope": "SFT-held-out subset only (PREREG 7.5 fallback), not the 3,207 H48 frames"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--init-key", default="hc_init_a"); ap.add_argument("--trained", default="HC-s0")
    ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=R.R.N_BOOT)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    d, audit = R.load_real(man)
    spec = {s["name"]: s for s in d.pop("_spec")}
    tr = spec[a.trained]
    ent = man["files"][a.init_key]
    ex = np.load(man["files"]["export"]["path"], allow_pickle=False)
    kpos = {k: i for i, k in enumerate(zip([str(x) for x in ex["log"]], [str(x) for x in ex["token"]]))}
    z = np.load(ent["path"], allow_pickle=False)
    nx, ny, _ = R.cell_grid(tr["cell"])
    if not (np.asarray(z["step"]) == 0).all() or not np.isfinite(z["logit"]).all() or z["logit"].shape[1:] != (nx, ny):
        raise SystemExit("REFUSED: init file step / finiteness / shape")
    N = len(kpos)
    lg, sd, keep = np.zeros((N, nx, ny), np.float32), np.zeros((N, nx, ny), np.float32), np.zeros(N, bool)
    for i, k in enumerate(zip([str(x) for x in z["log"]], [str(x) for x in z["token"]])):
        j = kpos[k]
        if keep[j]:
            raise SystemExit(f"REFUSED: duplicate key {k}")
        keep[j], lg[j], sd[j] = True, z["logit"][i], z["sdf_m"][i]
    d["arms"]["__init"] = {"logit": lg, "sdf": sd}
    res = s1f_on_subset(d, keep, tr["key"], "__init", tr["cell"], n_boot=a.n_boot)
    res.update({"init_file": ent["path"], "init_sha256": ent["sha256"], "manifest": os.path.abspath(a.manifest), "wrapper_sha256": R.sha256_file(os.path.abspath(__file__))})
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "hc_s1f_subset.json"), "w"), indent=1, default=float)
    print(json.dumps(res, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
