"""Build compact per-record tables for bank_s0 / bank_s1 (one npz member at a time) and validate them against bank.json.
Usage: python -I build_tables.py s0 [s1]
Writes raw/bank_table_<seed>.npz and raw/bank_table_<seed>.check.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bankio as B


def main(seeds):
    B.RAW.mkdir(parents=True, exist_ok=True)
    print("free RAM before:", round(B.require_ram(3.0), 2), "GB")
    tracks = B.load_tracks()
    print("tracks:", len(tracks), "clips")
    for seed in seeds:
        t0 = time.time()
        tab = B.build_table(seed, tracks)
        man = json.loads((B.BANK_DIRS[seed] / "bank.json").read_bytes())
        n = len(tab["tick"])
        chk = {"seed": seed, "n_records": int(n), "manifest_n_records": man["n_records"], "bank_sha256": str(tab["bank_sha256"])}
        assert n == man["n_records"], (n, man["n_records"])
        # reasons by tick vs the manifest
        rb = {}
        for tk in (1, 2):
            m = tab["tick"] == tk
            rb[str(tk)] = {B.REASONS[i]: int(((tab["reason"] == i) & m).sum()) for i in range(5)}
        chk["reasons_by_tick"] = rb
        chk["reasons_by_tick_equal_manifest"] = bool(rb == {k: {kk: vv for kk, vv in v.items()} for k, v in man["reasons_by_tick"].items()})
        # window pairing: each (clip, t0) must hold exactly ticks 1 and 2
        key = tab["clip_g"].astype(np.int64) * (1 << 20) + tab["t0"].astype(np.int64)
        u, cnt = np.unique(key, return_counts=True)
        chk["n_windows"] = int(len(u))
        chk["all_windows_have_two_records"] = bool((cnt == 2).all())
        chk["j_equals_t0_plus_10_tick"] = bool((tab["j"] == tab["t0"] + 10 * tab["tick"]).all())
        # v_own (p10f[-1,0]) vs kin3[0]
        d = np.abs(tab["v_own"] - tab["kin3_v"])
        fin = np.isfinite(d)
        chk["p10f_vs_kin3_speed"] = {"n_finite": int(fin.sum()), "n_nonfinite_v_own": int((~np.isfinite(tab["v_own"])).sum()),
                                      "n_nonfinite_kin3_v": int((~np.isfinite(tab["kin3_v"])).sum()),
                                      "max_abs_diff": float(d[fin].max()) if fin.any() else None, "n_diff_gt_1e-6": int((d[fin] > 1e-6).sum())}
        miss = [c for c in tab["clip_names"].tolist() if c not in tracks]
        chk["n_bank_clips"] = int(len(tab["clip_names"]))
        chk["bank_clips_missing_from_tracks"] = len(miss)
        chk["seconds"] = round(time.time() - t0, 1)
        np.savez(B.RAW / f"bank_table_{seed}.npz", **{k: v for k, v in tab.items()})
        (B.RAW / f"bank_table_{seed}.check.json").write_text(json.dumps(chk, indent=1))
        print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["s0", "s1"])
