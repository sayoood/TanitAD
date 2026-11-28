"""PRIVGEOM-CLEAN rehearsal identity (SEEN banked data only; NOT part of the read): the new runner's fits against PRIVGEOM-SAFE's banked OOF logits and banked descriptive numbers.
  L_s0 / L_s1        == banked run/fits/L_dac_s{0,1}.npy                  (the L arm of the new runner = the registered L arm)
  LGu_s0             == banked run/descr/fits/drop0_dac_s0.npy            (the companion = SAFE's `drop0` arm: Delta_F +0.0276 [+0.0130, +0.0430])
  LGc_s0 Delta_F     == banked run/descr/centred_real_dacGVnomin.json     (the primary = SAFE's centred 9-column arm: +0.0352 [+0.0194, +0.0495])
    python pgc_rehearsal_identity.py --rehearsal <run_rehearsal dir> --out <json>
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                       # noqa: E402

import pgc_lib as C                                                                      # noqa: E402

P = C.P
SAFE = "C:/Users/Admin/par18/privgeom/run"


def cmp(a, b):
    return {"bit_identical": bool(np.array_equal(a, b)), "max_abs_diff": float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()), "n": int(a.size)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    A = np.load(f"{SAFE}/join/arrays.npz", allow_pickle=False)
    lab = np.load(f"{SAFE}/join/labels_sub.npy")
    an = P.Analyzer(lab, A["code"], A["log_idx"], int(A["log_idx"].max()) + 1)
    R = a.rehearsal
    out = {"schema": "privgeom-clean-rehearsal-identity-v1", "evidence": "MEASURED on the banked SEEN arrays (41 logs); a rehearsal of the code path, not a result"}
    out["L_s0_vs_banked"] = cmp(np.load(f"{R}/fits/L_s0.npy"), np.load(f"{SAFE}/fits/L_dac_s0.npy"))
    out["L_s1_vs_banked"] = cmp(np.load(f"{R}/fits/L_s1.npy"), np.load(f"{SAFE}/fits/L_dac_s1.npy"))
    out["LGu_s0_vs_banked_drop0"] = cmp(np.load(f"{R}/fits/LGu_s0.npy"), np.load(f"{SAFE}/descr/fits/drop0_dac_s0.npy"))
    st = lambda p: an.per_log_stats(np.load(p), C.ROW)                                   # noqa: E731
    sL, sU, sC = st(f"{R}/fits/L_s0.npy"), st(f"{R}/fits/LGu_s0.npy"), st(f"{R}/fits/LGc_s0.npy")
    out["delta_F_LGu_s0"] = an.diff(sU, sL, "Q4")
    out["delta_F_LGc_s0"] = an.diff(sC, sL, "Q4")
    ban = json.load(open(f"{SAFE}/descr/centred_real_dacGVnomin.json", encoding="utf-8"))["centred_delta_F"]
    out["banked_centred_delta_F"] = ban
    out["banked_drop0_delta_F"] = {"point": 0.027626618242166612, "lo": 0.013003737430799144, "hi": 0.043003106832428056}
    out["LGc_matches_banked_centred_to_1e-9"] = bool(all(abs(out["delta_F_LGc_s0"][k] - ban[k]) < 1e-9 for k in ("point", "lo", "hi")))
    out["LGu_matches_banked_drop0_to_1e-9"] = bool(all(abs(out["delta_F_LGu_s0"][k] - out["banked_drop0_delta_F"][k]) < 1e-9 for k in ("point", "lo", "hi")))
    # the RowView over the census `feat_f10.npy` (used by the READ instead of copying the 356 MB tokens) returns exactly the rows of the banked `feat_sub.npy`
    import pgc_run as RUN
    rv = RUN.RowView(np.load(RUN.FEAT, mmap_mode="r"), A["idx"])
    fs = np.load(f"{SAFE}/join/feat_sub.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(6):
        bi = rng.permutation(len(A["idx"]))[:256]
        ok = ok and bool(np.array_equal(rv[bi], np.stack([fs[i] for i in bi])))
    out["rowview_equals_banked_feat_sub"] = ok
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "evidence"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
