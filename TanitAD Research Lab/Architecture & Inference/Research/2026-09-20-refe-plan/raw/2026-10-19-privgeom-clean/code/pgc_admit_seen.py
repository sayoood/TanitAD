"""PRIVGEOM-CLEAN: the admissibility check's rehearsal on PRIVGEOM-SAFE's banked SEEN arrays (41 logs; labels opened by that arm).  Opens no label of the 76.  Writes run/admit_seen.json:
the planted controls (with the top proof rules by recall), the RED arm (disabled check), the mutation table, the family ablation, and the candidate's best proof by family."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                       # noqa: E402

import pg_geom as G                                                                      # noqa: E402
import pgc_admit as AD                                                                   # noqa: E402
import pgc_lib as C                                                                      # noqa: E402

SAFE = "C:/Users/Admin/par18/privgeom"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    A = np.load(f"{SAFE}/run/join/arrays.npz", allow_pickle=False)
    lab = np.load(f"{SAFE}/run/join/labels_sub.npy")[..., C.ROW_COL]
    g = A["g"].astype(np.float64)
    block, leaky = C.clean_block(g), G.unsigned(g[..., :11])
    out = {"schema": "privgeom-clean-admit-seen-v1", "evidence": "MEASURED on PRIVGEOM-SAFE's banked SEEN arrays (41 logs, 316,224 proposals); no label of the 76 read", "n_proposals": int(lab.size), "n_unsafe": int((lab == 0).sum())}
    t0 = time.time()
    base = AD.a_test(block, lab, leaky)
    out["base"] = {"admissible": base["admissible"], "controls_ok": base["controls_ok"], "controls": base["controls"], "seconds": time.time() - t0,
                   "candidate": {k: base["candidate"][k] for k in ("flagged", "hits_by_family", "best_unsafe_proof_by_family", "top", "n_rules", "n_unsafe", "n_proposals", "safe_class_proofs_descriptive")}}
    print("base done", base["admissible"], base["controls_ok"], f"{time.time() - t0:.0f} s", flush=True)
    red = AD.a_test(block, lab, leaky, cfg=AD.AdmitConfig(disabled=True))
    out["red_arm_disabled"] = {"controls_ok": red["controls_ok"], "leaked_through": sorted(k for k, v in red["controls"].items() if v["expected_flagged"] and not v["flagged"]),
                               "wrongly_flagged": sorted(k for k, v in red["controls"].items() if not v["expected_flagged"] and v["flagged"])}
    mut = {}
    for name, cfg in AD.mutations().items():
        r = AD.a_test(block, lab, leaky, cfg=cfg)
        mut[name] = {"controls_ok": r["controls_ok"], "broken": not r["controls_ok"], "controls_read_wrong": sorted(k for k, v in r["controls"].items() if not v["read_right"])}
        print("mutation", name, "controls_ok", r["controls_ok"], flush=True)
    out["mutations"] = mut
    out["all_mutations_break_controls"] = bool(all(v["broken"] for v in mut.values()))
    out["family_ablation"] = AD.family_ablation(block, lab, leaky)
    print("family ablation done", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("admit_seen: admissible", base["admissible"], "all mutations break", out["all_mutations_break_controls"], "red leaked", out["red_arm_disabled"]["leaked_through"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
