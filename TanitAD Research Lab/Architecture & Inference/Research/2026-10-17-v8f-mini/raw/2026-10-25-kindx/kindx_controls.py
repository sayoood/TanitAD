"""kindx_controls.py -- PREREG_V8F_KINDX controls. Written BEFORE any KINDX eval existed; UNTESTED at the time of writing (the box was out of commit headroom).

  python -I kindx_controls.py 1     phase 1, after the evals own, none, all, union and BEFORE any single-kind eval: controls 1-3 of the PREREG (+ structure)
  python -I kindx_controls.py 2     phase 2, after the six single-kind evals: the fallback-window identity (windows with no v9 row: every oracle arm == own exactly)
  python -I kindx_controls.py --seed 1 1|2   the SAME controls on X_s1 (PREREG_V8F_KINDX's seed-1 confirmation, chain_v8f_kindx_s1.sh). Paths come from KINDX_OUT / KINDX_EV (the pulled evals) /
                                    KINDX_BK (the "banked" standard-eval files: at s1 the chain's own `evalmixo 1` evals, in EV) / KINDX_CKPT; the expected checkpoint sha256 is the one the chain
                                    MEASURED after training (OUT/ckpt_sha256.json, key M2b-mixo_s1_trained), which must equal the one after the evals, the file's own, and differ from X_s0's.

Writes OUT/controls.json (phase 1 key `phase1`, phase 2 key `phase2`; at s1 OUT/controls_s1.json). Exit 0 = every check passed, exit 2 = a control failed: STOP and report, read no gain.
Tolerance 1e-4 m on per-window ADE6 (CPU float32 against box 2's GPU: a prior CPU re-run of M2b matched box 2 to 1.6e-5 m).  Exactness (control 3) is == 0.0."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

X0_CKPT_SHA = "c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d"        # X_s0 (MIXO's seed-0 checkpoint): the s1 checkpoint must differ from it


def _seed_arg(argv):
    for i, x in enumerate(argv):
        if x == "--seed" and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith("--seed="):
            return x.split("=", 1)[1]
    return None


SEED = int(os.environ["KINDX_SEED"] if "KINDX_SEED" in os.environ else ((_seed_arg(sys.argv[1:]) if __name__ == "__main__" else None) or "0"))
if SEED not in (0, 1):
    raise SystemExit(f"KINDX controls: --seed must be 0 or 1, got {SEED}")
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
OUT = os.environ.get("KINDX_OUT", f"{SC}/kindx")
EV = os.environ.get("KINDX_EV", f"{OUT}/eval")
BK = os.environ.get("KINDX_BK", "C:/Users/Admin/box2_ship/logs/v8f_mixo/eval" if SEED == 0 else EV)
CKPT = os.environ.get("KINDX_CKPT", f"C:/Users/Admin/box2_ship/logs/v8f_mixo/runs/M2b_s0_mixo/ckpt_last.pt" if SEED == 0 else f"{OUT}/runs/M2b_s{SEED}_mixo/ckpt_last.pt")
T = f"M2b-mixo_s{SEED}"
XDIR = f"M2b_s{SEED}_mixo"


def expected_sha():
    """s0: the committed value. s1: the value the chain measured after training (it must also be the one after the evals), never one typed in advance."""
    if SEED == 0:
        return X0_CKPT_SHA, None
    ck = js(f"{OUT}/ckpt_sha256.json")
    a, b = ck.get(f"{T}_trained"), ck.get(f"{T}_after_evals")
    return a, (None if a and a == b else f"ckpt_sha256.json: trained {a!r} vs after_evals {b!r}")
TOL = 1e-4
KINDS = ("speed", "angle", "tte", "timegap", "geom", "leadgap")
N_WIN, N_EP = 2399, 141


def ld(p):
    return np.load(p)


def js(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def dev(a, b):
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


def ckpt_is_x(j):
    parts = str(j.get("ckpt", "")).replace("\\", "/").rstrip("/").split("/")
    return len(parts) >= 2 and parts[-2] == XDIR and parts[-1] == "ckpt_last.pt"


def phase1():
    r = {"tolerance_m": TOL}
    exp, why = expected_sha()
    r["checkpoint_sha256"] = {"observed": sha256(CKPT), "expected": exp}
    r["checkpoint_sha256"]["PASS"] = bool(exp) and why is None and r["checkpoint_sha256"]["observed"] == exp
    if why:
        r["checkpoint_sha256"]["why"] = why
    if SEED != 0:                                                                                     # X_s1 is another training run: the same file as X_s0 would mean no second seed was read
        r["checkpoint_sha256"]["differs_from_X_s0"] = r["checkpoint_sha256"]["observed"] != X0_CKPT_SHA
        r["checkpoint_sha256"]["PASS"] = bool(r["checkpoint_sha256"]["PASS"] and r["checkpoint_sha256"]["differs_from_X_s0"])
    own, none, allo, uni = (ld(f"{EV}/eval_{T}-kindx-{n}.npz") for n in ("own", "none", "all", "union"))
    j = {n: js(f"{EV}/eval_{T}-kindx-{n}.json") for n in ("own", "none", "all", "union")}
    b_std, b_pred, b_con, b_orc = (ld(f"{BK}/eval_{T}{s}.npz") for s in ("", "-tacpred", "-tacorc-constraints", "-tacorc"))
    # structure: one window list, the committed size, X's checkpoint, the manipulation each arm declares
    eid = own["eid"]
    st = {"n_windows": int(eid.size), "n_episodes": int(np.unique(eid).size),
          "eid_equal_banked": all(np.array_equal(z["eid"], eid) for z in (none, allo, uni, b_std, b_pred, b_con, b_orc)),
          "ckpt_is_X": all(ckpt_is_x(j[n]) for n in j),
          "own_is_pred": j["own"].get("tac") == "pred" and not j["own"].get("tac_manipulation")}
    man = {n: (j[n].get("tac_manipulation") or {}) for n in ("none", "all", "union")}
    st["manipulations"] = {n: {k: man[n].get(k) for k in ("mode", "groups", "n_code_dims_replaced", "alpha")} for n in man}
    st["declared_ok"] = (man["none"].get("n_code_dims_replaced") == 0 and man["none"].get("mode") == "oracle" and man["all"].get("n_code_dims_replaced") == 60
                         and man["union"].get("n_code_dims_replaced") == 26 and all(float(man[n].get("alpha", 0)) == 1.0 for n in man))
    st["PASS"] = bool(st["n_windows"] == N_WIN and st["n_episodes"] == N_EP and st["eid_equal_banked"] and st["ckpt_is_X"] and st["own_is_pred"] and st["declared_ok"])
    r["structure"] = st
    # control 1: X's own code (--tac pred) reproduces the banked standard eval (and the banked -tacpred) per window
    r["control_1_own_vs_banked"] = {f"ade6_max_abs_dev_vs_eval_{T}_m": dev(own["ade6"], b_std["ade6"]), "ade6_max_abs_dev_vs_tacpred_m": dev(own["ade6"], b_pred["ade6"]),
                                    "pred_max_abs_dev_vs_banked_m": dev(own["pred"], b_std["pred"]), "tolerance_m": TOL}
    r["control_1_own_vs_banked"]["PASS"] = bool(r["control_1_own_vs_banked"][f"ade6_max_abs_dev_vs_eval_{T}_m"] <= TOL and r["control_1_own_vs_banked"]["ade6_max_abs_dev_vs_tacpred_m"] <= TOL)
    # control 2: the union oracle reproduces the banked -tacorc-constraints, the full GT code the banked -tacorc
    r["control_2_oracles_vs_banked"] = {"union_vs_tacorc_constraints_m": dev(uni["ade6"], b_con["ade6"]), "all_vs_tacorc_m": dev(allo["ade6"], b_orc["ade6"]), "tolerance_m": TOL}
    r["control_2_oracles_vs_banked"]["PASS"] = bool(r["control_2_oracles_vs_banked"]["union_vs_tacorc_constraints_m"] <= TOL and r["control_2_oracles_vs_banked"]["all_vs_tacorc_m"] <= TOL)
    # control 3: the identity oracle (GT in no kind) gains EXACTLY 0.0 per window against this session's own-code eval
    g = np.asarray(own["ade6"], np.float64) - np.asarray(none["ade6"], np.float64)
    r["control_3_identity_oracle"] = {"max_abs_gain_per_window_m": float(np.abs(g).max()), "n_windows_with_nonzero_gain": int((g != 0.0).sum()), "pred_array_equal": bool(np.array_equal(own["pred"], none["pred"])),
                                      "gain_exactly_zero_per_window": bool((g == 0.0).all())}
    r["control_3_identity_oracle"]["PASS"] = bool(r["control_3_identity_oracle"]["gain_exactly_zero_per_window"] and r["control_3_identity_oracle"]["pred_array_equal"])
    r["PASS"] = all(r[k]["PASS"] for k in ("checkpoint_sha256", "structure", "control_1_own_vs_banked", "control_2_oracles_vs_banked", "control_3_identity_oracle"))
    return r


def phase2():
    own = ld(f"{EV}/eval_{T}-kindx-own.npz")
    r = {"per_arm": {}}
    for nm in [f"only-{k}" for k in KINDS] + ["union", "all"]:
        z = ld(f"{EV}/eval_{T}-kindx-{nm}.npz")
        fb = np.asarray(z["tac_oracle_fallback"], bool)
        g = np.asarray(own["ade6"], np.float64) - np.asarray(z["ade6"], np.float64)
        r["per_arm"][nm] = {"eid_equal": bool(np.array_equal(z["eid"], own["eid"])), "n_fallback_windows": int(fb.sum()), "max_abs_gain_on_fallback_windows_m": float(np.abs(g[fb]).max()) if fb.any() else 0.0,
                            "fallback_mask_equals_all": None}
    fb_all = np.asarray(ld(f"{EV}/eval_{T}-kindx-all.npz")["tac_oracle_fallback"], bool)
    for nm in r["per_arm"]:
        z = ld(f"{EV}/eval_{T}-kindx-{nm}.npz")
        r["per_arm"][nm]["fallback_mask_equals_all"] = bool(np.array_equal(np.asarray(z["tac_oracle_fallback"], bool), fb_all))
    r["PASS"] = all(v["eid_equal"] and v["max_abs_gain_on_fallback_windows_m"] == 0.0 and v["fallback_mask_equals_all"] for v in r["per_arm"].values())
    return r


if __name__ == "__main__":
    ph = sys.argv[-1]
    path = f"{OUT}/controls.json" if SEED == 0 else f"{OUT}/controls_s{SEED}.json"
    C = js(path) if os.path.exists(path) else {}
    C[f"phase{ph}"] = phase1() if ph == "1" else phase2()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(C, f, indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else v.get("PASS")) for k, v in C[f"phase{ph}"].items() if k in ("PASS", "control_1_own_vs_banked", "control_2_oracles_vs_banked", "control_3_identity_oracle", "structure", "checkpoint_sha256")}, indent=1, default=str))
    sys.exit(0 if C[f"phase{ph}"]["PASS"] else 2)
