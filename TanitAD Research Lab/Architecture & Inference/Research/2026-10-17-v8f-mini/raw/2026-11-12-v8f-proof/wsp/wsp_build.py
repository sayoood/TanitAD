"""wsp_build.py -- builds SOUP2, SOUPa0, SOUPa1, DOSE050, DOSE100 from the banked stopbal checkpoints (PREREG_V8F_PROOF.md H-V8F-PROOF-1 / -2).  CPU, no training.
  SOUP2   = 0.5 * (theta_NULL_s0 + theta_NULL_rep)                   every float tensor; integer buffers from member a (NULL_s0)
  SOUPa0  = (1-0)*a + 0*b      SOUPa1 = (1-1)*a + 1*b                 interpolation controls c1 (must be bit-identical to the members)
  DOSE050 = theta_NULL_s0 + 0.5 * (theta_G2_s0 - theta_NULL_s0)
  DOSE100 = theta_NULL_s0 + 1.0 * (theta_G2_s0 - theta_NULL_s0)      control: must equal BAL-G2_s0 bit for bit; if float rounding breaks it, built by exact copy (reported)
Controls run inline: c1, c2 (every tensor frozen in training is bit-identical in both members, else STOP), plus the list of tensors that differ between NULL_s0 and NULL_rep.
Writes <ckpt>/<TAG>/ckpt_last.pt (same dict layout as the member; `opt` emptied, `wsp` provenance added) + WSP_MANIFEST.json, and BUILD_REPORT.json beside this file.
Usage: python -I wsp_build.py      (refuses an existing <ckpt>/<TAG>; remove it by its literal path first)"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp_common as C  # noqa: E402
import torch  # noqa: E402


def bits_equal(x, y):
    return x.shape == y.shape and x.dtype == y.dtype and bool(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)))


def interp(a, b, alpha):
    """(1-alpha)*a + alpha*b on tensors that differ between the members; a tensor bit-identical in both is copied (the formula is the identity there, but 0*inf = NaN breaks it
    on the +-inf clip bounds of wm.stack.prior_lo_* / prior_hi_*: the first build failed c1 on exactly those six tensors)."""
    if not a.is_floating_point() or bits_equal(a, b):
        return a.clone()
    return (1.0 - alpha) * a + alpha * b


def main():
    rep = {"tool": "wsp_build.py", "commit_tree": C.COMMIT, "problems": [], "stop": False}
    ck = {t: C.load(d + "/ckpt_last.pt") for t, d in C.BANKS.items()}
    man = {t: json.load(open(d + "/BANK_MANIFEST.json", encoding="utf-8"))["files"]["ckpt_last.pt"]["sha256"] for t, d in C.BANKS.items()}
    rep["source_sha256"] = {t: C.sha(d + "/ckpt_last.pt") for t, d in C.BANKS.items()}
    rep["source_matches_manifest"] = {t: rep["source_sha256"][t] == man[t] for t in man}
    if not all(rep["source_matches_manifest"].values()):
        rep["problems"].append("a source checkpoint differs from its BANK_MANIFEST sha256")
    a, b, g, g2 = (ck[t]["model"] for t in ("NULL_s0", "NULL_rep", "G_s0", "G2_s0"))
    keys = list(a.keys())
    for t in ("NULL_rep", "G_s0", "G2_s0"):
        if list(ck[t]["model"].keys()) != keys:
            rep["problems"].append(f"{t}: state-dict keys differ from NULL_s0")
    c0 = ck["NULL_s0"]["cfg"]
    rep["cfg_differences_vs_NULL_s0"] = {t: {k: [str(c0.get(k)), str(ck[t]["cfg"].get(k))] for k in set(c0) | set(ck[t]["cfg"]) if c0.get(k) != ck[t]["cfg"].get(k)} for t in ("NULL_rep", "G_s0", "G2_s0")}
    rep["seed"] = {t: ck[t]["seed"] for t in ck}
    rep["init_ckpt_sha256"] = {t: ck[t]["init_ckpt"]["sha256"] for t in ck}
    if len(set(rep["init_ckpt_sha256"].values())) != 1:
        rep["problems"].append("init checkpoints differ: not same-init")
    m0 = C.build_model(ck["NULL_s0"])
    tn = set(C.trainable_names(m0))
    frozen = [k for k in keys if k not in tn]
    rep["n_tensors"], rep["n_trainable"], rep["n_not_trainable"] = len(keys), len(tn), len(frozen)
    rep["n_opt_state_entries_NULL_s0"] = len(ck["NULL_s0"]["opt"]["state"])
    rep["frozen_keys"] = frozen
    differ = [k for k in keys if not bits_equal(a[k], b[k])]
    rep["differ_NULL_s0_vs_NULL_rep"] = {"n": len(differ), "keys": differ}
    fr_bad = [k for k in frozen if not bits_equal(a[k], b[k])]
    rep["c2_frozen_bit_identical"] = {"n_frozen": len(frozen), "n_not_identical": len(fr_bad), "not_identical": fr_bad, "pass": not fr_bad}
    rep["c2_ckpt_level_arrays_equal"] = {x: bool((ck["NULL_s0"][x] == ck["NULL_rep"][x]).all()) for x in ("cons_mu", "cons_sd")}
    if fr_bad:
        rep["problems"].append("c2 FAILED: a frozen tensor differs between NULL_s0 and NULL_rep: not same-init. STOP.")
        rep["stop"] = True
    rep["trainable_tensors_bit_identical_between_members"] = [k for k in tn if bits_equal(a[k], b[k])]
    if rep["stop"]:
        json.dump(rep, open(C.WSP + "/BUILD_REPORT.json", "w"), indent=1)
        print("STOP (c2)", rep["problems"])
        raise SystemExit(2)
    sd = {"SOUP2": {k: (0.5 * (a[k] + b[k]) if (a[k].is_floating_point() and not bits_equal(a[k], b[k])) else a[k].clone()) for k in keys},
          "SOUPa0": {k: interp(a[k], b[k], 0.0) for k in keys}, "SOUPa1": {k: interp(a[k], b[k], 1.0) for k in keys},
          "DOSE050": {k: (a[k] + 0.5 * (g2[k] - a[k])) if (a[k].is_floating_point() and not bits_equal(a[k], g2[k])) else a[k].clone() for k in keys}}
    d100 = {k: (a[k] + 1.0 * (g2[k] - a[k])) if (a[k].is_floating_point() and not bits_equal(a[k], g2[k])) else a[k].clone() for k in keys}
    bad100 = [k for k in keys if not bits_equal(d100[k], g2[k])]
    mx = max((float((d100[k] - g2[k]).abs().max()) for k in bad100), default=0.0)
    rep["c_dose100"] = {"formula_bit_identical_to_BAL-G2_s0": not bad100, "n_tensors_not_identical": len(bad100), "max_abs_diff": mx,
                        "built_by": "exact copy of BAL-G2_s0 tensors" if bad100 else "formula (bit-identical)"}
    sd["DOSE100"] = {k: g2[k].clone() for k in keys} if bad100 else d100
    rep["c1"] = {"SOUPa0_bit_identical_to_NULL_s0": all(bits_equal(sd["SOUPa0"][k], a[k]) for k in keys),
                 "SOUPa1_bit_identical_to_NULL_rep": all(bits_equal(sd["SOUPa1"][k], b[k]) for k in keys),
                 "SOUPa0_not_identical": [k for k in keys if not bits_equal(sd["SOUPa0"][k], a[k])][:10],
                 "SOUPa1_not_identical": [k for k in keys if not bits_equal(sd["SOUPa1"][k], b[k])][:10]}
    rep["c1"]["pass"] = rep["c1"]["SOUPa0_bit_identical_to_NULL_s0"] and rep["c1"]["SOUPa1_bit_identical_to_NULL_rep"]
    if not rep["c1"]["pass"]:
        rep["problems"].append("c1 FAILED: alpha 0 / 1 not bit-identical to the members (see c1 lists)")
    rep["soup2_frozen_equal_members"] = all(bits_equal(sd["SOUP2"][k], a[k]) for k in frozen)
    rep["soup2_vs_interp_half_max_abs_diff"] = max(float((sd["SOUP2"][k] - interp(a[k], b[k], 0.5)).abs().max()) for k in keys if a[k].is_floating_point())
    prov = {"SOUP2": {"a": "NULL_s0", "b": "NULL_rep", "formula": "0.5*(a+b)"}, "SOUPa0": {"a": "NULL_s0", "b": "NULL_rep", "alpha": 0.0},
            "SOUPa1": {"a": "NULL_s0", "b": "NULL_rep", "alpha": 1.0}, "DOSE050": {"null": "NULL_s0", "lever": "G2_s0", "alpha": 0.5},
            "DOSE100": {"null": "NULL_s0", "lever": "G2_s0", "alpha": 1.0, "built_by": rep["c_dose100"]["built_by"]}}
    rep["written"] = {}
    for tag in C.BUILT:
        d = f"{C.CKPT}/{tag}"
        if os.path.exists(d):
            raise SystemExit(f"{d} exists: remove it by its literal path first")
        os.makedirs(d)
        out = dict(ck["NULL_s0"])
        out["model"] = type(ck["NULL_s0"]["model"])((k, sd[tag][k]) for k in keys)
        out["opt"] = {}
        out["arm"] = tag
        out["wsp"] = dict(prov[tag], source_sha256=rep["source_sha256"], built=time.strftime("%Y-%m-%d %H:%M:%S"))
        torch.save(out, d + "/ckpt_last.pt")
        h = C.sha(d + "/ckpt_last.pt")
        back = C.load(d + "/ckpt_last.pt")
        C.build_model(back)
        rt = all(bits_equal(back["model"][k], sd[tag][k]) for k in keys)
        json.dump({"tag": tag, "ckpt_sha256": h, "bytes": os.path.getsize(d + "/ckpt_last.pt"), "provenance": out["wsp"], "strict_load_ok": True, "roundtrip_bit_identical": rt},
                  open(d + "/WSP_MANIFEST.json", "w"), indent=1)
        rep["written"][tag] = {"sha256": h, "roundtrip_bit_identical": rt}
        print("WROTE", d, h[:12], "roundtrip", rt, flush=True)
    json.dump(rep, open(C.WSP + "/BUILD_REPORT.json", "w"), indent=1)
    print("DONE problems:", rep["problems"])


if __name__ == "__main__":
    main()
