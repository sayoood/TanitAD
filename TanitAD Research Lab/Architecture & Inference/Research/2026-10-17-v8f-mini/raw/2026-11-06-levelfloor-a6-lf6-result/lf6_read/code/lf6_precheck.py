#!/usr/bin/env python3
"""lf6_precheck.py -- gate 1 of the LF6 chain (PREREG_LEVELFLOOR_A6 sha256 fbf5de02...) and the per-rollout JSON check. Reuses lfo_precheck.py (LOADED BY SHA-PINNED PATH), which reuses lf_precheck.py: the checkpoints are the LF read's.
  python -I lf6_precheck.py                          gate 1: per seed the C0 and LLR checkpoints are PRESENT and equal the banked sha256. Writes raw/LF6_PRECHECK.json. exit 0 / 5 / 8.
  python -I lf6_precheck.py --rollout TAG [--out-root D]   one finished rollout: the output JSON's floor keys (kappa 1.0, the seed's LLR sha, the source, the lift cap DELTA, no noise keys on a heads arm, sigma 0 and no seed on a logged arm, the CEILING label), the planner sha256. exit 0 / 8.
Tags: RILFB05[C]_s{s}, RILFB10[C]_s{s} (heads, reindexed, DELTA 0.5 / 1.0) | LFB05[C]_s{s}, LFB10[C]_s{s} (heads, `logged`) | RILFOB05[C]_s{s}, RILFOB10[C]_s{s} (logged source, sigma 0, no seed, CEILING, reindexed)."""
import hashlib
import importlib.util
import json
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
MC = "E:/Projects/TanitAD-artifacts/dagland/motion_c"
LFO_PRECHECK_PY = f"{MC}/lfo_read/code/lfo_precheck.py"
LFO_PRECHECK_SHA = "4e2d10692341883e2b0f861b0c8d979d9d24f9214ee2edb1a685f7fb814fef27"
OUT_ROOT = "F:/TanitAD-scratch/motion_c/lf6_read/out"
KAPPA = 1.0
TAG_RE = re.compile(r"^(RILFOB05|RILFOB10|RILFB05|RILFB10|LFB05|LFB10)(C)?_s([01])$")
KIND = {"RILFB05": ("heads", 0.5), "RILFB10": ("heads", 1.0), "LFB05": ("heads", 0.5), "LFB10": ("heads", 1.0), "RILFOB05": ("logged", 0.5), "RILFOB10": ("logged", 1.0)}
DELTA_FLAG = {0.5: "0.5", 1.0: "1.0"}


def _load():
    got = hashlib.sha256(open(LFO_PRECHECK_PY, "rb").read()).hexdigest()
    if got != LFO_PRECHECK_SHA:
        raise SystemExit(f"REFUSED: {LFO_PRECHECK_PY} is {got}, not the pinned {LFO_PRECHECK_SHA}")
    spec = importlib.util.spec_from_file_location("lfo_precheck_for_lf6", LFO_PRECHECK_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LP = _load()
C0_SHA, floor_info, llr_sha_of, precheck = LP.C0_SHA, LP.floor_info, LP.llr_sha_of, LP.precheck


def parse_tag(tag):
    m = TAG_RE.match(tag)
    if not m:
        raise ValueError(f"not an LF6 rollout tag: {tag!r}")
    src, delta = KIND[m.group(1)]
    return {"base": m.group(1), "composite": bool(m.group(2)), "seed": int(m.group(3)), "source": src, "delta": delta, "kappa": KAPPA}


def check_rollout(out_root, tag, llr_pc=None):
    t = parse_tag(tag)
    s = t["seed"]
    doc = json.load(open(f"{out_root}/{tag}/t1v8f_{tag}.json", encoding="utf-8"))
    hdr = doc.get("t1v8f", doc)
    fi = floor_info(doc)
    done = json.load(open(f"{out_root}/done_{tag}.txt", encoding="utf-8"))
    want_llr = llr_sha_of(s, llr_pc)
    lf = fi["level_floor"]
    ck = str(fi["floor_ckpt"] or "").replace("\\", "/")
    try:
        kap_ok = abs(float(fi["floor_kappa"]) - KAPPA) < 1e-9
    except (TypeError, ValueError):
        kap_ok = False
    logged = t["source"] == "logged"
    flags = str(done.get("flags", ""))
    c = {"level_floor_on": bool(lf is True or str(lf).lower() == "on"), "floor_kappa_is_1_0": bool(kap_ok), "floor_ckpt_is_the_seeds_LLR": ck.endswith(f"/M2b_s{s}_recft_LLR/ckpt_last.pt"),
         "floor_ckpt_sha256_is_the_LLR_sha": fi["floor_ckpt_sha256"] == want_llr, "planner_ckpt_sha256_is_C0": done.get("ckpt_sha256") == C0_SHA[s],
         "header_floor_vhat_source_matches": hdr.get("floor_vhat_source") == t["source"], "done_flags_name_the_source": f"--floor-vhat-source {t['source']}" in flags,
         "header_floor_lift_cap_is_delta": hdr.get("floor_lift_cap") == t["delta"], "detail_floor_lift_cap_is_delta": (hdr.get("level_floor_detail") or {}).get("floor_lift_cap") == t["delta"],
         "done_flags_name_the_lift_cap": flags.endswith(f"--floor-lift-cap {DELTA_FLAG[t['delta']]}"),
         "header_label_matches": (hdr.get("label") == "CEILING") if logged else ("label" not in hdr),
         "detail_vision_only_matches": str((hdr.get("level_floor_detail") or {}).get("vision_only", "")).startswith("NO.") if logged else str((hdr.get("level_floor_detail") or {}).get("vision_only", "")).startswith("v_hat, presence"),
         "detail_names_the_a6_card": "PREREG_LEVELFLOOR_A6" in str((hdr.get("level_floor_detail") or {}).get("prereg", ""))}
    if logged:
        c["header_noise_level_is_zero"] = hdr.get("floor_vhat_noise") == 0.0
        c["header_noise_seed_is_none"] = hdr.get("floor_vhat_noise_seed") is None
        c["done_flags_name_no_noise"] = "--floor-vhat-noise" not in flags
    else:
        c["no_noise_keys_on_a_heads_arm"] = "floor_vhat_noise" not in hdr
    return c, {"floor_info": fi, "llr_sha256_wanted": want_llr, "c0_sha256_wanted": C0_SHA[s], "done_ckpt_sha256": done.get("ckpt_sha256"), "header_floor_vhat_source": hdr.get("floor_vhat_source"),
               "header_label": hdr.get("label"), "header_noise": [hdr.get("floor_vhat_noise"), hdr.get("floor_vhat_noise_seed")], "header_floor_lift_cap": hdr.get("floor_lift_cap")}


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if "--rollout" in a:
        tag = a[a.index("--rollout") + 1]
        oroot = a[a.index("--out-root") + 1] if "--out-root" in a else OUT_ROOT
        try:
            c, det = check_rollout(oroot, tag)
        except Exception as e:
            print(f"LF6-ROLLOUT-CHECK {tag} FAILED to read: {type(e).__name__}: {e}")
            return 8
        bad = [k for k, v in c.items() if not v]
        print(f"LF6-ROLLOUT-CHECK {tag}", "PASS" if not bad else f"FAILED {bad}", json.dumps(det["floor_info"]))
        return 0 if not bad else 8
    res = precheck()
    os.makedirs(f"{PK}/raw", exist_ok=True)
    json.dump(res, open(f"{PK}/raw/LF6_PRECHECK.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print("LF6 PRECHECK", "PASS" if res["PASS"] else "FAILED", "| absent:", res["absent"], "| failures:", res["failures"])
    return 0 if res["PASS"] else (5 if res["absent"] and not res["failures"] else 8)


if __name__ == "__main__":
    sys.exit(main())
