#!/usr/bin/env python3
"""lf5_precheck.py -- gate 1 of the LF5 chain (PREREG_LEVELFLOOR_A5 sha256 2aac1dbe...) and the per-rollout JSON check. Reuses lfo_precheck.py (LOADED BY SHA-PINNED PATH), which reuses lf_precheck.py: the checkpoints are the LF read's.
  python -I lf5_precheck.py                          gate 1: per seed the C0 and LLR checkpoints are PRESENT and equal the banked sha256. Writes raw/LF5_PRECHECK.json. exit 0 / 5 / 8.
  python -I lf5_precheck.py --rollout TAG [--out-root D]   one finished rollout: the output JSON's floor keys (kappa 1.0, the seed's LLR sha, the source, the noise level and generator seed, the CEILING label), the planner sha256. exit 0 / 8.
Tags: RILF100[C]_s{0,1} and LF100[C]_s{0,1} (heads source, no noise keys, no CEILING label) | RILFO100[C]_s{s} (logged, sigma 0.0, no seed) | RILFN100x10[C]_s{s}, RILFN100x20[C]_s{s} (logged, sigma 0.10 / 0.20, seed 20261106, CEILING)."""
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
OUT_ROOT = "F:/TanitAD-scratch/motion_c/lf5_read/out"
KAPPA = 1.0
NOISE_SEED = 20261106
SIG_FLAG = {0.10: "0.10", 0.20: "0.20"}
TAG_RE = re.compile(r"^(RILFN100x10|RILFN100x20|RILFO100|RILF100|LF100)(C)?_s([01])$")
KIND = {"RILF100": ("heads", None), "LF100": ("heads", None), "RILFO100": ("logged", 0.0), "RILFN100x10": ("logged", 0.10), "RILFN100x20": ("logged", 0.20)}


def _load():
    got = hashlib.sha256(open(LFO_PRECHECK_PY, "rb").read()).hexdigest()
    if got != LFO_PRECHECK_SHA:
        raise SystemExit(f"REFUSED: {LFO_PRECHECK_PY} is {got}, not the pinned {LFO_PRECHECK_SHA}")
    spec = importlib.util.spec_from_file_location("lfo_precheck_for_lf5", LFO_PRECHECK_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LP = _load()
C0_SHA, floor_info, llr_sha_of, precheck = LP.C0_SHA, LP.floor_info, LP.llr_sha_of, LP.precheck


def parse_tag(tag):
    m = TAG_RE.match(tag)
    if not m:
        raise ValueError(f"not an LF5 rollout tag: {tag!r}")
    src, sigma = KIND[m.group(1)]
    return {"base": m.group(1), "composite": bool(m.group(2)), "seed": int(m.group(3)), "source": src, "sigma": sigma, "kappa": KAPPA}


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
    c = {"level_floor_on": bool(lf is True or str(lf).lower() == "on"), "floor_kappa_is_1_0": bool(kap_ok), "floor_ckpt_is_the_seeds_LLR": ck.endswith(f"/M2b_s{s}_recft_LLR/ckpt_last.pt"),
         "floor_ckpt_sha256_is_the_LLR_sha": fi["floor_ckpt_sha256"] == want_llr, "planner_ckpt_sha256_is_C0": done.get("ckpt_sha256") == C0_SHA[s],
         "header_floor_vhat_source_matches": hdr.get("floor_vhat_source") == t["source"], "done_flags_name_the_source": f"--floor-vhat-source {t['source']}" in str(done.get("flags", "")),
         "header_label_matches": (hdr.get("label") == "CEILING") if logged else ("label" not in hdr),
         "detail_vision_only_matches": str((hdr.get("level_floor_detail") or {}).get("vision_only", "")).startswith("NO.") if logged else str((hdr.get("level_floor_detail") or {}).get("vision_only", "")).startswith("v_hat, presence")}
    if logged:
        sg = t["sigma"]
        c["header_noise_level_matches"] = hdr.get("floor_vhat_noise") == sg
        c["header_noise_seed_matches"] = (hdr.get("floor_vhat_noise_seed") == NOISE_SEED) if sg > 0 else (hdr.get("floor_vhat_noise_seed") is None)
        c["done_flags_name_the_noise"] = (f"--floor-vhat-noise {SIG_FLAG[sg]} --floor-vhat-noise-seed {NOISE_SEED}" in str(done.get("flags", ""))) if sg > 0 else ("--floor-vhat-noise" not in str(done.get("flags", "")))
        if sg > 0:
            c["noise_draws_equal_tick_rows"] = int((hdr.get("level_floor_detail") or {}).get("counts", {}).get("noise_draws", -1)) == int(hdr.get("level_floor_detail", {}).get("counts", {}).get("replans", -2))
    else:
        c["no_noise_keys_on_a_heads_arm"] = "floor_vhat_noise" not in hdr
    return c, {"floor_info": fi, "llr_sha256_wanted": want_llr, "c0_sha256_wanted": C0_SHA[s], "done_ckpt_sha256": done.get("ckpt_sha256"), "header_floor_vhat_source": hdr.get("floor_vhat_source"),
               "header_label": hdr.get("label"), "header_noise": [hdr.get("floor_vhat_noise"), hdr.get("floor_vhat_noise_seed")]}


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if "--rollout" in a:
        tag = a[a.index("--rollout") + 1]
        oroot = a[a.index("--out-root") + 1] if "--out-root" in a else OUT_ROOT
        try:
            c, det = check_rollout(oroot, tag)
        except Exception as e:
            print(f"LF5-ROLLOUT-CHECK {tag} FAILED to read: {type(e).__name__}: {e}")
            return 8
        bad = [k for k, v in c.items() if not v]
        print(f"LF5-ROLLOUT-CHECK {tag}", "PASS" if not bad else f"FAILED {bad}", json.dumps(det["floor_info"]))
        return 0 if not bad else 8
    res = precheck()
    os.makedirs(f"{PK}/raw", exist_ok=True)
    json.dump(res, open(f"{PK}/raw/LF5_PRECHECK.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print("LF5 PRECHECK", "PASS" if res["PASS"] else "FAILED", "| absent:", res["absent"], "| failures:", res["failures"])
    return 0 if res["PASS"] else (5 if res["absent"] and not res["failures"] else 8)


if __name__ == "__main__":
    sys.exit(main())
