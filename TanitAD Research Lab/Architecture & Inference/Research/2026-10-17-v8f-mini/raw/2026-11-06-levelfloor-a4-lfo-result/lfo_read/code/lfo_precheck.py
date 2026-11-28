#!/usr/bin/env python3
"""lfo_precheck.py -- gate 1 of the LFO chain (PREREG_LEVELFLOOR_A4 section 2) and the per-rollout JSON check. Reuses lf_precheck.py (LOADED BY SHA-PINNED PATH): the checkpoints are the LF read's (C0 = the planner, LLR = the floor heads).

  python -I lfo_precheck.py                          gate 1 (before any rollout): per seed the C0 and LLR checkpoints are PRESENT (absent = exit 5) and equal the banked sha256 (lf_precheck.precheck). Writes raw/LFO_PRECHECK.json. exit 0 / 5 / 8.
  python -I lfo_precheck.py --rollout TAG [--out-root D]   after one rollout (exit 0 / 8): the output JSON carries level_floor on, floor_kappa 0.75, floor_ckpt = the seed's LLR checkpoint, floor_ckpt sha256 = the LLR sha256 of
                                                    the seed, floor_vhat_source "logged" and label "CEILING"; the done marker's --ckpt sha256 = the C0 sha256 and its flags name `--floor-vhat-source logged` explicitly.
"""
import importlib.util
import json
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
MC = "E:/Projects/TanitAD-artifacts/dagland/motion_c"
LF_PRECHECK_PY = f"{MC}/lf_read/code/lf_precheck.py"
LF_PRECHECK_SHA = "be99a03a869b9f3835bd247c63646258e78050e478efa3fd5fbd4c4a9a4cabf0"
OUT_ROOT = "F:/TanitAD-scratch/motion_c/lfo_read/out"
KAPPA = 0.75
TAG_RE = re.compile(r"^RILFO075(C)?_s([01])$")


def _load_lf():
    import hashlib
    got = hashlib.sha256(open(LF_PRECHECK_PY, "rb").read()).hexdigest()
    if got != LF_PRECHECK_SHA:
        raise SystemExit(f"REFUSED: {LF_PRECHECK_PY} is {got}, not the pinned {LF_PRECHECK_SHA}")
    spec = importlib.util.spec_from_file_location("lf_precheck_for_lfo", LF_PRECHECK_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LP = _load_lf()
C0_SHA, floor_info, llr_sha_of, precheck = LP.C0_SHA, LP.floor_info, LP.llr_sha_of, LP.precheck


def parse_tag(tag):
    m = TAG_RE.match(tag)
    if not m:
        raise ValueError(f"not an LFO rollout tag: {tag!r}")
    return {"composite": bool(m.group(1)), "seed": int(m.group(2)), "kappa": KAPPA}


def check_rollout(out_root, tag, llr_pc=None):
    """Controls on one finished LFO rollout: the output JSON's floor keys against the dose and the seed's LLR checkpoint, the CEILING labelling, the done marker's planner checkpoint against C0's sha256. -> {name: bool}, details."""
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
    c = {"level_floor_on": bool(lf is True or str(lf).lower() == "on"), "floor_kappa_is_0_75": bool(kap_ok), "floor_ckpt_is_the_seeds_LLR": ck.endswith(f"/M2b_s{s}_recft_LLR/ckpt_last.pt"),
         "floor_ckpt_sha256_is_the_LLR_sha": fi["floor_ckpt_sha256"] == want_llr, "planner_ckpt_sha256_is_C0": done.get("ckpt_sha256") == C0_SHA[s],
         "header_floor_vhat_source_is_logged": hdr.get("floor_vhat_source") == "logged", "header_label_is_CEILING": hdr.get("label") == "CEILING",
         "detail_says_not_vision_only": str((hdr.get("level_floor_detail") or {}).get("vision_only", "")).startswith("NO."),
         "done_flags_name_the_logged_source": "--floor-vhat-source logged" in str(done.get("flags", ""))}
    return c, {"floor_info": fi, "llr_sha256_wanted": want_llr, "c0_sha256_wanted": C0_SHA[s], "done_ckpt_sha256": done.get("ckpt_sha256"), "header_floor_vhat_source": hdr.get("floor_vhat_source"), "header_label": hdr.get("label")}


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if "--rollout" in a:
        tag = a[a.index("--rollout") + 1]
        oroot = a[a.index("--out-root") + 1] if "--out-root" in a else OUT_ROOT
        try:
            c, det = check_rollout(oroot, tag)
        except Exception as e:
            print(f"LFO-ROLLOUT-CHECK {tag} FAILED to read: {type(e).__name__}: {e}")
            return 8
        bad = [k for k, v in c.items() if not v]
        print(f"LFO-ROLLOUT-CHECK {tag}", "PASS" if not bad else f"FAILED {bad}", json.dumps(det["floor_info"]))
        return 0 if not bad else 8
    res = precheck()
    os.makedirs(f"{PK}/raw", exist_ok=True)
    json.dump(res, open(f"{PK}/raw/LFO_PRECHECK.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print("LFO PRECHECK", "PASS" if res["PASS"] else "FAILED", "| absent:", res["absent"], "| failures:", res["failures"])
    return 0 if res["PASS"] else (5 if res["absent"] and not res["failures"] else 8)


if __name__ == "__main__":
    sys.exit(main())
