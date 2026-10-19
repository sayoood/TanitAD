#!/usr/bin/env python3
"""assemble PREREG_SEARCH_P2_CONFIRM_DRAFT.md from src/PREREG_DRAFT_TEMPLATE.md + raw/design_calc_table.md + src/DQS.md + the PINS block.  Numbers in the power table come from raw/design_calc.json via power_table.py, never retyped.
   python build_prereg.py --out ../PREREG_SEARCH_P2_CONFIRM_DRAFT.md"""
import argparse, hashlib, json, re, os
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
STEPB_PREREG = "C:/Users/Admin/par18/l2stepb/PREREG_L2_STEPB.md"
NEW_CODE = ("spc_launch.py", "spc_chain.py", "spc_gpu_stage.py", "spc_cid.py", "spc_analyse.py", "b_rest.py", "make_cid_sample.py", "design_calc.py", "cost_spc.py", "power_table.py", "test_spc.py", "mutation_check.py", "seal_spc.py")
def sha(p): return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
def step_b_pins():
    t = open(STEPB_PREREG, encoding="utf-8").read()
    blk = t.split("<!-- PINS-BEGIN -->")[1].split("<!-- PINS-END -->")[0]
    return [l for l in blk.splitlines() if l.strip() and not l.startswith("`")]
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    s = open(os.path.join(PKG, "src/PREREG_DRAFT_TEMPLATE.md"), encoding="utf-8").read()
    s = s.replace("@@POWER_TABLE@@", open(os.path.join(PKG, "raw/design_calc_table.md"), encoding="utf-8").read().rstrip())
    s = s.replace("@@DQS@@", open(os.path.join(PKG, "src/DQS.md"), encoding="utf-8").read().rstrip())
    pins = ["# Step B pins, copied verbatim from PREREG_L2_STEPB.md (the arm is IDENTICAL; seal_spc.py re-checks each against the file / constant)"] + step_b_pins()
    pins += ["# this package",
             "b_rest_tokens_sha256 3a85cc8124b3dd02297fffa176787b629369a75d82ed52c4440b32aa4cc70c4b",
             "carve_b_tokens_sha256 0629811ff5d05879eb62abc5726005ccc4924f03539423003503a099ba29577c",
             "step_a_tokens_sha256 211e26406f5b4eb083ae38b1c73839826d560042cd75ccf9ff6656262e7c0ffd",
             "bridge_bundle_script_sha256_lf 8e544b8676c18c537076a52243df26a761ae299397fc25ec69653eedd40aa237"]
    for k, f in (("raw:b_rest.json", "raw/b_rest.json"), ("raw:cid_sample.json", "raw/cid_sample.json"), ("raw:design_calc.json", "raw/design_calc.json"), ("raw:cost.json", "raw/cost.json")):
        pins.append(f"{k} {sha(os.path.join(PKG, f))}")
    for f in NEW_CODE:
        p = os.path.join(PKG, "code", f)
        pins.append(f"code:{f} " + (sha(p) if os.path.exists(p) else "<NOT YET WRITTEN>"))
    s = s.replace("@@PINS@@", "\n".join(pins))
    assert "@@" not in s
    open(a.out, "w", encoding="utf-8", newline="\n").write(s)
    print("ZZPREREG_DRAFT_OK", a.out, len(s), hashlib.sha256(s.encode()).hexdigest())
main()
