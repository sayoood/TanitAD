"""EP-ADMISSION-R1 step 1 check: `--rules draft` of the edited ep_admission.py reproduces the BASE module (blob b66988b2, sha256 a338f303...) bit for bit.

  (1) the 35 base RED mutations are carried over verbatim (DECISION_MUTATIONS[:35] equal) and every base function / constant the draft uses has the same source text;
  (2) SYNTHETIC: the same banks through the base `run_admission` and the edited `run_admission(rules='draft')`: the whole result JSON is equal except the new key `rules` (and, with the mutation
      control on, the C-DECISION block, whose mutation list is longer by construction);
  (3) REAL S0 (`--real`): `selftest-real` of the base and of the edited module with `--rules draft`, then both against the banked `selftest_real.json` of the P6-READ-PREP package: equal except
      `code_sha256`, `ram_available_gb` and the new `rules` key.
CPU only.  Usage: python verify_draft_bit_for_bit.py --base <base ep_admission.py> --out <json> [--real --p2-bank .. --ranked-logs .. --arrays-npz .. --landed-result .. --banked-selftest ..]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_RAW = os.path.abspath(os.path.join(HERE, "..", ".."))
EDITED = os.path.join(REPO_RAW, "2026-10-19-p6-read-prep", "code")
BASE_SHA256 = "a338f303359ba7ffe1380d21a1da3158d275a88c503dadac7e7420a99ca6ae79"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def canon(o):
    return json.dumps(o, sort_keys=True, default=str)


def diff(a, b, path=""):
    """First differing paths of two JSON-like trees (exact equality)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: only in B")
            elif k not in b:
                out.append(f"{path}/{k}: only in A")
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += diff(x, y, f"{path}[{i}]")
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and a != a and b != b):
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def function_source(src, name):
    m = re.search(rf"^def {name}\(.*?(?=^\S)", src[src.index(f"def {name}("):], re.S | re.M)       # up to the next column-0 line (the body and a multi-line signature are indented)
    return m.group(0) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--p2-bank")
    ap.add_argument("--ranked-logs")
    ap.add_argument("--arrays-npz")
    ap.add_argument("--landed-result")
    ap.add_argument("--banked-selftest")
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    res = {"base_path": a.base, "base_sha256": sha(a.base), "edited_path": os.path.join(EDITED, "ep_admission.py"), "edited_sha256": sha(os.path.join(EDITED, "ep_admission.py"))}
    assert res["base_sha256"] == BASE_SHA256, f"the base copy is not the pristine blob: {res['base_sha256']}"
    base, mine = load("ep_admission_base", a.base), load("ep_admission_edit", os.path.join(EDITED, "ep_admission.py"))
    ok = True

    # (1) mutations and source text of every draft function
    m1 = mine.DECISION_MUTATIONS[:len(base.DECISION_MUTATIONS)] == base.DECISION_MUTATIONS
    bsrc, msrc = open(a.base, encoding="utf-8").read(), open(os.path.join(EDITED, "ep_admission.py"), encoding="utf-8").read()
    names = ["a1_verdict", "a2_head_ok", "a2_verdict", "a1_form", "b_recovers", "null_fires_a1", "null_void", "admission_label", "next_arm", "load_bank_units", "align_candidate", "analyze_candidate",
             "pessimistic_picks", "control_reach", "a1_analytic_oc", "check_prereg", "read_ranked", "read_manifest", "make_stats"]
    same_src = {n: (function_source(bsrc, n) == function_source(msrc, n)) for n in names}
    consts = ["A1_BAR", "A2_LO", "A2_HI", "A2_HEADS", "B_UNSAFE_UB", "PLANT", "NULL_C_REPS", "NULL_E_REPS", "NULL_VOID_FIRES", "N_LOGS", "BOOT_B", "BOOT_SEED", "SELECT_PCT", "N_FOLDS", "RANKED_LOGS_SHA256",
              "EPRELIAB_LIB_SHA256", "LABEL_ORDER", "RULE", "SCHEMA"]
    same_const = {c: (getattr(base, c) == getattr(mine, c)) for c in consts}
    res["step1_source"] = {"base_mutations_carried_verbatim": m1, "n_base_mutations": len(base.DECISION_MUTATIONS), "n_edited_mutations": len(mine.DECISION_MUTATIONS), "unchanged_function_source": same_src,
                           "unchanged_constants": same_const}
    ok &= m1 and all(same_src.values()) and all(same_const.values())
    # the six functions that DID change are named, not hidden
    changed = ["control_id", "control_power", "run_null", "run_admission", "cmd_run", "cmd_selftest_real", "main"]
    res["step1_source"]["functions_edited_on_purpose"] = {n: (function_source(bsrc, n) != function_source(msrc, n)) for n in changed}

    # (2) synthetic banks, base vs edited --rules draft
    sys.path.insert(0, EDITED)
    import test_ep_admission as T
    rows = None
    syn = {}
    tmp = tempfile.mkdtemp(dir=a.workdir)
    try:
        rk = T.write_ranked(os.path.join(tmp, "ranked.tsv"))
        rows, _ = mine.read_ranked(rk)
        cases = {"calibrated": T.make_case(over=0.0), "overpredict_0.30": T.make_case(over=0.30), "over_0.04": T.make_case(over=0.04), "nc_unsafe_0.12": T.make_case(nc_unsafe=0.12),
                 "ttc_zero": T.make_case(ttc_unsafe=0.0), "few_changed": T.make_case(frac=0.05, over=0.02)}
        for cname, cs in cases.items():
            z6_p2, z6_c, lab, li, ch = cs
            pb = T.write_bank(os.path.join(tmp, cname, "p2"), z6_p2, lab, li, ckpt="p2")
            cb = T.write_bank(os.path.join(tmp, cname, "c"), z6_c, lab, li, ckpt="c", shuffle_seed=3)
            outs = {}
            for tag, mod, kw in (("base", base, {}), ("edited_draft", mine, {"rules": "draft"})):
                p2 = mod.load_bank_units(pb, rows)
                c = mod.align_candidate(p2, mod.load_bank_units(cb, rows))
                outs[tag] = json.loads(json.dumps(mod.jsonable(mod.run_admission(p2, {"c": c}, B=500, null_reps_e=20, skip_controls_mutation=True, log=lambda *_: None, **kw))))
            outs["edited_draft"].pop("rules")
            d = diff(outs["base"], outs["edited_draft"])
            syn[cname] = {"equal": not d, "label": outs["base"]["verdict"]["label"], "n_diff": len(d), "first_diffs": d[:5]}
            ok &= not d
        # with the mutation control ON the only difference allowed is C-DECISION's own list
        z6_p2, z6_c, lab, li, ch = cases["calibrated"]
        outs = {}
        for tag, mod, kw in (("base", base, {}), ("edited_draft", mine, {"rules": "draft"})):
            p2 = mod.load_bank_units(os.path.join(tmp, "calibrated", "p2"), rows)
            c = mod.align_candidate(p2, mod.load_bank_units(os.path.join(tmp, "calibrated", "c"), rows))
            outs[tag] = json.loads(json.dumps(mod.jsonable(mod.run_admission(p2, {"c": c}, B=500, null_reps_e=0, log=lambda *_: None, **kw))))
        outs["edited_draft"].pop("rules")
        cb_, ce_ = outs["base"]["controls"].pop("C-DECISION"), outs["edited_draft"]["controls"].pop("C-DECISION")
        d = diff(outs["base"], outs["edited_draft"])
        syn["with_mutation_control"] = {"equal_except_C-DECISION": not d, "n_diff": len(d), "first_diffs": d[:5], "C-DECISION_base": {"n_mutations": cb_["n_mutations"], "n_caught": cb_["n_caught"], "pass": cb_["pass"]},
                                        "C-DECISION_edited": {"n_mutations": ce_["n_mutations"], "n_caught": ce_["n_caught"], "pass": ce_["pass"]}}
        ok &= (not d) and cb_["pass"] and ce_["pass"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res["step2_synthetic"] = syn

    # (3) real S0
    if a.real:
        real = {}
        outs = {}
        wd = a.workdir or tempfile.mkdtemp()
        for tag, mod, extra in (("base", base, []), ("edited_draft", mine, ["--rules", "draft"])):
            o = os.path.join(wd, f"selftest_{tag}.json")
            rc = mod.main(["selftest-real", "--p2-bank", a.p2_bank, "--ranked-logs", a.ranked_logs, "--arrays-npz", a.arrays_npz, "--landed-result", a.landed_result, "--out", o, *extra])
            outs[tag] = json.load(open(o, encoding="utf-8"))
            real[f"rc_{tag}"] = rc
        banked = json.load(open(a.banked_selftest, encoding="utf-8")) if a.banked_selftest else None
        for t in outs.values():
            t.pop("code_sha256", None)
            t.pop("ram_available_gb", None)
        outs["edited_draft"].pop("rules", None)
        real["base_vs_edited_draft_n_diff"] = len(diff(outs["base"], outs["edited_draft"]))
        real["base_vs_edited_draft_first_diffs"] = diff(outs["base"], outs["edited_draft"])[:5]
        if banked is not None:
            banked.pop("code_sha256", None)
            banked.pop("ram_available_gb", None)
            real["base_rerun_vs_banked_n_diff"] = len(diff(banked, outs["base"]))
            real["base_rerun_vs_banked_first_diffs"] = diff(banked, outs["base"])[:5]
            real["edited_draft_vs_banked_n_diff"] = len(diff(banked, outs["edited_draft"]))
        real["pass_both"] = bool(real["rc_base"] == 0 and real["rc_edited_draft"] == 0)
        ok &= real["base_vs_edited_draft_n_diff"] == 0 and real["pass_both"]
        res["step3_real_s0"] = real
    res["pass"] = bool(ok)
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print("VERIFY_DRAFT_BIT_FOR_BIT pass=%s -> %s" % (res["pass"], a.out))
    return 0 if res["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
