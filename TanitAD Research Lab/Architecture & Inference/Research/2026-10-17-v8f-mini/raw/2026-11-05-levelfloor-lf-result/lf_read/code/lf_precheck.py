#!/usr/bin/env python3
"""lf_precheck.py -- gate 1 of the LF read chain (A3 card section 3) and the per-rollout JSON check.

  python -I lf_precheck.py                          gate 1 (before any rollout): per seed the C0 checkpoint (the planner) and the LLR checkpoint (the floor heads) are PRESENT (absent = exit 5),
                                                    their sha256 equals the banked ones: C0 = the constants below = SHA256SUMS.txt = the banked rollouts' ckpt_sha256 (when its done marker is present);
                                                    LLR = raw/LLR_PRECHECK.json of the LLR read = SHA256SUMS.txt.  Writes raw/LF_PRECHECK.json.  exit 0 PASS / 5 absent / 8 failed.
  python -I lf_precheck.py --rollout TAG [--out-root D]   after one rollout (exit 0 / 8): the rollout's output JSON carries level_floor on, floor_kappa == the role's dose, floor_ckpt == the seed's LLR checkpoint and
                                                    floor_ckpt sha256 == the LLR sha256 of the seed; the done marker's --ckpt sha256 == the C0 sha256.

The tool's JSON layout for the floor keys is read tolerantly (``floor_info``: top level or one level down; ``level_floor`` as a string, bool or {mode, kappa, ckpt, sha256}); a key that cannot be found is a failed check, never a pass.
"""
import hashlib
import json
import os
import re
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
DL = f"{ART}/dagland/motion_c"
RDB = os.environ.get("LF_RDB", f"{ART}/v8f-recover")
LLR_PRECHECK = os.environ.get("LF_LLR_PRECHECK", f"{DL}/llr_read/raw/LLR_PRECHECK.json")
OUT_ROOT = "F:/TanitAD-scratch/motion_c/lf_read/out"
BANKED_C0_DONE = "F:/TanitAD-scratch/recover_read/out/done_C0_s{s}.txt"           # the banked C0 rollout's marker (ckpt_sha256); optional evidence
C0_SHA = {0: "bc3f926b7d5bc034ebb7043e5ef186a9dcad29ef404341fcd0493c7a22ca9fdf", 1: "43a303980f063a495f4d745e66efb870cf7f2f9c25acc0a7998562fb4978be93"}      # SHA256SUMS.txt of M2b_s{s}_recft_C0
KAPPA = {"050": 0.5, "075": 0.75}
TAG_RE = re.compile(r"^(RI)?LF(050|075)(C)?_s([01])$")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sums_line(run_dir):
    """The sha256 SHA256SUMS.txt lists for ckpt_last.pt, None when the file or the line is absent."""
    p = f"{run_dir}/SHA256SUMS.txt"
    if not os.path.isfile(p):
        return None
    for line in open(p, encoding="utf-8"):
        t = line.split()
        if len(t) == 2 and t[1].lstrip("*") == "ckpt_last.pt":
            return t[0]
    return None


def parse_tag(tag):
    m = TAG_RE.match(tag)
    if not m:
        raise ValueError(f"not an LF rollout tag: {tag!r}")
    return {"reindexed": bool(m.group(1)), "dose": m.group(2), "kappa": KAPPA[m.group(2)], "composite": bool(m.group(3)), "seed": int(m.group(4))}


def _find(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            if isinstance(v, dict) and key in v:
                return v[key]
    return None


def floor_info(doc):
    """The four floor keys of a rollout's output JSON (the file written by ``--out-dir/<tag>/t1v8f_<tag>.json``): {level_floor, floor_kappa, floor_ckpt, floor_ckpt_sha256}; None where a key is absent."""
    lf, kap, ck, sha = (_find(doc, k) for k in ("level_floor", "floor_kappa", "floor_ckpt", "floor_ckpt_sha256"))
    if isinstance(lf, dict):
        kap = lf.get("kappa", lf.get("floor_kappa", kap))
        ck = lf.get("ckpt", lf.get("floor_ckpt", ck))
        sha = lf.get("sha256", lf.get("ckpt_sha256", lf.get("floor_ckpt_sha256", sha)))
        lf = lf.get("mode", lf.get("level_floor"))
    if isinstance(ck, dict):
        sha = ck.get("sha256", sha)
        ck = ck.get("path", ck.get("ckpt"))
    return {"level_floor": lf, "floor_kappa": kap, "floor_ckpt": ck, "floor_ckpt_sha256": sha}


def llr_sha_of(seed, path=None):
    pc = json.load(open(path or LLR_PRECHECK, encoding="utf-8"))
    return pc["seeds"][f"s{seed}"]["ckpt_sha256"]


def check_rollout(out_root, tag, llr_pc=None):
    """Controls on one finished rollout: the output JSON's floor keys against the role's dose and the seed's LLR checkpoint, the done marker's planner checkpoint against C0's sha256.  -> {name: bool}, details."""
    t = parse_tag(tag)
    s = t["seed"]
    doc = json.load(open(f"{out_root}/{tag}/t1v8f_{tag}.json", encoding="utf-8"))
    fi = floor_info(doc)
    done = json.load(open(f"{out_root}/done_{tag}.txt", encoding="utf-8"))
    want_llr = llr_sha_of(s, llr_pc)
    lf = fi["level_floor"]
    ck = str(fi["floor_ckpt"] or "").replace("\\", "/")
    try:
        kap_ok = abs(float(fi["floor_kappa"]) - t["kappa"]) < 1e-9
    except (TypeError, ValueError):
        kap_ok = False
    c = {"level_floor_on": bool(lf is True or str(lf).lower() == "on"), "floor_kappa_is_the_dose": bool(kap_ok), "floor_ckpt_is_the_seeds_LLR": ck.endswith(f"/M2b_s{s}_recft_LLR/ckpt_last.pt"),
         "floor_ckpt_sha256_is_the_LLR_sha": fi["floor_ckpt_sha256"] == want_llr, "planner_ckpt_sha256_is_C0": done.get("ckpt_sha256") == C0_SHA[s]}
    return c, {"floor_info": fi, "llr_sha256_wanted": want_llr, "c0_sha256_wanted": C0_SHA[s], "done_ckpt_sha256": done.get("ckpt_sha256")}


def precheck(rdb=None, llr_pc=None):
    rdb = rdb or RDB
    res = {"rdb": rdb, "seeds": {}, "failures": [], "absent": [], "clock": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        pc = json.load(open(llr_pc or LLR_PRECHECK, encoding="utf-8"))
        assert pc["PASS"] is True
    except Exception as e:
        res["failures"].append(f"LLR_PRECHECK.json unreadable or not PASS: {type(e).__name__}: {e}")
        pc = None
    for s in (0, 1):
        r = {}
        for run, want in (("C0", C0_SHA[s]), ("LLR", None if pc is None else pc["seeds"][f"s{s}"]["ckpt_sha256"])):
            d = f"{rdb}/M2b_s{s}_recft_{run}"
            ck = f"{d}/ckpt_last.pt"
            if not os.path.isfile(ck):
                res["absent"].append(ck)
                r[run] = {"present": False}
                continue
            got, sl = sha256_file(ck), sums_line(d)
            e = {"present": True, "ckpt_sha256": got, "pinned": want, "sha256sums_line": sl}
            if want is None or got != want:
                res["failures"].append(f"s{s} {run}: ckpt sha256 {got} != banked {want}")
            if sl is None or sl != got:
                res["failures"].append(f"s{s} {run}: SHA256SUMS.txt line {sl} != file {got}")
            if run == "C0":
                bp = BANKED_C0_DONE.format(s=s)
                if os.path.isfile(bp):
                    bm = json.load(open(bp, encoding="utf-8")).get("ckpt_sha256")
                    e["banked_C0_rollout_marker_ckpt_sha256"] = bm
                    if bm != got:
                        res["failures"].append(f"s{s} C0: the banked C0 rollout's marker says {bm}")
            r[run] = e
        res["seeds"][f"s{s}"] = r
    res["PASS"] = bool(not res["failures"] and not res["absent"])
    return res


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if "--rollout" in a:
        tag = a[a.index("--rollout") + 1]
        oroot = a[a.index("--out-root") + 1] if "--out-root" in a else OUT_ROOT
        try:
            c, det = check_rollout(oroot, tag)
        except Exception as e:
            print(f"LF-ROLLOUT-CHECK {tag} FAILED to read: {type(e).__name__}: {e}")
            return 8
        bad = [k for k, v in c.items() if not v]
        print(f"LF-ROLLOUT-CHECK {tag}", "PASS" if not bad else f"FAILED {bad}", json.dumps(det["floor_info"]))
        return 0 if not bad else 8
    res = precheck()
    os.makedirs(f"{PK}/raw", exist_ok=True)
    json.dump(res, open(f"{PK}/raw/LF_PRECHECK.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print("LF PRECHECK", "PASS" if res["PASS"] else "FAILED", "| absent:", res["absent"], "| failures:", res["failures"])
    return 0 if res["PASS"] else (5 if res["absent"] and not res["failures"] else 8)


if __name__ == "__main__":
    sys.exit(main())
