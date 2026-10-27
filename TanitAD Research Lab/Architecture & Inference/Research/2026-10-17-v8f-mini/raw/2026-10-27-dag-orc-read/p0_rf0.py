#!/usr/bin/env python3
"""p0_rf0.py -- P-0 of PREREG_V8F_DAG_ORC (launch precondition 1): the banked replicate floor RF0, computed from BANKED runs only, then frozen by sha256.

    RF0_r = max over seeds s of | d_r,s(DAG-0 - ORC-0) |         for every Read-A row r        (PREREG_V8F_DAG_ORC section 3)

DAG-0 and ORC-0 are the banked M2b-init pair: same argv except --out and the default-off --dagger-p 0, same seed, builds b43c7a9a / 853f2fc3.  d_r,s(X - Y) is
Read A's paired point of X - Y on row r (PREREG_V8F_DAG_REREAD section 2's reader, UNCHANGED: ref/dag2_read.py -> ReadA, over ref/readA_code/).  For the three S-POS
rows X - Y is the difference of the two runs' CTRV contrasts on the same windows: here the difference of the two runs' banked (S - CTRV, ticks>=1) points, read from
the sha-pinned DAG_READ.json (per_run_t1heads.DAG0 / .ORC0).  No DAG-ORC data is read or exists.

Controls (every one must PASS or nothing is written):
  PINS            every reader / banked input re-hashed against dag2_read.PINS (15 pins) + this package's staged copies
  PINS-TXT        the tool-tree files named in each T1 dir's PINS.txt re-hashed
  WINDOWS         all four dumps carry the same 2,399 windows / 141 episodes, bit-equal (clip_index, ws, eid, g, v0)
  FLOOR           P10's tracks (fl) are bit-equal across the four dumps
  IDENTITY        (DAG-0, DAG-0) and (ORC-0, ORC-0) read exactly +0 [0, 0] on every comps row and every TAC / DK row
  RC-DAG          DAG - P10 re-computed with this code equals the banked READ_A.json rows (0 differences at 1e-4)  [DAG's banked Read-A table]
  RC-DAG0         DAG-0 - P10 re-computed equals the banked beside_DAG0_minus_P10 rows (0 differences at 1e-4)
  SPOS-BANKED     the banked per-run S-POS records of DAG-0 and ORC-0 exist at both seeds (3 arcs)

    python p0_rf0.py --out P0_RF0.json            writes P0_RF0.json, P0_RF0.sha256 next to it
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, f"{HERE}/ref")
import dag2_read as D2  # noqa: E402  (the staged, sha-pinned copy of dag2t1/dag2_read.py: Read A's layer, unchanged)

ART = "E:/Projects/TanitAD-artifacts"
TREE = f"{ART}/dagt1_tree_b43c7a9a"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
JOIN = f"{TREE}/{PK}/raw/join_eval141.npz"
SEEDS = (0, 1)
F_ADE6 = 0.1207
SPOS_KEY = "S-POS vs CTRV (ticks>=1, within this run)"
SPOS_ROWS = ("S-POS@30", "S-POS@50", "S-POS@80")
DUMP_DAG0 = lambda s: f"{ART}/v8f-dagger-t1/dump_DAG0_s{s}_m10_causal_heads"       # noqa: E731
DUMP_ORC0 = lambda s: f"{ART}/v8f-orecover-t1/dump_ORC0_s{s}_m10_causal_heads"     # noqa: E731
DUMP_DAG = lambda s: f"{ART}/v8f-dagger-t1/dump_DAG_s{s}_m10_causal_heads"         # noqa: E731
PINS_TXT = {"DAG0": f"{ART}/v8f-dagger-t1/PINS.txt", "ORC0": f"{ART}/v8f-orecover-t1/PINS.txt"}
SCHEMA = "p0-rf0-v1"


def sha256_file(p):
    return D2.sha256_file(p)


def make_args():
    """The namespace dag2_read.ReadA / static_paths read (the staged, pinned copies; the banked inputs in place)."""
    return types.SimpleNamespace(
        readers=f"{ART}/dagreread/reader_src", reada_code=f"{HERE}/ref/readA_code", tool_root=TREE, lead_block=f"{HERE}/ref/b1_eval_lead_block_k30.npz", eval_join=JOIN,
        dag_read=f"{HERE}/ref/dag_read.py", dag_read_json=f"{ART}/v8f-dagger-t1/read/DAG_READ.json", reada_banked=f"{ART}/dagreread/raw/READ_A.json",
        t1kf_banked=f"{ART}/dagreread/raw/t1kf_DAG/T1KF_READ_A1_DAG.json")


def pins_txt_problems(pins_path):
    """PINS.txt = `tree=<dir>` + `file=<relpath> <sha256>` lines; re-hash each file under the tree.  -> (problems, n_checked)."""
    lines = [ln.strip() for ln in open(pins_path, encoding="utf-8") if ln.strip()]
    tree = next(ln.split("=", 1)[1] for ln in lines if ln.startswith("tree="))
    bad, n = [], 0
    for ln in lines:
        if ln.startswith("file="):
            rel, sha = ln[5:].rsplit(" ", 1)
            p = f"{tree}/{rel}"
            n += 1
            if not os.path.isfile(p):
                bad.append(f"{p} is missing")
            elif sha256_file(p) != sha:
                bad.append(f"{p}: sha256 {sha256_file(p)[:16]}..., PINS.txt {sha[:16]}...")
    return bad, n


def dir_digest(d):
    """sha256 over (name, sha256) of every ep*.npz in a dump dir, in sorted order; -> (digest, n_files)."""
    import glob
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    h = hashlib.sha256()
    for f in fs:
        h.update(os.path.basename(f).encode() + b"\0" + sha256_file(f).encode() + b"\n")
    return h.hexdigest(), len(fs)


def read_a_table(A, d_ref, dX, dY, pair_names, s_spos=None):
    """Read A's rows (all but S-POS) for X - Y on the windows of d_ref.  -> {label: row} (Read A's labels), via ReadA.pair_rows + ReadA.table (unchanged)."""
    T = A.T
    cx = A.tacdk_context(d_ref)
    PX, PY = T.prep(pair_names[0], dX["cl"], cx), T.prep(pair_names[1], dY["cl"], cx)
    blk, tk = A.pair_rows(dX["cl"], dY["cl"], d_ref, PX, PY)
    return blk, tk


def table_of(A, blk_by_seed, tk_by_seed):
    return A.table(blk_by_seed, tk_by_seed, {}, None)


def identity_ok(A, d):
    T, RA = A.T, A.RA
    cx = A.tacdk_context(d)
    P = T.prep("X", d["cl"], cx)
    idc = RA.ident_block(d["cl"], d["g"], d["eid"])
    idt = T.rows(P, P, d["eid"])
    return bool(all(v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0 for v in idc.values()) and all(T.is_zero(v) for v in idt.values()))


def compute(verbose=True):
    t0 = time.time()
    a = make_args()
    ctl = {}

    def add(k, ok, det=None):
        ctl[k] = {"PASS": bool(ok), "detail": det}
        if verbose:
            print(("PASS " if ok else "FAIL ") + k, str(det)[:200].replace("\n", " "), flush=True)

    # ---------------------------------------------------------------- pins
    probs = D2.pin_problems(D2.static_paths(a))
    add("PINS", not probs, probs or f"{len(D2.static_paths(a))} pinned inputs re-hashed (dag2_read.PINS)")
    staged = {"ref/dag2_read.py": "1425e8a3895c442b9841f3bcd6fd0893021d583e06485fde171b04bfa73fbc20"}
    bad = [k for k, v in staged.items() if sha256_file(f"{HERE}/{k}") != v]
    add("PINS-STAGED", not bad, bad or "the staged copy of dag2_read.py is the sha-pinned dag2t1/dag2_read.py")
    inputs = {}
    for k, p in PINS_TXT.items():
        pb, n = pins_txt_problems(p)
        add(f"PINS-TXT:{k}", not pb and n > 0, pb or f"{n} tool files re-hashed against {p}")
        inputs[f"PINS.txt:{k}"] = {"path": p, "sha256": sha256_file(p), "n_files": n}
    if any(not v["PASS"] for v in ctl.values()):
        return None, ctl
    # ---------------------------------------------------------------- Read A over the staged code
    A = D2.ReadA(a)
    np = A.np
    dagread = json.load(open(a.dag_read_json, encoding="utf-8"))
    banked = json.load(open(a.reada_banked, encoding="utf-8"))
    comps, tacdk, rows = {"D0O0": {}, "DGP": {}, "D0P": {}}, {"D0O0": {}, "DGP": {}, "D0P": {}}, {}
    idents = {}
    for s in SEEDS:
        dd0, do0, dg = A.dump(DUMP_DAG0(s)), A.dump(DUMP_ORC0(s)), A.dump(DUMP_DAG(s))
        for k, d in (("DAG0", dd0), ("ORC0", do0), ("DAG", dg)):
            inputs[f"dump:{k}:s{s}"] = dict(zip(("sha256_of_file_hashes", "n_files"), dir_digest(DUMP_DAG0(s) if k == "DAG0" else DUMP_ORC0(s) if k == "ORC0" else DUMP_DAG(s))), path=(DUMP_DAG0(s) if k == "DAG0" else DUMP_ORC0(s) if k == "ORC0" else DUMP_DAG(s)))
        same_w = {nm: bool(all(np.array_equal(d[k], dd0[k]) for k in ("clip_index", "ws", "eid", "g", "v0")) and len(d["eid"]) == 2399 and len(np.unique(d["eid"])) == 141)
                  for nm, d in (("ORC0", do0), ("DAG", dg))}
        add(f"WINDOWS:s{s}", all(same_w.values()), {"windows_bit_equal_to_DAG0": same_w, "n_windows": int(len(dd0["eid"])), "n_episodes": int(len(np.unique(dd0["eid"])))})
        add(f"FLOOR:s{s}", bool(np.array_equal(do0["fl"], dd0["fl"]) and np.array_equal(dg["fl"], dd0["fl"])), "P10's tracks (fl) are bit-equal in DAG-0, ORC-0 and DAG dumps")
        idents[s] = (identity_ok(A, dd0), identity_ok(A, do0))
        add(f"IDENTITY:s{s}", all(idents[s]), {"DAG0-DAG0 exactly 0": idents[s][0], "ORC0-ORC0 exactly 0": idents[s][1]})
        cx = A.tacdk_context(dd0)
        P = {"DAG0": A.T.prep("DAG0", dd0["cl"], cx), "ORC0": A.T.prep("ORC0", do0["cl"], cx), "DAG": A.T.prep("DAG", dg["cl"], cx), "P10": A.T.prep("P10", dd0["fl"], cx)}
        for k, (X, Y, PX, PY) in {"D0O0": (dd0["cl"], do0["cl"], P["DAG0"], P["ORC0"]), "DGP": (dg["cl"], dg["fl"], P["DAG"], P["P10"]),
                                  "D0P": (dd0["cl"], dd0["fl"], P["DAG0"], P["P10"])}.items():
            comps[k][s], tacdk[k][s] = A.pair_rows(X, Y, dd0, PX, PY)
    tabs = {"D0O0": A.table(comps["D0O0"], tacdk["D0O0"], {}, None), "DGP": A.table(comps["DGP"], tacdk["DGP"], dagread, "DAG"), "D0P": A.table(comps["D0P"], tacdk["D0P"], dagread, "DAG0")}
    # RC-DAG: DAG's banked Read-A table, row for row
    for key, ref, nm in (("DGP", banked["READ_A"]["rows"], "RC-DAG"), ("D0P", banked["beside_DAG0_minus_P10"]["rows"], "RC-DAG0")):
        diffs = {f"s{s}": D2.rows_equal(tabs[key][s], ref[f"s{s}"]) for s in SEEDS}
        mx = 0.0
        for s in SEEDS:
            for lab, r in tabs[key][s].items():
                b = ref[f"s{s}"].get(lab)
                if b and r.get("status") != "NOT COMPUTED" and b.get("status") != "NOT COMPUTED":
                    mx = max(mx, abs(float(r["delta"]) - float(b["delta"])), abs(float(r["lo"]) - float(b["lo"])), abs(float(r["hi"]) - float(b["hi"])))
        add(nm, not any(diffs.values()), {"differences": {k: v[:4] for k, v in diffs.items()}, "rows_compared_per_seed": len(ref["s0"]), "max_abs_diff_delta_lo_hi": mx, "tolerance": 1e-4})
    # S-POS: the difference of the two runs' banked CTRV contrasts (points)
    spos = {}
    for s in SEEDS:
        per = dagread["beside"]["per_seed"][f"s{s}"]["per_run_t1heads"]
        for arc in SPOS_ROWS:
            try:
                x, y = per["DAG0"][SPOS_KEY][arc], per["ORC0"][SPOS_KEY][arc]
                spos[(s, arc)] = {"delta": float(x["delta"]) - float(y["delta"]), "DAG0_contrast": float(x["delta"]), "ORC0_contrast": float(y["delta"]),
                                  "DAG0_separated_vs_CTRV": bool(x["separated"]), "ORC0_separated_vs_CTRV": bool(y["separated"])}
            except KeyError as e:
                spos[(s, arc)] = None
    add("SPOS-BANKED", all(v is not None for v in spos.values()), {"records": len(spos), "source": a.dag_read_json})
    if any(not v["PASS"] for v in ctl.values()):
        return None, ctl
    # ---------------------------------------------------------------- RF0
    out_rows = {}
    labels = list(tabs["D0O0"][0].keys())
    for lab in labels:
        per_seed, ds = {}, []
        for s in SEEDS:
            r = tabs["D0O0"][s][lab]
            if lab in SPOS_ROWS:
                sp = spos[(s, lab)]
                per_seed[f"s{s}"] = {"delta": sp["delta"], "kind": "difference of the two runs' banked CTRV contrasts (points)", **{k: v for k, v in sp.items() if k != "delta"}}
                ds.append(abs(sp["delta"]))
            else:
                per_seed[f"s{s}"] = {"delta": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]), "n": r["n"]}
                ds.append(abs(float(r["delta"])))
        rf0 = max(ds)
        fam = tabs["D0O0"][0][lab].get("family") or ("STRATEGIC" if lab in SPOS_ROWS else None)
        out_rows[lab] = {"family": fam, "direction": A.DIRECTIONS.get(lab, "lower"), "per_seed": per_seed, "RF0": rf0, "RF0_seed_that_binds": f"s{int(ds[1] > ds[0])}",
                         "F": F_ADE6 if lab == "ADE6" else 0.0}
    res = {"schema": SCHEMA, "claim_row": "H-V8F-DAG-ORC-1", "prereg": "Project Steering/PREREG_V8F_DAG_ORC.md @ 3b7b404e", "section": "3 (RF0_r), launch precondition 1",
           "definition": "RF0_r = max over s in {0,1} of |d_r,s(DAG-0 - ORC-0)|, Read A's paired point of DAG-0 - ORC-0 on row r (S-POS: the difference of the two runs' banked CTRV contrasts)",
           "pair": {"X": "DAG-0 (v8f-dagger-t1/dump_DAG0_s{s}_m10_causal_heads)", "Y": "ORC-0 (v8f-orecover-t1/dump_ORC0_s{s}_m10_causal_heads)",
                    "note": "banked M2b-init pair: same argv except --out and the default-off --dagger-p 0; same seed; builds b43c7a9a / 853f2fc3 (PREREG section 3)"},
           "tier": "T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan; plain). Never 'closed loop'.",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0, 141 episodes / 2,399 windows, per seed, paired. RF0 uses the POINT only; lo / hi / separated are carried for the record.",
           "evidence_class": "MEASURED (ours; banked dumps named in `inputs`)", "n_rows": len(out_rows), "rows": out_rows,
           "RF0": {lab: r["RF0"] for lab, r in out_rows.items()}, "controls": ctl, "inputs": inputs,
           "code": {"p0_rf0.py": sha256_file(os.path.abspath(__file__)), "dag2_read.py (ReadA, unchanged)": sha256_file(f"{HERE}/ref/dag2_read.py")}}
    return res, ctl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{HERE}/P0_RF0.json")
    ap.add_argument("--check-only", action="store_true", help="compute and print, write nothing")
    ns = ap.parse_args()
    res, ctl = compute()
    failed = [k for k, v in ctl.items() if not v["PASS"]]
    if res is None or failed:
        print("P-0 REFUSED (nothing written): failed controls:", failed, file=sys.stderr)
        return 2
    if ns.check_only:
        print(json.dumps(res["RF0"], indent=1))
        return 0
    with open(ns.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    sha = sha256_file(ns.out)
    with open(os.path.splitext(ns.out)[0] + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{sha}  {os.path.basename(ns.out)}\n")
    print(json.dumps(res["RF0"], indent=1))
    print(f"wrote {ns.out}  sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
