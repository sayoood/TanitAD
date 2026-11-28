#!/usr/bin/env python3
"""n0_read.py -- the read of N0, PREREG_V8F_DAG_ORC section 8 (landed 85d0b125).  REUSES dag2t1/dag2_read.py's layer A (ReadA, rows_equal, pins) unchanged; the only new logic is the composition below.

N0: DAG (checkpoints M2b_s{0,1}_dag) at T1-kin with --nav logged, vs the SAME checkpoint with --nav deadreckon (DAG's banked decisive causal dumps).  Paired episode-cluster bootstrap, B = 2,000
(rr_lib.PB, as Read A), 141 episodes / 2,399 windows, per seed.  X = nav-logged, Y = nav-deadreckon, delta = X - Y (negative = nav-logged better on an error row).
NAVMOVE = every row of C_s separated-better at BOTH seeds, C_0 = {curvature MAE, L2 curvature MAE (GT>=5.2), M1 crawl share}, C_1 = C_0 + {yaw-rate MAE}.  Nothing here moves a bar.

Controls (any failure => REFUSED, no verdict): K-ARGV, K-IDENT, K-RC, K-WIN (the four PREREG-named controls) -- plus informational checks reported beside (K-REACH, K-FLOOR, K-HDR, K-PINS).
usage: n0_read.py --out <N0_READ.json> [--ident-only]
"""
import hashlib
import json
import os
import shlex
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ART = "E:/Projects/TanitAD-artifacts"
N0 = f"{ART}/dag-orc-n0"
sys.path.insert(0, f"{ART}/dag2t1")
import dag2_read as D2  # noqa: E402

SEEDS = (0, 1)
TT = f"{ART}/dagt1_tree_b43c7a9a"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
SF = f"{ART}/sfresh_inputs"
PREREG_BLOB_SHA = None  # filled from the extracted blob at run time
CRAWL_C = D2.CRAWL_SET   # {0: (curvature MAE, L2 .., M1 ..), 1: (+ yaw-rate MAE)}  == PREREG_V8F_DAG_ORC section 4 / section 8 (unchanged from DAGGER2)
ALLOWED_ARGV_DIFF = {"--nav", "--tag", "--out-dir", "--dump-ticks"}
CHAIN_LOG = f"{ART}/dagt1/logs/chain.out"


def sha256_file(p):
    return D2.sha256_file(p)


def n0_dump(s):
    return f"{N0}/dump_N0DAG_s{s}_m10_causal_heads_navlogged"


def dr_dump(s):
    return f"{ART}/v8f-dagger-t1/dump_DAG_s{s}_m10_causal_heads"


def banked_argv(s):
    pref = f"start DAG_s{s}_m10_causal_heads"
    ln = [l for l in open(CHAIN_LOG, encoding="utf-8").read().splitlines() if pref in l]
    assert len(ln) == 1
    return shlex.split(ln[0].split("): ", 1)[1], posix=True)


def argv_control(s):
    """K-ARGV: the N0 argv (the file the runner executed) == DAG's banked deadreckon argv except --nav (deadreckon -> logged) and the tag / output paths."""
    new = [t.rstrip("\r") for t in open(f"{N0}/argv/N0_s{s}.argv", encoding="utf-8").read().split("\n") if t != ""]
    old = banked_argv(s)
    if len(new) != len(old):
        return False, {"reason": f"token counts differ {len(new)} vs {len(old)}"}
    diffs = [(i, old[i - 1], old[i], new[i]) for i in range(len(old)) if old[i] != new[i]]
    bad = [d for d in diffs if d[1] not in ALLOWED_ARGV_DIFF]
    nav_ok = any(d[1] == "--nav" and d[2] == "deadreckon" and d[3] == "logged" for d in diffs)
    out_ok = all(n0 in new[old.index("--out-dir") + 1] or True for n0 in (N0,))
    return (not bad) and nav_ok, {"differing_flags": {d[1]: [d[2], d[3]] for d in diffs}, "disallowed": bad, "n_tokens": len(new)}


def header_control(s):
    """Informational: the run JSON headers agree on every setting except nav (and its consequences: strict.route, open_channels)."""
    hn = json.load(open(f"{N0}/t1v8f_N0DAG_s{s}_m10_causal_heads_navlogged.json", encoding="utf-8"))["t1v8f"]
    hb = json.load(open(f"{ART}/v8f-dagger-t1/t1v8f_DAG_s{s}_m10_causal_heads.json", encoding="utf-8"))["t1v8f"]
    keys = ("ckpt", "kin_mode", "tier", "replan_steps", "replan_hz", "schedule", "scene", "limit", "n_windows", "model_keys_read", "envelope_ms")
    same = {k: hn.get(k) == hb.get(k) for k in keys}
    same["strict.kinematics/camera/speed_bucket"] = all(hn["strict"][k] == hb["strict"][k] for k in ("kinematics", "camera", "speed_bucket"))
    differ = {"nav": [hb["nav"], hn["nav"]], "strict.route": [hb["strict"]["route"], hn["strict"]["route"]], "open_channels": [hb["open_channels"], hn["open_channels"]],
              "windows_with_envelope_clip": [hb["windows_with_envelope_clip"], hn["windows_with_envelope_clip"]]}
    ok = all(same.values()) and hn["nav"] == "logged" and hb["nav"] == "deadreckon"
    return ok, {"same": same, "differ_by_design_or_result": differ}


def zero_rows(tab):
    """-> list of rows that are not exactly 0.0 [0.0, 0.0] (NOT COMPUTED rows are skipped, and counted)."""
    bad, n = [], 0
    for lab, r in tab.items():
        if r.get("status") == "NOT COMPUTED":
            continue
        n += 1
        if not (r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 and not r["separated"]):
            bad.append((lab, r["delta"], r["lo"], r["hi"], r["separated"]))
    return bad, n


def slim(r):
    keep = ("delta", "lo", "hi", "separated", "separated_better", "separated_worse", "n", "mean_arm", "mean_ref", "family", "direction", "status")
    return {k: r[k] for k in keep if k in r}


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ident-only", action="store_true")
    ap.add_argument("--prereg-blob", default=f"{N0}/PREREG_V8F_DAG_ORC_85d0b125.md")
    A_ = ap.parse_args()
    t_start = time.time()
    argv = ["--tool-root", TT, "--t1hd-read", f"{TT}/{PK}/raw/2026-10-26-t1hdread/t1hd_read.py", "--t1k-read", f"{TT}/{PK}/raw/2026-10-26-t1kin-hacc/t1k_read.py", "--eval-cache", f"{SF}/cache",
            "--tracks", f"{SF}/inputs/native_tracks.pt", "--eval-grid", f"{TT}/{PK}/raw/eval_grid_windows.json", "--eval-join", f"{TT}/{PK}/raw/join_eval141.npz", "--out", "unused"]
    a = D2.build_parser().parse_args(argv)
    ctl = {}

    def add(name, ok, detail):
        ctl[name] = {"PASS": bool(ok), "detail": detail}
        print(("PASS " if ok else "FAIL ") + name + "  " + str(detail)[:220].replace("\n", " "), flush=True)

    bad = D2.pin_problems(D2.static_paths(a))
    add("K-PINS(reader sources, dag_read, banked READ_A / DAG_READ / T1KF, lead block)", not bad, bad or f"{len(D2.static_paths(a))} pinned inputs re-hashed")
    A = D2.ReadA(a)
    np = A.np
    T = A.T
    # ---- K-ARGV (control 1)
    for s in SEEDS:
        ok, det = argv_control(s)
        add(f"K-ARGV:s{s}", ok, det)
    for s in SEEDS:
        ok, det = header_control(s)
        add(f"K-HDR:s{s}", ok, det)
    banked_reada = json.load(open(a.reada_banked, encoding="utf-8"))["READ_A"]["rows"]
    banked_dagread = json.load(open(a.dag_read_json, encoding="utf-8"))
    comps = {k: {} for k in ("NB", "NP", "DP", "BB")}   # NB = N0 - DAG(deadreckon); NP = N0 - P10; DP = DAG(deadreckon) - P10; BB = DAG(dr) - DAG(dr) identity
    tacdk = {k: {} for k in comps}
    reach = {}
    for s in SEEDS:
        dN = A.dump(n0_dump(s))
        dB = A.dump(dr_dump(s))
        # ---- K-WIN (control 4)
        same = all(np.array_equal(dN[k], dB[k]) for k in ("clip_index", "ws", "eid", "g", "v0"))
        add(f"K-WIN:s{s}", bool(same and len(dN["eid"]) == 2399 and len(np.unique(dN["eid"])) == 141 and len(dB["eid"]) == 2399 and len(np.unique(dB["eid"])) == 141),
            {"n_windows_N0": int(len(dN["eid"])), "n_episodes_N0": int(len(np.unique(dN["eid"]))), "n_windows_banked": int(len(dB["eid"])), "n_episodes_banked": int(len(np.unique(dB["eid"]))),
             "note": "clip_index, ws, eid, g, v0 bit-equal between the new nav-logged dump and DAG's banked deadreckon dump"})
        add(f"K-FLOOR:s{s}(info)", bool(np.array_equal(dN["fl"], dB["fl"])), "P10's floor tracks (fl) bit-equal between the two runs (informational: not a PREREG-named control)")
        diff_w = np.any(dN["cl"] != dB["cl"], axis=(1, 2))
        reach[s] = {"windows_with_changed_cl_track": int(diff_w.sum()), "of": int(len(diff_w)), "max_abs_cl_diff_m": float(np.nanmax(np.abs(dN["cl"] - dB["cl"])))}
        add(f"K-REACH:s{s}(info)", bool(diff_w.sum() > 0), reach[s])
        cx = A.tacdk_context(dB)
        PN = T.prep("N0", dN["cl"], cx)
        PB = T.prep("DAG", dB["cl"], cx)
        PP = T.prep("P10", dB["fl"], cx)
        for k, (X, Y, PX, PY) in {"NB": (dN["cl"], dB["cl"], PN, PB), "NP": (dN["cl"], dB["fl"], PN, PP), "DP": (dB["cl"], dB["fl"], PB, PP), "BB": (dB["cl"], dB["cl"], PB, PB)}.items():
            comps[k][s], tacdk[k][s] = A.pair_rows(X, Y, dB, PX, PY)
        print(f"s{s}: pairs done {time.time() - t_start:.0f}s", flush=True)
    spos_ref = banked_dagread
    tabs = {"NB": A.table(comps["NB"], tacdk["NB"], banked_dagread, None), "NP": A.table(comps["NP"], tacdk["NP"], banked_dagread, None),
            "DP": A.table(comps["DP"], tacdk["DP"], banked_dagread, "DAG"), "BB": A.table(comps["BB"], tacdk["BB"], banked_dagread, None)}
    # ---- K-IDENT (control 2): the reader on (banked deadreckon vs itself) reads exactly 0.0 on every row
    for s in SEEDS:
        badz, nrows = zero_rows(tabs["BB"][s])
        # the comps/tac rows the table builder does not carry (M2 block, L3, ADE2 beside) are covered by RA.ident_block / T.rows identity too
        dB = A.dump(dr_dump(s))
        idc = A.RA.ident_block(dB["cl"], dB["g"], dB["eid"])
        cx = A.tacdk_context(dB)
        idt = T.rows(T.prep("DAG", dB["cl"], cx), T.prep("DAG", dB["cl"], cx), dB["eid"])
        ok = (not badz) and all(v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0 for v in idc.values()) and all(T.is_zero(v) for v in idt.values())
        add(f"K-IDENT:s{s}", ok, {"table_rows_exactly_zero": nrows - len(badz), "table_rows": nrows, "not_zero": badz[:5], "comps_ident_rows": len(idc), "tac_dk_ident_rows": len(idt),
                                  "STRATEGIC": "the 3 S-POS rows are NOT COMPUTED in a paired table (unpaired within-run vs CTRV), so they are outside the pair identity"})
    # ---- K-RC (control 3): layer A reproduces DAG's banked Read-A DAG - P10 table from the banked dumps with 0 differences (19 rows)
    for s in SEEDS:
        diffs = D2.rows_equal(tabs["DP"][s], banked_reada[f"s{s}"])
        mx = 0.0
        for lab, r in banked_reada[f"s{s}"].items():
            q = tabs["DP"][s].get(lab)
            if q and q.get("status") != "NOT COMPUTED" and r.get("status") != "NOT COMPUTED":
                mx = max(mx, abs(float(q["delta"]) - float(r["delta"])), abs(float(q["lo"]) - float(r["lo"])), abs(float(q["hi"]) - float(r["hi"])))
        add(f"K-RC:s{s}", not diffs, {"rows_compared": len(banked_reada[f"s{s}"]), "tolerance": 1e-4, "differences": diffs[:6], "max_abs_diff_over_delta_lo_hi": mx})
    # ---- C-SET: the crawl set C is the banked set of separated-worse rows of DAG - P10 (minus the OACC rows): informational but cheap
    for s in SEEDS:
        derived = sorted(l for l, r in banked_reada[f"s{s}"].items() if r.get("separated_worse") and l not in D2.EXEMPT_NEWLY)
        add(f"C-SET:s{s}(info)", derived == sorted(CRAWL_C[s]), {"derived_from_banked_READ_A": derived, "PREREG_C": sorted(CRAWL_C[s])})
    named = [k for k in ctl if not k.endswith("(info)") and not k.startswith("K-PINS") and not k.startswith("K-HDR") and not k.startswith("C-SET")]
    failed_named = [k for k in named if not ctl[k]["PASS"]]
    failed_other = [k for k in ctl if not ctl[k]["PASS"] and k not in failed_named]
    print("named controls:", named, "failed:", failed_named, "other failed:", failed_other, flush=True)
    refused = bool(failed_named)
    out = {"read": "N0 (PREREG_V8F_DAG_ORC section 8 @ 85d0b125)", "claim_row": "H-V8F-DAG-ORC-1 (precondition 3)",
           "tier": "T1-kin: self-action OPEN loop, plain (no post-map), m=10 periodic, causal speed bucket, logged scene; nav = LOGGED for N0 (an OPEN channel: route countdown from the logged human position), DEAD-RECKONED for the reference. Never closed loop.",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0, 141 episodes / 2,399 windows, per seed, paired; same checkpoint, deterministic planner => no training-run and no inference-run term; "
                        "it answers 'would another draw of EPISODES say this?' only.",
           "sign": "X = DAG nav-logged, Y = DAG nav-deadreckon (same checkpoint); delta = X - Y; negative = nav-logged lower error; separated_better = separated AND delta on the better side of the row's direction",
           "controls": ctl, "named_controls_failed": failed_named, "other_controls_failed": failed_other}
    if refused:
        out["OUTCOME"] = "REFUSED"
        out["note"] = "a PREREG-named control failed: no row is reported as a verdict"
    seeds_rows = {}
    for s in SEEDS:
        rows = tabs["NB"][s]
        c = CRAWL_C[s]
        dec = {}
        for lab in c:
            r = rows[lab]
            dec[lab] = {**slim(r), "sep_better_check": bool(D2.sep_better(r, "lower"))}
        seeds_rows[f"s{s}"] = {"C_s": list(c), "rows": dec, "all_C_separated_better": all(dec[l]["separated_better"] for l in c)}
    navmove = all(seeds_rows[f"s{s}"]["all_C_separated_better"] for s in SEEDS)
    out["decisive"] = seeds_rows
    out["yaw_rate_at_s0_beside"] = slim(tabs["NB"][0]["yaw-rate MAE"])
    if not refused:
        out["NAVMOVE"] = bool(navmove)
        out["OUTCOME"] = "NAVMOVE" if navmove else "NOT NAVMOVE"
    out["beside"] = {"read_a_N0_minus_DAGdr": {f"s{s}": {l: slim(r) for l, r in tabs["NB"][s].items()} for s in SEEDS},
                     "read_a_N0_minus_P10": {f"s{s}": {l: slim(r) for l, r in tabs["NP"][s].items()} for s in SEEDS},
                     "read_a_DAGdr_minus_P10_recomputed": {f"s{s}": {l: slim(r) for l, r in tabs["DP"][s].items()} for s in SEEDS},
                     "reach": reach,
                     "M2_and_pooled_M1": {f"s{s}": {"N0_minus_DAG": {k: v for k, v in tabs["NB"][s]["M1 crawl share"].items() if k in ("pooled_M1_arm", "pooled_M1_P10", "M2")}} for s in SEEDS}}
    out["provenance"] = {"reader": os.path.abspath(__file__), "reader_sha256": sha256_file(os.path.abspath(__file__)), "dag2_read_sha256": sha256_file(f"{ART}/dag2t1/dag2_read.py"),
                         "prereg_blob_sha256": sha256_file(A_.prereg_blob) if os.path.isfile(A_.prereg_blob) else None, "wall_s": time.time() - t_start,
                         "evidence_class": "MEASURED (ours; dumps under dag-orc-n0/ and v8f-dagger-t1/)"}
    os.makedirs(os.path.dirname(os.path.abspath(A_.out)), exist_ok=True)
    json.dump(out, open(A_.out, "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("OUTCOME", out["OUTCOME"], "wrote", A_.out, sha256_file(A_.out), flush=True)
    return 0 if not refused else 2


if __name__ == "__main__":
    sys.exit(main())
