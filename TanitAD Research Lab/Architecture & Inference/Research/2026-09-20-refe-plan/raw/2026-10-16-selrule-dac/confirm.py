#!/usr/bin/env python3
"""confirm.py -- the ONE navtest confirm of PREREG_SELRULE_DAC.md, using ONLY the rule frozen in FIT.json.

    python confirm.py                 # FULL mode: refuses unless <table-dir>/P2FULL_DONE exists (P2's 12,146-token 64-proposal table)
    python confirm.py --smoke         # PLUMBING ONLY on the 509-token pilot table: shapes, fields, gates, seam schema.  Computes and prints NO PDMS delta
                                      # and writes none (a 509-token subset of P2's zero tokens is not the registered confirm set).

Order (full mode): marker -> FIT.json sha256 -> DISJOINTNESS (RL-val logs  INTERSECT  navtest logs must be empty, else STOP) -> table fields -> gates
(token set, valid, finite, B == table pick on >= 99 %, mean PDMS at the table pick == P2's banked 88.82 +- 0.01, PDMS column consistent with the sub-scores)
-> selrule.confirm_compute (paired log-cluster bootstrap, B 10,000, seed 20260927, over the logs) -> family leg (families6.py on B's and the chosen
rule's executed-pick seams; proxy_eval.fam_adverse) -> the committed outcome table (selrule.outcome) -> CONFIRM.json / CONFIRM.md.
Run it ONCE.  It refuses to overwrite an existing CONFIRM.json.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selrule as SR  # noqa: E402

PK = SR.PK
RAW = os.path.join(PK, "raw")
RV, RR, PE = SR.RV, SR.RR, SR.PE
FULL_DIR = os.path.join(RAW, "2026-10-16-p2-full-table")
PILOT_DIR = os.path.join(RAW, "2026-10-15-p2-zero-tail", "p2_own_table")
TOKENS_JSON = os.path.join(RAW, "p2_r2_instrument", "baseline", "navtest_full_tokens.json")
TOKENS_MD5 = "3a8b3653fd70891e50a230a37eb76ae8"                                  # stage3_full.FULL_PINS
N_TOKENS, N_LOGS, M = 12146, 136, 64
EXPORT_CANDIDATES = ("D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz", "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz")
SUB_NAMES = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
REQUIRED = {"token": (None,), "pdms": (None, M), "sub": (None, M, 6), "valid": (None, M), "logits": (None, M, 6), "proposals": (None, M, 8, 3), "pick": (None,), "sub_names": (6,)}
B_EQ_TABLE_PICK_MIN = 0.99
STRATEGIC = ("PAUSED by the PI (2026-09-19, PI_DECISION_QUEUE item 25) and UNAVAILABLE in NAVSIM (no strategic decision is scored); reported as such, never as a pass")


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Stop(Exception):
    """A gate failed: the confirm STOPS, nothing is read."""


def load_fit(path):
    side = path + ".sha256"
    if not (os.path.exists(path) and os.path.exists(side)):
        raise Stop(f"FIT.json or its .sha256 sidecar is missing ({path})")
    want = open(side, encoding="utf-8").read().split()[0]
    got = SR.sha256_file(path)
    if got != want:
        raise Stop(f"FIT.json sha256 {got} != sidecar {want}: the frozen choice was altered")
    fit = json.load(open(path, encoding="utf-8"))
    if not fit.get("frozen") or fit["chosen"]["spec"][0] not in ("V", "K"):
        raise Stop("FIT.json is not a frozen fit with a V / K choice")
    return fit, got


def load_token_map(path=TOKENS_JSON):
    if md5_file(path) != TOKENS_MD5:
        raise Stop(f"{path}: md5 is not the pinned navtest token list")
    d = json.load(open(path, encoding="utf-8"))
    toks = sorted(d["tokens"])
    tl = d["token_log"]
    if len(toks) != N_TOKENS or len(set(tl[t] for t in toks)) != N_LOGS:
        raise Stop(f"token list is not {N_TOKENS} tokens / {N_LOGS} logs")
    return toks, tl


def disjointness(fit, token_log):
    nav = set(token_log.values())
    rl = set(fit["rl_val"]["logs"])
    inter = sorted(nav & rl)
    return {"rl_val_logs": len(rl), "navtest_logs": len(nav), "intersection": len(inter), "intersection_logs": inter[:10]}


def load_table(table_dir, table_name="table.npz", proposals_name="proposals.npz"):
    tp = os.path.join(table_dir, table_name)
    if not os.path.exists(tp):
        raise Stop(f"{tp} does not exist")
    z = np.load(tp, allow_pickle=False)
    miss = [k for k in REQUIRED if k not in z.files]
    if miss:
        raise Stop(f"table fields missing: {miss}; present: {sorted(z.files)}")
    T = {k: z[k] for k in REQUIRED}
    n = len(T["token"])
    for k, shp in REQUIRED.items():
        want = tuple(n if s is None else s for s in shp)
        if T[k].shape != want:
            raise Stop(f"table field {k}: shape {T[k].shape} != {want}")
    names = tuple(str(x) for x in T["sub_names"])
    if names != SUB_NAMES:
        raise Stop(f"sub_names {names} != the registered column order {SUB_NAMES}")
    T["token"] = np.asarray([str(t) for t in T["token"]])
    pp = os.path.join(table_dir, proposals_name)
    T["_proposals_npz"] = None
    if os.path.exists(pp):
        zp = np.load(pp, allow_pickle=False)
        T["_proposals_npz"] = {k: zp[k] for k in zp.files if k in ("token", "fingerprint", "sampling")}
        T["_proposals_npz"]["token"] = np.asarray([str(t) for t in T["_proposals_npz"]["token"]])
    T["_sha256"] = {"table.npz": SR.sha256_file(tp), "proposals.npz": SR.sha256_file(pp) if os.path.exists(pp) else None}
    return T


def load_commands(tokens, paths=EXPORT_CANDIDATES):
    for p in paths:
        if os.path.exists(p):
            exp = json.load(gzip.open(p, "rt", encoding="utf-8"))["tokens"]
            miss = [t for t in tokens if t not in exp]
            if miss:
                return None, f"{len(miss)} tokens missing from the export {p}", None
            cmd = np.array([int(np.argmax(exp[t]["ego_statuses"][-1]["driving_command"])) for t in tokens])
            return cmd, "ok", {"path": p, "sha256": SR.sha256_file(p)}
    return None, "the navtest input export was not found at " + " or ".join(EXPORT_CANDIDATES), None


def check_table_basics(T, expected_tokens, token_log, full):
    g = {}
    tok = list(T["token"])
    g["n_tokens"] = len(tok)
    g["tokens_unique"] = len(set(tok)) == len(tok)
    if full:
        g["token_set_equals_expected_12146"] = set(tok) == set(expected_tokens)
    else:
        g["tokens_subset_of_expected_12146"] = set(tok) <= set(expected_tokens)
    g["every_token_has_a_log"] = all(t in token_log for t in tok)
    g["n_logs_in_table"] = len({token_log[t] for t in tok if t in token_log})
    g["valid_all_true"] = bool(T["valid"].all())
    g["pdms_finite"] = bool(np.isfinite(T["pdms"]).all())
    g["sub_finite"] = bool(np.isfinite(T["sub"]).all())
    g["logits_finite"] = bool(np.isfinite(T["logits"]).all())
    g["pdms_in_0_1"] = bool(((T["pdms"] >= 0) & (T["pdms"] <= 1)).all())
    g["sub_in_0_1"] = bool(((T["sub"] >= 0) & (T["sub"] <= 1)).all())
    g["pick_in_range"] = bool(((T["pick"] >= 0) & (T["pick"] < M)).all())
    g["proposals_finite"] = bool(np.isfinite(T["proposals"]).all())
    return g


def b_equals_table_pick(T):
    P = RV.sigmoid(T["logits"])
    B = SR.picks(("B",), P)
    return int((B == T["pick"]).sum()), int(len(B)), P, B


def build_seam(path, tokens, fingerprint, poses, sampling, arm):
    np.savez(path, token=np.asarray(tokens), fingerprint=np.asarray(fingerprint), poses=np.asarray(poses, np.float32), sampling=np.asarray(sampling, np.float64), arm=np.asarray(arm))


def aligned_fingerprint(T):
    p = T["_proposals_npz"]
    if p is None or "fingerprint" not in p:
        return None, None
    pos = {t: i for i, t in enumerate(p["token"])}
    if not all(t in pos for t in T["token"]):
        return None, None
    return np.asarray([p["fingerprint"][pos[t]] for t in T["token"]]), np.asarray(p["sampling"])


def families_leg(T, B, C, out_dir, export_path):
    """families6.py on the executed-pick seams of B and of the chosen rule -> (list of adverse 'family.metric' names or None, status dict)."""
    st = {"computed": False}
    try:
        sys.path.insert(0, os.path.join(PK, "eval"))
        import stage3_full as S3
    except Exception as e:  # noqa: BLE001
        st["reason"] = f"stage3_full import failed: {type(e).__name__}: {e}"
        return None, st
    f6 = S3.resolve_families6()
    fp, sampling = aligned_fingerprint(T)
    if f6 is None or not os.path.exists(S3.PY_T) or export_path is None or fp is None:
        st["reason"] = {"families6.py": str(f6), "tanitad_python": os.path.exists(S3.PY_T), "export": export_path, "fingerprint_available": fp is not None}
        return None, st
    S3.EXPORT = export_path
    od = Path(out_dir) / "families"
    od.mkdir(parents=True, exist_ok=True)
    ar = np.arange(len(B))
    fams = {}
    for lab, pk in (("selrule_B", B), ("selrule_chosen", C)):
        sp = od / f"{lab}_seam_in.npz"
        build_seam(sp, T["token"], fp, T["proposals"][ar, pk], sampling, lab)
        fams[lab] = S3.compute_families(sp, lab, od, set(T["token"].tolist()))
    if not all(fams.values()):
        st["reason"] = "families6 produced no JSON for: " + ", ".join(k for k, v in fams.items() if not v)
        return None, st
    r = S3.families_vs(fams["selrule_B"], fams["selrule_chosen"])
    adv = [f"{a['family']}.{a['metric']}" for a in r.get("adverse", [])]
    st.update({"computed": True, "instrument": str(f6), "components_compared": r.get("components_compared"), "adverse": r.get("adverse"), "tactical_no_interval": r.get("tactical_no_interval"),
               "reading": "proxy_eval.fam_adverse, W = B, P = the chosen rule: unpaired interval non-overlap (conservative)", "strategic": STRATEGIC})
    return adv, st


# ----------------------------------------------------------------------------------------------------------------- SMOKE (plumbing only)
def smoke(a):
    fit, fit_sha = load_fit(a.fit)
    toks, tl = load_token_map()
    dis = disjointness(fit, tl)
    T = load_table(a.pilot_dir)
    g = check_table_basics(T, toks, tl, full=False)
    eq, n, P, B = b_equals_table_pick(T)
    spec = tuple(fit["chosen"]["spec"])
    C = SR.picks(spec, P)
    cmd, why, exp_info = load_commands(list(T["token"]))
    fp, sampling = aligned_fingerprint(T)
    ar = np.arange(n)
    seam_shapes = {"B_poses": list(T["proposals"][ar, B].shape), "chosen_poses": list(T["proposals"][ar, C].shape)}
    try:
        sys.path.insert(0, os.path.join(PK, "eval"))
        import stage3_full as S3
        f6 = S3.resolve_families6()
        fam_ready = {"families6.py": str(f6) if f6 else None, "tanitad_python_exists": os.path.exists(S3.PY_T)}
    except Exception as e:  # noqa: BLE001
        fam_ready = {"error": f"{type(e).__name__}: {e}"}
    out = {"mode": "SMOKE: plumbing only on the 509-token pilot table; NO PDMS delta, NO bootstrap, NO outcome is computed or stored",
           "fit_sha256": fit_sha, "frozen_rule": fit["chosen"]["name"], "frozen_spec": list(spec),
           "disjointness": dis, "table_shapes": {k: list(T[k].shape) for k in REQUIRED}, "table_dtypes": {k: str(T[k].dtype) for k in REQUIRED},
           "table_gates": g, "B_equals_table_pick": {"equal": eq, "n": n}, "picks": {"B_dtype": str(B.dtype), "chosen_dtype": str(C.dtype), "in_range": bool(((B >= 0) & (B < M)).all() and ((C >= 0) & (C < M)).all())},
           "command_field": {"available": cmd is not None, "status": why, "n_with_command": int(len(cmd)) if cmd is not None else 0,
                             "values_present": sorted(set(cmd.tolist())) if cmd is not None else []},
           "seam_schema": {"fingerprint_available": fp is not None, "sampling_shape": list(sampling.shape) if sampling is not None else None, **seam_shapes},
           "families_instrument_ready": fam_ready}
    out["PLUMBING_OK"] = bool(dis["intersection"] == 0 and all(v for k, v in g.items() if isinstance(v, bool)) and eq == n and out["picks"]["in_range"] and cmd is not None
                              and fp is not None and fam_ready.get("families6.py") and fam_ready.get("tanitad_python_exists"))
    with open(os.path.join(HERE, "SMOKE_PLUMBING.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"disjointness: RL-val logs {dis['rl_val_logs']}, navtest logs {dis['navtest_logs']}, intersection {dis['intersection']}")
    print(f"table shapes: { {k: tuple(T[k].shape) for k in REQUIRED} }")
    print(f"table gates: { {k: v for k, v in g.items()} }")
    print(f"B == table pick on {eq}/{n}; frozen rule {fit['chosen']['name']} applied: picks in range {out['picks']['in_range']}")
    print(f"command field: {out['command_field']}")
    print(f"seam schema: {out['seam_schema']}; families instrument: {fam_ready}")
    print(f"ZZCONFIRM_SMOKE_{'OK' if out['PLUMBING_OK'] else 'NOT_OK'}")
    return 0 if out["PLUMBING_OK"] else 1


# ----------------------------------------------------------------------------------------------------------------- FULL confirm
def full(a):
    t0 = time.time()
    marker = os.path.join(a.table_dir, "P2FULL_DONE")
    if not os.path.exists(marker):
        raise Stop(f"{marker} does not exist: P2's full table is not built yet (the confirm does not run on anything else)")
    if os.path.exists(os.path.join(a.table_dir, "P2FULL_FAILED")):
        raise Stop("P2FULL_FAILED exists next to P2FULL_DONE: the table is not trustworthy")
    out_json = os.path.join(a.out_dir, "CONFIRM.json")
    if os.path.exists(out_json):
        raise Stop(f"{out_json} exists: the confirm is ONE read and has been read")
    os.makedirs(a.out_dir, exist_ok=True)
    fit, fit_sha = load_fit(a.fit)
    if a.expect_fit_sha256 and fit_sha != a.expect_fit_sha256:
        raise Stop(f"FIT.json sha256 {fit_sha} != --expect-fit-sha256 {a.expect_fit_sha256}")
    toks, tl = load_token_map()
    dis = disjointness(fit, tl)
    print(f"DISJOINTNESS: RL-val logs {dis['rl_val_logs']} INTERSECT navtest logs {dis['navtest_logs']} = {dis['intersection']}", flush=True)
    if dis["intersection"] != 0:
        raise Stop(f"RL-val and navtest share {dis['intersection']} logs ({dis['intersection_logs']}): the study STOPS")
    T = load_table(a.table_dir, a.table_name, a.proposals_name)
    g = check_table_basics(T, toks, tl, full=True)
    eq, n, P, B0 = b_equals_table_pick(T)
    g["B_equals_table_pick"] = {"equal": eq, "n": n, "share": eq / n}
    pp_pick = float(100 * SR.at_pick(T["pdms"], T["pick"]).mean())
    g["mean_PDMS_x100_at_table_pick"] = pp_pick
    g["p2_banked_x100"] = SR.P2_BANKED_PDMS_X100
    g["pdms_vs_recomputed_max_abs"] = float(np.abs(T["pdms"] - RR.pdms_v1(T["sub"][..., 0], T["sub"][..., 1], T["sub"][..., 2], T["sub"][..., 3], T["sub"][..., 4])).max())
    bad = [k for k, v in g.items() if isinstance(v, bool) and not v]
    if g["n_tokens"] != N_TOKENS:
        bad.append("n_tokens")
    if eq / n < B_EQ_TABLE_PICK_MIN:
        bad.append("B_equals_table_pick<99%")
    if abs(pp_pick - SR.P2_BANKED_PDMS_X100) > SR.P2_BANKED_TOL:
        bad.append("mean_PDMS_at_table_pick_not_P2_banked")
    if g["n_logs_in_table"] != N_LOGS:
        bad.append("n_logs")
    if bad:
        raise Stop(f"gates failed, nothing was read: {bad}; {g}")
    logs = np.asarray([tl[t] for t in T["token"]])
    cmd, cmd_why, cmd_info = load_commands(list(T["token"]))
    res, B, C = SR.confirm_compute(logs, cmd, T["pdms"], T["sub"], T["logits"], tuple(fit["chosen"]["spec"]), T["pick"])
    fam_adv, fam_status = (None, {"computed": False, "reason": "--no-families"}) if a.no_families else families_leg(T, B, C, a.out_dir, cmd_info["path"] if cmd_info else None)
    cmp_ = res["compare"]
    outc = SR.outcome(cmp_["PDMS"]["ci95"], cmp_["adverse_subscores"], fam_adv, fam_adv is not None)
    out = {"tool": "confirm/1", "tier": "T1 (NAVSIM v1.1 PDMS, ego pseudo-simulation of an open-loop plan, P2's own 64-proposal table, 12,146 tokens / 136 logs)",
           "evidence": "MEASURED (ours); the ONE pre-registered read, rule frozen in FIT.json before this table existed",
           "frozen_rule": fit["chosen"]["name"], "fit_sha256": fit_sha, "fit_rl_val": {"delta_vs_B": fit["chosen"]["delta_vs_B"], "note": "u400 scorer, pre-P2 fan: a transfer, not P2 (Amendment 1, A1)"},
           "estimator": "paired log-cluster bootstrap over the 136 navtest logs, 10,000 resamples, 95 %, seed 20260927 (proxy_eval.boot's stream). It answers 'another draw of LOGS' only: "
                        "P2 is ONE training run (H-ESTIM-SEED-1) and the rule is deterministic (no inference variance).",
           "disjointness": dis, "gates": g, "command_field_status": cmd_why, "result": res, "families": fam_status, "outcome": outc,
           "four_families": {"longitudinal": "EP (paired, result.compare.subscores.EP) + families6 speed / along-track via the family leg", "lateral": "DAC (paired) + families6 heading / cross-track via the family leg",
                             "tactical": "selection quality random / actual / best-of-64 (result.selection_quality)", "strategic": STRATEGIC},
           "provenance": {"table": T["_sha256"], "tokens_md5": TOKENS_MD5, "export": cmd_info, "selrule_py_sha256": SR.sha256_file(os.path.join(HERE, "selrule.py")),
                          "confirm_py_sha256": SR.sha256_file(os.path.join(HERE, "confirm.py")), "fit_selrule_py_sha256": fit["prereg"]["selrule_py_sha256"], "seconds": round(time.time() - t0, 1)}}
    with open(out_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    pd = cmp_["PDMS"]
    md = [f"# CONFIRM: {fit['chosen']['name']} vs B on P2's full navtest table (T1, MEASURED)", "",
          f"**Outcome: {outc['outcome']}** -- {outc['why']}", "",
          f"- DeltaPDMS x100 {pd['mean']:+.4f} [{pd['ci95'][0]:+.4f}, {pd['ci95'][1]:+.4f}] (paired log-cluster bootstrap, 136 logs, B 10,000, seed 20260927); picks changed {cmp_['picks_changed']}/{res['n_tokens']}",
          f"- zero tokens B {cmp_['zero_tokens']['b_zero']} -> chosen {cmp_['zero_tokens']['a_zero']}; DAC=0 {cmp_['dac0']['b']} -> {cmp_['dac0']['a']}; NC=0 {cmp_['nc0']['b']} -> {cmp_['nc0']['a']}",
          f"- EP {cmp_['subscores']['EP']['mean']:+.4f} [{cmp_['subscores']['EP']['ci95'][0]:+.4f}, {cmp_['subscores']['EP']['ci95'][1]:+.4f}]; TTC {cmp_['subscores']['TTC']['mean']:+.4f} [{cmp_['subscores']['TTC']['ci95'][0]:+.4f}, {cmp_['subscores']['TTC']['ci95'][1]:+.4f}]",
          f"- adverse PDMS sub-scores: {', '.join(cmp_['adverse_subscores']) or 'none'}; family leg: {'adverse ' + ', '.join(fam_adv) if fam_adv else ('none adverse' if fam_adv is not None else 'NOT COMPUTED')}",
          "- oracle controls (DeltaPDMS x100, share of the +3.54 SEL ceiling): " + "; ".join(f"{k} {v['delta_vs_B']['mean']:+.3f} ({100 * v['share_of_SEL_ceiling_3.54']:.0f} %)" for k, v in res["oracles"].items()),
          f"- selection quality (random / actual / best-of-64), B: {res['selection_quality']['B']}; chosen: {res['selection_quality']['chosen']}",
          f"- STRATEGIC: {STRATEGIC}", "", "Estimator caveat: answers 'another draw of LOGS' only; P2 is one training run; the rule's parameter was fit on the u400 scorer (Amendment 1 A1)."]
    with open(os.path.join(a.out_dir, "CONFIRM.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print(f"ZZCONFIRM_DONE outcome={outc['outcome']} dPDMS={pd['mean']:+.4f} [{pd['ci95'][0]:+.4f},{pd['ci95'][1]:+.4f}] -> {a.out_dir}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="plumbing only on the pilot table; prints and stores NO PDMS delta")
    ap.add_argument("--fit", default=os.path.join(HERE, "FIT.json"))
    ap.add_argument("--table-dir", default=FULL_DIR)
    ap.add_argument("--table-name", default="table.npz")
    ap.add_argument("--proposals-name", default="proposals.npz")
    ap.add_argument("--pilot-dir", default=PILOT_DIR)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "confirm_out"))
    ap.add_argument("--expect-fit-sha256", default="")
    ap.add_argument("--no-families", action="store_true")
    a = ap.parse_args(argv)
    try:
        return smoke(a) if a.smoke else full(a)
    except Stop as e:
        print(f"ZZCONFIRM_STOP {e}", flush=True)
        return 6


if __name__ == "__main__":
    sys.exit(main())
