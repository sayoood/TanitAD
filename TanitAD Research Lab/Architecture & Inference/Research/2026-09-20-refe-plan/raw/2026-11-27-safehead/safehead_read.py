#!/usr/bin/env python3
"""L2 SAFE-HEAD navtest TABLE READ (PREREG_SAFEHEAD.md 4-6).  CPU only, no GPU, no new seam: a pick's PDMS is the anchor table's tracked value for that proposal.

    prep     python safehead_read.py prep --table "<p5c_next>/propscore/full_C1_s*of2.jsonl" --out <run>/navtest_logs.txt
             (the 136 navtest logs, for the trainer's CTRL-LEAK; also re-checks the anchor table's sha256)
    read     python safehead_read.py read --table ... --proposals <stage2_p5c_c1.proposals.npz> --seam <stage2_p5c_c1.npz> --navtest-index <sealed navtest index>
                 --train-index <the sealed index the heads trained on> --heldout-logs <splits/sft_heldout_logs.txt> --run-s0 <dir> --run-s1 <dir> --run-red <dir>
                 --eval-dir <PK/eval> --card <PREREG_SAFEHEAD.md> --out <dir>
             -> READ_SAFEHEAD.json (every control, M1-M5, bars, the PROVISIONAL outcome) + seams/seam_{SH0,SH1,C1pick}.npz for the families chain
    confirm  python safehead_read.py confirm --read <dir>/READ_SAFEHEAD.json --pass-rows "<fam>/pass/rows_s*.jsonl" --families <fam>/families --tactical <fam>/tactical.json
                 --strategic <fam>/strategic.json --out <dir>
             -> CTRL-T2 (the harness equals the table at the new picks, 1e-6) + the four families -> VERDICT_SAFEHEAD.json (the FINAL outcome)

The only freedom here is mechanics: every bar, control and outcome is the card's, fixed before any SH number existed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUBS = ("nc", "dac", "ttc", "ep", "c", "ddc")
JUDGED = ("nc", "dac", "ttc", "ep", "c")            # G and R; DDC is reported only (it enters neither the aggregate nor PDMS)
# --- the card's pins (PREREG_SAFEHEAD.md 2, 5, 6) ------------------------------------------------------------------------------------
C1_SHA256 = "f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4"
TABLE_SHA256 = {"full_C1_s0of2.jsonl": "9d1239a250c021b89480b0079f2d66ac7b319a3eaee73c80a3deac208b89b42e",
                "full_C1_s1of2.jsonl": "77588849ce5a3080e5ec941f71ba1b52f380fdc04d406849416e8c73c99a99d0"}
PROPOSALS_SHA256 = "b258e50a722101c090861c6f1f0c26d05b38de966fffeb84f0ea17c3dce95007"      # stage2_p5c_c1.proposals.npz (C1's banked dump)
SEAM_SHA256 = "8df306ff940e50d6699942b9e61836c8193b4ed5ef21c32ec0d755859d349995"           # stage2_p5c_c1.npz (C1's banked seam)
C1_PDMS = 92.55477798611808                          # MODEL_REGISTRY 14.6 C1 92.5548; = the table at C1's pick, MEASURED 2026-11-26
N_TOKENS, N_LOGS = 12146, 136
F_REFE = 0.1158                                      # PREREG_C1R_READ.md:35 (one draw, 1 df)
DET_TOL, FEAT_MAX, FEAT_P99, T2_TOL = 1e-4, 1e-4, 1e-4, 1e-6       # s is float32 (DATA_CONTRACT.md v1; CTRL-LOGIT 7.6e-6 MEASURED by the data agent)
ROWS = (0, 1, 3, 5)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _refe(pkg_refe: Path):
    sys.path.insert(0, str(pkg_refe))
    import safehead_data as D
    return D


def load_table(pattern: str, pins=TABLE_SHA256, check_sha=True) -> tuple:
    files = sorted(glob.glob(pattern))
    shas = {Path(f).name: sha256_file(f) for f in files}
    probs = []
    if check_sha:
        if set(shas) != set(pins):
            probs.append(f"table shards {sorted(shas)} != pinned {sorted(pins)}")
        probs += [f"{k} sha256 {v[:12]} != pinned {pins[k][:12]}" for k, v in shas.items() if k in pins and v != pins[k]]
    R = {}
    for f in files:
        for ln in open(f, encoding="utf-8"):
            r = json.loads(ln)
            if "error" not in r:
                R[r["token"]] = r
    return R, shas, probs


def boot_of(eval_dir: str):
    sys.path.insert(0, str(Path(eval_dir) / "p5c"))
    import p5c_read
    if (p5c_read.SEED, p5c_read.NBOOT) != (20260927, 10000):
        raise SystemExit(f"p5c_read.Boot is not the card's estimator (seed {p5c_read.SEED}, draws {p5c_read.NBOOT})")
    return p5c_read.Boot, sha256_file(Path(eval_dir) / "p5c" / "p5c_read.py")


def ci_diff(bt, a, b) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    m = bt.draws(d)
    return {"point": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}


def outcome(m: dict, controls_pass: bool, chosen0: int) -> dict:
    """The card's committed reading (5), mechanically.  `m`: {"s0": {"pdms": ci, "sub": {k: ci}}, "s1": ...}."""
    if not controls_pass:
        return {"outcome": "INVALID", "why": "a control FAILED: every number is descriptive only", "next_arm": "fix the failing control under a dated amendment, then re-read"}
    if chosen0 == 0:
        return {"outcome": "NULL-HELDOUT", "why": "the held-out rule chose step 0 for seed 0: SH_s0 IS C1, navtest is not read for a decision",
                "next_arm": "SH-MLP (card 8.1)"}
    d0, d1 = m["s0"]["pdms"], m["s1"]["pdms"]
    F = abs(m["s0"]["pdms_point_abs"] - m["s1"]["pdms_point_abs"])
    bar2 = 2 * max(F, F_REFE)
    P1 = d0["lo"] > 0
    P2 = d0["point"] >= bar2
    adverse0 = [k for k in JUDGED if m["s0"]["sub"][k]["hi"] < 0]
    G = not adverse0
    adverse1 = [k for k in JUDGED if m["s1"]["sub"][k]["hi"] < 0] + (["pdms"] if d1["hi"] < 0 else [])
    R = d1["point"] > 0 and not adverse1
    bars = {"P1": P1, "P2": P2, "G": G, "R": R, "F": F, "bar_P2": bar2, "adverse_s0": adverse0, "adverse_s1": adverse1}
    if d0["hi"] < 0 or not G:
        o = {"outcome": "ADVERSE", "next_arm": "SH-TOPK (card 8.2), the named adverse sub-scores carried into its guard"}
    elif P1 and P2 and G and R:
        o = {"outcome": "ADOPT", "next_arm": "L3 EP-GEOM inside the SH-scored safe set (card 8.4); TRK+SH / ENS+SH are future cards"}
    elif P1 and G:
        o = {"outcome": "SEPARATED-SMALL", "next_arm": "SH-MLP (card 8.1), 3 seeds"}
    else:
        o = {"outcome": "NULL", "next_arm": "SH-MLP (card 8.1)"}
    o["bars"] = bars
    return o


def cmd_prep(a) -> int:
    R, shas, probs = load_table(a.table, check_sha=not a.no_sha)
    if probs:
        raise SystemExit("REFUSED (CTRL-SHA): " + "; ".join(probs))
    logs = sorted({r["log"] for r in R.values()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(logs) + "\n", encoding="utf-8")
    print(json.dumps({"navtest_logs": len(logs), "tokens": len(R), "out": a.out, "table_sha256": shas}, indent=1))
    print("ZZSH_PREP_OK")
    return 0


def _run_head(D, run: Path, arm: str) -> dict:
    sel = json.loads((run / "heldout_select.json").read_text(encoding="utf-8"))
    rj = json.loads((run / "run.json").read_text(encoding="utf-8"))
    st = int(sel["chosen_step"])
    hp = run / f"head_step{st:05d}.npz"
    listed = dict(reversed(ln.split("  ")) for ln in (run / "heads.sha256").read_text(encoding="utf-8").splitlines() if ln.strip())
    h = D.load_head(hp)
    probs = []
    if listed.get(hp.name) != sha256_file(hp):
        probs.append(f"{arm}: {hp.name} sha256 != heads.sha256")
    m = h["meta"]
    if m.get("smoke") is not False:
        probs.append(f"{arm}: the head is a gate SMOKE head, never a decision head")
    if m.get("init_sha256") != C1_SHA256:
        probs.append(f"{arm}: init sha256 {str(m.get('init_sha256'))[:12]} is not C1's")
    if list(m.get("rows", [])) != list(ROWS):
        probs.append(f"{arm}: rows {m.get('rows')} != {list(ROWS)}")
    want_seed, want_perm = {"s0": (0, -1), "s1": (1, -1), "red": (0, 20261127)}[arm]
    if (m.get("seed"), m.get("permute_labels_seed")) != (want_seed, want_perm):
        probs.append(f"{arm}: seed/permute {m.get('seed')}/{m.get('permute_labels_seed')} != {want_seed}/{want_perm}")
    tok = rj.get("token") or {}
    if tok.get("profile") != {"s0": "refe_safehead_s0", "s1": "refe_safehead_s1", "red": "refe_safehead_red"}[arm]:
        probs.append(f"{arm}: the run was not launched on its own gate token ({tok.get('profile')})")
    return {"head": h, "step": st, "select": sel, "path": str(hp), "sha256": sha256_file(hp), "index_sha256": m.get("index_sha256"),
            "token_sha256": tok.get("token_sha256"), "problems": probs}


def cmd_read(a) -> int:
    D = _refe(Path(a.refe))
    Boot, boot_sha = boot_of(a.eval_dir)
    ctrl: dict = {}
    # ---- CTRL-SHA ------------------------------------------------------------------------------------------------------------------
    R, tshas, tprobs = load_table(a.table, check_sha=not a.no_sha)
    psha = sha256_file(a.proposals)
    ssha = sha256_file(a.seam)
    sha_probs = list(tprobs) + ([] if a.no_sha or psha == PROPOSALS_SHA256 else [f"proposals sha256 {psha[:12]} != pinned {PROPOSALS_SHA256[:12]}"]) \
        + ([] if a.no_sha or ssha == SEAM_SHA256 else [f"seam sha256 {ssha[:12]} != pinned {SEAM_SHA256[:12]}"])
    runs = {k: _run_head(D, Path(p), k) for k, p in (("s0", a.run_s0), ("s1", a.run_s1), ("red", a.run_red))}
    tidx_sha = sha256_file(a.train_index)
    for k, r in runs.items():
        sha_probs += r["problems"]
        if r["index_sha256"] != tidx_sha:
            sha_probs.append(f"{k}: trained on index {str(r['index_sha256'])[:12]}, not --train-index {tidx_sha[:12]}")
    if a.card and a.card_sha256 and sha256_file(a.card) != a.card_sha256:
        sha_probs.append("the card's sha256 is not the frozen one")
    ctrl["CTRL-SHA"] = {"pass": not sha_probs, "problems": sha_probs, "table": tshas, "proposals": psha, "train_index": tidx_sha,
                        "navtest_index": sha256_file(a.navtest_index), "heads": {k: r["sha256"] for k, r in runs.items()}, "boot_p5c_read": boot_sha}
    # ---- the anchor (C1's table, banked dump) --------------------------------------------------------------------------------------
    z = np.load(a.proposals, allow_pickle=False)
    tok = z["token"].astype(str)
    n = len(tok)
    ar = np.arange(n)
    BL, BP = z["logits"].astype(np.float32), z["pick"].astype(np.int64)
    miss = [t for t in tok if t not in R]
    tr = {k: np.array([R[t]["tr"][k] for t in tok], np.float64) for k in SUBS + ("pdms",)} if not miss else None
    logs = np.array([R[t]["log"] for t in tok]) if not miss else None
    tpick = np.array([R[t]["pick"] for t in tok]) if not miss else None
    seam = np.load(a.seam, allow_pickle=False)
    set_ok = (not miss and n == a.n_tokens and len(set(logs)) == a.n_logs and np.array_equal(tpick, BP)
              and np.array_equal(seam["token"].astype(str), tok) and np.array_equal(z["proposals"][ar, BP], seam["poses"]))
    ctrl["CTRL-SET"] = {"pass": bool(set_ok), "n": n, "n_logs": int(len(set(logs))) if logs is not None else None, "missing_in_table": len(miss),
                        "table_pick_equals_dump": bool(tpick is not None and np.array_equal(tpick, BP))}
    if not set_ok:
        _write(a.out, {"controls": ctrl, "outcome": {"outcome": "INVALID", "why": "CTRL-SET"}})
        print("ZZSH_READ INVALID CTRL-SET")
        return 3
    # ---- the navtest s cache -------------------------------------------------------------------------------------------------------
    c = D.load_cache(a.navtest_index, splits=("navtest",), require_sealed=True, labels=False)
    ci = {t: i for i, t in enumerate(c["token"])}
    cmiss = [t for t in tok if t not in ci]
    if cmiss:
        raise SystemExit(f"the navtest cache lacks {len(cmiss)} of the {n} table tokens (first {cmiss[0]})")
    I = np.array([ci[t] for t in tok])
    S, CL, CP = c["s"][I], c["logits"][I], c["pick"][I]
    dl = float(np.abs(CL - BL).max())
    ctrl["CTRL-DET-NT"] = {"pass": bool(np.array_equal(CP, BP) and dl <= DET_TOL), "pick_agree": int((CP == BP).sum()), "n": n, "max_abs_logit": dl, "bar": DET_TOL}
    h0 = runs["s0"]["head"]
    fd = np.abs(D.direct_logits(S, h0["W0"], h0["b0"]) - BL)
    ctrl["CTRL-FEAT-NT"] = {"pass": bool(fd.max() <= FEAT_MAX and np.percentile(fd, 99) <= FEAT_P99), "max_abs": float(fd.max()), "p99_abs": float(np.percentile(fd, 99)),
                            "bars": [FEAT_MAX, FEAT_P99]}
    # ---- CTRL-ID: the untrained head through THIS read --------------------------------------------------------------------------------
    L0 = D.delta_logits(BL, S, h0["W0"], h0["b0"], h0["W0"], h0["b0"], ROWS)
    k0 = D.select(L0)
    pa = tr["pdms"][ar, BP] * 100                                            # the anchor: C1's seam pick, its tracked table value (= the harness CSV, CTRL_T of full_C1_read.json)
    p0 = tr["pdms"][ar, k0] * 100
    ctrl["CTRL-ID"] = {"pass": bool(np.array_equal(L0, BL) and np.array_equal(k0, BP) and np.array_equal(p0, pa) and abs(float(pa.mean()) - a.c1_pdms) <= 5e-5),
                       "logits_bitexact": bool(np.array_equal(L0, BL)), "pick_agree": int((k0 == BP).sum()), "per_token_pdms_exact": bool(np.array_equal(p0, pa)),
                       "pdms": float(p0.mean()), "anchor_pdms": float(pa.mean()), "registry_pdms": a.c1_pdms}
    # ---- CTRL-LEAK -----------------------------------------------------------------------------------------------------------------
    tc = D.read_index(a.train_index)
    nav_logs, nav_tok = set(logs), set(tok)
    tl = {r["log"] for r in tc}
    H = set(D.read_log_list(a.heldout_logs))
    ho = {r["log"] for r in tc if r["split"] == "heldout"}
    lk = {"train_logs_in_navtest": len(tl & nav_logs), "train_tokens_in_navtest": len({r["token"] for r in tc} & nav_tok),
          "heldout_not_sft": len(ho - H), "train_in_sft_heldout": len({r["log"] for r in tc if r["split"] == "train"} & H),
          "train_heldout_overlap": len({r["log"] for r in tc if r["split"] == "train"} & ho), "n_train_logs": len(tl - ho), "n_heldout_logs": len(ho)}
    ctrl["CTRL-LEAK"] = {"pass": all(lk[k] == 0 for k in ("train_logs_in_navtest", "train_tokens_in_navtest", "heldout_not_sft", "train_in_sft_heldout",
                                                          "train_heldout_overlap")), **lk}
    # ---- CTRL-REACH (the runs' own logs) and CTRL-RED (held-out only) --------------------------------------------------------------
    reach = {}
    for k in ("s0", "s1", "red"):
        M = [json.loads(x) for x in (Path(getattr(a, f"run_{k}")) / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        reach[k] = {"steps": len(M), "min_grad": min(min(m["grad_abs"].values()) for m in M), "max_frozen_grad": max(max(m["frozen_grad_abs"].values()) for m in M)}
    ctrl["CTRL-REACH"] = {"pass": all(v["min_grad"] > 0 and v["max_frozen_grad"] == 0.0 and v["steps"] == 2000 for v in reach.values()), **reach}
    rs, r0 = runs["red"]["select"]["per_step"], runs["s0"]["select"]["per_step"]
    fin = str(max(int(x) for x in rs))
    red = {"final_step": int(fin), **{f"auc_{x}": [rs["0"][f"auc_{x}"], rs[fin][f"auc_{x}"]] for x in ("nc", "dac", "ttc")},
           "heldout_safety_miss_rate": [rs["0"]["safety_miss_rate"], rs[fin]["safety_miss_rate"]], "s0_step0_equals_red_step0": rs["0"] == r0["0"]}
    ctrl["CTRL-RED"] = {"pass": bool(all(red[f"auc_{x}"][1] < red[f"auc_{x}"][0] for x in ("nc", "dac", "ttc"))
                                     and red["heldout_safety_miss_rate"][1] > red["heldout_safety_miss_rate"][0] and red["s0_step0_equals_red_step0"]), **red}
    # ---- the arms on navtest -------------------------------------------------------------------------------------------------------
    bt = Boot(logs)
    picks, meas, LS = {}, {}, {"C1": L0}
    for k in ("s0", "s1"):
        h = runs[k]["head"]
        Lk = D.delta_logits(BL, S, h["W"], h["b"], h["W0"], h["b0"], ROWS)
        LS["SH0" if k == "s0" else "SH1"] = Lk
        kk = D.select(Lk)
        picks[k] = kk
        pd = tr["pdms"][ar, kk] * 100
        meas[k] = {"chosen_step": runs[k]["step"], "pdms_point_abs": float(pd.mean()), "pdms": ci_diff(bt, pd, pa),
                   "sub": {s: ci_diff(bt, tr[s][ar, kk] * 100, tr[s][ar, BP] * 100) for s in SUBS},
                   "switch_rate": float((kk != BP).mean()),
                   "direct_form_pick_agree": int((D.select(D.direct_logits(S, h["W"], h["b"])) == kk).sum())}
    controls_pass = all(v["pass"] for v in ctrl.values())
    out = outcome(meas, controls_pass, runs["s0"]["step"])
    # ---- M5 + TACTICAL selection quality (descriptive / family) -----------------------------------------------------------------------
    safe = (tr["nc"] == 1) & (tr["dac"] == 1) & (tr["ttc"] == 1)
    orc = tr["pdms"].max(1) * 100
    m5, tac = {}, {}
    for k, kk in (("C1", BP), ("SH0", picks["s0"]), ("SH1", picks["s1"])):
        m5[k] = {"safety_fail_with_alternative": int(((~safe[ar, kk]) & safe.any(1)).sum()),
                 **{f"pooled_auc_{D.ROW_NAMES[r]}": D.rank_auc(LS[k][..., r], tr[D.ROW_NAMES[r]]) for r in ROWS}}
        tac[k] = {"regret": orc - tr["pdms"][ar, kk] * 100, "top1": (tr["pdms"][ar, kk] * 100 == orc).astype(float),
                  "zeroed": ((tr["pdms"][ar, kk] == 0) & (tr["pdms"].max(1) > 0)).astype(float)}
    for k in ("s0", "s1"):
        nm = "SH0" if k == "s0" else "SH1"
        kk = picks[k]
        m5[nm]["fixed"] = int(((~safe[ar, BP]) & safe[ar, kk]).sum())
        m5[nm]["created"] = int((safe[ar, BP] & ~safe[ar, kk]).sum())
    tac_ci = {nm: {q: ci_diff(bt, tac[nm][q], tac["C1"][q]) for q in ("regret", "top1", "zeroed")} for nm in ("SH0", "SH1")}
    # ---- seams for the families chain (8-pose, like-for-like, build_rule_seams.py form) ----------------------------------------------
    sd = Path(a.out) / "seams"
    sd.mkdir(parents=True, exist_ok=True)
    for nm, kk in (("C1pick", BP), ("SH0", picks["s0"]), ("SH1", picks["s1"])):
        np.savez(sd / f"seam_{nm}.npz", token=tok, fingerprint=seam["fingerprint"], poses=z["proposals"][ar, kk], sampling=seam["sampling"], arm=np.array(nm))
    np.savez(sd / "picks.npz", token=tok, C1=BP, SH0=picks["s0"], SH1=picks["s1"])
    rep = {"card": "PK/raw/2026-11-27-safehead/PREREG_SAFEHEAD.md", "tier": "T1 (NAVSIM v1.1 PDM, LQR-tracked; the anchor table's per-proposal tracked scores)",
           "estimator": "paired log-cluster bootstrap, p5c_read.Boot, 10,000 draws, seed 20260927, 95 % percentile, full_set means; it answers 'another draw of LOGS'",
           "population": f"{n} navtest tokens, {len(set(logs))} logs", "controls": ctrl, "M1_M3": meas, "M5": m5, "tactical_selection_quality": tac_ci,
           "outcome_provisional": out, "pending": "CTRL-T2 and the four families (`confirm`)", "evidence": "MEASURED (this read)"}
    _write(a.out, rep)
    print(f"ZZSH_READ {out['outcome']} (provisional) s0 {meas['s0']['pdms']} s1 {meas['s1']['pdms']['point']:+.4f} controls "
          f"{'PASS' if controls_pass else 'FAIL ' + str([k for k, v in ctrl.items() if not v['pass']])}")
    return 0


def _write(out, rep):
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "READ_SAFEHEAD.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


def cmd_confirm(a) -> int:
    rep = json.loads(Path(a.read).read_text(encoding="utf-8"))
    pk = np.load(Path(a.read).parent / "seams" / "picks.npz", allow_pickle=False)
    R, _, _ = load_table(a.table, check_sha=False)
    rows = {}
    for f in sorted(glob.glob(a.pass_rows)):
        for ln in open(f, encoding="utf-8"):
            r = json.loads(ln)
            if "error" not in r:
                rows[r["token"]] = r
    t2 = {}
    for arm in ("SH0", "SH1", "C1pick"):
        kk = pk["C1" if arm == "C1pick" else arm]
        mx, miss = 0.0, 0
        for t, k in zip(pk["token"].astype(str), kk):
            if t not in rows or arm not in rows[t].get("arms", {}):
                miss += 1
                continue
            got = rows[t]["arms"][arm]["tracked"]
            mx = max(mx, max(abs(float(got[s]) - float(R[t]["tr"][s][int(k)])) for s in SUBS + ("pdms",)))
        t2[arm] = {"max_abs": mx, "missing": miss, "pass": miss == 0 and mx <= T2_TOL}
    ctrl = dict(rep["controls"])
    ctrl["CTRL-T2"] = {"pass": all(v["pass"] for v in t2.values()), **t2}
    fam = {"LONGITUDINAL_LATERAL": _maybe_json(a.families), "TACTICAL_command": _maybe_json(a.tactical), "TACTICAL_selection_quality": rep.get("tactical_selection_quality"),
           "STRATEGIC": _maybe_json(a.strategic)}
    missing = [k for k, v in fam.items() if v is None]
    meas = rep["M1_M3"]
    o = outcome(meas, all(v["pass"] for v in ctrl.values()), meas["s0"]["chosen_step"])
    if missing:
        o = {"outcome": "INCOMPLETE", "why": f"family output(s) missing: {missing} (an eval without its four families is not a result)", "provisional": o}
    v = {**rep, "controls": ctrl, "families": fam, "outcome": o, "pending": None}
    (Path(a.out) / "VERDICT_SAFEHEAD.json").write_text(json.dumps(v, indent=1, default=float), encoding="utf-8")
    print(f"ZZSH_VERDICT {o['outcome']}")
    return 0


def _maybe_json(p):
    if not p:
        return None
    pp = Path(p)
    if pp.is_dir():
        return {f.name: json.loads(f.read_text(encoding="utf-8")) for f in sorted(pp.glob("*.json"))} or None
    return json.loads(pp.read_text(encoding="utf-8")) if pp.is_file() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("prep", "read", "confirm"):
        p = sub.add_parser(nm)
        p.add_argument("--table", required=True)
        p.add_argument("--no-sha", action="store_true", help="SYNTHETIC TESTS ONLY: skip the anchor pins (CTRL-SHA then reads FAIL-free only on synthetic input)")
        if nm == "prep":
            p.add_argument("--out", required=True)
        if nm == "read":
            for f in ("--proposals", "--seam", "--navtest-index", "--train-index", "--heldout-logs", "--run-s0", "--run-s1", "--run-red", "--eval-dir", "--out"):
                p.add_argument(f, required=True)
            p.add_argument("--refe", default=str(HERE.parent.parent / "refe"), help="PK/refe (safehead_data.py)")
            p.add_argument("--card", default="")
            p.add_argument("--card-sha256", default="")
            p.add_argument("--c1-pdms", type=float, default=C1_PDMS)
            p.add_argument("--n-tokens", type=int, default=N_TOKENS)
            p.add_argument("--n-logs", type=int, default=N_LOGS)
        if nm == "confirm":
            for f in ("--read", "--pass-rows", "--out"):
                p.add_argument(f, required=True)
            for f in ("--families", "--tactical", "--strategic"):
                p.add_argument(f, default="")
    a = ap.parse_args(argv)
    return {"prep": cmd_prep, "read": cmd_read, "confirm": cmd_confirm}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
