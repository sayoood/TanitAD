"""kindx_read.py -- PREREG_V8F_KINDX reader. Written BEFORE any KINDX eval existed (the box was out of commit headroom); UNTESTED at the time of writing.

  python -I kindx_read.py [--selftest-b]

Reuses KIND's instrument (`v8f_kind_read.py` at tip eeb2043e, IMPORTED, never copied): the paired episode-cluster bootstrap (`taniteval.ci`, B = 2,000, 141 episodes), `clears`, `beside`
(LAMBDA, distance keeping K = 10, the paired LON / LAT family blocks, tactical, strategic), `load_strata` / `strata_read` (Amendment 1), `additivity_interval`. Only its module constants
ARM / SEEDS are re-pointed at the one X arm.

DECIDES (PREREG_V8F_KINDX, X = M2b-mixo_s0, T0, one seed): gain_k(X) = ADE6_X(own) - ADE6_X(GT in kind k only).
  CANDIDATE     some kind k has gain_k(X) >= 2F = 0.2415 m (the committed bar, KIND's) AND separated (paired CI excludes 0)
  NO CANDIDATE  otherwise
BESIDE (decides nothing): gain_union, additivity A_X = sum_k gain_k / gain_union, necessity_k(X) (only if the six leave-one-out evals exist: the PREREG's beside table names it but its
10-eval compute list has none; run them AFTER the outcome is written, see HANDOFF), gain_k(X) - gain_k(B s0) paired (B's KIND evals), the four families, the Amendment 1 strata.
Order: controls.json must say PASS (phase 1 AND phase 2) -> the deciding read is written to READ.json FIRST -> only then the beside block is computed and appended.
--selftest-b: run the deciding core on B s0's banked KIND evals as a stand-in for X and require it to reproduce kind_read.json (rows.s0.gain.*.point) to 1e-9."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
OUT = f"{SC}/kindx"
CODE = f"{OUT}/code"
EV = f"{OUT}/eval"
BE = f"{OUT}/b_kind/eval"
T = "M2b-mixo_s0"
PKGCODE = f"{CODE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code"
B_BOOT = 2000
BAR_2F = 0.2415                                   # PREREG_V8F_KINDX: 2F, committed (= KIND's)
NEXT = {"CANDIDATE": ("train X at seed 1 on box 2 (MIXO's argv, `--seed 1`, behind its launch gate) and re-read gain_k at s1. k is a CARRIER only if it clears 2F at s1 too (KIND's both-seed rule). "
                      "A carrier gets a T-side arm aimed at it, under its own PREREG, along KIND's routing: SPEED -> T's speed variance; ANGLE, GEOM or TTE -> the encoder (LoRA); "
                      "TIMEGAP or LEADGAP -> lead perception. Several candidates are ranked by gain_k(X) and all go to the s1 read"),
        "NO CANDIDATE": ("single-kind T-side arms stay unrankable even with an O that consumes partial codes; X is not trained at s1 for this purpose. "
                         "The longitudinal line (KSCORE, then OACC) remains the top lever")}

for _p in (f"{CODE}/taniteval", f"{CODE}/stack"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_s = importlib.util.spec_from_file_location("v8f_kind_read_for_kindx", f"{PKGCODE}/v8f_kind_read.py")
K = importlib.util.module_from_spec(_s)
_s.loader.exec_module(K)
K.ARM, K.SEEDS = "M2b-mixo", (0,)
KINDS = K.KINDS
OWN, ALL, UNION = K.OWN, K.ALL, K.UNION            # the suffixes KIND's helpers key on: kind-own / kind-all / kind-union (+ kind-only-<k>)
assert abs(K.BAR_2F - BAR_2F) < 1e-12


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def refuse(msg):
    print(f"KINDX READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_x():
    """X's evals keyed by KIND's suffixes. Each json is checked: X's checkpoint, the manipulation the arm declares, tactical + strategic families present."""
    from tanitad.models import v8f_mini as M
    R = {}
    names = {OWN: "own", ALL: "all", UNION: "union", **{f"kind-only-{k}": f"only-{k}" for k in KINDS}}
    for suf, nm in names.items():
        E = K.load_eval(EV, f"{T}-kindx-{nm}", need_json=True)
        j = E["json"]
        parts = str(j.get("ckpt", "")).replace("\\", "/").rstrip("/").split("/")
        if not (len(parts) >= 2 and parts[-2] == "M2b_s0_mixo" and parts[-1] == "ckpt_last.pt"):
            refuse(f"{E['name']}: json says ckpt {j.get('ckpt')!r}, not X = M2b_s0_mixo/ckpt_last.pt")
        spec = K.spec_of(suf)
        m = j.get("tac_manipulation") or {}
        if spec is None:
            if j.get("tac") != "pred" or m:
                refuse(f"{E['name']}: not T's own code")
        else:
            n = len(M.code_group_index(spec))
            if m.get("mode") != "oracle" or m.get("groups") != spec or m.get("n_code_dims_replaced") != n or abs(float(m.get("alpha", 0)) - 1.0) > 1e-9:
                refuse(f"{E['name']}: manipulation {m} is not oracle / {spec} / {n} dims / alpha 1")
        for fam in ("tactical", "strategic"):
            if not (j.get("raw") or {}).get(fam):
                refuse(f"{E['name']}: no {fam} family: a missing family is refused, never dropped")
        R[suf] = E
    eid = R[OWN]["eid"]
    for E in R.values():
        if not np.array_equal(E["eid"], eid):
            refuse(f"{E['name']}: not on one window list")
    if (eid.size, np.unique(eid).size) != (K.EVAL_WINDOWS, K.EVAL_EPISODES):
        refuse(f"{eid.size} windows over {np.unique(eid).size} episodes, committed {K.EVAL_WINDOWS} / {K.EVAL_EPISODES}")
    fb = [E["fb"] for s, E in R.items() if s != OWN]
    if any(x is None for x in fb) or any(not np.array_equal(x, fb[0]) for x in fb):
        refuse("the oracle evals disagree about which windows have no v9 row")
    return R, eid


def load_b(eid):
    """B = M2b_s0's KIND evals (banked on box 2, npz only), on the same window list and the same no-v9-row windows."""
    Rb = {}
    for suf, nm in {OWN: "kind-own", ALL: "kind-all", UNION: "kind-union", **{f"kind-only-{k}": f"kind-only-{k}" for k in KINDS}}.items():
        Rb[suf] = K.load_eval(BE, f"M2b_s0-{nm}")
        if not np.array_equal(Rb[suf]["eid"], eid):
            refuse(f"B's {nm} is not on X's window list")
    return Rb


def core(R, eid, pf, n_boot):
    """THE DECIDING READ: gain_k per kind (paired, own - only-k)."""
    own = R[OWN]["ade6"]
    gain = {k: K.PGT.paired(pf, own, R[f"kind-only-{k}"]["ade6"], eid, n_boot) for k in KINDS}
    cand = sorted([k for k in KINDS if K.clears(gain[k], BAR_2F)], key=lambda k: -gain[k]["point"])
    out = "CANDIDATE" if cand else "NO CANDIDATE"
    return gain, cand, out


def selftest_b(pf):
    """The reader's core on B s0's banked KIND evals must reproduce kind_read.json (KIND's committed gains) exactly."""
    Rb = load_b(K.load_eval(BE, "M2b_s0-kind-own")["eid"])
    gain, cand, out = core(Rb, Rb[OWN]["eid"], pf, B_BOOT)
    ref = jl(f"{OUT}/b_kind/kind_read.json")["rows"]["s0"]["gain"]
    dev = {k: abs(gain[k]["point"] - ref[k]["point"]) for k in KINDS}
    ok = all(v < 1e-9 for v in dev.values()) and all(gain[k]["lo"] == ref[k]["lo"] and gain[k]["hi"] == ref[k]["hi"] for k in KINDS)
    print(json.dumps({"selftest_b": "PASS" if ok else "FAIL", "max_point_dev": max(dev.values()), "outcome_on_B": out, "candidates_on_B": cand}, indent=1))
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-b", action="store_true")
    ap.add_argument("--no-beside", action="store_true", help="write the deciding block only")
    a0 = ap.parse_args()
    from taniteval.ci import paired_episode_cluster_bootstrap as pf
    if a0.selftest_b:
        return selftest_b(pf)
    ctl = jl(f"{OUT}/controls.json")
    for ph in ("phase1", "phase2"):
        if not ctl.get(ph, {}).get("PASS"):
            refuse(f"controls.json {ph} is not PASS: no gain may be read")
    R, eid = load_x()
    gain, cand, out = core(R, eid, pf, B_BOOT)
    own, uni, allo = R[OWN]["ade6"], R[UNION]["ade6"], R["kind-all"]["ade6"] if "kind-all" in R else R[ALL]["ade6"]
    g_union = K.PGT.paired(pf, own, uni, eid, B_BOOT)
    g_all = K.PGT.paired(pf, own, allo, eid, B_BOOT)
    s_gain = float(sum(gain[k]["point"] for k in KINDS))
    A = (s_gain / g_union["point"]) if g_union["point"] > 0 else None
    res = {"schema": "v8f-kindx-read-v1", "prereg": "PREREG_V8F_KINDX.md", "tier": "T0 (GT-conditioned oracles): a diagnostic of how O consumes T's code, never a driving claim",
           "estimator": f"paired episode-cluster bootstrap, B={B_BOOT} (episodes), 141 episodes / 2,399 windows; answers 'another draw of EPISODES' only (H-ESTIM-SEED-1); ONE seed: no lever claim from s0",
           "arm": "X = M2b-mixo_s0", "checkpoint_sha256": ctl["phase1"]["checkpoint_sha256"]["observed"], "bar_2F_m": BAR_2F, "controls": {k: ctl[k]["PASS"] for k in ("phase1", "phase2")},
           "ade6_m": {"own": float(own.mean()), "union": float(uni.mean()), "all": float(allo.mean())},
           "gain": gain, "OUTCOME": out, "candidates_ranked": cand, "committed_next": NEXT[out],
           "gain_union": g_union, "gain_all": g_all, "sum_gain": s_gain, "A_X": A}
    res["reading"] = (f"PREREG_V8F_KINDX: {out}" + (f" ({', '.join(cand)})" if cand else "") + "; gain_k(X) m: " +
                      ", ".join(f"{k} {gain[k]['point']:+.3f}{'' if gain[k]['separated'] else ' ns'}" for k in KINDS) + f"; union {g_union['point']:+.3f}; A_X " +
                      ("undefined" if A is None else f"{A:.2f}") + f"; 2F {BAR_2F}.")
    with open(f"{OUT}/READ.json", "w", encoding="utf-8", newline="\n") as f:      # the deciding block is on disk BEFORE any beside read is computed
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(res["reading"])
    if a0.no_beside:
        return 0
    # ---------------------------------------------------------------- beside (decides nothing)
    bes = {}
    bes["A_X_interval"] = K.additivity_interval(R, eid, B_BOOT)
    loo = {k: os.path.exists(f"{EV}/eval_{T}-kindx-loo-{k}.npz") for k in KINDS}
    if all(loo.values()):
        nec = {}
        for k in KINDS:
            E = K.load_eval(EV, f"{T}-kindx-loo-{k}")
            nec[k] = K.PGT.paired(pf, E["ade6"], allo, eid, B_BOOT)
        bes["necessity"] = nec
    else:
        bes["necessity"] = {"status": "NOT COMPUTED", "reason": "the PREREG's 10 evals contain no leave-one-out arm; necessity_k = ADE6(GT everywhere except k) - ADE6(full GT) needs the six loo evals"}
    Rb = load_b(eid)
    if not all(np.array_equal(Rb[s]["fb"], R[ALL]["fb"]) for s in (ALL, UNION)):
        refuse("B's oracle evals and X's disagree about which windows have no v9 row")
    gb = {k: Rb[OWN]["ade6"] - Rb[f"kind-only-{k}"]["ade6"] for k in KINDS}
    bes["gain_B_s0"] = {k: K.PGT.paired(pf, Rb[OWN]["ade6"], Rb[f"kind-only-{k}"]["ade6"], eid, B_BOOT) for k in KINDS}
    bes["gain_union_B_s0"] = K.PGT.paired(pf, Rb[OWN]["ade6"], Rb[UNION]["ade6"], eid, B_BOOT)
    bes["shift_gain_X_minus_B"] = {k: K.PGT.paired(pf, own - R[f"kind-only-{k}"]["ade6"], gb[k], eid, B_BOOT) for k in KINDS}
    bes["shift_gain_union_X_minus_B"] = K.PGT.paired(pf, own - uni, Rb[OWN]["ade6"] - Rb[UNION]["ade6"], eid, B_BOOT)
    # families, LAMBDA, distance keeping, strata: KIND's own functions on X's arms
    a = argparse.Namespace(eval_dir=EV, runs_dir=None, out=None, ckpt_sha=None, n_boot=B_BOOT, dk_n_boot=B_BOOT, no_families=False, strata=str(K.STRATA_DEFAULT), no_strata=False,
                           n_windows=K.EVAL_WINDOWS, n_episodes=K.EVAL_EPISODES, lead_block=None, p10_pred=None, eval_grid=str(K.RAW / "eval_grid_windows.json"),
                           eval_kin=str(K.RAW / "eval141_kin.npz"), eval_join=str(K.RAW / "join_eval141.npz"), gt_npz=None)
    DK, WC = K._modules()
    famfn = K.PGT._famfn(a)
    ctx = K.PGT.build_ctx(a, DK, eid, need_p10=False)
    bes["families"] = K.beside(a, DK, ctx, {0: R}, 0, pf, famfn)
    S = K.load_strata(a.strata, eid, ctx["G"]["frame"], [R[ALL]["fb"]])
    bes["strata"] = K.strata_read(pf, {0: R}, eid, S, B_BOOT)
    bes["families_note"] = "tactical and strategic are identical across the oracle arms by construction (the oracle replaces only the code O reads); lateral / longitudinal are paired vs X's own code"
    res["beside"] = bes
    with open(f"{OUT}/READ.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("beside written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
