"""kindx_read.py -- PREREG_V8F_KINDX reader. Written BEFORE any KINDX eval existed (the box was out of commit headroom); UNTESTED at the time of writing.

  python -I kindx_read.py [--selftest-b]
  python -I kindx_read.py --seed 1 [--check-inputs] [--no-beside]    the SEED-1 CONFIRMATION of CANDIDATE (see below); everything it does not say differently is the s0 code, parametrised

Reuses KIND's instrument (`v8f_kind_read.py` at tip eeb2043e, IMPORTED, never copied): the paired episode-cluster bootstrap (`taniteval.ci`, B = 2,000, 141 episodes), `clears`, `beside`
(LAMBDA, distance keeping K = 10, the paired LON / LAT family blocks, tactical, strategic), `load_strata` / `strata_read` (Amendment 1), `additivity_interval`. Only its module constants
ARM / SEEDS are re-pointed at the one X arm.

DECIDES (PREREG_V8F_KINDX, X = M2b-mixo_s0, T0, one seed): gain_k(X) = ADE6_X(own) - ADE6_X(GT in kind k only).
  CANDIDATE     some kind k has gain_k(X) >= 2F = 0.2415 m (the committed bar, KIND's) AND separated (paired CI excludes 0)
  NO CANDIDATE  otherwise
BESIDE (decides nothing): gain_union, additivity A_X = sum_k gain_k / gain_union, necessity_k(X) (only if the six leave-one-out evals exist: the PREREG's beside table names it but its
10-eval compute list has none; run them AFTER the outcome is written, see HANDOFF), gain_k(X) - gain_k(B s0) paired (B's KIND evals), the four families, the Amendment 1 strata.
Order: controls.json must say PASS (phase 1 AND phase 2) -> the deciding read is written to READ.json FIRST -> only then the beside block is computed and appended.
--selftest-b: run the deciding core on B s0's banked KIND evals as a stand-in for X and require it to reproduce kind_read.json (rows.s0.gain.*.point) to 1e-9.

THE SEED-1 MODE (`--seed 1`, or KINDX_SEED=1; PREREG_V8F_KINDX, CANDIDATE row: "train X at seed 1 on box 2 ... and re-read gain_k at s1. k is a CARRIER only if it clears 2F at s1 too (KIND's both-seed
rule)"). The deciding core is the s0 core, unchanged (gain_k = ADE6_X(own) - ADE6_X(GT in kind k only), paired episode-cluster bootstrap, `clears` = the unrounded point >= 2F = 0.2415 m AND the
paired CI excludes 0). What is new is the OUTCOME, applied to the s0 candidates read from the BANKED s0 READ.json (KINDX_S0_READ; it must say CANDIDATE, with X_s0's checkpoint, the 2F bar and both
control phases PASS):
  CARRIER        some s0 candidate k also clears 2F at s1   -> a T-side arm aimed at it, under its own PREREG, along KIND's routing (the PREREG's text, quoted)
  NOT A CARRIER  no s0 candidate clears 2F at s1            -> none is a carrier; the PREREG commits no step for this case, so the reader says so and quotes KIND's MIXED row; a kind that
                                                               clears at s1 only is reported, and is a carrier at neither seed (KIND Amendment 2)
Refusals (rc 2, nothing written): a failed control phase, an eval of another checkpoint / manipulation / window list, a missing family, an s0 read that is not the banked CANDIDATE.
Order: controls_s1.json must say PASS (phase 1 AND phase 2) -> the deciding block is written to READ_s1.json FIRST -> only then the beside block (gain_B(s1), shift X - B, families, strata, the
s0 -> s1 swing of each gain) is computed and appended. Nothing in the beside block decides anything."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

def _seed_arg(argv):
    for i, x in enumerate(argv):
        if x == "--seed" and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith("--seed="):
            return x.split("=", 1)[1]
    return None


#: the seed this process reads: KINDX_SEED, else `--seed` when run as a script, else 0 (every s0 path, byte for byte)
SEED = int(os.environ["KINDX_SEED"] if "KINDX_SEED" in os.environ else ((_seed_arg(sys.argv[1:]) if __name__ == "__main__" else None) or "0"))
if SEED not in (0, 1):
    raise SystemExit(f"KINDX READ REFUSED: --seed must be 0 or 1, got {SEED}")
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
OUT = os.environ.get("KINDX_OUT", f"{SC}/kindx")
CODE = os.environ.get("KINDX_CODE", f"{OUT}/code")
EV = os.environ.get("KINDX_EV", f"{OUT}/eval")
BE = os.environ.get("KINDX_BE", f"{OUT}/b_kind/eval")
T = f"M2b-mixo_s{SEED}"
XDIR = f"M2b_s{SEED}_mixo"
HERE = os.path.dirname(os.path.abspath(__file__))
S0_READ = os.environ.get("KINDX_S0_READ", os.path.join(HERE, "READ.json"))        # the banked s0 read (raw/2026-10-25-kindx/READ.json)
S0_EV = os.environ.get("KINDX_S0_EV", os.path.join(HERE, "eval"))                  # the banked s0 evals (beside only)
X0_CKPT_SHA = "c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d"
PKGCODE = f"{CODE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code"
B_BOOT = 2000
BAR_2F = 0.2415                                   # PREREG_V8F_KINDX: 2F, committed (= KIND's)
NEXT = {"CANDIDATE": ("train X at seed 1 on box 2 (MIXO's argv, `--seed 1`, behind its launch gate) and re-read gain_k at s1. k is a CARRIER only if it clears 2F at s1 too (KIND's both-seed rule). "
                      "A carrier gets a T-side arm aimed at it, under its own PREREG, along KIND's routing: SPEED -> T's speed variance; ANGLE, GEOM or TTE -> the encoder (LoRA); "
                      "TIMEGAP or LEADGAP -> lead perception. Several candidates are ranked by gain_k(X) and all go to the s1 read"),
        "NO CANDIDATE": ("single-kind T-side arms stay unrankable even with an O that consumes partial codes; X is not trained at s1 for this purpose. "
                         "The longitudinal line (KSCORE, then OACC) remains the top lever")}
#: the seed-1 outcome (see the docstring). The CANDIDATE rule is quoted from NEXT; the not-a-carrier text is the reader's reading of the PREREG's "only if" and KIND's MIXED row, and says so.
NEXT_S1 = {"CARRIER": NEXT["CANDIDATE"],
           "NOT A CARRIER": ("k is a CARRIER only if it clears 2F at s1 too (KIND's both-seed rule), and no candidate did: there is no carrier, so no T-side arm aimed at a kind follows from this PREREG. "
                             "PREREG_V8F_KINDX commits no step for this case; KIND's MIXED row is the nearest committed text ('no branch from this PREREG; each seed reported'), and the next step is "
                             "a Master Mind / PI decision, not a branch this instrument takes")}

for _p in (f"{CODE}/taniteval", f"{CODE}/stack"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_s = importlib.util.spec_from_file_location("v8f_kind_read_for_kindx", f"{PKGCODE}/v8f_kind_read.py")
K = importlib.util.module_from_spec(_s)
_s.loader.exec_module(K)
K.ARM, K.SEEDS = "M2b-mixo", (SEED,)
KINDS = K.KINDS
OWN, ALL, UNION = K.OWN, K.ALL, K.UNION            # the suffixes KIND's helpers key on: kind-own / kind-all / kind-union (+ kind-only-<k>)
assert abs(K.BAR_2F - BAR_2F) < 1e-12


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def refuse(msg):
    print(f"KINDX READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_x(ev=None, t=None, xdir=None):
    """X's evals keyed by KIND's suffixes (default: this seed's, in EV). Each json is checked: X's checkpoint, the manipulation the arm declares, tactical + strategic families present."""
    ev, t, xdir = ev or EV, t or T, xdir or XDIR
    from tanitad.models import v8f_mini as M
    R = {}
    names = {OWN: "own", ALL: "all", UNION: "union", **{f"kind-only-{k}": f"only-{k}" for k in KINDS}}
    for suf, nm in names.items():
        E = K.load_eval(ev, f"{t}-kindx-{nm}", need_json=True)
        j = E["json"]
        parts = str(j.get("ckpt", "")).replace("\\", "/").rstrip("/").split("/")
        if not (len(parts) >= 2 and parts[-2] == xdir and parts[-1] == "ckpt_last.pt"):
            refuse(f"{E['name']}: json says ckpt {j.get('ckpt')!r}, not X = {xdir}/ckpt_last.pt")
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
    """B = M2b_s<seed>'s KIND evals (banked on box 2, npz only), on the same window list and the same no-v9-row windows."""
    Rb = {}
    for suf, nm in {OWN: "kind-own", ALL: "kind-all", UNION: "kind-union", **{f"kind-only-{k}": f"kind-only-{k}" for k in KINDS}}.items():
        Rb[suf] = K.load_eval(BE, f"M2b_s{SEED}-{nm}")
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
    Rb = load_b(K.load_eval(BE, f"M2b_s{SEED}-kind-own")["eid"])
    gain, cand, out = core(Rb, Rb[OWN]["eid"], pf, B_BOOT)
    ref = jl(f"{OUT}/b_kind/kind_read.json")["rows"][f"s{SEED}"]["gain"]
    dev = {k: abs(gain[k]["point"] - ref[k]["point"]) for k in KINDS}
    ok = all(v < 1e-9 for v in dev.values()) and all(gain[k]["lo"] == ref[k]["lo"] and gain[k]["hi"] == ref[k]["hi"] for k in KINDS)
    print(json.dumps({"selftest_b": "PASS" if ok else "FAIL", "max_point_dev": max(dev.values()), "outcome_on_B": out, "candidates_on_B": cand}, indent=1))
    return 0 if ok else 2


def _sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def s0_read_checked():
    """The banked s0 read the seed-1 outcome is applied to. It must be the committed CANDIDATE of X_s0 (else the confirmation has nothing to confirm and the chain should not have run)."""
    if not os.path.exists(S0_READ):
        refuse(f"the banked s0 read {S0_READ} does not exist")
    r = jl(S0_READ)
    why = []
    if r.get("schema") != "v8f-kindx-read-v1":
        why.append(f"schema {r.get('schema')!r}")
    if r.get("OUTCOME") != "CANDIDATE":
        why.append(f"OUTCOME {r.get('OUTCOME')!r}: X is trained at s1 only on a CANDIDATE")
    cand = r.get("candidates_ranked")
    if not (isinstance(cand, list) and cand and all(k in KINDS for k in cand) and len(set(cand)) == len(cand)):
        why.append(f"candidates_ranked {cand!r}")
    if r.get("checkpoint_sha256") != X0_CKPT_SHA:
        why.append(f"checkpoint_sha256 {r.get('checkpoint_sha256')!r} is not X_s0's")
    if not (isinstance(r.get("bar_2F_m"), (int, float)) and abs(float(r["bar_2F_m"]) - BAR_2F) < 1e-12):
        why.append(f"bar_2F_m {r.get('bar_2F_m')!r} != {BAR_2F}")
    if r.get("controls") != {"phase1": True, "phase2": True}:
        why.append(f"controls {r.get('controls')!r}")
    g = r.get("gain") or {}
    for k in KINDS:
        if not (isinstance(g.get(k), dict) and all(x in g[k] for x in ("point", "lo", "hi", "separated"))):
            why.append(f"gain.{k} is not a paired block")
    if why:
        refuse("the banked s0 read is not the CANDIDATE this confirmation is for: " + "; ".join(why))
    return r


def confirm(gain, s0):
    """THE SEED-1 OUTCOME (PREREG_V8F_KINDX CANDIDATE row, KIND's both-seed rule). `gain` = {kind: paired block} at s1; `s0` = the banked s0 read.
    A kind is a CARRIER iff it was a candidate at s0 AND clears 2F at s1 (`clears`: the unrounded point >= 2F and the paired CI excludes 0). The candidates keep the s0 rank (gain_k(X_s0))."""
    cand0 = list(s0["candidates_ranked"])
    clears1 = {k: bool(K.clears(gain[k], BAR_2F)) for k in KINDS}
    carriers = [k for k in cand0 if clears1[k]]
    return {"outcome": "CARRIER" if carriers else "NOT A CARRIER", "carriers": carriers, "candidates_s0": cand0, "clears_s1": clears1,
            "candidates_not_confirmed": [k for k in cand0 if not clears1[k]], "clears_at_s1_only": [k for k in KINDS if clears1[k] and k not in cand0]}


def s1_reading(c, gain, g_union, A):
    parts = [f"PREREG_V8F_KINDX seed-1 confirmation: {c['outcome']}" + (f" ({', '.join(c['carriers'])})" if c["carriers"] else "")]
    parts.append("gain_k(X_s1) m: " + ", ".join(f"{k} {gain[k]['point']:+.3f}{'' if gain[k]['separated'] else ' ns'}" for k in KINDS))
    parts.append(f"s0 candidates {c['candidates_s0']}; 2F {BAR_2F}; union {g_union['point']:+.3f}; A_X " + ("undefined" if A is None else f"{A:.2f}"))
    if c["clears_at_s1_only"]:
        parts.append(f"clears at s1 only (a carrier at neither seed): {c['clears_at_s1_only']}")
    return "; ".join(parts) + "."


def check_inputs_s1(out_path):
    """The pre-flight of the chain (run BEFORE any training, while box 2 is idle): the banked s0 read, B's seed-1 KIND evals (own, all, union, six single kinds) on the committed window list,
    the strata file, the instrument imports. X_s1's own evals do not exist yet and are not looked at."""
    s0 = s0_read_checked()
    own = K.load_eval(BE, "M2b_s1-kind-own")
    eid = own["eid"]
    if (eid.size, np.unique(eid).size) != (K.EVAL_WINDOWS, K.EVAL_EPISODES):
        refuse(f"B s1's KIND evals hold {eid.size} windows over {np.unique(eid).size} episodes, committed {K.EVAL_WINDOWS} / {K.EVAL_EPISODES}")
    Rb = load_b(eid)
    fb = [Rb[s]["fb"] for s in (ALL, UNION, *[f"kind-only-{k}" for k in KINDS])]
    if any(x is None for x in fb) or any(not np.array_equal(x, fb[0]) for x in fb):
        refuse("B s1's oracle evals disagree about which windows have no v9 row")
    if not os.path.exists(str(K.STRATA_DEFAULT)):
        refuse(f"the strata file {K.STRATA_DEFAULT} does not exist")
    from taniteval.ci import paired_episode_cluster_bootstrap as pf        # noqa: F401  (the instrument imports)
    res = {"schema": "v8f-kindx-s1-check-inputs-v1", "s0_read": S0_READ, "s0_read_sha256": _sha256(S0_READ), "s0_candidates_ranked": s0["candidates_ranked"], "bar_2F_m": BAR_2F,
           "b_s1_kind_evals": sorted(Rb), "n_windows": int(eid.size), "n_episodes": int(np.unique(eid).size), "strata_file": str(K.STRATA_DEFAULT),
           "b_s1_gain_own_minus_only_speed_m": float((Rb[OWN]["ade6"] - Rb["kind-only-speed"]["ade6"]).mean())}
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print(f"KINDX S1 INPUTS OK: the banked s0 read is CANDIDATE {s0['candidates_ranked']} (X_s0 {X0_CKPT_SHA[:12]}); B s1's KIND evals ({len(Rb)} arms) on {res['n_windows']} windows / {res['n_episodes']} episodes; strata file present.")
    return 0


def beside_block(a_pf, R, eid, loo_note):
    """Everything that decides nothing (identical for both seeds; the B side is this seed's B)."""
    pf = a_pf
    own, uni, allo = R[OWN]["ade6"], R[UNION]["ade6"], R[ALL]["ade6"]
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
        bes["necessity"] = {"status": "NOT COMPUTED", "reason": loo_note}
    Rb = load_b(eid)
    if not all(np.array_equal(Rb[s]["fb"], R[ALL]["fb"]) for s in (ALL, UNION)):
        refuse(f"B's oracle evals and X's disagree about which windows have no v9 row")
    gb = {k: Rb[OWN]["ade6"] - Rb[f"kind-only-{k}"]["ade6"] for k in KINDS}
    bes[f"gain_B_s{SEED}"] = {k: K.PGT.paired(pf, Rb[OWN]["ade6"], Rb[f"kind-only-{k}"]["ade6"], eid, B_BOOT) for k in KINDS}
    bes[f"gain_union_B_s{SEED}"] = K.PGT.paired(pf, Rb[OWN]["ade6"], Rb[UNION]["ade6"], eid, B_BOOT)
    bes["shift_gain_X_minus_B"] = {k: K.PGT.paired(pf, own - R[f"kind-only-{k}"]["ade6"], gb[k], eid, B_BOOT) for k in KINDS}
    bes["shift_gain_union_X_minus_B"] = K.PGT.paired(pf, own - uni, Rb[OWN]["ade6"] - Rb[UNION]["ade6"], eid, B_BOOT)
    a = argparse.Namespace(eval_dir=EV, runs_dir=None, out=None, ckpt_sha=None, n_boot=B_BOOT, dk_n_boot=B_BOOT, no_families=False, strata=str(K.STRATA_DEFAULT), no_strata=False,
                           n_windows=K.EVAL_WINDOWS, n_episodes=K.EVAL_EPISODES, lead_block=None, p10_pred=None, eval_grid=str(K.RAW / "eval_grid_windows.json"),
                           eval_kin=str(K.RAW / "eval141_kin.npz"), eval_join=str(K.RAW / "join_eval141.npz"), gt_npz=None)
    DK, WC = K._modules()
    famfn = K.PGT._famfn(a)
    ctx = K.PGT.build_ctx(a, DK, eid, need_p10=False)
    bes["families"] = K.beside(a, DK, ctx, {SEED: R}, SEED, pf, famfn)
    S = K.load_strata(a.strata, eid, ctx["G"]["frame"], [R[ALL]["fb"]])
    bes["strata"] = K.strata_read(pf, {SEED: R}, eid, S, B_BOOT)
    bes["families_note"] = "tactical and strategic are identical across the oracle arms by construction (the oracle replaces only the code O reads); lateral / longitudinal are paired vs X's own code"
    return bes


def swing_vs_s0(pf, R1, eid, s0):
    """s1 only, beside: the paired difference gain_k(X_s1) - gain_k(X_s0) per kind on the same windows (the seed-to-seed swing of the gain), from the banked s0 evals; and the PREREG_V8F_KIND
    named-in-advance speed side read on X's two seeds. Not computable (and said so) if the banked s0 evals are not in the tree."""
    try:
        R0, eid0 = load_x(ev=S0_EV, t="M2b-mixo_s0", xdir="M2b_s0_mixo")
    except SystemExit as e:
        return {"status": "NOT COMPUTED", "reason": f"the banked s0 evals could not be loaded from {S0_EV} (rc {e.code})"}
    if not np.array_equal(eid0, eid):
        return {"status": "NOT COMPUTED", "reason": "the banked s0 evals are not on the s1 window list"}
    out = {"estimator": "paired episode-cluster bootstrap on the per-window difference of the two seeds' gains (same windows); the two X are different training runs, so this is descriptive of the seed swing, not a test"}
    out["swing_gain_s1_minus_s0"] = {k: K.PGT.paired(pf, R1[OWN]["ade6"] - R1[f"kind-only-{k}"]["ade6"], R0[OWN]["ade6"] - R0[f"kind-only-{k}"]["ade6"], eid, B_BOOT) for k in KINDS}
    g = [float(s0["gain"]["speed"]["point"]), float((R1[OWN]["ade6"] - R1["kind-only-speed"]["ade6"]).mean())]
    hi, lo = K.SPEED_HI, K.SPEED_LO
    v = "ASSOCIATION-RIGHT" if all(x >= hi for x in g) else ("DOSE-MODEL-RIGHT" if all(x <= lo for x in g) else "NEITHER")
    out["speed_side_read_X"] = {"read": v, "gain_speed_m": g, "bars_m": {"association_right_at_least": hi, "dose_model_right_at_most": lo}, "decides": "nothing alone (PREREG_V8F_KIND)"}
    return out


def _dump(path, res):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, choices=(0, 1), default=None, help="which X to read (must agree with KINDX_SEED when that is set); default 0")
    ap.add_argument("--selftest-b", action="store_true")
    ap.add_argument("--no-beside", action="store_true", help="write the deciding block only")
    ap.add_argument("--check-inputs", action="store_true", help="seed 1 only: the pre-flight of the chain (before any training)")
    ap.add_argument("--out", default=None, help="the result json (default OUT/READ.json at s0, OUT/READ_s1.json at s1)")
    a0 = ap.parse_args(argv)
    if a0.seed is not None and a0.seed != SEED:
        refuse(f"--seed {a0.seed} but this process was configured for seed {SEED} (KINDX_SEED): one of them is wrong")
    from taniteval.ci import paired_episode_cluster_bootstrap as pf
    if a0.selftest_b:
        return selftest_b(pf)
    if a0.check_inputs:
        if SEED != 1:
            refuse("--check-inputs is the seed-1 pre-flight")
        return check_inputs_s1(a0.out or f"{OUT}/check_inputs_s1.json")
    ctl_path = f"{OUT}/controls.json" if SEED == 0 else f"{OUT}/controls_s{SEED}.json"
    out_path = a0.out or (f"{OUT}/READ.json" if SEED == 0 else f"{OUT}/READ_s{SEED}.json")
    s0 = s0_read_checked() if SEED == 1 else None
    ctl = jl(ctl_path)
    for ph in ("phase1", "phase2"):
        if not ctl.get(ph, {}).get("PASS"):
            refuse(f"{os.path.basename(ctl_path)} {ph} is not PASS: no gain may be read")
    R, eid = load_x()
    gain, cand, out = core(R, eid, pf, B_BOOT)
    own, uni, allo = R[OWN]["ade6"], R[UNION]["ade6"], R["kind-all"]["ade6"] if "kind-all" in R else R[ALL]["ade6"]
    g_union = K.PGT.paired(pf, own, uni, eid, B_BOOT)
    g_all = K.PGT.paired(pf, own, allo, eid, B_BOOT)
    s_gain = float(sum(gain[k]["point"] for k in KINDS))
    A = (s_gain / g_union["point"]) if g_union["point"] > 0 else None
    if SEED == 0:
        res = {"schema": "v8f-kindx-read-v1", "prereg": "PREREG_V8F_KINDX.md", "tier": "T0 (GT-conditioned oracles): a diagnostic of how O consumes T's code, never a driving claim",
               "estimator": f"paired episode-cluster bootstrap, B={B_BOOT} (episodes), 141 episodes / 2,399 windows; answers 'another draw of EPISODES' only (H-ESTIM-SEED-1); ONE seed: no lever claim from s0",
               "arm": "X = M2b-mixo_s0", "checkpoint_sha256": ctl["phase1"]["checkpoint_sha256"]["observed"], "bar_2F_m": BAR_2F, "controls": {k: ctl[k]["PASS"] for k in ("phase1", "phase2")},
               "ade6_m": {"own": float(own.mean()), "union": float(uni.mean()), "all": float(allo.mean())},
               "gain": gain, "OUTCOME": out, "candidates_ranked": cand, "committed_next": NEXT[out],
               "gain_union": g_union, "gain_all": g_all, "sum_gain": s_gain, "A_X": A}
        res["reading"] = (f"PREREG_V8F_KINDX: {out}" + (f" ({', '.join(cand)})" if cand else "") + "; gain_k(X) m: " +
                          ", ".join(f"{k} {gain[k]['point']:+.3f}{'' if gain[k]['separated'] else ' ns'}" for k in KINDS) + f"; union {g_union['point']:+.3f}; A_X " +
                          ("undefined" if A is None else f"{A:.2f}") + f"; 2F {BAR_2F}.")
        loo_note = "the PREREG's 10 evals contain no leave-one-out arm; necessity_k = ADE6(GT everywhere except k) - ADE6(full GT) needs the six loo evals"
    else:
        c = confirm(gain, s0)
        res = {"schema": "v8f-kindx-read-s1-v1", "prereg": "PREREG_V8F_KINDX.md (CANDIDATE row: the seed-1 confirmation)", "tier": "T0 (GT-conditioned oracles): a diagnostic of how O consumes T's code, never a driving claim",
               "estimator": f"paired episode-cluster bootstrap, B={B_BOOT} (episodes), 141 episodes / 2,399 windows; answers 'another draw of EPISODES' only (H-ESTIM-SEED-1); the SECOND SEED (a new training run) is the replicate KIND's both-seed rule asks for",
               "arm": f"X = M2b-mixo_s{SEED}", "checkpoint_sha256": ctl["phase1"]["checkpoint_sha256"]["observed"], "bar_2F_m": BAR_2F, "controls": {k: ctl[k]["PASS"] for k in ("phase1", "phase2")},
               "ade6_m": {"own": float(own.mean()), "union": float(uni.mean()), "all": float(allo.mean())},
               "gain": gain, "s0": {"read": S0_READ, "sha256": _sha256(S0_READ), "candidates_ranked": c["candidates_s0"],
                                    "gain": {k: {x: s0["gain"][k][x] for x in ("point", "lo", "hi", "separated")} for k in KINDS}},
               "OUTCOME": c["outcome"], "carriers": c["carriers"], "clears_s1": c["clears_s1"], "candidates_not_confirmed": c["candidates_not_confirmed"], "clears_at_s1_only": c["clears_at_s1_only"],
               "committed_rule_quoted": NEXT["CANDIDATE"], "committed_next": NEXT_S1[c["outcome"]],
               "carrier_routes": {k: K.KIND_ROUTE[k] for k in c["carriers"]},
               "gain_union": g_union, "gain_all": g_all, "sum_gain": s_gain, "A_X": A}
        res["reading"] = s1_reading(c, gain, g_union, A)
        loo_note = "the seed-1 chain runs no leave-one-out arm (the PREREG's 10 evals contain none); necessity_k needs the six loo evals"
    _dump(out_path, res)                                    # the deciding block is on disk BEFORE any beside read is computed
    print(res["reading"])
    if a0.no_beside:
        return 0
    # ---------------------------------------------------------------- beside (decides nothing)
    res["beside"] = beside_block(pf, R, eid, loo_note)
    if SEED == 1:
        res["beside"]["seed_swing_vs_s0"] = swing_vs_s0(pf, R, eid, s0)
    _dump(out_path, res)
    print("beside written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
