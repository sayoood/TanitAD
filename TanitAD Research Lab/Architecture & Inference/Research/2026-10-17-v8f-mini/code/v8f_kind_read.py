"""PREREG_V8F_KIND, committed before its data: which KIND of T's constraint cells is worth ADE6 to O?

  python v8f_kind_read.py --eval-dir <eval> --ckpt-sha ckpt_sha256.json --out kind_read.json
  python v8f_kind_read.py --check-inputs --eval-dir <eval> --runs-dir <runs> --out x.json     (the banked M2b side alone, before any box-2 run)

EVAL ONLY, ZERO TRAINING, on the two banked checkpoints M2b_s0 and M2b_s1 (T0: every oracle arm reads the GT; a diagnostic of how O consumes T's code, never a
driving-performance claim). The 26 constraint cells of T's code are partitioned into six KINDS (`v8f_mini.CODE_KINDS`): speed 5, angle 3, tte 5, timegap 2, geom 9,
leadgap 2. Per seed, c = M2b_s<seed>, the evals are `eval_<c>-kind-<name>` with name in

  own          T's own code (`--tac` off, the standard eval)         identity control 1: must reproduce the banked eval_<c>
  all          the full oracle (GT in all 60 code dims)               identity control 2: must reproduce the banked eval_<c>-tacorc
  only-<k>     GT in kind k's cells only, T's own code elsewhere     (6)
  loo-<k>      GT everywhere EXCEPT kind k (`all-<k>`)                (6)
  union        GT in all 26 constraint cells, T's decisions           (1)

= 15 evals per seed, 30 in all. Per seed (paired, same windows, episode-cluster bootstrap B = 2,000):
  gain_k      = ADE6(own) - ADE6(only-k)          (positive = the kind's GT helps)
  necessity_k = ADE6(loo-k) - ADE6(all)           (positive = O misses the kind when it is the one left wrong)
  additivity A = sum_k gain_k / gain_union,  gain_union = ADE6(own) - ADE6(union)

THE COMMITTED READING (PREREG_V8F_KIND.md, verbatim conditions; 2F = 0.2415 m is the committed bar, not recomputed):
  ONE KIND CARRIES   some kind k has gain_k >= 2F, separated, at BOTH seeds
  DIFFUSE            no kind clears 2F at both seeds, and A >= 0.8 at both
  INTERACTIVE        no kind clears 2F at both seeds, and A < 0.8 at both
  MIXED              otherwise (the seeds disagree on A's side, or A is undefined because gain_union <= 0 at a seed)
Named in advance, decides nothing alone: gain_speed >= 0.20 m at both seeds => the cross-arm association was right and the size-agnostic dose model wrong for
speed; <= 0.05 m at both => the reverse. Beside every oracle arm, deciding nothing: the four families (LON incl. distance keeping K = 10, LAT, TACTICAL, STRATEGIC:
NOT PAUSED for v8F-mini) and LAMBDA (the along-track(2 s) difference to the full oracle on the K = 10 LEAD windows; for the all-latcons oracle it is exactly PGT's leak).
The confound is stated in the output: O was trained on WHOLE codes, so a single-kind oracle hands it a MIXED code it never trained on and gain_k is a LOWER bound.

REFUSALS (rc 2, one prefix): an identity control that misses exactly +0 [0, 0] (per-window ade6 equal AND the paired delta 0 [0, 0]); a kind partition that is not a
partition of the 26 cells (restated here from the PREREG's table by cell NAME, then checked against `v8f_mini.CODE_KINDS` index by index and against `constraints`);
an eval whose window set / episode list differs from the banked one (eid, or the committed 2,399 windows over 141 episodes), or whose checkpoint is not
<runs>/M2b_s<seed>/ckpt_last.pt, or whose manipulation record is not the arm's (oracle / spec / n dims / alpha 1); a checkpoint sha256 (the chain's `--ckpt-sha` json)
that differs from the banked run's; a missing eval; a missing family (tactical / strategic in an eval's json, longitudinal / lateral / distance keeping in the read).
ADDED GUARDS (same direction; they refuse, they never decide): the banked F reproduces the PREREG's 0.1207 m (`known_values`); every oracle arm has the same
no-v9-row windows. Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000 (EPISODES): another draw of EPISODES, never another training run
(H-ESTIM-SEED-1); this instrument has no training, its seed replicate is the second banked checkpoint. A's interval is DESCRIPTIVE (the same episode draws).

AMENDMENT 1 (PREREG_V8F_KIND.md, written before any KIND data), BESIDE, DECIDING NOTHING: every single-kind gain_k and the union gain, per WINDOW STRATUM on eval-141 (`strata` in the output):
  JUNCTION  a dominant turn segment in the label band with turn_is_turn = 1     CURVE  curve = 1 (build_v9_labels.py:828-831; a bend the nav never announces)     NO-TURN  every other window
read at the T-tick label row the tactical layer reads (rowT = off + t - a_T). The per-window labels are `raw/2026-10-25-kind/window_strata_eval141.npz`, built from the v9 release by
`raw/2026-10-25-kind/mk_window_strata.py` (the shipped join does not carry the two columns). Each stratum: n windows, n episodes, and the paired episode-cluster bootstrap WITHIN the
stratum (B = 2,000, the stratum's own episodes); a stratum with fewer than 10 episodes is reported with its n and NO interval. The stratum read is computed from the same per-window ade6
as the deciding read and feeds nothing back: the outcome and the refusals above are unchanged (a test permutes the stratum labels and requires the outcome identical). A strata file that
is not the eval grid's (eid / tick / v9-row presence / label-derived stratum) is refused like any other input that is not the banked one."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
RAW = PKG / "raw"
ARM, SEEDS = "M2b", (0, 1)
B = 2000
BAR_2F = 0.2415                          # PREREG_V8F_KIND: 2F, committed (F = |ADE6 M2b_s0 - M2b_s1| = 0.1207 m on the banked evals)
ADD_BAR = 0.8                            # additivity A: DIFFUSE at >= 0.8, INTERACTIVE below
SPEED_HI, SPEED_LO = 0.20, 0.05          # the named-in-advance SPEED side-read
EVAL_WINDOWS, EVAL_EPISODES = 2399, 141  # the committed eval-141 grid
IDX_2S = 9                               # pred[:, 9] = t + 2.0 s (0.2 s steps): the K = 10 block's 2 s horizon
K_DK = 10
STRATA = ("JUNCTION", "CURVE", "NO-TURN")             # Amendment 1; stratum code 0 / 1 / 2 in the strata file
MIN_STRATUM_EPISODES = 10                             # Amendment 1: fewer episodes -> n only, no interval
STRATA_DEFAULT = RAW / "2026-10-25-kind" / "window_strata_eval141.npz"
RELEASE_MD5 = "6b5c7f207cffc3b7eb3cd527fd433599"      # the v9 eval139 release the strata were built from
# The PREREG's partition table VERBATIM, by cell name. Restated here on purpose: a model whose CODE_KINDS had a cell moved must be REFUSED by this reader,
# not silently followed (a test pins this table equal to the model's, and mutates the model to prove the refusal).
KIND_CELLS = {
    "speed": ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms"),
    "angle": ("lat_theta_deg", "turn_dyaw_deg", "turn_exit_psi_deg"),
    "tte": ("turn_t_start_s", "turn_t_end_s", "lc_t_cross_s", "lon_t_reach_s", "lon_t_start_s"),
    "timegap": ("lead_tg_s", "lead_tg_min_s"),
    "geom": ("turn_d_start_m", "turn_len_m", "turn_r_arc_m", "turn_exit_x_m", "turn_exit_y_m", "lc_d_cross_m", "lon_d_reach_m", "stop_x_m", "stop_y_m"),
    "leadgap": ("lead_gap_m", "lead_gap_min_m"),
}
KINDS = tuple(KIND_CELLS)
N_CONS = 26
# The banked checkpoints (raw/full/MANIFEST.md): the chain asserts them on box 2 before and after the evals and writes the observed values to the `--ckpt-sha` json.
BANKED_CKPT_SHA = {"M2b_s0": "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8",
                   "M2b_s1": "6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6"}
OWN, ALL, UNION = "kind-own", "kind-all", "kind-union"
FAMILIES = ("longitudinal", "lateral", "tactical", "strategic")
OUTCOMES = ("ONE KIND CARRIES", "DIFFUSE", "INTERACTIVE", "MIXED")
# The committed next steps, VERBATIM from PREREG_V8F_KIND.md's table (a test compares them with the file).
NEXT_TEXT = {
    "ONE KIND CARRIES": "a T-side arm aimed at kind k, under its own PREREG. SPEED points to T's speed variance (two-seed code average / EMA); ANGLE, GEOM or TTE to the encoder (raises LoRA's prior); TIMEGAP or LEADGAP to lead perception",
    "DIFFUSE": "no single T-side lever; next is O-side: train O on CELL-WISE mixed codes, under its own PREREG",
    "INTERACTIVE": "O needs most cells right at once; same O-side next, stronger prior; T-side single-kind arms deprioritised",
    "MIXED": "no branch from this PREREG; each seed reported; the SPEED read below still stands"}
KIND_ROUTE = {"speed": "T's speed variance (two-seed code average / EMA)", "angle": "the encoder (raises LoRA's prior)", "geom": "the encoder (raises LoRA's prior)",
              "tte": "the encoder (raises LoRA's prior)", "timegap": "lead perception", "leadgap": "lead perception"}
CONFOUND = ("O was trained on WHOLE codes: GT with probability p_gt (1.0 -> 0.2), else T's own, one switch per sample over the whole code (v8f_mini.py:620-622 at tip "
            "a61d3170). A single-kind oracle hands O a MIXED code it never trained on, so gain_k is a LOWER bound on kind k's worth. ONE KIND CARRIES is robust to that (a lower "
            "bound that clears 2F clears it). DIFFUSE and INTERACTIVE cannot tell 'the kinds are individually small' from 'O cannot use a mixed code': both point to the same "
            "O-side arm, which removes the mismatch, and this instrument re-reads its result.")


def refuse(msg):
    print(f"KIND READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# The programme's own instruments, IMPORTED (never copied): PGT's reader brings the paired contrast with its unrounded `point`, the lead-block context, the K = 10
# distance-keeping call, the family block, and the banked-side check (runs, evals, LAMBDA / S5 / F known values, the identity control). Their refusals are routed
# through THIS reader's refuse (one prefix, one rc).
PGT = _load("v8f_pgt_read_for_kind", HERE.parent / "v8f_pgt_read.py")
PGT.refuse = refuse


def _modules():
    return PGT._modules()


def arm_suffixes():
    """The 15 per-seed eval suffixes, in the order the queue runs them: own, all, 6 single-kind, 6 leave-one-out, union."""
    return [OWN, ALL] + [f"kind-only-{k}" for k in KINDS] + [f"kind-loo-{k}" for k in KINDS] + [UNION]


def spec_of(suffix):
    """The `--tac-groups` spec an oracle arm was run with; None for T's own code."""
    if suffix == OWN:
        return None
    if suffix == ALL:
        return "all"
    if suffix == UNION:
        return "constraints"
    if suffix.startswith("kind-only-"):
        return suffix[len("kind-only-"):]
    if suffix.startswith("kind-loo-"):
        return "all-" + suffix[len("kind-loo-"):]
    raise KeyError(suffix)


# ------------------------------------------------------------------------------------------------ the partition
def check_partition(M):
    """The kinds PARTITION the 26 constraint cells and their union IS `constraints`, index for index. Restated from the PREREG's table by name, then compared with the model's."""
    names = [n for k in KINDS for n in KIND_CELLS[k]]
    if len(names) != N_CONS or len(set(names)) != N_CONS or set(names) != set(M.CONS_NAMES):
        dup = sorted({n for n in names if names.count(n) > 1})
        refuse(f"the PREREG's kind table is not a partition of the {N_CONS} constraint cells (cells {len(names)}, distinct {len(set(names))}, repeated {dup}, "
               f"missing {sorted(set(M.CONS_NAMES) - set(names))})")
    c0 = M.N_GOAL + M.N_LAT + M.N_LON
    kinds = getattr(M, "CODE_KINDS", None)
    if not isinstance(kinds, dict) or tuple(kinds) != KINDS:
        refuse(f"v8f_mini.CODE_KINDS is {sorted(kinds) if isinstance(kinds, dict) else kinds!r}, the PREREG's kinds are {list(KINDS)}")
    for k in KINDS:
        want = sorted(c0 + M.CI[n] for n in KIND_CELLS[k])
        if sorted(kinds[k]) != want:
            refuse(f"v8f_mini.CODE_KINDS[{k!r}] = {sorted(kinds[k])} is not the PREREG's {k} cells {want}: a cell moved between kinds")
    union = sorted(i for k in KINDS for i in kinds[k])
    cons = list(M.code_group_index("constraints"))
    if union != cons or len(union) != len(set(union)):
        refuse(f"the union of the six kinds is not the `constraints` composite index for index ({len(union)} indices vs {len(cons)})")
    for k in KINDS:
        if list(M.code_group_index(k)) != sorted(kinds[k]):
            refuse(f"code_group_index({k!r}) does not return the kind's cells")
        rest = [i for i in range(M.CODE_DIM) if i not in set(kinds[k])]
        if list(M.code_group_index(f"all-{k}")) != rest:
            refuse(f"code_group_index('all-{k}') is not the complement of the kind")
    return {"cells": {k: list(KIND_CELLS[k]) for k in KINDS}, "n_cells": {k: len(KIND_CELLS[k]) for k in KINDS}, "n_total": len(names), "union_is_constraints": True}


# ------------------------------------------------------------------------------------------------ inputs
def _jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ckpt_ok(js, tag):
    """The eval's json `ckpt` names <runs>/M2b_s<seed>/ckpt_last.pt: an eval of another checkpoint under this tag is refused."""
    ck = js.get("ckpt")
    parts = str(ck or "").replace("\\", "/").rstrip("/").split("/")
    return len(parts) >= 2 and parts[-2] == tag and parts[-1] == "ckpt_last.pt", ck


def load_eval(eval_dir, name, need_json=False):
    pz, pj = os.path.join(eval_dir, f"eval_{name}.npz"), os.path.join(eval_dir, f"eval_{name}.json")
    if not os.path.exists(pz):
        refuse(f"eval_{name}.npz is missing from {eval_dir}")
    z = np.load(pz)
    for k in ("eid", "ade6", "pred"):
        if k not in z.files:
            refuse(f"eval_{name}.npz carries no {k}")
    js = None
    if os.path.exists(pj):
        js = _jload(pj)
    elif need_json:
        refuse(f"eval_{name}.json is missing from {eval_dir}")
    ade = np.asarray(z["ade6"], np.float64)
    if not np.isfinite(ade).all():
        refuse(f"eval_{name}.npz holds a non-finite ade6")
    return {"name": name, "eid": np.asarray(z["eid"]), "ade6": ade, "pred": np.asarray(z["pred"], np.float64), "json": js,
            "fb": np.asarray(z["tac_oracle_fallback"], bool) if "tac_oracle_fallback" in z.files else None}


def check_arm(M, c, suffix, E):
    """One new arm's json: it is this tag, on this checkpoint, with the manipulation the arm declares, and it carries the TACTICAL and STRATEGIC families."""
    name, js = E["name"], E["json"]
    if js.get("tag") not in (None, name):
        refuse(f"eval_{name}.json says tag {js.get('tag')!r}")
    ok, ck = _ckpt_ok(js, c)
    if not ok:
        refuse(f"eval_{name}.json scored {ck!r}, not <runs>/{c}/ckpt_last.pt")
    spec = spec_of(suffix)
    m = js.get("tac_manipulation")
    if spec is None:
        if js.get("tac") != "pred" or m:
            refuse(f"eval_{name}.json is not T's own code (tac {js.get('tac')!r}, manipulation {m!r}): the identity control needs `--tac` off")
    else:
        n = len(M.code_group_index(spec))
        m = m or {}
        alpha = m.get("alpha")
        if m.get("mode") != "oracle" or m.get("groups") != spec or m.get("n_code_dims_replaced") != n or alpha is None or abs(float(alpha) - 1.0) > 1e-9:
            refuse(f"eval_{name}.json: manipulation {m.get('mode')} / {m.get('groups')} / {m.get('n_code_dims_replaced')} dims / alpha {alpha}, "
                   f"expected oracle / {spec} / {n} / 1.0")
    raw = js.get("raw") or {}
    for fam in ("tactical", "strategic"):
        if not raw.get(fam):
            refuse(f"eval_{name}.json carries no {fam} family (raw.{fam}): a missing family is refused, never dropped")


def load_all(eval_dir, M):
    """-> R[seed][suffix] for the 15 new evals, B[seed] = {'std', 'orc'} the banked controls, eid. Every file on ONE window list."""
    R, Bk, eid = {}, {}, None
    for s in SEEDS:
        c = f"{ARM}_s{s}"
        Bk[s] = {"std": load_eval(eval_dir, c), "orc": load_eval(eval_dir, f"{c}-tacorc")}
        R[s] = {}
        for suf in arm_suffixes():
            E = load_eval(eval_dir, f"{c}-{suf}", need_json=True)
            check_arm(M, c, suf, E)
            R[s][suf] = E
        for E in list(Bk[s].values()) + list(R[s].values()):
            if eid is None:
                eid = E["eid"]
            elif not np.array_equal(E["eid"], eid):
                refuse(f"eval_{E['name']}.npz is not on the banked window list (eid differs): the window set or the episode list moved")
        for E in R[s].values():
            if E["pred"].shape != Bk[s]["std"]["pred"].shape or E["ade6"].shape != Bk[s]["std"]["ade6"].shape:
                refuse(f"eval_{E['name']}.npz has pred {E['pred'].shape} / ade6 {E['ade6'].shape}, the banked eval has {Bk[s]['std']['pred'].shape} / {Bk[s]['std']['ade6'].shape}")
        fb = [E["fb"] for suf, E in R[s].items() if suf != OWN]
        if any(x is None for x in fb):
            refuse(f"an oracle eval of {c} carries no tac_oracle_fallback: the windows without a v9 row cannot be matched")
        if any(not np.array_equal(x, fb[0]) for x in fb):
            refuse(f"the oracle evals of {c} disagree about which windows have no v9 row")
    return R, Bk, eid


def check_counts(eid, a):
    n, ne = int(eid.size), int(np.unique(eid).size)
    if (n, ne) != (a.n_windows, a.n_episodes):
        refuse(f"the evals hold {n} windows over {ne} episodes, the committed eval-141 grid has {a.n_windows} over {a.n_episodes}")
    return n, ne


def check_ckpt_sha(path):
    if not path:
        refuse("--ckpt-sha is required: the chain's json of the checkpoint sha256 it measured on box 2 (the banked runs' shas are asserted here)")
    if not os.path.exists(path):
        refuse(f"--ckpt-sha {path} does not exist")
    got = _jload(path)
    out = {}
    for tag, want in BANKED_CKPT_SHA.items():
        have = got.get(tag)
        if have != want:
            refuse(f"the checkpoint {tag} has sha256 {str(have)[:16]}..., the banked run's is {want[:16]}...: this is not the banked checkpoint")
        out[tag] = have
    return out


# ------------------------------------------------------------------------------------------------ the identity controls
def _exact_zero(r):
    return r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 and r["point"] == 0.0


def identity_controls(pf, R, Bk, eid, n_boot):
    """T's own code (`--tac` off) must reproduce the banked eval; the full oracle must reproduce the banked -tacorc eval: per window, ADE6 delta exactly +0 [0, 0]."""
    out = {}
    for s in SEEDS:
        c = f"{ARM}_s{s}"
        out[f"s{s}"] = {}
        for nm, suf, banked in (("own_vs_banked_eval", OWN, Bk[s]["std"]), ("all_vs_banked_tacorc", ALL, Bk[s]["orc"])):
            new = R[s][suf]
            same = bool(np.array_equal(new["ade6"], banked["ade6"]))
            r = PGT.paired(pf, new["ade6"], banked["ade6"], eid, n_boot)
            if not same or not _exact_zero(r):
                d = float(np.abs(new["ade6"] - banked["ade6"]).max())
                refuse(f"IDENTITY CONTROL FAILED: eval_{new['name']} - eval_{banked['name']} reads {r['delta']} [{r['lo']}, {r['hi']}] (point {r['point']}, per-window max |dev| "
                       f"{d:.3g} m), not exactly +0 [0, 0]: the box-2 re-run does not reproduce the banked eval")
            out[f"s{s}"][nm] = {**{x: r[x] for x in ("delta", "lo", "hi", "point")}, "per_window_equal": same,
                                "pred_max_abs_dev_m": float(np.abs(new["pred"] - banked["pred"]).max())}
    return out


def known_values(F):
    """The banked M2b seed floor reproduces the PREREG's (0.1207 m): the inputs are the ones the PREREG was written against."""
    if abs(F - PGT.KNOWN["F"]) > PGT.KNOWN_TOL["F"]:
        refuse(f"F reads {F:.5f} m on the banked M2b evals, the PREREG's is {PGT.KNOWN['F']:.4f} m: not the inputs the PREREG was written against")
    return {"checked": True, "F": F, "banked": PGT.KNOWN["F"], "tolerance": PGT.KNOWN_TOL["F"]}


# ------------------------------------------------------------------------------------------------ the reads
def seed_reads(pf, R, eid, n_boot):
    """gain_k, necessity_k, the union gain and additivity A at ONE seed (paired, same windows)."""
    own, allo, uni = R[OWN]["ade6"], R[ALL]["ade6"], R[UNION]["ade6"]
    gain = {k: PGT.paired(pf, own, R[f"kind-only-{k}"]["ade6"], eid, n_boot) for k in KINDS}
    nec = {k: PGT.paired(pf, R[f"kind-loo-{k}"]["ade6"], allo, eid, n_boot) for k in KINDS}
    g_union = PGT.paired(pf, own, uni, eid, n_boot)
    g_all = PGT.paired(pf, own, allo, eid, n_boot)
    s_gain = float(sum(gain[k]["point"] for k in KINDS))
    A = (s_gain / g_union["point"]) if g_union["point"] > 0 else None
    return {"ade6": {"own": float(own.mean()), "all": float(allo.mean()), "union": float(uni.mean())}, "gain": gain, "necessity": nec, "gain_union": g_union, "gain_all": g_all,
            "sum_gain": s_gain, "A": A, "A_defined": A is not None}


def additivity_interval(R, eid, n_boot):
    """DESCRIPTIVE interval on A: the same episode draws as the paired estimator (ci._draws, seed 0). The deciding A is the point estimate."""
    from taniteval import ci
    uniq, idx = ci.episode_index(eid)
    own, uni = R[OWN]["ade6"], R[UNION]["ade6"]
    ones = [R[f"kind-only-{k}"]["ade6"] for k in KINDS]
    vals = []
    for sel in ci._draws(uniq, idx, n_boot, 0):
        mo = own[sel].mean()
        gu = mo - uni[sel].mean()
        if gu > 0:
            vals.append(sum(mo - o[sel].mean() for o in ones) / gu)
    if len(vals) < max(2, n_boot // 2):
        return {"status": "NOT-DEFINED", "reason": f"gain_union > 0 on only {len(vals)} of {n_boot} draws", "estimator": "descriptive"}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "n_draws": len(vals), "estimator": "descriptive: percentile of A over the paired estimator's own episode draws"}


def clears(g, bar=BAR_2F):
    """gain_k clears the bar: point >= 2F (the UNROUNDED difference) and separated (the interval excludes 0) on the gain side."""
    return bool(g["point"] >= bar and g["separated"] and g["point"] > 0)


def decide(rows, bar=BAR_2F, add_bar=ADD_BAR):
    """The committed table, from the per-seed gains and A. rows[seed] = {'gain': {kind: paired}, 'A': float | None}."""
    ok = {s: {k: clears(rows[s]["gain"][k], bar) for k in KINDS} for s in SEEDS}
    carriers = [k for k in KINDS if all(ok[s][k] for s in SEEDS)]
    A = [rows[s]["A"] for s in SEEDS]
    if carriers:
        out = "ONE KIND CARRIES"
    elif all(x is not None and x >= add_bar for x in A):
        out = "DIFFUSE"
    elif all(x is not None and x < add_bar for x in A):
        out = "INTERACTIVE"
    else:
        out = "MIXED"
    one_seed = {k: [s for s in SEEDS if ok[s][k]] for k in KINDS if len([s for s in SEEDS if ok[s][k]]) == 1}
    return {"outcome": out, "carriers": carriers, "clears": {f"s{s}": ok[s] for s in SEEDS}, "kinds_clearing_at_one_seed_only": {k: [f"s{s}" for s in v] for k, v in one_seed.items()},
            "A": A, "A_bar": add_bar, "two_F_bar": bar,
            "per_seed_row": {f"s{s}": ("ONE KIND (" + "+".join(k for k in KINDS if ok[s][k]) + ")") if any(ok[s].values())
                             else ("A UNDEFINED" if rows[s]["A"] is None else ("DIFFUSE" if rows[s]["A"] >= add_bar else "INTERACTIVE")) for s in SEEDS}}


def speed_side_read(rows):
    """Named in advance, decides nothing alone."""
    g = [rows[s]["gain"]["speed"]["point"] for s in SEEDS]
    if all(x >= SPEED_HI for x in g):
        v = "ASSOCIATION-RIGHT"
        why = "gain_speed >= 0.20 m at both seeds: the cross-arm association (0.78-0.90 m per SD^2) was right and the size-agnostic dose model (0.033 m) wrong for speed"
    elif all(x <= SPEED_LO for x in g):
        v = "DOSE-MODEL-RIGHT"
        why = "gain_speed <= 0.05 m at both seeds: the size-agnostic dose model (0.033 m) was right and the cross-arm association wrong for speed"
    else:
        v = "NEITHER"
        why = "gain_speed is between 0.05 m and 0.20 m at one or both seeds: neither estimator is confirmed for speed"
    return {"read": v, "gain_speed_m": g, "bars_m": {"association_right_at_least": SPEED_HI, "dose_model_right_at_most": SPEED_LO}, "reading": why, "decides": "nothing alone"}


# ------------------------------------------------------------------------------------------------ beside: the four families and LAMBDA
def _check_family_block(name, blk):
    miss = [f for f in FAMILIES if not blk.get(f)]
    if miss:
        refuse(f"the family block of {name} is missing {miss}: a missing family is refused, never dropped")


def beside(a, DK, ctx, R, s, pf, famfn):
    """Per oracle arm at one seed, deciding nothing: LAMBDA, LON (paired family block + distance keeping K = 10), LAT, TACTICAL, STRATEGIC."""
    c = f"{ARM}_s{s}"
    names = {suf: f"{c}-{suf}" for suf in R[s]}
    eid, lead = ctx["eid"], ctx["lead"]
    preds = {names[suf]: R[s][suf]["pred"] for suf in R[s]}
    saved = {names[suf]: R[s][suf]["ade6"] for suf in R[s]}
    pairs = [(names[suf], names[OWN]) for suf in R[s] if suf != OWN]
    try:
        d10 = DK.read_dk(preds=preds, gt=ctx["G"]["gt"], eid=eid, clip=ctx["G"]["clip"], frame=ctx["G"]["frame"], v0=ctx["G"]["v0"], block=ctx["blk10"], k=K_DK, dt=ctx["G"]["dt"],
                         n_boot=a.dk_n_boot, seed=0, saved_ade6=saved, p10_floor_ade6=None, pairs=pairs)
    except DK.Refusal as e:
        refuse(f"v8f_dk_read: {e}")
    along = {suf: R[s][suf]["pred"][:, IDX_2S, 0] for suf in R[s]}
    out = {}
    for suf in R[s]:
        nm = names[suf]
        lam = along[suf] - along[ALL]
        rec = {"mean_lead_m": float(lam[lead].mean()), "n_lead_windows": int(lead.sum()),
               "paired_vs_full_oracle": PGT.paired(pf, along[suf][lead], along[ALL][lead], eid[lead], a.n_boot)} if lead.any() else {"status": "NOT-DEFINED", "reason": "no LEAD window in the K = 10 block"}
        blk = {"lambda": rec,
               "longitudinal": {"distance_keeping_k10": {"arm": d10["read"]["arms"].get(nm), "pair_vs_own": d10["read"]["pairs"].get(f"{nm}__minus__{names[OWN]}")}},
               "tactical": R[s][suf]["json"]["raw"]["tactical"], "strategic": R[s][suf]["json"]["raw"]["strategic"]}
        if not blk["longitudinal"]["distance_keeping_k10"]["arm"]:
            refuse(f"the distance-keeping read has no arm block for {nm}")
        if suf != OWN and not blk["longitudinal"]["distance_keeping_k10"]["pair_vs_own"]:
            refuse(f"the distance-keeping read has no pair block for {nm} - {names[OWN]}")
        if famfn is not None:
            fb = famfn(R[s][suf]["pred"], R[s][OWN]["pred"], eid)
            blk["longitudinal"]["paired_vs_own"] = fb.get("longitudinal")
            blk["lateral"] = {"paired_vs_own": fb.get("lateral")}
            if not blk["longitudinal"]["paired_vs_own"] or not blk["lateral"]["paired_vs_own"]:
                refuse(f"the paired family block of {nm} has no longitudinal / lateral component: a missing family is refused")
        else:
            blk["lateral"] = {"status": "SKIPPED (--no-families, a TEST input; the chain never passes it)"}
        _check_family_block(nm, blk)
        out[suf] = blk
    return out


# ------------------------------------------------------------------------------------------------ Amendment 1: the window strata (beside, deciding nothing)
def stratum_codes(curve, turn_is_turn, has):
    """0 JUNCTION / 1 CURVE / 2 NO-TURN per window from the two label columns read at the T-tick row. JUNCTION wins a tie; a window with no v9 row is NO-TURN (every other window).
    Restated here (not imported from mk_window_strata.py) so that a strata file whose `stratum` is not this function of its own label columns is REFUSED."""
    s = np.full(np.shape(curve), 2, np.int8)
    s[(np.asarray(curve) == 1) & np.asarray(has, bool)] = 1
    s[(np.asarray(turn_is_turn) == 1) & np.asarray(has, bool)] = 0
    return s


def load_strata(path, eid, frame, fbs):
    """The per-window stratum labels of the eval-141 grid, VALIDATED against the evals: one stratum per window; the same episode index per window; the same tick (frame = 2t);
    `stratum` equal to its own label columns; the v9-row presence equal to the oracle evals' own fallback mask (`has == ~tac_oracle_fallback`, which ties the file's T-tick row
    pairing to the rows the evaluator actually read); and the provenance names the release the PREREG's strata were built from. Anything else is refused: a stratum split on the
    wrong windows is a wrong answer that reads like a right one."""
    if not path or not os.path.exists(str(path)):
        refuse(f"--strata {path} does not exist (raw/2026-10-25-kind/window_strata_eval141.npz is built by mk_window_strata.py from the v9 release)")
    z = np.load(str(path), allow_pickle=False)
    need = ("stratum", "curve", "turn_is_turn", "has", "eid", "t", "provenance")
    miss = [k for k in need if k not in z.files]
    if miss:
        refuse(f"the strata file {path} is missing {miss}")
    s, cu, ti, has = (np.asarray(z[k]) for k in ("stratum", "curve", "turn_is_turn", "has"))
    n = int(np.asarray(eid).size)
    if not all(x.shape == (n,) for x in (s, cu, ti, has, z["eid"], z["t"])):
        refuse(f"the strata file holds {s.shape[0] if s.ndim else '?'} windows, the evals hold {n}")
    if not np.array_equal(np.asarray(z["eid"]), np.asarray(eid)):
        refuse("the strata file is not on the evals' windows (episode index differs)")
    if frame is not None and not np.array_equal(2 * np.asarray(z["t"], np.int64), np.asarray(frame, np.int64)):
        refuse("the strata file is not on the evals' windows (tick differs: frame != 2t)")
    if not np.isin(s, (0, 1, 2)).all():
        refuse("the strata file holds a stratum code outside {0 JUNCTION, 1 CURVE, 2 NO-TURN}")
    if not np.array_equal(s, stratum_codes(cu, ti, has.astype(bool))):
        refuse("the strata file's stratum is not JUNCTION (turn_is_turn = 1) / CURVE (curve = 1) / NO-TURN (the rest) of its own label columns")
    for fb in fbs:
        if fb is None:
            refuse("an oracle eval carries no tac_oracle_fallback: the strata file's T-tick row pairing cannot be checked against the evaluator's")
        if not np.array_equal(has.astype(bool), ~np.asarray(fb, bool)):
            refuse("the strata file's v9-row presence is not the oracle evals' own (has != ~tac_oracle_fallback): the T-tick row pairing is not the evaluator's")
    try:
        prov = json.loads(str(z["provenance"]))
    except ValueError:
        refuse("the strata file's provenance is not json")
    if prov.get("release_md5") != RELEASE_MD5:
        refuse(f"the strata file was built from release {prov.get('release_md5')}, the PREREG's labels are {RELEASE_MD5}")
    return {"stratum": s.astype(np.int8), "has": has.astype(bool), "prov": prov, "path": str(path)}


def stratum_gains(pf, arms, own_key, eid, mask, n_boot):
    """gain = ADE6(own) - ADE6(arm) inside one stratum: the paired episode-cluster bootstrap over the stratum's OWN episodes. Fewer than MIN_STRATUM_EPISODES episodes: the point
    (descriptive mean difference) and the n, NO interval."""
    ne = int(np.unique(eid[mask]).size)
    own = arms[own_key]["ade6"][mask]
    out = {}
    for k, E in arms.items():
        if k == own_key:
            continue
        if ne < MIN_STRATUM_EPISODES:
            out[k] = {"point": float(own.mean() - E["ade6"][mask].mean()), "interval": f"NONE: fewer than {MIN_STRATUM_EPISODES} episodes in the stratum"}
        else:
            r = PGT.paired(pf, own, E["ade6"][mask], eid[mask], n_boot)
            out[k] = {x: r[x] for x in ("delta", "lo", "hi", "point", "separated")}
    return out


def strata_read(pf, R, eid, S, n_boot):
    """Per seed and per stratum: n windows, n episodes, the six single-kind gains and the union gain (paired, within the stratum), and each stratum's CONTRIBUTION to the pooled
    gain (m): n_stratum / N * gain_stratum, which sums over the three strata to the pooled gain exactly. Nothing here is an input of `decide`."""
    s, has = S["stratum"], S["has"]
    out = {"definition": {"JUNCTION": "turn_is_turn = 1 at the T-tick label row (a dominant turn segment in the label band)", "CURVE": "curve = 1 at the T-tick label row",
                          "NO-TURN": "every other window (includes the windows with no v9 row, where the oracle falls back to T's own code and the gain is exactly 0)"},
           "decides": "nothing: the committed outcome table and the refusals are unchanged",
           "estimator": f"paired episode-cluster bootstrap WITHIN the stratum, B={n_boot}, the stratum's own episodes; fewer than {MIN_STRATUM_EPISODES} episodes: n and no interval",
           "labels": {"file": os.path.basename(S["path"]), "release_md5": S["prov"].get("release_md5"), "definition": S["prov"].get("definition"),
                      "stratum_sha256": hashlib.sha256(np.ascontiguousarray(s).tobytes()).hexdigest()},
           "strata": {}, "read_with": ("If the angle / tte / geom gains sit in CURVE windows, the lever is perception of bends (the encoder: it raises LoRA's prior), not a route input. "
                                       "If they sit in JUNCTION windows, a predicted-goal input stays on the list. (PREREG_V8F_KIND Amendment 1; descriptive, decides nothing.)")}
    n_all = int(s.size)
    for si, name in enumerate(STRATA):
        m = s == si
        ne = int(np.unique(eid[m]).size) if m.any() else 0
        blk = {"n_windows": int(m.sum()), "n_episodes": ne, "n_windows_without_v9_row": int((~has[m]).sum()), "interval": ("computed" if ne >= MIN_STRATUM_EPISODES
               else f"NONE: {ne} episodes < {MIN_STRATUM_EPISODES}"), "seeds": {}}
        for sd in SEEDS:
            if not m.any():
                blk["seeds"][f"s{sd}"] = {"status": "NOT-DEFINED", "reason": "no window in the stratum"}
                continue
            arms = {OWN: R[sd][OWN], UNION: R[sd][UNION], **{f"kind-only-{k}": R[sd][f"kind-only-{k}"] for k in KINDS}}
            g = stratum_gains(pf, arms, OWN, eid, m, n_boot)
            blk["seeds"][f"s{sd}"] = {"gain": {k: g[f"kind-only-{k}"] for k in KINDS}, "gain_union": g[UNION]}
        out["strata"][name] = blk
    contrib = {}
    for sd in SEEDS:
        own = R[sd][OWN]["ade6"]
        contrib[f"s{sd}"] = {}
        for key in [f"kind-only-{k}" for k in KINDS] + [UNION]:
            d = own - R[sd][key]["ade6"]
            contrib[f"s{sd}"]["union" if key == UNION else key[len("kind-only-"):]] = {**{name: float(d[s == si].sum() / n_all) for si, name in enumerate(STRATA)}, "pooled_gain": float(d.mean())}
    out["contribution_to_pooled_gain_m"] = {"seeds": contrib, "note": "n_stratum / N * gain_stratum per stratum; the three sum to pooled_gain (the deciding gain_k) exactly"}
    return out


# ------------------------------------------------------------------------------------------------ the read
def read(a, pf, famfn, DK):
    DK._paths()                                        # <repo>/stack and <repo>/taniteval on sys.path
    from tanitad.models import v8f_mini as M
    part = check_partition(M)
    sha = check_ckpt_sha(a.ckpt_sha)
    R, Bk, eid = load_all(a.eval_dir, M)
    n_win, n_ep = check_counts(eid, a)
    z = PGT.paired(pf, Bk[0]["std"]["ade6"], Bk[0]["std"]["ade6"], eid, a.n_boot)
    if not _exact_zero(z):
        refuse(f"the paired instrument reads {z['delta']} [{z['lo']}, {z['hi']}] for an arm against itself, not +0 [0, 0]")
    F = abs(float(Bk[0]["std"]["ade6"].mean()) - float(Bk[1]["std"]["ade6"].mean()))
    known = known_values(F)
    idc = identity_controls(pf, R, Bk, eid, a.n_boot)
    rows = {f"s{s}": seed_reads(pf, R[s], eid, a.n_boot) for s in SEEDS}
    for s in SEEDS:
        rows[f"s{s}"]["A_interval"] = additivity_interval(R[s], eid, a.n_boot)
    dec = decide({s: rows[f"s{s}"] for s in SEEDS})
    side = speed_side_read({s: rows[f"s{s}"] for s in SEEDS})
    ctx = PGT.build_ctx(a, DK, eid, need_p10=False)
    bes = {f"s{s}": beside(a, DK, ctx, R, s, pf, famfn) for s in SEEDS}
    # Amendment 1, BESIDE: computed from the same per-window ade6, AFTER `dec`, and feeding nothing back into it
    if a.no_strata:
        strata = {"status": "SKIPPED (--no-strata, a TEST input; the chain never passes it)"}
    else:
        S = load_strata(a.strata, eid, ctx["G"]["frame"], [R[s][ALL]["fb"] for s in SEEDS])
        strata = strata_read(pf, R, eid, S, a.n_boot)
    out_name = dec["outcome"]
    if out_name == "ONE KIND CARRIES":
        nxt = (f"ONE KIND CARRIES ({', '.join(dec['carriers'])}): " + "; ".join(f"{k} -> {KIND_ROUTE[k]}" for k in dec["carriers"]) +
               ". Next: a T-side arm aimed at the carrying kind, under its own PREREG. (" + NEXT_TEXT[out_name] + ")")
    else:
        nxt = f"{out_name}: {NEXT_TEXT[out_name]}"
    if out_name in ("DIFFUSE", "INTERACTIVE"):
        nxt += ". CONFOUND: " + CONFOUND
    fmt = lambda f: " / ".join(f(rows[f"s{s}"]) for s in SEEDS)                                              # noqa: E731
    out = dict(schema="v8f-kind-read-v1", prereg="PREREG_V8F_KIND.md", arm=ARM, seeds=list(SEEDS),
               tier="T0 (every oracle arm reads GT): a diagnostic of how O consumes T's code, never a driving-performance claim",
               estimator=f"paired episode-cluster bootstrap, B={a.n_boot} (episodes); answers 'another draw of EPISODES' only (H-ESTIM-SEED-1); this instrument has no training, its seed "
                         f"replicate is the second banked checkpoint",
               n_windows=n_win, n_episodes=n_ep, partition=part, bar_2F_m=BAR_2F, bar_source="PREREG_V8F_KIND.md (committed, not recomputed)", F_measured=F, two_F_measured=2 * F,
               known_values=known, checkpoint_sha256=sha, identity_control=idc, rows=rows, decision=dec, DECISION=out_name, VERDICT=out_name, speed_side_read=side,
               confound=CONFOUND, beside=bes, strata=strata, families_skipped=bool(famfn is None),
               strategic_family="NOT PAUSED: un-paused for v8F-mini (PI ruling 2026-10-17); read from each arm's eval json (beside.*.*.strategic)",
               lambda_definition=f"LAMBDA(arm) = along-track(2 s) of the arm minus that of the full oracle, mean over the {int(ctx['lead'].sum())} LEAD windows of the K = 10 block; "
                                 f"for the all-latcons oracle it is PGT's latcons leak", n_lead_windows_k10=int(ctx["lead"].sum()), next=nxt)
    out["reading"] = (f"PREREG_V8F_KIND: {out_name}" + (f" ({', '.join(dec['carriers'])})" if dec["carriers"] else "") + "; gain_k (m, M2b s0 / s1): " +
                      ", ".join(f"{k} " + fmt(lambda r, k=k: f"{r['gain'][k]['point']:+.3f}{'' if r['gain'][k]['separated'] else ' ns'}") for k in KINDS) +
                      "; union " + fmt(lambda r: f"{r['gain_union']['point']:+.3f}") + "; A " +
                      fmt(lambda r: "undefined" if r["A"] is None else f"{r['A']:.2f}") + f" (bar {ADD_BAR}); 2F {BAR_2F}; SPEED side-read {side['read']}.")
    return out


# ------------------------------------------------------------------------------------------------ --check-inputs
def check_inputs(a, pf, famfn, DK, WC):
    """The banked M2b side alone, before any box-2 run: the PREREG's kind table is a partition and the model agrees; PGT's banked-side check (the runs and evals of BOTH seeds load,
    the K = 10 lead context builds, F / LAMBDA / S5 known values reproduce, the paired instrument reads exactly 0 on M2b against itself including every family cell);
    the two identity-control targets (the standard eval and the -tacorc eval) exist on one window list of the committed size."""
    DK._paths()                                        # <repo>/stack and <repo>/taniteval on sys.path
    from tanitad.models import v8f_mini as M
    part = check_partition(M)
    banked = PGT.check_inputs(a, DK, WC, pf, famfn)
    Bk, eid = {}, None
    for s in SEEDS:
        Bk[s] = {"std": load_eval(a.eval_dir, f"{ARM}_s{s}"), "orc": load_eval(a.eval_dir, f"{ARM}_s{s}-tacorc")}
        for E in Bk[s].values():
            if eid is None:
                eid = E["eid"]
            elif not np.array_equal(E["eid"], eid):
                refuse(f"eval_{E['name']}.npz is not on the same eval grid as the other banked evals (eid differs)")
    n_win, n_ep = check_counts(eid, a)
    F = abs(float(Bk[0]["std"]["ade6"].mean()) - float(Bk[1]["std"]["ade6"].mean()))
    known_values(F)
    strata = {"status": "SKIPPED (--no-strata, a TEST input)"}
    if not a.no_strata:
        S = load_strata(a.strata, eid, DK.ground_truth(a)["frame"], [Bk[s]["orc"]["fb"] for s in SEEDS])
        strata = {"file": os.path.basename(S["path"]), "n_windows": int(S["stratum"].size), "release_md5": S["prov"].get("release_md5"),
                  "counts": {nm: int((S["stratum"] == i).sum()) for i, nm in enumerate(STRATA)},
                  "episodes": {nm: int(np.unique(eid[S["stratum"] == i]).size) for i, nm in enumerate(STRATA)},
                  "n_without_v9_row": int((~S["has"]).sum()), "stratum_sha256": hashlib.sha256(np.ascontiguousarray(S["stratum"]).tobytes()).hexdigest()}
    return {"base": ARM, "partition": part, "n_windows": n_win, "n_episodes": n_ep, "F": F, "two_F_bar": BAR_2F, "banked": banked,
            "banked_ade6": {f"s{s}": {"own": float(Bk[s]["std"]["ade6"].mean()), "all": float(Bk[s]["orc"]["ade6"].mean())} for s in SEEDS},
            "strata": strata, "expected_evals_per_seed": len(arm_suffixes()), "expected_evals": len(arm_suffixes()) * len(SEEDS), "expected_suffixes": arm_suffixes(),
            "banked_ckpt_sha256": BANKED_CKPT_SHA}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-inputs", action="store_true", help="validate the banked M2b side alone (and the identity control) and exit")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--runs-dir", help="the banked M2b runs (summary / argv / selfcheck): required by --check-inputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-sha", help="the chain's json {M2b_s0: sha256, M2b_s1: sha256} measured on box 2: required by the read")
    ap.add_argument("--n-boot", type=int, default=B, help="B of the deciding contrasts (default 2,000)")
    ap.add_argument("--dk-n-boot", type=int, default=B, help="B of the descriptive distance-keeping blocks (read_dk)")
    ap.add_argument("--no-families", action="store_true", help="TEST INPUT: skip the paired LON / LAT family blocks (the chain never passes it)")
    ap.add_argument("--strata", default=str(STRATA_DEFAULT), help="Amendment 1: per-window stratum labels of the eval-141 grid (default: the repo's, built by mk_window_strata.py)")
    ap.add_argument("--no-strata", action="store_true", help="TEST INPUT: skip the Amendment 1 stratum read (the chain never passes it)")
    ap.add_argument("--n-windows", type=int, default=EVAL_WINDOWS, help="the committed eval grid's window count")
    ap.add_argument("--n-episodes", type=int, default=EVAL_EPISODES, help="the committed eval grid's episode count")
    ap.add_argument("--lead-block", help="the K = 10 B1 lead block (default: the banked 2026-09-02 block)")
    ap.add_argument("--p10-pred", default=None, help="P10-yaw-4512 eval predictions (default: the repo's); needed by --check-inputs only")
    ap.add_argument("--eval-grid", default=str(RAW / "eval_grid_windows.json"))
    ap.add_argument("--eval-kin", default=str(RAW / "eval141_kin.npz"))
    ap.add_argument("--eval-join", default=str(RAW / "join_eval141.npz"))
    ap.add_argument("--gt-npz", help="TEST INPUT: npz with gt, eid, clip, frame, v0, dt in place of the rebuild")
    a = ap.parse_args(argv)
    try:
        DK, WC = _modules()
        a.p10_pred = a.p10_pred or str(DK.P10_DEFAULT)
        from taniteval.ci import paired_episode_cluster_bootstrap as pairfn
        famfn = None if a.no_families else PGT._famfn(a)
        if a.check_inputs:
            if not a.runs_dir:
                ap.error("--runs-dir is required with --check-inputs")
            res = check_inputs(a, pairfn, famfn, DK, WC)
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(DK._clean(res), f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            b = res["banked"]
            print(f"KIND INPUTS OK: base {ARM}; F {res['F']:.4f} (2F bar {BAR_2F}); LAMBDA(M2b all-latcons) {b['lambda_m2b'][0]:+.4f} / {b['lambda_m2b'][1]:+.4f}; "
                  f"{res['n_windows']} windows over {res['n_episodes']} episodes; partition 26 = " + " + ".join(str(len(KIND_CELLS[k])) for k in KINDS) +
                  f"; expects {res['expected_evals']} evals ({res['expected_evals_per_seed']} per seed); identity control exactly +0 [0, 0]; strata " +
                  ("not read" if "counts" not in res["strata"] else ", ".join(f"{nm} {res['strata']['counts'][nm]} windows/{res['strata']['episodes'][nm]} episodes" for nm in STRATA)))
            return 0
        res = read(a, pairfn, famfn, DK)
    except (OSError, KeyError, ValueError) as e:
        print(f"KIND READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:
        return int(e.code or 2)
    except Exception as e:                           # an inherited instrument's OWN Refusal (v8f_dk_read.Refusal): still a refusal, rc 2, never a traceback
        if type(e).__name__ != "Refusal":
            raise
        print(f"KIND READ REFUSED: {e}", file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(DK._clean(res), f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("reading", "next")}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
