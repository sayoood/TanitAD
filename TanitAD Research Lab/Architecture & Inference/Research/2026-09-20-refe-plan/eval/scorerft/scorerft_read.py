#!/usr/bin/env python3
"""SCORER-FT T-vs-C read: per-token arrays -> shared-draw paired log-cluster bootstrap -> the card's clauses P1 / G1 / P2 / P3 and outcomes O1-O6.

Implements PREREG_SCORER_FT.md (FROZEN 2026-11-20, sha256 adc1b2ae8d3cc9b3dfed00a20a3c39197428cda8492be34af3bb9ed24010ec91) sections 5-10.
It decides nothing the card did not decide; every place where the card is silent is listed under `card_questions` in the output.
Built on eval/p5c/p5c_read.py (Boot, per_token, contrast, entry_adverse) and eval/field/field_read.py (ctrl_set, ctrl_seam, seam_geometry, strategic_block):
IMPORTED, never forked or edited. The selection-quality instruments (head_auc.py:53-56 Spearman, regret_decomp.py split) are re-stated here as per-token
arrays so that they ride the SAME draw matrix as every other estimand (tests assert they equal the originals).

ARMS     C = refe_scorerft_c (seed 0, --rank-w 0), T = refe_scorerft_t (seed 0), Cs / Ts = the seed-1 pair (Stage 2, optional).
INPUT    transp_score rows (p5c_read_prep.py convert + score) carrying the arms; per arm: seam + proposals dump + harness CSV + 64-proposal table
         (p5c_prop_score.py) + the pod's argv file; C's step-0 digest; T's seam re-run on 200 tokens (CTRL-DET). Scores are printed x 100.
ESTIMATOR paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, ONE draw matrix for every arm and contrast.
         It answers "would another draw of LOGS say this?" ONLY. Another TRAINING run = the seed pair (P2, P3). Another INFERENCE run = none (deterministic argmax; CTRL-DET).
TIER     T1: NAVSIM v1 PDMS of the executed LQR-tracked plan, open-loop pseudo-simulation. Not closed loop, not T0.
CONTROLS CTRL-T, CTRL-SET, CTRL-ARGV, CTRL-DET, CTRL-W0 (+ CTRL-SEAM, CTRL-CSV, CTRL-ID of the instrument): any FAIL or any missing one in a real read = INVALID / REFUSED.
Run with --dry-run "<text>" on stand-in arms: every output is stamped DRY RUN - NOT A SCORER-FT VERDICT and carries no SCORER-FT claim.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (os.environ.get("SCORERFT_READ_P5C_DIR", ""), os.path.join(HERE, "..", "p5c")):
    if _d and os.path.exists(os.path.join(_d, "p5c_read.py")):
        sys.path.insert(0, os.path.abspath(_d))
        break
for _d in (os.environ.get("SCORERFT_READ_FIELD_DIR", ""), os.path.join(HERE, "..", "field")):
    if _d and os.path.exists(os.path.join(_d, "field_read.py")):
        sys.path.insert(0, os.path.abspath(_d))
        break
import p5c_read as PR                                     # noqa: E402
from p5c_read import Boot, Refused, SEED, NBOOT, SUBS, G1_SUBS, METRICS, SCORERS   # noqa: E402
import field_read as FR                                   # noqa: E402

PREREG_SHA256 = "adc1b2ae8d3cc9b3dfed00a20a3c39197428cda8492be34af3bb9ed24010ec91"
DRY_STAMP = "DRY RUN — NOT A SCORER-FT VERDICT"
TIER = "T1 (NAVSIM v1 PDMS, open-loop pseudo-simulation of the executed LQR-tracked plan; not closed loop, not T0)"
ANSWERS = ("another draw of LOGS only; another TRAINING run = the seed pair (Ts - Cs, F = |Cs - C|; H-ESTIM-SEED-1); "
           "another INFERENCE run = none (deterministic argmax, CTRL-DET)")
N_NAVTEST, L_NAVTEST = 12146, 136
GOAL = 94.55
M1_MOVED_DELTA = 0.10
CSV_BAR = 1e-9
DET_N_MIN = 200
STEPS_EXPECTED = 1000
ONE_SEED = "one seed per arm: training variance was NOT measured for this arm"
LOOK = "look 1 of H-SCORERFT-1 (Stage 1: T, C)"
LOOK2 = "look 2 of H-SCORERFT-1 (Stage 2: Ts, Cs)"
MULTIPLICITY = "multiplicity across cards is not corrected"
HEADS = {"nc": 0, "dac": 1, "ep": 2, "ttc": 3, "c": 4}                     # head_auc.py:21 / agg32 order of the logits
CSV_SUB = (("nc", "no_at_fault_collisions"), ("dac", "drivable_area_compliance"), ("ddc", "driving_direction_compliance"),
           ("ep", "ego_progress"), ("ttc", "time_to_collision_within_bound"), ("c", "comfort"), ("pdms", "score"))
NEXT_ARM = {
    "O1": "SCORER-FT-2: add `ttc` to the heads (the 1.38 term), from T's final, same design, own card (~14.2 L40-h + control); `L_rank` carries into the next generator arm (TEACHER-N7 / comfort)",
    "O2": "one more seed pair (T s2, C s2; ~28.3 L40-h): seeds, not windows. Still swamped => O4's next",
    "O5": "drop the head behind the adverse sub-score: EP adverse => `gate` only; NC / DAC / TTC adverse => `ep` only. C is reused (same init / seed / steps), ~14.2 L40-h",
    "O6": "`L_rank` on a second scorer pass over DETACHED `visual_ctx` (rank stops at the scorer decoder; BCE path unchanged), new flag `--rank-detach-ctx`, own card; C reused, ~14.2 L40-h",
    "O3": "decision-level term: pick-anchored pairwise on the `navsim_v1` aggregate (every j with PDMS_j > PDMS_pick vs the current pick, weight PDMS_j - PDMS_pick), a new head key `pick` in the same `rank_loss.py`; C reused, ~14.2 L40-h",
    "O4": "the selection line closes. Generation TEACHER-N7 (`PREREG_P5C_NEXT` section 5 rank 3), OR explicit proposal-geometry inputs to the EP head (architecture: PI). Both are named blockers on the PI and the L40",
}
OUTCOME_VERDICT = {
    "O1": "ADOPT", "O2": "SWAMPED, NOT PROVEN", "O5": "ADVERSE, NOT ADOPTED", "O6": "FAIL: the trunk path hurt generation",
    "O3": "FAIL: the heads rank, the pick does not", "O4": "FAIL: the representation path does not carry the margin at this dose",
}
ALLOWED_TC = {"--rank-w", "--out", "--rank-heads", "--rank-ep-gap", "--rank-gate-topk"}      # card section 3: argv(T) - argv(C)
ALLOWED_SEED = {"--seed", "--out"}                                                           # card section 3: argv(Xs) - argv(X)


# ------------------------------------------------------------------------------------------------------ small helpers
def ci_s(x):
    return "%.4f [%.4f, %.4f]" % (x["point"], x["lo"], x["hi"])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def wdraws(bt: Boot, v, w) -> np.ndarray:
    """ratio of sums of v over the drawn logs on the tokens with weight w (0/1): the SAME draw matrix as Boot.draws, for statistics defined on a subset."""
    v, w = np.asarray(v, float), np.asarray(w, float)
    s = np.bincount(bt.li, weights=np.where(w > 0, v, 0.0) * w, minlength=bt.L)
    n = np.bincount(bt.li, weights=w, minlength=bt.L)
    den = n[bt.D].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s[bt.D].sum(1) / den


def wci(bt: Boot, v, w) -> dict:
    w = np.asarray(w, float)
    m = wdraws(bt, v, w)
    k = int((w > 0).sum())
    if k == 0:
        return {"status": "NOT COMPUTABLE", "reason": "no token defines this statistic", "n_tokens": 0}
    pt = float(np.sum(np.where(w > 0, np.asarray(v, float), 0.0) * w) / w.sum())
    return {"point": pt, "lo": float(np.nanpercentile(m, 2.5)), "hi": float(np.nanpercentile(m, 97.5)), "n_tokens": k}


def wdiff(bt: Boot, va, wa, vb, wb) -> dict:
    """a - b where each side is a ratio of sums over ITS OWN defined tokens, on the shared draws (paired per draw)."""
    wa, wb = np.asarray(wa, float), np.asarray(wb, float)
    if (wa > 0).sum() == 0 or (wb > 0).sum() == 0:
        return {"status": "NOT COMPUTABLE", "reason": "a side defines no token", "n_a": int((wa > 0).sum()), "n_b": int((wb > 0).sum())}
    d = wdraws(bt, va, wa) - wdraws(bt, vb, wb)
    pa = float(np.sum(np.where(wa > 0, va, 0.0) * wa) / wa.sum())
    pb = float(np.sum(np.where(wb > 0, vb, 0.0) * wb) / wb.sum())
    return {"point": pa - pb, "lo": float(np.nanpercentile(d, 2.5)), "hi": float(np.nanpercentile(d, 97.5)), "n_a": int((wa > 0).sum()), "n_b": int((wb > 0).sum())}


# ------------------------------------------------------------------------------------------------------ argv (CTRL-ARGV)
def parse_argv(x) -> dict:
    """argv as a list / a whitespace string / a path to a JSON list or a text file -> {flag: tuple(values)}; `--f=v` is split. Order is not significant."""
    if isinstance(x, str) and os.path.exists(x):
        txt = open(x, encoding="utf-8").read()
        try:
            j = json.loads(txt)
            x = j["argv"] if isinstance(j, dict) and "argv" in j else j
        except ValueError:
            x = txt.split()
    if isinstance(x, str):
        x = x.split()
    toks = []
    for t in x:
        t = str(t)
        if t.startswith("--") and "=" in t:
            a, b = t.split("=", 1)
            toks += [a, b]
        else:
            toks.append(t)
    out, cur = {}, None
    for t in toks:
        if t.startswith("--"):
            cur = t
            out.setdefault(cur, ())
        elif cur is not None:
            out[cur] = out[cur] + (t,)
    return out


def _f(v):
    try:
        return float(v[0]) if len(v) == 1 else None
    except ValueError:
        return None


def _diff_keys(a: dict, b: dict) -> set:
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def ctrl_argv(argv: dict, steps: int = STEPS_EXPECTED) -> dict:
    """argv = {"C": ..., "T": ..., optional "Cs", "Ts"} (parsed). The difference from the card's section 3 must be EXACTLY: T vs C = {--rank-w, --rank-heads ep,gate,
    --rank-ep-gap 0.02, --rank-gate-topk 8, --out}; Xs vs X = {--seed 1, --out}. C names --rank-w 0.0; every arm has --init-from (one value) and --steps == 1000."""
    fails, detail = [], {}
    for need in ("C", "T"):
        if need not in argv:
            return {"control": "CTRL-ARGV", "pass": False, "fails": [f"argv of arm {need} not supplied"], "detail": {}}
    C, T = argv["C"], argv["T"]
    d = _diff_keys(C, T)
    detail["T_minus_C_flags"] = sorted(d)
    if d - ALLOWED_TC:
        fails.append(f"T vs C differ in flags outside the card: {sorted(d - ALLOWED_TC)}")
    if "--rank-w" not in C or _f(C["--rank-w"]) != 0.0:
        fails.append(f"C must name --rank-w 0.0 (got {C.get('--rank-w')})")
    if "--rank-w" not in T or not (_f(T["--rank-w"]) or 0.0) > 0.0:
        fails.append(f"T must name --rank-w w* > 0 (got {T.get('--rank-w')})")
    for k in ("--rank-heads", "--rank-ep-gap", "--rank-gate-topk"):
        if k in C:
            fails.append(f"C carries {k}: only T may")
    if T.get("--rank-heads") not in (("ep,gate",), ("ep", "gate")):
        fails.append(f"T --rank-heads must be ep,gate (got {T.get('--rank-heads')})")
    if "--rank-ep-gap" not in T or _f(T["--rank-ep-gap"]) != 0.02:
        fails.append(f"T --rank-ep-gap must be 0.02 (got {T.get('--rank-ep-gap')})")
    if "--rank-gate-topk" not in T or _f(T["--rank-gate-topk"]) != 8.0:
        fails.append(f"T --rank-gate-topk must be 8 (got {T.get('--rank-gate-topk')})")
    for nm, a in argv.items():
        if "--init-from" not in a or len(a["--init-from"]) != 1:
            fails.append(f"{nm}: --init-from missing or not a single value")
        if _f(a.get("--steps", ())) != float(steps):
            fails.append(f"{nm}: --steps must be {steps} (got {a.get('--steps')})")
        if "--out" not in a:
            fails.append(f"{nm}: no own --out")
    outs = [a.get("--out") for a in argv.values()]
    if len(set(outs)) != len(outs):
        fails.append("two arms share one --out")
    for base, seeded in (("C", "Cs"), ("T", "Ts")):
        if seeded in argv:
            b, s = argv[base], argv[seeded]
            dd = _diff_keys(b, s)
            detail[f"{seeded}_minus_{base}_flags"] = sorted(dd)
            if dd - ALLOWED_SEED:
                fails.append(f"{seeded} vs {base} differ in flags outside the card: {sorted(dd - ALLOWED_SEED)}")
            if _f(s.get("--seed", ())) != 1.0:
                fails.append(f"{seeded}: --seed must be 1 (got {s.get('--seed')})")
            if base in argv and (_f(b.get("--seed", ("0",))) or 0.0) != 0.0:
                fails.append(f"{base}: seed must be 0 (got {b.get('--seed')})")
    if ("Cs" in argv) != ("Ts" in argv):
        fails.append("Cs and Ts must be supplied together")
    if "Cs" in argv and "Ts" in argv:
        dd = _diff_keys(argv["Cs"], argv["Ts"])
        if dd - ALLOWED_TC:
            fails.append(f"Ts vs Cs differ in flags outside the card: {sorted(dd - ALLOWED_TC)}")
    return {"control": "CTRL-ARGV", "pass": not fails, "fails": fails, "detail": detail, "rule": "card section 3, exactly (no extra flag, no missing flag)"}


# ------------------------------------------------------------------------------------------------------ CTRL-W0, CTRL-DET
HEX64 = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")


def extract_digest(x: str) -> str | None:
    """a 64-hex digest from a path (JSON or text) or from the string itself; the FIRST one found."""
    txt = open(x, encoding="utf-8").read() if os.path.exists(x) else x
    m = HEX64.search(txt)
    return m.group(0) if m else None


def ctrl_w0(c_digest: str | None, ref: str | None, others: dict | None = None) -> dict:
    """C's step-0 digest == C1's (the sha256 of C1's final, SCORERFT_BASE_SHA256). `others` = optional {arm: digest} of T / Cs / Ts, reported with the same rule."""
    fails = []
    d, r = extract_digest(c_digest) if c_digest else None, extract_digest(ref) if ref else None
    if not d:
        fails.append("no 64-hex step-0 digest of C found")
    if not r:
        fails.append("no 64-hex C1 reference digest found")
    if d and r and d != r:
        fails.append(f"C step-0 digest {d[:12]} != C1 {r[:12]}")
    oth = {}
    for k, v in (others or {}).items():
        o = extract_digest(v)
        oth[k] = {"digest": o, "equals_C1": bool(o and r and o == r)}
        if not (o and r and o == r):
            fails.append(f"{k} step-0 digest {str(o)[:12]} != C1")
    return {"control": "CTRL-W0", "pass": not fails, "fails": fails, "c_digest": d, "c1_digest": r, "others": oth,
            "note": "the step-0 artifact format is not registered: the first 64-hex string in the supplied file is read (card question Q-S2)"}


def ctrl_det(seam_full: str, seam_rerun: str, n_min: int = DET_N_MIN) -> dict:
    """T's seam re-run on >= 200 tokens is IDENTICAL: every rerun token is in the full seam and its 8 poses are equal bit for bit."""
    a = np.load(seam_full, allow_pickle=False)
    b = np.load(seam_rerun, allow_pickle=False)
    ta, tb = [str(t) for t in a["token"]], [str(t) for t in b["token"]]
    ia = {t: i for i, t in enumerate(ta)}
    miss = [t for t in tb if t not in ia]
    fails = []
    if len(tb) < n_min:
        fails.append(f"re-run holds {len(tb)} tokens < {n_min}")
    if len(set(tb)) != len(tb):
        fails.append("re-run repeats a token")
    if miss:
        fails.append(f"{len(miss)} re-run tokens are absent from the full seam (first {miss[0]})")
    mx, nbit = None, None
    if not miss and tb:
        pa = np.asarray(a["poses"], float)[[ia[t] for t in tb]]
        pb = np.asarray(b["poses"], float)
        if pa.shape != pb.shape:
            fails.append(f"pose shapes {pa.shape} != {pb.shape}")
        else:
            nbit = int((pa != pb).any((1, 2)).sum())
            mx = float(np.abs(pa - pb).max())
            if nbit:
                fails.append(f"{nbit} tokens differ (max |d| {mx})")
    return {"control": "CTRL-DET", "pass": not fails, "fails": fails, "n_tokens": len(tb), "n_bar": n_min, "tokens_not_bit_identical": nbit, "max_abs_pose_diff": mx,
            "seam_full": os.path.abspath(seam_full), "seam_rerun": os.path.abspath(seam_rerun), "seam_full_sha256": FR.sha256_file(seam_full), "seam_rerun_sha256": FR.sha256_file(seam_rerun)}


# ------------------------------------------------------------------------------------------------------ tables, dumps, CTRL-T
TAB_KEYS = ("nc", "dac", "ddc", "ttc", "c", "ep", "pdms")


def load_table_arrays(paths, tokens) -> dict:
    """64-proposal table rows (p5c_prop_score.py) -> arrays in `tokens` order. Streams the file: never holds the parsed rows."""
    want = {t: i for i, t in enumerate(tokens)}
    n = len(tokens)
    out = {f"{s}_{k}": np.full((n, 64), np.nan) for s in ("tr", "tp") for k in TAB_KEYS}
    out["pick"] = np.full(n, -1, int)
    out["tp20"] = {k: np.full(n, np.nan) for k in TAB_KEYS}
    seen = np.zeros(n, bool)
    for fp in paths:
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                if "error" in r or r.get("token") not in want:
                    continue
                i = want[r["token"]]
                for s in ("tr", "tp"):
                    for k in TAB_KEYS:
                        out[f"{s}_{k}"][i] = r[s][k]
                out["pick"][i] = int(r["pick"])
                for k in TAB_KEYS:
                    out["tp20"][k][i] = r["tp20_pick"][k]
                seen[i] = True
    if not seen.all():
        miss = [tokens[i] for i in np.where(~seen)[0]]
        raise Refused(f"proposal table lacks {len(miss)} of {n} tokens (first {miss[0]})")
    for k, v in out.items():
        if k not in ("pick", "tp20") and not np.isfinite(v).all():
            raise Refused(f"proposal table: non-finite {k}")
    return out


def load_dump(dump_path: str, tokens) -> dict:
    d = np.load(dump_path, allow_pickle=False)
    di = {str(t): i for i, t in enumerate(d["token"])}
    miss = [t for t in tokens if t not in di]
    if miss:
        raise Refused(f"dump {dump_path} lacks {len(miss)} tokens (first {miss[0]})")
    r = np.array([di[t] for t in tokens])
    lg = np.asarray(d["logits"])[r].astype(np.float32)
    if lg.ndim != 3 or lg.shape[1] != 64 or lg.shape[2] < 5:
        raise Refused(f"dump {dump_path}: logits shape {lg.shape}")
    return {"logits": lg, "pick": np.asarray(d["pick"])[r]}


def read_csv(path: str) -> dict:
    out = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if r["valid"] == "True" and len(r["token"]) == 16:
            out[r["token"]] = {k: float(r[c]) for k, c in CSV_SUB}
    return out


def ctrl_t(T: dict, csv_path: str, tokens, dump_pick, arm: str, bar: float = CSV_BAR) -> dict:
    """the table's tracked row AT THE PICK == the harness CSV row to 1e-9 on every token (all 7 metrics), and the table's pick == the dump's pick."""
    C = read_csv(csv_path)
    miss = [t for t in tokens if t not in C]
    if miss:
        return {"control": "CTRL-T", "arm": arm, "pass": False, "fails": [f"harness CSV lacks {len(miss)} tokens (first {miss[0]})"]}
    ar = np.arange(len(tokens))
    pick = T["pick"]
    mx = {k: float(max(abs(T[f"tr_{k}"][i, pick[i]] - C[t][k]) for i, t in enumerate(tokens))) for k, _ in CSV_SUB}
    bad = [t for i, t in enumerate(tokens) if any(abs(T[f"tr_{k}"][i, pick[i]] - C[t][k]) > bar for k, _ in CSV_SUB)]
    same_pick = bool(np.array_equal(pick, np.asarray(dump_pick)))
    fails = []
    if bad:
        fails.append(f"{len(bad)} tokens over {bar} (first {bad[0]})")
    if not same_pick:
        fails.append("table pick != dump pick")
    _ = ar
    return {"control": "CTRL-T", "arm": arm, "pass": not fails, "fails": fails, "n_tokens": len(tokens), "n_over_bar": len(bad), "bar": bar, "max_abs_diff": mx,
            "table_pick_equals_dump_pick": same_pick, "csv": os.path.abspath(csv_path), "csv_sha256": FR.sha256_file(csv_path)}


# ------------------------------------------------------------------------------------------------------ the instruments as per-token arrays
def spearman(a, b):
    """head_auc.py:34-37, verbatim (ordinal ranks via argsort(argsort), NaN when < 3 points or a constant)."""
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def auc_within(s, y):
    """head_auc.py:28-33, verbatim."""
    pos, neg = s[y > 0.5], s[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def mech_arrays(T: dict, D: dict) -> dict:
    """per-token arrays of one arm from its 64-proposal table T and proposals dump D (logits, pick). Definitions: regret_decomp.py (regret split, zeroed pick,
    TTC-0 pick with a TTC-1 alternative), head_auc.py (within-token AUC, EP Spearman over safe proposals, confidently-wrong share). weights w_* = 1 where defined."""
    n = len(T["pick"])
    ar = np.arange(n)
    pk = T["pick"]
    tr = {k: T[f"tr_{k}"] for k in TAB_KEYS}
    P = 1.0 / (1.0 + np.exp(-np.asarray(D["logits"], np.float32)))          # float32, as head_auc.py:62
    pdm = tr["pdms"] * 100.0
    oracle = pdm.max(1)
    pick_pdms = pdm[ar, pk]
    regret = oracle - pick_pdms
    o = tr["pdms"].argmax(1)
    z = (tr["nc"][ar, pk] < .5) | (tr["dac"][ar, pk] < .5)
    lin = ~z & (tr["nc"][ar, o] > .5) & (tr["dac"][ar, o] > .5) & (tr["nc"][ar, pk] > .5) & (tr["dac"][ar, pk] > .5)     # regret_decomp.py:11 (NC = 0.5 is not a gate pass)
    comp = {k: np.where(lin, w * (tr[k][ar, o] - tr[k][ar, pk]) / 12 * 100, 0.0) for k, w in (("ep", 5), ("ttc", 5), ("c", 2))}
    out = {"n": n, "oracle": oracle, "pick": pick_pdms, "regret": regret,
           "regret_gate": np.where(z, regret, 0.0), "regret_ep": comp["ep"], "regret_ttc": comp["ttc"], "regret_c": comp["c"],
           "regret_other": np.where(~z & ~lin, regret, 0.0),
           "top1": 100.0 * (np.abs(tr["pdms"][ar, pk] - tr["pdms"].max(1)) <= 1e-9), "zeroed": 100.0 * z,
           "ttc0_alt": 100.0 * (~z & (tr["ttc"][ar, pk] < .5) & (tr["ttc"].max(1) > .5)), "random": pdm.mean(1)}      # rates in percent of tokens
    for h in ("nc", "dac", "ttc"):
        v = np.array([auc_within(P[i, :, HEADS[h]], (tr[h][i] > 0.5).astype(float)) for i in range(n)])
        out[f"auc_{h}"], out[f"w_auc_{h}"] = np.nan_to_num(v), (~np.isnan(v)).astype(float)
    ok = (tr["nc"] > .5) & (tr["dac"] > .5)
    v = np.array([spearman(P[i, ok[i], HEADS["ep"]], tr["ep"][i, ok[i]]) if ok[i].sum() >= 3 else np.nan for i in range(n)])
    out["ep_spearman"], out["w_ep_spearman"] = np.nan_to_num(v), (~np.isnan(v)).astype(float)
    resc = z & (tr["pdms"].max(1) > 0)
    for h in ("nc", "dac"):
        val, w = np.zeros(n), np.zeros(n)
        for i in np.where(resc & (tr[h][ar, pk] < .5))[0]:
            good = tr[h][i] > .5
            val[i] = 100.0 * float(P[i, pk[i], HEADS[h]] > np.median(P[i, good, HEADS[h]]))
            w[i] = 1.0
        out[f"conf_wrong_{h}"], out[f"w_conf_wrong_{h}"] = val, w
    out["zeroed_by_dac"] = 100.0 * (tr["dac"][ar, pk] < .5)
    out["zeroed_by_nc"] = 100.0 * (tr["nc"][ar, pk] < .5)
    out["comfort20"] = T["tp20"]["c"] * 100.0
    out["comfort8"] = T["tp_c"][ar, pk] * 100.0
    return out


FULL = ("oracle", "pick", "regret", "regret_gate", "regret_ep", "regret_ttc", "regret_c", "regret_other", "top1", "zeroed", "ttc0_alt", "random", "zeroed_by_dac", "zeroed_by_nc")
SUBSET = ("auc_nc", "auc_dac", "auc_ttc", "ep_spearman", "conf_wrong_nc", "conf_wrong_dac")


def mech_levels(bt, A) -> dict:
    n1 = np.ones(A["n"])
    out = {k: wci(bt, A[k], n1) for k in FULL}
    for k in SUBSET:
        out[k] = wci(bt, A[k], A[f"w_{k}"])
    return out


def mech_contrast(bt, A, B) -> dict:
    """A - B on the shared draws: M1 (EP Spearman, on the tokens defined in BOTH), M2, M3 (oracle - pick and its split), M4 (oracle-of-64), and the rest of the selection quality."""
    n1 = np.ones(A["n"])
    out = {k: wci(bt, A[k] - B[k], n1) for k in FULL}
    both = lambda k: A[f"w_{k}"] * B[f"w_{k}"]      # noqa: E731
    out["ep_spearman_both_defined"] = wci(bt, A["ep_spearman"] - B["ep_spearman"], both("ep_spearman"))
    for k in SUBSET:
        out[k] = wdiff(bt, A[k], A[f"w_{k}"], B[k], B[f"w_{k}"])
    return out


def mech_decide(con: dict) -> dict:
    """card section 7. M1 moved iff delta >= +0.10 AND CI-lo > 0; M3 moved iff CI-hi < 0 (oracle - pick); M4 damaged iff CI-hi < 0 (oracle-of-64)."""
    m1 = con["ep_spearman_both_defined"]
    m3, m4 = con["regret"], con["oracle"]
    m1_moved = bool(m1.get("point") is not None and m1["point"] >= M1_MOVED_DELTA and m1["lo"] > 0)
    m3_moved = bool(m3["hi"] < 0)
    m4_damaged = bool(m4["hi"] < 0)
    return {"M1_ep_spearman": {"delta": m1, "moved": m1_moved, "rule": f"delta >= +{M1_MOVED_DELTA} and CI-lo > 0 (tokens with >= 3 safe proposals in BOTH arms)"},
            "M2_boundary": {"zeroed_pick_rate_pp": con["zeroed"], "zeroed_by_dac": con["zeroed_by_dac"], "zeroed_by_nc": con["zeroed_by_nc"],
                            "confidently_wrong_dac_share": con["conf_wrong_dac"], "confidently_wrong_nc_share": con["conf_wrong_nc"], "reported_only": True},
            "M3_regret": {"oracle_minus_pick": m3, "gate": con["regret_gate"], "ep": con["regret_ep"], "ttc": con["regret_ttc"], "c": con["regret_c"], "other": con["regret_other"],
                          "moved": m3_moved, "rule": "CI-hi < 0 of T - C regret (oracle - pick)"},
            "M4_generation_guard": {"oracle_of_64": m4, "damaged": m4_damaged, "rule": "CI-hi < 0 of T - C oracle-of-64"},
            "mechanism_moved": bool(m1_moved or m3_moved), "m4_damaged": m4_damaged}


def selection_qualifiers(con: dict) -> list:
    """separated adverse selection-quality cells (T worse than C): stated in the verdict line as a qualifier, never a clause."""
    adv = []
    for k, bad_if in (("regret", "lo>0"), ("zeroed", "lo>0"), ("oracle", "hi<0"), ("top1", "hi<0"), ("pick", "hi<0"), ("ep_spearman", "hi<0")):
        c = con[k]
        if "point" in c and ((bad_if == "lo>0" and c["lo"] > 0) or (bad_if == "hi<0" and c["hi"] < 0)):
            adv.append(f"TACTICAL.selection.{k}")
    return adv


# ------------------------------------------------------------------------------------------------------ the clauses and the outcomes
def goal_flag(lv: dict) -> dict:
    p, hi = lv["point"], lv["hi"]
    return {"goal": GOAL, "level_tracked": lv, "status": "BELOW GOAL" if p < GOAL else "AT/ABOVE GOAL (level)", "ci_hi_below_goal": bool(hi < GOAL)}


def g1_adverse(con: dict) -> list:
    return [m for m in G1_SUBS if con[m]["tracked"]["hi"] < 0]


def decide(con0: dict, mech: dict | None, controls_ok, rep: dict | None = None) -> dict:
    """card sections 5 and 9. O1, O2, O5, O6, O3, O4 evaluated IN THAT ORDER. `rep` = {"con1": contrast Ts - Cs, "F": |Cs - C| point, "s_p": ...} or None (Stage 1)."""
    d0 = con0["pdms"]["tracked"]
    adv0 = g1_adverse(con0)
    p1, g1 = bool(d0["lo"] > 0), not adv0
    cl = {"P1_lo_gt_0": p1, "P1_delta": d0, "G1": "PASS" if g1 else "FAIL", "G1_adverse": adv0, "P2": "PENDING (Stage 2)", "P3": "PENDING (Stage 2)"}
    if controls_ok is not True:
        return {"verdict": "INVALID", "outcome": None, "wording": "a read control failed or is missing: no verdict; every Delta is descriptive only", "clauses": cl, "next_arm": None}
    if rep is not None:
        d1 = rep["con1"]["pdms"]["tracked"]
        adv1 = g1_adverse(rep["con1"])
        d = (d0["point"] + d1["point"]) / 2.0
        p2 = bool(d1["lo"] > 0 and not adv1)
        p3 = bool(d > 2.0 * rep["F"])
        cl.update(P2="PASS" if p2 else "FAIL", P2_delta=d1, P2_G1_adverse=adv1, P3="PASS" if p3 else "FAIL", P3_d=d, P3_F=rep["F"], P3_2F=2.0 * rep["F"], s_p_beside=rep.get("s_p"))
    mech_ok = mech is not None
    if p1 and g1:
        if rep is None:
            return {"verdict": "STAGE-2-TRIGGERED", "outcome": "PENDING", "wording": "P1 and G1 hold: Stage 2 (Cs, Ts) is triggered; P2 and P3 are PENDING (Stage 2). Never ADOPT on one seed pair. " + ONE_SEED,
                    "clauses": cl, "next_arm": "run Cs then Ts (conditional Stage 2, ~28.3 L40-h, PI go 2026-11-20); the verdict is O1 or O2 after them"}
        o = "O1" if (cl["P2"] == "PASS" and cl["P3"] == "PASS") else "O2"
    elif p1:
        o = "O5"
    elif not mech_ok:
        return {"verdict": "INCOMPLETE", "outcome": None, "wording": "not P1, and no 64-proposal tables: O6 / O3 / O4 cannot be told apart (M4, M1, M3 need them)", "clauses": cl, "next_arm": None}
    elif mech["m4_damaged"]:
        o = "O6"
    elif mech["mechanism_moved"]:
        o = "O3"
    else:
        o = "O4"
    w = f"{o}: {OUTCOME_VERDICT[o]}"
    if o == "O5":
        w += " (adverse: " + ", ".join(adv0) + ")"
    if o in ("O3", "O4", "O5", "O6") and rep is None:
        w += ". " + ONE_SEED + " (a Stage-1 outcome is accepted on one seed, decided before data)"
    if o == "O1":
        w += " as a scorer-training lever on the C1 recipe; it does NOT clear C1's own GUARD-FAIL (TM+C: transparent comfort, the longitudinal speed bias)"
    nxt = NEXT_ARM[o]
    if o == "O5":
        nxt = ("EP adverse => `gate` only. " if "ep" in adv0 else "") + ("NC / DAC / TTC adverse => `ep` only. " if any(m in adv0 for m in ("nc", "dac", "ttc")) else "") + \
              ("C adverse: the card names no head for it (card question Q-S5). " if "c" in adv0 else "") + "C is reused, ~14.2 L40-h"
    return {"verdict": o, "outcome": o, "wording": w, "clauses": cl, "next_arm": nxt}


def pair_rep(bt, P, c, t, cs, ts) -> dict:
    con1 = PR.contrast(bt, P, ts, cs)
    F = abs(float(np.mean(P[cs]["tracked"]["pdms"]) - np.mean(P[c]["tracked"]["pdms"])))
    d0 = float(np.mean(P[t]["tracked"]["pdms"]) - np.mean(P[c]["tracked"]["pdms"]))
    d1 = float(np.mean(P[ts]["tracked"]["pdms"]) - np.mean(P[cs]["tracked"]["pdms"]))
    return {"con1": con1, "F": F, "d0": d0, "d1": d1,
            "s_p": {"value": abs(d0 - d1) / math.sqrt(2.0), "definition": "sample sd of the two seed-level deltas (n = 2), |d0 - d1| / sqrt(2); the card names s_p without defining it (card question Q-S4); reported beside P3, decides nothing"}}


# ------------------------------------------------------------------------------------------------------ CTRL-ID
def ctrl_id(P, logs, MA: dict | None, arm: str, n_boot: int) -> dict:
    """the read of an arm against ITSELF: every Delta exactly 0.0 (PDMS, six sub-scores, both scorers, every mechanism read), no P1."""
    bt = Boot(logs, n_boot)
    con = PR.contrast(bt, P, arm, arm)
    nz = FR.zero_exact({m: {s: con[m][s] for s in ("tracked", "transparent", "TD")} for m in METRICS})
    if MA is not None:
        mc = mech_contrast(bt, MA, MA)
        nz += ["mech." + x for x in FR.zero_exact({k: ({q: v[q] for q in ("point", "lo", "hi")} if "point" in v else {}) for k, v in mc.items()})]
    v = decide(con, mech_decide(mech_contrast(bt, MA, MA)) if MA is not None else None, True, None)
    ok = not nz and v["outcome"] in ("O4", None) and not v["clauses"]["P1_lo_gt_0"]
    return {"control": "CTRL-ID", "arm": arm, "pass": bool(ok), "nonzero_deltas": nz, "verdict": v["verdict"], "P1": v["clauses"]["P1_lo_gt_0"]}


# ------------------------------------------------------------------------------------------------------ families
def relabel_tactical(t: dict, m: dict | None = None) -> dict:
    """p5c_tactical.py keys its arms C1 / C0 / base; here C1 -> T, C0 -> C (base stays: in the chain it is C1's own seam, the init). Drops its STRATEGIC stub."""
    m = m or {"C1": "T", "C0": "C"}

    def rn(x):
        if isinstance(x, dict):
            return {m.get(k, k).replace("C1-", "T-").replace("C0-", "C-").replace("paired_C1_minus_x", "paired_T_minus_x"): rn(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rn(v) for v in x]
        return x
    r = rn({k: v for k, v in t.items() if k != "strategic"})
    r["labelling"] = "arm names relabelled by scorerft_read.relabel_tactical: C1 -> T, C0 -> C (p5c_tactical.py hard-codes C1/C0); its STRATEGIC stub dropped (STRATEGIC is reported in full)"
    r["pooled"] = False
    return r


def adverse_cell(name: str, e: dict):
    """is T separated WORSE than C on this component (B = T, A = C)? v2 (2026-11-20): ONE shared rule, field_read.entry_adverse_pol (= p5c_read.entry_adverse with the
    three lead-agent safety margins min_ttc_s / headway_min_m / time_gap_min_s read higher-is-better; the correction is applied there, so it holds on an old p5c_read
    export too). MEASURED on famci_C1_vs_C0.json: min_ttc_s ROSE 13.67 -> 13.97, lo > 0, and the inherited default 'lower' read it as adverse. This file no longer carries its own copy."""
    return FR.entry_adverse_pol(name, e)


def fam_cells(j: dict | None, labels=("C", "T")) -> dict:
    """LONGITUDINAL / LATERAL / TACTICAL components of the TRACKED-EXECUTED frame from a fam_ci JSON (diff = B - A = T - C). Adverse via the vendored polarity rule
    (LONGITUDINAL / LATERAL; TACTICAL accuracy / kappa only: per-class recall / precision would take the default polarity, which is wrong for them)."""
    if j is None:
        return {"status": "NOT SUPPLIED", "reason": "no fam_ci JSON (T vs C)", "adverse": []}
    if j.get("label_a") not in (None, labels[0]) or j.get("label_b") not in (None, labels[1]):
        return {"status": "NOT-EVALUATED", "reason": f"fam_ci labels {j.get('label_a')!r}/{j.get('label_b')!r} are not {labels!r}: the diff direction is unproven", "adverse": []}
    fr = (j.get("frames") or {}).get("executed_tracked")
    if not fr:
        return {"status": "NOT-EVALUATED", "reason": "fam_ci JSON has no executed_tracked frame", "adverse": []}
    out = {"status": "COMPUTED", "frame": "executed_tracked (fam_ci diff = B - A = T - C)", "estimator": j.get("estimator"), "adverse": [], "families": {}}
    for fam, key in (("LONGITUDINAL", "longitudinal"), ("LATERAL", "lateral"), ("TACTICAL", "tactical")):
        comp = {}
        for k, e in (fr["families"].get(key) or {}).items():
            if not isinstance(e, dict) or "status" not in e:
                continue
            polarity_ok = key != "tactical" or k.endswith("accuracy") or k.endswith("kappa")
            adv = adverse_cell(k, e) if (polarity_ok and k.split(".")[-1] not in ("gt_progress_mean_m", "t0_axis_gt_self_ratio")) else None
            comp[k] = {"status": e["status"], "n_tokens": e.get("n_tokens"), "T": e.get("B"), "C": e.get("A"), "diff_T_minus_C": e.get("diff_B_minus_A"), "adverse": adv}
            if adv:
                out["adverse"].append(f"{fam}.{k}")
        out["families"][fam] = comp
    lead = {k: v for k, v in out["families"]["LONGITUDINAL"].items() if k in ("headway_min_m", "time_gap_min_s", "min_ttc_s", "lead_viol", "lead_at_risk", "lead_progress_ratio")}
    out["lead_agent_components"] = {"components": lead, "rule": "computed where a lead exists with its n, refused with the reason where none does (families_run.py); never silently dropped"}
    return out


def strategic_block(j: dict | None, n: int, path: str | None = None) -> dict:
    """STRATEGIC IN FULL (PI unpaused 2026-11-16): the strategic_family.py read, reported, no bar. Refuses a JSON that still says PAUSED."""
    if j is None:
        return {"status": "NOT SUPPLIED", "reason": "no strategic_family.py JSON (--arm T=<T seam> --base C=<C seam>)", "n": n}
    b = FR.strategic_block(j, n)
    ctl = json.dumps(j.get("controls", {}))[:20000].upper()
    b["status"] = "INVALID (not quotable)" if "INVALID" in ctl else "REPORTED IN FULL (no bar; PI ruling 2026-11-16)"
    b["file_sha256"] = FR.sha256_file(path) if path else None
    b["tokens"] = j.get("tokens")
    return b


def goal_setting_block(t: dict | None, n: int) -> dict:
    g = (t or {}).get("goal_setting") or {}
    return {"status": "NOT COMPUTABLE", "reason": g.get("reason") or "REFe has no tactical goal head, and NAVSIM carries no manoeuvre label beyond the driving command; the command is an INPUT of the agent", "n": g.get("n") or n}


# ------------------------------------------------------------------------------------------------------ the read
def _arm_names(a):
    return {"T": a.t, "C": a.c, "Ts": a.ts, "Cs": a.cs}


def run_read(a) -> dict:
    nm = {k: v for k, v in _arm_names(a).items() if v}
    if (a.ts is None) != (a.cs is None):
        raise Refused("--ts and --cs must be supplied together")
    uniq = list(dict.fromkeys(nm.values()))
    dry = bool(a.dry_run)
    paths = [p for g in a.rows for p in sorted(glob.glob(g))]
    rows, errs = FR.load_slim(paths, uniq)
    if errs:
        raise Refused(f"{len(errs)} tokens have only error rows (first {errs[0]})")
    P = PR.per_token(rows, uniq)
    tokens, logs = P["tokens"], P["logs"]
    n = len(tokens)
    self_read = a.t == a.c
    need = []
    if not dry:
        for k in nm:
            kk = k.lower()
            for flag in ("seam", "dump", "csv", "proptable", "argv"):
                if not getattr(a, f"{flag}_{kk}", None):
                    need.append(f"--{flag}-{kk}")
        for flag, v in (("--w0-digest-c", a.w0_digest_c), ("--w0-ref", a.w0_ref), ("--det-seam-t", a.det_seam_t), ("--seam-ledger", a.seam_ledger),
                        ("--fam-ci", a.fam_ci), ("--strategic", a.strategic), ("--tactical", a.tactical)):
            if not v:
                need.append(flag)
        if a.ts and not a.fam_ci_s:
            need.append("--fam-ci-s")
        if need:
            raise Refused("a real read refuses without: " + ", ".join(need))
    # --- per-arm inputs
    geo, TAB, DMP, MA = {}, {}, {}, {}
    for k, name in nm.items():
        kk = k.lower()
        sp, dp, tp = getattr(a, f"seam_{kk}"), getattr(a, f"dump_{kk}"), getattr(a, f"proptable_{kk}")
        if sp and dp:
            geo[k] = FR.seam_geometry(sp, dp, tokens)
            DMP[k] = load_dump(dp, tokens)
        elif not dry or k in ("T", "C"):
            raise Refused(f"arm {k}: --seam-{kk} and --dump-{kk} are required")
        if tp:
            TAB[k] = load_table_arrays([p for g in tp for p in sorted(glob.glob(g))], tokens)
    # --- controls
    toks_by = {f"rows:{v}": tokens for v in uniq}
    for k, g in geo.items():
        toks_by[f"dump:{k}"] = [t for t in tokens if t in set(g["dump_tokens"])]
    set_ctl = FR.ctrl_set(toks_by, None if dry else N_NAVTEST, logs, None if dry else L_NAVTEST)
    for k, g in geo.items():
        if (not dry) and set(g["seam_tokens"]) != set(tokens):
            set_ctl["pass"] = False
            set_ctl["population_problems"].append(f"seam:{k}: seam token set != rows token set")
    controls = [set_ctl]
    t_ctl, csv_ctl, unavailable = [], [], []
    for k, name in nm.items():
        kk = k.lower()
        cp = getattr(a, f"csv_{kk}")
        if k in TAB and cp:
            t_ctl.append(ctrl_t(TAB[k], cp, tokens, DMP[k]["pick"], k))
        elif not dry:
            raise Refused(f"arm {k}: table and --csv-{kk} are required in a real read (CTRL-T)")
        else:
            unavailable.append({"control": f"CTRL-T[{k}]", "reason": f"no table or no --csv-{kk} for this stand-in arm: a real read REFUSES without them"})
        if cp:
            csv_ctl.append(dict(FR.ctrl_csv(rows, name, cp, tokens), arm_name=name, arm=k))
    controls += t_ctl + csv_ctl
    if geo:
        controls.append(FR.ctrl_seam(geo, a.seam_ledger, a.convert_report, {k: nm[k] for k in geo}))
    argv = {k: parse_argv(getattr(a, f"argv_{k.lower()}")) for k in nm if getattr(a, f"argv_{k.lower()}", None)}
    if argv:
        controls.append(ctrl_argv(argv))
    else:
        unavailable.append({"control": "CTRL-ARGV", "reason": "no pod argv files exist for these arms (stand-ins): a real read REFUSES without --argv-t/-c(/-ts/-cs)"})
    if a.w0_digest_c and a.w0_ref:
        controls.append(ctrl_w0(a.w0_digest_c, a.w0_ref))
    else:
        unavailable.append({"control": "CTRL-W0", "reason": "no step-0 digest of C and no C1 reference digest for these arms (stand-ins): a real read REFUSES without --w0-digest-c and --w0-ref"})
    if a.det_seam_t and a.det_full_seam_t:
        controls.append(ctrl_det(a.det_full_seam_t, a.det_seam_t))
    elif a.det_seam_t and geo.get("T"):
        controls.append(ctrl_det(geo["T"]["seam"], a.det_seam_t))
    else:
        unavailable.append({"control": "CTRL-DET", "reason": "no re-run seam of T on 200 tokens (--det-seam-t): a real read REFUSES without it"})
    MAt = MAc = None
    if "T" in TAB and "C" in TAB:
        MA = {k: mech_arrays(TAB[k], DMP[k]) for k in TAB}
        MAt, MAc = MA["T"], MA["C"]
    controls.append(ctrl_id(P, logs, MA.get("T"), a.t, a.n_boot))
    if not self_read:
        controls.append(ctrl_id(P, logs, MA.get("C"), a.c, a.n_boot))
    controls_ok = False if any(not c["pass"] for c in controls) else (True if (dry or not unavailable) else None)
    # --- estimands
    bt = Boot(logs, a.n_boot)
    levels = {k: {m: PR.level(bt, P, v, m) for m in METRICS} for k, v in nm.items()}
    con0 = PR.contrast(bt, P, a.t, a.c)
    mech = mc = None
    if MAt is not None:
        mc = mech_contrast(bt, MAt, MAc)
        mech = mech_decide(mc)
    rep = pair_rep(bt, P, a.c, a.t, a.cs, a.ts) if a.ts else None
    v = decide(con0, mech, controls_ok, rep)
    if self_read and (v["clauses"]["P1_lo_gt_0"] or any(c["control"] == "CTRL-ID" and not c["pass"] for c in controls)):
        raise Refused(f"self-read (T == C == {a.t}) found a non-zero Delta or a P1: {v['verdict']}")
    mc_s = None
    if a.ts and "Ts" in TAB and "Cs" in TAB:
        mc_s = mech_contrast(bt, mech_arrays(TAB["Ts"], DMP["Ts"]), mech_arrays(TAB["Cs"], DMP["Cs"]))
    # --- comfort, both forms (descriptive; enters no clause)
    comfort = None
    if MAt is not None:
        n1 = np.ones(n)
        comfort = {"rule": "descriptive; enters NO clause (G1's C is NAVSIM's tracked comfort sub-score); C1's own transparent-comfort GUARD-FAIL stays open on the lineage",
                   "forms": {"transparent_20_pose_plan": "table tp20_pick.c (the 20-pose pick scored without the tracker; the P5-C pass-1 transparent row)",
                             "transparent_8_submitted_poses": "table tp.c[pick] (the 8 submitted poses executed as given)"}, "level_x100": {}, "delta_T_minus_C": {}}
        for form, key in (("transparent_20_pose_plan", "comfort20"), ("transparent_8_submitted_poses", "comfort8")):
            comfort["level_x100"][form] = {"T": bt.ci(MAt[key]), "C": bt.ci(MAc[key])}
            comfort["delta_T_minus_C"][form] = bt.ci(MAt[key] - MAc[key])
        comfort["rows_transparent_c_equals_20_pose"] = {k: float(np.abs(P[v_]["transparent"]["c"] - MA[k]["comfort20"]).max()) for k, v_ in nm.items() if k in MA}
    # --- families
    fam_j = json.load(open(a.fam_ci, encoding="utf-8")) if a.fam_ci else None
    fam_js = json.load(open(a.fam_ci_s, encoding="utf-8")) if a.fam_ci_s else None
    fc = fam_cells(fam_j, tuple(a.fam_labels or ("C", "T")))
    tact = relabel_tactical(json.load(open(a.tactical, encoding="utf-8"))) if a.tactical else None
    strat = strategic_block(json.load(open(a.strategic, encoding="utf-8")) if a.strategic else None, n, a.strategic)
    sel = None
    if mc is not None:
        sel = {"definition": "from each arm's 64-proposal table (p5c_prop_score.py) and proposals dump; tracked PDMS x100; never pooled with another family",
               "levels": {"T": mech_levels(bt, MAt), "C": mech_levels(bt, MAc)}, "T_minus_C": mc,
               "separated_adverse": selection_qualifiers(mc)}
    quals = list(fc.get("adverse", [])) + (sel["separated_adverse"] if sel else [])
    fams = {"pooled": False, "rule": "each family stands alone; no composite; a separated adverse cell is a verdict QUALIFIER, only G1 blocks ADOPT (card section 6)",
            "LONGITUDINAL": (fc["families"]["LONGITUDINAL"] if "families" in fc else fc), "LATERAL": (fc["families"]["LATERAL"] if "families" in fc else fc),
            "lead_agent_components": fc.get("lead_agent_components"),
            "TACTICAL": {"(a)_manoeuvre_decision_command_adherence": tact if tact is not None else {"status": "NOT SUPPLIED", "reason": "no p5c_tactical.py JSON"},
                         "(a2)_fam_ci_tactical_components": (fc["families"]["TACTICAL"] if "families" in fc else fc),
                         "(b)_selection_quality": sel if sel is not None else {"status": "NOT COMPUTED", "reason": "no 64-proposal tables"},
                         "(c)_goal_setting": goal_setting_block(tact, n)},
            "STRATEGIC": strat, "separated_adverse_qualifiers": quals,
            "Ts_minus_Cs_fam_ci": (fam_cells(fam_js, ("Cs", "Ts")) if fam_js else None)}
    goals = {k: goal_flag(levels[k]["pdms"]["tracked"]) for k in nm}
    res = {"instrument": "scorerft_read.py", "card": "PREREG_SCORER_FT.md (FROZEN 2026-11-20)", "card_sha256": PREREG_SHA256, "evidence": "MEASURED on the rows supplied; " + TIER, "tier": TIER,
           "estimator": {"kind": "paired log-cluster bootstrap", "draws": a.n_boot, "seed": SEED, "interval": "95 % percentile",
                         "statistic": "ratio of sums over drawn logs (rates, shares and subset statistics too)", "shared": "ONE draw matrix for every arm and contrast", "answers": ANSWERS},
           "ledger": LOOK2 if a.ts else LOOK, "multiplicity": MULTIPLICITY, "n_tokens": n, "n_logs": bt.L, "units": "PDMS, sub-scores x 100",
           "arms": nm, "self_read": self_read, "stage": 2 if a.ts else 1, "controls": controls, "controls_all_pass": controls_ok,
           "controls_not_available": unavailable, "levels": levels, "goal": goals,
           "T_minus_C": {m: con0[m] for m in METRICS}, "mechanism": mech, "mechanism_Ts_minus_Cs": mc_s,
           "replicate": ({"Ts_minus_Cs": {m: rep["con1"][m] for m in METRICS}, "F_abs_Cs_minus_C": rep["F"], "s_p": rep["s_p"], "d_mean": (rep["d0"] + rep["d1"]) / 2.0}
                         if rep else {"status": "PENDING (Stage 2)", "note": "Cs, Ts not supplied: P2 and P3 print PENDING; ADOPT is not reachable"}),
           "comfort_transparent": comfort, "families": fams, "verdict": v, "ddc": {"T_minus_C": con0["ddc"], "note": "DDC is reported, not a guard (navsim_v1 ignores it)"},
           "adopt_caveat": "ADOPT would not clear C1's own GUARD-FAIL (TM+C: transparent comfort, the longitudinal speed bias); those stay open on the lineage",
           "snooping": "the lever, heads and sizes were chosen from C1's navtest diagnostics (SNOOPED); the arms train on navtrain only; this read tests trained models, not a rule fit on navtest"}
    res["card_questions"] = [
        "Q-S1 outcome for a CONTROL-INVALID read is INVALID, no outcome; Stage 1 with P1 and G1 prints STAGE-2-TRIGGERED, never ADOPT, with P2 / P3 PENDING (Stage 2), as the brief states.",
        "Q-S2 the step-0 digest artifact format is not registered: CTRL-W0 reads the FIRST 64-hex string in the supplied file and compares it to the C1 reference (SCORERFT_BASE_SHA256).",
        "Q-S3 CTRL-DET: the re-run seam must hold >= 200 tokens, all in T's full seam, with poses bit-identical (the card says 'T's seam re-run on 200 tokens is identical').",
        "Q-S4 s_p is not defined by the card: reported beside P3 as |d0 - d1| / sqrt(2) (n = 2), decides nothing; P6's 0.179 is another rig and is not imported.",
        "Q-S5 O5 names no head for a separated adverse C sub-score; the read says so rather than choosing.",
        "Q-S6 M2 'DAC / NC boundary share on rescuable zeroed picks' is read as head_auc.py's share of rescuable zeroed picks whose pick has a head output above the passing median (the 70.2 % of the card), plus the zeroed-by-DAC / by-NC rates; UNVERIFIED reading.",
        "Q-S7 M1 is the paired T - C difference on the tokens with >= 3 safe proposals in BOTH arms (the safe set differs per arm); each arm's own level (own defined set) is beside it.",
        "Q-S8 BELOW GOAL: the arm's tracked PDMS POINT < 94.55 (the brief's 'under 94.55'); whether the CI-hi also lies below is printed.",
        "Q-S9 P3 uses a strict '>' and F as the point |mean(Cs) - mean(C)| (as the field card's Q-F5).",
        "Q-S11 POLARITY DEFECT found in the inherited rule: p5c_read.fam_direction (vendored from proxy_eval) has no entry for min_ttc_s / headway_min_m / time_gap_min_s and defaults to 'lower is better'; on C1 vs C0 it read a RISE of min_ttc_s (+0.297 [0.157, 0.436], safer) as separated adverse. This read takes the corrected rule from field_read.entry_adverse_pol (the ONE shared definition; descriptive qualifiers only); P5-C's G2 list is NOT re-derived here and its min_ttc_s entry should be re-read.",
        "Q-S10 a separated adverse cell of TACTICAL fam_ci components is read for accuracy / kappa only (polarity-safe); per-class recall / precision and goal-setting components are descriptive.",
    ]
    if dry:
        res["STAMP"] = DRY_STAMP
        res["DRY_RUN"] = {"label": a.dry_run, "claim": "none: the arms are stand-ins, the verdict string is a pipeline exercise", "verdict_as_computed": v["verdict"],
                          "controls_not_available_for_stand_ins": [u["control"] for u in unavailable]}
        res["verdict"] = dict(v, verdict="DRY RUN:" + v["verdict"])
    return res


# ------------------------------------------------------------------------------------------------------ rendering
def verdict_line(r: dict) -> str:
    v = r["verdict"]
    cl = v["clauses"]
    d0 = cl["P1_delta"]
    tl = r["levels"]["T"]["pdms"]["tracked"]
    parts = []
    if "STAMP" in r:
        parts.append(r["STAMP"])
    parts.append(f"SCORERFT-READ outcome={v['verdict']} stage={r['stage']} n={r['n_tokens']} logs={r['n_logs']} d0_tracked={d0['point']:.4f} [{d0['lo']:.4f}, {d0['hi']:.4f}]")
    parts.append(f"P1={'PASS' if cl['P1_lo_gt_0'] else 'FAIL'} G1={cl['G1']}{(' adverse=' + ','.join(cl['G1_adverse'])) if cl['G1_adverse'] else ''} P2={cl['P2']} P3={cl['P3']}")
    if r["mechanism"]:
        m = r["mechanism"]
        parts.append(f"M1_moved={m['M1_ep_spearman']['moved']} M3_moved={m['M3_regret']['moved']} M4_damaged={m['m4_damaged']} mechanism_moved={m['mechanism_moved']}")
    g = r["goal"]["T"]
    parts.append(f"T={tl['point']:.2f} {g['status']}")
    q = r["families"]["separated_adverse_qualifiers"]
    parts.append("qualifiers: " + (", ".join(q[:6]) + (f" (+{len(q) - 6} more, see JSON)" if len(q) > 6 else "") if q else "none"))
    s = r["families"]["STRATEGIC"].get("status")
    parts.append(f"strategic: {s}")
    parts.append("controls_all_pass=" + str(r["controls_all_pass"]))
    if r["stage"] == 1 and v["verdict"] not in ("INVALID", "INCOMPLETE"):
        parts.append(ONE_SEED)
    parts.append(r["ledger"] + "; " + MULTIPLICITY)
    return " | ".join(parts)


def text_report(r: dict) -> str:
    L = []
    if "STAMP" in r:
        L += [r["STAMP"], "(" + r["DRY_RUN"]["label"] + ")", ""]
    v = r["verdict"]
    L.append(f"VERDICT: {v['verdict']} | {v['wording']}")
    L.append(f"NEXT ARM (committed in the card, Rule Zero): {v.get('next_arm')}")
    L.append(f"tier: {r['tier']}")
    L.append(f"estimator: {r['estimator']['kind']}, {r['estimator']['draws']} draws, seed {r['estimator']['seed']}; answers: {r['estimator']['answers']}")
    L.append(f"{r['ledger']}; {r['multiplicity']}; card sha256 {r['card_sha256']}")
    L.append(f"n tokens {r['n_tokens']}, logs {r['n_logs']}; arms {r['arms']}; stage {r['stage']}")
    L.append("controls: " + ", ".join(f"{c['control']}={'PASS' if c['pass'] else 'FAIL'}" for c in r["controls"]) + f"  (all pass: {r['controls_all_pass']})")
    for u in r["controls_not_available"]:
        L.append(f"  NOT AVAILABLE: {u['control']}: {u['reason']}")
    cl = v["clauses"]
    d0 = cl["P1_delta"]
    L.append(f"P1 d0 = T - C PDMS (tracked): {ci_s(d0)} -> {'PASS' if cl['P1_lo_gt_0'] else 'FAIL'}")
    L.append(f"G1 (T - C, tracked, none separated adverse): {cl['G1']} {cl['G1_adverse']}")
    for m in G1_SUBS + ("ddc",):
        L.append(f"  sub {m}: tracked {ci_s(r['T_minus_C'][m]['tracked'])} | transparent ({PR.NO_TRACKER}) {ci_s(r['T_minus_C'][m]['transparent'])}" + ("  [DDC: reported, not a guard]" if m == "ddc" else ""))
    L.append(f"P2: {cl['P2']}" + (f" Ts - Cs {ci_s(cl['P2_delta'])} G1-adverse {cl['P2_G1_adverse']}" if "P2_delta" in cl else ""))
    L.append(f"P3: {cl['P3']}" + (f" d = {cl['P3_d']:.4f}, F = {cl['P3_F']:.4f}, 2F = {cl['P3_2F']:.4f}; s_p beside it: {cl['s_p_beside']['value']:.4f}" if "P3_d" in cl else ""))
    for k, g in r["goal"].items():
        L.append(f"GOAL {GOAL}: {k} {g['level_tracked']['point']:.2f} [{g['level_tracked']['lo']:.2f}, {g['level_tracked']['hi']:.2f}] -> {g['status']}")
    m = r["mechanism"]
    if m:
        L.append(f"M1 EP Spearman T - C: {ci_s(m['M1_ep_spearman']['delta'])} moved={m['M1_ep_spearman']['moved']}")
        L.append(f"M2 zeroed-pick rate T - C (pp): {ci_s(m['M2_boundary']['zeroed_pick_rate_pp'])}")
        L.append(f"M3 regret (oracle - pick) T - C: {ci_s(m['M3_regret']['oracle_minus_pick'])} moved={m['M3_regret']['moved']}")
        L.append(f"M4 oracle-of-64 T - C: {ci_s(m['M4_generation_guard']['oracle_of_64'])} damaged={m['m4_damaged']}")
    else:
        L.append("M1-M4: NOT COMPUTED (no 64-proposal tables)")
    c = r["comfort_transparent"]
    if c:
        for f_, d in c["delta_T_minus_C"].items():
            L.append(f"transparent comfort T - C, {f_}: {ci_s(d)} (descriptive, no clause)")
    f = r["families"]
    L.append("families (each stands alone, never pooled):")
    for k in ("LONGITUDINAL", "LATERAL"):
        b = f[k]
        L.append(f"  {k}: " + (f"{b.get('status')} ({b.get('reason')})" if "status" in b and "reason" in b else f"{len(b)} components"))
    L.append("  lead-agent (headway / time-gap / TTC): " + ", ".join(f"{k}={e['status']} n={e['n_tokens']}" for k, e in ((f.get("lead_agent_components") or {}).get("components") or {}).items()))
    L.append("  TACTICAL (b) selection quality: " + ("computed" if "T_minus_C" in f["TACTICAL"]["(b)_selection_quality"] else "NOT COMPUTED") + f"; (c) goal-setting: {f['TACTICAL']['(c)_goal_setting']['status']} (n = {f['TACTICAL']['(c)_goal_setting']['n']})")
    L.append("  STRATEGIC: " + str(f["STRATEGIC"].get("status")))
    L.append("  separated adverse qualifiers: " + (", ".join(f["separated_adverse_qualifiers"]) or "none"))
    L.append(r["adopt_caveat"])
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True)
    for k in ("t", "c"):
        ap.add_argument(f"--{k}", required=True, help=f"arm name of {k.upper()} in the rows")
    for k in ("ts", "cs"):
        ap.add_argument(f"--{k}", default=None)
    for k in ("t", "c", "ts", "cs"):
        for f_ in ("seam", "dump", "csv", "argv"):
            ap.add_argument(f"--{f_}-{k}", default=None)
        ap.add_argument(f"--proptable-{k}", nargs="*", default=None)
    ap.add_argument("--w0-digest-c", default=None, help="file or string holding C's step-0 digest (CTRL-W0)")
    ap.add_argument("--w0-ref", default=None, help="file or string holding C1's sha256 (SCORERFT_BASE_SHA256)")
    ap.add_argument("--det-seam-t", default=None, help="T's seam re-run on >= 200 tokens (CTRL-DET)")
    ap.add_argument("--det-full-seam-t", default=None, help="the seam it is compared with (default: --seam-t)")
    ap.add_argument("--seam-ledger", default=None)
    ap.add_argument("--convert-report", default=None)
    ap.add_argument("--fam-ci", default=None, help="fam_ci.py JSON with --a C --b T")
    ap.add_argument("--fam-ci-s", default=None, help="fam_ci.py JSON with --a Cs --b Ts")
    ap.add_argument("--fam-labels", nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--tactical", default=None, help="p5c_tactical.py JSON run with --seam C1=<T seam> C0=<C seam> base=<C1 seam>")
    ap.add_argument("--strategic", default=None, help="strategic_family.py read JSON")
    ap.add_argument("--dry-run", default=None)
    ap.add_argument("--n-boot", type=int, default=NBOOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--verdict-line", default=None)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        res = run_read(a)
    except Refused as e:
        print("REFUSED:", e, file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=float)
    if a.text:
        with open(a.text, "w", encoding="utf-8", newline="\n") as f:
            f.write(text_report(res))
    line = verdict_line(res)
    if a.verdict_line:
        with open(a.verdict_line, "w", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
    print("ZZSFT_READ " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
