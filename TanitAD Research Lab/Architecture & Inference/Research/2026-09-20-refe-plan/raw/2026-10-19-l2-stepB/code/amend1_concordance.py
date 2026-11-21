#!/usr/bin/env python3
"""L2 STEP B, AMENDMENT 1 (SEALED, REPORTED ONLY): the AGGREGATE CONCORDANCE of the search outputs.

`PREREG_L2_STEPB_AMENDMENT1.md` (sealed 2026-10-19, before ZZSTEPB_DONE) adds ONE reported-only read beside dPDMS and the four families: for each arm (L2-s0, L2-s1, P2HEAD) and each search seed, the pair
(anchor = P2's pick, candidate = the search output) per token, scored by the ARM'S OWN scorer (6 heads) and by the TRUE labels, then A3 / A4 / A5 with L1's `stepA.admission_terms` (imported UNCHANGED, the L1
code is sha-pinned and re-verified here: never copied).  It never gates, never changes Step B's verdict, and is computed by THIS script AFTER `ZZSTEPB_DONE`.

  A3  the arm's aggregate (planner V1_W = 5,5,2) against the TRUE PDMS; a pair counts only if the true PDMS differs by >= 0.05, a scorer tie counts 0.5
  A4  among pairs whose aggregate rises by >= 0.02: the share whose true PDMS falls by >= Step A's threshold (the exploit rate the search actually realised)
  A5  among those same pairs: the share where true NC or DAC goes 1 -> 0
Pooled over the 3 search seeds within an arm, and each per-seed value beside it.  Estimator: the paired log-cluster bootstrap of `stepA` (whole logs, B = 10,000, seed 20260927, the SAME draws for every quantity
because they come from `stepA.boot_indices`), plus the paired difference of each L2 arm against P2HEAD.  It answers 'another draw of LOGS' and nothing else (not another head seed, not another search seed: the
per-seed values are the search-seed spread).

INPUTS: Step B's run dir (`search_<arm>.npz`, `labels_<arm>.npz`, `chain_stepB.log`) and the carve-B bundle.  The arm's scorer logits at P2's pick (`pred_logit_pick`) and at the search output (`pred_logit_out`) ARE
banked by `stepB.run_search` (the chain's writer), so by default NOTHING is re-scored.  The identity that guards them:
  * structural, always (no model): shapes / finiteness / token + pick + label-file alignment; an UNMOVED output's banked logits equal the pick's bit for bit; the non-EP columns of the L2 arms' pick logits equal
    P2HEAD's bit for bit (only the EP row differs); the searches / labels / bundle / head / checkpoint / config shas are the SEALED pins;
  * model identity (`--identity-tokens N`, or ALL tokens in `--logits rescore`): the SAME `score_fn` path (`stepB.build_scorer`: head installed by `l2_scorer.install_ep_head`, sha-checked, CPU) re-scores P2's
    pick and the moved outputs and must reproduce the banked logits BIT FOR BIT, else NOT COMPUTABLE (exit 3) with the reason.
`--logits rescore` (or `auto` when `pred_logit_out` is absent) uses the re-scored logits, after the identity held at EVERY token.

Refusals: the sealed amendment's .sha256 must match its text (and the main PREREG's its own): exit 5, nothing written.  No `ZZSTEPB_DONE` in the chain log: exit 4 (`--allow-no-done` for a stopped chain).
Outputs (`--out-dir`): `amend1_result.json` + `AMEND1.md` (section "AMENDMENT 1: aggregate concordance of the search outputs (reported only)").  CPU only, no lock, no GPU.
Never touches the chain, never writes into the run dir unless `--out-dir` says so.

    CUDA_VISIBLE_DEVICES=-1 python amend1_concordance.py --run-dir <run> --bundle <bundle.npz> --out-dir <dir> [--identity-tokens 16] [--logits auto|banked|rescore]
Exit codes: 0 computed; 3 NOT COMPUTABLE; 4 precondition refused (chain not done); 5 seal / L1-code refusal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

sys.dont_write_bytecode = True                       # never rewrite .pyc files the running chain's children import

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import analyse_stepB as AN  # noqa: E402  (log_index = Step B's own log indexing; STAMPS)
import stepB  # noqa: E402  (puts L1's code on sys.path)
import stepB_chain as CH  # noqa: E402  (the chain's own ARMS, pins parser and paths: one source of truth)

A, T = stepB.A, stepB.T                            # L1's stepA / tts_cem, imported unchanged

SECTION = "AMENDMENT 1: aggregate concordance of the search outputs (reported only)"
MAIN_MD, MAIN_SHA = "PREREG_L2_STEPB.md", "PREREG_L2_STEPB.sha256"
AMEND_MD, AMEND_SHA = "PREREG_L2_STEPB_AMENDMENT1.md", "PREREG_L2_STEPB_AMENDMENT1.sha256"
KEYS = ("A3", "A4", "A5")
ARMS = CH.ARMS                                     # (("L2s0", "L2", "s0"), ("L2s1", "L2", "s1"), ("P2HEAD", "P2HEAD", None))
DISPLAY = {"L2s0": "L2-s0", "L2s1": "L2-s1", "P2HEAD": "P2HEAD"}
EXPECT_LOGS = 24
DONE_RE = re.compile(r"^ZZSTEPB_DONE(?:\s|$)", re.M)         # the marker's PRESENCE only: the verdict word after it is never captured, never printed
EP = stepB.l2_scorer.EP
EXIT = {"computed": 0, "not_computable": 3, "precondition": 4, "seal": 5}
ESTIMATOR = ("paired log-cluster bootstrap: whole LOGS resampled with the SAME draws for every quantity (stepA.boot_indices; B = 10,000, seed 20260927); a difference is the per-resample difference of the two "
             "ratios of sums.  It answers 'another draw of LOGS' and nothing else: not another head seed (H-ESTIM-SEED-1), not another search seed (the per-seed values are that spread).")


class SealRefused(RuntimeError):
    """the sealed text or L1's code is not what was sealed: exit 5, nothing written"""


class PreconditionRefused(RuntimeError):
    """Step B is not done: exit 4, nothing written"""


class NotComputable(RuntimeError):
    """an identity or an input failed: NOT COMPUTABLE with the reason (exit 3)"""


def sha256_file(p: str) -> str:
    return stepB.l2_scorer.sha256_file(p)


# ------------------------------------------------------------------------------------------------------------------------------ the seals
def check_seals(pkg: str) -> dict:
    """the amendment's .sha256 must equal the sha256 of its text, and so must the main PREREG's (the amendment changes nothing of it).  CRLF-folded and raw bytes are both accepted (the chain's own convention).
    Returns the two shas, the main text and its PINS block."""
    got = {}
    for key, md, sh in (("main", MAIN_MD, MAIN_SHA), ("amendment", AMEND_MD, AMEND_SHA)):
        pm, ps = os.path.join(pkg, md), os.path.join(pkg, sh)
        if not (os.path.isfile(pm) and os.path.isfile(ps)):
            raise SealRefused(f"SEAL_MISSING: {pm} / {ps}: no amendment read without its sealed text")
        raw = open(pm, "rb").read()
        lf = raw.replace(b"\r\n", b"\n")
        rec = open(ps, encoding="utf-8").read().split()
        if not rec or not re.fullmatch(r"[0-9a-fA-F]{64}", rec[0]):
            raise SealRefused(f"SEAL_BAD_RECORD: {ps} does not start with a sha256")
        ok = {hashlib.sha256(raw).hexdigest(), hashlib.sha256(lf).hexdigest()}
        if rec[0].lower() not in ok:
            raise SealRefused(f"SEAL_MISMATCH: the {key} text's sha256 {sorted(ok)[0][:12]} != the recorded {rec[0][:12]}: it was edited after sealing")
        got[key] = {"sha256": rec[0].lower(), "text": lf.decode("utf-8")}
    if CH.SEALED_MARK not in got["main"]["text"] or CH.DRAFT_MARK in got["main"]["text"]:
        raise SealRefused(f"SEAL_NOT_SEALED: the main text must carry {CH.SEALED_MARK!r} and not {CH.DRAFT_MARK!r}")
    try:
        pins = CH.parse_pins(got["main"]["text"])
    except ValueError as e:
        raise SealRefused(f"SEAL_NO_PINS: {e}")
    return {"main_sha256": got["main"]["sha256"], "amendment_sha256": got["amendment"]["sha256"], "pins": pins}


def check_l1_code() -> dict:
    try:
        return stepB.verify_l1_code()
    except Exception as e:
        raise SealRefused(f"L1_CODE: {e}")


def chain_done(path: str) -> bool:
    txt = CH.read_text(path)
    return bool(txt is not None and DONE_RE.search(txt))


# ------------------------------------------------------------------------------------------------------------------------------ loading
def _meta(so: dict) -> dict:
    try:
        return json.loads(str(so["meta"]))
    except Exception as e:
        raise NotComputable(f"search file meta unreadable ({type(e).__name__})")


def load_arm(run_dir: str, name: str) -> dict:
    sp, lp = os.path.join(run_dir, f"search_{name}.npz"), os.path.join(run_dir, f"labels_{name}.npz")
    for p in (sp, lp):
        if not os.path.isfile(p):
            raise NotComputable(f"missing input {p}")
    so = {k: v for k, v in np.load(sp, allow_pickle=False).items()}
    lab = {k: v for k, v in np.load(lp, allow_pickle=False).items()}
    return {"name": name, "so": so, "lab": lab, "search_path": sp, "labels_path": lp, "meta": _meta(so)}


REQ_SEARCH = ("token", "bundle_index", "pick", "seeds", "exec", "pick_exec", "moved", "pred_logit_pick", "meta")
REQ_LABELS = ("sub", "pdms", "token", "bundle_index", "search_sha256", "identity")


def validate_arm(arm: dict, bundle: dict, n_bundle: int) -> None:
    """structural checks of ONE arm against the bundle (shapes, finiteness, token / pick / label-file alignment, the label identity record).  Raises NotComputable."""
    nm, so, lab = arm["name"], arm["so"], arm["lab"]
    for k in REQ_SEARCH:
        if k not in so:
            raise NotComputable(f"{nm}: search file lacks '{k}'")
    for k in REQ_LABELS:
        if k not in lab:
            raise NotComputable(f"{nm}: labels file lacks '{k}'")
    ex = np.asarray(so["exec"])
    if ex.ndim != 4 or ex.shape[2:] != (20, 3):
        raise NotComputable(f"{nm}: exec has shape {ex.shape}, expected [n,S,20,3]")
    n, S = ex.shape[:2]
    if n != n_bundle or not np.array_equal(np.asarray(so["bundle_index"]), np.arange(n_bundle)):
        raise NotComputable(f"{nm}: the search file does not cover the bundle's {n_bundle} tokens in order (n={n})")
    if np.asarray(so["moved"]).shape != (n, S) or np.asarray(so["pred_logit_pick"]).shape != (n, 6) or np.asarray(so["pick_exec"]).shape != (n, 20, 3) or np.asarray(so["seeds"]).shape != (S,):
        raise NotComputable(f"{nm}: moved / pred_logit_pick / pick_exec / seeds shapes do not match exec {ex.shape}")
    if "pred_logit_out" in so and np.asarray(so["pred_logit_out"]).shape != (n, S, 6):
        raise NotComputable(f"{nm}: pred_logit_out has shape {np.asarray(so['pred_logit_out']).shape}, expected {(n, S, 6)}")
    if not np.isfinite(np.asarray(so["pred_logit_pick"], np.float64)).all() or ("pred_logit_out" in so and not np.isfinite(np.asarray(so["pred_logit_out"], np.float64)).all()):
        raise NotComputable(f"{nm}: non-finite banked logits")
    if np.asarray(lab["sub"]).shape != (n, S, 6) or np.asarray(lab["pdms"]).shape != (n, S):
        raise NotComputable(f"{nm}: label shapes {np.asarray(lab['sub']).shape} / {np.asarray(lab['pdms']).shape}, expected {(n, S, 6)} / {(n, S)}")
    if not (np.isfinite(np.asarray(lab["sub"], np.float64)).all() and np.isfinite(np.asarray(lab["pdms"], np.float64)).all()):
        raise NotComputable(f"{nm}: non-finite labels (a partial labelling is not analysed)")
    toks = [str(t) for t in np.asarray(bundle["token"])]
    if [str(t) for t in so["token"]] != toks or [str(t) for t in lab["token"]] != toks:
        raise NotComputable(f"{nm}: the search / labels tokens are not the bundle's tokens in order")
    if not np.array_equal(np.asarray(lab["bundle_index"]), np.asarray(so["bundle_index"])):
        raise NotComputable(f"{nm}: labels bundle_index differs from the search file's")
    if not np.array_equal(np.asarray(so["pick"], np.int64), np.asarray(bundle["pick"], np.int64)):
        raise NotComputable(f"{nm}: the search file's pick is not the bundle's pick (P2's pick must be the anchor)")
    if str(lab["search_sha256"]) != sha256_file(arm["search_path"]):
        raise NotComputable(f"{nm}: the labels are not bound to this search file (search_sha256 differs)")
    try:
        ident = json.loads(str(lab["identity"]))
    except Exception:
        raise NotComputable(f"{nm}: the labels' identity record is unreadable")
    if not (ident.get("passed") is True and ident.get("control_ok") is True):
        raise NotComputable(f"{nm}: the labeller's planted identity (or its control) did not pass: these labels are not admissible")


def structural_identities(arms: dict) -> dict:
    """the model-free identities on the BANKED logits (each raises NotComputable):
      (1) an UNMOVED output IS the pick: exec == pick_exec and, where banked, pred_logit_out == pred_logit_pick, BIT FOR BIT
      (2) the three arms share the same pick plan and the same search seeds
      (3) only the EP row differs between an L2 arm and P2HEAD: the other five columns of pred_logit_pick are equal BIT FOR BIT"""
    rec = {"unmoved_outputs_checked": {}, "p2head_vs_l2_non_ep_columns_bitwise_equal": {}}
    seeds0, pe0 = None, None
    for nm, arm in arms.items():
        so = arm["so"]
        moved = np.asarray(so["moved"], bool)
        um = ~moved
        if um.any():
            if not np.array_equal(np.asarray(so["exec"])[um], np.broadcast_to(np.asarray(so["pick_exec"])[:, None], np.asarray(so["exec"]).shape)[um]):
                raise NotComputable(f"identity FAILED ({nm}): an unmoved output's exec is not the pick's exec")
            if "pred_logit_out" in so:
                lo, lp = np.asarray(so["pred_logit_out"], np.float64), np.asarray(so["pred_logit_pick"], np.float64)
                bad = np.argwhere(um & ~np.all(lo == lp[:, None, :], axis=2))
                if len(bad):
                    i, s = bad[0]
                    raise NotComputable(f"identity FAILED ({nm}): token {so['token'][i]} seed index {s} is UNMOVED (the output IS the pick) but its banked output logits differ from the banked pick logits")
        rec["unmoved_outputs_checked"][nm] = int(um.sum())
        if seeds0 is None:
            seeds0, pe0 = np.asarray(so["seeds"]), np.asarray(so["pick_exec"])
        else:
            if not np.array_equal(seeds0, np.asarray(so["seeds"])):
                raise NotComputable(f"{nm}: search seeds differ across arms")
            if not np.array_equal(pe0, np.asarray(so["pick_exec"])):
                raise NotComputable(f"{nm}: the pick's executed plan differs across arms (the anchor must be the same P2 pick)")
    if tuple(int(s) for s in seeds0) != tuple(int(s) for s in T.SEARCH_SEEDS):
        raise NotComputable(f"search seeds {list(seeds0)} are not the registered {list(T.SEARCH_SEEDS)}")
    if "P2HEAD" in arms:
        others = [c for c in range(6) if c != EP]
        p2 = np.asarray(arms["P2HEAD"]["so"]["pred_logit_pick"], np.float64)
        for nm, arm in arms.items():
            if nm == "P2HEAD":
                continue
            lp = np.asarray(arm["so"]["pred_logit_pick"], np.float64)
            ok = bool(np.array_equal(lp[:, others], p2[:, others]))
            rec["p2head_vs_l2_non_ep_columns_bitwise_equal"][nm] = ok
            if not ok:
                i = int(np.argmax(np.abs(lp[:, others] - p2[:, others]).max(1)))
                raise NotComputable(f"identity FAILED ({nm}): the banked pick logits' non-EP columns differ from P2HEAD's at token {arm['so']['token'][i]} (only the EP row may differ between an L2 arm and plain P2)")
    return rec


def check_pins(arms: dict, pins: dict, bundle: dict, bundle_sha: str, planted_carve: bool) -> dict:
    """every merged search file must bind the SEALED head / checkpoint / config / carve and THIS bundle (the chain's own `search_valid` rule, re-asserted here on the files this script reads)"""
    carve = pins["carve_sha256"]
    if not planted_carve and carve != stepB.SEALED_CARVE_SHA256:
        raise NotComputable("the sealed carve pin differs from stepB.SEALED_CARVE_SHA256")
    try:
        clause = stepB.check_data_clause({"token": np.asarray(bundle["token"])}, carve)
    except SystemExit as e:
        raise NotComputable(f"DATA CLAUSE: {e}")
    heads = {}
    for name, kind, s in ARMS:
        m = arms[name]["meta"]
        if m.get("arm") != kind:
            raise NotComputable(f"{name}: the search file's arm is {m.get('arm')!r}, expected {kind!r}")
        want_head = pins[f"head_{s}_sha256"] if s else None
        if m.get("head_sha256") != want_head:
            raise NotComputable(f"{name}: the search file's head sha256 {str(m.get('head_sha256'))[:12]} is not the sealed pin {str(want_head)[:12]}")
        for k, pk in (("ckpt_sha256", "ckpt_sha256"), ("config_sha256", "config_sha256")):
            if m.get(k) != pins[pk]:
                raise NotComputable(f"{name}: the search file's {k} {str(m.get(k))[:12]} is not the sealed pin {pins[pk][:12]}")
        if m.get("bundle_sha256") != bundle_sha:
            raise NotComputable(f"{name}: the search file was made on another bundle (sha256 {str(m.get('bundle_sha256'))[:12]} != {bundle_sha[:12]})")
        if (m.get("data_clause") or {}).get("carve_sha256") != carve:
            raise NotComputable(f"{name}: the search file's data clause is not the sealed carve")
        heads[name] = want_head
    return {"carve_sha256": carve, "data_clause": clause, "heads": heads}


# ------------------------------------------------------------------------------------------------------------------------------ the model identity / re-score (the SAME score_fn path as the chain's search)
def build_arm_scorer(arm: dict, bundle_path: str, ctx: str, ckpt: str, refe_dir: str, heads_dir: str, scorer_mod=None):
    m = arm["meta"]
    if sha256_file(ckpt) != m["ckpt_sha256"]:
        raise NotComputable(f"{arm['name']}: the checkpoint {ckpt!r} is not the one the search used (sha256 differs)")
    cj = os.path.join(os.path.dirname(os.path.abspath(ctx)), "ctx.json")
    if os.path.isfile(cj):
        want = json.load(open(cj, encoding="utf-8")).get("ctx_sha256")
        if want and sha256_file(ctx) != want:
            raise NotComputable(f"{arm['name']}: ctx.npy's sha256 differs from ctx.json (a torn or changed ctx)")
    head = None
    if m["arm"] == "L2":
        s = [s for n, _, s in ARMS if n == arm["name"]][0]
        head = os.path.join(heads_dir, f"final_head_{s}.npz")
    ns = argparse.Namespace(arm=m["arm"], head=head, head_sha256=m.get("head_sha256"), bundle=bundle_path, ctx=ctx, ckpt=ckpt, ckpt_sha256=m["ckpt_sha256"], refe_dir=refe_dir, autocast="none")
    try:
        return stepB.build_scorer(ns, scorer_mod=scorer_mod)
    except stepB.l2_scorer.HeadRefused as e:
        raise NotComputable(f"{arm['name']}: the head was refused at load ({e})")


def rescore_rows(scorer, bundle: dict, so: dict, rows) -> tuple:
    """re-score P2's pick and every MOVED output exactly as `stepB.run_search` did: the complete 64-set, the pick's slot replaced by the banked scorer input, the pick's row read"""
    S = np.asarray(so["moved"]).shape[1]
    pick_re, out_re = np.zeros((len(rows), 6)), np.zeros((len(rows), S, 6))
    for j, i in enumerate(rows):
        bi = int(so["bundle_index"][i])
        scorer.bind(bi)
        fan = np.asarray(bundle["traj_raw"][bi], np.float32)
        pick = int(bundle["pick"][bi])
        base = np.asarray(scorer(np.array(fan, copy=True)), np.float64)
        pick_re[j] = base[pick]
        for s in range(S):
            out_re[j, s] = A.score_replaced(scorer, fan, pick, so["scorer_in"][i, s])[pick] if bool(so["moved"][i, s]) else base[pick]
    return pick_re, out_re


def model_identity(arm: dict, bundle: dict, scorer, rows) -> tuple:
    """re-score `rows` and require BIT-FOR-BIT equality with the banked pick logits (and with the banked output logits where banked).  Raises NotComputable with the first failing token."""
    so, nm = arm["so"], arm["name"]
    pick_re, out_re = rescore_rows(scorer, bundle, so, rows)
    bank = np.asarray(so["pred_logit_pick"], np.float64)[rows]
    if not np.array_equal(pick_re, bank):
        d = np.abs(pick_re - bank).max(1)
        j = int(np.argmax(d))
        raise NotComputable(f"identity FAILED ({nm}): the re-scored logits at P2's pick differ from the banked pred_logit_pick at token {so['token'][rows[j]]} (max |diff| {d[j]:.3e}, {int((d > 0).sum())} of {len(rows)} tokens differ): the banked logits are not reproduced bit for bit")
    rec = {"tokens_checked": int(len(rows)), "pick_logits_bitwise_equal": True, "max_abs_diff_pick": 0.0, "out_logits_bitwise_equal": None}
    if "pred_logit_out" in so:
        bo = np.asarray(so["pred_logit_out"], np.float64)[rows]
        if not np.array_equal(out_re, bo):
            d = np.abs(out_re - bo).max(2)
            j, s = np.unravel_index(int(np.argmax(d)), d.shape)
            raise NotComputable(f"identity FAILED ({nm}): the re-scored logits at the search output differ from the banked pred_logit_out at token {so['token'][rows[j]]} seed index {s} (max |diff| {d[j, s]:.3e})")
        rec["out_logits_bitwise_equal"] = True
    return rec, out_re


# ------------------------------------------------------------------------------------------------------------------------------ the pairs
def build_pairs(bundle: dict, arm: dict, li: np.ndarray, out_logit: np.ndarray) -> tuple:
    """(P, log_idx, seed_idx, counts) for one arm: anchor = P2's pick (its BANKED label, the arm's own logits at the pick), candidate = the search output (its label, the arm's own logits at the output).  An output
    IDENTICAL to the pick (unmoved, or exec == pick_exec) contributes to no pair; an anchor whose banked label is invalid is dropped (stepA.assemble_pairs's rule) and counted."""
    so, lab = arm["so"], arm["lab"]
    n, S = np.asarray(so["moved"]).shape
    bi = np.asarray(so["bundle_index"], np.int64)
    pick = np.asarray(bundle["pick"], np.int64)[bi]
    a_sub = np.asarray(bundle["sub"], np.float64)[bi, pick]                                  # [n,6]  the pick's BANKED label
    a_pdms = np.asarray(bundle["pdms"], np.float64)[bi, pick]                                # [n]
    valid = np.asarray(bundle["valid"], bool)[bi, pick] if "valid" in bundle else np.ones(n, bool)
    anchor_ok = valid & np.isfinite(a_sub).all(1) & np.isfinite(a_pdms)
    identical = (~np.asarray(so["moved"], bool)) | np.all(np.asarray(so["exec"]) == np.asarray(so["pick_exec"])[:, None], axis=(2, 3))     # [n,S]
    keep = (~identical) & anchor_ok[:, None]
    ii, ss = np.nonzero(keep)
    P = {"a_logit": np.asarray(so["pred_logit_pick"], np.float64)[ii], "p_logit": np.asarray(out_logit, np.float64)[ii, ss],
         "a_sub": a_sub[ii], "p_sub": np.asarray(lab["sub"], np.float64)[ii, ss], "a_pdms": a_pdms[ii], "p_pdms": np.asarray(lab["pdms"], np.float64)[ii, ss]}
    counts = {"n_tokens": int(n), "n_seeds": int(S), "n_pairs_total": int(n * S), "n_identical_excluded": int((identical & anchor_ok[:, None]).sum()), "n_dropped_invalid_anchor": int((~anchor_ok).sum() * S),
              "n_pairs": int(len(ii))}
    return P, li[ii], ss, counts


def _empty(P: dict) -> bool:
    return len(P["a_pdms"]) == 0


def arm_sums(P: dict, log_idx, seed_idx, n_logs: int, S: int) -> dict:
    """per-log (num, den) of A3 / A4 / A5 from L1's `stepA.admission_terms` (UNCHANGED) and `stepA.per_log_sums`: pooled, and one per search seed"""
    if _empty(P):
        z = (np.zeros(n_logs), np.zeros(n_logs))
        return {"pooled": {k: z for k in KEYS}, "per_seed": [{k: z for k in KEYS} for _ in range(S)], "pair_counts": {"A3_counted": 0.0, "A3_concordant_weight": 0.0, "A4_rise": 0.0, "A4_rise_and_fall": 0.0, "A5_rise_and_flip": 0.0}}
    terms = A.admission_terms(P)
    sub = {k: terms[k] for k in KEYS}
    pooled = A.per_log_sums(sub, log_idx, n_logs)
    per_seed = [A.per_log_sums(sub, log_idx, n_logs, mask=(np.asarray(seed_idx) == s)) for s in range(S)]
    pc = {"A3_counted": float(terms["A3"][1].sum()), "A3_concordant_weight": float(terms["A3"][0].sum()), "A4_rise": float(terms["A4"][1].sum()), "A4_rise_and_fall": float(terms["A4"][0].sum()),
          "A5_rise_and_flip": float(terms["A5"][0].sum())}
    return {"pooled": pooled, "per_seed": per_seed, "pair_counts": pc}


def diff_ci(num_a, den_a, num_b, den_b, idx, alpha: float = 0.05) -> dict:
    """paired difference ratio_a - ratio_b of two ratios of sums on the SAME resampled logs (idx = stepA.boot_indices).  A resample counts only if both denominators are > 0."""
    na, da, nb, db = (np.asarray(x, np.float64) for x in (num_a, den_a, num_b, den_b))
    pa = float(na.sum() / da.sum()) if da.sum() > 0 else float("nan")
    pb = float(nb.sum() / db.sum()) if db.sum() > 0 else float("nan")
    ok = (da[idx].sum(1) > 0) & (db[idx].sum(1) > 0)
    if not ok.any():
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_valid_resamples": 0}
    r = na[idx].sum(1)[ok] / da[idx].sum(1)[ok] - nb[idx].sum(1)[ok] / db[idx].sum(1)[ok]
    return {"point": pa - pb, "lo": float(np.quantile(r, alpha / 2)), "hi": float(np.quantile(r, 1 - alpha / 2)), "n_valid_resamples": int(ok.sum())}


def _clean(o):
    """JSON-safe: numpy -> python, NaN / inf -> None (an undefined ratio, e.g. a zero denominator)"""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def ci_dict(sums_pair: tuple, idx) -> dict:
    d = A.ratio_ci(sums_pair[0], sums_pair[1], idx)
    d["num"] = float(np.asarray(sums_pair[0]).sum())
    return d


def spread(points) -> dict:
    p = [x for x in points if x is not None and math.isfinite(x)]
    return {"points": list(points), "n_finite": len(p), "range": (max(p) - min(p)) if len(p) > 1 else None}


# ------------------------------------------------------------------------------------------------------------------------------ the whole computation
def compute(run_dir: str, bundle_path: str, pkg: str = CH.PKG, logits: str = "auto", identity_tokens: int = 0, scorer_mod=None, heads_dir: str = CH.HEADS_DIR, ckpt: str = CH.CKPT,
            ctx: str | None = None, refe_dir: str = CH.REFE_DIR, threads: int = 4, expect_logs: int | None = EXPECT_LOGS, B: int = A.N_BOOT, seed: int = A.SEED, planted_carve: bool = False,
            require_done: bool = True, chain_log: str | None = None, seals: dict | None = None) -> dict:
    """the whole read.  Raises SealRefused / PreconditionRefused / NotComputable; otherwise returns the result dict (status COMPUTED).  `planted_carve` and `scorer_mod` are for the tests (no CLI flag reaches them)."""
    t0 = time.time()
    seals = seals or check_seals(pkg)
    l1 = check_l1_code()
    log = chain_log or os.path.join(run_dir, "chain_stepB.log")
    if require_done and not chain_done(log):
        raise PreconditionRefused(f"no ZZSTEPB_DONE marker in {log}: this amendment is computed after Step B is done (--allow-no-done only for a stopped chain)")
    if logits not in ("auto", "banked", "rescore"):
        raise NotComputable(f"unknown --logits {logits!r}")
    bundle = A.load_npz(bundle_path)
    n_bundle = len(bundle["token"])
    bundle_sha = sha256_file(bundle_path)
    arms = {name: load_arm(run_dir, name) for name, _, _ in ARMS}
    for arm in arms.values():
        validate_arm(arm, bundle, n_bundle)
    pin_rec = check_pins(arms, seals["pins"], bundle, bundle_sha, planted_carve)
    struct = structural_identities(arms)
    have_banked = all("pred_logit_out" in a["so"] for a in arms.values())
    mode = ("banked" if have_banked else "rescore") if logits == "auto" else logits
    if mode == "banked" and not have_banked:
        raise NotComputable("--logits banked, but pred_logit_out is not banked in every search file")
    if mode == "rescore" and any("scorer_in" not in a["so"] for a in arms.values()):
        raise NotComputable("--logits rescore needs the banked scorer inputs (scorer_in), which the search files lack")
    idl = None
    out_logits = {name: (np.asarray(a["so"]["pred_logit_out"], np.float64) if "pred_logit_out" in a["so"] else None) for name, a in arms.items()}
    identity = {"structural": struct, "model": {}, "logits_source": mode}
    if mode == "rescore" or identity_tokens > 0:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise NotComputable("CPU only: set CUDA_VISIBLE_DEVICES=-1 (the re-score never uses the GPU)")
        import torch
        torch.set_num_threads(int(threads))
        ctx = ctx or os.path.join(os.path.dirname(os.path.abspath(bundle_path)), "ctx.npy")
        rows = list(range(n_bundle)) if mode == "rescore" else sorted({int(round(x)) for x in np.linspace(0, n_bundle - 1, min(int(identity_tokens), n_bundle))})
        for name, arm in arms.items():
            sc = build_arm_scorer(arm, bundle_path, ctx, ckpt, refe_dir, heads_dir, scorer_mod)
            rec, out_re = model_identity(arm, bundle, sc, rows)
            identity["model"][name] = rec
            if mode == "rescore":
                out_logits[name] = out_re
            del sc
    identity["passed"] = True
    # ------------------------------------------------------------------ the statistics: ONE set of bootstrap draws for every quantity
    li_all, n_logs, log_names = AN.log_index(bundle, np.arange(n_bundle))
    if expect_logs is not None and n_logs != expect_logs:
        raise NotComputable(f"the bundle has {n_logs} logs, the sealed estimator is the {expect_logs}-log bootstrap")
    idx = A.boot_indices(n_logs, B, seed)
    S = int(np.asarray(arms["P2HEAD"]["so"]["moved"]).shape[1])
    sums, res_arms = {}, {}
    for name, kind, s in ARMS:
        arm = arms[name]
        P, lidx, sidx, counts = build_pairs(bundle, arm, li_all, out_logits[name])
        sm = arm_sums(P, lidx, sidx, n_logs, S)
        sums[name] = sm
        per_seed = []
        for si in range(S):
            row = {"seed": int(arm["so"]["seeds"][si])}
            for k in KEYS:
                row[k] = ci_dict(sm["per_seed"][si][k], idx)
            per_seed.append(row)
        res_arms[name] = {"display": DISPLAY[name], "head_seed": s, "counts": {**counts, **{k: v for k, v in sm["pair_counts"].items()}},
                          "pooled": {k: ci_dict(sm["pooled"][k], idx) for k in KEYS}, "per_seed": per_seed,
                          "seed_spread": {k: spread([r[k]["point"] if math.isfinite(r[k]["point"]) else None for r in per_seed]) for k in KEYS}}
    diffs = {}
    for name, _, s in ARMS:
        if name == "P2HEAD":
            continue
        pooled = {k: diff_ci(*sums[name]["pooled"][k], *sums["P2HEAD"]["pooled"][k], idx) for k in KEYS}
        per_seed = []
        for si in range(S):
            row = {"seed": int(arms[name]["so"]["seeds"][si])}
            for k in KEYS:
                row[k] = diff_ci(*sums[name]["per_seed"][si][k], *sums["P2HEAD"]["per_seed"][si][k], idx)
            per_seed.append(row)
        diffs[name] = {"pooled": pooled, "per_seed": per_seed}
    inputs = {"run_dir": os.path.abspath(run_dir), "bundle": {"path": os.path.abspath(bundle_path), "sha256": bundle_sha, "n_tokens": n_bundle, "n_logs": n_logs},
              "search_files": {n: {"path": os.path.abspath(a["search_path"]), "sha256": sha256_file(a["search_path"])} for n, a in arms.items()},
              "label_files": {n: {"path": os.path.abspath(a["labels_path"]), "sha256": sha256_file(a["labels_path"])} for n, a in arms.items()},
              "heads_sha256": pin_rec["heads"], "carve_sha256": pin_rec["carve_sha256"], "ckpt_sha256": seals["pins"]["ckpt_sha256"], "config_sha256": seals["pins"]["config_sha256"],
              "l1_code_sha256": l1, "script_sha256": sha256_file(os.path.abspath(__file__)), "chain_done_checked": bool(require_done)}
    result = {"status": "COMPUTED", "section": SECTION, "reported_only": True, "gated": False, "changes_step_b_verdict": False, "main_prereg_sha256": seals["main_sha256"], "amendment_sha256": seals["amendment_sha256"],
              "estimator": {"kind": "paired log-cluster bootstrap", "unit": "log", "n_logs": n_logs, "B": B, "seed": seed, "answers": ESTIMATOR}, "units": "ratios in [0, 1]; differences are L2 arm minus P2HEAD",
              "definitions": {"A3": "arm aggregate vs TRUE PDMS concordance (pairs with |dPDMS| >= 0.05; scorer tie = 0.5)", "A4": "of pairs whose aggregate rises >= 0.02: share whose true PDMS falls by >= Step A's 0.10",
                              "A5": "of the same pairs: share where true NC or DAC goes 1 -> 0"},
              "seeds": [int(x) for x in arms["P2HEAD"]["so"]["seeds"]], "identity": identity, "arms": res_arms, "differences_L2_minus_P2HEAD": diffs, "inputs": inputs,
              "stamps": AN.STAMPS, "elapsed_s": round(time.time() - t0, 2), "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    return _clean(result)


# ------------------------------------------------------------------------------------------------------------------------------ the report
def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _cell(d):
    if d is None or d.get("point") is None:
        return "n/a (den 0)"
    lo, hi = d.get("lo"), d.get("hi")
    return f"{d['point']:.4f} [{_f(lo)}, {_f(hi)}]" if lo is not None else f"{d['point']:.4f} [n/a]"


def render_md(r: dict) -> str:
    L = [f"## {SECTION}", ""]
    if r.get("status") != "COMPUTED":
        L += [f"**NOT COMPUTABLE.** Reason: {r.get('reason')}", "",
              f"Sealed text: `{AMEND_MD}` sha256 `{r.get('amendment_sha256')}`.  REPORTED ONLY; it changes no gate, bar, arm or verdict of `{MAIN_MD}` (sha256 `{r.get('main_prereg_sha256')}`).",
              "Per the amendment, an identity that fails means the addition is reported as NOT COMPUTABLE, with the reason above; no number is quoted.", ""]
        return "\n".join(L) + "\n"
    est, idn = r["estimator"], r["identity"]
    L += [f"**REPORTED ONLY. It is never gated and it changes no gate, bar, arm or verdict of `{MAIN_MD}` (sha256 `{r['main_prereg_sha256']}`); Step B's analysis and verdict line stand exactly as sealed.**  "
          f"Sealed text: `{AMEND_MD}` sha256 `{r['amendment_sha256']}`.", "",
          "- **Why this section exists (INHERITED from the sealed amendment, not re-measured here):** `MM_RULING_L2_FRESH.md` / `PREREG_L2_FRESH.md` (sha256 `5006ed84...`) committed, before READ 1 was read, that if the paired dA3 upper CI were below 0 on both "
          "seeds the Step B amendment would report the aggregate concordance of the search outputs beside dPDMS; READ 1 fired that condition.  This section carries it out; it is a record, not a test.",
          "- **Evidence class:** MEASURED (ours; the inputs and their sha256 are listed under Provenance).",
          f"- **Tier / stamp:** {r['stamps']}",
          f"- **Estimator:** {est['answers']}  ({est['n_logs']} logs, B = {est['B']:,}, seed {est['seed']}.)",
          f"- **What a pair is:** anchor = P2's pick (its BANKED label), candidate = the search output (its label); both scored by the ARM'S OWN scorer (6 heads, the pick's slot replaced inside the complete 64-set); an output identical to the pick contributes to no pair.  "
          f"Terms are L1's `stepA.admission_terms`, unchanged (L1 code sha-pinned and re-verified).",
          f"- **Logits source:** `{idn['logits_source']}`.  Identity (passed): structural identities on the banked logits"
          + ("" if not idn["model"] else "; model identity (re-score on CPU, same `score_fn` path, bit for bit) on " + ", ".join(f"{k}: {v['tokens_checked']} tokens" for k, v in idn["model"].items())) + ".",
          f"- **Definitions:** A3 = {r['definitions']['A3']}; A4 = {r['definitions']['A4']}; A5 = {r['definitions']['A5']}.", ""]
    names = [n for n, _, _ in ARMS]
    L += ["### Pooled over the 3 search seeds: point [95 % CI] (pairs counted)", "", "| arm | A3 concordance | A4 exploit rate | A5 NC/DAC-breach rate |", "|---|---|---|---|"]
    for n in names:
        a = r["arms"][n]
        L.append(f"| {a['display']} | " + " | ".join(f"{_cell(a['pooled'][k])} (n={int(a['pooled'][k]['den'])})" for k in KEYS) + " |")
    L += ["", "n counts: A3 = pairs whose true PDMS differs by >= 0.05; A4 / A5 = pairs whose aggregate rises by >= 0.02.", "",
          "### L2 minus P2HEAD, paired (pooled over the 3 search seeds): point [95 % CI]", "", "| L2 arm | dA3 | dA4 | dA5 |", "|---|---|---|---|"]
    for n in ("L2s0", "L2s1"):
        d = r["differences_L2_minus_P2HEAD"][n]["pooled"]
        L.append(f"| {DISPLAY[n]} - P2HEAD | " + " | ".join(_cell(d[k]) for k in KEYS) + " |")
    L += ["", "### Per search seed (the search-seed spread): point [95 % CI]", "", "| arm | seed | A3 | A4 | A5 |", "|---|---|---|---|---|"]
    for n in names:
        for row in r["arms"][n]["per_seed"]:
            L.append(f"| {r['arms'][n]['display']} | {row['seed']} | " + " | ".join(_cell(row[k]) for k in KEYS) + " |")
    L += ["", "Spread of the per-seed points (max - min over defined seeds): " + "; ".join(f"{DISPLAY[n]} " + ", ".join(f"{k} {_f(r['arms'][n]['seed_spread'][k]['range'])}" for k in KEYS) for n in names) + ".", "",
          "### L2 minus P2HEAD, paired, per search seed: point [95 % CI]", "", "| L2 arm | seed | dA3 | dA4 | dA5 |", "|---|---|---|---|---|"]
    for n in ("L2s0", "L2s1"):
        for row in r["differences_L2_minus_P2HEAD"][n]["per_seed"]:
            L.append(f"| {DISPLAY[n]} - P2HEAD | {row['seed']} | " + " | ".join(_cell(row[k]) for k in KEYS) + " |")
    L += ["", "### Pair counts", "", "| arm | tokens x seeds | identical to the pick (excluded) | invalid anchor (dropped) | pairs | A3 counted | A3 concordant weight | rise pairs | rise and fall | rise and NC/DAC 1->0 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        c = r["arms"][n]["counts"]
        L.append(f"| {DISPLAY[n]} | {c['n_pairs_total']} | {c['n_identical_excluded']} | {c['n_dropped_invalid_anchor']} | {c['n_pairs']} | {c['A3_counted']:.0f} | {c['A3_concordant_weight']:.1f} | {c['A4_rise']:.0f} | {c['A4_rise_and_fall']:.0f} | {c['A5_rise_and_flip']:.0f} |")
    inp = r["inputs"]
    L += ["", "### Provenance", "", f"- bundle `{inp['bundle']['path']}` sha256 `{inp['bundle']['sha256']}` ({inp['bundle']['n_tokens']} tokens, {inp['bundle']['n_logs']} logs); carve `{inp['carve_sha256'][:12]}`; checkpoint `{inp['ckpt_sha256'][:12]}`; "
          f"heads " + ", ".join(f"{k}: `{(v or 'plain P2')[:12]}`" for k, v in inp["heads_sha256"].items()) + ".",
          "- search files: " + "; ".join(f"{k} `{v['sha256'][:12]}`" for k, v in inp["search_files"].items()) + ".  labels: " + "; ".join(f"{k} `{v['sha256'][:12]}`" for k, v in inp["label_files"].items()) + ".",
          f"- script `amend1_concordance.py` sha256 `{inp['script_sha256'][:12]}`; L1 code " + ", ".join(f"{k} `{v[:12]}`" for k, v in inp["l1_code_sha256"].items()) + ".", ""]
    return "\n".join(L) + "\n"


def write_outputs(out_dir: str, result: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, text in (("amend1_result.json", json.dumps(result, indent=1) + "\n"), ("AMEND1.md", render_md(result))):
        p = os.path.join(out_dir, name)
        tmp = p + f".tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True, help="Step B's run dir (search_<arm>.npz, labels_<arm>.npz, chain_stepB.log); read only")
    ap.add_argument("--bundle", required=True, help="the carve-B bundle.npz the chain used")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pkg", default=CH.PKG, help="dir holding the sealed PREREG_L2_STEPB*.md and their .sha256")
    ap.add_argument("--logits", default="auto", choices=("auto", "banked", "rescore"))
    ap.add_argument("--identity-tokens", type=int, default=0, help="re-score this many evenly spaced tokens per arm on CPU and require the banked logits bit for bit (rescore mode: ALL tokens)")
    ap.add_argument("--heads-dir", default=CH.HEADS_DIR)
    ap.add_argument("--ckpt", default=CH.CKPT)
    ap.add_argument("--ctx", default=None, help="default: ctx.npy beside the bundle")
    ap.add_argument("--refe-dir", default=CH.REFE_DIR)
    ap.add_argument("--threads", type=int, default=4, help="the chain's search used 4: keep it for a bit-for-bit identity")
    ap.add_argument("--expect-logs", type=int, default=EXPECT_LOGS)
    ap.add_argument("--chain-log", default=None)
    ap.add_argument("--allow-no-done", action="store_true", help="compute although the chain log has no ZZSTEPB_DONE (a stopped chain)")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    os.environ["MKL_NUM_THREADS"] = str(a.threads)
    seals = None
    try:
        seals = check_seals(a.pkg)
        res = compute(a.run_dir, a.bundle, pkg=a.pkg, logits=a.logits, identity_tokens=a.identity_tokens, heads_dir=a.heads_dir, ckpt=a.ckpt, ctx=a.ctx, refe_dir=a.refe_dir, threads=a.threads,
                      expect_logs=a.expect_logs, require_done=not a.allow_no_done, chain_log=a.chain_log, seals=seals)
    except SealRefused as e:
        print(f"ZZAMEND1_REFUSED {e}", flush=True)
        return EXIT["seal"]
    except PreconditionRefused as e:
        print(f"ZZAMEND1_REFUSED {e}", flush=True)
        return EXIT["precondition"]
    except NotComputable as e:
        res = {"status": "NOT_COMPUTABLE", "section": SECTION, "reported_only": True, "gated": False, "changes_step_b_verdict": False, "reason": str(e), "main_prereg_sha256": seals["main_sha256"],
               "amendment_sha256": seals["amendment_sha256"], "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        write_outputs(a.out_dir, res)
        print(f"ZZAMEND1_NOT_COMPUTABLE {e}", flush=True)
        return EXIT["not_computable"]
    write_outputs(a.out_dir, res)
    print(f"ZZAMEND1_DONE {os.path.join(a.out_dir, 'amend1_result.json')} (reported only; logits {res['identity']['logits_source']})", flush=True)
    return EXIT["computed"]


if __name__ == "__main__":
    sys.exit(main())
