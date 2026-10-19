#!/usr/bin/env python3
"""L2-FRESH-STEPA step 7: the READ.  SEALED-ONLY.  A1-A5 per head on the fresh carve, L1 section 4 + Amendment 1's `interval_aware` rule applied to the L2 head on BOTH seeds (the verdict = the most
severe of the two), P2 as the reference, the PAIRED A1 and A3 differences (L2 minus P2, ungated), the A3 pre-stated consequence, the log-OOF COMPANION (reported-only), the exposure stamps and the four
families (STRATEGIC PAUSED).

THE SEAL GUARD (tested RED, `check_seal`): refuses (exit 5) unless
  (1) the amendment file's sha256 (CRLF folded to LF) equals --prereg-sha256,
  (2) the file carries the positive marker "Status: SEALED" and NOT the draft marker "Status: DRAFT",
  (3) the file NAMES (so the seal covers) the carve's token-list sha256, both final heads' sha256, P2's head (file sha of the L2 reference AND the content sha256 of its arrays), and the log-OOF fold-heads
      file + manifest sha256, exactly as recorded on disk,
  (4) the file NAMES, on a line with its file name, the LF sha256 of every code file this read runs (CODE_PIN_FILES: this file, logoof_companion.py, score_fresh.py, l2lib.py, stepA.py) AND each of those equals
      the file actually on disk: editing any of them after the seal turns the read into a refusal,
  (5) (real read only) C-FEAT passed on this run and the run dir's scores come from THIS feats file.
`--seal-check-only` runs (1)-(4) and stops: it reads NO label, NO score, NO run-dir file (the dry run that proves the guard accepts the sealed file and refuses a mutated copy).
The read itself reads only what step 6 banked; it never trains anything and reaches no GPU.

    python analyse_fresh.py --run-dir <W> --prereg <PREREG_L2_FRESH.md> --prereg-sha256 <hex> --sample <fresh_sample*.json> [--out <W>] [--heads <dir>] [--logoof-dir <dir>]
    python analyse_fresh.py --seal-check-only --prereg <PREREG_L2_FRESH.md> --prereg-sha256 <hex> --sample <fresh_sample*.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L2_CODE = os.environ.get("L2_CODE", "C:/Users/Admin/par18/l2/code")
sys.path.insert(0, L2_CODE)
sys.path.insert(0, HERE)
import l2lib  # noqa: E402
import stepA  # noqa: E402
import logoof_companion as LC  # noqa: E402

DRAFT_MARKER = "Status: DRAFT"
SEALED_MARKER = "Status: SEALED"
SEEDS = ("s0", "s1")
SEVERITY = {"O1": 0, "O3": 1, "O2": 2}
COMPANION_PREFIX = "LOGOOF_"
HEADS_DIR = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA/heads"
LOGOOF_DIR = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA/heads"                      # logoof_heads.npz + logoof_heads.json (a copy of raw/logoof/ in the package)
P2_HEAD_REFERENCE = "D:/Projects/TanitAD-artifacts/l2_offprop/p2_score_head.npz"
CODE_PIN_FILES = ("analyse_fresh.py", "logoof_companion.py", "score_fresh.py", "l2lib.py", "stepA.py")
STAMP = ("SEEN-SCENE, SELECTION tier (L1 A1.3); token-fresh, LOG-SHARED: B is NOT a fresh-log test (its 24 logs are Step A's and the heads' fit logs); ONE training seed per head (H-ESTIM-SEED-1)")


class SealRefused(SystemExit):
    """a seal refusal: a SystemExit whose EXIT STATUS is 5 (a bare `SystemExit("msg")` exits with status 1) and whose str() is the reason, so the tests can read it and the CLI wrapper can print it"""

    def __init__(self, msg: str):
        super().__init__(5)
        self.msg = msg

    def __str__(self):
        return self.msg


def sha_lf(p):
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def code_paths() -> dict:
    return {"analyse_fresh.py": os.path.join(HERE, "analyse_fresh.py"), "logoof_companion.py": os.path.join(HERE, "logoof_companion.py"), "score_fresh.py": os.path.join(HERE, "score_fresh.py"),
            "l2lib.py": os.path.join(L2_CODE, "l2lib.py"), "stepA.py": os.path.abspath(stepA.__file__)}


def current_code_pins() -> dict:
    """LF sha256 of every code file this read runs, as it is on disk NOW"""
    return {n: sha_lf(p) for n, p in code_paths().items()}


def check_seal(prereg: str, sha_given: str, pins: dict, code_pins: dict | None = None) -> dict:
    """-> info dict or SystemExit(5).  `pins` = names -> hex strings the sealed text must contain verbatim; `code_pins` = file name -> LF sha256 of the file on disk now, which must appear on a line that also
    names the file."""
    if not prereg or not os.path.exists(prereg):
        raise SealRefused(f"SEAL REFUSED (5): amendment file not found: {prereg!r}")
    got = sha_lf(prereg)
    if got != str(sha_given).lower():
        raise SealRefused(f"SEAL REFUSED (5): amendment sha256 {got[:12]} != --prereg-sha256 {str(sha_given)[:12]}")
    txt = open(prereg, encoding="utf-8").read()
    if DRAFT_MARKER in txt:
        raise SealRefused(f"SEAL REFUSED (5): the amendment still carries the draft marker {DRAFT_MARKER!r}")
    if SEALED_MARKER not in txt:
        raise SealRefused(f"SEAL REFUSED (5): the amendment does not carry the seal marker {SEALED_MARKER!r}")
    missing = [k for k, v in pins.items() if not v or v not in txt]
    if missing:
        raise SealRefused(f"SEAL REFUSED (5): the sealed text does not name (so does not cover) {missing}")
    lines = txt.splitlines()
    stale = [n for n, h in (code_pins or {}).items() if not any((n in ln and h in ln) for ln in lines)]
    if stale:
        raise SealRefused(f"SEAL REFUSED (5): the sealed text does not name the CURRENT sha256 of code file(s) {stale} (edited after the seal, or never sealed)")
    return {"prereg": prereg, "prereg_sha256": got, "pins": pins, "code_pins": dict(code_pins or {})}


def seal_pins(sample: dict, s0_sha: str, s1_sha: str, p2_head_file_sha: str, p2_head_content_sha: str, logoof_npz_sha: str, logoof_json_sha: str) -> dict:
    return {"carve token-list sha256": sample["tokens_sha256"], "final head s0 sha256": s0_sha, "final head s1 sha256": s1_sha, "P2 head file sha256 (L2 reference)": p2_head_file_sha,
            "P2 head content sha256 (arrays)": p2_head_content_sha, "log-OOF fold-heads file sha256": logoof_npz_sha, "log-OOF fold-heads manifest sha256": logoof_json_sha}


def head_content_sha(path: str) -> str:
    z = np.load(path, allow_pickle=False)
    return l2lib.sha256_arrays(np.asarray(z["weight"], np.float32), np.asarray(z["bias"], np.float32))


def disk_pins(sample: dict, heads_dir: str, logoof_dir: str, p2_head: str = P2_HEAD_REFERENCE) -> dict:
    """the pins as the files on disk say they are; reads NO label and NO score (head files, the reference P2 head, the fold-heads file, the carve's token list)"""
    toks = sorted(str(t) for t in sample["tokens"])
    if hashlib.sha256("\n".join(toks).encode()).hexdigest() != sample["tokens_sha256"]:
        raise SealRefused("SEAL REFUSED (5): the sample file's token list does not hash to its own tokens_sha256")
    fh = json.load(open(os.path.join(heads_dir, "final_heads.json")))["heads"]
    hs = {}
    for s in SEEDS:
        p = os.path.join(heads_dir, f"final_head_{s}.npz")
        if sha_file(p) != fh[s]["sha256"]:
            raise SealRefused(f"SEAL REFUSED (5): final head {s} on disk differs from final_heads.json")
        hs[s] = fh[s]["sha256"]
    return seal_pins(sample, hs["s0"], hs["s1"], sha_file(p2_head), head_content_sha(p2_head), sha_file(os.path.join(logoof_dir, LC.HEADS_NPZ)), sha_file(os.path.join(logoof_dir, LC.HEADS_JSON)))


def combine_seeds(outcomes: dict) -> dict:
    """the committed combination: the L2 head is judged on BOTH seeds; the verdict is the MOST SEVERE of the two (O2 > O3 > O1), so O1 needs both seeds at O1"""
    worst = max(outcomes.values(), key=lambda o: SEVERITY[o])
    return {"per_seed": dict(outcomes), "combined": worst, "seeds_agree": len(set(outcomes.values())) == 1}


def paired_terms_diff(D, arm_a, arm_b, key):
    """paired (A - B) log-cluster difference of a ratio-of-sums admission quantity (`A1`/`A2`/`A3`...) on the SAME resamples; descriptive, ungated"""
    ta = stepA.per_log_sums({key: stepA.admission_terms(l2lib._arm_pairs(D, *arm_a))[key]}, D.pair_log, D.n_logs)[key]
    tb = stepA.per_log_sums({key: stepA.admission_terms(l2lib._arm_pairs(D, *arm_b))[key]}, D.pair_log, D.n_logs)[key]
    return l2lib.paired_diff_ci(ta[0], ta[1], tb[0], tb[1], D.boot_idx)


def a3_consequence(arms: dict) -> dict | None:
    """MM ruling (pre-stated, no new gate): A3 stays REPORTED (paired, log-cluster).  If the paired A3 difference's UPPER CI end is below 0 on BOTH final-head seeds, the read RECORDS an EP-RELIAB-class
    risk, and the Step B amendment must report the aggregate concordance of the search outputs beside dPDMS.  Evaluated on the FINAL heads (L2_s0, L2_s1); the companion is not consulted."""
    seeds = [n for n in arms if n.startswith("L2_") and "A3_minus_P2_paired" in arms[n]]
    if not seeds:
        return None
    his = {n: arms[n]["A3_minus_P2_paired"]["hi"] for n in seeds}
    trig = bool(all(h < 0 for h in his.values()))
    return {"rule": "paired dA3 (L2 - P2) upper CI end < 0 on BOTH seeds => record an EP-RELIAB-class risk; the Step B amendment must then report the aggregate concordance of the search outputs beside dPDMS (no gate)",
            "dA3_upper_per_seed": his, "dA3_point_per_seed": {n: arms[n]["A3_minus_P2_paired"]["point"] for n in seeds}, "triggered": trig,
            "consequence": ("EP-RELIAB-CLASS RISK RECORDED: the Step B amendment must report the aggregate concordance of the search outputs beside dPDMS" if trig else "not triggered")}


def evaluate_all(D, scores: dict, exposure: dict | None = None) -> dict:
    """pure given Data + the EP scores {arm: (ep_in, ep_p)}; used by the sealed read AND by the instrument-validity test on Step A's own data.  Arms named LOGOOF_* are the log-OOF COMPANION: scored and
    reported like any other, but REPORTED-ONLY: no outcome is assigned to them and they never enter the L2 verdict."""
    res = {"arms": {}, "bootstrap": {"B": l2lib.BOOT_B, "seed": l2lib.BOOT_SEED, "unit": "log", "n_logs": int(D.n_logs), "answers": "another draw of LOGS only; one training seed (H-ESTIM-SEED-1)"}}
    ev_p2 = l2lib.evaluate_arm(D, *scores["P2"])
    ref = l2lib.ref_terms_of(ev_p2)
    for name, (ei, ep) in scores.items():
        ev = ev_p2 if name == "P2" else l2lib.evaluate_arm(D, ei, ep, ref)
        companion = name.startswith(COMPANION_PREFIX)
        a = {"A": {tag: {k: ev[tag][k] for k in ("A1", "A2", "A3", "A4", "A5", "A5b")} for tag in ("all", "lon", "lat")}, "n_pairs": ev["n_pairs"], "gated": not companion,
             "outcome_interval_aware": None if companion else stepA.classify(ev["all"]["A1"], ev["all"]["A4"], ev["all"]["A5"], "interval_aware"),
             "outcome_if_literal": None if companion else stepA.classify(ev["all"]["A1"], ev["all"]["A4"], ev["all"]["A5"], "literal"),
             "lon_A1_by_speed_scale": ev["lon_A1_by_speed_scale"], "lat_DAC_concordance": ev["lat_DAC_concordance"], "infan_top8_ep_concordance": ev["infan_top8_ep_concordance"],
             "tactical": ev["tactical"], "strategic": ev["strategic"]}
        if name != "P2":
            a["A1_minus_P2_paired"] = ev["A1_minus_P2_paired"]
            a["A3_minus_P2_paired"] = paired_terms_diff(D, (ei, ep), scores["P2"], "A3")
            a["infan_minus_P2_paired"] = ev["infan_minus_P2_paired"]
        res["arms"][name] = a
    l2 = {n: v["outcome_interval_aware"] for n, v in res["arms"].items() if n.startswith("L2_")}
    l2_lit = {n: v["outcome_if_literal"] for n, v in res["arms"].items() if n.startswith("L2_")}
    if l2:
        res["L2_outcome"] = combine_seeds(l2)
        res["L2_outcome_if_literal"] = combine_seeds(l2_lit)
        res["a3_consequence"] = a3_consequence(res["arms"])
    res["strategic_family"] = "PAUSED (PI ruling 2026-09-19, ITEM 25); reported as PAUSED, not as a pass"
    res["exposure"] = exposure or {}
    return res


def _ci(d):
    return f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def render_md(res: dict, title: str = "L2-FRESH-STEPA read") -> str:
    """the result as a markdown table: one row per head, A1-A5 (all), A1 lon / lat, in-fan, paired differences vs P2 (ungated), the outcome by the DECIDING rule (interval_aware) and, for the record only,
    the literal reading; then the A3 consequence and the log-OOF companion (reported-only)"""
    L = [f"# {title}", "", f"**{STAMP}**", "",
         f"Estimator: paired log-cluster bootstrap over {res['bootstrap']['n_logs']} logs, B={res['bootstrap']['B']}, seed {res['bootstrap']['seed']}; it answers 'another draw of LOGS' only (H-ESTIM-SEED-1: one training seed per head; the two L2 seeds are near-replicates).",
         "", "| head | A1 all | A1 lon | A1 lat | A2 | A3 | A4 | A5 | in-fan top-8 EP | A1 - P2 (paired) | A3 - P2 (paired) | outcome: interval_aware (DECIDES) / literal (record only) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, a in res["arms"].items():
        A = a["A"]
        oc = f"{a['outcome_interval_aware']} / {a['outcome_if_literal']}" if a.get("gated", True) else "reported-only, no outcome"
        L.append(f"| {n} | {_ci(A['all']['A1'])} | {_ci(A['lon']['A1'])} | {_ci(A['lat']['A1'])} | {_ci(A['all']['A2'])} | {_ci(A['all']['A3'])} | {_ci(A['all']['A4'])} | {_ci(A['all']['A5'])} | "
                 f"{_ci(a['infan_top8_ep_concordance'])} | {_ci(a['A1_minus_P2_paired']) if 'A1_minus_P2_paired' in a else '-'} | {_ci(a['A3_minus_P2_paired']) if 'A3_minus_P2_paired' in a else '-'} | {oc} |")
    if "L2_outcome" in res:
        L += ["", f"**L2 outcome (interval_aware, most severe of the two seeds): {res['L2_outcome']['combined']}** (per seed {res['L2_outcome']['per_seed']}). Literal reading, for the record only, decides nothing: {res['L2_outcome_if_literal']['combined']}.", ""]
    ac = res.get("a3_consequence")
    if ac:
        L += [f"**A3 (reported, ungated):** paired dA3 upper CI end per final-head seed {ac['dA3_upper_per_seed']}; pre-stated consequence: {ac['consequence']}.", ""]
    comp = res.get("companion")
    if comp:
        L += [f"**Log-OOF companion (reported-only, never gated):** {comp.get('status', '')}"] + ([f"Omitted: {comp['omitted']}"] if comp.get("omitted") else [f"Fold heads {comp.get('heads_npz_sha256', '')[:12]}; rows LOGOOF_* above."]) + [""]
    L += ["Gates: A1 >= 0.80, A4 <= 0.10, A5 <= 0.05 (L1 prereg section 4, Amendment 1 `interval_aware`).", f"STRATEGIC: {res['strategic_family']}",
          f"Exposure: {json.dumps(res.get('exposure', {}).get('stamp', ''))}"]
    return "\n".join(L) + "\n"


def load_scores(run_dir):
    z = np.load(f"{run_dir}/fresh_ep_scores.npz", allow_pickle=False)
    arms = sorted({k[len("ep_in_"):] for k in z.files if k.startswith("ep_in_")})
    return {a: (z[f"ep_in_{a}"], z[f"ep_p_{a}"]) for a in arms}


def companion_scores(D, logoof_dir: str) -> tuple:
    """-> ({LOGOOF_s0: (ep_in, ep_p), ...}, record).  Each token of D is scored by the fold head whose held-out fold contains its LOG.  Omitted (empty dict + reason) if the carve's logs are not the heads' logs."""
    h = LC.load_heads(os.path.join(logoof_dir, LC.HEADS_NPZ), os.path.join(logoof_dir, LC.HEADS_JSON))
    man = h["manifest"]
    ok, why = LC.compatible_logs(D.log_names, man["log_names"])
    rec = {"status": "reported-only, never gated (MM ruling 2026-10-19)", "heads_npz_sha256": man["heads_npz_sha256"], "reproduction_pass": man["reproduction"]["pass"], "log_compat": why}
    if not ok:
        rec["omitted"] = why
        return {}, rec
    fold_of_log = {l: int(f) for l, f in zip(man["log_names"], man["fold_of_log"])}
    fold = np.array([fold_of_log[str(x)] for x in D.bundle["log"]], np.int64)
    if not np.array_equal(fold, D.fold):
        raise SystemExit("log-OOF companion: the fold of a token from the heads' manifest differs from l2lib's fold_of_logs (the fold assignment changed)")
    out = {}
    for s, heads in h["heads"].items():
        out[f"{COMPANION_PREFIX}s{s}"] = LC.score_by_fold(D.tok_in, D.tok_p, fold, heads)
    rec["fold_sizes_tokens"] = [int((fold == f).sum()) for f in range(LC.N_FOLDS)]
    return out, rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--prereg-sha256", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--heads", default=HEADS_DIR)
    ap.add_argument("--logoof-dir", default=LOGOOF_DIR)
    ap.add_argument("--seal-check-only", action="store_true", help="run the seal guard on the files on disk and stop: reads no label, no score, no run-dir file")
    a = ap.parse_args(argv)
    sample = json.load(open(a.sample))
    if a.seal_check_only:
        seal = check_seal(a.prereg, a.prereg_sha256, disk_pins(sample, a.heads, a.logoof_dir), current_code_pins())
        print("ZZSEAL_ACCEPTED", json.dumps({"prereg_sha256": seal["prereg_sha256"], "code_pins": seal["code_pins"], "pins": {k: v[:12] for k, v in seal["pins"].items()}}), flush=True)
        return 0
    if not a.run_dir:
        raise SystemExit("--run-dir is required for the read")
    W_ = a.run_dir
    sc_rec = json.load(open(f"{W_}/fresh_ep_scores.json"))
    ref_pins = disk_pins(sample, a.heads, a.logoof_dir)
    # the run's own record must agree with the files the seal names: the heads it scored, and the P2 head ARRAYS it scored with (the file bytes of its copy may differ; the arrays may not)
    if sc_rec["heads"]["s0"]["sha256"] != ref_pins["final head s0 sha256"] or sc_rec["heads"]["s1"]["sha256"] != ref_pins["final head s1 sha256"]:
        raise SystemExit("the run's scores were made with different final heads than the ones the seal names")
    if head_content_sha(f"{W_}/feats/p2_score_head.npz") != ref_pins["P2 head content sha256 (arrays)"]:
        raise SystemExit("the run's P2 head arrays differ from the reference P2 head the seal names")
    seal = check_seal(a.prereg, a.prereg_sha256, ref_pins, current_code_pins())
    if not sc_rec.get("pass"):
        raise SystemExit("C-FEAT did not pass on this run: no read")
    if sc_rec["feats_sha256"] != sha_file(f"{W_}/feats/feats.npz"):
        raise SystemExit("fresh_ep_scores.json was made from a different feats.npz")
    D = l2lib.load_real(f"{W_}/feats/feats.npz", run_dir=W_)
    toks = [str(t) for t in D.bundle["token"]]
    if hashlib.sha256("\n".join(sorted(toks)).encode()).hexdigest() != sample["tokens_sha256"]:
        raise SystemExit("the run's bundle tokens are not the sealed carve")
    exposure = {"g0": sample.get("g0"), "mode": sample.get("mode"), "stepA_overlap": sample.get("stepA_overlap"), "stamp": STAMP}
    scores = load_scores(W_)
    comp_scores, comp_rec = companion_scores(D, a.logoof_dir)
    scores.update(comp_scores)
    res = evaluate_all(D, scores, exposure)
    res["companion"] = comp_rec
    res["seal"] = seal
    out = a.out or W_
    json.dump(l2lib.jsonable(res), open(os.path.join(out, "FRESH_RESULT.json"), "w"), indent=1)
    open(os.path.join(out, "FRESH_RESULT.md"), "w", encoding="utf-8").write(render_md(res))
    print("ZZFRESH_READ_DONE", json.dumps({"L2_outcome": res.get("L2_outcome"), "literal": res.get("L2_outcome_if_literal")}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SealRefused as e:
        print(e.msg, file=sys.stderr, flush=True)
        sys.exit(5)
