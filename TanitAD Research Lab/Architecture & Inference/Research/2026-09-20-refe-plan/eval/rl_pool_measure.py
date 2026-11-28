#!/usr/bin/env python3
"""RL-S1 priority 1: MEASURE which navtrain scenes each candidate base's SCORER trained on, and what is left for RL.

Reads only banked artifacts (no pod, no network):
  --bank     onpolicy_sets_logs_full.tar.zst   (the 190,712 on-policy set lines; md5 4fc97993... on the pod side)
  --grow     refe_train_grow.tar.zst           (TargetBank rows targets_rank{0,1}.jsonl; fixed scorer_targets*.jsonl keys)
  --final    refe-final-2026-10-01/            (metrics.jsonl: the per-epoch on-policy bank sizes of the main run)
  --splits   <PKG>/splits                      (navtrain.yaml, sft_heldout_logs.txt)
  --navtest  navtest_full_tokens.json          (12,146 tokens + token_log)
Writes <out>/rl_pool_by_base.json, <out>/navtrain_token_table.tsv.gz, <out>/served_keys.tsv.gz.

Needs Python >= 3.14 (`compression.zstd`) or the `zstandard` package.  The head-field parse is select_bank_subset.head_fields, the
relabeller's own, so the served-key definition (per (log, token, step, rank) the row with the largest (ckpt_step, label_version)) cannot drift.
    python eval/rl_pool_measure.py --bank ... --grow ... --final ... --navtest ... --out raw/rl_pool
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "refe"))
import select_bank_subset as SB  # noqa: E402  -- the relabeller's own head parse

# SFT-4 constants, each with its source (see the JSON for the evidence class)
SFT4_BATCH, SFT4_ACCUM, SFT4_U = 8, 8, 400                 # eval/PREREG_SFT4.md l.44; u400 = update 400 (RESULT_SFT4.md)
SFT4_TOTAL_UPDATES = 2335                                   # STOP_CONSEQUENCES: "update ~1,000 of 2,335"
SFT4_LABELS_AT_ARMING = 146185                              # PREREG_SFT4.md Amendment 4
MAIN_RUN_END_AT = "2026-10-01T00:18:49Z"                    # refe-final-2026-10-01/summary.json finished_at


@contextlib.contextmanager
def gz_text(path):
    """gzip text writer with a fixed header (mtime 0, no embedded name): the same content gives the same bytes, so a manifest's sha256 of this file is stable."""
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        tw = io.TextIOWrapper(gz, encoding="utf-8", newline="\n")
        try:
            yield tw
        finally:
            tw.flush()
            tw.detach()


def open_zstd(path):
    try:
        from compression import zstd
        return zstd.ZstdFile(path, "rb")
    except ImportError:
        import zstandard
        return zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))


def md5_of(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_of(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


AT = re.compile(r'"at":\s*"([^"]*)"')


def scan_bank(path):
    """One dict per set line: file, line, log, token, step, rank, ckpt_step, label_version, at."""
    rows = []
    with open_zstd(path) as zf:
        tf = tarfile.open(fileobj=zf, mode="r|")
        for m in tf:
            if not (m.isfile() and m.name.endswith(".jsonl") and os.path.basename(m.name).startswith("onpolicy_")):
                continue
            base = os.path.basename(m.name)
            for i, raw in enumerate(tf.extractfile(m)):
                s = raw.decode("utf-8", "replace")
                f = SB.head_fields(s[:2000] + s[-400:])
                if not {"log_name", "step", "ckpt_step"} <= set(f):
                    continue
                at = (AT.findall(s[-600:]) or [""])[-1]
                rows.append({"file": base, "line": i, "log": f["log_name"], "token": f.get("token", ""), "step": int(f["step"]),
                             "rank": int(f.get("rank", 0)), "ckpt": int(f["ckpt_step"]), "lv": int(f.get("label_version", 1)), "at": at})
    return rows


KEYRX = {k: re.compile(r'"%s":\s*("([^"]*)"|-?\d+)' % k) for k in ("log_name", "token", "step", "rank")}


def _key_of(s):
    d = {}
    for k, rx in KEYRX.items():
        mm = rx.search(s)
        d[k] = (mm.group(2) if mm.group(2) is not None else mm.group(1)) if mm else ""
    return d


def scan_grow(path):
    """TargetBank rows (targets_rank*.jsonl) and the FIXED scorer-target keys (scorer_targets*.jsonl) of the main run's bank."""
    tb, fixed = set(), set()
    with open_zstd(path) as zf:
        tf = tarfile.open(fileobj=zf, mode="r|")
        for m in tf:
            if not m.isfile():
                continue
            b = os.path.basename(m.name)
            if re.fullmatch(r"targets_rank[01]\.jsonl", b):
                dst, n = tb, 4000
            elif re.fullmatch(r"scorer_targets(_rank1)?\.jsonl", b):
                dst, n = fixed, 400
            else:
                continue
            for raw in tf.extractfile(m):
                d = _key_of(raw[:n].decode("utf-8", "replace") if dst is fixed else raw.decode("utf-8", "replace"))
                dst.add((d["log_name"], d["token"], int(d["step"] or 0), int(d["rank"] or 0)))
    return tb, fixed


def read_yaml_lists(path):
    logs, toks, mode = [], [], None
    for ln in open(path, encoding="utf-8"):
        s = ln.rstrip("\n")
        if s.startswith("log_names:"):
            mode = "l"; continue
        if s.startswith("tokens:"):
            mode = "t"; continue
        m = re.match(r"\s+-\s+'([^']+)'", s)
        if m and mode == "l":
            logs.append(m.group(1))
        elif m and mode == "t":
            toks.append(m.group(1))
        elif s and not s.startswith(" ") and mode:
            mode = None
    return logs, toks


def hyper_unseen(n, m, s):
    """P(a token with s labelled sets has none of them among m draws without replacement from n sets)."""
    p = 1.0
    for i in range(s):
        p *= (n - m - i) / (n - i)
    return p


def epoch_bank_sizes(final_dir):
    out = []
    for ln in open(os.path.join(final_dir, "metrics.jsonl"), encoding="utf-8"):
        try:
            d = json.loads(ln.replace("NaN", "null"))
        except json.JSONDecodeError:
            continue
        if d.get("event") == "bank" and "onpolicy_sets" in d:
            out.append({"epoch": d["epoch"], "at": d["at"], "onpolicy_sets": d["onpolicy_sets"]})
    return out


def analyse(a):
    ho = set(open(os.path.join(a.splits, "sft_heldout_logs.txt"), encoding="utf-8").read().split())
    y_logs, y_toks = read_yaml_lists(os.path.join(a.splits, "navtrain.yaml"))
    yset = set(y_toks)
    nt = json.load(open(a.navtest, encoding="utf-8"))
    nt_toks, nt_log = set(nt["tokens"]), nt["token_log"]
    nt_logs = set(nt_log.values()) if isinstance(nt_log, dict) else set()
    rows = scan_bank(a.bank)
    tb, fixed = scan_grow(a.grow)

    # served = the largest (ckpt_step, label_version) per (log, token, step, rank); first-labelled time = min 'at' over ALL rows of the key
    best, first_at, order = {}, {}, {}
    fo = {n: i for i, n in enumerate(sorted({r["file"] for r in rows}))}                  # onpolicy_relabel_pdm: files = sorted(glob)
    for r in rows:
        k = (r["log"], r["token"], r["step"], r["rank"])
        rk = (r["ckpt"], r["lv"])
        if k not in best or best[k][0] < rk:
            best[k] = (rk, r["file"], r["line"], r["at"])
        if r["at"] and (k not in first_at or r["at"] < first_at[k]):
            first_at[k] = r["at"]
    served_train = {k: v for k, v in best.items() if k[0] not in ho}
    served_ho = {k: v for k, v in best.items() if k[0] in ho}
    tok_log = {}
    for (lg, tk, _s, _r) in tb:
        tok_log[tk] = lg
    for k in best:
        tok_log.setdefault(k[1], k[0])
    toks_tr = {k[1] for k in served_train}
    toks_ho_served = {k[1] for k in served_ho}
    toks_served = toks_tr | toks_ho_served
    tb_toks = {k[1] for k in tb if k[3] == 0}
    ho_toks_map = {t for t, lg in tok_log.items() if lg in ho}

    R = {"evidence_key": {"M": "MEASURED, ours, artifact path in 'sources'", "E": "ESTIMATED (arithmetic on measured inputs, assumption named)",
                          "I": "INHERITED, not re-verified", "U": "UNVERIFIED / not recoverable"}}
    R["sources"] = {
        "bank": {"path": a.bank, "md5": md5_of(a.bank), "note": "pod-side md5 4fc979939c5f9b5140ba59a952da77fb per MD5SUMS.txt"},
        "grow": {"path": a.grow, "md5": md5_of(a.grow)},
        "navtrain_yaml_sha256": sha256_of(os.path.join(a.splits, "navtrain.yaml")),
        "sft_heldout_logs_sha256": sha256_of(os.path.join(a.splits, "sft_heldout_logs.txt")),
        "navtest_tokens": a.navtest, "final_metrics": os.path.join(a.final, "metrics.jsonl")}
    R["navtrain"] = {
        "yaml_tokens": len(y_toks), "yaml_tokens_unique": len(yset), "yaml_logs": len(y_logs),
        "yaml_tokens_in_navtest": len(yset & nt_toks), "navtest_tokens": len(nt_toks), "navtest_logs": len(nt_logs),
        "yaml_logs_in_navtest": len(set(y_logs) & nt_logs) if nt_logs else None,
        "heldout_logs": len(ho), "heldout_logs_in_yaml": len(ho & set(y_logs)),
        "target_bank_rows": len(tb), "target_bank_rank0_tokens": len(tb_toks), "target_bank_tokens_in_yaml": len(tb_toks & yset),
        "yaml_tokens_not_in_target_bank": len(yset - tb_toks), "evidence": "M"}
    R["navtrain"]["tokens_with_known_log"] = len(yset & set(tok_log))
    R["navtrain"]["tokens_in_heldout_logs"] = len(yset & ho_toks_map)
    pop = yset - nt_toks - ho_toks_map
    R["navtrain"]["population_tokens_minus_navtest_minus_heldout_logs"] = len(pop)
    R["navtrain"]["population_note"] = ("the 251 yaml tokens absent from the target bank have no recoverable log (navtrain.yaml pairs no token with a log); "
                                        "they are counted in the population, so the heldout-log subtraction is a lower bound by at most that many")

    # ---- the bank
    R["bank"] = {"rows": len(rows), "served_keys": len(best), "served_train_sets": len(served_train), "served_in_heldout_logs": len(served_ho),
                 "reproduces_p2r_carve": {"177836": len(best) == 177836, "174912": len(served_train) == 174912, "2924": len(served_ho) == 2924},
                 "label_versions_of_served": dict(collections.Counter(v[0][1] for v in best.values())),
                 "served_by_rank": dict(collections.Counter(k[3] for k in served_train)),
                 "served_tokens_train": len(toks_tr), "served_tokens_all": len(toks_served), "served_tokens_heldout_logs": len(toks_ho_served),
                 "served_train_tokens_in_yaml": len(toks_tr & yset), "served_train_logs": len({k[0] for k in served_train}),
                 "sets_per_token_train": dict(collections.Counter(collections.Counter(k[1] for k in served_train).values())),
                 "ckpt_step_of_served_train_top": collections.Counter(v[0][0] for v in served_train.values()).most_common(6),
                 "evidence": "M"}
    # tokens with NO served set at all (the 'needs a fresh dump in either variant' class)
    no_set = yset - toks_served
    no_set_nt = no_set - nt_toks - ho_toks_map          # eligible: outside navtest and the held-out logs
    R["no_onpolicy_set"] = {"tokens_in_yaml_with_no_served_set_at_all": len(no_set),
                            "of_which_not_in_target_bank_(no_recoverable_log)": len(no_set - tb_toks),
                            "of_which_in_target_bank_(rank0_row_but_no_set)": len(no_set & tb_toks),
                            "of_which_in_heldout_logs": len(no_set & ho_toks_map),
                            "target_bank_rows_without_a_served_set": len(tb - set(best)),
                            "logs_of_those_tokens_(known_log_only)": len({tok_log[t] for t in no_set if t in tok_log}), "evidence": "M"}

    # ---- main run: the shared L0 scorer (model_final) -- what it could have trained on
    ep = epoch_bank_sizes(a.final)
    model = []
    for e in ep:
        n_av = sum(1 for k, v in first_at.items() if v <= e["at"])
        model.append({**e, "keys_first_labelled_by_then": n_av})
    last = ep[-1]
    seen_after = {k for k, v in first_at.items() if v > last["at"]}
    tok_first = collections.defaultdict(lambda: "9999")
    for k, v in first_at.items():
        if k[0] not in ho:
            tok_first[k[1]] = min(tok_first[k[1]], v)
    toks_after_main = {t for t in toks_tr if tok_first[t] > last["at"]}
    R["main_run_scorer_exposure_L0"] = {
        "shared_by_both_bases": True,
        "fixed_bank_keys_with_scorer_targets": len(fixed), "target_bank_keys": len(tb),
        "fixed_bank_keys_in_target_bank": len(fixed & tb),
        "main_run_switch": "step 3,708 (epoch 11->12): scorer-mode fixed -> onpolicy (train.log l.3807-3808, DECLARED RECIPE CHANGE)",
        "per_epoch_bank_vs_my_first_label_time": model,
        "validation": "keys_first_labelled_by_then is MY reconstruction from each row's own 'at'; onpolicy_sets is the trainer's own count at that epoch boundary",
        "last_bank_reload_at": last["at"], "main_run_finished_at": MAIN_RUN_END_AT,
        "served_train_tokens_whose_every_set_was_first_labelled_after_the_last_reload": len(toks_after_main),
        "served_keys_first_labelled_after_the_last_reload": len(seen_after), "evidence": "M"}

    # ---- variant B: P2 = 3 epochs over scenes, pdm-only, ALL 174,912 served training sets
    p2_unseen = pop - toks_tr
    p2_unseen_logs = {tok_log[t] for t in p2_unseen if t in tok_log}
    R["variant_B_P2"] = {
        "trained_sets": len(served_train), "trained_tokens": len(toks_tr), "trained_logs": len({k[0] for k in served_train}),
        "basis": "PREREG_P2.md l.41,47 (174,912 served sets, 3 epochs over scenes) and p2r_carve.py; the list is the banked bank minus the 24 held-out logs",
        "unseen_tokens": len(p2_unseen), "unseen_tokens_with_known_log": len([t for t in p2_unseen if t in tok_log]),
        "unseen_tokens_log_unknown": len([t for t in p2_unseen if t not in tok_log]), "unseen_logs_known": len(p2_unseen_logs),
        "unseen_logs_with_zero_trained_tokens": len({lg for lg in p2_unseen_logs if lg not in {k[0] for k in served_train}}),
        "vs_review_plan_51000": round(len(p2_unseen) / 51000.0, 4), "evidence": "M (counts) / I (that P2 trains on all 174,912: its own prereg)"}

    # ---- variant A: SFT-4 u400 = update 400 of batch 8 x accum 8 = 25,600 sets drawn without replacement (seed 0)
    m_draw = SFT4_U * SFT4_BATCH * SFT4_ACCUM
    n_lo, n_hi = (SFT4_TOTAL_UPDATES - 1) * 64 + 1, SFT4_TOTAL_UPDATES * 64
    n_mid = (n_lo + n_hi) // 2
    # processing order of the PDM relabel: sorted files, then line order (onpolicy_relabel_pdm.main); labelled prefix = first n_mid train keys
    ordered = sorted(served_train.items(), key=lambda kv: (fo[kv[1][1]], kv[1][2]))
    prefix = [k for k, _ in ordered[:n_mid]]
    tail = [k for k, _ in ordered[n_mid:]]
    s_cnt = collections.Counter(k[1] for k in prefix)
    exp_unseen = sum(hyper_unseen(n_mid, m_draw, s) for s in s_cnt.values()) + len(toks_tr - set(s_cnt))
    lab_toks = set(s_cnt)
    tail_only = toks_tr - lab_toks
    R["variant_A_u400"] = {
        "stage": "SFT-4 arm B update 400 (model_sft4_B_u400.pt, md5 8e382eeb...)",
        "sets_drawn_by_u400": m_draw, "basis": "400 updates x batch 8 x accum 8, 1 epoch shuffle=True without replacement, torch.Generator seed 0 (scorer_finetune.py l.488-489)",
        "pool_size_n_range": [n_lo, n_hi], "pool_size_basis": "ceil(n/64) = 2,335 total updates (STOP_CONSEQUENCES_2026-10-11.md), so 149,377 <= n <= 149,440; "
                                                           "PDM labels at arming were 146,185 of 174,912 (PREREG_SFT4 Amendment 4), the rest written before launch",
        "exact_draw_recoverable": False,
        "why_not": "the permutation depends on the exact ordered tr_idx at launch (n and which sets carried a PDM label that minute); the SFT-4 `data` event and "
                   "sft.log were never banked (STOP_CONSEQUENCES), and train_pdm lived only on the pod",
        "tokens_certainly_unseen_lower_bound": len(toks_tr) - m_draw, "tokens_certainly_unseen_lower_bound_basis": "pigeonhole: 25,600 sets touch at most 25,600 tokens",
        "tokens_touched_upper_bound": min(m_draw, len(toks_tr)),
        "est_processing_order_prefix_n": n_mid, "est_tail_sets_unlabelled": len(tail), "est_tokens_with_all_train_sets_in_tail": len(tail_only),
        "est_expected_unseen_tokens": round(exp_unseen, 1), "est_expected_touched_tokens": round(len(toks_tr) - exp_unseen, 1),
        "est_expected_unseen_fraction_of_train_tokens": round(exp_unseen / len(toks_tr), 4),
        "est_basis": "hypergeometric on the labelled prefix (relabel order = sorted files, then lines; ESTIMATED, not recorded anywhere)",
        "evidence": "M for the draw size and bounds; E for the expectations; the exact list is U"}
    # RL population for A at the SFT-4 level, and the L0 level
    R["pool_by_definition"] = {
        "definitions": {
            "D1_unseen_by_the_fine_tune_stage_only": "tokens the SFT-4 u400 / P2 fine-tune did not draw (the review's wording: 'navtrain scenes the SFT-4 scorer did not train on')",
            "D2_unseen_by_the_whole_scorer_lineage": "D1 AND not in model_final's main-run scorer training (fixed bank epochs 1-11, on-policy sets epochs 12-27)"},
        "A_D1": {"certain_lower_bound": len(toks_tr) - m_draw + len(p2_unseen), "expected_E": round(exp_unseen + len(p2_unseen), 1),
                 "note": "lower bound = pigeonhole unseen tokens of the train sets + the tokens with no train set; the individual tokens are NOT identifiable"},
        "B_D1": {"count": len(p2_unseen), "identifiable": True, "note": "P2 sees every served training token for 3 epochs"},
        "D2_upper_bound_both_bases": {
            "tokens": len(toks_after_main) + len(no_set_nt),
            "parts": {"every_served_train_set_first_labelled_after_the_last_main_run_bank_reload": len(toks_after_main), "no_served_set_at_all": len(no_set_nt)},
            "identifiable": True,
            "note": "an UPPER bound on tokens the main run's scorer never trained on in its ON-POLICY phase (epochs 12-27). The fixed bank of epochs 1-11 holds "
                    "scorer targets for 189,858 of 189,888 target-bank keys and 120,264 scorer frames were in the epoch-10 snapshot, so the true D2 count is lower "
                    "by an amount this artifact cannot resolve (the snapshot's keys are not banked). It does not depend on the base: it is the shared L0 floor."},
        "review_plan_scenes": 51000,
        "reading": ("B: no pool (about 4 % of the plan, and below the 3,000 scenes RL-val alone needs). "
                    "A: a pool of at least 75,983 tokens exists in count under D1, but its members cannot be named, because the u400 draw is not recoverable; "
                    "any RL-train/RL-val partition of A is a random partition with ~24 % expected overlap with the 25,600 sets u400 drew. "
                    "Under D2 neither base has a pool near 51,000.")}

    # frozen-fan sets: proposals from the final checkpoint (ckpt_step 10075 = the step where the main run ended)
    ff = {k: v for k, v in served_train.items() if v[0][0] == 10075}
    R["final_checkpoint_sets"] = {"ckpt_step": 10075, "served_train_sets": len(ff), "tokens": len({k[1] for k in ff}), "by_rank": dict(collections.Counter(k[3] for k in ff)),
                                  "first_label_after_main_run_end": sum(1 for k in ff if first_at.get(k, "") > MAIN_RUN_END_AT),
                                  "rank0_plain_view_sets": sum(1 for k in ff if k[3] == 0),
                                  "note": "sets whose 64 proposals came from the step-10,075 checkpoint = the fan RL-S1 freezes. ckpt_last@10075 == model_final.pt: MEASURED, "
                                          "all 276 trainable tensors bit-identical and equal frozen-trunk fingerprints (raw/rl_pool/ckpt_last_vs_model_final.json, eval/rl_ckpt_equality.py). "
                                          "Rank-1 sets are route/goal-augmented views; RL scenes use rank 0. Their PDM labels lived only on the pod", "evidence": "M"}

    # ---- artifacts
    os.makedirs(a.out, exist_ok=True)
    with gz_text(os.path.join(a.out, "served_keys.tsv.gz")) as fo_:
        # the log is dropped: a navtrain token belongs to exactly one log (join on token via navtrain_token_table.tsv.gz)
        fo_.write("token\tstep\trank\tckpt_step\tlabel_version\tfirst_label_at\theldout_log\torder_pos\n")
        pos = {k: i for i, (k, _) in enumerate(ordered)}
        for k, v in sorted(best.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][3])):
            fo_.write("\t".join([k[1], str(k[2]), str(k[3]), str(v[0][0]), str(v[0][1]), first_at.get(k, ""), str(int(k[0] in ho)),
                                 str(pos.get(k, ""))]) + "\n")
    with gz_text(os.path.join(a.out, "navtrain_token_table.tsv.gz")) as fo_:
        fo_.write("token\tlog\tin_navtest\theldout_log\tserved_train_sets\tserved_heldout_sets\thas_final_ckpt_set\tfirst_label_at\tprefix_labelled_sets_est\n")
        per_tr = collections.Counter(k[1] for k in served_train)
        per_ho = collections.Counter(k[1] for k in served_ho)
        ff_t = {k[1] for k in ff}
        for t in sorted(yset | toks_served, key=lambda x: (tok_log.get(x, "~"), x)):         # sorted by log: the repeated log names compress away
            fo_.write("\t".join([t, tok_log.get(t, ""), str(int(t in nt_toks)), str(int(tok_log.get(t, "") in ho)), str(per_tr.get(t, 0)),
                                 str(per_ho.get(t, 0)), str(int(t in ff_t)), tok_first[t] if t in tok_first else "", str(s_cnt.get(t, 0))]) + "\n")
    with open(os.path.join(a.out, "rl_pool_by_base.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1)
    return R


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--grow", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--splits", default=os.path.join(PKG, "splits"))
    ap.add_argument("--navtest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R = analyse(a)
    print(json.dumps({k: R[k] for k in ("navtrain", "bank", "no_onpolicy_set", "variant_B_P2", "variant_A_u400", "pool_by_definition")}, indent=1)[:6000])
    print("ZZRLPOOL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
