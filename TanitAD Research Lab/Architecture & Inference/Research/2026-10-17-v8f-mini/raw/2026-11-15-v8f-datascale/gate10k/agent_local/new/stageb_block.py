# ------------------------------------------------------------------------------------------------ stage B: the bank (C-BANK) and the trainer's own stage-B G-LIVE (G-DAGGER)  [gate patch 2026-11-17]
BANK_SHARDS = 4                       # MM ruling 2026-11-17: every bank is built as the recipe of record builds banks: 4 shards, the 4 processes at the same time, then merge


def bank_problems(bank_dir, init_ckpt, dry) -> list:
    """C-BANK: the stage-B bank is complete (not PARTIAL), has n_shards == 4 with every shard file re-hashed, covers the arm's TRAIN list (clips and windows of the dry build), carries
    PASSing K-B4 / K-B5, zero K-B0 / K-B1 mismatches in every shard, an eval-141 overlap of 0, coverage >= 0.5, and was generated from this arm's own --init-ckpt (K-B2). The trainer re-checks
    K-B2 / K-B3 / COVERAGE at launch; this is the checker's own read of the same file, so a bank that the trainer would only refuse after the stage-B build is refused at the token."""
    bad = []
    bj, pj = os.path.join(bank_dir or "", "bank.json"), os.path.join(bank_dir or "", "PROVENANCE.json")
    if not (os.path.isfile(bj) and os.path.isfile(pj)):
        return [f"no bank.json / PROVENANCE.json under {bank_dir!r}: the bank was not built (or a control failed at the merge)"]
    b, p = json.load(open(bj, encoding="utf-8")), json.load(open(pj, encoding="utf-8"))
    if b.get("n_shards") != BANK_SHARDS:
        bad.append(f"bank n_shards {b.get('n_shards')} != {BANK_SHARDS} (the recipe of record's bank build)")
    if b.get("partial") or (b.get("common") or {}).get("partial"):
        bad.append("the bank is PARTIAL (--limit-clips): it may not train an arm")
    com = b.get("common") or {}
    tr = com.get("train") or {}
    if dry:
        if tr.get("n_clips") != dry.get("n_clips") or tr.get("n_windows") != dry.get("n_windows"):
            bad.append(f"bank TRAIN list {tr.get('n_clips')} clips / {tr.get('n_windows')} windows != the dry build's {dry.get('n_clips')} / {dry.get('n_windows')}")
        if tr.get("train_sha256") != dry.get("train_sha256"):
            bad.append("bank train_sha256 differs from the dry build's")
    else:
        bad.append("no dry record to compare the bank's TRAIN list with")
    if (com.get("eval") or {}).get("overlap") != 0 or (com.get("eval") or {}).get("n_clips") != 141:
        bad.append(f"bank eval-141 overlap {(com.get('eval') or {}).get('overlap')}, n_clips {(com.get('eval') or {}).get('n_clips')}")
    cov = b.get("coverage") or {}
    if cov.get("blocked") or not (cov.get("fraction", 0.0) >= 0.5):
        bad.append(f"bank COVERAGE {cov.get('fraction')} < 0.5 (BLOCKED)")
    if init_ckpt and os.path.isfile(init_ckpt) and com.get("ckpt_sha256") != sha_file(init_ckpt):
        bad.append("K-B2: the bank's generating checkpoint is not this arm's --init-ckpt")
    for sh in b.get("shards") or []:
        f = os.path.join(bank_dir, sh.get("file", ""))
        if not os.path.isfile(f) or sha_file(f) != sh.get("file_sha256"):
            bad.append(f"shard {sh.get('k')}: file absent or its sha256 differs from bank.json")
    if len(b.get("shards") or []) != BANK_SHARDS:
        bad.append(f"bank lists {len(b.get('shards') or [])} shards")
    ctl = p.get("controls") or {}
    for k in ("K-B4", "K-B5"):
        if not (ctl.get(k) or {}).get("PASS"):
            bad.append(f"control {k} does not read PASS in PROVENANCE.json")
    for i in range(BANK_SHARDS):
        sp = os.path.join(bank_dir, f"shard_{i:03d}of{BANK_SHARDS:03d}.provenance.json")
        if not os.path.isfile(sp):
            bad.append(f"shard {i}: no provenance file")
            continue
        s = json.load(open(sp, encoding="utf-8"))
        if (s.get("kb0_tick0_identity") or {}).get("mismatch") != 0:
            bad.append(f"shard {i}: K-B0 mismatches {(s.get('kb0_tick0_identity') or {}).get('mismatch')}")
        if (s.get("kb1_replay") or {}).get("mismatch_chunks") != 0:
            bad.append(f"shard {i}: K-B1 mismatched chunks {(s.get('kb1_replay') or {}).get('mismatch_chunks')}")
    if p.get("bank_sha256") != sha_file(bj):
        bad.append("PROVENANCE.json bank_sha256 != the sha256 of bank.json")
    return bad


def dagger_problems(sc: dict, toks) -> list:
    """G-DAGGER: the trainer's own ``--selfcheck-only`` record of the STAGE-B argv: declared_vs_built PASS; ``acc_probe`` PASS with the declared w_acc and a non-zero head_o gradient; ``dagger_probe`` active
    and PASS with a non-zero loss / gradient difference and every control EXACTLY 0.0 (the p = 0 identity, the repeat, the frozen prior, the world-model auxiliary); the bank controls it read (K-B2 / K-B3 /
    coverage) present and not blocked. Deliberate regression: a swap whose ``apply`` is a no-op makes the trainer's probe raise (the selfcheck then holds no passing record) -> RED here."""
    bad = []
    if sc.get("declared_vs_built") != "PASS":
        bad.append(f"declared_vs_built reads {sc.get('declared_vs_built')!r}")
    ap, dp = sc.get("acc_probe") or {}, sc.get("dagger_probe") or {}
    if not ap.get("pass"):
        bad.append("acc_probe does not read pass")
    elif not (ap.get("head_o_grad_diff", 0.0) > 0.0 and abs(float(ap.get("w_acc_declared", -1)) - 49.6969) < 1e-9):
        bad.append(f"acc_probe: w_acc {ap.get('w_acc_declared')}, head_o_grad_diff {ap.get('head_o_grad_diff')}")
    if dp.get("dagger") == "off" or not dp.get("pass") or "skipped" in dp:
        bad.append(f"dagger_probe is not active and passing: {str(dp)[:200]}")
    else:
        if not (dp.get("loss_diff", 0.0) != 0.0 and dp.get("grad_diff_all", 0.0) > 0.0):
            bad.append(f"dagger_probe: loss_diff {dp.get('loss_diff')}, grad_diff_all {dp.get('grad_diff_all')}: the swap does not reach the loss")
        for k in ("control_repeat_loss_diff", "control_repeat_grad_diff", "control_p0_loss_diff", "control_p0_grad_diff", "control_prior_grad_diff"):
            if dp.get(k) != 0.0:
                bad.append(f"dagger_probe control {k} reads {dp.get(k)}, expected exactly 0.0")
        if dp.get("control_wm_grad_diff") not in (0.0, None):
            bad.append(f"dagger_probe control control_wm_grad_diff reads {dp.get('control_wm_grad_diff')}")
        if dp.get("control_p0_same_object") is not True:
            bad.append("dagger_probe: p = 0 did not return the same batch object")
    ctl = (sc.get("info") or {}).get("dagger") or {}
    if not ctl or (ctl.get("coverage") or {}).get("blocked") or not (ctl.get("coverage") or {}).get("fraction", 0) >= 0.5:
        bad.append("the selfcheck carries no bank controls, or the bank coverage is blocked")
    pa = parse_argv(toks).get("--dagger-p")
    if not pa or not float(pa[0]) > 0:
        bad.append("the stage-B argv has no --dagger-p > 0")
    return bad

