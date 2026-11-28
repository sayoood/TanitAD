# ================================================================================================ P6-HN-S: the offset hard-negative term + the frozen-trunk context cache (ALL default OFF)
# (raw/2026-10-18-p6hn-prereg/PREREG_P6HN_SCREEN.md 2.4, 3, 4.3, 4.4, 5 + amendment B; SCREEN_NOTES.md checklist item 3.)
#   --hn-bank PATH   arm B ONLY gains L_hn = hn_bank.hn_term (the MASKED TWO-CALL term: the fan call is this trainer's own, unchanged; a second call over [fan ; hn] keeps only the hard-negative rows; weight 1/(B*K*6) per element).
#   --ctx-cache DIR  the frozen trunk's visual_ctx is SERVED from the cache (ctx_cache.CtxCache) instead of recomputed: arm A through model.score_trajectories (the function REFe.forward itself calls), arm B and the base through
#                    their own scorer calls, on the SAME cached tensor, so the arms stay paired by construction.
# Neither flag named = this trainer exactly (refe/test_hn_screen_wiring.py I3 runs the real main() of the pre-edit source and of this one and compares every row and tensor bit for bit).
# HOW IT IS WIRED WITHOUT TOUCHING THE LOOP: main() REBINDS `fwd`, `train_losses` and `log` (the originals stay byte-identical, so every pin on them holds) and wraps the datasets in ScreenItems, which appends the cached context and the
# bank's rows to each item AFTER the stock nine elements.  The ON path is therefore: argparse -> a.hn_bank -> hn_obj (HN.load_bank, refused on another partition / label_version / fan fingerprint / changed bytes) -> ScreenItems
# (hn_xy [B, 8, T, 3], hn_tg [B, 8, 6], hn_m [B, 8], the bank row) -> fwd() (stashes the tail) -> train_losses() (HN.hn_term on arm B's scorer) -> lB -> the loop's own backward().
HN_DESTS = ("hn_bank", "ctx_cache")
HN_ARM = "B"
HN_PARTITION_ID = "9995a39aeeac3833dd3f679b75e883b056e3d4d235bb9dc986578a81a7a256fa"          # the frozen RL-S1 n16000 partition (refe/rls1_checks.py PARTITION, TL-1): the only partition the screen is registered on
HN_LABEL_VERSION = 5                                                                           # the fan labels the screen trains on (PREREG 2.3); a bank of another version is refused
HN_PROBE_SCENES = 64                                                                           # G15: scenes per side (RL-train, RL-val) compared live vs cached (PREREG 4.4)


class HNEnv:
    """the handles the G14 / G15 probes need (they live in main()'s closure); plain attributes"""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class ScreenItems(torch.utils.data.Dataset):
    """Wraps a TargetBank-like dataset.  Item = the stock tuple (nine elements) + (cached visual_ctx bf16 [P, D])? + (hn_xy [8, T, 3], hn_tg [8, 6], hn_m [8], bank row)?  -- the cache part iff `cache`, the bank part iff `bank`
    (the TRAINING items only: the held-out wrapper gets no bank, so evaluation can never read a hard negative, gate I4).  `rows` / `onpolicy` pass through, so the trainer's own index arithmetic is unchanged."""

    def __init__(self, base, cache=None, bank=None, ckpt_of=None):
        self.base, self.cache, self.bank, self.ckpt_of = base, cache, bank, (ckpt_of or {})

    @property
    def rows(self):
        return self.base.rows

    @property
    def onpolicy(self):
        return self.base.onpolicy

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = tuple(self.base[i])
        k = key(self.base.rows[i])
        if self.cache is not None:
            item = item + (self.cache.get_ctx([k])[0],)
        if self.bank is not None:
            b = self.bank.batch([k], [self.ckpt_of[k]])
            item = item + (b["hn_xy"][0], b["hn_tg"][0], b["hn_m"][0], torch.tensor(self.bank.index[k], dtype=torch.long))
        return item


def hn_refusals(a, ap):
    """Every reason --hn-bank / --ctx-cache must not start, as messages ([] = fine).  PREREG_P6HN_SCREEN 4.4: --hn-bank with --epo, with --lam != 0 or with --aux-drv; (the bank of 0 hard-negative rows is refused where the
    bank is loaded).  Beyond the register: arm B's OTHER loss must be plain BCE (--b-mode listnet), or B - A would not be the bank alone.  --ctx-cache is not wired for --aux-drv (the head reads the live capture)."""
    bad = []
    if a.hn_bank:
        if a.epo:
            bad.append("--hn-bank with --epo: arm B would carry two extra terms, and the screen's contrast B - A is the bank alone")
        if a.lam != 0.0:
            bad.append("--hn-bank needs --lam 0: arm B's other loss must be plain BCE (a ListNet term would make B - A more than the bank)")
        if a.b_mode != "listnet":
            bad.append(f"--hn-bank needs --b-mode listnet (with --lam 0 that is plain BCE); the argv has --b-mode {a.b_mode}")
        if a.aux_drv:
            bad.append("--hn-bank with --aux-drv: not wired (the head reads the live capture of the scoring context)")
    if a.ctx_cache and a.aux_drv:
        bad.append("--ctx-cache with --aux-drv: not wired (the head reads the live capture of the scoring context, which a cache does not produce)")
    return bad


def hn_head_rows(armB):
    """(sum |grad| of score_head rows 0-5, of rows 6+ = the lane output), weights and biases together; a head without a gradient reads 0.0"""
    h = armB.score_head
    if h.weight.grad is None:
        return 0.0, 0.0
    w, b = h.weight.grad.detach().abs().double(), h.bias.grad.detach().abs().double()
    return float(w[:6].sum() + b[:6].sum()), float(w[6:].sum() + b[6:].sum())


def hn_probe_batch(env):
    """a REAL training batch of --batch scenes with the cached context and the bank's rows attached, scenes that HAVE a hard negative first (G14 needs n_hn > 0)"""
    bank = env.bank
    any_hn = bank.hn_mask().any(1)
    first, rest = [], []
    for i in env.tr_idx:
        (first if bool(any_hn[bank.index[key(env.ds.rows[i])]]) else rest).append(i)
    sel = (first + rest)[:env.a.batch]
    return torch.utils.data.default_collate([env.ds[i] for i in sel])


def g14_hn_live_and_identity(env):
    """G14 (P6-HN-S preflight), one dict of RAW FIELDS that the launch gate re-judges (gate_checks.judge_g14): it never trusts this `ok`.  Every probe runs on a REAL batch with n_hn > 0 through the RUN'S OWN assembly
    (`env.train_losses`, the rebound function the loop calls): the term alone is the DIFFERENCE of that assembly and the stock one on one fresh forward, so deleting the line that adds the term leaves it identically zero.
      grad table   the term alone: finite and non-zero on arm B's score_q_mlp, score_dec, score_head rows 0-5; EXACTLY 0.0 on the lane row (the control that proves the probe can read zero), on every arm-A parameter and on the base;
                   the trunk / generator have requires_grad False (G1); arm A's own BCE gradient in the same probe reads non-zero
      controls     the bank OFF and the bank EMPTY (all-False mask): the term and every gradient read EXACTLY 0.0
      two steps    two AdamW steps of the run's assembly with the bank off / empty leave arm B bit-identical to arm A; with the bank on they differ; the probe is shown able to read one ulp
      analytic     (amendment B2) on a copy of arm B with set-level self-attention NEUTRALISED, a bank row that duplicates fan row j adds exactly row j's share of the fan BCE
      pad          the served mask equals the bank's own mask for the served scenes, and junk (NaN) written into the bank's pad slots leaves the term bit-identical
      fan / rows   the fan rows of the masked call equal the one-call fan logits bit for bit; a hard-negative row's logits do not depend on the other hard-negative rows
      D1           the rows that entered the term = the rows the bank holds for the served scenes; evaluation items carry no bank tail"""
    import hn_bank as HN
    a, model, armB, cur, bank = env.a, env.model, env.armB, env.cur, env.bank
    hb = hn_probe_batch(env)
    g: dict = {"mutate": [], "scorer": list(SCORER), "lane_head": bool(a.lane_head)}

    def fresh():
        s0, sA, sB, tg = env.fwd(hb)
        return s0, sA, sB, tg, tuple(cur["tail"])

    def zero():
        armB.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

    def probe(mode):
        s0, sA, sB, tg, tail = fresh()
        keep_tail, keep_off = cur["tail"], cur["off"]
        try:
            if mode == "empty":
                cur["tail"] = (tail[0], tail[1], torch.zeros_like(tail[2]), tail[3])
            cur["off"] = mode == "off"
            ref = env.train_losses_stock(s0, sA, sB, tg, None, 0)
            on = env.train_losses(s0, sA, sB, tg, None, 0)
        finally:
            cur["tail"], cur["off"] = keep_tail, keep_off
        return on[1] - ref[1], on[0]

    vals, rb, rh, ra, base_g, frozen_g, a_ctrl = {}, {}, {}, {}, {}, {}, None
    for mode in ("on", "off", "empty"):
        zero()
        term, loss_a = probe(mode)
        vals[mode] = float(term.detach())
        if bool(term.requires_grad):
            term.backward(retain_graph=True)
        rb[mode], rh[mode], ra[mode] = module_grad_abs(armB, SCORER), hn_head_rows(armB), module_grad_abs(model, SCORER)
        base_g[mode] = sum(1 for p in env.base.parameters() if p.grad is not None)
        frozen_g[mode] = sum(1 for n_, p in model.named_parameters() if n_.split(".")[0] not in SCORER and p.grad is not None)
        if mode == "on":
            model.zero_grad(set_to_none=True)
            if loss_a.requires_grad:
                loss_a.backward()
            a_ctrl = module_grad_abs(model, SCORER)
    zero()
    on = rb["on"]
    g["n_hn"] = int(cur["tail"][2].sum()) if len(cur["tail"]) == 4 else 0
    g["grad_abs_sum"] = {"score_q_mlp": on["score_q_mlp"], "score_dec": on["score_dec"], "score_head_rows_0_5": rh["on"][0], "score_head_row_6_lane": rh["on"][1],
                         "armA_all": float(sum(ra["on"].values())), "base_params_with_grad": base_g["on"], "frozen_params_with_grad": frozen_g["on"]}
    g["grads_finite"] = bool(all(math.isfinite(v) for v in list(on.values()) + list(rh["on"])))
    g["A_bce_control"] = a_ctrl
    g["value_on"] = vals["on"]
    g["controls"] = {m: {"value": vals[m], "B_all": float(sum(rb[m].values()) + sum(rh[m])), "A_all": float(sum(ra[m].values()))} for m in ("off", "empty")}
    g["trunk_requires_grad"] = bool(any(p.requires_grad for n_, p in model.named_parameters() if n_.split(".")[0] not in SCORER))

    # ---- two optimiser steps (the run's own assembly): B == A with the bank off / empty, B != A with it on, and the probe can read one ulp
    def two_steps(mode):
        sdA = {m: copy.deepcopy(getattr(model, m).state_dict()) for m in SCORER}
        sdB = {m: copy.deepcopy(getattr(armB, m).state_dict()) for m in SCORER}
        pA, pB = [p for m in SCORER for p in getattr(model, m).parameters()], list(armB.parameters())
        oA, oB = torch.optim.AdamW(pA, lr=a.lr, weight_decay=0.01), torch.optim.AdamW(pB, lr=a.lr, weight_decay=0.01)
        was_det, was_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        det_strict = {"warn_only": False}                    # G12's setting, spelt apart so the G12 warn-only mutation text (test_aux_g12_bf16 WARN) stays ONE site in this file
        torch.use_deterministic_algorithms(True, **det_strict)
        try:
            for _ in range(2):
                s0, sA, sB, tg, tail = fresh()
                keep_tail, keep_off = cur["tail"], cur["off"]
                try:
                    if mode == "empty":
                        cur["tail"] = (tail[0], tail[1], torch.zeros_like(tail[2]), tail[3])
                    cur["off"] = mode == "off"
                    lA, lB, _ln, _ep = env.train_losses(s0, sA, sB, tg, None, 0)
                finally:
                    cur["tail"], cur["off"] = keep_tail, keep_off
                (lA + lB).backward()
                for o, ps in ((oA, pA), (oB, pB)):
                    torch.nn.utils.clip_grad_norm_(ps, 1.0)
                    o.step()
                    o.zero_grad(set_to_none=True)
            same = bool(all(torch.equal(x, y) for x, y in zip(pA, pB)))
            ulp = [p.detach().clone() for p in pA]
            z0 = ulp[0].view(-1)
            z0[0] = torch.nextafter(z0[0], torch.tensor(float("inf"), dtype=z0.dtype, device=z0.device))
            reads_ulp = bool(not all(torch.equal(x, y) for x, y in zip(ulp, pA)))
        finally:
            torch.use_deterministic_algorithms(was_det, warn_only=was_warn)
            for m in SCORER:
                getattr(model, m).load_state_dict(sdA[m])
                getattr(armB, m).load_state_dict(sdB[m])
            zero()
        return same, reads_ulp
    off_same, ulp_ok = two_steps("off")
    empty_same, _ = two_steps("empty")
    on_same, _ = two_steps("on")
    g["two_step"] = {"off_equal_A": off_same, "empty_equal_A": empty_same, "on_differs_from_A": bool(not on_same), "probe_reads_one_ulp": ulp_ok}

    # ---- the pieces below read the term directly (fp32 where the identity is exact), on the batch the last fresh() served
    s0, sA, sB, tg, tail = fresh()
    cxy, sctx = cur["cxy"], cur["sctx"]
    cxy32, sctx32 = cxy.float(), sctx.float()          # the direct probes below run in fp32 with NO autocast (the cached context is bf16; their identities are exact or 1e-5 in fp32); the run's own term runs under the trainer's autocast
    hxy, htg, hm, hrow = (x.to(env.dev) for x in tail)
    B_, K_ = int(cxy.shape[0]), int(cxy.shape[1])
    truth = torch.as_tensor(bank.hn_mask()[hrow.cpu().numpy()], device=env.dev)                  # the bank's OWN mask for the served scenes
    g["d1"] = {"rows_in_term": int(hm.sum()), "rows_expected": int(truth.sum()), "batch": B_, "K": K_}
    g["pad"] = {"served_mask_equals_bank": bool(torch.equal(hm.bool(), truth)), "pad_slots": int((~truth).sum())}
    with torch.no_grad():
        clean = HN.hn_term(armB, cxy32, hxy, htg, hm, sctx32).loss
        junk_xy, junk_tg = hxy.clone(), htg.clone()
        junk_xy[~truth] = float("nan")
        junk_tg[~truth] = 7.5
        junk = HN.hn_term(armB, cxy32, junk_xy, junk_tg, hm, sctx32).loss
        g["pad"]["perturbed_loss_equal"] = bool(torch.isfinite(junk) and torch.equal(clean, junk))
        # (1) B's fan logits AS THE LOSS USES THEM (fwd's sxB) vs the one-call path, same shape and same autocast: bit-identical by construction (the term never touches the fan call)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            one_ac = armB(cxy, sctx)
        g["fan_call_max_abs_diff"] = float((sB.float() - one_ac.float()).abs().max())
        # (2) the masked call's fan rows must not see any hn slot: the SAME 72-row call with the hn slots zeroed gives bit-identical fan rows (same shapes, same kernels, so the identity is exact)
        both = torch.cat([cxy32, torch.where(hm.bool()[..., None, None], hxy, torch.zeros_like(hxy))], 1)
        both0 = torch.cat([cxy32, torch.zeros_like(hxy)], 1)
        blocked = HN.hn_attn_mask(K_, int(hxy.shape[1]), device=cxy.device)
        two = HN.score_set(armB, both, sctx32, blocked)[:, :K_]
        two0 = HN.score_set(armB, both0, sctx32, blocked)[:, :K_]
        g["fan_rows_hn_isolation_max_abs"] = float((two.float() - two0.float()).abs().max())
        # (3) INFORMATIONAL (not gated): the masked call's fan rows vs the 64-row one-call -- different GEMM shapes may differ in the last bits, so this is a scale, not an identity
        one32 = armB(cxy32, sctx32)
        g["fan_rows_vs_one_call_max_abs"] = float((two.float() - one32.float()).abs().max())
        g["fan_rows_scale"] = float(one32.float().abs().max())
        # a hard-negative row does not depend on the OTHER hard-negative rows: keep only the first valid row of a scene that has >= 2
        bi = next((b for b in range(B_) if int(hm[b].sum()) >= 2), None)
        if bi is None:
            g["hn_row_independence_max_abs"] = None
        else:
            full = HN.hn_scores(armB, cxy32, hxy, hm.bool(), sctx32)
            h0 = int(torch.nonzero(hm[bi])[0])
            one_m = torch.zeros_like(hm.bool())
            one_m[bi, h0] = True
            alone = HN.hn_scores(armB, cxy32, hxy, one_m, sctx32)
            g["hn_row_independence_max_abs"] = float((full[bi, h0].float() - alone[bi, h0].float()).abs().max())
    # ---- the analytic identity (amendment B2): a neutralised copy of arm B, a bank row that duplicates fan row j
    Bn = copy.deepcopy(armB)
    for blk in Bn.score_dec:
        blk.self_attn.out_proj.weight.data.zero_()
        blk.self_attn.out_proj.bias.data.zero_()
    with torch.no_grad():
        j = min(3, K_ - 1)
        dxy = torch.zeros(B_, 8, cxy32.shape[2], cxy32.shape[3], device=env.dev)
        dtg = torch.zeros(B_, 8, 6, device=env.dev)
        dm = torch.zeros(B_, 8, dtype=torch.bool, device=env.dev)
        dxy[:, 0], dtg[:, 0], dm[:, 0] = cxy32[:, j], tg[:, j, :6].float(), True
        lhn = HN.hn_term(Bn, cxy32, dxy, dtg, dm, sctx32).loss.float()
        fan_logits = Bn(cxy32, sctx32)[:, j, :6].float()
        share = F.binary_cross_entropy_with_logits(fan_logits, tg[:, j, :6].float(), reduction="sum") / float(B_ * K_ * 6)
    g["dup_identity"] = {"loss": float(lhn), "share": float(share), "abs_diff": float((lhn - share).abs()), "j": j}
    # ---- evaluation reads no bank: the held-out items carry nothing but the stock nine (+ the cached context)
    want = 9 + (1 if env.cache is not None else 0)
    sel = env.ho_idx[:3]
    g["eval_reads_bank"] = {"held_items_checked": len(sel), "held_items_with_hn_tail": int(sum(1 for i in sel if len(env.ds_ho[i]) != want))}
    g["ok"] = bool(g["n_hn"] > 0 and g["grads_finite"] and g["pad"]["served_mask_equals_bank"] and g["pad"]["perturbed_loss_equal"] and g["fan_call_max_abs_diff"] == 0.0 and g["fan_rows_hn_isolation_max_abs"] == 0.0
                   and (g["hn_row_independence_max_abs"] in (None, 0.0)) and g["d1"]["rows_in_term"] == g["d1"]["rows_expected"] and g["eval_reads_bank"]["held_items_with_hn_tail"] == 0
                   and all(v > 0.0 for v in (on["score_q_mlp"], on["score_dec"], rh["on"][0])) and rh["on"][1] == 0.0 and g["grad_abs_sum"]["armA_all"] == 0.0
                   and g["grad_abs_sum"]["base_params_with_grad"] == 0 and g["grad_abs_sum"]["frozen_params_with_grad"] == 0 and not g["trunk_requires_grad"]
                   and all(c["value"] == 0.0 and c["B_all"] == 0.0 and c["A_all"] == 0.0 for c in g["controls"].values())
                   and g["two_step"]["off_equal_A"] and g["two_step"]["empty_equal_A"] and g["two_step"]["on_differs_from_A"] and g["two_step"]["probe_reads_one_ulp"]
                   and g["dup_identity"]["abs_diff"] <= 1e-5 * max(1.0, g["dup_identity"]["share"]) and g["dup_identity"]["share"] > 0.0
                   and bool(a_ctrl) and all(math.isfinite(v) and v > 0.0 for v in a_ctrl.values()))
    return g


def g15_ctx_cache_identity(env):
    """G15 (P6-HN-S preflight), RAW FIELDS re-judged by gate_checks.judge_g15.  On HN_PROBE_SCENES RL-train and HN_PROBE_SCENES RL-val scenes the cached context and the base / A / B logits read from the cache equal the LIVE trunk
    forward's bit for bit (micro-batches of --batch, the trainer's own size: a batch-dependent kernel would break bit equality and must show here, not in the run); the SAME comparison with the cache's rows ROLLED across the batch (a
    row-shuffled cache) must read far from it; the manifest binds the checkpoint's trunk fingerprint and the calib file."""
    a, cache = env.a, env.cache
    inf = {"ctx": 0.0, "base": 0.0, "A": 0.0, "B": 0.0}
    shuf = {"ctx": 0.0, "base": 0.0, "A": 0.0, "B": 0.0}
    n_side = {}
    for name, ds_side, idx in (("train", env.ds, env.tr_idx), ("val", env.ds_ho, env.ho_idx)):
        sel = idx[::max(1, len(idx) // HN_PROBE_SCENES)][:HN_PROBE_SCENES]
        n_side[name] = len(sel)
        for s in range(0, len(sel), a.batch):
            batch = torch.utils.data.default_collate([ds_side[i] for i in sel[s:s + a.batch]])
            body, tail = tuple(batch[:9]), tuple(batch[9:])
            with torch.no_grad():
                live = env.fwd_live(body)
                ctx_live = env.cap["sctx"].detach().clone()
                cach = env.fwd(batch)
                ctx_c = tail[0].to(env.dev)
                rolled = env.fwd(body + (tail[0].roll(1, 0),) + tail[1:])
            inf["ctx"] = max(inf["ctx"], float((ctx_c.float() - ctx_live.float()).abs().max()))
            shuf["ctx"] = max(shuf["ctx"], float((tail[0].roll(1, 0).to(env.dev).float() - ctx_live.float()).abs().max()))
            for nm, x, y, z in zip(("base", "A", "B"), live[:3], cach[:3], rolled[:3]):
                inf[nm] = max(inf[nm], float((x - y).abs().max()))
                shuf[nm] = max(shuf[nm], float((x - z).abs().max()))
    calib_run = None
    if a.calib and os.path.isfile(a.calib):
        h = hashlib.sha256()
        with open(a.calib, "rb") as f:
            for b_ in iter(lambda: f.read(1 << 22), b""):
                h.update(b_)
        calib_run = h.hexdigest()
    miss_tr = sum(1 for i in env.tr_idx if key(env.ds.rows[i]) not in cache.index)
    miss_ho = sum(1 for i in env.ho_idx if key(env.ds.rows[i]) not in cache.index)
    g = {"n_train": n_side["train"], "n_val": n_side["val"], "micro_batch": int(a.batch), "ctx_max_abs_diff": inf["ctx"], "logit_max_abs_diff": {k: inf[k] for k in ("base", "A", "B")},
         "shuffled": {"ctx_max_abs_diff": shuf["ctx"], "logit_max_abs_diff": {k: shuf[k] for k in ("base", "A", "B")}},
         "trunk_fingerprint_manifest": cache.man["trunk_fingerprint"], "trunk_fingerprint_checkpoint": env.trunk_fp, "dtype": cache.man["dtype"], "n_scenes_manifest": int(cache.man["n_scenes"]),
         "manifest_sha256": env.cache_manifest_sha256, "covers_run": {"missing_train": miss_tr, "missing_val": miss_ho}, "calib_sha256_manifest": cache.man.get("calib_sha256"), "calib_sha256_run": calib_run,
         "mutate": []}
    g["ok"] = bool(inf["ctx"] == 0.0 and all(inf[k] == 0.0 for k in ("base", "A", "B")) and shuf["ctx"] > 10 * inf["ctx"] and all(shuf[k] > 10 * inf[k] for k in ("base", "A", "B"))
                   and g["trunk_fingerprint_manifest"] == g["trunk_fingerprint_checkpoint"] and miss_tr == 0 and miss_ho == 0 and g["dtype"] == "bf16"
                   and min(n_side.values()) >= min(HN_PROBE_SCENES, min(len(env.tr_idx), len(env.ho_idx))))
    return g


