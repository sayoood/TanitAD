"""Apply the A2 edits to scorer_finetune.py (base d3227241).  usage: python apply_sf.py <path to scorer_finetune.py>   (LF in, LF out; every anchor must match exactly once)"""
import sys
P = sys.argv[1]
s = open(P, encoding="utf-8", newline="").read()


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:70], s.count(old))
    s = s.replace(old, new)


rep('def expected_reward_loss(sx, tg, tau_s, sel=None):', 'def expected_reward_loss(sx, tg, tau_s, sel=None, rfac=None, selfac=None):')
rep('''    p = torch.sigmoid(sx)
    s = torch.log((agg_v1(p) if sel is None else sel(p)).clamp_min(1e-6))
    pi = torch.softmax(s / tau_s, dim=1)
    return -(pi * agg_v1(tg[..., :6])).sum(1).mean()
''', '''    p = torch.sigmoid(sx)
    sc = agg_v1(p) if sel is None else sel(p)
    if selfac is not None:                                   # A2 (--beh-listwise): x prod_k sigmoid(behaviour logit_k)^act_k
        sc = sc * selfac
    s = torch.log(sc.clamp_min(1e-6))
    pi = torch.softmax(s / tau_s, dim=1)
    r = agg_v1(tg[..., :6])
    if rfac is not None:                                     # A2 (--beh-listwise): R_i = agg_v1(labels) * prod_k q_ik^act_ik
        r = r * rfac
    return -(pi * r).sum(1).mean()
''')

A2 = r'''# ================================================================================================ A2 (a2/WIRING_PLAN.md): the five behaviour-term heads.  ALL levers default OFF = this trainer exactly.
#   --beh-heads            grow score_head to 12 rows AFTER load (6 PDM + lane + 5 behaviour; rows 7..11 zero weight, bias = logit of the train label mean).  Arm A never trains rows 7..11 (its BCE reads :6/:7).
#   --w-beh W              arm B ONLY: += W * beh_bce (masked BCE of rows 7..11 on the labelled q, normalised by the ACTIVE entries).  0.0 = exactly zero gradient on rows 7..11.
#   --beh-listwise         arm B ONLY, needs --b-mode expreward --lam > 0: R_i = agg_v1(tg) * prod q^act, selection score x prod sigmoid(beh logit)^act (an inapplicable factor is 1).
#   --beh-listwise-stall   q_stall (col 3) joins the listwise product.  DEFAULT EXCLUDED (Master Mind 2026-11-08: q_stall varies across proposals in 3 of 62 scenes, so it has ~no within-scene ranking signal).
# HOW IT IS WIRED WITHOUT TOUCHING THE STOCK LOSS: main() REBINDS `fwd` and `train_losses` (make_beh_wrappers; the originals stay as they are).  The rebound fwd peels the (bq, bact) tail, stashes the 12-wide logits in
# `cur`, and hands the stock code the stock width (6, or 7 with --lane-head); the rebound train_losses adds the behaviour term to arm B only.  The labels exist ONLY in the loss: nothing in model.py / planner.py / eval/ names them.
BEH_ORDER = ("q_free", "q_gap", "q_osc", "q_stall", "q_comf")      # == behaviour_terms.ORDER (test_a2_wiring pins it)
BEH_ROW0, BEH_N = 7, 5
BEH_HEAD_ROWS = BEH_ROW0 + BEH_N
BEH_GLOB = "behaviour_*.jsonl"                                       # behaviour_label.BEH_PREFIX + "*.jsonl" (pinned by test_a2_wiring)
BEH_TERMS_VERSION = "behaviour/1"
BEH_REWARD_COLS = (0, 1, 2, 4)                                       # q_stall (col 3) excluded by default
BEH_REWARD_COLS_STALL = (0, 1, 2, 3, 4)
BEH_PASS_Q = 1.0                                                     # held-out AUC: a proposal is 'clean' on a term iff q >= 1 (unpenalised); OPEN for PREREG_A2
BEH_DESTS = ("beh_labels_train", "beh_labels_heldout", "w_beh", "beh_listwise", "beh_listwise_stall")


def beh_refusals(a, ap):
    """Every reason the A2 levers must not start ([] = fine)."""
    bad = []
    if not a.beh_heads:
        named = [d for d in BEH_DESTS if getattr(a, d) != ap.get_default(d)]
        return [f"A2 levers {named} are set but --beh-heads is not: they would be silently ignored"] if named else bad
    if not (a.beh_labels_train and a.beh_labels_heldout):
        bad.append("--beh-heads needs --beh-labels-train AND --beh-labels-heldout (a head with no held-out readout cannot be evaluated)")
    if not (a.w_beh >= 0.0 and math.isfinite(a.w_beh)):
        bad.append("--w-beh must be a finite number >= 0")
    if a.w_beh == 0.0 and not a.beh_listwise:
        bad.append("--beh-heads with --w-beh 0 and no --beh-listwise: the heads would be built and never trained on")
    if a.beh_listwise and (a.b_mode != "expreward" or a.lam <= 0.0):
        bad.append("--beh-listwise needs --b-mode expreward and --lam > 0 (it changes the expected-reward term; without it the lever does nothing)")
    if a.beh_listwise_stall and not a.beh_listwise:
        bad.append("--beh-listwise-stall needs --beh-listwise")
    if a.epo or a.aux_drv or a.hn_bank or a.ctx_cache:
        bad.append("A2 is the plain scorer fine-tune (+ L_pair / listwise): --epo / --aux-drv / --hn-bank / --ctx-cache change the batch layout and are not wired with --beh-heads")
    return bad


def extend_beh_head(model, dev, bias):
    """grow the 6- or 7-row score head to 12: rows 0..n-1 copied verbatim, row 6 (when the head had 6) zero weight and bias 1.5 (an unused placeholder), rows 7..11 zero weight and bias = `bias` (the logit of the
    train label mean, so the first BCE is not a calibration transient).  A head that already has 12 rows is left as it is."""
    old = model.score_head
    n = int(old.out_features)
    if n == BEH_HEAD_ROWS:
        return old
    assert n in (6, 7), f"cannot grow a {n}-row score head"
    new = nn.Linear(old.in_features, BEH_HEAD_ROWS).to(dev)
    with torch.no_grad():
        new.weight.zero_(); new.bias.zero_()
        new.weight[:n].copy_(old.weight); new.bias[:n].copy_(old.bias)
        if n == 6:
            new.bias[6] = 1.5
        new.bias[BEH_ROW0:].copy_(torch.as_tensor(bias, dtype=new.bias.dtype))
    model.score_head = new
    for p in model.score_head.parameters():
        p.requires_grad_(True)
    return new


def _beh_record(d, n_prop):
    """one validated behaviour line -> (q [M, 5] float32, act [M, 5] bool) or None"""
    try:
        if d.get("terms_version") != BEH_TERMS_VERSION:
            return None
        q, act = np.asarray(d["q"], np.float32), np.asarray(d["act"]).astype(bool)
        if q.shape != (n_prop, BEH_N) or act.shape != (n_prop, BEH_N) or not np.isfinite(q).all() or q.min() < 0.0 or q.max() > 1.0 + 1e-6:
            return None
    except (KeyError, ValueError, TypeError):
        return None
    return q, act


def load_beh_terms(path, bank_by, n_prop):
    """`<path>/behaviour_*.jsonl` (behaviour_label.py): {"kind":"behaviour","key":[log,token,step,rank],"ckpt_step":c,"label_version":v,"q":[M][5],"act":[M][5],"terms_version":"behaviour/1"}.
    Same staleness rule as load_rl_terms (same key AND ckpt_step).  Returns ({key: (q, act)}, stats); a set with no usable record is simply absent (BehItems then serves act = False)."""
    terms: dict = {}
    st = {"files": 0, "lines": 0, "matched": 0, "stale": 0, "superseded": 0, "incomplete": 0, "not_in_bank": 0, "bad_lines": 0, "duplicates": 0}
    for fp in sorted(glob.glob(os.path.join(path, BEH_GLOB))):
        st["files"] += 1
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    st["bad_lines"] += 1
                    continue
                if not isinstance(d, dict) or d.get("kind") != "behaviour":
                    continue
                st["lines"] += 1
                k = tuple(d.get("key", ()))
                if len(k) != 4:
                    st["incomplete"] += 1
                    continue
                k = (k[0], k[1], int(k[2]), int(k[3]))
                e = bank_by.get(k)
                if e is None:
                    st["not_in_bank"] += 1
                    continue
                ck = int(d.get("ckpt_step", -1))
                if ck != int(e[0]):
                    st["superseded" if ck < int(e[0]) else "stale"] += 1
                    continue
                rec = _beh_record(d, n_prop)
                if rec is None:
                    st["incomplete"] += 1
                    continue
                if k in terms:
                    st["duplicates"] += 1
                terms[k] = rec
    st["matched"] = len(terms)
    return terms, st


def beh_bias_init(terms):
    """logit of the mean of q over the ACTIVE entries of each term (clamped to [0.01, 0.99]); 0.0 for a term with no active entry"""
    out = []
    for c in range(BEH_N):
        num = den = 0.0
        for q, act in terms.values():
            num += float(q[:, c][act[:, c]].sum()); den += float(act[:, c].sum())
        m = min(max(num / den, 0.01), 0.99) if den > 0 else 0.5
        out.append(math.log(m / (1.0 - m)))
    return out


class BehItems(torch.utils.data.Dataset):
    """Wraps a TargetBank-like dataset.  Item = the stock nine elements + (bq [M, 5] float32, bact [M, 5] bool) looked up by key(row).  A set with no record gets act = False everywhere (a mask, never a
    fabricated 1).  `rows` / `onpolicy` pass through, so the trainer's index arithmetic is unchanged."""

    def __init__(self, base, terms):
        self.base, self.terms = base, terms

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
        rec = self.terms.get(key(self.base.rows[i]))
        m = int(item[4].shape[0])
        q, act = (rec if rec is not None else (np.ones((m, BEH_N), np.float32), np.zeros((m, BEH_N), bool)))
        return item + (torch.from_numpy(np.ascontiguousarray(q)), torch.from_numpy(np.ascontiguousarray(act)))


def beh_bce(sx_beh, bq, bact):
    """BCE of the behaviour logits [B, M, 5] on the labelled q, over the ACTIVE entries only, divided by their number (a 92 %-inactive term must not be diluted: D-RL-COLL-SPARSE-1).  No active entry -> 0.0 WITH graph."""
    l = F.binary_cross_entropy_with_logits(sx_beh, bq, reduction="none")
    return torch.where(bact, l, torch.zeros_like(l)).sum() / bact.sum().clamp_min(1).to(l.dtype)


def beh_factors(bq, bact, sx_beh, cols):
    """(rfac [B, M] detached = prod q^act, selfac [B, M] = prod sigmoid(logit)^act) over `cols`; an inapplicable factor is exactly 1.0"""
    idx = list(cols)
    act, q = bact[..., idx], bq[..., idx]
    one = torch.ones_like(q)
    return torch.where(act, q, one).prod(-1).detach(), torch.where(act, torch.sigmoid(sx_beh[..., idx]), one).prod(-1)


def beh_heldout_auc(pred, bq, bact, logs):
    """per-term held-out within-set AUC of the head's logit ranking CLEAN (q >= BEH_PASS_Q) above penalised proposals, over the ACTIVE entries of each set; sets with one class are skipped.
    -> {term: {"auc": mean over sets, "sets": n, "by_log": {log: mean}}}.  pred/bq/bact: [N, M, 5] CPU tensors, logs: N names."""
    out = {}
    for c, nm in enumerate(BEH_ORDER):
        by, vals = {}, []
        for i in range(pred.shape[0]):
            m = bact[i, :, c]
            if int(m.sum()) < 2:
                continue
            v = set_auc(pred[i, :, c][m], (bq[i, :, c][m] >= BEH_PASS_Q - 1e-6).float())
            if v is not None:
                vals.append(v); by.setdefault(logs[i], []).append(v)
        out[nm] = {"auc": float(np.mean(vals)) if vals else None, "sets": len(vals), "by_log": {k: float(np.mean(v)) for k, v in sorted(by.items())}}
    return out


def make_beh_wrappers(a, cur, fwd_stock, train_losses_stock, dev):
    """-> (fwd, train_losses): the rebound pair of an --beh-heads run.  `cur` is the run's state dict: cur['w'] (the weight the loss multiplies), cur['listwise'], cur['cols'] (the columns of the listwise product) are what
    the loss READS (the data event reports them, G-DVB compares them with the argv); cur['collect'] (a list) makes fwd() record held-out head outputs."""
    ncol = 7 if a.lane_head else 6

    def fwd(batch):
        body, tail = tuple(batch[:9]), tuple(batch[9:])
        sx0, sxA, sxB, ctg = fwd_stock(body)
        cur["sxA_beh"], cur["sxB_beh"] = sxA[..., BEH_ROW0:BEH_HEAD_ROWS], sxB[..., BEH_ROW0:BEH_HEAD_ROWS]
        if len(tail) == 2:
            cur["bq"], cur["bact"] = tail[0].to(dev, non_blocking=True).float(), tail[1].to(dev, non_blocking=True).bool()
            if cur.get("collect") is not None:
                cur["collect"].append((cur["sxB_beh"].detach().float().cpu(), cur["sxA_beh"].detach().float().cpu(), cur["bq"].cpu(), cur["bact"].cpu()))
        else:
            cur["bq"] = cur["bact"] = None
        out = (sx0, sxA[..., :ncol], sxB[..., :ncol], ctg)
        cur["sxB_ret"] = out[2]
        return out

    def train_losses(sx0, sxA, sxB, tg, rows, upd, lam_pg=None, pair_w=None, beh_w=None, beh_listwise=None):
        """the stock assembly + (arm B ONLY) the behaviour terms.  beh_w=None -> cur['w'] (0 = no term, nothing computed); an EXPLICIT beh_w (the G17 probe) is never short-circuited, so 0.0 still builds the graph
        and its gradient must read EXACTLY 0.0.  beh_listwise=None -> cur['listwise']."""
        if sxB is not cur.get("sxB_ret") or cur.get("bq") is None:
            raise RuntimeError("train_losses: sxB is not the output of the latest rebound fwd() (or the batch carries no behaviour labels)")
        lw = cur["listwise"] if beh_listwise is None else beh_listwise
        if lw:
            cur["rfac"], cur["selfac"] = beh_factors(cur["bq"], cur["bact"], cur["sxB_beh"], cur["cols"])
        try:
            lA, lB, ln, ep = train_losses_stock(sx0, sxA, sxB, tg, rows, upd, lam_pg, pair_w)
        finally:
            cur.pop("rfac", None); cur.pop("selfac", None)
        w = cur["w"] if beh_w is None else beh_w
        if beh_w is not None or w != 0.0:
            lB = lB + w * beh_bce(cur["sxB_beh"], cur["bq"], cur["bact"])
        return lA, lB, ln, ep
    return fwd, train_losses


'''
rep('def key(r):\n    return (r.get("log_name", ""),', A2 + 'def key(r):\n    return (r.get("log_name", ""),')
rep('er = expected_reward_loss(sxB_full, tg, a.tau_s, sel=sel_score)', 'er = expected_reward_loss(sxB_full, tg, a.tau_s, sel=sel_score, rfac=beh_cur.get("rfac"), selfac=beh_cur.get("selfac"))')
rep('    cap: dict = {}\n    model.score_dec[0]', '    cap: dict = {}\n    beh_cur: dict = {}                                       # A2: the rebound fwd/train_losses state ({} when --beh-heads is off: losses6 then reads None)\n    model.score_dec[0]')
rep('if not (n_out == 7 and k_.startswith("score_head.")):', 'if not (n_out in (7, BEH_HEAD_ROWS) and k_.startswith("score_head.")):')
rep('    AUX.add_aux_args(ap)\n    ap.add_argument("--aux-drv-lr"', '''    # A2 (a2/WIRING_PLAN.md): ALL DEFAULT OFF = this trainer exactly.
    ap.add_argument("--beh-labels-train", default="", help="A2: DIR of behaviour_*.jsonl side files (behaviour_label.py) for --onpolicy")
    ap.add_argument("--beh-labels-heldout", default="", help="A2: DIR of behaviour_*.jsonl side files for --heldout (the held-out per-term AUC readout)")
    ap.add_argument("--beh-heads", action="store_true", help="A2: grow score_head to 12 rows (rows 7..11 = the five behaviour terms)")
    ap.add_argument("--w-beh", type=float, default=0.0, help="A2: arm B ONLY, weight of the masked behaviour BCE (0.0 = rows 7..11 receive exactly zero gradient)")
    ap.add_argument("--beh-listwise", action="store_true", help="A2: arm B ONLY, the expected-reward term uses R = agg_v1 * prod q^act and the selection score x prod sigmoid(beh)^act (needs --b-mode expreward --lam > 0)")
    ap.add_argument("--beh-listwise-stall", action="store_true", help="A2: q_stall joins the listwise product (default: excluded, it varies across proposals in 3 of 62 scenes)")
    AUX.add_aux_args(ap)
    ap.add_argument("--aux-drv-lr"''')
rep('    bad_cli = bad_cli + pair_refusals(a)                                 # SCORER1: --pair-* named inconsistently never starts\n',
    '    bad_cli = bad_cli + pair_refusals(a)                                 # SCORER1: --pair-* named inconsistently never starts\n    bad_cli = bad_cli + beh_refusals(a, ap)                              # A2: --beh-* named inconsistently never starts\n')
rep('''    if a.lane_head:
        extend_lane_head(model, dev)
    armB = ScorerCopy(model).to(dev)''', '''    if a.lane_head:
        extend_lane_head(model, dev)
    beh_tr = beh_ho = beh_st_tr = beh_st_ho = beh_bias = None
    if a.beh_heads:                                          # A2: the labels are read BEFORE the head is grown (its bias is the logit of the train label mean)
        import behaviour_gate as BG
        _opn = T.OnPolicyBank(a.onpolicy, cfg.n_proposals, cfg.horizon_steps, lane_labels=a.lane_labels_train or None, pdm_labels=a.pdm_labels_train or None, teacher_lane_labels=a.teacher_lane_train or None)
        beh_tr, beh_st_tr = load_beh_terms(a.beh_labels_train, _opn.by, cfg.n_proposals)
        beh_bias = beh_bias_init(beh_tr)
        extend_beh_head(model, dev, beh_bias)
        del _opn
    armB = ScorerCopy(model).to(dev)''')
rep('''    # RL-S1: the reward config and the D / K / L side terms.''', '''    if a.beh_heads:                                          # A2: + (bq, bact) on the TRAINING and held-out items; the navtest quarantine is checked on every row of both sides BEFORE a label is used
        beh_ho, beh_st_ho = load_beh_terms(a.beh_labels_heldout, ho.by, cfg.n_proposals)
        if not beh_tr or not beh_ho:
            print(f"REFUSED: --beh-labels-train matched {len(beh_tr)} sets and --beh-labels-heldout {len(beh_ho)}: a head with no labels on a side is not an experiment"); return 4
        BG.assert_batch({"log_name": ds.rows[i].get("log_name"), "token": ds.rows[i].get("token")} for i in tr_idx + ho_idx)
        ds, ds_ho = BehItems(ds, beh_tr), BehItems(ds_ho, beh_ho)
    # RL-S1: the reward config and the D / K / L side terms.''')
rep('''    if aux is not None:
        # BUILT facts only (G-DVB compares them with the argv)''', '''    if a.beh_heads:                                          # A2: BUILT facts (read off the objects the run holds); absent without --beh-heads
        beh_cur.update(w=float(a.w_beh), listwise=bool(a.beh_listwise), cols=(BEH_REWARD_COLS_STALL if a.beh_listwise_stall else BEH_REWARD_COLS))
        info["beh"] = {"order": list(BEH_ORDER), "terms_version": BEH_TERMS_VERSION, "head_rows": int(model.score_head.out_features), "row0": BEH_ROW0, "w": beh_cur["w"], "listwise": beh_cur["listwise"],
                       "listwise_cols": list(beh_cur["cols"]), "stall_in_reward": 3 in beh_cur["cols"], "bias_init": beh_bias, "train": beh_st_tr, "heldout": beh_st_ho,
                       "train_active": [int(sum(int(act[:, c].sum()) for _q, act in beh_tr.values())) for c in range(BEH_N)]}
    if aux is not None:
        # BUILT facts only (G-DVB compares them with the argv)''')
rep('''    @torch.no_grad()
    def evaluate(tag, limit=None):''', '''    if a.beh_heads:                                          # A2: the ON path.  REBINDS fwd / train_losses (the stock ones stay as they were; with no flag nothing below runs)
        fwd_stock_beh, train_losses_stock_beh = fwd, train_losses
        fwd, train_losses = make_beh_wrappers(a, beh_cur, fwd_stock_beh, train_losses_stock_beh, dev)

    @torch.no_grad()
    def evaluate(tag, limit=None):''')
rep('''        if aux is None:
            return evaluate(tag)
        cap["aux_buf"] = []''', '''        if a.beh_heads:                                      # A2: evaluate() unchanged; the rebound fwd records the held-out behaviour logits of the very batches it forwards (no second trunk pass)
            beh_cur["collect"] = []
            try:
                out_ = evaluate(tag)
                buf_ = beh_cur["collect"]
            finally:
                beh_cur["collect"] = None
            if sum(int(x_[0].shape[0]) for x_ in buf_) != len(ho_idx):
                raise RuntimeError(f"beh eval: {sum(int(x_[0].shape[0]) for x_ in buf_)} head outputs for {len(ho_idx)} held-out sets -- the row alignment with evaluate() is broken")
            logs_h = [ds.rows[i].get("log_name", "") for i in ho_idx]
            res_h = {nm_: beh_heldout_auc(torch.cat([x_[j_] for x_ in buf_]), torch.cat([x_[2] for x_ in buf_]), torch.cat([x_[3] for x_ in buf_]), logs_h) for nm_, j_ in (("B", 0), ("A", 1))}
            log({"event": "beh_eval", "tag": tag, "n": len(ho_idx), "pass_q": BEH_PASS_Q, "auc": res_h})
            return out_
        if aux is None:
            return evaluate(tag)
        cap["aux_buf"] = []''')
open(P, "w", encoding="utf-8", newline="").write(s)
print("ok")
