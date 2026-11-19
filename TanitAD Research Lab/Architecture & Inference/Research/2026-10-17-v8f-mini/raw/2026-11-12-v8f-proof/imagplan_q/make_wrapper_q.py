#!/usr/bin/env python
"""make_wrapper_q.py -- builds v8f_t1_eval_imagplan_q.py (note PREREG_V8F_PROOF_ADD40_N7, source cand-cvq) from a COPY of ../imagplan/v8f_t1_eval_imagplan.py.
Every edit is asserted to match exactly once; the original is not changed. imag_plan.py is NOT patched: the rule is called with source cand-cv for cand-cvq
(the rule of the card is the same; only the gap that enters it differs).  Run: python make_wrapper_q.py  (writes next to this file)."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "imagplan", "v8f_t1_eval_imagplan.py")
OUT = os.path.join(HERE, "v8f_t1_eval_imagplan_q.py")

EDITS = []


def E(old, new):
    EDITS.append((old, new))


E('import v8f_t1_eval_rule as W  # noqa: E402\n',
  'import v8f_t1_eval_rule as W  # noqa: E402\nimport qhead_model as QM  # noqa: E402  (note ADD40_N7: the trained quantile head of card ADD42, one version, never a copy of its code)\n')
E('MODES = ("off",) + IP.SOURCES\n', 'MODES = ("off",) + IP.SOURCES + ("cand-cvq",)      # cand-cvq: note ADD40_N7 (cand-cv with the gap min(g_head, q_head))\n')
E('"n_nonfinite", "changed", "n_yaw_lim")\n', '"n_nonfinite", "changed", "n_yaw_lim", "g_head", "q_head", "g_used")\n')
E('''        if source not in IP.SOURCES:
            raise ValueError(f"source must be one of {IP.SOURCES}")
''', '''        if source not in IP.SOURCES + ("cand-cvq",):
            raise ValueError(f"source must be one of {IP.SOURCES + ('cand-cvq',)}")
''')
E('p45=None, action_def="logged", family="scale"):\n', 'p45=None, action_def="logged", family="scale", qhead=None, q_const=None):\n')
E('        self.cand_source, self.T = source, T\n', '''        self.q_on = source == "cand-cvq"
        if self.q_on and qhead is None:
            raise SystemExit("REFUSED: cand-cvq needs the quantile head file (IP_QHEAD_FILE)")
        if (not self.q_on) and (qhead is not None or q_const is not None):
            raise SystemExit("REFUSED: a quantile head or constant with a source other than cand-cvq would change nothing")
        self.qhead, self.q_const, self.q_dev = qhead, q_const, None
        self.cand_source, self.T = ("cand-cv" if self.q_on else source), T      # the rule of cand-cv, unchanged (note N7 section 2)
''')
E('    # ---------------------------------------------------------------- p4 / p5 (tick 0 only, every window, any gate)\n', '''    def _read_q(self, feats, g_head):
        """q_head in metres for each row of the batch, read at the TICK frame: the quantile head on z1 of the last `hist` frames of the batch (the expression of the banked head read);
        or the constant IP_QHEAD_CONST (controls q1, q2) in place of the head output."""
        g = np.asarray(g_head)
        if self.q_const is not None:
            return np.full(g.shape, float(self.q_const), np.float32)
        hist = int(self.model.wm.lead_head.hist)
        if self.qhead.hist != hist:
            raise SystemExit("REFUSED: the quantile head reads %d frames, the banked head %d" % (self.qhead.hist, hist))
        dev = feats.device
        if self.q_dev != str(dev):
            self.qhead = self.qhead.to(dev).eval()
            self.q_dev = str(dev)
        with self.torch.no_grad():
            f = feats[:, -hist:]
            z1 = self.model.wm.z1(self.model.adapter(f.float()) + self.model.pos[None, None])
            o = self.qhead(z1)[:, 0, 0].float().cpu().numpy()
        return (o * QM.GAP_SCALE_M).astype(np.float32)

    # ---------------------------------------------------------------- p4 / p5 (tick 0 only, every window, any gate)
''')
E('        p_src = np.full(B, np.nan)\n', '        p_src = np.full(B, np.nan)\n        g_gate, gh_c, qh_c = None, np.full(B, np.nan), np.full(B, np.nan)\n')
E('            p, g_h, c_h, ok = W.head_values(self.model, batch["feats"], batch["frame_ok"])\n', '''            p, g_h, c_h, ok = W.head_values(self.model, batch["feats"], batch["frame_ok"])
            gh_c = np.asarray(g_h, np.float64)
            if self.q_on:
                q_h = self._read_q(batch["feats"], g_h)
                g_h = QM.used_gap(g_h, q_h)                      # g_used = min(g_head, q_head)
                g_gate, qh_c = gh_c, np.asarray(q_h, np.float64)  # the gate (gap at most 60 m) stays on the head gap: the active set is that of cand-cv (note N7 control q5)
''')
E('(g_src <= IP.G_MAX_M)', '((g_src if g_gate is None else g_gate) <= IP.G_MAX_M)')
E('n_yaw_lim=self._yl.copy(), safe=out.safe,', 'n_yaw_lim=self._yl.copy(), g_head=gh_c, q_head=qh_c, g_used=g_src, safe=out.safe,')
E('p45=p45, action_def=opts["action_def"], family=opts["family"])', 'p45=p45, action_def=opts["action_def"], family=opts["family"], qhead=opts["qhead"], q_const=opts["q_const"])')
E('def install(T, source, a, opts, J=None, log=None, p45=None):\n', '''def load_qhead_env(mode, ckpt, env=None):
    """Note ADD40_N7: IP_QHEAD_FILE (the head file; required for cand-cvq, refused for any other mode) and IP_QHEAD_CONST (metres; replaces the head OUTPUT, controls q1 = 1e9, q2 = 0).
    Checks of the rule wrapper v8f_t1_eval_rule_qhead.py: not an evaluation-split path, fit record next to the file, checkpoint sha256 of the record equals the run's, head sha256 in the
    record, and (note N7 section 2) coverage on the check clips in [0.15, 0.25] (h5 true) -- else the head is not read.  Returns (head or None, const or None, meta dict)."""
    env = os.environ if env is None else env
    f, c = env.get("IP_QHEAD_FILE", ""), env.get("IP_QHEAD_CONST", "")
    if mode != "cand-cvq":
        if f or c != "":
            raise SystemExit("REFUSED: IP_QHEAD_FILE / IP_QHEAD_CONST with mode %s would change nothing" % mode)
        return None, None, {}
    if not f:
        raise SystemExit("REFUSED: cand-cvq needs IP_QHEAD_FILE")
    if not os.path.isfile(f):
        raise SystemExit("REFUSED: no head file " + f)
    if QM.is_eval_path(f):
        raise SystemExit("REFUSED: a head file of the evaluation split: " + f)
    rec_p = f[:-4] + ".json" if f.endswith(".npz") else f + ".json"
    if not os.path.isfile(rec_p):
        raise SystemExit("REFUSED: no fit record " + rec_p)
    rec = json.load(open(rec_p, encoding="utf-8"))
    sha_head = QM.sha256_file(f)
    if sha_head not in (rec.get("head_file_sha256") or {}).values():
        raise SystemExit("REFUSED: the head file does not match the sha256 of its fit record")
    if rec.get("h5_check_coverage_in_0.15_0.25") is not True:
        raise SystemExit("REFUSED: the fit record does not say check coverage inside [0.15, 0.25]: the head is not read")
    ck_rec = (rec.get("states_record") or {}).get("ckpt_sha256")
    if not ck_rec:
        raise SystemExit("REFUSED: the fit record has no checkpoint sha256")
    if ckpt and os.path.isfile(ckpt) and QM.sha256_file(ckpt) != ck_rec:
        raise SystemExit("REFUSED: the head was fitted on states of checkpoint %s, this run is another" % ck_rec)
    const = float(c) if c != "" else None
    return QM.load_head(f), const, {"qhead_file": f, "qhead_sha256": sha_head, "qhead_fit_record": rec_p, "qhead_const": np.nan if const is None else const,
                                    "qhead_model_sha256": QM.sha256_file(QM.__file__)}


def install(T, source, a, opts, J=None, log=None, p45=None):
''')
E('    if mine.imag_plan == "off":\n        extras = [n for n, bad in', '''    qh, qc, qmeta = load_qhead_env(mine.imag_plan, a.ckpt)
    if mine.imag_plan == "off":
        extras = [n for n, bad in''')
E('"noise_seed": mine.noise_after_tick, "family": mine.cand_family}', '"noise_seed": mine.noise_after_tick, "family": mine.cand_family, "qhead": qh, "q_const": qc}')
E('"imag_shift_m": mine.imag_shift_m, "noise_after_tick": mine.noise_after_tick, "pinned_path": path,', '"imag_shift_m": mine.imag_shift_m, "noise_after_tick": mine.noise_after_tick, **qmeta, "pinned_path": path,')


def main():
    s = open(SRC, encoding="utf-8", newline="").read()
    crlf = "\r\n" in s
    if crlf:
        s = s.replace("\r\n", "\n")
    for i, (old, new) in enumerate(EDITS):
        n = s.count(old)
        if n != 1:
            sys.exit("EDIT %d matches %d times (need exactly 1): %r" % (i, n, old[:70]))
        s = s.replace(old, new)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(s)
    print("wrote", OUT, "edits", len(EDITS), "sha256", hashlib.sha256(s.encode("utf-8")).hexdigest()[:16])


if __name__ == "__main__":
    main()
