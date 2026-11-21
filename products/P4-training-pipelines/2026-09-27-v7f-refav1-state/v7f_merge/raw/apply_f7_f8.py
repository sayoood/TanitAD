"""F7 + F8 (found by the v7f gate once nav was really built, 2026-09-27):
F7  the S-W world-model losses roll predictor_op through SHARED helpers (stage_a_losses, rollout_transitions) that never
    forward nav, so the operative nav tensors (nav.embed/arg_proj + the operative projection/gate) got NO gradient in
    S-W and S-T freezes that group -> the operative layer never learned from the nav command. Fix: bind the operative
    nav embedding to the predictor for those three call sites via a thin delegating wrapper (the shared helpers stay
    untouched). With nav off the wrapper is not used (same object) -> byte-identical.
F8  synthetic_train_batch carried no nav keys -> the trainer's own --dry-run died with NavTokenMissing on every nav launch.
EOL preserved; every anchor must match exactly once or the patch refuses."""
import sys
from pathlib import Path

CRLF = chr(13) + chr(10)
p = Path("C:/Users/Admin/v7f_merge/stack/scripts/train_v6_staged.py")
eol = CRLF if CRLF.encode() in p.read_bytes() else chr(10)
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    c = s.count(old)
    if c != n:
        sys.exit(f"anchor matched {c} (want {n}) -- refusing:\n{old[:200]}")
    s = s.replace(old, new)


# --- F7: the wrapper, defined right before v6_loss_step --------------------------------------------------------
rep("def v6_loss_step(stack: V6Stack, batch: dict, *, stage: str,",
    '''class _NavBoundPredictor:
    """⭐ R2 / F7 (2026-09-27): the operative predictor WITH the nav conditioning bound, for the SHARED world-model loss
    helpers (``train_stage_a.stage_a_losses``, ``metric_dynamics.rollout_transitions``), which call
    ``predictor(window, actions)`` and know nothing of nav. MEASURED by the v7f gate: without it the operative nav
    tensors received NO gradient in S-W (and S-T freezes their group), so the operative layer never learned from the
    nav command. Delegates every other attribute (``parameters()``, ``training`` ...) to the wrapped predictor. Each
    helper rolls arms SEPARATELY on the same batch, so the per-row nav tensor always matches the batch."""

    def __init__(self, predictor, nav_cond):
        self._p, self._nav = predictor, nav_cond

    def __call__(self, *args, **kwargs):
        kwargs.setdefault("nav_cond", self._nav)
        return self._p(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._p, name)


def v6_loss_step(stack: V6Stack, batch: dict, *, stage: str,''')

# the bound predictor, computed once per step right after the shared forward
rep('''    out = stack.forward(frames=batch["frames"], actions=_lift3(''',
    '''    # ⭐ R2 / F7: the operative predictor the WM-loss helpers roll -- nav-bound when the stack has nav, else the
    # SAME object (byte-identical). Not detached: in S-W these losses are what trains the operative nav path.
    _pred_op = stack.predictor_op
    if getattr(stack, "nav", None) is not None and batch.get("nav_token") is not None:
        _pred_op = _NavBoundPredictor(stack.predictor_op,
                                      stack.nav(batch["nav_token"], batch["nav_args"], "operative"))
    out = stack.forward(frames=batch["frames"], actions=_lift3(''')

rep('''        L1 = stage_a_losses(
            stack.predictor_op, stack.step_readout_op, o1_states,''',
    '''        L1 = stage_a_losses(
            _pred_op, stack.step_readout_op, o1_states,''')
rep('''        trans = rollout_transitions(stack.predictor_op, states, aw3, fa3,''',
    '''        trans = rollout_transitions(_pred_op, states, aw3, fa3,''')
rep('''                tn = rollout_transitions(stack.predictor_op, states, aw3,''',
    '''                tn = rollout_transitions(_pred_op, states, aw3,''')

# --- F8: nav keys in the dry-run's synthetic batch (drawn LAST, only when the stack has nav) --------------------
rep('''        if b >= 2:
            out["v_max_valid"][-1] = 0.0
    if device is not None:
        out = {kk: ([t.to(device) for t in v] if isinstance(v, list)''',
    '''        if b >= 2:
            out["v_max_valid"][-1] = 0.0
    # ⭐ R2 / F8 (2026-09-27): a nav stack REQUIRES the nav channel in forward, and this batch carried none, so the
    # trainer's own --dry-run died with NavTokenMissing on every nav launch (MEASURED by the v7f gate). Drawn LAST and
    # only when the conditioner exists, so every pre-existing batch is byte-identical.
    if getattr(stack, "nav", None) is not None:
        out["nav_token"] = torch.randint(0, len(stack.nav.tokens), (b,), generator=g)
        out["nav_args"] = torch.rand(b, 2, generator=g)
    if device is not None:
        out = {kk: ([t.to(device) for t in v] if isinstance(v, list)''')

open(p, "w", encoding="utf-8", newline=eol).write(s)
print("F7 + F8 applied")
