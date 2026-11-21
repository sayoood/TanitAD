"""One-shot editor: adds the PREREG_V8F_HACC_LAT arm-B speed gate (a scalar beside hacc_w) to the COPY of v8f_mini.py."""
import sys

p = "C:/Users/Admin/hlat/code/stack/tanitad/models/v8f_mini.py"
s = open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


# 1. the buffer
rep('''        self.register_buffer("hacc_w", None)       # PREREG_V8F_HACC post-map weights [30, 2]: ABSENT by default (a None buffer is not in the state dict: every earlier checkpoint loads and runs bit for bit)
''', '''        self.register_buffer("hacc_w", None)       # PREREG_V8F_HACC post-map weights [30, 2]: ABSENT by default (a None buffer is not in the state dict: every earlier checkpoint loads and runs bit for bit)
        self.register_buffer("hacc_gate_v", None)  # PREREG_V8F_HACC_LAT arm B: the speed gate v* (m/s), a 0-d scalar beside hacc_w: ABSENT by default (not in the state dict, forward_infer bit-identical); NEVER read by _forward
''')

# 2. module-level helpers after hacc_apply
rep('''    return torch.lerp(prior, plan, w.expand_as(prior))
''', '''    return torch.lerp(prior, plan, w.expand_as(prior))


#: PREREG_V8F_HACC_LAT arm B: the axis the speed gate switches off (1 = cross, the ego frame's y; the along axis is never gated)
HACC_GATE_AXIS = 1


def hacc_check_gate(v) -> float:
    """The ONE validator of the HACC speed gate v* (m/s): returns it as a python float, or REFUSES with ``ValueError`` -- not a real number, NaN or negative. ``+inf`` IS legal (the gate fires at every speed:
    PREREG_V8F_HACC_LAT control C-B2) and so is 0 (``v0 < 0`` never holds: the gate never fires, control C-B1)."""
    try:
        x = float(v.detach().cpu().item() if isinstance(v, Tensor) else v)
    except (TypeError, ValueError, RuntimeError) as e:
        raise ValueError(f"the HACC speed gate is not a real number: {e}") from e
    if x != x or x < 0.0:
        raise ValueError(f"the HACC speed gate v* must be >= 0 (or +inf), got {x!r}")
    return x
''')

# 3. forward_infer
rep('''        g = _KeyGuard(batch)
        out = self._forward(g, 0.0, False)
        out["_keys_read"] = set(g.read)
        return self._hacc_post_map(out)

    def _hacc_post_map(self, out: dict) -> dict:''', '''        g = _KeyGuard(batch)
        out = self._forward(g, 0.0, False)
        out["_keys_read"] = set(g.read)
        return self._hacc_post_map(out, None if self.hacc_gate_v is None else g["p10f"][:, -1, 0])

    def _hacc_post_map(self, out: dict, v0: Tensor | None = None) -> dict:''')

rep('''        compares the two) and NEVER in ``_forward``: training and the loss read O's raw plan."""
        if self.hacc_w is None:
            return out
        w30 = self.hacc_w.to(device=out["traj60"].device, dtype=out["traj60"].dtype)
        out["traj60_o"], out["traj30_o"] = out["traj60"], out["traj30"]
        out["traj60"] = hacc_apply(out["traj60"], upsample_prior(out["p10"]), hacc_w_midpoints(w30))
        out["traj30"] = subsample_02(out["traj60"])
        return out
''', '''        compares the two) and NEVER in ``_forward``: training and the loss read O's raw plan.

        PREREG_V8F_HACC_LAT arm B: with the optional scalar ``hacc_gate_v`` (v*) set, the CROSS weight of a sample is 0 (the cross axis falls back to the prior P) when its measured ego speed ``v0`` --
        ``p10f[:, -1, 0]``, the speed feature the prior already reads, handed in by :meth:`forward_infer` / :meth:`forward_diag` -- is below v*; the along axis is untouched. The gate is a function of
        the measured ego motion alone (no new batch key, no label, no situation-classifier output). With it absent the lines below are the ones HACC shipped, bit for bit. A gate with no ``hacc_w``
        is REFUSED (it would silently do nothing), and so is a set gate without ``v0``."""
        if self.hacc_w is None:
            if self.hacc_gate_v is not None:
                raise ValueError("hacc_gate_v is set but hacc_w is absent: the gate acts on the post-map and would silently do nothing")
            return out
        w30 = self.hacc_w.to(device=out["traj60"].device, dtype=out["traj60"].dtype)
        out["traj60_o"], out["traj30_o"] = out["traj60"], out["traj30"]
        w60 = hacc_w_midpoints(w30)
        if self.hacc_gate_v is not None:
            if v0 is None:
                raise ValueError("hacc_gate_v is set: the post-map needs the measured ego speed v0 of every sample")
            fire = v0.to(device=w60.device, dtype=w60.dtype) < self.hacc_gate_v.to(device=w60.device, dtype=w60.dtype)       # [B]
            w60 = w60[None].expand(fire.shape[0], -1, -1).clone()
            w60[:, :, HACC_GATE_AXIS] = torch.where(fire[:, None], torch.zeros_like(w60[:, :, HACC_GATE_AXIS]), w60[:, :, HACC_GATE_AXIS])
            out["hacc_gate_fired"] = fire
        out["traj60"] = hacc_apply(out["traj60"], upsample_prior(out["p10"]), w60)
        out["traj30"] = subsample_02(out["traj60"])
        return out
''')

# 4. setters, clear, load
rep('''    def clear_hacc_weights(self) -> None:
        """Back to the plain O plan: the buffer is removed (so it leaves the state dict)."""
        self.hacc_w = None
''', '''    @property
    def hacc_gate_active(self) -> bool:
        """True iff the HACC speed gate v* is set (PREREG_V8F_HACC_LAT arm B)."""
        return self.hacc_gate_v is not None

    def set_hacc_gate(self, v) -> None:
        """Set the speed gate v* (m/s, :func:`hacc_check_gate`: >= 0 or +inf) as a 0-d float32 buffer beside ``hacc_w``. REFUSED with ``ValueError`` when ``hacc_w`` is absent or ``v`` is invalid, and the model
        is left unchanged."""
        x = hacc_check_gate(v)
        if self.hacc_w is None:
            raise ValueError("set_hacc_gate needs hacc_w: set the post-map weights first")
        self.register_buffer("hacc_gate_v", torch.tensor(x, dtype=torch.float32, device=self.basis.device))

    def clear_hacc_gate(self) -> None:
        """Back to the ungated HACC post-map: the scalar is removed (so it leaves the state dict)."""
        self.hacc_gate_v = None

    def clear_hacc_weights(self) -> None:
        """Back to the plain O plan: the buffer is removed (so it leaves the state dict). The gate is meaningless without the weights, so it goes too."""
        self.hacc_w = None
        self.hacc_gate_v = None
''')

rep('''        key = prefix + "hacc_w"
        if key in state_dict:
            hacc_check_weights(state_dict[key])
            if self.hacc_w is None:
                self.register_buffer("hacc_w", torch.zeros(K30, 2, device=self.basis.device))
''', '''        key = prefix + "hacc_w"
        if key in state_dict:
            hacc_check_weights(state_dict[key])
            if self.hacc_w is None:
                self.register_buffer("hacc_w", torch.zeros(K30, 2, device=self.basis.device))
        gkey = prefix + "hacc_gate_v"
        if gkey in state_dict:                                   # PREREG_V8F_HACC_LAT: a state dict that carries the gate is validated and needs the weights beside it
            hacc_check_gate(state_dict[gkey])
            if key not in state_dict:
                raise ValueError("a state dict carries hacc_gate_v without hacc_w: the gate acts on the post-map")
            if self.hacc_gate_v is None:
                self.register_buffer("hacc_gate_v", torch.zeros((), device=self.basis.device))
''')

# 5. forward_diag
rep('''        return self._hacc_post_map(out)                      # the same post-map as forward_infer (absent => out as is), so the identity under T's own code survives''',
    '''        return self._hacc_post_map(out, None if self.hacc_gate_v is None else g["p10f"][:, -1, 0])     # the same post-map as forward_infer (absent => out as is), so the identity under T's own code survives''')
open(p, "w", encoding="utf-8", newline="").write(s.replace("\n", "\r\n"))
print("ok")
