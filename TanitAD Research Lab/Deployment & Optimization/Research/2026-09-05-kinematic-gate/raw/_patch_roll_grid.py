"""Fix the roll probe's TWO grid errors, both of the `dt`/scope family.

1. ⛔ `ARM_HORIZONS` is uniform at 0.5 s ONLY over the first four slots (5/10/15/20 frames at
   10 Hz, asserted at rl_refcv3_min.py:610); the tail slots are the 6 s horizons on a different
   spacing. Deriving speed and arc over all 8 slots with DT = 0.5 s inflates the anchors' own
   speed to a mean of 35.73 m/s (max 61.99) and makes the re-integration inconsistent -- which
   is exactly what C3 caught.
2. ⛔ Anchor slot 0 is at t = 0.5 s, not the origin, so `anchors[:,1:] - anchors[:,:-1]` drops
   the FIRST segment (origin -> 0.5 s). Prepending the origin fixes it.

⇒ The probe now operates on the 5-point 2 s prefix `with_origin(anchors[:, :4])`, which is the
same object every fan-safety metric is computed on. C3 is re-run and must PASS before any
number from the rolled sweep is admissible.
"""
import io
import os

os.chdir("C:/Users/Admin/kingate/raw")
p = "v0_roll_bank_probe.py"
s = io.open(p, encoding="utf-8").read()

a = '''    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    key = find_anchors(sd)
    anchors = sd[key].detach().float()                          # [N, S, 2]
    bank = anchors[None].expand(W, -1, -1, -1).contiguous()

    # ---- C3: the roll must be a NO-OP at each anchor's own mean speed ---------------
    seg = (anchors[:, 1:, :] - anchors[:, :-1, :]).norm(dim=-1)
    own_speed = seg.mean(dim=-1) / DT                           # [N] per-anchor mean speed
    self_roll = roll_at_speed(anchors, own_speed)               # [N, N, S, 2]
    self_diag = self_roll[torch.arange(N), torch.arange(N)]     # [N, S, 2]
    c3_err = float((self_diag - anchors).norm(dim=-1).mean())
    c3_max = float((self_diag - anchors).norm(dim=-1).max())

    rolled = roll_at_speed(anchors, v0)                          # [W, N, S, 2]'''
b = '''    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    key = find_anchors(sd)
    anchors8 = sd[key].detach().float()                         # [N, 8, 2]
    # ⛔ THE GRID IS UNIFORM ONLY OVER THE FIRST FOUR SLOTS (rl_refcv3_min.py:610 asserts
    # ARM_HORIZONS[:4] == [5, 10, 15, 20] frames at 10 Hz = 0.5 s spacing); the tail is the
    # 6 s horizon set on a different spacing. And slot 0 is at t = 0.5 s, not the origin.
    # Both are handled by working on the 5-point 2 s prefix, which is the object every
    # fan-safety metric is computed on anyway.
    anchors = D.with_origin(anchors8[:, :NS, :])                # [N, 5, 2], origin first
    bank8 = anchors8[None].expand(W, -1, -1, -1).contiguous()

    # ---- C3: the roll must be a NO-OP at each anchor's own mean speed ---------------
    seg = (anchors[:, 1:, :] - anchors[:, :-1, :]).norm(dim=-1)
    own_speed = seg.mean(dim=-1) / DT                           # [N] per-anchor mean speed
    self_roll = roll_at_speed(anchors, own_speed)               # [N, N, 5, 2]
    self_diag = self_roll[torch.arange(N), torch.arange(N)]     # [N, 5, 2]
    c3_err = float((self_diag - anchors).norm(dim=-1).mean())
    c3_max = float((self_diag - anchors).norm(dim=-1).max())

    rolled5 = roll_at_speed(anchors, v0)                         # [W, N, 5, 2]
    # back to the 8-slot layout the sweep indexes: the 2 s prefix is rolled, the tail is
    # carried over unchanged and is NEVER scored (every metric below reads [..., :NS, :]).
    rolled = bank8.clone()
    rolled[:, :, :NS, :] = rolled5[:, :, 1:, :]
    bank = bank8'''
assert a in s
s = s.replace(a, b)

a = '''    W, N, S = fan.shape[0], fan.shape[1], fan.shape[2]
    NS = D.N_REWARD_SLOTS
    v0 = bk["v0"].float()'''
b = '''    W, N, S = fan.shape[0], fan.shape[1], fan.shape[2]
    NS = D.N_REWARD_SLOTS
    v0 = bk["v0"].float()
    N = fan.shape[1]'''
assert a in s
s = s.replace(a, b)

# C3 must gate the report, not decorate it
a = '''            "PASS": bool(c3_err < 0.25),'''
b = '''            "PASS": bool(c3_err < 0.25),
            "gates_the_rolled_sweep": True,'''
assert a in s
s = s.replace(a, b)

a = '''    for name, rows in (("FIXED bank (speed-blind, as shipped)", fixed),
                       ("v0-ROLLED bank (same shape, window's own speed)", roll)):'''
b = '''    _c3ok = controls["C3_roll_is_noop_at_own_speed"]["PASS"]
    if not _c3ok:
        print("")
        print("  !! C3 FAILED -> THE v0-ROLLED SWEEP IS INADMISSIBLE AND IS NOT QUOTED.")
        print("     A broken integrator would read as a spectacular finding here; that is")
        print("     what this control exists to stop. Printed below for the record only.")
    for name, rows in (("FIXED bank (speed-blind, as shipped)", fixed),
                       ("v0-ROLLED bank%s" % ("" if _c3ok else "  [INADMISSIBLE, C3 FAILED]"),
                        roll)):'''
assert a in s
s = s.replace(a, b)

a = '''           "controls": controls,
           "sweep_fixed_bank": fixed, "sweep_v0_rolled_bank": roll}'''
b = '''           "controls": controls,
           "rolled_sweep_admissible": bool(controls["C3_roll_is_noop_at_own_speed"]["PASS"]),
           "sweep_fixed_bank": fixed, "sweep_v0_rolled_bank": roll}'''
assert a in s
s = s.replace(a, b)

io.open(p, "w", encoding="utf-8").write(s)
print("patched v0_roll_bank_probe.py")
