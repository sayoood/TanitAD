"""Apply the R1 patch (refcv8's hard speed CLAMP on the emitted plan) to the pristine tip files.

Reads `tree/` (git archive of f7f943e), writes the edited files into `tree_patched/` (for the tests) AND `out/`
(the deliverable). Idempotent: always starts from the pristine bytes. Anchors must match exactly once.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _edit as E  # noqa: E402

ROWS: list[dict] = []

# --------------------------------------------------------------------------------------------------------------- #
# 1. refcv6_selection.py : the pure clamp, beside planned_max_speed / SpeedCeilingFilter (ONE dt derivation)        #
# --------------------------------------------------------------------------------------------------------------- #
REL = "stack/tanitad/refs/refcv6_selection.py"
tip = E.read(REL)
d = E.edit(tip, '    "planned_max_speed", "step_durations_s",\n',
           '    "cap_plan_to_limit",\n')
d = E.edit(d, "        return keep, tele\n", '''

# ---------------------------------------------------------------------------
# 2b. the HARD CAP on the EMITTED plan -- refcv8's equivalent of v7F's R1 `plan_vmax_cap`
# ---------------------------------------------------------------------------

def cap_plan_to_limit(traj: Tensor, v_limit_ms: Tensor, *, horizons: Sequence[int],
                      tick_s: float = 0.1, v0: Tensor | None = None,
                      brake_ms2: float = 0.0) -> tuple[Tensor, dict]:
    """Clamp the EMITTED plan so no waypoint interval is faster than the fed ceiling. -> (capped, diag).

    PI R1 (2026-09-27): the max speed "must not be exceeded". :class:`SpeedCeilingFilter` cannot promise that:
    it only CHOOSES among the fan, and a row with no compliant candidate keeps its whole fan (the "exempt"
    rows the CEIL-OBEY instrument counts and never folds into the bar), so the plan it emits there can be
    above the ceiling. This function is the missing half: a projection of the SELECTED plan, v7F's
    `plan_speed_cap.cap_speed_profile` semantics expressed on refcv8's waypoint grid.

    ``traj`` ``[B, S, 2]`` metres, ego frame, origin at t0. ``horizons`` are its waypoint TICKS: the interval
    durations come from :func:`step_durations_s`, the SAME derivation :func:`planned_max_speed` (and so the
    CEIL-OBEY statistic) uses -- one rule, never a typed 0.1 (a 5x error). ``v_limit_ms`` ``[B]``, ``+inf`` =
    no ceiling for that row (the X15 rule: unknown is never a fabricated limit).

    * per interval ``i`` the displacement vector is scaled by ``min(1, allowed_i * dt_i / |d_i|)`` and the
      path is re-integrated: headings are kept, progress is compressed;
    * a row whose plan is already under the ceiling is returned BIT-IDENTICAL (``torch.where`` on the
      per-row binding mask), so a non-binding cap equals no cap;
    * ``allowed_i = max(limit, v0 - brake_ms2 * t_mid_i)`` when ``v0`` and ``brake_ms2 > 0`` are given: an ego
      already above the ceiling brakes at ``brake_ms2`` until under it (v7F: "brakes at -a_max until it is
      under"); that residual is a property of the INPUT state and is REPORTED (``plan_cap_v0_over_limit``),
      never hidden. With ``brake_ms2 = 0`` the allowance is the ceiling itself (strict).
    * the cap holds on the WAYPOINT grid (0.5 s steps here), which is the grid CEIL-OBEY measures on.
    """
    if traj.dim() != 3 or traj.shape[-1] != 2:
        raise ValueError(f"traj must be [B, S, 2], got {tuple(traj.shape)}")
    dts = step_durations_s(horizons, tick_s)
    if len(dts) != traj.shape[1]:
        raise ValueError(f"[plan-cap] {len(dts)} horizons but the plan has {traj.shape[1]} waypoints; a "
                         f"mismatched grid pairs waypoint i with interval j")
    b, s, _ = traj.shape
    lim = v_limit_ms.reshape(-1).to(traj.dtype)
    if lim.shape[0] != b:
        raise ValueError(f"v_limit_ms must be [B={b}], got {tuple(v_limit_ms.shape)}")
    if bool(torch.isnan(lim).any()) or bool((lim <= 0).any()):
        raise ValueError("[plan-cap] every row's limit must be > 0 m/s or +inf (no cap). A NaN or non-positive "
                         "limit is not a speed limit; it would brake every plan to a standstill.")
    dt = torch.tensor(dts, device=traj.device, dtype=traj.dtype)                      # [S]
    t_mid = torch.cumsum(dt, 0) - 0.5 * dt                                             # [S]
    allow = lim[:, None].expand(b, s)
    over = None
    if v0 is not None and float(brake_ms2) > 0.0:
        v0r = v0.reshape(-1).to(traj.dtype)
        allow = torch.maximum(allow, v0r[:, None] - float(brake_ms2) * t_mid[None, :])
        over = v0r > lim
    seq = torch.cat([torch.zeros_like(traj[:, :1]), traj], dim=1)                      # [B, S+1, 2]
    disp = seq[:, 1:] - seq[:, :-1]                                                    # [B, S, 2]
    ln = disp.norm(dim=-1)                                                             # [B, S]
    room = allow * dt[None, :]                                                         # metres per interval
    scale = torch.where(ln > room, room / ln.clamp_min(1e-12), torch.ones_like(ln))
    capped = torch.cumsum(disp * scale[..., None], dim=1)
    binding = (scale < 1.0).any(dim=1)                                                 # [B]
    out = torch.where(binding[:, None, None], capped, traj)
    with torch.no_grad():
        before = planned_max_speed(traj.detach()[:, None], horizons=horizons, tick_s=tick_s)[:, 0]
        after = planned_max_speed(out.detach()[:, None], horizons=horizons, tick_s=tick_s)[:, 0]
        diag = {
            "plan_cap_binding": binding.detach(),
            "plan_cap_speed_before": before,
            "plan_cap_speed_after": after,
            "plan_cap_excess_ms": (after - lim.detach()).clamp_min(0.0),
        }
        if over is not None:
            diag["plan_cap_v0_over_limit"] = over.detach() & torch.isfinite(lim.detach())
    return out, diag
''')
E.record(ROWS, REL, tip, d, "edit")
E.write(REL, d)

# --------------------------------------------------------------------------------------------------------------- #
# 2. refc_v3.py : the config field, the limit export, the clamp in the E9 block, the build guards                   #
# --------------------------------------------------------------------------------------------------------------- #
REL = "stack/tanitad/refs/refc_v3.py"
tip = E.read(REL)
d = E.edit(tip, "    max_speed_onehot_v6: bool = False\n", '''
    # --- v7F R1 EQUIVALENT (PI 2026-09-27: the max speed "must not be exceeded"); refcv8 audit 2026-10-18 ---
    # ⛔ `speed_ceiling_filter` only CHOOSES among the fan: a row with no compliant candidate keeps its whole
    # fan (`refcv6_selection.SpeedCeilingFilter`, `e9_rank`), so the EMITTED plan can exceed the fed ceiling.
    # True adds a CLAMP on the emitted plan (`refcv6_selection.cap_plan_to_limit`), INFERENCE ONLY
    # (`not self.training`: the label is the ego's own realised max, so the training plan must stay uncapped).
    # Default False = byte-identical to every banked arm. Needs `max_speed_onehot_v6` (the ceiling channel).
    plan_vmax_clamp: bool = False
''')
d = E.edit(d, "        self.core = refc.RefCModel(cfg.core)\n", '''        if getattr(cfg, "plan_vmax_clamp", False):
            # ⛔ a clamp whose limit is +inf on every row is BUILT and INERT -- the false-provenance class
            if not getattr(cfg, "max_speed_onehot_v6", False):
                raise ValueError(
                    "[refcv8] plan_vmax_clamp without max_speed_onehot_v6: the clamp's limit IS the fed 4-way "
                    "set speed; with no channel every row's limit is +inf and the clamp is built and INERT.")
            # ⛔ the refcv7 scorer pick REPLACES out["traj"] after E9, so it would emit an unclamped plan
            if bool(getattr(cfg, "refcv7_select", False)):
                raise ValueError(
                    "[refcv8] plan_vmax_clamp with refcv7_select: the scorer's pick replaces the emitted plan "
                    "after the E9 block where the clamp sits -- the cap would not reach the plan.")
''', mode="before")
d = E.edit(d, "            if ego_state is not None:\n                es = ego_state.to(ref.dtype)\n"
              "                keep = es[:, 4:5]\n", '''            if bool(getattr(self.cfg, "plan_vmax_clamp", False)):
                cache["plan_v_lim_ms"] = v_lim.detach()          # the E9 block clamps the EMITTED plan to it
''', mode="before")
d = E.edit(d, "        traj = fan[torch.arange(b, device=fan.device), idx]\n        out.update(cache)\n",
           '''        traj = fan[torch.arange(b, device=fan.device), idx]
        if bool(getattr(self.cfg, "plan_vmax_clamp", False)) and not self.training:
            # ⛔ v7F R1 EQUIVALENT. The argmax filter above CHOOSES; this CLAMPS. A row with no compliant
            # candidate keeps its whole fan (counted in `e9_ceil_dead_frac`), so without this the emitted plan
            # can exceed the fed ceiling there. Non-binding rows are returned bit-identical.
            _pl = cache.get("plan_v_lim_ms")
            if _pl is None:
                raise ValueError("[refcv8] plan_vmax_clamp is on but the scene hook exported no limit "
                                 "(`plan_v_lim_ms`): the clamp would silently not run.")
            _dec = self.core.decoder
            out["traj_precap"] = traj
            traj, _cd = v6sel.cap_plan_to_limit(
                traj, _pl, horizons=_dec.anchor_horizons, tick_s=_dec.anchor_dt, v0=v0,
                brake_ms2=float(_dec.sel.accel_max))
            out.update(_cd)
        out.update(cache)
''', mode="replace")
E.record(ROWS, REL, tip, d, "edit")
E.write(REL, d)

# --------------------------------------------------------------------------------------------------------------- #
# 3. refc_v3_train.py (CRLF) : the flag, its pin, its config.json stamp                                            #
# --------------------------------------------------------------------------------------------------------------- #
REL = "stack/scripts/refc_v3_train.py"
tip = E.read(REL)
d = E.edit(tip, "        cfg.core.speed_ceiling_filter = True\n", '''    # ⭐ v7F R1 EQUIVALENT (audit 2026-10-18; PI 2026-09-27 "must not be exceeded"): the CLAMP on the emitted
    # plan, beside the argmax filter above. INFERENCE ONLY. Needs the ceiling channel (else every limit is +inf).
    if bool(getattr(args, "plan_vmax_clamp", False)):
        if not bool(getattr(args, "max_speed_input_v6", False)):
            raise SystemExit(
                "[v3] ⛔ --plan-vmax-clamp without --max-speed-input-v6: the clamp's limit IS the fed 4-way set "
                "speed; with no channel every row's limit is +inf and the clamp is built and INERT.")
        cfg.plan_vmax_clamp = True
''')
d = E.edit(d, "    # ---- refcv7 (PI 2026-09-19): DrivoR's proven planning heads ---------- #\n", '''    g6t.add_argument("--plan-vmax-clamp", action="store_true",
                     help="v7F R1 equivalent (PI 2026-09-27: the max speed 'must not be exceeded'): CLAMP the "
                          "EMITTED plan to the fed ceiling at INFERENCE (refcv6_selection.cap_plan_to_limit), "
                          "beside --speed-ceiling-filter, which only CHOOSES among the fan and leaves a row with "
                          "no compliant candidate uncapped. Needs --max-speed-input-v6. The TRAINING forward "
                          "stays uncapped. Default OFF = byte-identical. CHANGES THE ARM.")
''', mode="before")
d = E.edit(d, '        "speed_ceiling_filter": bool(getattr(core, "speed_ceiling_filter", False)),\n',
           '        "plan_vmax_clamp": bool(getattr(cfg, "plan_vmax_clamp", False)),\n')
E.record(ROWS, REL, tip, d, "edit")
E.write(REL, d)

# --------------------------------------------------------------------------------------------------------------- #
# 4. declared_vs_built.py (CRLF) : the G-DVB entry                                                                 #
# --------------------------------------------------------------------------------------------------------------- #
REL = "stack/tanitad/train/declared_vs_built.py"
tip = E.read(REL)
d = E.edit(tip, '_b("goal_point_inject", _c_goal_point)\n', '''# v7F R1 equivalent (audit 2026-10-18): argv --plan-vmax-clamp == the BUILT model's cfg.plan_vmax_clamp
_b("plan_vmax_clamp", _c_attr("plan_vmax_clamp", "plan_vmax_clamp",
                              lambda a: bool(_a(a, "plan_vmax_clamp", False)),
                              owner=lambda m: m.cfg,
                              where="model.cfg.plan_vmax_clamp (the E9 block clamps the emitted plan)"))
''', mode="before")
E.record(ROWS, REL, tip, d, "edit")
E.write(REL, d)

# --------------------------------------------------------------------------------------------------------------- #
# 5. test_declared_vs_built.py (CRLF) : the registry-count pin moves 278 -> 279 (a new lever adds its entry)       #
# --------------------------------------------------------------------------------------------------------------- #
REL = "stack/tests/test_declared_vs_built.py"
tip = E.read(REL)
d = E.edit(tip, "    assert len(dvb.REGISTRY) == 278\n",
           "    # + v7F R1 equivalent (audit 2026-10-18): --plan-vmax-clamp (\"built\": model.cfg.plan_vmax_clamp) (278 -> 279)\n"
           "    assert len(dvb.REGISTRY) == 279\n", mode="replace")
E.record(ROWS, REL, tip, d, "edit")
E.write(REL, d)

Path(E.OUT / "Project Steering/audits/2026-10-18-refcv8-v7f-fixes/raw").mkdir(parents=True, exist_ok=True)
(E.OUT / "Project Steering/audits/2026-10-18-refcv8-v7f-fixes/raw/r1_edited_files.json").write_text(
    json.dumps(ROWS, indent=1), encoding="utf-8")
print(json.dumps(ROWS, indent=1))
