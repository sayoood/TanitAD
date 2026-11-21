# -*- coding: utf-8 -*-
"""Apply the --max-speed-input wiring to a LOCAL copy of refc_v3.py.

Exact-anchor replacement only: every anchor must appear EXACTLY ONCE or the
script refuses. A fuzzy patch into a contended file is how a diff lands on
something that moved under it.
"""
from __future__ import annotations

import io
import sys

P = sys.argv[1]
src = io.open(P, encoding="utf-8").read()
N = 0


def rep(anchor: str, new: str, label: str) -> None:
    global src, N
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(f"ANCHOR {label!r}: found {n} times, need exactly 1")
    if new in src:
        raise SystemExit(f"ANCHOR {label!r}: replacement already present")
    src = src.replace(anchor, new, 1)
    N += 1


# --------------------------------------------------------------------- A: import
rep(
    "from tanitad.refs import goal_point as gp\nfrom tanitad.refs import refc\n",
    "from tanitad.refs import goal_point as gp\n"
    "from tanitad.refs import max_speed_input as msi\n"
    "from tanitad.refs import refc\n",
    "import")

# --------------------------------------------------------------------- B: config
rep(
    "    ego_state_inject: bool = False\n"
    "    d_ego: int = 32                   # ego embed width (~= d_nav's 64 / 2)\n",
    "    ego_state_inject: bool = False\n"
    "    d_ego: int = 32                   # ego embed width (~= d_nav's 64 / 2)\n"
    "\n"
    "    # --- ⭐⭐ E16: the MAX-SPEED (map/nav posted-limit) INPUT ---------------\n"
    "    # PI 2026-09-01, reaffirmed 2026-09-06: *\"introduce an input in the models\n"
    "    # which corresponds the upper band value of speed band\"*, then *\"good idea\n"
    "    # with quantization\"*.\n"
    "    #\n"
    "    # ⛔ DEFAULT FALSE. With it off this class builds today's refcv3/refcv4b/\n"
    "    # refcv5 BIT-IDENTICALLY — no module is constructed, so not one RNG draw\n"
    "    # moves — which matters because a live 40,284-step run resumes through this\n"
    "    # file. Pinned by tests/test_max_speed_wiring.py::test_v3_parity_when_off,\n"
    "    # which compares against the PRE-PATCH file itself.\n"
    "    #\n"
    "    # ⭐ WHY IT IS ADMISSIBLE. `v_max_ms` STANDS IN FOR A MAP/NAV SPEED-LIMIT\n"
    "    # SERVICE exactly as `nav_command` stands in for the nav system, and the PI\n"
    "    # has ruled that class is an INPUT, not a training signal. We have no map;\n"
    "    # this is the simulator for one.\n"
    "    #\n"
    "    # ⚠️ WHAT IT IS TRAINED ON, STATED SO NOTHING IS OVERSOLD. The label's\n"
    "    # provenance is `ego-future`: max of the ego's OWN REALISED speed over\n"
    "    # [anchor+2 s, +6 s]. That is a TRAIN/DEPLOY MISMATCH, not a leak — at\n"
    "    # deployment the supplier is a posted limit the driver may not reach.\n"
    "    # Quantization is COSMETIC as a leak fix (bin + v0 recovers R^2 0.9702 vs\n"
    "    # v0-alone 0.8789, i.e. 75.4 % of the future survives) and REAL as a\n"
    "    # SEMANTICS fix (raw v_hi sits BELOW the ego's own current speed on 34.8 %\n"
    "    # of clips — incoherent for a ceiling; quantized, 8.4 %).\n"
    "    # ⚠️ AND ITS RESIDUAL DEFECT: snapping UP from a STOPPED ego reports the\n"
    "    # LOWEST limit (75 % of intersection clips get <= 30 km/h where a map would\n"
    "    # say 50), so the channel can teach \"slow ego => low limit\". Not fixable in\n"
    "    # this design; it needs a real map.\n"
    "    #\n"
    "    # ⛔ EVAL OBLIGATION, inseparable from this edge and identical to E13's: any\n"
    "    # max-speed-conditioned result carries a SHUFFLE control (serve another\n"
    "    # clip's ceiling) and a WITHHOLD control (valid = 0). Without them \"the arm\n"
    "    # improved\" cannot be separated from \"the arm gained a parameter\".\n"
    "    max_speed_input: bool = False\n"
    "    max_speed_cfg: msi.MaxSpeedConfig = field(\n"
    "        default_factory=msi.MaxSpeedConfig)\n",
    "config")

# --------------------------------------------------------------------- C: flat guard
rep(
    "        self.core = refc.RefCModel(cfg.core)\n"
    "        if not cfg.hier:\n"
    "            return\n",
    "        # ⛔ E16 IS A HIERARCHY EDGE, AND A FLAT BUILD MUST SAY SO RATHER THAN\n"
    "        # DROP IT. The two injection sites are `z_tac` and `ctx`, both of which\n"
    "        # exist only under `hier`; a flat arm would construct the conditioner,\n"
    "        # never call it, and report as +max-speed while running without it —\n"
    "        # the same silent-drop class the `ego_state` / `nav_args` / `gp_point`\n"
    "        # guards below refuse. Raised BEFORE the `not cfg.hier` return so it is\n"
    "        # reached on the flat path too (the `preflight`-only lesson).\n"
    "        if getattr(cfg, \"max_speed_input\", False) and not cfg.hier:\n"
    "            raise ValueError(\n"
    "                \"max_speed_input=True on a FLAT build: the ceiling's two\"\n"
    "                \" injection sites (z_tac, ctx) exist only in the hierarchy,\"\n"
    "                \" so the channel would be built and never read. Use\"\n"
    "                \" --arm hier, or turn the flag off.\")\n"
    "        self.core = refc.RefCModel(cfg.core)\n"
    "        if not cfg.hier:\n"
    "            return\n",
    "flat-guard")

# --------------------------------------------------------------------- D: constructor
rep(
    "        else:\n"
    "            self.ego_inj = None\n",
    "        else:\n"
    "            self.ego_inj = None\n"
    "        # ⭐ E16 — the speed ceiling gets its own conditioner, structurally\n"
    "        # identical to `nav_to_tac`/`nav_to_str` (E13) and `ego_to_tac`/\n"
    "        # `ego_to_str` (E11'). `MaxSpeedConditioner` zero-inits BOTH output\n"
    "        # projections itself, so an ON build's EMISSION is bit-identical at\n"
    "        # step 0 and an ON-vs-OFF comparison is attributable.\n"
    "        # ⛔ ONE CONSTRUCTION SITE, GATED. The D-ROLL-1 regression was a head\n"
    "        # built unconditionally: 11,286 params no recorded `param_breakdown`\n"
    "        # named, which made refcv4b, three refcv3 checkpoints and a LIVE\n"
    "        # refcv5 run unrollable. The gate is the flag, and\n"
    "        # `param_breakdown_v3` reads the BUILT OBJECT rather than re-deriving\n"
    "        # this condition, so the ledger cannot drift from the constructor.\n"
    "        if cfg.max_speed_input:\n"
    "            self.max_speed_cond = msi.MaxSpeedConditioner(\n"
    "                cfg.d_tac, d_ctx, cfg.max_speed_cfg)\n"
    "        else:\n"
    "            self.max_speed_cond = None\n",
    "ctor")

# --------------------------------------------------------------------- E: _hook sig
rep(
    "    def _hook(self, cache: dict, nav_cmd: Tensor | None = None,\n"
    "              ego_state: Tensor | None = None,\n"
    "              nav_args: Tensor | None = None):\n",
    "    def _hook(self, cache: dict, nav_cmd: Tensor | None = None,\n"
    "              ego_state: Tensor | None = None,\n"
    "              nav_args: Tensor | None = None,\n"
    "              v_max_ms: Tensor | None = None,\n"
    "              v_max_valid: Tensor | None = None):\n",
    "hook-sig")

# --------------------------------------------------------------------- F: injection
rep(
    "                z_tac_raw = z_tac_raw + self.ego_to_tac(ego_e)    # -> tactical\n"
    "                ctx = ctx + self.ego_to_str(ego_e)                # -> strategic\n",
    "                z_tac_raw = z_tac_raw + self.ego_to_tac(ego_e)    # -> tactical\n"
    "                ctx = ctx + self.ego_to_str(ego_e)                # -> strategic\n"
    "            # ⭐⭐ E16 — THE MAX-SPEED CEILING REACHES THE SAME TWO NODES.\n"
    "            # It enters EXACTLY where `v0` and the nav command already enter and\n"
    "            # nowhere else; nothing downstream is reshaped.\n"
    "            # ⚠️ MEASURED on the linear readout: a SHUFFLED ceiling collapses\n"
    "            # EXACTLY onto ego-only on every longitudinal target (delta +0.0002,\n"
    "            # NOT separated), so the shuffle control has teeth — which is why\n"
    "            # it is an obligation on every result this edge produces.\n"
    "            ms_injected = False\n"
    "            if getattr(self, \"max_speed_cond\", None) is not None \\\n"
    "                    and v_max_ms is not None:\n"
    "                ms_tac, ms_str = self.max_speed_cond(v_max_ms, v_max_valid)\n"
    "                z_tac_raw = z_tac_raw + ms_tac                    # -> tactical\n"
    "                ctx = ctx + ms_str                                # -> strategic\n"
    "                ms_injected = True\n",
    "inject")

# --------------------------------------------------------------------- G: cache stamp
rep(
    "                         ego_injected=bool(ego_e is not None),\n"
    "                         nav_injected=bool(nav_t is not None),\n",
    "                         ego_injected=bool(ego_e is not None),\n"
    "                         nav_injected=bool(nav_t is not None),\n"
    "                         max_speed_injected=bool(ms_injected),\n",
    "cache")

# --------------------------------------------------------------------- H: forward sig
rep(
    "                agent_gt: dict | None = None,\n"
    "                nav_args: Tensor | None = None) -> dict:\n",
    "                agent_gt: dict | None = None,\n"
    "                nav_args: Tensor | None = None,\n"
    "                v_max_ms: Tensor | None = None,\n"
    "                v_max_valid: Tensor | None = None) -> dict:\n",
    "fwd-sig")

# --------------------------------------------------------------------- I: refusal
rep(
    "        if not self.cfg.hier:\n"
    "            return self.core(frames, nav_cmd, v0, steps=steps, lan=lan,\n",
    "        # ⛔ SAME REFUSAL AS `ego_state`, `gp_point` AND `nav_args` ABOVE, AND\n"
    "        # FOR THE SAME REASON. A build without the E16 seam would SILENTLY DROP\n"
    "        # the ceiling and the arm would read as \"+max-speed does not help\" while\n"
    "        # never having had it — a refutation manufactured by a wiring gap.\n"
    "        if v_max_ms is not None and not getattr(self.cfg,\n"
    "                                                \"max_speed_input\", False):\n"
    "            raise ValueError(\n"
    "                \"v_max_ms was supplied but this build has no E16 seam \"\n"
    "                \"(`cfg.max_speed_input` is False) - it would be SILENTLY \"\n"
    "                \"DROPPED and the arm would report as +max-speed-input while \"\n"
    "                \"running without a ceiling. Build with --max-speed-input, or \"\n"
    "                \"stop passing v_max_ms.\")\n"
    "        if v_max_ms is None and getattr(self.cfg, \"max_speed_input\", False) \\\n"
    "                and nav_cmd is not None:\n"
    "            raise ValueError(\n"
    "                \"this build is --max-speed-input but no v_max_ms reached the \"\n"
    "                \"forward while a nav token did. Refusing rather than \"\n"
    "                \"defaulting: a silent 0.0 in the value channel reads as 'the \"\n"
    "                \"limit here is 0 m/s - stop', which is a LIE, not a missing \"\n"
    "                \"value. Pass v_max_valid=0 for 'no limit known'.\")\n"
    "        if not self.cfg.hier:\n"
    "            return self.core(frames, nav_cmd, v0, steps=steps, lan=lan,\n",
    "fwd-refuse")

# --------------------------------------------------------------------- J: call site
rep(
    "                        hierarchy_hook=self._hook(cache, nav_cmd, ego_state,\n"
    "                                                  nav_args),\n",
    "                        hierarchy_hook=self._hook(cache, nav_cmd, ego_state,\n"
    "                                                  nav_args, v_max_ms,\n"
    "                                                  v_max_valid),\n",
    "call-site")

# --------------------------------------------------------------------- K: ledger
rep(
    "        if getattr(model, \"tac_goal_tok_head\", None) is not None:\n"
    "            out[\"tac_goal_tok_head\"] = cnt(model.tac_goal_tok_head)\n",
    "        if getattr(model, \"tac_goal_tok_head\", None) is not None:\n"
    "            out[\"tac_goal_tok_head\"] = cnt(model.tac_goal_tok_head)\n"
    "        # ⭐ E16 — the max-speed conditioner, on its own ledger line and read\n"
    "        # off the BUILT OBJECT (never by re-deriving `cfg.max_speed_input`).\n"
    "        # ⛔ THIS LINE IS THE ROLLABILITY CONTRACT. D-ROLL-1: parameters that\n"
    "        # exist and are not accounted make every banked checkpoint unrollable,\n"
    "        # because `refcv3_arm.cross_check_config` compares the recorded ledger\n"
    "        # against a rebuild key-for-key and correctly refuses a mismatch.\n"
    "        # `test_param_breakdown_smoke_sums` fires the moment they are missing.\n"
    "        if getattr(model, \"max_speed_cond\", None) is not None:\n"
    "            out[\"max_speed_inject\"] = cnt(model.max_speed_cond)\n",
    "ledger")

io.open(P, "w", encoding="utf-8", newline="").write(src)
print(f"OK: {N} hunks applied to {P}")
