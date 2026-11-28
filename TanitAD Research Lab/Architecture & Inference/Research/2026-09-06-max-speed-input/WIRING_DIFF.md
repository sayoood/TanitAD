# ESCALATION — the `--max-speed-input` wiring into `refc_v3.py`

**Owner action required. This patch is NOT applied.** `stack/tanitad/refs/refc_v3.py`
has live siblings and a live training resumes through it, so the diff is filed
rather than landed. Everything it needs already exists and is committed:
`stack/tanitad/refs/max_speed_input.py` (the channel) and
`stack/tests/test_max_speed_input.py` (29 tests, incl. the OFF proof and its
demonstrated-red mutation control).

⚠️ **It is four hunks, not two.** The two INJECTION SITES are tactical and
strategic — both inside one hunk in `_hook` — but a working patch also needs the
config flag, the constructor, and one `forward` parameter. Calling it "a two-site
diff" would understate what an owner has to review, so it is counted honestly.

**Anchors verified against `refc_v3.py` at commit `a1f5ade`** (lines 891-899,
1031-1033, 1107-1109, 1277-1287, 1388-1391). If those anchors have moved, re-derive
before applying — do not fuzzy-match a patch into a file that has changed under it.

---

## HUNK 1 — the flag (`RefCV3Config`, beside `ego_state_inject` / `d_ego`)

```python
    ego_state_inject: bool = False
    d_ego: int = 32                   # ego embed width (~= d_nav's 64 / 2)

+   # --- ⭐ E16: the MAX-SPEED (map/nav posted-limit) INPUT (PI 2026-09-01,
+   # reaffirmed 2026-09-06) ------------------------------------------------
+   # ⛔ DEFAULT FALSE. With it off this class builds today's refcv3/refcv4b
+   # BIT-IDENTICALLY, which matters because a live run resumes through this
+   # file. Zero-init projections, exactly like E13 (nav) and E11' (ego), so an
+   # ON build is also bit-identical AT STEP 0 and an ON-vs-OFF comparison is
+   # attributable.
+   #
+   # ⭐ WHY IT IS ADMISSIBLE. `v_hi_ms` STANDS IN FOR A MAP/NAV SPEED-LIMIT
+   # SERVICE exactly as `nav_command` stands in for the nav system, and that
+   # class is an INPUT, not a training signal (PI). We have no map; this is the
+   # simulator for one.
+   #
+   # ⛔ THE SOURCE IS OPTIMISTIC BY CONSTRUCTION and `quantized` mode is the
+   # fix: the raw value is max(the ego's OWN realised speed over 2-6 s), so it
+   # is never violated (MEASURED: 0 / 4,572 clips) and is BELOW the ego's
+   # current speed on 34.8 % of clips, which is incoherent for a ceiling.
+   # Snapping UP to the pinned posted-limit ladder cuts that to 8.4 % and makes
+   # the ceiling violable on 0.61 %. Evidence and the ladder-selection table:
+   # `TanitAD Research Lab/Architecture & Inference/Research/
+   #  2026-09-06-max-speed-input/`.
+   max_speed_input: bool = False
+   max_speed_cfg: msi.MaxSpeedConfig = field(
+       default_factory=msi.MaxSpeedConfig)
```

plus, at the top of the file with the other `tanitad.refs` imports:

```python
+from tanitad.refs import max_speed_input as msi
```

---

## HUNK 2 — the constructor (`RefCV3Model.__init__`, after the `ego_inj` block, ~line 899)

```python
        else:
            self.ego_inj = None
+
+       # ⭐ E16 — the speed ceiling gets its own conditioner, structurally
+       # identical to `nav_to_tac`/`nav_to_str` and `ego_to_tac`/`ego_to_str`.
+       # `MaxSpeedConditioner` zero-inits BOTH output projections itself.
+       if cfg.max_speed_input:
+           self.max_speed_cond = msi.MaxSpeedConditioner(
+               cfg.d_tac, d_ctx, cfg.max_speed_cfg)
+       else:
+           self.max_speed_cond = None
```

---

## HUNK 3 — the two injection sites (`_hook`, after the E11' ego block, ~line 1109)

```python
    def _hook(self, cache: dict, nav_cmd: Tensor | None = None,
              ego_state: Tensor | None = None,
-             nav_args: Tensor | None = None):
+             nav_args: Tensor | None = None,
+             v_max_ms: Tensor | None = None,
+             v_max_valid: Tensor | None = None):
```

```python
                z_tac_raw = z_tac_raw + self.ego_to_tac(ego_e)    # -> tactical
                ctx = ctx + self.ego_to_str(ego_e)                # -> strategic
+           # ⭐⭐ E16 — THE MAX-SPEED CEILING REACHES THE SAME TWO NODES.
+           # It enters EXACTLY where `v0` and the nav command already enter and
+           # nowhere else; nothing downstream is reshaped.
+           # ⛔ EVAL OBLIGATION, inseparable from this edge and identical to
+           # E13's: any max-speed-conditioned result carries a SHUFFLE control
+           # (serve another clip's ceiling) and a WITHHOLD control (valid = 0).
+           # Without them "the arm improved" cannot be separated from "the arm
+           # gained a parameter". MEASURED on the linear readout: the shuffled
+           # channel collapses EXACTLY onto ego-only on every longitudinal
+           # target (delta +0.0002, not separated), so the control has teeth.
+           ms_e = None
+           if self.max_speed_cond is not None and v_max_ms is not None:
+               ms_tac, ms_str = self.max_speed_cond(v_max_ms, v_max_valid)
+               z_tac_raw = z_tac_raw + ms_tac                    # -> tactical
+               ctx = ctx + ms_str                                # -> strategic
+               ms_e = True
```

and, in the `cache.update(...)` block that already records `ego_injected` /
`nav_injected`:

```python
+                        max_speed_injected=bool(ms_e),
```

---

## HUNK 4 — `forward` (signature ~line 1277, call site ~line 1390)

```python
                 agent_gt: dict | None = None,
-                nav_args: Tensor | None = None) -> dict:
+                nav_args: Tensor | None = None,
+                v_max_ms: Tensor | None = None,
+                v_max_valid: Tensor | None = None) -> dict:
```

```python
                        hierarchy_hook=self._hook(cache, nav_cmd, ego_state,
-                                                 nav_args),
+                                                 nav_args, v_max_ms,
+                                                 v_max_valid),
```

---

## The loader side (`refc_v3_train.py`) — one read, and it must REFUSE

```python
+from tanitad.refs import max_speed_input as msi
+
+# ⛔ UNITS ARE REQUIRED, NOT ASSUMED. `read_max_speed_field` raises
+# MaxSpeedUnitsMissing on a payload that declares none. Do NOT wrap this in a
+# try/except that defaults to m/s -- that is the 396 g anchor table again.
+smi = rec.get("speed_max_input") or {}
+got = msi.read_max_speed_field(smi, mode=args.max_speed_mode)
+v_max_ms = got["value_ms"] if got["valid"] else 0.0
+v_max_valid = 1.0 if got["valid"] else 0.0
```

and the CLI flag:

```python
+p.add_argument("--max-speed-input", action="store_true",
+               help="feed the map/nav posted-limit ceiling (default OFF)")
+p.add_argument("--max-speed-mode", choices=msi.MODES,
+               default=msi.DEFAULT_MODE,
+               help="quantized (default, snaps UP to real posted limits) or "
+                    "raw (unquantized, for comparison only)")
```

and into `config.json`, so an arm is identifiable from its own artifacts:

```python
+cfg_json["max_speed_input"] = msi.artifact_meta(args.max_speed_mode)
```

---

## Acceptance for whoever lands it

1. `pytest stack/tests/test_max_speed_input.py -q` — 29 passed.
2. `python "<pkg>/code/mutation_control_demo.py"` — baseline PASS, mutated FAIL,
   restored PASS. **If the mutated run does not go red, the OFF proof is vacuous
   and must not be quoted.**
3. Add a `test_v3_parity_when_off` to the applied patch, in
   `test_goal_point_wiring.py`'s shape: flag OFF ⇒ no `max_speed_*` key in
   `state_dict()`, and `max_speed_injected is False`. ⛔ That test is the ONE
   thing this package could NOT provide, because it can only be written against
   the patched file.
4. Any arm trained with the flag ON reports the SHUFFLE and WITHHOLD controls
   beside its headline, or it is not evidence.
