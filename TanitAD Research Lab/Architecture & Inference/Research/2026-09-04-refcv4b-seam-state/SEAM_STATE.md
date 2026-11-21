# refcv4b — the measured state of the hierarchy seam, and two record gaps

`MEASURED 2026-09-04 by the Master Mind against branch HEAD and the LIVE run's own
config/argv on pod tanitad-refcv3. Run: /workspace/experiments/refcv4b-b1-v72-40k,
step ~700 of 40,284.`

## Why this note exists

The refcv4b launch resolved "Defect A" (an 8-wide tactical vocabulary being indexed by
a function whose positional contract is 3-wide) with option **(a)**: call
`derive_man5_logprobs` only on the kin3 vocabulary it is defined on. The launch report
stated the honest cost as *"the tactical brain no longer drives the anchor prior; it
reaches the decoder through E7/E9 only."*

That sentence is **true of the EXTERNAL tactical port and false of the seam as a whole**,
and the difference decides what this arm may claim about the programme's thesis. Read from
source rather than inferred.

## What is actually live (MEASURED, source + live config)

| flag | value | where established |
|---|---|---|
| `hierarchy` | **True** | `refc.py:490` default; `refc_v3.py:640` refuses `hier` without it |
| `factored_maneuver` | **True** | forced at `refc_v3.py:437` and `:615` — *"both arms, never a lever"* (overrides the `refc.py:493` default of False) |
| `graft_maneuver` | **True** (H19) | `refc.py:491` default |
| `graft_prior_center` | **True** | `refc.py:511` default |
| `tactical_speed_input` | **False** | live `config.json` → `ego.tactical_speed_input` |

The executed path (`refc.py:2290-2305`):

```
tac_in = pooled                          # tactical_speed_input False -> VISION-PURE (E11)
h_tac  = tactical_trunk(tac_in)
lat_logits = lat_head(h_tac)             # 3-wide kin3
lon_logits = lon_head(h_tac)             # 3-wide kin3
man_logits = derive_man5_logprobs(lat_logits, lon_logits)     # contract satisfied
reweight   = maneuver_logits if maneuver_logits is not None else man_logits
```

⇒ **H19 IS LIVE AND CORRECTLY FED.** The anchor prior is reweighted every step by the
model's own factored kin3 tactical heads, on a vision-only input. What is dead is the
**external** port: nothing fills `maneuver_logits`, so the `invert_man5` branch
(`refc.py:2306-2313`, an outside tactical brain speaking the 5-way surface) never runs.

## The claim this arm can and cannot support

- ✅ **CAN test**: does the model's own tactical prediction improve its own anchor
  selection — an INTERNAL hierarchy seam, measurable by ablating `graft_maneuver`.
- ⛔ **CANNOT test**: whether the **v7 8-wide tactical vocabulary** drives selection. That
  vocabulary reaches the decoder only through E7/E9 latent conditioning in this arm.

⚠️ Note the irony worth keeping: the comment at `refc.py:2306-2309` says a silently-ignored
external prior *"is exactly the class of bug this seam exists to remove"* — and under v7
the port is simply never filled, which is the same outcome reached quietly rather than by a
bug. Not a code defect; a wiring gap, and it belongs to the hierarchy-wiring stream.

## Two record gaps found on the way (both real, both cheap)

1. **`config.json` does not serialise the seam booleans.** `hierarchy`, `graft_maneuver`,
   `factored_maneuver`, `graft_prior_center`, `lan_enable`, `goal_str` are all absent at
   every nesting level; only `selection.*`, `ego.*` and `tac_vocab_version` are recorded.
   The arm is reconstructible today **only** by knowing that `refc_v3.py` forces them. A
   run record that cannot rebuild its own model config is not a run record.
   ⚠️ Absence was confirmed by walking the whole tree, not by one key lookup — the first
   probe read `<absent>` for all twelve flags purely because they sit one level down.
2. **`anchors.pt` carries no units field** — keys are exactly `anchors` and `controls`,
   and column 1 is lateral acceleration (m/s²), not curvature. Read as curvature the same
   bytes give **396 g** at 36 m/s and 104/117 anchors over a μ=0.7 friction circle; read
   correctly they give **0.31 g** and 0/117. Both tables look plausible. I produced the
   396 g one from the shipped file this morning.
   ⭐ **The narrowing that matters:** `config.json['argv']` **does** record
   `--anchor-control-units alat`, and the trainer's load line prints
   `units=alat, (accel, lateral accel)`. So the **run** is unambiguous and reproducible;
   only the standalone `.pt`, read in isolation, is not. The fix is a units field in the
   artifact, not a correction to the run.

## Evidence class

All rows MEASURED: source read from `git cat-file blob HEAD:stack/tanitad/refs/refc.py`
(146,231 B) and `…/refc_v3.py`; flags from the live run's `config.json`; units from its
`argv` and from `train.log`'s anchor-load line. Anchor identity verified by content —
`torch.equal(ckpt["model"]["core.decoder.anchors"], anchors.pt["anchors"])` **True** at
step 500, both hashing to `51f930dc6f3564ff`.

⛔ No number here is a driving result. The vocabulary's held-out oracle-in-vocabulary
figure is MODEL-FREE and is a ceiling, not an achievement; this arm has no T1 number.
