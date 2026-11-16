# BUILD_SPEC: STRAT1 (nav text + detected strategic-information heads), v8f-mini policy fine-tune (Architecture agent, 2026-11-16)

For the Sonnet implementer. Design only; nothing here is built, trained or launched. Card: `PREREG_V8F_STRAT1_DRAFT.md`. Labels: `LABEL_SPEC.md`. Reads: `READ_PLAN.md`.
Evidence classes: MEASURED (M, artifact named), INHERITED (I), ESTIMATED (E), HYPOTHESIS (H), UNVERIFIED (U). Every `file:line` is at the base commit unless stated.

## 0. Base commit (decision, escalated)

**Base: `cdbff971fa6dad4d20b937fe8abda6cb9f622adb`** (`origin/agent/v8f-otband-20261109`), not `81cee6c1` (`agent/v8f-actsens-20261115`).

* The control is the **recipe of record OBD-CTL** = DAG-OACC + `--detach-upper` (I, register row `H-V8F-OBD-1`, "recipe of record OBD-CTL"). Its code of record is `cdbff971` (`PREREG_V8F_OBD.md` line 3; M, `t2v8f_pulls/obd_OBD-CTL_s*/RUN.done` `commit` = `cdbff971...`). Its gate read `otband_probe` and `acc_probe` (M, `obd_OBD-CTL_s0/selfcheck.json`), which exist only on this line.
* `81cee6c1` = ladder tip `1faa1a06` + two ACTSENS commits (`--wm-idm-gamma`, `--wm-sampler`), all WM-only and default-off (M, `git log 1faa1a06..81cee6c1`). It lacks the three OTBAND commits `8e9499e0`, `202849da`, `cdbff971` (M, `git merge-base cdbff971 81cee6c1` = `1faa1a06`). A policy arm uses none of the ACTSENS code.
* Both lines carry `1faa1a06` (window drop on latent-cache-not-ok), so the window set is the same on either base (I; the implementer re-checks `train_window_list_sha256` in the selfcheck).
* `agent/arch-inf-20260803` (`b84521ef`) has the v8f-mini files of the merge base `6a3fa0d7` (M, `git diff --stat 6a3fa0d7 b84521ef` touches only `launch_gate.py` among them), so it lacks both lines.
* A merge of the two lines would be new, ungated code; not done.
* `v8f_mini_eval.py` and `v8f_t1_eval.py` are byte-identical at `cdbff971` and `81cee6c1` (M, `cmp`).
* The implementer works on a new branch from `cdbff971` (suggested `agent/v8f-strat1-20261116`), stages only, never commits or pushes.

**If the MM rules for `81cee6c1` instead**, every change below applies by anchor (function name); line numbers then shift.

## 1. Information flow (binding: goal-input rule, vision-only rule)

```
feats (front camera tokens), frame_ok ──► adapter ──► adapt ─┬─► S, T, O readers (unchanged) ──► ckpt, code, traj
                                                             └─► mean of NOW frame tokens ──► ctx_head ──► ctx_logits ──► loss + eval ONLY
nav (5 ch; source v9 OR text) ──► e_nav ──► S, T, O (unchanged wiring)
labels (context table, navtok) ──► loss / reader only
```

* `ctx_logits` is never passed to `run_s`, `run_t`, `run_o`, `e_code`, `e_ck` or any goal path. A test perturbs `ctx_head` weights and requires every planner output to stay bit-identical (section 9, T-ISO).
* Training coupling: with `--ctx-detach off` the ctx loss reaches the shared adapter, which S, T and O read. That is a training-time coupling, not an inference-time input. It is declared, and arm N1 (text nav, no heads) measures its effect on driving.
* The ctx head reads `feats` and `frame_ok` only. It reads no ego, nav, limit, metadata, clock or GPS.
* The nav text is a router product (GPS trace matched to OSM routing; I, NAVTOK_REPORT section headline 1). It is not built from the situation classifier. The goal and situation paths stay information-disjoint.

## 2. New flags (trainer), defaults, and G-DVB entries

Add to `build_parser()` after `--aux-cond` (`v8f_mini_train.py:189`). Add each to `FLAG_CHECKS` (`:70`, after the `lead_head` line `:115`). `declared_vs_built` refuses any parser dest missing from `FLAG_CHECKS` (`:1657-1660`).

| flag | dest | default | off-state | FLAG_CHECKS kind | read back from |
|---|---|---|---|---|---|
| `--nav-source {v9,text}` | `nav_source` | `v9` | v9 = the join's nav, bit for bit | `data:nav_source` | `info["nav_source"]` = the digest record `nav_source_record(J)` (section 5) |
| `--nav-text-npz PATH` | `nav_text_npz` | `""` | not read | `path` | must exist iff text, must be `""` iff v9 |
| `--nav-text-sha256 HEX` | `nav_text_sha256` | `""` | | `data:nav_text_sha256` | sha256 of the file actually loaded |
| `--ctx-heads {off,on}` | `ctx_heads` | `off` | nothing built, no key | `lever:ctx_heads` | `model.built_levers().get("ctx_heads","off")` |
| `--w-ctx FLOAT` | `w_ctx` | `0.0` | not in the total | `cfg:w_ctx` | built config; `> 0` iff on, exactly `0.0` iff off |
| `--ctx-labels PATH` | `ctx_labels` | `""` | not read | `path` | must exist iff on |
| `--ctx-labels-sha256 HEX` | `ctx_labels_sha256` | `""` | | `data:ctx_labels_sha256` | sha256 of the file loaded |
| `--ctx-lr-mult FLOAT` | `ctx_lr_mult` | `1.0` | no extra optimiser group | `opt:ctx_lr_mult` | the optimiser group named `ctx`: its lr / `--lr` |
| `--ctx-detach {off,on}` | `ctx_detach` | `off` | head trains the adapter | `lever:ctx_detach` | `built_levers()["ctx_detach"]` |

* `bad_ctx(a, model, data_info)` (new, after `bad_lead` `:1384`; called in `declared_vs_built` after `bad_lead` `:1653`) refuses: on with `w_ctx == 0`; off with `w_ctx != 0`; off with any `ctx_head.*` key; on without a head; on without `--ctx-labels`; sha mismatch; `ctx_lr_mult != 1` with the head off; `ctx_detach on` with the head off; a head whose `_ctx` source names a forbidden input; `nav_source text` without the npz or with a sha mismatch; `nav_source v9` with an npz named.
* Rule ("a flag whose default disables it must be named in argv"): **every STRAT1 argv names all nine flags**, the control included (section 10).

## 3. Model: `stack/tanitad/models/v8f_mini.py`

**Constants** (after `STOK_KEYS`, `:165`):
```python
CTX_VARS = ("country", "weather", "surface", "illumination", "period_of_day")   # order FROZEN = LABEL_SPEC section 2
CTX_N = (25, 4, 3, 4, 5)                  # = len(vocab_<var>) of strat_labels_table.npz (sha256 8055a321...); a test pins it
CTX_DIM = sum(CTX_N)                      # 41
CTX_VAR_W = (1.0, 1.0, 0.5, 1.0, 1.0)     # surface at half weight (borderline label, LABEL_SPEC); a constant, not a flag
CTX_TWILIGHT_ID = 3                       # illumination id masked in the loss (card: 0/9 recalled)
CTX_KEY_PREFIXES = ("ctx_head.",)
CTX_FORBIDDEN_NAMES = LEVEL_FORBIDDEN_NAMES + ("ctx_y", "nav_source", "country", "hour", "month", "platform")
CTX_ALLOWED_KEYS = frozenset({"frame_ok"})
CTX_INIT_SEED = 20261116
NAV_SOURCES = ("v9", "text")
CTX_HEAD_MODES = ("off", "on")
```
* `NEW_LEVER_KEY_PREFIXES` (`:154`): append `+ CTX_KEY_PREFIXES`.

**Config** (`V8FMiniConfig`, fields after `w_level` `:245`): `ctx_heads: str = "off"`, `w_ctx: float = 0.0`, `ctx_detach: str = "off"`, `nav_source: str = "v9"` (a DATA lever: the model never reads it; the checkpoint records it, as `window_start`). `__post_init__` (after the `w_lead` check, about `:282`): `ctx_heads`/`ctx_detach` in `CTX_HEAD_MODES`, `w_ctx` finite and >= 0, `nav_source in NAV_SOURCES`. The checkpoint stores `cfg.__dict__` (`v8f_mini_train.py:3522`), so the fields round-trip; an older checkpoint has none and builds `off` / `v9`.

**Build** (`__init__`, after the `e_stok` block `:868-870`, before `self.p_gt = 1.0` `:871`):
```python
self.ctx_head = None
if cfg.ctx_heads == "on":                     # built LAST, under a forked RNG: every module above and the global stream after it are the off model's
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(CTX_INIT_SEED)
        self.ctx_head = _mlp(d, d, CTX_DIM)   # Linear(256,256) GELU Linear(256,41): 76,329 params (arithmetic, d = 256)
```
`d = 256` for M2b (M: adapter 275,712 params = LN 2,048 + Linear 262,400 + pos 10,240 + time 1,024, `obd_OBD-CTL_s0/selfcheck.json` `levers.params`).

**Head** (new method after `_stok_tok`, `:953`):
```python
def _ctx(self, b, adapt: Tensor, last: Tensor) -> Tensor:
    tok = self._at(adapt, last)
    if self.cfg.ctx_detach == "on":
        tok = tok.detach()
    return self.ctx_head(tok.mean(1)).float()          # [B, CTX_DIM]; VISION ONLY (pinned by ctx_head_source_problems)
```
**Forward** (`_forward`, after the lead block `:1034-1035`): `if self.ctx_head is not None: out["ctx_logits"] = self._ctx(b, adapt, last_now)`. Nothing else in `_forward` changes. `forward_infer` returns it; `INFERENCE_INPUT_KEYS` (`:176`) is unchanged.

**Loss** (`loss`, after the lead block ending `:1211`):
```python
if cfg.ctx_heads == "on":
    if "ctx_y" not in batch:
        raise ValueError("ctx_heads on but the batch carries no ctx_y: the trainer stages it (ctx_stage) before any ego stage")
    y = batch["ctx_y"].long()                           # [B, 5], -1 = masked
    fok = batch["frame_ok"][:, -1]
    num, den, o = 0.0, 0.0, 0
    for v, (n, w) in enumerate(zip(CTX_N, CTX_VAR_W)):
        lg, t = out["ctx_logits"][:, o:o + n], y[:, v]
        m = (t >= 0) & fok
        ce = F.cross_entropy(lg, t.clamp_min(0), reduction="none")
        T[f"ctx/{CTX_VARS[v]}"] = (ce * m).sum() / m.sum().clamp_min(1)
        num, den, o = num + w * T[f"ctx/{CTX_VARS[v]}"], den + w, o + n
    T["ctx"] = num / den
```
The twilight mask is applied in the label arrays (section 4), not here. **Total** (after `:1244`), ONE line a mutation deletes: `total = total + cfg.w_ctx * T["ctx"] if cfg.ctx_heads == "on" else total`.

**Groups / levers**: `trainable_groups` (`:1279`): `if self.ctx_head is not None: g["CTX"] = [self.ctx_head]`. `built_levers` (before `return lv` `:1348`): when built, `ctx_heads="on"`, `w_ctx`, `ctx_detach`, `ctx_dim = self.ctx_head[2].out_features`, `ctx_vision_only_source = not ctx_head_source_problems()`. Absent when off (the key set of an off model is the tip's).

**Guards** (after `assert_lead_head_vision_only`, `:1558`):
* `ctx_head_source_problems(fn=None)`: `level_head_source_problems`'s AST walk (`:1428`) over `V8FMini._ctx` against `CTX_FORBIDDEN_NAMES`.
* `assert_ctx_head_vision_only(model, batch)`: the three clauses of `assert_level_head_vision_only` (`:1446`): static source; key-guarded read set equals `CTX_ALLOWED_KEYS`; `ctx_logits` bit-identical when every non-vision key (`p10f, nav, limit, limit_ok, act, pose*, base, kin3, a_t, a_s`, labels) is replaced, and different when `feats` is.
* `ctx_isolation_problems(model, batch)`: `forward_infer` twice, the second after adding N(0, 1) to every `ctx_head` parameter (forked RNG). `traj30`, `traj60`, `ckpt`, `goal_logits`, `lat_logits`, `lon_logits`, `cons_pred`, `code` must be bit-identical; `ctx_logits` must differ. Non-empty = the head feeds the planner or goal path.

## 4. Data: `stack/tanitad/data/v8f_mini_data.py`

New functions after `lead_label_arrays` (`:277`). **No change to `item` or `collate`.**

* `ctx_label_arrays(J, path, expect_sha256, split="train") -> dict`: re-hash the file, refuse a mismatch; assert `<split>_clip == J["clip"]`, `<split>_off == J["off"]`, `<split>_n == J["n"]`; build `y` int8 `[N, 5]` in `CTX_VARS` order with `-1` where `flag != 0` or `<var>_valid` is False, and illumination `== CTX_TWILIGHT_ID` set to `-1`; return `{"off", "y", "sha256", "n_valid": {var: int}, "n_twilight_masked": int}`. Never load `visibility` or any metadata column.
* `nav_text_arrays(J, path, expect_sha256, split) -> (nav [N,5] float32, nav_d [N] float32, rec)`: re-hash, assert the same clip/off/n equality, assert channel ranges (one-hot rows on 0-2 or all zeros; ch3 in [0, 3]; ch4 in {0, 1}; no NaN), return `rec = {"sha256", "nav_digest": sha256(nav.tobytes()), "n_text_ne_v9": int((nav != J["nav"]).any(1).sum())}`.

## 5. Offline nav-text builder (new): `stack/scripts/build_navtext_nav.py`

CPU only, numpy and torch (for `native_tracks.pt`). Runs once per split where the tracks live; output under 15 MB (E: train 461,518 x 5 float32 = 9.2 MB).

**Inputs**: `navtok_table.npz` (sha256 `2836369e3e14a8c3c49a6b1a42116c36cda522ca42f98a6bcbc8c4e4d45c8992`, I), the join (`join_train.npz` md5 `9ebc7a48cbca45cc634eab2aab5a3960`, M from `obd_OBD-CTL_s0/gate_token.json`; `join_eval141.npz`), the tracks (`native_tracks.pt` md5 `189f4e68913c0163912a509203963f8d`, M same file; the eval tracks the T1 eval reads, path U). Track row `2j` is cache step `j` (`v8f_t1_eval.py:164` `tr[:2*t0+1]`).

**Rule (fixed here, before any number)**, per step with table flag 0 (any other flag: all five channels 0, `nav_d` NaN, as the join's no-row steps):
1. `side`: `TURN_L, TURN_SHARP_L, TURN_SLIGHT_L` -> L; `TURN_R, TURN_SHARP_R, TURN_SLIGHT_R` -> R; every other class (UTURN, CONTINUE*, KEEP*, MERGE*, RAMP*, RB_*, ROTARY_EXIT, UNPARSED) -> none. This is NAVTOK's `coarse()` with UTURN set to none, as the draft N1.
2. No side: `[1, 0, 0, 0, 0]` (FOLLOW), `nav_d` NaN.
3. Side, distance present: **countdown** `d = max(0, dist_m - (arc(now_s) - arc(now_s - age_s)))` on the logged track's cumulative arc, interpolated in time (row times from `J["now_s"]` at even rows, linear between). A row start before the first finite time clamps to it (count reported).
4. Side, no distance (26 distinct bare texts, I): `d = 0`.
5. `d_ann = max(30.0, 6.0 * v_now)`, `v_now` = track speed at row `2j` (the v9 rule's literals, `build_v9_labels.py:44-45`, I).
6. Token = side if `d <= d_ann` else FOLLOW; `ch3 = clip(d / 100, 0, 3)`; `ch4 = 1`; `nav_d = d`.

**Why the countdown and not the raw text distance** (a change from `PREREG_NAVTEXT_DRAFT` N1, escalated): the text row holds until the next row, and the row age at a step is median 2.09 s, p90 15.45 s, p99 19.65 s (M, this session, `navtok_table.npz` `train_age_s` over flag-0 steps). At 10 m/s a p90 row is 150 m stale. The router a car would carry counts down live; the countdown is that, from logged odometry. The T1 eval's dead-reckoning already counts the same channel down from own motion (`v8f_t1_eval.py:216-220`), so training and T1 then share one semantics.

**Output** `navtext_nav.npz`: `{split}_clip/off/n` (copied), `{split}_nav` [N,5], `{split}_nav_d` [N], `{split}_rowclamp` (bool), `meta_json` (rule constants, class-to-side table, input sha256/md5s). Plus `navtext_nav_report.json` with its own sha256 and the QA below.

**G-CLOCK-NAV (launch blocker, both outcomes fixed now)**: on train steps where the v9 token is L or R, the text side equals it, and both distances are valid, `median |d_countdown - v9 nav_d|` must be `<= median |dist_raw - v9 nav_d|`. The deliberate-regression arm is the same QA with the sign flipped (`dist + travelled`): it must FAIL. If the minus form fails or the plus form passes, the time base or the arc is wrong: no launch. The report also gives the count of steps where the countdown changed the token against the raw rule (informational).

## 6. Trainer: `stack/scripts/v8f_mini_train.py`

1. **Parser + FLAG_CHECKS + make_config**: section 2; `make_config` (`:424`, beside `level_head=a.level_head` `:435`) passes `ctx_heads, w_ctx, ctx_detach, nav_source`.
2. **Nav overwrite**, after the join is loaded (`:3081-3090`) and before `--exclude` (`:3091`):
   ```python
   info["nav_source"] = {"source": a.nav_source, "join_nav_digest": sha256(J["nav"].tobytes())}
   if a.nav_source == "text":
       J["nav"], J["nav_d"], rec = D.nav_text_arrays(J, a.nav_text_npz, a.nav_text_sha256, "train")
       info["nav_source"].update(rec)
   ```
   The assignment is ONE line a mutation deletes (test T-NAV-MUT).
3. **Ctx labels**: `CL = make_ctx_labels(a, cfg, J, info)` beside `LL = make_lead_labels(...)` (`:3287`); `None` when off (nothing read). `ctx_stage(batch, CL)` (new, after `lead_stage` `:2155`): `rows = CL["off"][ci] + t`; returns `torch.as_tensor(CL["y"][rows])`. It is called on the probe batch (beside `:3343-3344` and `:3358`) and in the loop beside `lead_lab = lead_stage(b, LL)` (`:3563`), BEFORE `ego.apply` / `dag.apply` / `comp.apply` / `rcv.apply`.
4. **Init checkpoint**: `load_init_ckpt` (`:789`) already admits exactly the `NEW_LEVER_KEY_PREFIXES` keys as fresh; extend the gate line `:763` to `... or a.ctx_heads == "on"`. Do NOT add `nav_source` or the ctx fields to `INIT_CKPT_MATCH` (`:772`): a v9 M2b checkpoint is the declared start state of a text arm.
5. **Optimiser** (`:3312-3315`): with the head on, remove `ctx_head` parameters from `decay` / `nodecay` and append `{"params": ctx_decay, "weight_decay": a.wd, "lr": a.lr * a.ctx_lr_mult, "name": "ctx", "lr_mult": a.ctx_lr_mult}` and its no-decay twin. The loop already applies `g.get("lr_mult", 1.0)` per group (the LoRA precedent). With the head off, `groups` is the tip's list, bit for bit. Extend the read-back at `:1733` (`want_lrs`) and assert the ctx groups hold exactly the `ctx_head` parameters.
6. **G-LIVE**: `live_check` (`:1761`): `if cfg.ctx_heads == "on": expect.add("ctx")` beside `:1794-1795`; same in the LoRA twin (`:3041-3042`). Add `"CTX": "ctx_head"` to `groups_grad`.
7. **`ctx_head_probe(model, batch, cfg)`** (new, after `lead_head_probe` `:2164`), written to `sc["ctx_head_probe"]` beside `:3469-3470` (lever-on only):
   * (a) `total(w) - total(0) == w_ctx * T["ctx"]` (rtol 1e-3);
   * (b) `sum |grad ctx_head| > 0` at `w_ctx`, and **exactly 0.0** at `w_ctx = 0` (control);
   * (c) adapter: `sum |grad_adapter(w) - grad_adapter(0)| > 0` with detach off, **exactly 0.0** with detach on (control, run on a copy with `ctx_detach` set);
   * (d) all-masked control: `ctx_y` all -1 gives `T["ctx"] == 0.0` and head gradient **exactly 0.0**;
   * (e) `assert_ctx_head_vision_only` and `ctx_isolation_problems` empty;
   * (f) label staging: `ctx_y` of the probe batch equals an independent re-read of the table at `(ci, t)`.
8. **`nav_source_probe(model, batch, J, info)`** (new), to `sc["nav_source_probe"]` for every argv (the control's records `source v9`):
   * (a) `sha256(J["nav"])` equals the npz's declared digest (text) or the join's (v9);
   * (b) text: `n_text_ne_v9 > 0` (zero = the overwrite did not happen); v9: `== 0`;
   * (c) the probe batch's `nav` equals an independent read of the source rows at its history steps;
   * (d) reach: `sum |grad e_nav| > 0`, and `forward_infer` with `nav` set to zeros differs from the real one on `ckpt`.
9. **Checkpoint blob** (`:3521`): `cfg` carries the new fields; add `blob["nav_text_sha256"]`, `blob["ctx_labels_sha256"]` when used. `summary.json` (`:3694` style): `nav_source`, `ctx_heads`, `w_ctx`, per-var valid counts, `n_text_ne_v9`.

## 7. Evaluators

**`stack/scripts/v8f_mini_eval.py`** (T0; `cmd_run` `:233`):
* After `cfg = M.V8FMiniConfig(**ck["cfg"])` (`:237`): new `--nav-text-npz`, `--nav-text-sha256`. REFUSE if `cfg.nav_source == "text"` and no npz, or `"v9"` and an npz is named.
* After `ej = D.load_join(a.eval_join)` (`:267`): text: `ej["nav"], ej["nav_d"], _ = D.nav_text_arrays(ej, ..., "eval")`.
* New `--nav-withheld` (all five channels 0, the trained no-row state); refused together with `--nav-shuffle`. The existing donor path (`:274-276`, `donor_indices(..., seed=0)`) is unchanged.
* `outs` (`:282`): `"ctx_logits"` when `model.ctx_head is not None`. The `eval_<tag>.npz` (`:380`) adds `win_ci`, `win_t`, `nav_variant` (`real|donor|zero`) and `nav_source`.

**`stack/scripts/v8f_t1_eval.py`** (T1-kin):
* The same refusal and overwrite after `cfg` (`:506`) and `J = D.load_join(a.eval_join)` (`:515`).
* New `--nav-variant {real,donor,zero}`, default `real` = the tool's output bit for bit. donor: `WindowState.nav0` is the donor window's row (`donor_indices`, episode-disjoint, seed 0), dead-reckoned by own arc as now (`:216-220`). zero: `nav0 = 0`.
* New `--dump-nav` (default off): per tick, add `nav_in` (the `[5]` vector fed at NOW) to the `--dump-ticks` rows. With `--dump-heads` and a ctx model, add `ctx_logits`.

**G-EVAL**: `strat1_check.py geval --ckpt C` loads the checkpoint through each evaluator's own code path (`v8f_mini_eval.py run` and `v8f_t1_eval.py run` on 8 grid windows, CPU). It asserts a strict load, `built_levers()` equal to the selfcheck's `levers`, and `forward_infer` outputs bit-equal to the trainer's on the same 8 windows (CPU, 2 threads). Deliberate-regression arm: a checkpoint whose `cfg` drops `ctx_heads` must be REFUSED (unexpected keys).

## 8. Gate: `strat1_check.py` (pattern of `obd_check.py`, raw/2026-11-10-v8f-obd/box2/)

`launch_gate.py` has no v8f-mini profile (its `v8f` profile is rung alpha on the refcv8 recipe, `launch_gate.py:881-895` at `b84521ef`, M). Every v8f-mini arm to date was gated by a study checker around the trainer's own selfcheck (`obd_check.py gate`, `actsens_check.py gate`; M, `gate_token.json` schemas `obd-gate-token-v1`, `actsens-gate-token-v1`). STRAT1 follows that precedent with all seven G-checks mapped (ESCALATED: the MM confirms this satisfies the PI's 2026-09-26 rule).

| check | in STRAT1 |
|---|---|
| G-HYG | `V8FMiniConfig.__post_init__` refuses bad values; `V8FMiniConfig(**cfg)` refuses unknown keys (dataclass `TypeError`); test T-HYG |
| G-DVB | trainer `declared_vs_built` incl. `bad_ctx`, `FLAG_CHECKS` completeness; checker asserts every argv lever against `selfcheck.levers` |
| G-LIVE | `live_check` (all terms incl. `ctx`, `ctx/<var>`; no unexpected dead parameter), `ctx_head_probe`, `nav_source_probe`, the inherited `acc_probe`, `otband_probe` (detach-only), `dagger_probe` |
| G-CLOCK | the two label-time sources: `ctx_label_arrays` alignment asserts + LABEL_SPEC time base (ASSUMED, bound 1.6 %); G-CLOCK-NAV (section 5) |
| G-EVAL | section 7 |
| G-CKPT | save, load, bit-equal state dict and cfg; resume test (the trainer's `V8F_TRAIN_CRASH_AT` hook) on a lever-on synthetic run |
| G-SUITE | `pytest stack/tests` on a clean export of the branch tip, run in the pod environment, 0 regressions against `cdbff971` run in the same environment (memory note: pod-only test dependences exist) |

Token: `strat1-gate-token-v1`, bound to commit, argv sha256, data sha256/md5s (v9 labels, lead, speed, join, p10, tracks, navtext npz, ctx labels, M2b init, DAgger bank), tree manifest, trainer sha256, selfcheck sha256. Minted only by the checker. **Pilot**: 200 steps per first arm of each kind; the measured s/step projects the chain; the chain needs `<= 10 h` projected pod time, else STOP and report (E rule, mirrors OBD's 12 h permit). **Post**: moved-identity check from the checkpoint: every non-ctx group moved; `ctx_head.*` moved iff on; no `ctx_head.*` key in C0.

## 9. Tests (`stack/tests/test_v8f_strat1.py`, new; plus snapshot)

Each mechanism has a test and a mutation that must turn it RED (run by `tests/run_mutation_strat1.py`, copies the file, applies one edit, runs the named test, restores).

| id | asserts | mutation that must turn it RED |
|---|---|---|
| T-OFF | OFF == TIP: with all nine flags at default, state dict, loss terms, total, `forward_infer`, gradients bit-equal to the `cdbff971` module snapshot (`v8f_mini_tip_snapshot_strat1.txt`, sha256 pinned), same seed; no new key | build `ctx_head` unconditionally |
| T-RNG | ctx on: every non-ctx init tensor and the global RNG state after `__init__` equal the off model's | remove `fork_rng` |
| T-HYG | bad `ctx_heads`, negative `w_ctx`, unknown `nav_source`, unknown cfg key all raise | delete the `nav_source` check |
| T-CTX-LOSS | the ctx term reaches `total` with weight `w_ctx`; per-var CE equals a hand computation on a 3-sample fixture; twilight and -1 targets contribute exactly 0 | delete the ONE total line |
| T-GRAD | head gradient > 0 at `w_ctx`; exactly 0.0 at `w_ctx = 0`; adapter gradient difference exactly 0.0 with `ctx_detach on` | drop the `.detach()` in `_ctx` |
| T-VIS | `assert_ctx_head_vision_only` passes; source guard lists nothing | make `_ctx` add `b["p10f"]` |
| T-ISO | `ctx_isolation_problems` empty | feed `ctx_logits` into the ego token of `run_o` |
| T-LBL | `ctx_label_arrays` on a fixture: masks flag != 0, invalid, twilight; refuses a shifted `off` and a wrong sha | skip the twilight mask |
| T-STAGE | `ctx_stage` before the DAgger swap: a swapped row keeps its logged label | call it after `dag.apply` |
| T-NAV | `nav_text_arrays` refuses misalignment and a wrong sha; text and v9 differ on a fixture | none (refusal test) |
| T-NAV-MUT | the trainer's probe sees the text nav (n_text_ne_v9 > 0, digest equal) | delete the overwrite line |
| T-NAVB | builder rule on a fixture: countdown, d_ann, UTURN none, bare text d = 0, flag != 0 zeros | `+ travelled` (G-CLOCK-NAV must then FAIL on the fixture) |
| T-DAG | DAgger rebuild on a text-nav batch: channels 0-2 and 4 untouched, ch3 in [0, 3] | none (inherited rule) |
| T-INIT | M2b-shaped checkpoint into a ctx-on model: fresh keys exactly `ctx_head.*`; into ctx-off: strict | drop `CTX_KEY_PREFIXES` from `NEW_LEVER_KEY_PREFIXES` |
| T-OPT | ctx groups hold exactly the head params at `lr * mult`; off = the tip's groups | put the head in `decay` too |
| T-DVB | each new flag mis-declared is refused by `declared_vs_built` | remove one `FLAG_CHECKS` entry |
| T-CKPT | save/load round trip bit-equal incl. cfg fields; ctx-on checkpoint into an off cfg refused | drop `ctx_heads` from the saved cfg |
| T-EVAL | both evaluators refuse a nav-source mismatch; `--nav-variant real` is the tool's output bit for bit; `zero` and `donor` change `nav` only | make `real` read the donor |
| T-SNAP | the existing snapshot tests (`test_v8f_level_head.py`, `test_v8f_lead_head.py`, `test_v8f_otband.py`) admit the new flags at their defaults and nothing else | none |

Run with `CUDA_VISIBLE_DEVICES=-1`, `--basetemp` on F: two levels deep, and delete it after (CLAUDE.md delete rule).

## 10. Frozen argvs (pod paths; the checker renders them)

**C0** = `obd_OBD-CTL_s{k}/argv.txt` (M) with box-2 paths mapped to the pod, plus:
`--nav-source v9 --nav-text-npz "" --nav-text-sha256 "" --ctx-heads off --w-ctx 0 --ctx-labels "" --ctx-labels-sha256 "" --ctx-lr-mult 1 --ctx-detach off`

**S1** = C0 with:
`--nav-source text --nav-text-npz <navtext_nav.npz> --nav-text-sha256 <its sha> --ctx-heads on --w-ctx 0.5 --ctx-labels <strat_labels_table.npz> --ctx-labels-sha256 8055a321df6db1725ea17abde2999d48bcb68d43f1a480a69ed42dd0cf184eff --ctx-lr-mult 10 --ctx-detach off`

**N1** = C0 with only the nav-source group of S1 (ctx off).

Doses (fixed here, not tuned):
* `w_ctx 0.5`. At init, `T["ctx"]` is the weighted mean CE = (ln 25 + ln 4 + 0.5 ln 3 + ln 4 + ln 5) / 4.5 = 1.81 (arithmetic). Half of that is 4.4 % of the OBD-CTL step-0 total 20.44 (M, `obd_OBD-CTL_s0/selfcheck.json` `acc_probe.total_w`); E, it falls as the CE falls.
* `w_ctx` sets the head's pull on the shared adapter. Adam normalises the head's own update, so the head's learning speed is set by `--ctx-lr-mult`.
* `ctx-lr-mult 10` (head lr 1e-3): the head starts from random, inside a 3,000-step fine-tune at lr 1e-4 whose other groups are pre-trained.

Seeds (the DAG-OACC convention, M from `obd_*/argv.txt`): `s0` = `M2b_s0` + bank `s0` + `--dagger-seed 0` + `--seed 0`; `s1` = `M2b_s1` + bank `s1` + 1 + 1; the replicate `rep` = init 0, bank `s0`, `--dagger-seed 0`, `--seed 2` (OBD-NULL_rep's form).

## 11. Venue (the LoRA pod; nothing here touches it)

What runs there, per arm, one job at a time: checker preflight, `--selfcheck-only` gate, the pilot (first arm of each kind), the 3,000-step run, post-check, bank (pull to the dev box by sha256). Reads run on the dev box CPU.

Preconditions the checker asserts before the first arm, each UNVERIFIED today:
* `/root/data/cache_pod` holds the 4,572-clip train cache in `ClipStore` format, with the same `read_manifest` order and `train_sha256` `37f4f9ee...` (M on box 2, `summary.json`);
* the pod holds byte-equal copies of every md5/sha in section 8, shipped with sha256;
* the GPU is idle (`nvidia-smi` name check: RTX 4000 Ada);
* `memory.max` read, as for the other pod (50 GB cap there);
* `F:` stays >= 25 GB on the dev box for pulls (E: <= 1.5 GB per banked arm, OBD's stated bound).
