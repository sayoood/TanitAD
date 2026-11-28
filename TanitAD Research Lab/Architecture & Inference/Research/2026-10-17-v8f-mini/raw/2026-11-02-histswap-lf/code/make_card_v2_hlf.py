#!/usr/bin/env python3
"""make_card_v2_hlf.py -- writes dagland/histswap/DRAFT_RECOVERY_CARD_v2.md: the landed v1 (read, never modified) plus two new sections, "Evidence since v1" and "T1-kin validity risk".
Every number in the new sections is read from raw/HISTSWAP_LF_T1.json, raw/HISTSWAP_T1.json, logs/chain.out or the frozen card; file:line citations are checked against the files they cite (fail loud).

    python -I code/make_card_v2_hlf.py
"""
import hashlib
import json
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE).replace("\\", "/")
DAG = os.path.dirname(PK).replace("\\", "/")
HS = f"{DAG}/histswap"
TREE = "F:/TanitAD-scratch/histswap_lf/tree_dag"
sys.path.insert(0, f"{DAG}/smooth_t1/code")
import smooth_t1_read as SR  # noqa: E402

GD = SR.load_gd()
J = json.load(open(f"{PK}/raw/HISTSWAP_LF_T1.json", encoding="utf-8"))
H = json.load(open(f"{HS}/raw/HISTSWAP_T1.json", encoding="utf-8"))
V1B = open(f"{HS}/DRAFT_RECOVERY_CARD.md", "rb").read()
V1 = V1B.decode("utf-8")
CELLS = ("CTRL_s0", "CTRL_s1", "SOUP_s0", "SOUP_s1")


def die(m):
    raise SystemExit("UNVERIFIED / PROSE-FACT FAILED: " + m)


def rng(xs, f="%+.3f"):
    return (f + " to " + f) % (min(xs), max(xs))


def line_of(path, needle):
    """1-based line number of the unique line of `path` containing `needle` (fails loud unless exactly one)."""
    hits = [i + 1 for i, ln in enumerate(open(path, encoding="utf-8", errors="replace").read().splitlines()) if needle in ln]
    if len(hits) != 1:
        die("%s: %r found on %d lines" % (path, needle, len(hits)))
    return hits[0]


TOOL = f"{PK}/code/v8f_t1_eval_histswap_lf.py"
DATA = f"{TREE}/stack/tanitad/data/v8f_mini_data.py"
MODEL = f"{TREE}/stack/tanitad/models/v8f_mini.py"
for p in (TOOL, DATA, MODEL):
    if not os.path.exists(p):
        die("missing " + p)
if hashlib.sha256(open(TOOL, "rb").read()).hexdigest() != "6c66f0a527b3b08c640550dd4d065c45f43186d1af5220830f38bd933245719f":
    die("the tool differs from the sha pinned in FROZEN_SHA.txt")
LN = {
    "tool_build_item": line_of(TOOL, "def build_item("),
    "tool_ds_item": line_of(TOOL, "it = ds.item(st.ci, st.j, a_t, a_s)"),
    "tool_frozen_if": line_of(TOOL, 'if scene == "frozen":'),
    "tool_frozen_assign": line_of(TOOL, 'it["feats"], it["frame_ok"] = st.scene0'),
    "tool_collate": line_of(TOOL, "full = D.collate(items)"),
    "tool_scene_modes": line_of(TOOL, "SCENE_MODES = "),
    "tool_strict": line_of(TOOL, '"camera": a.scene == "frozen"'),
    "data_class": line_of(DATA, "class V8FMiniWindows:"),
    "data_idx": line_of(DATA, "idx = np.arange(t - self.hmax + 1, t + 1)"),
    "data_idc": line_of(DATA, "idc = np.clip(idx, 0, T - 1)"),
    "data_feats": line_of(DATA, 'it = {"feats": d["feats"][idc], "frame_ok": okf, "p10f": p10f,'),
    "data_collate": line_of(DATA, 'out["feats"] = stack("feats", torch.float16)'),
    "model_encode": line_of(MODEL, 'adapt = self.adapter(b["feats"].float()) + self.pos[None, None]'),
    "model_scene": line_of(MODEL, "def _scene(self, adapt: Tensor, ok: Tensor, last: Tensor):"),
    "model_scene_pos": line_of(MODEL, "pos = last[:, None] - (cfg.hist - 1 - torch.arange(cfg.hist, device=adapt.device))[None]"),
    "model_run_o": line_of(MODEL, "def run_o(self, b, adapt, last_o, ck_o, ck_ok, code):"),
    "model_run_o_scene": line_of(MODEL, 'sc, pad = self._scene(adapt, b["frame_ok"], i)'),
    "model_call_o": line_of(MODEL, "traj60, p30 = self.run_o(b, adapt, last_now, ck_for_o, ck_ok, code)"),
    "model_last_now": line_of(MODEL, "last_now = torch.full((B,), Fn - 1, dtype=torch.long, device=dev)"),
}
hdr = json.load(open("F:/TanitAD-scratch/histswap_lf/out/HCV_DAGOACC_s0/t1v8f_HCV_DAGOACC_s0.json", encoding="utf-8"))["t1v8f"]
if hdr["scene"] != "logged" or hdr["strict"]["camera"] is not False or not any("human" in x for x in hdr["open_channels"]):
    die("the run header does not carry scene logged / strict.camera False / the open-channel note")
walls = [int(x) for x in re.findall(r"ZZHLF-RUN-DONE \S+ wall=(\d+)s", open(f"{PK}/logs/chain.out", encoding="utf-8", errors="replace").read())]
if len(walls) != 9:
    die("chain.out does not list 9 finished rollouts")
mw = sum(walls) / len(walls)

arms = J["arms"]
SH = {a: [arms[f"{c}:{a}"]["m1_cell"]["share_closed"] for c in CELLS] for a in ("HCV", "HLVL")}
XY = {a: [arms[f"{c}:{a}"]["m1_cell"]["xy_delta"] for c in CELLS] for a in ("HCV", "HLVL")}
AD = {a: [arms[f"{c}:{a}"]["rows"]["X_minus_Y"]["ADE6"]["delta"] for c in CELLS] for a in ("HCV", "HLVL")}
M1R, INT, OUT = J["m1_reading"], J["interpretation"], J["outcome"]
HSH = [H["arms"][c]["m1_cell"]["share_closed"] for c in CELLS]
HAD = [H["arms"][c]["rows"]["X_minus_Y"]["ADE6"]["delta"] for c in CELLS]
if (M1R["HCV"]["reading"], M1R["HLVL"]["reading"], INT["reading"], H["outcome"]["outcome"]) != ("NOT CLOSES", "CLOSES", "B_LEVEL", "SUPPORTED"):
    die("the fixed prose assumes HISTSWAP SUPPORTED, HCV NOT CLOSES, HLVL CLOSES, reading B_LEVEL")
trig_h = sorted(set.union(*[set(arms[f"{c}:HLVL"]["read_xy_X_minus_Y"]["non_crawl_separated_worse"]) for c in CELLS]))
if not trig_h or not all(r.startswith("S-POS") for r in trig_h):
    die("the prose assumes HLVL's HARMS label rests on S-POS rows alone")
dP = [J["descriptive"][f"{c}:HLVL"]["P_x30_mean_m_ticks_ge1"]["X"] - J["descriptive"][f"{c}:HLVL"]["P_x30_mean_m_ticks_ge1"]["Y"] for c in CELLS]
dO = [J["descriptive"][f"{c}:HLVL"]["o_residual_x30_mean_m_ticks_ge1"]["X"] - J["descriptive"][f"{c}:HLVL"]["o_residual_x30_mean_m_ticks_ge1"]["Y"] for c in CELLS]
if not (all(x > 0 for x in dP) and all(x < 0 for x in dO)):
    die("the prose assumes HLVL's prior end point rises and O's delta moves down at 4 of 4 cells")
STAMP = "Tier T1-kin / paired episode-cluster bootstrap (B 2,000, seed 0; answers another draw of EPISODES only) / MEASURED (ours)"

# ---------------------------------------------------------------------------------------------- new text
V2_STATUS = """**v2 status (2026-11-02, written after the HISTSWAP and HISTSWAP-LF reads).** Sections 1-7 below are v1 **word for word** (v1 sha256 `%s` is unchanged on disk); v2 adds this status block, section 0 **Evidence since v1** and section 8 **T1-kin validity risk**, and changes only the title line. It is still a DRAFT: not frozen, not launched, nothing here trains, queues or touches box 2.
Gate to freezing, as v2 reads it: (a) HISTSWAP read SUPPORTED (an ORACLE; met in that sense) and its leak-free follow-up HISTSWAP-LF read (B), see section 0; (b) the PI's box-2 go, **still open at writing** (UNVERIFIED whether an answer arrived after this file was written); (c) NEW: the frame-freeze control of section 8 is frozen together with this card, and the **level source (perception or training)** is named as the card's unknown (the HISTSWAP-LF card's committed next step under (B)).
""" % hashlib.sha256(V1B).hexdigest()

S0 = f"""## 0. Evidence since v1

{STAMP} (the descriptive blocks are point means with no interval). Paths: `dagland/histswap/RESULTS.md` + `raw/HISTSWAP_T1.json`; `dagland/histswap_lf/RESULTS.md` + `raw/HISTSWAP_LF_T1.json`.

**0.1 HISTSWAP (H-HISTSWAP-1): SUPPORTED, an ORACLE.** Handing the network the LOGGED ego history at every tick (the plan still committed to the executed state) closes **{H['outcome']['mean_share_closed']:.3f}** of Y's M1 crawl-share excess over P10 (cells {rng(HSH, '%.3f')}; M1 X - Y separated-better at {H['outcome']['n_cells_separated_better']} of 4 cells) and moves ADE6 by {rng(HAD)} m (an oracle's ADE6, not performance). It shows that the executed-speed drift enters through the ego-history input. The logged history carries the human's realised speed at every tick, so it could not tell *the executed lag is the cause* from *any human-motion history fixes it* (v1 section 7, unknown 2).

**0.2 HISTSWAP-LF (H-HISTSWAP-LF-1): reading (B), the speed LEVEL is missing; the trend copy is ruled out as the recovery.** Two zero-training arms replace the ego-history input by a constant-twist history (speed and yaw rate held, acceleration 0), anchored at the executed pose:
* **HCV** (deployable; speed = the EXECUTED speed): **NOT CLOSES.** M1 X - Y is separated-WORSE at 4 of 4 cells ({rng(XY['HCV'], '%+.4f')}), share closed {rng(SH['HCV'], '%.3f')} (mean {M1R['HCV']['mean_share_closed']:.3f}), ADE6 X - Y {rng(AD['HCV'])} m. Its harm sits in the launch class.
* **HLVL** (ORACLE; speed = the LOGGED speed, level only): **CLOSES.** M1 X - Y separated-better at 4 of 4 cells ({rng(XY['HLVL'], '%+.4f')}), share closed {rng(SH['HLVL'], '%.3f')} (mean {M1R['HLVL']['mean_share_closed']:.3f}), ADE6 X - Y {rng(AD['HLVL'])} m. HISTSWAP's full logged history closes {H['outcome']['mean_share_closed']:.3f}, so the level alone is about {100 * M1R['HLVL']['mean_share_closed'] / H['outcome']['mean_share_closed']:.0f} % of it; the rest is what the logged speed profile and yaw rows add.
* **The two arms differ in exactly one number, the speed, and that number separates a harmful arm from a closing one.** The committed reading is (B): the speed level must come from perception or training, not from the history form. The outcome label is HARMS for both arms on both bases; HLVL's label rests on the {', '.join(trig_h)} rows alone (UNVERIFIED rows, STRATEGIC family PAUSED by ITEM 25), HCV's on ADE6 (+{min(AD['HCV']):.2f} m or more at every cell) and longitudinal rows.
* **Caveats that travel with it.** HLVL is an oracle and its closure is near-mechanical: the prior's 6 s end point rises ({rng(dP)} m) while O's own delta moves down ({rng(dO)} m) at 4 of 4 cells, so the prior's anchor carries the movement (descriptive, all windows). HCV removes ALL acceleration information, not only the trend, so *trend alone* is untested (HTRND not run); and an exactly constant history is off the training distribution.

**0.3 What this changes in this card (HYPOTHESIS).**
* v1 section 1's premise, *O follows its own slowed ego history*, is refined: what the network lacks is the speed **level**; the **form** of the history is not the lever (HCV). The remedy under test stays the same, perturb the history and recover to the logged path and speed, but its success now **requires that O gets the level from somewhere other than the executed-state history, i.e. from the images** (scene tokens), or learns to discount a deviated history speed. That is the card's named unknown.
* v1 section 7, unknown 2 (the leak-confounded positive reading): a leak-free arm now exists (HCV) and does not close; the level-only oracle (HLVL) does. The fine-tune's own read is still the first test of whether training can supply the level; it is now exposed to the T1-kin validity risk of section 8.
* Unchanged: the design (sections 4-6), the costs and the PI go (open). No history-form lever is pursued (the HISTSWAP-LF card's committed rule under (B)).
"""

S8 = f"""## 8. T1-kin validity risk (NEW in v2): a recovery arm can read the human's speed from the frames

**8.1 The risk.** In T1-kin the camera features are LOGGED: they advance at the human's pace even when the executed ego lags (the run header of every HISTSWAP-LF rollout carries `scene` = `{hdr['scene']}`, `strict.camera` = {hdr['strict']['camera']}, and the open-channel note *"{hdr['open_channels'][0]}"*; tool `strict` dict at `{os.path.basename(TOOL)}:{LN['tool_strict']}`). A model trained to recover the logged speed from a deviated history can pass M1 at T1-kin by **reading the human's speed from frame motion** between the frames its scene tokens carry. **Real driving does not offer that signal**: there the frames show the car's own motion, which is the executed speed the history already reports. The training signal of this card rewards exactly that shortcut (a deviated history speed paired with a target at the logged speed, with the logged frames consistent with the logged speed). So a positive M1 read of R at T1-kin may be an artefact of the rig. (HYPOTHESIS; whether a recovery-trained O would take the shortcut is UNVERIFIED, and Y's M1 excess with the logged frames available is consistent with the base model not using it, which says nothing about a recovery-trained one.) The HISTSWAP-LF read is not exposed (HCV did not close, so there is no gain for frames to explain, and HLVL is an oracle); the risk bites the **trained** arm R.

**8.2 Where the frames that feed O's scene tokens are assembled (file:line; tree `{TREE}`, the tree the nine HISTSWAP-LF rollouts ran from; v1's `run_o` :698-712 agrees with it; the E: tree is older and its lines differ).**
1. `{os.path.basename(TOOL)}:{LN['tool_build_item']}` `build_item`; **:{LN['tool_ds_item']}** `it = ds.item(st.ci, st.j, a_t, a_s)` builds the per-tick item; items are collated at :{LN['tool_collate']} (`D.collate`).
2. `stack/tanitad/data/v8f_mini_data.py` (class `V8FMiniWindows` :{LN['data_class']}): **:{LN['data_idx']}-{LN['data_idc']}** the history index `idx = arange(t - hmax + 1, t + 1)`, clipped; **:{LN['data_feats']}** `"feats": d["feats"][idc]` = the logged per-frame feature rows of the last `hmax` frames (`hmax` = 10); `collate` :{LN['data_collate']} casts them to fp16.
3. `stack/tanitad/models/v8f_mini.py`: `_encode` **:{LN['model_encode']}** adapts all `hmax` rows; `_scene` :{LN['model_scene']} (**:{LN['model_scene_pos']}** gathers the `cfg.hist` frames, 4 per v1 section 3, ending at index `last`) builds the scene tokens; `run_o` :{LN['model_run_o']} calls it at **:{LN['model_run_o_scene']}** with `i = last_o`, and `last_now = Fn - 1` (:{LN['model_last_now']}) is passed at :{LN['model_call_o']}: **O reads the `hist` frames ending at the tick frame**. T and S read the frames ending at `j - a_t` / `j - a_s` through the same `_scene`.
4. **Existing switch, not the control:** `--scene frozen` (`SCENE_MODES` :{LN['tool_scene_modes']}; branch :{LN['tool_frozen_if']}-{LN['tool_frozen_assign']}) repeats the TICK-0 `feats` for the whole rollout. It removes frame motion but also replaces the scene by a stale one, so it confounds the read with scene staleness; it is kept only as the stricter secondary bound (8.3).
UNVERIFIED: the checkpoints' `cfg.hist` (v1 section 3 says 4; the run headers do not carry it); that the per-frame feature rows were produced by a per-frame encoder with no temporal mixing (the indexing site is per frame; the producing encoder was not read here); and the new branch below is **not written**.

**8.3 Pre-registered control: frame-freeze (DRAFT; frozen with the card; zero new training).**
* **What it does.** A new `--scene` value (name provisional: `hold_now`) in the T1 tool, applied in `build_item` right after the item is built (after :{LN['tool_ds_item']}, beside the branch at :{LN['tool_frozen_if']}): every row of `it["feats"]` is replaced by the **current frame's row, repeated** (`feats[k] := feats[-1]` for all `hmax` rows), so the `hist` frames O reads (and the frames T and S read) are the tick-`j` frame repeated and **no frame motion is available**. `frame_ok` is left unchanged (token count and padding pattern are identical; only the frame content changes). `strict.camera` stays False (the camera is still logged; a separate `camera_motion: false` entry is added).
* **Arms (per seed s = 0, 1; 8 rollouts at about {mw:.0f} s each on this rig, about {8 * mw / 60:.0f} min sequential at 16 threads, MEASURED from `dagland/histswap_lf/logs/chain.out`).** `FZ-R_s` and `FZ-C0_s` (the section 4 checkpoints R_s and C0_s, `--scene hold_now`); `FZ-Y_s` (DAG-OACC_s itself: the freeze's own effect on the init); `FZ-HLVL_s` (DAG-OACC_s with `--histswap cv_logged --scene hold_now`: the **known-answer positive control**, whose level enters through the history input, not the frames). The logged-frame reads of R_s and C0_s (section 5) are the comparison.
* **Controls (computed before any row; a failure refuses the read).** KFZ1 every batch's `feats` rows equal the last row (max abs deviation exactly 0.0); KFZ2 the apply line deleted turns a test RED (mutation) and `--scene logged` / the flag absent is bit-identical to the existing tool; KFZ3 the freeze reached the network (FZ-Y plan differs from the logged-frame Y plan by > 1e-3 m at ticks >= 1); KFZ4 **FZ-HLVL keeps its closure** (M1 X - Y separated-better at both seeds and share closed >= 0.5): if the freeze itself breaks the HLVL read, the control is uninformative and is REFUSED, not interpreted.
* **Read.** G_L = M1 crawl share of R - C0 under logged frames (the section 5 read); G_F = the same under frame-freeze (FZ-R - FZ-C0), each against its own frame mode's (C0 - P10) excess; same rows, same estimator, same four families.
  * **ROBUST:** G_F separated-better at both seeds AND mean share closed >= 0.5 (the HISTSWAP form).
  * **ARTEFACT:** the logged-frame read passes its bar (v1 section 5 primary), and under freeze G_F is separated-better at fewer than 2 seeds OR mean share closed < 0.25.
  * **PARTIAL:** anything else; the gain survives at reduced size and the capability claim is limited to the surviving part.
  (Thresholds proposed; they are frozen with the card, before any FZ number.)

**8.4 The reading rule (committed).** **A recovery arm whose M1 gain disappears under frame-freeze is a T1-kin artefact, not a driving capability.** Both outcomes are committed in advance: **ROBUST** => the gain is not carried by frame motion (necessary, not sufficient: see 8.5); **ARTEFACT** => R is reported as a rig artefact, and the next lever is the same recovery arm trained with the frame-hold applied **in training** too, so the level must come from the single current frame or from kinematics (Rule Zero: the verdict ships with its next arm).

**8.5 What the control does not remove (disclosed).** (i) The current frame still depicts the human's position at tick `j`, not the executed pose, and a single frame can carry motion blur; a position or blur cue survives (`--scene frozen` removes the position cue at the price of a stale scene and is the secondary bound; its power is UNVERIFIED). (ii) Four identical frames with distinct time embeddings are off the training distribution; the FZ-Y / FZ-C0 versus logged-frame comparison measures the freeze's own effect, and R - C0 within one frame mode cancels it only to first order. (iii) Frame-freeze says nothing about re-perception (T2 is not provisioned); ROBUST at T1-kin is a necessary condition for a driving claim, not a sufficient one.

**8.6 Why this arm is not ORC or DAG-ORC (restating v1 section 1, one paragraph; all HYPOTHESIS).** ORC and DAG-ORC gave O slowed histories with the targets untouched (the logged future in the NOW frame) and read HARMS, DAG-ORC leaving the crawl rows uncleared; nothing in them separated *recovering the level from perception* from *reading the human's speed off the logged frames*, and the HISTSWAP-LF read now says the history form is not the lever and the level must come from elsewhere. This arm differs in what it teaches and in how the read is protected: the deviation law is empirical (the model's own executed-minus-logged deviations, not a hand-set speed scale), the target is re-expressed in the perturbed frame with a rejoin blend so recovery is taught explicitly, a lateral / heading offset moves the NOW anchor (UNBUILT), the init is DAG-OACC with a matched zero-perturbation control and an optional replicate floor, and, new in v2, a gain is read through the frame-freeze control so that frame motion at T1-kin cannot be mistaken for recovery. None of these differences is measured to matter, and the honest prior is that this arm can read HARMS too.
"""

# ---------------------------------------------------------------------------------------------- splice into v1
lines = V1.split("\n")
if not lines[0].startswith("# DRAFT_RECOVERY_CARD:"):
    die("v1 title line changed")
i1 = next(i for i, ln in enumerate(lines) if ln.startswith("## 1. The idea"))
head, body = lines[:i1], lines[i1:]
head[0] = head[0].replace("# DRAFT_RECOVERY_CARD:", "# DRAFT_RECOVERY_CARD_v2:", 1)
out = "\n".join(head).rstrip("\n") + "\n\n" + V2_STATUS.rstrip("\n") + "\n\n" + S0.rstrip("\n") + "\n\n" + "\n".join(body).rstrip("\n") + "\n\n" + S8.rstrip("\n") + "\n"
# v1's sections 1-7 must be carried verbatim
if "\n".join(body).rstrip("\n") not in out:
    die("v1 body not carried verbatim")
open(f"{HS}/DRAFT_RECOVERY_CARD_v2.md", "w", encoding="utf-8", newline="\n").write(out)
print("DRAFT_RECOVERY_CARD_v2.md written", out.count("\n"), "lines; v1 sha256", hashlib.sha256(V1B).hexdigest()[:16])
print("LINES", LN)
