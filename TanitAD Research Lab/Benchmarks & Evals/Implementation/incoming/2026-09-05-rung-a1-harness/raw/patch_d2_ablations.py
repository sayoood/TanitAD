"""RUNG A1 / DELIVERABLE 2 — the missing eval-time ablation CLI flags.

``Project Steering/PREREG_REFCV4B_HIERARCHY_EVAL.md`` §3 registers 12 ablation
arms. Three had a flag (``--with-navzero``, ``--no-navshuf``,
``--with-navflip``) and FULL needs none; the other EIGHT had no way to be run
from the command line. The prereg's own §7 escalation concedes seven of them —
and OMITS the eighth, the frame-blind deliberate regression, which is the arm
the panel's validity rests on (a gate that has never been shown to FAIL an
image-blind arm certifies nothing).

Idempotent, exact-match, read-back verified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# (1) the registry — the authoritative list is the PREREG's §3, not its §7      #
# --------------------------------------------------------------------------- #
REGISTRY = ('''#: ⛔ ARMS THAT DO NOT EXIST FOR THIS MODEL. Written into the record with the''', '''#: ⭐⭐ THE EVAL-TIME ABLATIONS — ``PREREG_REFCV4B_HIERARCHY_EVAL.md`` §3.
#:
#: ⛔ WHY THIS DICT IS THE REGISTRY AND THE PREREG'S §7 IS NOT. §7's escalation
#: names SEVEN switches ({gstr_zero, gstr_shuffle, e7_off, e9_off, h19_off,
#: ego_zero, sel_refined}) and OMITS the eighth — the frame-blind DELIBERATE
#: REGRESSION, §3's last row, on which the whole panel's validity depends: a
#: gate that has never been shown to FAIL an image-blind arm certifies nothing
#: (H-ECHO-4, where an ADE-scored gate once passed an echoing arm). An
#: incomplete list of what is missing is the same failure class as an
#: incomplete list of what is present. ⇒ §3 is authoritative; this dict mirrors
#: §3 and ``stack/tests/test_refcv3_ablations.py`` pins the bijection.
#:
#: Every ablation is EVAL-TIME and leaves the rest of the forward bit-identical.
#: ⚠️ An eval-time knockout measures the trained model's RELIANCE on an edge —
#: a lower bound on what the edge bought in training and an upper bound on
#: nothing — so each row states the regime it creates and whether training ever
#: produced it (`seen_in_training`), exactly as the prereg's §3 column does.
ABLATIONS = {
    "gstr_zero": {
        "prereg_arm": "g_str-ZERO",
        "mechanism": ("forward hook on `str_goal_head` returning (1, 0, 0), so "
                      "the model's own normalisation yields the straight-ahead "
                      "constant g_str = (1, 0, 0) before E4/E7"),
        "seen_in_training": "at init only (zero-init FiLM)",
        "tests": "the strategic goal conditions the tactical/operative levels (E4)",
        "applies_at": "model",
        "requires": "hier build (str_goal_head exists only on the goal cascade)",
    },
    "gstr_shuffle": {
        "prereg_arm": "g_str-SHUFFLE",
        "mechanism": ("the same hook, fed the g_str another WINDOW produced — "
                      "read from a banked FULL dump (--gstr-bank) and permuted "
                      "with --gstr-shuffle-seed. ⛔ NOT a batch permutation: "
                      "this harness's batch rows are the nav CONDITIONINGS of "
                      "ONE window, so permuting them would permute nav, not "
                      "windows, and would silently answer a different question"),
        "seen_in_training": "no",
        "tests": "the goal carries WINDOW-specific information downstream",
        "applies_at": "model",
        "requires": "hier build AND --gstr-bank <FULL dump dir> on the same grid",
    },
    "e7_off": {
        "prereg_arm": "E7-OFF",
        "mechanism": ("the hierarchy hook returns `target_latent=None`, so "
                      "refc.py:2208 leaves it None and the decoder skips the "
                      "FiLM entirely (refc.py:1508 `tgt_film is not None and "
                      "target_latent is not None`)"),
        "seen_in_training": "at init only",
        "tests": "the tactical latent conditions the decoder",
        "applies_at": "model",
        "requires": "hier build",
    },
    "e9_off": {
        "prereg_arm": "E9-OFF",
        "mechanism": ("`model.goal_gate` set to 0.0, so refc_v3.py:1037's "
                      "graft = goal_gate * score is exactly zero and the "
                      "blended rank collapses to sel_score"),
        "seen_in_training": "at init (the gate is zero-init)",
        "tests": "goal-distance selection improves the pick",
        "applies_at": "model",
        "requires": "hier build",
    },
    "h19_off": {
        "prereg_arm": "H19-OFF",
        "mechanism": ("`decoder.maneuver_to_anchor = None`, removing the "
                      "kin3-derived 5-way prior from the anchor confidence "
                      "(refc.py:1572 gates on it)"),
        "seen_in_training": "no",
        "tests": "the model's own tactical prediction improves its selection",
        "applies_at": "model",
        "requires": "a decoder built with graft_maneuver=True",
    },
    "ego_zero": {
        "prereg_arm": "EGO-ZERO",
        "mechanism": ("ego_state[:, 4] = 0 (keep = 0, the X15 regime — the "
                      "model itself then zeroes the values beside the flag) "
                      "AND v0 withheld at the core (v0=None, which is how "
                      "refc.py derives keep=0 there). ⚠️ The model-free "
                      "controls still integrate the MEASURED v0: they are "
                      "controls, not arms, and must read bit-identically "
                      "across ablations"),
        "seen_in_training": "yes (ego_dropout 0.5)",
        "tests": "the ego channels are used (E11') — a robustness read, NOT a "
                 "thesis test",
        "applies_at": "call-site",
        "requires": "an ego_state_inject build",
    },
    "sel_refined": {
        "prereg_arm": "SEL-REFINED",
        "mechanism": ("`decoder.sel.refined = True` and "
                      "`decoder.sel.score_emitted = True` (0 params) — rank the "
                      "refined fan by the refined confidence read from the "
                      "EMITTED estimate"),
        "seen_in_training": "no",
        "tests": "the selection surface is the lever the implementation audit named",
        "applies_at": "model",
        "requires": ("diffusion steps > 0 — ⛔ at steps == 0 refc.py:1596 "
                     "leaves `refined is conf` BY CONSTRUCTION and :1630 gates "
                     "score_emitted on `steps > 0`, so the switch would parse "
                     "and change nothing. REFUSED there rather than silently "
                     "inert: a flag that does nothing is worse than a missing "
                     "one"),
    },
    "frames_blind": {
        "prereg_arm": "DELIBERATE REGRESSION",
        "mechanism": ("every observed frame replaced by the window's own "
                      "scalar mean, so the encoder sees a constant image and "
                      "the arm is an echo BY CONSTRUCTION"),
        "seen_in_training": "no",
        "tests": ("⭐⭐ THAT THE INSTRUMENTS CAN FAIL. If this arm PASSES the "
                  "echo gate, or reads FOLLOWS_NAV with a high compliance, the "
                  "PANEL IS VOID (PREREG_REFC_V4 §7 OUTCOME IV). The "
                  "model-free controls ha / ha0 / ha0_ext read no frames and "
                  "must come back BIT-IDENTICAL to the FULL run — that is the "
                  "internal control on this arm"),
        "applies_at": "call-site",
        "requires": "nothing",
    },
}
#: The four §3 rows that ALREADY had a route before Rung A1, so the completeness
#: test can assert a bijection with §3's twelve rather than only with the eight.
ABLATIONS_PREEXISTING = {
    "FULL": "no flag — the final checkpoint as trained IS the default run",
    "nav-ZERO": "arm `os_navzero`, on by default; --no-navzero removes it",
    "nav-SHUFFLE": "arm `os_navshuf`, on by default; --no-navshuf removes it",
    "nav-FLIP": "--with-navflip",
}
#: ⛔ ARMS THAT DO NOT EXIST FOR THIS MODEL. Written into the record with the''')

# --------------------------------------------------------------------------- #
# (2) the machinery                                                            #
# --------------------------------------------------------------------------- #
MACHINERY = ('''# --------------------------------------------------------------------------- #
# grid + kinematics                                                            #
# --------------------------------------------------------------------------- #
def grid_slots(horizons, grid: str) -> dict:''', '''# --------------------------------------------------------------------------- #
# the eval-time ablations (PREREG_REFCV4B_HIERARCHY_EVAL.md §3)                  #
# --------------------------------------------------------------------------- #
class AblationState:
    """What the roll loop needs to know, and the per-window channel the g_str
    hook reads. Held on ONE object so the hook and the loop cannot drift."""

    def __init__(self):
        self.names: list[str] = []
        self.frames_blind = False
        self.ego_zero = False
        self.gstr_mode: str | None = None     # None | "zero" | "shuffle"
        self.gstr_target = None               # [3] for the CURRENT window
        self.gstr_bank: dict = {}             # (clip_id, ws) -> [3]
        self.gstr_perm: dict = {}             # (clip_id, ws) -> (clip_id, ws)
        self.n_gstr_windows = 0
        self.verified = {}                    # ablation -> what was observed


def resolve_ablations(a) -> list[str]:
    """``--ablate`` + the explicitly-named ``--ablate-frames``, deduplicated.

    ``--ablate-frames`` exists as its OWN flag because the frame-blind
    deliberate regression is the arm the panel's validity rests on and it must
    be nameable without remembering an enum member."""
    names = list(getattr(a, "ablate", None) or [])
    if getattr(a, "ablate_frames", False) and "frames_blind" not in names:
        names.append("frames_blind")
    unknown = [n for n in names if n not in ABLATIONS]
    if unknown:
        raise SystemExit(f"[refcv3_arm] unknown ablation(s) {unknown}; known: "
                         f"{sorted(ABLATIONS)}")
    return names


def load_gstr_bank(bank_dir: str, seed: int):
    """``(clip_id, ws) -> g_str [3]`` from a banked FULL dump, plus the seeded
    permutation OVER WINDOWS.

    ⛔ The permutation is over windows because that is the registered mechanism.
    A batch-level permutation in THIS harness would permute the nav
    conditionings of one window — a different intervention wearing the same
    name, which is the `--with-navzero` defect this tool already carries a
    warning about."""
    man_p = os.path.join(bank_dir, "manifest.json")
    if not os.path.exists(man_p):
        raise SystemExit(f"[refcv3_arm] --gstr-bank {bank_dir} has no "
                         f"manifest.json — it is not a refcv3 dump")
    with open(man_p, encoding="utf-8") as fh:
        man = json.load(fh)
    by_fi = {int(e["file_index"]): e["clip_id"] for e in man.get("episodes", [])}
    bank: dict = {}
    for f in sorted(glob.glob(os.path.join(bank_dir, "ep*.npz"))):
        fi = int(os.path.basename(f)[2:5])
        dp = os.path.join(bank_dir, "decisions", f"ep{fi:03d}.npz")
        if not os.path.exists(dp):
            raise SystemExit(f"[refcv3_arm] --gstr-bank: {dp} is missing; the "
                             f"bank needs the decisions sidecar (gstr_nav_true)")
        with np.load(f) as d, np.load(dp) as dd:
            if "gstr_nav_true" not in dd.files:
                raise SystemExit(
                    f"[refcv3_arm] --gstr-bank: {dp} carries no "
                    f"`gstr_nav_true` — the bank was rolled on a FLAT build, "
                    f"which has no strategic goal to shuffle")
            ws = np.asarray(d["ws"]).astype(np.int64).reshape(-1)
            gs = np.asarray(dd["gstr_nav_true"]).astype(np.float32)
            cid = by_fi.get(fi)
            if cid is None:
                raise SystemExit(f"[refcv3_arm] --gstr-bank: no clip_id for "
                                 f"file_index {fi} in the bank manifest")
            if gs.shape[0] != ws.shape[0]:
                raise SystemExit(f"[refcv3_arm] --gstr-bank: {fi} has "
                                 f"{gs.shape[0]} goals for {ws.shape[0]} windows")
            for j, w in enumerate(ws.tolist()):
                bank[(str(cid), int(w))] = gs[j].reshape(-1)[:3]
    if not bank:
        raise SystemExit(f"[refcv3_arm] --gstr-bank {bank_dir} yielded no windows")
    keys = sorted(bank)
    perm = np.random.default_rng(int(seed)).permutation(len(keys))
    mapping = {keys[i]: keys[int(perm[i])] for i in range(len(keys))}
    n_same = sum(1 for k, v in mapping.items() if k == v)
    stats = {"n_windows": len(keys), "seed": int(seed),
             "n_fixed_points": int(n_same),
             "frac_changed": round(1.0 - n_same / max(1, len(keys)), 6),
             "bank_dir": bank_dir,
             "⚠️": ("a fixed point feeds a window ITS OWN goal; those windows "
                    "are not ablated and the changed fraction is the honest n")}
    return bank, mapping, stats


def _gstr_forward_hook(state: "AblationState"):
    """Replaces ``str_goal_head``'s output so the model's OWN normalisation
    (bearing = g[:2]/||g[:2]||, dist = tanh(g[2])) reproduces the intended
    ``g_str`` EXACTLY — the intervention is on the strategic goal, never on the
    normalisation."""
    import torch

    def hook(_mod, _inp, out):
        if state.gstr_mode is None:
            return None
        g = out.new_zeros(out.shape[0], 3)
        if state.gstr_mode == "zero":
            g[:, 0] = 1.0                    # bearing (1, 0); tanh(0) = 0
            return g
        tgt = state.gstr_target
        if tgt is None:
            raise RuntimeError("gstr_shuffle: no banked goal set for this "
                               "window — the loop and the hook are out of step")
        bx, by, dist = float(tgt[0]), float(tgt[1]), float(tgt[2])
        n = math.hypot(bx, by)
        if n < 1e-9:
            raise RuntimeError(f"gstr_shuffle: banked bearing has norm {n}")
        dist = max(-1.0 + 1e-6, min(1.0 - 1e-6, dist))
        g[:, 0], g[:, 1] = bx / n, by / n
        g[:, 2] = math.atanh(dist)
        return g

    return hook


def apply_ablations(model, cfg, names, *, steps: int, feed_ego: bool,
                    gstr_bank: str | None = None, gstr_seed: int = 0):
    """Apply the eval-time ablations and return ``(state, record)``.

    ⛔ EVERY precondition is checked and REFUSED loudly. A flag that parses and
    changes nothing is worse than a missing one: it produces a table that reads
    like a knockout and is a copy of the FULL arm."""
    state = AblationState()
    state.names = list(names)
    per: dict = {}
    if not names:
        return state, {"applied": [], "requested": [],
                       "prereg": "PREREG_REFCV4B_HIERARCHY_EVAL.md §3",
                       "is": "FULL — the final checkpoint as trained"}
    hier = bool(getattr(cfg, "hier", False))
    dec = getattr(getattr(model, "core", None), "decoder", None)
    for n in names:
        spec = ABLATIONS[n]
        ev: dict = {"prereg_arm": spec["prereg_arm"],
                    "mechanism": spec["mechanism"],
                    "seen_in_training": spec["seen_in_training"]}
        if n in ("gstr_zero", "gstr_shuffle", "e7_off", "e9_off") and not hier:
            raise SystemExit(f"[refcv3_arm] ablation {n!r} needs a HIER build; "
                             f"this checkpoint is flat and the edge it ablates "
                             f"does not exist. Refusing rather than reporting "
                             f"an unchanged arm as a knockout.")
        if n == "gstr_zero":
            state.gstr_mode = "zero"
            model.str_goal_head.register_forward_hook(_gstr_forward_hook(state))
            ev["injected_g_str"] = [1.0, 0.0, 0.0]
        elif n == "gstr_shuffle":
            if not gstr_bank:
                raise SystemExit(
                    "[refcv3_arm] ablation 'gstr_shuffle' needs --gstr-bank "
                    "<a FULL dump dir on the SAME grid>: the registered "
                    "mechanism permutes g_str ACROSS WINDOWS, and this "
                    "harness's batch rows are the nav conditionings of ONE "
                    "window. Permuting them would permute nav, not windows.")
            state.gstr_mode = "shuffle"
            state.gstr_bank, state.gstr_perm, ev["bank"] = \\
                load_gstr_bank(gstr_bank, gstr_seed)
            model.str_goal_head.register_forward_hook(_gstr_forward_hook(state))
        elif n == "e7_off":
            _orig = model._hook

            def _hook_no_e7(cache, nav_cmd=None, ego_state=None, _o=_orig):
                inner = _o(cache, nav_cmd, ego_state)

                def wrapped(pooled_seq, ctx):
                    hk = dict(inner(pooled_seq, ctx))
                    hk["target_latent"] = None      # refc.py:2208 keeps it None
                    return hk
                return wrapped

            model._hook = _hook_no_e7
            ev["effect"] = ("the decoder's tgt_film is skipped "
                            "(refc.py:1508 gates on target_latent is not None)")
        elif n == "e9_off":
            before = float(model.goal_gate.detach().reshape(()).item())
            model.goal_gate.data.fill_(0.0)
            ev["goal_gate_before"] = before
            ev["goal_gate_after"] = 0.0
            if before == 0.0:
                ev["INERT"] = (
                    "the trained goal_gate was ALREADY exactly 0.0, so E9 "
                    "contributed nothing to this checkpoint and knocking it "
                    "out cannot change the forward. That is a FINDING about "
                    "the checkpoint (Caveat-B: the zero-init gate never "
                    "opened), NOT a defect of the switch — and it means the "
                    "E9-OFF row of the panel is uninformative and must be "
                    "reported as such, never as 'no effect, therefore the "
                    "seam is inert at eval'.")
                _p(f"  ⛔ ablation e9_off is INERT: {ev['INERT']}")
        elif n == "h19_off":
            if dec is None or getattr(dec, "maneuver_to_anchor", None) is None:
                raise SystemExit(
                    "[refcv3_arm] ablation 'h19_off' needs a decoder built "
                    "with graft_maneuver=True; `maneuver_to_anchor` is None "
                    "here, so the prior it removes is already absent.")
            ev["removed"] = type(dec.maneuver_to_anchor).__name__
            dec.maneuver_to_anchor = None
        elif n == "sel_refined":
            sel = getattr(dec, "sel", None)
            if sel is None:
                raise SystemExit("[refcv3_arm] ablation 'sel_refined': the "
                                 "decoder carries no selection config")
            if int(steps) <= 0:
                raise SystemExit(
                    "[refcv3_arm] ablation 'sel_refined' needs diffusion "
                    "steps > 0. At steps == 0 refc.py:1596 leaves `refined is "
                    "conf` BY CONSTRUCTION and :1630 gates score_emitted on "
                    "`steps > 0`, so the switch would parse and change "
                    "NOTHING. Refusing: a flag that does nothing is worse "
                    "than a missing one.")
            ev["before"] = {"refined": bool(sel.refined),
                            "score_emitted": bool(sel.score_emitted)}
            sel.refined = True
            sel.score_emitted = True
            ev["after"] = {"refined": True, "score_emitted": True}
            ev["params_changed"] = 0
        elif n == "ego_zero":
            if not feed_ego:
                raise SystemExit(
                    "[refcv3_arm] ablation 'ego_zero' needs an "
                    "ego_state_inject (v4) build: this build is fed no ego "
                    "block, so zeroing its keep bit would change nothing.")
            state.ego_zero = True
            ev["effect"] = ("ego_state[:, 4] = 0 on every forward AND v0=None "
                            "at the core (which is how refc.py derives keep=0 "
                            "there); the model-free controls keep the MEASURED "
                            "v0 and must stay bit-identical across ablations")
        elif n == "frames_blind":
            state.frames_blind = True
            ev["effect"] = ("frames := their own scalar mean; ha / ha0 / "
                            "ha0_ext read no frames and MUST come back "
                            "bit-identical to the FULL run — that is the "
                            "internal control on the deliberate regression")
        per[n] = ev
    rec = {"applied": list(names), "requested": list(names),
           "prereg": "PREREG_REFCV4B_HIERARCHY_EVAL.md §3",
           "per_ablation": per,
           "⛔ read_with": ("EVERY number in this record was produced under the "
                           "ablation(s) above. An eval-time knockout measures "
                           "the trained model's RELIANCE on an edge — a lower "
                           "bound on what the edge bought in training and an "
                           "upper bound on nothing."),
           "tier_unchanged": ("the ablations do not change what is fed from the "
                              "future; every arm keeps its ARM_TIERS stamp")}
    _p(f"[ablation] {names} — {'; '.join(ABLATIONS[n]['prereg_arm'] for n in names)}")
    return state, rec


# --------------------------------------------------------------------------- #
# grid + kinematics                                                            #
# --------------------------------------------------------------------------- #
def grid_slots(horizons, grid: str) -> dict:''')

# --------------------------------------------------------------------------- #
# (3) the call sites                                                           #
# --------------------------------------------------------------------------- #
EDITS: list[tuple[str, str, str]] = [
    REGISTRY,
    MACHINERY,
    # ---- `math` is needed by the g_str hook's inverse normalisation ---------
    (
        "import math",
        '''import json
import os''',
        '''import json
import math
import os''',
    ),
    # ---- apply, right after the model is built ------------------------------
    (
        "apply in run_dump",
        '''    model, cfg, targs, prov = load_model(a.ckpt, a.config, dev, a.allow_nonstrict)''',
        '''    model, cfg, targs, prov = load_model(a.ckpt, a.config, dev, a.allow_nonstrict)
    # ⭐⭐ THE EVAL-TIME ABLATIONS (PREREG_REFCV4B_HIERARCHY_EVAL.md §3), applied
    # to the LOADED model before a single window is rolled, so the whole dump is
    # one regime and the stamp below describes all of it.
    _abl_names = resolve_ablations(a)
    abl_state, abl_rec = apply_ablations(
        model, cfg, _abl_names, steps=int(prov["decoder_steps"]),
        feed_ego=bool(getattr(cfg, "ego_state_inject", False)),
        gstr_bank=getattr(a, "gstr_bank", None),
        gstr_seed=int(getattr(a, "gstr_shuffle_seed", 0) or 0))''',
    ),
    # ---- the dump-dir guard -------------------------------------------------
    (
        "dump dir guard",
        '''    os.makedirs(a.dump_dir, exist_ok=True)
    os.makedirs(os.path.join(a.dump_dir, "decisions"), exist_ok=True)''',
        '''    # ⛔ A DUMP DIRECTORY IS ONE REGIME. Writing an ablated roll on top of a
    # FULL one leaves ep*.npz from both and a manifest describing only the
    # second — an unreadable mixture that looks like a complete dump. Refuse.
    _prev_man = os.path.join(a.dump_dir, "manifest.json")
    if os.path.exists(_prev_man):
        try:
            with open(_prev_man, encoding="utf-8") as _fh:
                _prev = ((json.load(_fh).get("ablation") or {}).get("applied"))
        except (OSError, ValueError):
            _prev = "__unreadable__"
        if _prev != "__unreadable__" and list(_prev or []) != list(_abl_names):
            raise SystemExit(
                f"[refcv3_arm] ⛔ {a.dump_dir} already holds a dump rolled "
                f"under ablation {_prev!r}, and this run is {_abl_names!r}. "
                f"Two regimes in one directory is an unreadable mixture that "
                f"looks like a complete dump. Use a separate --dump-dir per "
                f"ablation arm.")
    os.makedirs(a.dump_dir, exist_ok=True)
    os.makedirs(os.path.join(a.dump_dir, "decisions"), exist_ok=True)
    with open(os.path.join(a.dump_dir, "ABLATION.txt"), "w",
              encoding="utf-8") as _fh:
        _fh.write((", ".join(_abl_names) if _abl_names else "FULL") + "\\n")''',
    ),
    # ---- frames_blind -------------------------------------------------------
    (
        "frames blind",
        '''            fr = tr.frames_to_device(item["frames"][None], dev)   # [1, W, C, H, W]''',
        '''            fr = tr.frames_to_device(item["frames"][None], dev)   # [1, W, C, H, W]
            if abl_state.frames_blind:
                # ⭐⭐ THE DELIBERATE REGRESSION: a constant image. The arm is an
                # echo BY CONSTRUCTION, and if the echo gate PASSES it the panel
                # is VOID (PREREG_REFC_V4 §7 OUTCOME IV).
                fr = torch.full_like(fr, float(fr.mean()))''',
    ),
    # ---- gstr target + ego_zero --------------------------------------------
    (
        "per-window ablation channel",
        '''            ego_b = ego_z = None
            if feed_ego:
                _es = v3mod.ego_state_from_batch(
                    {"pose_last": item["pose_last"].float()[None],
                     "actions": item["actions"].float()[None]}, device=dev)
                ego_b = _es.expand(b, -1).contiguous() if b > 1 else _es
                ego_z = _es''',
        '''            # ⭐ the g_str SHUFFLE's per-window channel: the banked goal of the
            # window this one is permuted onto. Set BEFORE the forward, read by
            # the hook on str_goal_head, so the loop and the hook cannot drift.
            if abl_state.gstr_mode == "shuffle":
                _key = (str(clip_ids[e_i]), int(t0))
                _src = abl_state.gstr_perm.get(_key)
                if _src is None:
                    raise SystemExit(
                        f"[refcv3_arm] gstr_shuffle: window {_key} is not in "
                        f"the bank. The bank must be a FULL dump over the SAME "
                        f"episodes AND the same window grid (--window-stride "
                        f"and --grid), or the permutation is meaningless.")
                abl_state.gstr_target = abl_state.gstr_bank[_src]
                abl_state.n_gstr_windows += 1
            ego_b = ego_z = None
            if feed_ego:
                _es = v3mod.ego_state_from_batch(
                    {"pose_last": item["pose_last"].float()[None],
                     "actions": item["actions"].float()[None]}, device=dev)
                if abl_state.ego_zero:
                    # X15: keep = 0 means WITHHELD, and the model re-applies the
                    # flag to the values, so zeroing the bit is the whole
                    # intervention on this side.
                    _es = _es.clone()
                    _es[:, 4] = 0.0
                ego_b = _es.expand(b, -1).contiguous() if b > 1 else _es
                ego_z = _es
            # ⛔ v0 WITHHELD AT THE CORE is the other half of EGO-ZERO: refc.py
            # derives `keep` from `v0 is not None`, so a pre-zeroed v0 would
            # arrive with keep = 1 — the file says so itself. Pass None.
            _v0_fed = None if abl_state.ego_zero else v0_t''',
    ),
    (
        "forward v0",
        '''                out = model(fr_b, nav_cmd=nav_t if nav_on else None,
                            v0=v0_t, steps=steps, ego_state=ego_b)''',
        '''                out = model(fr_b, nav_cmd=nav_t if nav_on else None,
                            v0=_v0_fed, steps=steps, ego_state=ego_b)''',
    ),
    (
        "forward v0 navzero",
        '''                    out_z = model(fr, nav_cmd=None,
                                  v0=v0_t[:1], steps=steps, ego_state=ego_z)''',
        '''                    out_z = model(fr, nav_cmd=None,
                                  v0=None if abl_state.ego_zero else v0_t[:1],
                                  steps=steps, ego_state=ego_z)''',
    ),
    # ---- the injection self-check, once, on the first forward ---------------
    (
        "injection self-check",
        '''            # ⭐ THE DEPLOYED SELECTION IS out["traj"] AND NOTHING ELSE''',
        '''            # ⭐ DID THE ABLATION ACTUALLY REACH THE FORWARD? Checked against
            # the model's OWN emitted g_str, once, rather than assumed. A hook
            # registered on the wrong module raises nothing and reports nothing.
            if abl_state.gstr_mode is not None and not abl_state.verified:
                _got = out.get("g_str")
                if _got is None:
                    raise SystemExit("[refcv3_arm] the g_str ablation is armed "
                                     "but the forward emits no `g_str` — the "
                                     "hook cannot be verified, so the arm is "
                                     "not admissible")
                _got = _got.float()[0].detach().cpu().numpy().tolist()
                _want = ([1.0, 0.0, 0.0] if abl_state.gstr_mode == "zero"
                         else [float(x) for x in abl_state.gstr_target[:3]])
                _err = max(abs(_got[j] - _want[j]) for j in range(3))
                if _err > 1e-4:
                    raise SystemExit(
                        f"[refcv3_arm] the g_str ablation did NOT take: the "
                        f"model emitted {_got} where {_want} was injected "
                        f"(max abs err {_err:.3g}). Refusing to roll an arm "
                        f"whose intervention cannot be demonstrated.")
                abl_state.verified["g_str"] = {"emitted": _got,
                                               "injected": _want,
                                               "max_abs_err": float(_err)}
                _p(f"  [ablation] g_str injection VERIFIED on the first window: "
                   f"emitted {[round(x, 4) for x in _got]} (err {_err:.2g})")
            # ⭐ THE DEPLOYED SELECTION IS out["traj"] AND NOTHING ELSE''',
    ),
    # ---- the manifest stamp -------------------------------------------------
    (
        "manifest stamp",
        '''        "ego_state_fed": feed_ego,''',
        '''        "ego_state_fed": feed_ego,
        # ⭐⭐ THE ABLATION PROVENANCE. A result must never be readable without
        # knowing which ablation produced it, so this rides in the manifest AND
        # is copied onto every per-arm block by analyze_refcv3.
        "ablation": dict(abl_rec, verified=abl_state.verified,
                         n_gstr_windows=abl_state.n_gstr_windows),''',
    ),
    # ---- the analysis stamp -------------------------------------------------
    (
        "analysis stamp",
        '''    ref["manifest"] = manifest
    rec["refcv3"] = ref
    return rec''',
        '''    ref["manifest"] = manifest
    # ⭐⭐ THE ABLATION STAMP TRAVELS TO EVERY ARM BLOCK. A per-arm number that
    # can be read without its ablation is a number that WILL be read without it.
    _abl = (manifest or {}).get("ablation") or {
        "applied": None,
        "⛔ provenance": ("UNKNOWN — this dump's manifest predates the ablation "
                         "stamp (Rung A1, 2026-09-05). That is NOT evidence "
                         "that no ablation was applied; absence of the field is "
                         "absence of the record."),
    }
    ref["ablation"] = _abl
    for _blk in (rec.get("arms") or {}).values():
        _blk["ablation"] = _abl
    rec["refcv3"] = ref
    return rec''',
    ),
    # ---- the CLI ------------------------------------------------------------
    (
        "cli",
        '''    ap.add_argument("--allow-nonstrict", action="store_true")''',
        '''    # ---- ⭐⭐ the eval-time ablations (PREREG_REFCV4B_HIERARCHY_EVAL.md §3) --
    ap.add_argument("--ablate", nargs="*", default=[], metavar="NAME",
                    choices=sorted(ABLATIONS),
                    help="eval-time ablation(s) to apply to the loaded model "
                         "before the roll; the rest of the forward stays "
                         "bit-identical. One regime per --dump-dir. Choices: "
                         + ", ".join(f"{k} ({v['prereg_arm']})"
                                     for k, v in sorted(ABLATIONS.items())))
    ap.add_argument("--ablate-frames", action="store_true",
                    help="⭐⭐ THE DELIBERATE REGRESSION (= --ablate "
                         "frames_blind), given its own flag because the panel's "
                         "VALIDITY rests on it: a gate that has never been "
                         "shown to FAIL an image-blind arm certifies nothing. "
                         "Every frame becomes its own scalar mean; ha / ha0 / "
                         "ha0_ext must come back bit-identical.")
    ap.add_argument("--gstr-bank", default=None, metavar="FULL_DUMP_DIR",
                    help="a banked FULL dump on the SAME episodes and window "
                         "grid, read for `gstr_nav_true`; REQUIRED by --ablate "
                         "gstr_shuffle, whose registered mechanism permutes "
                         "g_str ACROSS WINDOWS (this harness's batch rows are "
                         "the nav conditionings of ONE window)")
    ap.add_argument("--gstr-shuffle-seed", type=int, default=0,
                    help="seed for the g_str window permutation")
    ap.add_argument("--allow-nonstrict", action="store_true")''',
    ),
]


def _apply(path: Path, edits, label: str) -> str:
    raw = path.read_bytes()
    if b"\r" in raw:
        raise SystemExit(f"[{label}] REFUSING: {path} contains CR")
    src = path.read_text(encoding="utf-8")
    done, todo = [], []
    for item in edits:
        name, old, new = (item if len(item) == 3 else ("edit", item[0], item[1]))
        if new in src:
            done.append(name)
        elif old in src:
            todo.append((name, old, new))
        else:
            raise SystemExit(f"[{label}] REFUSING: anchor {name!r} not found "
                             f"and its replacement is not present either")
    if not todo:
        return f"[{label}] already applied ({len(done)}/{len(edits)})"
    for name, old, new in todo:
        if src.count(old) != 1:
            raise SystemExit(f"[{label}] REFUSING: anchor {name!r} occurs "
                             f"{src.count(old)} times, need exactly 1")
        src = src.replace(old, new, 1)
    path.write_text(src, encoding="utf-8", newline="")
    back = path.read_text(encoding="utf-8")
    missing = [n for n, _o, nw in
               [(i if len(i) == 3 else ("edit", i[0], i[1])) for i in edits]
               if nw not in back]
    if missing:
        raise SystemExit(f"[{label}] read-back FAILED for {missing}")
    return (f"[{label}] applied {[n for n, _, _ in todo]}, "
            f"{len(done)} already present; read-back OK, {len(back)} chars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    named = [("registry", *REGISTRY), ("machinery", *MACHINERY)] + EDITS[2:]
    print(_apply(Path(a.root) / "taniteval" / "tools" / "refcv3_arm.py",
                 named, "refcv3_arm"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
