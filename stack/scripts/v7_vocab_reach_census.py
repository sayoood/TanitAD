#!/usr/bin/env python3
"""A COMPLETE per-token reach census for the FROZEN v7 vocabulary.

⭐⭐ WHY THIS EXISTS. ``D-TLIGHT-1`` -- a label minted on 779 of 4,572 clips,
frozen into the vocabulary, and reached by NO loss -- was found by accident.
``stack/tests/test_tactical_label_reach.py`` pinned it afterwards, but it is
scoped to **four traffic-light tokens and four supervised heads**. Every other
token could be in the identical state and nothing in the repo would notice.

⛔⛔ THE RULE THAT GOVERNS THIS FILE -- **A CROSS-CHECK MUST BE DERIVED
INDEPENDENTLY OF THE VALUE IT CHECKS** (CLAUDE.md; four measured instances in
one night, one of them a census whose test pinned a set *against the status
that set produced* -- permanently blind to an extra member).

⇒ **Nothing here asks the vocabulary module whether a token is trained.** Two
derivations, then a comparison:

  EMISSION  read from the v7.2 label BLOBS. Every string in a token-bearing
            field is counted, vocabulary member or not -- so an OFF-VOCABULARY
            emission (the ``REDUCE_TO_FOLLOW_ROUTE`` class) is visible.

  REACH     three INDEPENDENT LEGS per surface, all of which must hold:
            (1) PROJECTION -- run the real consumer function and require it to
                turn the token into a valid target/input cell;
            (2) CONSUMER   -- assert, by literal marker strings, that a TRAINER
                actually calls that projection and its loss. Each file read
                carries a SAME-BREATH CONTROL marker that must be found, so a
                "marker absent" verdict can never be a failed read;
            (3) GRADIENT   -- where a built module is named, require the
                measured gradient census (``--gradreach``) to show the module
                actually receiving gradient under the arm's own argv.
                ``p.grad is None`` is the discriminator ``tac_goal_head.py``'s
                own docstring names for *"this head is not wired"*.

The vocabulary's declarations (``ROLE_OF``, ``NOT_YET_EXTRACTABLE``,
``TACTICAL_GOAL_NEEDS_PERCEPTION``, ``TACTICAL_GOAL_UNDERPOWERED``, the two
floors) are read ONLY at the end and ONLY to be COMPARED. Every disagreement is
a finding; none can change a class.

USAGE
-----
    python stack/scripts/v7_vocab_reach_census.py \
        --train <s2_labels_v7.2_train.jsonl.gz> \
        --eval  <s2_labels_v7.2_eval.jsonl.gz> \
        --repo-root <repo> \
        --gradreach <gradreach_live.json> \
        --live-config <refcv5_v2_launch_config.json> \
        --out census.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# THE CLASSES -- literals, never an expression over the code under test.
# --------------------------------------------------------------------------- #
CLASS_TRAINING = "TRAINING_SIGNAL"
CLASS_INPUT = "INFERENCE_INPUT"
CLASS_AUDIT = "AUDIT_OR_METRIC_ONLY"
CLASS_UNREACHED = "UNREACHED"
CLASS_NOT_EMITTED = "NOT_EMITTED"
ALL_CLASSES = (CLASS_TRAINING, CLASS_INPUT, CLASS_AUDIT, CLASS_UNREACHED,
               CLASS_NOT_EMITTED)


# =========================================================================== #
# DERIVATION 1 -- EMISSION, from the BLOBS ONLY
# =========================================================================== #
#: Token-bearing JSON paths, named from the RECORD SCHEMA (``s2-geom-v7``), not
#: from the vocabulary. ⛔ A path is where a STRING lives; whether that string
#: is a legal token is a separate question this function never asks.
TOKEN_FIELDS: tuple[tuple[str, str], ...] = (
    ("a_str.token", "strategic action"),
    ("g_str.token", "strategic goal"),
    ("a_tac.lat", "tactical lat action"),
    ("a_tac.lon", "tactical lon action"),
    ("g_tac.goals.<KEY>", "tactical goal"),
    ("nav_command.token", "nav command"),
)


def _read_records(path: str | Path) -> list[dict]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def emission_census(paths: dict[str, str]) -> dict[str, Any]:
    """``{split: path}`` -> per-token counts per FIELD, straight from the blob."""
    by_split: dict[str, dict] = {}
    for split, path in paths.items():
        recs = _read_records(path)
        counts: dict[str, Counter] = {f: Counter() for f, _ in TOKEN_FIELDS}
        n_speed_block = 0
        for r in recs:
            for f, _layer in TOKEN_FIELDS:
                if f == "g_tac.goals.<KEY>":
                    for k in ((r.get("g_tac") or {}).get("goals") or {}):
                        counts[f][str(k)] += 1
                    continue
                head, tail = f.split(".", 1)
                v = (r.get(head) or {}).get(tail)
                if isinstance(v, str):
                    counts[f][v] += 1
            if r.get("speed_max_input"):
                n_speed_block += 1
        by_split[split] = {
            "path": str(path), "n_records": len(recs),
            #: ⛔ E16's channel. The v7.2 release carries the block on ZERO
            #: records, and ``refc_v3_train.py`` REFUSES ``--max-speed-input``
            #: when not one window receives a value -- so the flag is not
            #: merely "not passed", it is UNRUNNABLE on this corpus.
            "n_records_with_speed_max_input": n_speed_block,
            "counts": {f: dict(c) for f, c in counts.items()},
        }
    return by_split


# =========================================================================== #
# DERIVATION 2 -- REACH
# =========================================================================== #
#: One consumer SURFACE = one executable projection from a label field into a
#: tensor the model sees, plus the TRAINER that calls it.
#:
#: ``markers``  literal strings that must be present in ``trainer`` for a real
#:              consumer to exist. ⛔ Written as LITERALS, never as an
#:              expression over the code under test.
#: ``control``  a same-breath control string that MUST be found in the same
#:              file. If it is absent the file was not really read and the
#:              verdict is INCONCLUSIVE, never "marker missing".
#: ``grad_module``  the built module whose gradient the ``--gradreach`` census
#:              must show reaching, when that evidence is supplied.
SURFACES: dict[str, dict[str, Any]] = {
    "tac_lat_ce": {
        "kind": "loss_target",
        "layer": "tactical lat action",
        "projection": "v7_labels.tactical_class_ids(label, t)[0]",
        "trainer": "stack/scripts/refc_v3_train.py",
        "markers": ('F.cross_entropy(out["lat_logits_tac"], lat_t)',
                    'batch["lat_v7"]',
                    'len(v7l.HEADS["tac_lat"])'),
        "control": "def compute_losses_v3(",
        "grad_module": "lat_head_tac",
        "enabled_by": ("--v7-labels",),
        "note": "refc line. Head width refc_v3.py:982 from "
                "v6.tactical_lat_actions(v7.0).",
    },
    "tac_lon_ce": {
        "kind": "loss_target",
        "layer": "tactical lon action",
        "projection": "v7_labels.tactical_class_ids(label, t)[1]",
        "trainer": "stack/scripts/refc_v3_train.py",
        "markers": ('F.cross_entropy(out["lon_logits_tac"], lon_t)',
                    'batch["lon_v7"]'),
        "control": "def compute_losses_v3(",
        "grad_module": "lon_head_tac",
        "enabled_by": ("--v7-labels",),
        "note": "refc line. Head width refc_v3.py:983.",
    },
    "tac_goal_bce": {
        "kind": "loss_target",
        "layer": "tactical goal",
        "projection": "v7_labels.tactical_goal_targets(label, t)",
        "trainer": "stack/scripts/refc_v3_train.py",
        #: ⛔ THE TWO CALLS A TRAINER WOULD HAVE TO MAKE. The head
        #: (`TacGoalTokenHead`) and the loss (`tac_goal_loss`) and the emitter
        #: (`TacGoalEmitter`) all EXIST; what decides reach is whether the
        #: TRAINER calls them.
        "markers": ("tac_goal_loss", "TacGoalEmitter"),
        "control": "def compute_losses_v3(",
        "grad_module": "tac_goal_tok_head",
        "enabled_by": ("--v7-labels", "--tac-goal-tok-head"),
        "note": "D-TACGOAL-1. refc_v3.py:1320 computes tac_goal_logits; "
                "nothing consumes them.",
    },
    "tac_action_ce_v6": {
        "kind": "loss_target",
        "layer": "tactical lat/lon action",
        "projection": "V72WindowSupervision -> batch['tac_lat_id'/'tac_lon_id']",
        "trainer": "stack/scripts/train_v6_staged.py",
        #: the trainer's OWN comment at :2411 says *"No loss term reads them
        #: yet"*; this marker is what that sentence would look like in code.
        "markers": ('cross_entropy(out["a_lat"]',),
        "control": "V72_TACTICAL_BATCH_KEYS",
        "grad_module": None,
        "enabled_by": ("--s2-labels",),
        "note": "v6 line. The tensors are landed; the CE is a pre-registered "
                "follow-up (train_v6_staged.py:2410-2415).",
    },
    "str_goal_ce_v6": {
        "kind": "loss_target",
        "layer": "strategic goal",
        "projection": "v7_labels.HEADS['str_goal'].index(token)",
        "trainer": "stack/scripts/train_v6_staged.py",
        "markers": ('s2_goal_loss(out["g_str"], out["a_str"], batch,',
                    "g_tokens=tuple(stack.vocab_str.tokens)",
                    'HEADS["str_goal"]'),
        "control": "def s2_goal_loss(",
        "grad_module": None,
        "enabled_by": ("--s2-labels", "w_s2_goal>0"),
        "note": "v6 line ONLY. Gated on w_s2_goal, DEFAULT 0.0 -- and the term "
                "is behind `if w.w_s2_goal:`, so at the default the head's "
                "p.grad is None (the guarded-term trap).",
    },
    "str_action_ce_v6": {
        "kind": "loss_target",
        "layer": "strategic action",
        "projection": "v7_labels.HEADS['str_action'].index(token)",
        "trainer": "stack/scripts/train_v6_staged.py",
        "markers": ('s2_goal_loss(out["g_str"], out["a_str"], batch,',
                    "a_tokens=tuple(stack.vocab_a_str.tokens)",
                    'HEADS["str_action"]'),
        "control": "def s2_goal_loss(",
        "grad_module": None,
        "enabled_by": ("--s2-labels", "w_s2_goal>0"),
        "note": "v6 line ONLY, same gate as str_goal_ce_v6.",
    },
    "str_ce_refc": {
        "kind": "loss_target",
        "layer": "strategic goal/action",
        "projection": "(none -- refc has no strategic TOKEN head)",
        "trainer": "stack/scripts/refc_v3_train.py",
        #: refc's `str_goal_head` is `nn.Linear(d_ctx, 3)` -- a GEOMETRIC
        #: bearing/distance readout supervised from LAN, not a softmax over
        #: the 8 strategic goal tokens. This marker is what a token head
        #: would look like; its absence is the finding.
        "markers": ('HEADS["str_goal"]',),
        "control": "def compute_losses_v3(",
        "grad_module": None,
        "enabled_by": ("--v7-labels",),
        "note": "refc line. `--goal-str` supervises g_str against a LAN "
                "bearing/distance target (strategic_goal_loss), NOT against "
                "the v7 strategic token.",
    },
    "nav_input": {
        "kind": "model_input",
        "layer": "nav command",
        "projection": "v7_labels.NavEmitter(...)(ep_idx)",
        "trainer": "stack/scripts/refc_v3_train.py",
        "markers": ("refb.NAV_COMMANDS.index(NAV_TOKEN_TO_LEGACY[tok])",
                    "assert_nav_token_alignment"),
        "control": "def compute_losses_v3(",
        "grad_module": "nav_to_tac",
        "enabled_by": ("--v7-labels", "--nav-from-v7"),
        "note": "⭐ THE NAV COMMAND IS AN INPUT, NOT A TRAINING SIGNAL (PI). "
                "Its provenance on this corpus is 'ego-future' on 100 % of "
                "records, so it is an ORACLE input, gated by "
                "load_v7_labels(allow_oracle_nav=True) and STAMPED.",
    },
    "goal_audit": {
        "kind": "audit",
        "layer": "tactical goal",
        "projection": "v7_labels._goal_audit(record['g_tac'])",
        #: ⛔ THE CONSUMER IS THE READER, NOT THE STORE. `v7_labels.py` only
        #: PUTS the goal set into `V7Label.audit['goal_flags']`; storing is not
        #: consuming. The one real reader is the census script below -- MEASURED
        #: by a repo-wide scan for `goal_flags` (5 hits / 4 files: the store,
        #: this reader, and two tests), with a `def ` control at 17,726 hits so
        #: the zero elsewhere is a claim about the CONTENT, not the search.
        #: ⚠️ `stack/tanitad/eval/constraints.py` names `g_tac.goals.SPEED_BAND`
        #: in its module DOCSTRING only -- to explain why it may NOT be fed --
        #: and `four_families.TARGET_SPEED_BANDS_MPS` is an unrelated metric
        #: tolerance constant. Neither reads the label.
        "trainer": "stack/scripts/tactical_label_census.py",
        "markers": ("goal_flags", "_flat_tactical_fields"),
        "control": "def _surface_map(",
        "grad_module": None,
        "enabled_by": ("(audit tool, no training flag)",),
        "note": "V7Label.audit['goal_flags'] -- the loader's own contract line "
                "says 'audit-only, NEVER a training input'.",
    },
}


def assert_consumers(repo_root: Path) -> dict[str, Any]:
    """Leg 2 -- does a TRAINER actually call the projection and its loss?

    ⛔ A ZERO IS A CLAIM ABOUT THE SEARCH UNTIL THE READ IS ASSERTED. Every
    file read carries a control marker that must be found; if it is not, the
    surface is INCONCLUSIVE and never "absent".
    """
    out: dict[str, Any] = {}
    cache: dict[str, str | None] = {}
    for sid, spec in SURFACES.items():
        rel = spec["trainer"]
        if rel not in cache:
            p = repo_root / rel
            try:
                cache[rel] = p.read_text(encoding="utf-8", errors="strict")
            except Exception as exc:                       # noqa: BLE001
                cache[rel] = None
                print(f"[census] ⛔ UNREADABLE {p}: {exc}", file=sys.stderr)
        src = cache[rel]
        if src is None:
            out[sid] = {"status": "INCONCLUSIVE",
                        "why": f"{rel} could not be read"}
            continue
        ctl = spec["control"]
        if ctl not in src:
            out[sid] = {"status": "INCONCLUSIVE",
                        "why": f"same-breath control {ctl!r} not found in "
                               f"{rel} -- the read is not trustworthy, so a "
                               f"missing marker proves nothing"}
            continue
        found = {m: (m in src) for m in spec["markers"]}
        out[sid] = {
            "status": "CONSUMED" if all(found.values()) else "NO_CONSUMER",
            "file": rel,
            "control_found": True,
            "markers": found,
            "missing": sorted(m for m, ok in found.items() if not ok),
        }
    return out


class ReachProbe:
    """Leg 1 -- EXECUTE the real projection against a real loaded label."""

    def __init__(self, labels, manifest):
        import torch  # noqa: PLC0415
        self.torch = torch
        self.labels = labels
        self.manifest = manifest
        self.template = labels[0]

    def _replace(self, **kw):
        import dataclasses  # noqa: PLC0415
        return dataclasses.replace(self.template, **kw)

    @staticmethod
    def _row_nonzero(lin, row: int) -> bool:
        g = lin.weight.grad
        return bool(g is not None and g[row].abs().sum().item() > 0.0)

    # -- surfaces --------------------------------------------------------- #
    def tac_action_ce(self, token: str, axis: str) -> dict[str, Any]:
        import torch, torch.nn as nn, torch.nn.functional as F  # noqa: PLC0415,E401
        from tanitad.data import v7_labels as v7l  # noqa: PLC0415
        head = "tac_lat" if axis == "lat" else "tac_lon"
        lab = self._replace(**{f"tac_{axis}": token})
        try:
            ids = v7l.tactical_class_ids(lab, lab.t0_s)
        except ValueError:
            return {"projects": False,
                    "why": "the consumer's own .index() refuses the token"}
        cid = ids[0] if axis == "lat" else ids[1]
        if cid == v7l.IGNORE_ID:
            return {"projects": False, "why": "projected to IGNORE_ID"}
        width = len(v7l.HEADS[head])
        torch.manual_seed(0)
        lin = nn.Linear(8, width)
        loss = F.cross_entropy(lin(torch.ones(1, 8)), torch.tensor([cid]))
        loss.backward()
        return {"projects": self._row_nonzero(lin, cid), "class_id": int(cid),
                "head_width": int(width)}

    def tac_goal_bce(self, token: str) -> dict[str, Any]:
        import torch  # noqa: PLC0415
        from tanitad.data import v7_labels as v7l  # noqa: PLC0415
        from tanitad.refs.tac_goal_head import (  # noqa: PLC0415
            TacGoalTokenHead, tac_goal_loss)
        toks = tuple(v7l.TAC_GOAL_TOKENS)
        if token not in toks:
            return {"projects": False,
                    "why": "not in TacGoalTokenHead's own token tuple"}
        i = toks.index(token)
        lab = self._replace(tac_goals=frozenset({token}),
                            tac_goal_meta={token: {"provenance": "geometry"}})
        y, w = v7l.tactical_goal_targets(lab, lab.t0_s, negatives="measured")
        torch.manual_seed(0)
        head = TacGoalTokenHead(8)
        loss, n_sup = tac_goal_loss(head(torch.ones(1, 8)),
                                    torch.tensor([list(y)]),
                                    torch.tensor([list(w)]))
        loss.backward()
        lin = head.net if hasattr(head.net, "weight") else head.net[-1]
        return {"projects": self._row_nonzero(lin, i) and float(y[i]) == 1.0,
                "class_id": int(i), "head_width": int(head.n_tokens),
                "y_i": float(y[i]), "w_i": float(w[i]),
                "n_supervised_cells": int(n_sup)}

    def tac_action_ce_v6(self, token: str) -> dict[str, Any]:
        """v6's projection is the SAME ``tactical_class_ids``; the difference
        is entirely in leg 2 (no trainer loss reads the landed tensors)."""
        r = self.tac_action_ce(token, "lat")
        if not r.get("projects"):
            r = self.tac_action_ce(token, "lon")
        return r

    def strategic_ce(self, token: str, head_name: str) -> dict[str, Any]:
        import torch, torch.nn as nn, torch.nn.functional as F  # noqa: PLC0415,E401
        from tanitad.data import v7_labels as v7l  # noqa: PLC0415
        toks = v7l.HEADS[head_name]
        if token not in toks:
            return {"projects": False,
                    "why": f"{head_name}: OffVocabularyToken (.index raises)"}
        cid = toks.index(token)
        torch.manual_seed(0)
        lin = nn.Linear(8, len(toks))
        F.cross_entropy(lin(torch.ones(1, 8)),
                        torch.tensor([cid])).backward()
        return {"projects": self._row_nonzero(lin, cid), "class_id": int(cid),
                "head_width": int(len(toks))}

    def nav_input(self, token: str) -> dict[str, Any]:
        """⭐ THE CONTROL IS THE POINT: a constant emitter would pass a "did it
        produce a tensor" check and fails "another token gives another tensor".

        ⚠️ ``NavConditioner`` ZERO-INITS its per-layer output projection by
        design (loss-continuity), so ``forward()`` is 0 for EVERY token at
        init and an output-difference test would read a design choice as an
        absence. The admissible tests are the SHARED code path
        (``encode``, not zero-inited) and the gradient into the embedding row
        once the projection is off zero -- i.e. after step 1 of any run.
        """
        import dataclasses, torch  # noqa: PLC0415,E401
        from tanitad.data import v7_labels as v7l  # noqa: PLC0415
        from tanitad.models.nav_conditioning import NavConditioner  # noqa: PLC0415,E501
        oracle = dict(self.template._oracle)
        base = dict(oracle.get("nav_command") or {})
        if not base:
            return {"projects": False, "why": "template has no nav_command"}

        def emit(tok: str):
            nav = dict(base, token=tok)
            lab = dataclasses.replace(
                self.template, _oracle={**oracle, "nav_command": nav})
            em = v7l.NavEmitter([lab], self.manifest, {0: lab.clip_id})
            return em(torch.tensor([0]))

        try:
            ids, args = emit(token)
        except v7l.NavTokenMissing as exc:
            return {"projects": False, "why": f"NavEmitter refused: {exc}"}
        other = next(t for t in v7l.NAV_TOKENS if t != token)
        ids2, _ = emit(other)
        changed = bool((ids != ids2).any().item())
        torch.manual_seed(0)
        cond = NavConditioner(d_model=16)
        e1, e2 = cond.encode(ids, args), cond.encode(ids2, args)
        encode_differs = bool((e1 - e2).abs().sum().item() > 0)
        # open the zero-init gate the way step 1 of training does
        with torch.no_grad():
            for ln in cond.layers:
                cond.layer_proj[ln].weight.normal_(0.0, 0.1)
        cond.zero_grad()
        cond(ids, args, layer="tactical").sum().backward()
        row = int(ids[0].item())
        g = cond.embed.weight.grad
        grad_row = bool(g is not None and g[row].abs().sum().item() > 0.0)
        return {"projects": changed and encode_differs and grad_row,
                "token_id": row, "changes_vs_control": changed,
                "encode_differs": encode_differs,
                "grad_reaches_embedding_row": grad_row,
                "control_token": other,
                "zero_init_note": "NavConditioner.layer_proj is zero-init by "
                                  "design; forward() is 0 for EVERY token at "
                                  "init, so the output-difference test is "
                                  "run with the projection opened."}

    def goal_audit(self, token: str) -> dict[str, Any]:
        from tanitad.data import v7_labels as v7l  # noqa: PLC0415
        out = v7l._goal_audit(
            {"goals": {token: {"disputed": True, "provenance": "vlm-cot"}}})
        return {"projects": token in out}

    def none(self, token: str) -> dict[str, Any]:
        return {"projects": False, "why": "no projection exists"}


def _probe(probe: ReachProbe, sid: str, token: str) -> dict:
    return {
        "tac_lat_ce": lambda: probe.tac_action_ce(token, "lat"),
        "tac_lon_ce": lambda: probe.tac_action_ce(token, "lon"),
        "tac_goal_bce": lambda: probe.tac_goal_bce(token),
        "tac_action_ce_v6": lambda: probe.tac_action_ce_v6(token),
        "str_goal_ce_v6": lambda: probe.strategic_ce(token, "str_goal"),
        "str_action_ce_v6": lambda: probe.strategic_ce(token, "str_action"),
        "str_ce_refc": lambda: probe.none(token),
        "nav_input": lambda: probe.nav_input(token),
        "goal_audit": lambda: probe.goal_audit(token),
    }[sid]()


#: Which surfaces a token is a CANDIDATE for, by the blob FIELD it appears in.
FIELD_TO_SURFACES: dict[str, tuple[str, ...]] = {
    "a_str.token": ("str_action_ce_v6", "str_ce_refc"),
    "g_str.token": ("str_goal_ce_v6", "str_ce_refc"),
    "a_tac.lat": ("tac_lat_ce", "tac_action_ce_v6"),
    "a_tac.lon": ("tac_lon_ce", "tac_action_ce_v6"),
    "g_tac.goals.<KEY>": ("tac_goal_bce", "goal_audit"),
    "nav_command.token": ("nav_input",),
}
#: ...and by the vocabulary tuple, so a token the blob NEVER emits is still
#: probed -- that is how a DEAD LOGIT (a live class with no data) is found.
TUPLE_TO_SURFACES: dict[str, tuple[str, ...]] = {
    "STRATEGIC_GOAL_TOKENS_V7": ("str_goal_ce_v6", "str_ce_refc"),
    "STRATEGIC_ACTION_TOKENS_V7": ("str_action_ce_v6", "str_ce_refc"),
    "TACTICAL_GOAL_TOKENS_V7": ("tac_goal_bce", "goal_audit"),
    "TACTICAL_LAT_ACTIONS_V7": ("tac_lat_ce", "tac_action_ce_v6"),
    "TACTICAL_LON_ACTIONS_V7": ("tac_lon_ce", "tac_action_ce_v6"),
    "NAV_COMMAND_TOKENS": ("nav_input",),
}


def _surface_reached(sid: str, proj: dict, consumer: dict,
                     grad: dict | None) -> tuple[bool, str]:
    """All three legs, ANDed. Returns ``(reached, why_not)``."""
    if not proj.get("projects"):
        return False, f"leg1 PROJECTION: {proj.get('why', 'no valid target')}"
    st = consumer.get("status")
    if st == "INCONCLUSIVE":
        return False, f"leg2 CONSUMER: INCONCLUSIVE -- {consumer.get('why')}"
    if st != "CONSUMED":
        return False, (f"leg2 CONSUMER: no trainer calls it "
                       f"(missing {consumer.get('missing')})")
    gm = SURFACES[sid].get("grad_module")
    if gm and grad is not None:
        row = (grad.get("modules") or {}).get(gm)
        if row is None:
            return False, (f"leg3 GRADIENT: module {gm!r} absent from the "
                           f"gradient census")
        if row["verdict"] != "GRADIENT REACHES":
            return False, (f"leg3 GRADIENT: {gm} -> {row['verdict']} "
                           f"(grad_none={row['n_grad_none']}/"
                           f"{row['n_tensors']}, |g|={row['grad_abs_sum']})")
    return True, ""


def _class_of(kinds: set[str], n_total: int) -> str:
    if n_total == 0:
        return CLASS_NOT_EMITTED
    if "loss_target" in kinds:
        return CLASS_TRAINING
    if "model_input" in kinds:
        return CLASS_INPUT
    if "audit" in kinds:
        return CLASS_AUDIT
    return CLASS_UNREACHED


def build_census(labels, manifest, emission, vocab_tuples, consumers,
                 grad) -> dict[str, Any]:
    probe = ReachProbe(labels, manifest)
    emitted: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for split, blk in emission.items():
        for field, counts in blk["counts"].items():
            for tok, n in counts.items():
                emitted[tok].setdefault(field, {})[split] = n
    declared: dict[str, list[str]] = defaultdict(list)
    for name, toks in vocab_tuples.items():
        for t in toks:
            declared[t].append(name)

    rows: dict[str, Any] = {}
    for tok in sorted(set(emitted) | set(declared)):
        fields = emitted.get(tok, {})
        n_by_split = {s: 0 for s in emission}
        for per_split in fields.values():
            for s, n in per_split.items():
                n_by_split[s] = n_by_split.get(s, 0) + n
        n_total = sum(n_by_split.values())

        cand: set[str] = set()
        for f in fields:
            cand |= set(FIELD_TO_SURFACES.get(f, ()))
        for name in declared.get(tok, ()):
            cand |= set(TUPLE_TO_SURFACES.get(name, ()))

        surf: dict[str, Any] = {}
        kinds: set[str] = set()
        for sid in sorted(cand):
            proj = _probe(probe, sid, tok)
            reached, why = _surface_reached(sid, proj, consumers[sid], grad)
            surf[sid] = {"reached": reached, "why_not": why,
                         "kind": SURFACES[sid]["kind"],
                         "enabled_by": list(SURFACES[sid]["enabled_by"]),
                         "trainer": SURFACES[sid]["trainer"],
                         "leg1_projection": proj,
                         "leg2_consumer": consumers[sid].get("status"),
                         "leg3_grad_module": SURFACES[sid].get("grad_module")}
            if reached:
                kinds.add(SURFACES[sid]["kind"])

        cls = _class_of(kinds, n_total)
        rows[tok] = {
            "token": tok, "class": cls,
            "in_vocab_tuples": sorted(declared.get(tok, ())),
            "off_vocabulary": not declared.get(tok),
            "emitted_in_fields": sorted(fields),
            "n_by_split": n_by_split, "n_total": n_total,
            "reached_kinds": sorted(kinds),
            "reached_surfaces": sorted(s for s, r in surf.items()
                                       if r["reached"]),
            "blocked_surfaces": {s: r["why_not"] for s, r in surf.items()
                                 if not r["reached"] and r["why_not"]},
            #: ⛔ THE WORSE FINDING: a class with a live logit and no data.
            "dead_logit_in": sorted(s for s, r in surf.items()
                                    if r["reached"] and n_total == 0),
            #: ⛔⛔ THE D-TLIGHT-1 CONDITION, STATED PER TOKEN: emitted into
            #: the corpus and reached by NO loss. An `AUDIT_OR_METRIC_ONLY`
            #: token is in this condition just as much as an `UNREACHED` one --
            #: an audit reader is not a gradient.
            #: ⚠️ ``model_input`` is EXCLUDED on purpose. The nav command is
            #: an INPUT by PI ruling and reaching no loss is CORRECT for it;
            #: counting it here would manufacture three false findings and
            #: bury the eighteen real ones.
            "d_tlight_condition": bool(
                n_total > 0 and "loss_target" not in kinds
                and "model_input" not in kinds),
            "training_reach": ("loss" if "loss_target" in kinds else "NONE"),
            "surfaces": surf,
        }
    return rows


# =========================================================================== #
# DECLARATION vs MEASUREMENT -- read the declarations HERE and nowhere else
# =========================================================================== #
def disagreements(rows: dict[str, Any], V7) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(tok, kind, declared, measured):
        out.append({"token": tok, "kind": kind, "declared": declared,
                    "measured": measured})

    for tok, row in sorted(rows.items()):
        cls, role = row["class"], V7.ROLE_OF.get(tok)
        if role == "input" and cls not in (CLASS_INPUT, CLASS_NOT_EMITTED):
            add(tok, "ROLE_OF says INPUT but the measured class differs",
                role, cls)
        if role in ("goal", "action") and cls == CLASS_INPUT:
            add(tok, "⛔ declared a LABEL but measured as an INPUT -- "
                     "candidate LEAK", role, cls)
        nye = tok in V7.NOT_YET_EXTRACTABLE
        if nye and row["n_total"] > 0:
            add(tok, "declared NOT_YET_EXTRACTABLE but the blob EMITS it",
                "NOT_YET_EXTRACTABLE", f"n={row['n_total']}")
        if (not nye) and row["n_total"] == 0 and not row["off_vocabulary"]:
            add(tok, "NOT declared NOT_YET_EXTRACTABLE and the blob emits it "
                     "ZERO times", "extractable", "n=0")
        if row["off_vocabulary"]:
            add(tok, "⛔ EMITTED but NOT in the frozen vocabulary",
                "(absent from ALL_V7_TOKENS)",
                f"n={row['n_total']} in {row['emitted_in_fields']}")
        if row["dead_logit_in"]:
            add(tok, "⛔ DEAD LOGIT -- a head carries the class and the corpus "
                     "never emits it", "-", row["dead_logit_in"])
    # the perception declaration vs the blob's own provenance
    return out


def provenance_divergence(labels, V7) -> dict[str, Any]:
    """``TACTICAL_GOAL_NEEDS_PERCEPTION`` (declared) vs the blob's provenance."""
    cot: set[str] = set()
    geo: set[str] = set()
    for lb in labels:
        for t, m in (lb.tac_goal_meta or {}).items():
            p = (m or {}).get("provenance")
            if p == "vlm-cot":
                cot.add(t)
            elif p == "geometry":
                geo.add(t)
    declared = set(V7.TACTICAL_GOAL_NEEDS_PERCEPTION)
    return {"declared_needs_perception": sorted(declared),
            "cot_backed_in_blob": sorted(cot),
            "geometry_backed_in_blob": sorted(geo),
            "cot_backed_but_NOT_declared": sorted(cot - declared),
            "declared_but_geometry_only_in_blob": sorted(declared - cot)}


def underpowered_report(rows, V7) -> dict[str, Any]:
    """n against the two floors. ⛔ Reported WITH n, never as a rate."""
    tr, me = int(V7.GOAL_MIN_N_FOR_TRAINING), int(V7.GOAL_MIN_N_FOR_METRIC)
    toks = {}
    for tok, row in rows.items():
        if "TACTICAL_GOAL_TOKENS_V7" not in row["in_vocab_tuples"]:
            continue
        n = row["n_by_split"].get("train", 0)
        toks[tok] = {"n_train": n, "n_total": row["n_total"],
                     "verdict": ("ABSENT" if n == 0 else
                                 "UNDERPOWERED_FOR_TRAINING" if n < tr else
                                 "UNSCOREABLE_AS_A_RATE" if n < me else "OK")}
    derived_under = sorted(t for t, v in toks.items()
                           if v["verdict"] in ("UNDERPOWERED_FOR_TRAINING",
                                               "UNSCOREABLE_AS_A_RATE"))
    declared_under = sorted(V7.TACTICAL_GOAL_UNDERPOWERED)
    return {"GOAL_MIN_N_FOR_TRAINING": tr, "GOAL_MIN_N_FOR_METRIC": me,
            "tokens": toks,
            "derived_below_metric_floor": derived_under,
            "declared_TACTICAL_GOAL_UNDERPOWERED": declared_under,
            "declared_but_not_derived":
                sorted(set(declared_under) - set(derived_under)),
            "derived_but_not_declared":
                sorted(set(derived_under) - set(declared_under))}


# =========================================================================== #
# THE LIVE ARM -- from ITS OWN argv, never from what the trainer COULD do
# =========================================================================== #
#: surface -> flags that must ALL be in the arm's argv. ⛔ Literals.
LIVE_FLAG_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "tac_lat_ce": ("--v7-labels",),
    "tac_lon_ce": ("--v7-labels",),
    "tac_goal_bce": ("--v7-labels", "--tac-goal-tok-head"),
    "nav_input": ("--v7-labels", "--nav-from-v7"),
    "goal_audit": ("--v7-labels",),
    "str_ce_refc": ("--v7-labels",),
    #: ⛔ a DIFFERENT TRAINER. No refc argv can ever enable these, and saying
    #: so explicitly is the point of this table.
    "str_goal_ce_v6": ("--IMPOSSIBLE-this-is-train_v6_staged",),
    "str_action_ce_v6": ("--IMPOSSIBLE-this-is-train_v6_staged",),
    "tac_action_ce_v6": ("--IMPOSSIBLE-this-is-train_v6_staged",),
}


def live_arm_table(rows, argv: list[str], trainer: str) -> dict[str, Any]:
    present = {a for a in argv if a.startswith("--")}
    enabled = {s: all(f in present for f in req)
               for s, req in LIVE_FLAG_REQUIREMENTS.items()}
    per_token = {}
    for tok, row in sorted(rows.items()):
        live = sorted(s for s, r in row["surfaces"].items()
                      if r["reached"] and enabled.get(s, False))
        per_token[tok] = {"consumed_by_live_arm": bool(live),
                          "live_surfaces": live, "n_total": row["n_total"],
                          "repo_class": row["class"]}
    return {"trainer": trainer, "argv": argv, "flags_present": sorted(present),
            "surfaces_enabled_by_argv": enabled, "per_token": per_token,
            "n_tokens_not_consumed": sum(
                1 for v in per_token.values()
                if not v["consumed_by_live_arm"])}


# =========================================================================== #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--gradreach", default=None)
    ap.add_argument("--live-config", default=None)
    ap.add_argument("--live-trainer", default="stack/scripts/refc_v3_train.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    from tanitad.data import v7_labels as v7l
    from tanitad.models import vocab_v7 as V7

    paths = {"train": a.train}
    if a.eval:
        paths["eval"] = a.eval
    print(f"[census] leg 0 -- EMISSION from {list(paths)} ...", flush=True)
    emission = emission_census(paths)
    for s, blk in emission.items():
        print(f"[census]   {s}: {blk['n_records']} records; "
              f"speed_max_input on {blk['n_records_with_speed_max_input']}")

    print("[census] loading through the REAL consumer (load_v7_labels) ...",
          flush=True)
    labels, manifest = v7l.load_v7_labels(a.train, allow_oracle_nav=True)
    print(f"[census]   md5={manifest.md5} n={manifest.n_records} "
          f"schema={manifest.schema_version} vocab={manifest.vocab}")

    print("[census] leg 2 -- CONSUMER assertions (with same-breath controls)",
          flush=True)
    consumers = assert_consumers(Path(a.repo_root))
    for sid, r in sorted(consumers.items()):
        print(f"[census]   {sid:20s} {r['status']}"
              + (f"  missing={r.get('missing')}"
                 if r.get("missing") else ""))

    grad = None
    if a.gradreach:
        grad = json.loads(Path(a.gradreach).read_text(encoding="utf-8"))
        print(f"[census] leg 3 -- GRADIENT census from {a.gradreach}")

    vocab_tuples = {
        "STRATEGIC_GOAL_TOKENS_V7": tuple(V7.STRATEGIC_GOAL_TOKENS_V7),
        "STRATEGIC_ACTION_TOKENS_V7": tuple(V7.STRATEGIC_ACTION_TOKENS_V7),
        "TACTICAL_GOAL_TOKENS_V7": tuple(V7.TACTICAL_GOAL_TOKENS_V7),
        "TACTICAL_LAT_ACTIONS_V7": tuple(V7.TACTICAL_LAT_ACTIONS_V7),
        "TACTICAL_LON_ACTIONS_V7": tuple(V7.TACTICAL_LON_ACTIONS_V7),
        "NAV_COMMAND_TOKENS": tuple(V7.NAV_COMMAND_TOKENS),
    }
    print("[census] leg 1 -- PROJECTION probes ...", flush=True)
    rows = build_census(labels, manifest, emission, vocab_tuples, consumers,
                        grad)

    by_class = Counter(r["class"] for r in rows.values())
    out: dict[str, Any] = {
        "_schema": "v7-vocab-reach-census/2",
        "_evidence": "MEASURED. Emission read from the blobs; reach decided by "
                     "three independent legs -- executed projection, asserted "
                     "trainer consumer (with same-breath read controls), and "
                     "the measured per-module gradient census.",
        "_rule": "A CROSS-CHECK MUST BE DERIVED INDEPENDENTLY OF THE VALUE IT "
                 "CHECKS. No class below is decided by a vocabulary "
                 "declaration; declarations are compared afterwards and every "
                 "disagreement is reported as a finding.",
        "classes": list(ALL_CLASSES),
        "emission": emission,
        "label_manifest": manifest.to_dict(),
        "surfaces": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                         for kk, vv in v.items()}
                     for k, v in SURFACES.items()},
        "consumer_assertions": consumers,
        "gradient_census": grad,
        "tokens": rows,
        "class_counts": {c: by_class.get(c, 0) for c in ALL_CLASSES},
        "class_members": {c: sorted(t for t, r in rows.items()
                                    if r["class"] == c) for c in ALL_CLASSES},
        #: ⛔⛔ THE HEADLINE LIST. Every member is a finding: a token the
        #: corpus carries and no loss can see.
        "d_tlight_condition_tokens": sorted(
            t for t, r in rows.items() if r["d_tlight_condition"]),
        "disagreements": disagreements(rows, V7),
        "provenance_divergence": provenance_divergence(labels, V7),
        "underpowered": underpowered_report(rows, V7),
    }
    if a.live_config:
        cfg = json.loads(Path(a.live_config).read_text(encoding="utf-8"))
        out["live_arm"] = live_arm_table(rows, list(cfg.get("argv") or []),
                                         a.live_trainer)

    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[census] wrote {a.out}")
    print(f"[census] class counts: {out['class_counts']}")
    print(f"[census] disagreements: {len(out['disagreements'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
