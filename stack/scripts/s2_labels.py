"""S2 strategic-goal LABELS — the trainer-side loader + the clip join.

THE CONSUMER of `s2-strategic-v1` (S2_STRATEGIC_GAP.md §1.2, produced by
`…/2026-08-16-s2-v1-labels/`): reads the label JSONL(s) + `clip_index.json`,
validates every record against the REAL v6 vocabularies (never a pinned copy —
the trainer imports `tanitad.models.v6` anyway, so drift is impossible here by
construction), joins clip UUIDs onto the trainer's episode ids, and hands
`train_v6_staged.v6_loss_step` its per-window batch keys
(`g_str_id … s2_valid`).

⛔ THE JOIN IS STABLE-ID ONLY. `build_v2_providers(stable_ids=True)` — the
default on the trainer path (`build_train_episodes` never overrides it) — gives
every episode `tanitad.data.v2_dataset.stable_episode_id(clip_id)` (blake2b>>1,
63-bit, collision-free). The 16-bit `episode_id` BAKED INTO the v2ep payloads
COLLIDES: 69/2400 train and 7/600 val clips share a legacy id
(`clip_index.json:_legacy_collisions`, MEASURED by the label build). An episode
that still carries a legacy id is therefore REFUSED outright — even one that is
unambiguous among the 801 LABELED clips can collide with an UNLABELED corpus
clip this loader cannot see, and a silent wrong-clip join would supervise the
strategic head with another scene's goal. Rebuilding the sidecar manifest
(`load_or_build_manifest(rebuild=True)`) upgrades any v2 cache to stable ids
for free; there is no admissible reason to join through the legacy id.

⛔ ROUTE_TO IS REFUSED, mirroring `colab/s2_schema.validate()`: G1 is CLOSED
(0/31) and `vocab_str` has no categorical arg channel (v6.py), so a ROUTE_TO
label is unsupervisable — a file carrying one is refused at load, never
silently skipped. Re-opening ROUTE_TO is a deliberate reviewed edit to the
schema module AND to this mirror, never a drive-by.

⛔ GOAL/SITUATION DISJOINTNESS (BINDING, Sayed 2026-08-03) is re-asserted ON
THE BYTES THIS PROCESS CONSUMES, not inherited from the builder: the goal
payload (g_str/a_str, and ONLY the payload — scanning the whole record would
match the record's own `disjointness` stamp key, the polling-monitor trap) must
not name a situation classifier, and the per-record stamp must assert False.

⚠️ TIMELINE. `t0_s` and `valid_window_s` live on the RAW clip timeline (the
bridged ego npz the labels were derived at, `t0_idx = 80` @ 10 Hz). A v2ep
provider drops the first ``n_stack - 1`` frames (`_scan_meta`: ``poses[k:]``),
so raw index = provider index + (n_stack − 1). ``S2WindowSupervision`` applies
that offset per episode, read off the episode's own channel count — a window's
"now" is its LAST frame (`pose_last`), the same instant every other label in
the batch is anchored to.

Validation scope: shape/type/vocabulary/id-consistency/mask discipline/
disjointness/ROUTE_TO are enforced here (the trainer's contract). The
per-token ALLOWED-SLOT table (`ARG_SLOT_SPEC`) is the BUILDER's law and stays
in `colab/s2_schema.py` — re-pinning it here would be a second copy that
drifts; every shipped record already passed it at build AND in the 797/797
re-scan (S2_V1_LABELS.md §3.3).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # stack root

from tanitad.models.v6 import (  # noqa: E402
    GOAL_ARG_SLOTS, STRATEGIC_ACTION_TOKENS, STRATEGIC_GOAL_TOKENS)

try:
    from tanitad.data.v2_dataset import stable_episode_id
except ImportError:                                       # pragma: no cover
    # the train_p8_occupancy.episode_uid_of_clip pattern: v2_dataset imports
    # torchvision.io at module level, which dev boxes lack. The SAME two-line
    # formula (blake2b digest_size=8 >> 1, v2_dataset.py:95-96); pinned equal
    # to the canonical one in tests/test_v6_s2_loss.py whenever both import —
    # and load_s2_labels additionally cross-checks EVERY index entry's
    # recorded stable id against this function, so a drifted copy refuses
    # loudly instead of joining nothing.
    import hashlib

    def stable_episode_id(clip_id: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(clip_id.encode("utf-8"), digest_size=8).digest(),
            "big") >> 1

__all__ = [
    "SCHEMA_VERSION", "IGNORE_ID", "ROUTE_TO_ID", "S2LabelError",
    "S2Row", "S2LabelSet", "S2WindowSupervision", "load_s2_labels",
    "assert_payload_disjoint",
]

SCHEMA_VERSION = "s2-strategic-v1"
INDEX_NAME = "clip_index.json"
#: torch's ``cross_entropy`` default ``ignore_index`` — a window outside the
#: validity band (or in an unlabeled clip) contributes NOTHING, by the same
#: IGNORE discipline the arg mask uses one level down.
IGNORE_ID = -100
ROUTE_TO_ID = STRATEGIC_GOAL_TOKENS.index("ROUTE_TO")
#: 16-bit legacy ids are ``int.from_bytes(clip_id[:4])`` < 2**32; a stable id
#: below 2**32 has probability ~2**-31 per clip. The bound separates the two
#: id families cheaply; the AUTHORITATIVE separation is membership in the
#: index's own maps, which is what the join actually consults.
_LEGACY_ID_BOUND = 1 << 32
_DISJOINT_NEEDLES = ("situation", "sitclf")


class S2LabelError(SystemExit):
    """A refusal of the S2 label artifact or its join.

    Its own subclass (the `ResumeLineageError` pattern) so a chain script can
    tell "the label artifact is bad / unjoinable" apart from every other
    `SystemExit` refusal in the trainer."""


def assert_payload_disjoint(rec: dict) -> None:
    """BINDING: no situation-classifier output may reach a goal/action field.

    Scans ONLY the goal payload (g_str / a_str) — the record's OWN
    ``disjointness`` stamp key contains the word ``situation``, so a
    whole-record scan would fire on the stamp that asserts compliance
    (CLAUDE.md's polling-monitor trap: the searched token must stay disjoint
    from the marker). Mirror of `colab/s2_schema.assert_disjoint`, re-run on
    the bytes THIS process consumes."""
    blob = json.dumps({k: rec.get(k) for k in ("g_str", "a_str")}).lower()
    for needle in _DISJOINT_NEEDLES:
        if needle in blob:
            raise S2LabelError(
                f"[s2] ⛔ goal/situation disjointness violated in "
                f"{rec.get('clip_id')!r}: {needle!r} appears in the goal "
                f"payload. A goal input must not carry the situation "
                f"classifier's output in any form (BINDING, Sayed "
                f"2026-08-03). This label file is refused whole.")


def _check_block(blk, tokens: tuple, where: str, clip: str
                 ) -> tuple[int, list[float], list[float]]:
    """One g_str/a_str block -> ``(token_id, args[8], mask[8])`` or a refusal."""
    if not isinstance(blk, dict):
        raise S2LabelError(f"[s2] ⛔ {clip}: {where} is not a dict")
    tok = blk.get("token")
    if tok not in tokens:
        raise S2LabelError(
            f"[s2] ⛔ {clip}: {where}.token {tok!r} is not in the v6 "
            f"vocabulary {tokens} — the labels and the model disagree on the "
            f"vocabulary, and training on a remapped guess is not an option.")
    if tok == "ROUTE_TO":
        raise S2LabelError(
            f"[s2] ⛔ {clip}: {where}.token is ROUTE_TO, which is GATED "
            f"(G1 CLOSED at 0/31 — sign text unverifiable; and vocab_str has "
            f"no categorical arg channel for its text_token_id arg). "
            f"`s2_schema.validate()` refuses it at build; this loader "
            f"mirrors the refusal so a hand-edited file cannot smuggle it to "
            f"the head. Emit the geometry token or NONE_ABSTAIN with a "
            f"reason, never ROUTE_TO.")
    tid = blk.get("token_id")
    if tid != tokens.index(tok):
        raise S2LabelError(
            f"[s2] ⛔ {clip}: {where}.token_id {tid!r} != "
            f"{tokens.index(tok)} for {tok!r} — the redundant id exists "
            f"precisely so a drifted vocabulary is a refusal, not a "
            f"silently-wrong CE target.")
    args, mask = blk.get("args"), blk.get("arg_mask")
    if not (isinstance(args, list) and len(args) == GOAL_ARG_SLOTS
            and all(isinstance(x, (int, float)) for x in args)):
        raise S2LabelError(f"[s2] ⛔ {clip}: {where}.args must be "
                           f"[{GOAL_ARG_SLOTS}] floats, got {args!r}")
    if not (isinstance(mask, list) and len(mask) == GOAL_ARG_SLOTS
            and all(m in (0, 1) for m in mask)):
        raise S2LabelError(f"[s2] ⛔ {clip}: {where}.arg_mask must be "
                           f"[{GOAL_ARG_SLOTS}] of 0/1, got {mask!r}")
    for i, m in enumerate(mask):
        if not m and float(args[i]) != 0.0:
            raise S2LabelError(
                f"[s2] ⛔ {clip}: {where} slot {i} is UNSET but carries "
                f"{args[i]} — an unset slot is 0.0 by convention (the masked "
                f"L1 would ignore it, but a value smuggled into an unset slot "
                f"is a schema violation upstream and is refused, not "
                f"laundered).")
    return int(tid), [float(x) for x in args], [float(m) for m in mask]


@dataclass(frozen=True)
class S2Row:
    """One clip's supervision, tensor-ready."""
    clip_id: str
    split: str
    g_id: int
    g_args: Tensor            # [8] float32
    g_mask: Tensor            # [8] float32
    a_id: int
    a_args: Tensor
    a_mask: Tensor
    t0_s: float
    band: tuple[float, float]
    g_token: str
    a_token: str
    g_provenance: str
    a_provenance: str


class S2LabelSet:
    """The loaded, validated label corpus + the id maps for the join."""

    def __init__(self, rows_by_stable: dict[int, S2Row],
                 legacy_ids: dict[int, tuple[str, ...]],
                 t0_s: float, band: tuple[float, float],
                 source: dict):
        self.rows_by_stable = rows_by_stable
        self.legacy_ids = legacy_ids
        self.t0_s = float(t0_s)
        self.band = (float(band[0]), float(band[1]))
        self.source = source

    def __len__(self) -> int:
        return len(self.rows_by_stable)

    def token_census(self) -> dict:
        """Per-token record counts — PER FAMILY, NEVER POOLED (the
        four-metric-families discipline applied to labels)."""
        g: dict[str, int] = {}
        a: dict[str, int] = {}
        for r in self.rows_by_stable.values():
            g[r.g_token] = g.get(r.g_token, 0) + 1
            a[r.a_token] = a.get(r.a_token, 0) + 1
        return {"g_str": dict(sorted(g.items())),
                "a_str": dict(sorted(a.items()))}

    def provenance_census(self) -> dict:
        """Raw material for the S-S gate's goal-provenance audit."""
        g: dict[str, int] = {}
        a: dict[str, int] = {}
        for r in self.rows_by_stable.values():
            g[r.g_provenance] = g.get(r.g_provenance, 0) + 1
            a[r.a_provenance] = a.get(r.a_provenance, 0) + 1
        return {"g_str": dict(sorted(g.items())),
                "a_str": dict(sorted(a.items()))}

    def report(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "n_records": len(self.rows_by_stable),
            "t0_s": self.t0_s, "valid_window_s": list(self.band),
            "token_census_records": self.token_census(),
            "provenance_census_records": self.provenance_census(),
            "join": "stable_episode_id (blake2b>>1) ONLY — legacy 16-bit ids "
                    "are refused (69/2400 + 7/600 collide)",
            "disjointness": "payload-only scan PASSED on every consumed "
                            "record + per-record stamp asserted False",
            **self.source,
            "_evidence_class": "MEASURED (ours; this load)",
        }

    def supervision(self, episodes, *, window: int, dt: float,
                    index) -> "S2WindowSupervision":
        return S2WindowSupervision(self, episodes, window=window, dt=dt,
                                   index=index)


def load_s2_labels(path) -> S2LabelSet:
    """Load + validate `s2-strategic-v1` labels and their join index.

    ``path`` is either the labels DIRECTORY (containing ``clip_index.json``
    and one or more ``s2_labels_*.jsonl``) or ONE ``.jsonl`` file (the index
    must sit beside it). Every refusal names the record and the rule; a label
    artifact that half-loads is worse than one that refuses whole."""
    p = Path(path)
    if not p.exists():
        raise S2LabelError(f"[s2] ⛔ --s2-labels {p} does not exist")
    if p.is_dir():
        idx_path = p / INDEX_NAME
        files = sorted(p.glob("s2_labels_*.jsonl"))
        if not files:
            raise S2LabelError(
                f"[s2] ⛔ {p} contains no s2_labels_*.jsonl — nothing to "
                f"supervise with. The v1 delivery ships "
                f"s2_labels_aug120.jsonl / s2_labels_w120val.jsonl.")
    else:
        idx_path = p.parent / INDEX_NAME
        files = [p]
    if not idx_path.exists():
        raise S2LabelError(
            f"[s2] ⛔ {idx_path} is MISSING. Without the clip index the "
            f"labels are UNJOINABLE — trainer episodes carry only an int "
            f"episode id, labels key on the clip UUID, and the index is the "
            f"only bridge (S2_STRATEGIC_GAP.md §1.2: 'without it the labels "
            f"are unjoinable and the term silently never fires'). Refusing "
            f"rather than silently never firing.")
    try:
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception as e:                                    # noqa: BLE001
        raise S2LabelError(f"[s2] ⛔ {idx_path} unreadable: "
                           f"{type(e).__name__}: {e}")
    clips = idx.get("clips")
    if not isinstance(clips, dict) or not clips:
        raise S2LabelError(f"[s2] ⛔ {idx_path} has no 'clips' map — not a "
                           f"clip_index.json")
    t0_s = float(idx.get("_t0_s", 8.0))
    band = tuple(idx.get("_valid_window_s", (-2.0, 2.0)))

    # ---- the id maps, with the index's own stable ids CROSS-CHECKED --------
    stable_of: dict[str, int] = {}
    legacy_ids: dict[int, list[str]] = {}
    excluded: set[str] = set()
    for cid, ent in clips.items():
        st = stable_episode_id(cid)
        recorded = ent.get("episode_id_stable")
        if recorded is not None and int(recorded) != st:
            raise S2LabelError(
                f"[s2] ⛔ {idx_path}: episode_id_stable for {cid} is "
                f"{recorded}, but tanitad.data.v2_dataset.stable_episode_id "
                f"computes {st}. The index and the code disagree on the hash "
                f"— one of them drifted, and a join under a drifted hash is "
                f"a silent zero-match. Rebuild the index or fix the drift.")
        if st in stable_of.values():                # ~2**-63 per pair; cheap
            raise S2LabelError(f"[s2] ⛔ stable-id collision inside "
                               f"{idx_path} at {cid} — refusing the join.")
        stable_of[cid] = st
        leg = ent.get("episode_id_legacy")
        if leg is not None:
            legacy_ids.setdefault(int(leg), []).append(cid)
        if ent.get("excluded"):
            excluded.add(cid)

    # ---- the records -------------------------------------------------------
    rows: dict[int, S2Row] = {}
    seen: set[str] = set()
    for f in files:
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(),
                                  start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception as e:                            # noqa: BLE001
                raise S2LabelError(f"[s2] ⛔ {f.name}:{ln} is not JSON: {e}")
            if rec.get("schema_version") != SCHEMA_VERSION:
                raise S2LabelError(
                    f"[s2] ⛔ {f.name}:{ln} schema_version "
                    f"{rec.get('schema_version')!r} != {SCHEMA_VERSION!r} — "
                    f"this loader consumes exactly the format the label "
                    f"build validated, nothing adjacent.")
            cid = rec.get("clip_id")
            if not cid or cid not in stable_of:
                raise S2LabelError(
                    f"[s2] ⛔ {f.name}:{ln}: clip_id {cid!r} is not in "
                    f"{INDEX_NAME} — an unjoinable label is a label that "
                    f"silently never fires, refused instead.")
            if cid in excluded:
                raise S2LabelError(
                    f"[s2] ⛔ {f.name}:{ln}: {cid} is marked EXCLUDED in the "
                    f"index (the triple-empty val records whose NONE_ABSTAIN "
                    f"is a default-of-absence, not a judgement) — a label "
                    f"for it is a manufactured abstain and is refused.")
            if cid in seen:
                raise S2LabelError(
                    f"[s2] ⛔ {f.name}:{ln}: duplicate record for {cid} — "
                    f"two labels for one clip is an ambiguous target.")
            seen.add(cid)
            dj = rec.get("disjointness")
            if not (isinstance(dj, dict)
                    and dj.get("situation_classifier_output_used") is False):
                raise S2LabelError(
                    f"[s2] ⛔ {f.name}:{ln}: {cid} carries no asserted "
                    f"disjointness stamp — the builder must state "
                    f"situation_classifier_output_used: false per record "
                    f"(BINDING).")
            assert_payload_disjoint(rec)
            g_id, g_args, g_mask = _check_block(
                rec.get("g_str"), STRATEGIC_GOAL_TOKENS, "g_str", cid)
            a_id, a_args, a_mask = _check_block(
                rec.get("a_str"), STRATEGIC_ACTION_TOKENS, "a_str", cid)
            r_t0 = float(rec.get("t0_s", t0_s))
            r_band = tuple(rec.get("valid_window_s", band))
            if not (r_band[0] <= 0.0 <= r_band[1]):
                raise S2LabelError(f"[s2] ⛔ {f.name}:{ln}: valid_window_s "
                                   f"{r_band} must bracket 0")
            rows[stable_of[cid]] = S2Row(
                clip_id=cid, split=clips[cid].get("label_split", "?"),
                g_id=g_id,
                g_args=torch.tensor(g_args, dtype=torch.float32),
                g_mask=torch.tensor(g_mask, dtype=torch.float32),
                a_id=a_id,
                a_args=torch.tensor(a_args, dtype=torch.float32),
                a_mask=torch.tensor(a_mask, dtype=torch.float32),
                t0_s=r_t0, band=(float(r_band[0]), float(r_band[1])),
                g_token=STRATEGIC_GOAL_TOKENS[g_id],
                a_token=STRATEGIC_ACTION_TOKENS[a_id],
                g_provenance=str(rec["g_str"].get("provenance")),
                a_provenance=str(rec["a_str"].get("provenance")))
    if not rows:
        raise S2LabelError(f"[s2] ⛔ {files}: zero records loaded")
    return S2LabelSet(rows, {k: tuple(v) for k, v in legacy_ids.items()},
                      t0_s, band,
                      {"labels_files": [str(f) for f in files],
                       "clip_index": str(idx_path),
                       "n_index_clips": len(clips),
                       "n_index_excluded": len(excluded)})


class S2WindowSupervision:
    """The per-window join: episodes × the label set -> batch keys.

    Built ONCE after the corpus; ``batch(idx)`` then assembles the seven
    tensors for a sampled window list in O(batch). A window is supervised
    (``s2_valid``) iff its episode joined AND its "now" (the last window
    frame, on the RAW clip timeline) falls inside the label's validity band.
    """

    def __init__(self, labels: S2LabelSet, episodes, *, window: int,
                 dt: float, index):
        self.labels = labels
        self._window = int(window)
        self._dt = float(dt)
        self._index = list(index)
        self._rows: list[S2Row | None] = []
        self._offs: list[int] = []
        legacy_hits: list[tuple[int, int]] = []
        for e_i, ep in enumerate(episodes):
            eid = int(ep.episode_id)
            row = labels.rows_by_stable.get(eid)
            if row is None and eid < _LEGACY_ID_BOUND \
                    and eid in labels.legacy_ids:
                legacy_hits.append((e_i, eid))
            self._rows.append(row)
            # raw clip index = provider index + (n_stack - 1): the provider
            # drops the first n_stack-1 frames (v2_dataset._scan_meta
            # ``poses[k:]``), and t0/band live on the RAW timeline.
            ch = int(ep.frames.shape[1])
            self._offs.append(max(ch // 3 - 1, 0))
        if legacy_hits:
            e_i, eid = legacy_hits[0]
            raise S2LabelError(
                f"[s2] ⛔ {len(legacy_hits)} episode(s) carry LEGACY 16-bit "
                f"ids (first: episode {e_i} id {eid}, which the index maps "
                f"to {labels.legacy_ids[eid]}). The legacy id COLLIDES "
                f"(69/2400 train + 7/600 val clips share one) and even an "
                f"id unique among the LABELED clips can collide with an "
                f"unlabeled corpus clip this loader cannot see — a silent "
                f"wrong-clip join would supervise the strategic head with "
                f"another scene's goal. The trainer path builds providers "
                f"with stable_ids=True; rebuild the cache manifest "
                f"(load_or_build_manifest(rebuild=True)) instead of joining "
                f"through the legacy id.")
        self.n_episodes = len(self._rows)
        self.n_matched_episodes = sum(r is not None for r in self._rows)
        # window-level accounting, per token, NEVER pooled — one pass, once.
        self.n_windows = len(self._index)
        self.n_windows_in_band = 0
        g_w: dict[str, int] = {}
        a_w: dict[str, int] = {}
        for e_i, t in self._index:
            row = self._rows[e_i]
            if row is None or not self._in_band(row, e_i, t):
                continue
            self.n_windows_in_band += 1
            g_w[row.g_token] = g_w.get(row.g_token, 0) + 1
            a_w[row.a_token] = a_w.get(row.a_token, 0) + 1
        self.window_token_census = {"g_str": dict(sorted(g_w.items())),
                                    "a_str": dict(sorted(a_w.items()))}

    def _in_band(self, row: S2Row, e_i: int, t: int) -> bool:
        t_now_s = (t + self._window - 1 + self._offs[e_i]) * self._dt
        return (row.t0_s + row.band[0] <= t_now_s
                <= row.t0_s + row.band[1])

    def report(self) -> dict:
        return {
            "n_episodes": self.n_episodes,
            "n_matched_episodes": self.n_matched_episodes,
            "n_windows": self.n_windows,
            "n_windows_in_band": self.n_windows_in_band,
            "window_token_census": self.window_token_census,
            "window": self._window, "dt": self._dt,
            "raw_offset_note": "raw = provider + (n_stack-1); t0/band are on "
                               "the raw clip timeline (t0_idx=80 @ 10 Hz)",
            "_evidence_class": "MEASURED (ours; this join)",
        }

    def batch(self, idx) -> dict[str, Tensor]:
        """Batch keys for ``v6_loss_step`` (CPU; caller moves to device)."""
        n = len(idx)
        out = {
            "g_str_id": torch.full((n,), IGNORE_ID, dtype=torch.long),
            "g_str_args": torch.zeros(n, GOAL_ARG_SLOTS),
            "g_str_arg_mask": torch.zeros(n, GOAL_ARG_SLOTS),
            "a_str_id": torch.full((n,), IGNORE_ID, dtype=torch.long),
            "a_str_args": torch.zeros(n, GOAL_ARG_SLOTS),
            "a_str_arg_mask": torch.zeros(n, GOAL_ARG_SLOTS),
            "s2_valid": torch.zeros(n, dtype=torch.bool),
        }
        for j, i in enumerate(idx):
            e_i, t = self._index[int(i)]
            row = self._rows[e_i]
            if row is None or not self._in_band(row, e_i, int(t)):
                continue
            out["s2_valid"][j] = True
            out["g_str_id"][j] = row.g_id
            out["g_str_args"][j] = row.g_args
            out["g_str_arg_mask"][j] = row.g_mask
            out["a_str_id"][j] = row.a_id
            out["a_str_args"][j] = row.a_args
            out["a_str_arg_mask"][j] = row.a_mask
        return out
