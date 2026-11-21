"""The FULL Alpamayo2-Super augmentation, keyed by clip UUID.

⛔⛔ WHY THIS EXISTS — I DESCRIBED A DERIVED EXPORT AND CALLED IT THE DATASET.
MEASURED 2026-08-23 (RETRACTION_LOG C142). For weeks the label pipeline read a
per-clip export carrying four fields (`cot`, `lane`, `lateral`, `longitudinal`)
and I concluded, in code and in three reports, that *"`meta_action` is NOT
reachable locally"* and that *"the amount of data with Alpamayo is limited"*.

Both were wrong. The dataset is `Sayood/tanitad-alpamayo2-augmentation` on the
PI's HF account — **23,644 rows over 4,729 clips = 26.3 hours of driving**, in
FIVE tasks, of which the pipeline was using parts of one:

| task | rows | what it carries |
|---|---|---|
| `meta_action` | 4,729 | the 3-axis action + the CoT sentence |
| `trajectory` | 4,729 | Alpamayo's own predicted path, ADE/FDE vs GT |
| `auto_labeling` | 4,729 | `chain_of_causation` (100 %), component + motion analysis |
| `vqa` | 4,729 | 21 semantic categories, ~5 Q/A per clip |
| `grounding_via_vqa` | 4,728 | ⭐ **2D BOUNDING BOXES on 93.3 % of clips** |

⭐ **THE BOXES ARE REAL PERCEPTION, AND THEY ARE ACCURATE.** Car 2,292 ·
Pedestrian 1,079 · traffic light 889 · Truck 175 · Bike-with-rider 146 · Bus 40.
Verified by cropping the box region at full resolution on two independent clips:
on `683d37fb` the `Pedestrian` box lands exactly on a person walking in the road
at night, on `5355f3cc` the `Car` box lands exactly on the lead vehicle. A CoT
sentence is a generative claim; this is image-space evidence.

⛔⛔ **BUT THERE IS EXACTLY ONE GROUNDING QUESTION PER CLIP, AND IT IS SAMPLED.**
4,411 clips carry ONE distinct question; 318 carry none. The pedestrian question
is asked on only **1,165 of 4,729** clips, and **3,246 clips with boxes were
never asked about pedestrians at all**.

⛔⛔ **CORRECTED 2026-09-06 — 10 OF THAT "318 CARRY NONE" ARE A PARSE
FAILURE, NOT AN ABSENCE, AND THE PERCEPTION IS RECOVERABLE.** MEASURED over the
4,728 `grounding_via_vqa` rows: **4,411 box payloads parse, 10 do NOT (0.212 %),
307 genuinely carry none** (4,411 + 10 + 307 = 4,728). All ten fail the same
way — the payload has lost its leading `[{"bbox_2d": ` and begins mid-object,
e.g. `[325, 653, 491, 756], "label": "Bus"}]`. **Prepending that prefix recovers
10 of 10**; a deliberately WRONG prefix recovers 0 of 10 (the control).
Recovered labels: Truck 3 - car 2 - lead vehicle 2 - Bus 1 - Car 1 - Bike with
rider 1.

⚠️ **WHY IT IS NOT COSMETIC:** this module's own rule is that grounding can
CONFIRM a token and can NEVER REFUTE one, because a missing box overwhelmingly
means the question was not asked. A box row that FAILED TO PARSE is
indistinguishable from that ⇒ a dropped row silently converts *perception we
hold* into *perception we were never offered*, on exactly the clips where a
vehicle WAS boxed.

⭐ **FOUND BY THE `box_json_failed` COUNTER ADDED THE SAME DAY**, which is the
argument for the counters: the `except: arr = []` branch had been dropping these
ten since the module was written and nothing could see it. ⛔ **NO REPAIR IS
SHIPPED HERE** — patching a truncated generative output is a corpus-owner
decision, not a loader's, and inventing one would manufacture labels. The shape,
the recipe and the control are banked at
`—/Research/2026-09-06-cot-loader/raw/box_drop_probe.json` +
`box_recovery.json`; the fix belongs upstream in the export.

⇒ **Grounding can CONFIRM a token. It can NEVER REFUTE one.** A missing box of
kind X overwhelmingly means the question about X was not asked, not that X is
absent. I first wrote a `contradicted` state for "has boxes, none of kind X" —
that is invalid and is removed. This is the same sampled-bank scope error I
correctly identified for `vqa` in this very file and then failed to apply to
`grounding_via_vqa`, which has ONE question per clip rather than five.

⚠️ **THE ANCHOR IS 5.1 s, NOT 8.0 s.** `t0_us` is 5,100,000 on every row; the
s2 label anchor is `RAW_T0_S = 8.0`. Alpamayo describes the scene 2.9 s EARLIER
than our labels do — at 10 m/s, 29 m apart. Any agreement statistic computed
without `ALPAMAYO_T0_S` is comparing two different moments, and every such
number I published before today did exactly that.

⛔⛔ **AND UNTIL 2026-09-06 THE LOADER ANSWERED A MISSING SOURCE FILE
WITH `return {}` — SO EVERY CoT TOKEN VANISHED WITH NO ERROR.** On any
machine without the hard-coded local `RECORDS` path the augmentation read as
legitimately empty, and downstream *"this corpus states no speed limit"* and
*"the records file was not there"* were INDISTINGUISHABLE.

⭐ That is this programme's dominant failure class in a loader costume:
a search tool reporting *"no matches"* for files it could not open, a
`grep -c` returning 0 from an unreadable file, a committer exiting 0 having
committed nothing. In every one the EMPTY RESULT READS AS AN ANSWER.
⇒ **`_load()` now RAISES `AlpamayoRecordsUnavailable`.** The only empty is
the explicit, STAMPED opt-in (`TANITAD_ALPAMAYO_RECORDS_OPTIONAL=1`), which
warns and is reported by `load_report()` / `coverage()['records_available']`.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

RECORDS = "C:/Users/Admin/tanitad-data/alpamayo/records.parquet"
MANIFEST = "C:/Users/Admin/tanitad-data/alpamayo/selection_manifest.json"

#: md5 of the parquet this module was written against — MEASURED 2026-09-06
#: on the local mirror and independently against HF snapshot `cedbf57c…`:
#: 25,970,018 B, 23,644 rows, 4,729 distinct clips. Quoted in the failure
#: message so an operator can tell a MISSING file from a DIFFERENT one.
RECORDS_MD5 = "9f13474723b880eec7fcc09a7be478d8"

#: Point the loader elsewhere (a pod, an off-Drive clone, a test). Read at
#: CALL time, never at import time, so a test can set it and `cache_clear()`.
RECORDS_ENV = "TANITAD_ALPAMAYO_RECORDS"
#: ⚠️ The ONLY route to an empty result, and it is STAMPED rather than
#: silent: `_load()` warns, and `load_report()` / `coverage()` report
#: `records_available: False` with the path that was tried. A SILENT opt-out
#: would simply reinstate the defect this module exists to have removed.
RECORDS_OPTIONAL_ENV = "TANITAD_ALPAMAYO_RECORDS_OPTIONAL"


class AlpamayoRecordsUnavailable(RuntimeError):
    """The source parquet could not be read. ⛔ NEVER answered with `{}`.

    ⛔ An empty mapping here is INDISTINGUISHABLE from a corpus that
    genuinely carries no CoT, no boxes and no `meta_action`. The honest
    output of a failed read is INCONCLUSIVE, never ABSENT — so it raises.

    ⚠️ Callers that legitimately tolerate the empty case must opt in
    explicitly via `RECORDS_OPTIONAL_ENV`, which is warned and stamped.
    """


#: What the last `_load()` actually did — populated on EVERY path, the
#: failing one included, so a consumer can always tell an empty result apart
#: from an unread one. See `load_report()`.
_LOAD_REPORT: dict = {}


def records_path() -> str:
    """The parquet this module will read, `RECORDS_ENV` override honoured."""
    return os.environ.get(RECORDS_ENV) or RECORDS


def load_report() -> dict:
    """Provenance of the last load: path, availability, rows/clips, and the
    per-reason SWALLOW COUNTS.

    ⭐ The counters exist because a per-row `except: continue` is the
    same defect at row scale: a corrupted parquet would yield FEWER clips with
    no error at all. They are counted and stamped rather than thresholded —
    inventing a tolerance here would be the uncalibrated-threshold defect.
    """
    return dict(_LOAD_REPORT)

#: Alpamayo's own anchor on the raw clip timeline. NOT our 8.0 s s2 anchor.
ALPAMAYO_T0_S = 5.1

#: Box labels that ground a VRU claim, lower-cased at comparison time.
VRU_LABELS = ("pedestrian", "bike with rider", "motorcycle with rider", "cyclist")
VEHICLE_LABELS = ("car", "truck", "bus", "van", "lead vehicle")
LIGHT_LABELS = ("traffic light",)


@dataclass
class AlpamayoClip:
    """Everything the augmentation holds for one clip."""

    clip_id: str
    t0_s: float = ALPAMAYO_T0_S
    #: meta_action axes, e.g. {"longitudinal": "Gentle Deceleration",
    #:                        "lateral": "Steer Right", "lane": "Lane Keep"}
    meta_action: dict[str, str] = field(default_factory=dict)
    cot: str | None = None
    chain_of_causation: str | None = None
    components_analysis: str | None = None
    motion_analysis: str | None = None
    #: [(category, question, answer), ...]
    vqa: list[tuple[str, str, str]] = field(default_factory=list)
    #: [(question, label, [x0,y0,x1,y1]), ...] in a **0-1000 NORMALISED**
    #: space — NOT pixels. See `boxes_px()`.
    boxes: list[tuple[str, str, list[int]]] = field(default_factory=list)
    #: Alpamayo's own trajectory error against the recorded future
    ade_m: float | None = None
    fde_m: float | None = None

    # -- the axes, normalised -------------------------------------------------
    @property
    def lateral(self) -> str | None:
        """`left` | `right` | `straight`, or None if the axis is absent."""
        v = (self.meta_action.get("lateral") or "").lower()
        if not v:
            return None
        if "left" in v:
            return "left"
        if "right" in v:
            return "right"
        return "straight"

    @property
    def longitudinal(self) -> str | None:
        """`accel` | `decel` | `constant` | `stop`, or None."""
        v = (self.meta_action.get("longitudinal") or "").lower()
        if not v:
            return None
        if "stop" in v or "stationary" in v:
            return "stop"
        if "decel" in v or "brak" in v:
            return "decel"
        if "accel" in v:
            return "accel"
        return "constant"

    @property
    def lane(self) -> str | None:
        """`keep` | `change_left` | `change_right`, or None."""
        v = (self.meta_action.get("lane") or "").lower()
        if not v:
            return None
        if "left" in v:
            return "change_left"
        if "right" in v:
            return "change_right"
        return "keep"

    # -- grounding ------------------------------------------------------------
    def grounded(self, kind: str) -> bool:
        """Is there a BOX supporting a claim of this kind?

        ⭐ This is the fusion gate. A CoT sentence saying "yielding to a
        pedestrian" is a generative claim; the same clip carrying a
        `Pedestrian` box is perception. Only the second promotes a token out
        of ``disputed``.
        """
        want = {"vru": VRU_LABELS, "vehicle": VEHICLE_LABELS,
                "traffic_light": LIGHT_LABELS}.get(kind, ())
        return any(any(w in lab.lower() for w in want) for _, lab, _ in self.boxes)

    def box_labels(self) -> list[str]:
        return sorted({lab.strip().lower() for _, lab, _ in self.boxes})

    def boxes_px(self, width: int, height: int
                 ) -> list[tuple[str, str, tuple[float, float, float, float]]]:
        """Boxes in PIXELS for a `width` x `height` frame.

        ⛔⛔ THE BOXES ARE 0-1000 NORMALISED, NOT PIXELS — and on a 1920x1080
        frame the raw numbers FIT, so nothing complains. MEASURED 2026-08-23:
        rendering `[535, 667, 556, 728]` as pixels put a `Pedestrian` box on a
        blank wall, and I read that as *"Alpamayo mislocalises"*. Rescaled by
        `/1000`, the same box lands **exactly on a real pedestrian walking in
        the road** — confirmed by cropping the region at full resolution on two
        independent clips (`683d37fb` pedestrian, `5355f3cc` lead vehicle).

        The localisation is accurate. The renderer was wrong.

        ⇒ **"It fits inside the frame" is not a test of a coordinate space.**
        Same family as reading `df` on a pod: a plausible-looking number in the
        wrong units, accepted because nothing errored.
        """
        sx, sy = width / 1000.0, height / 1000.0
        return [(q, lab, (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy))
                for q, lab, b in self.boxes]

    @property
    def box_question(self) -> str | None:
        """The ONE grounding question asked of this clip.

        ⚠️ There is exactly one (4,411 clips have 1 distinct question, 318 have
        none). It is a SAMPLED question, so the absence of a box of some class
        is NOT evidence that the class is absent — most often that question was
        simply never asked. See `alpamayo_fusion.ground_tokens`.
        """
        return self.boxes[0][0] if self.boxes else None

    def vqa_answer(self, category: str, pattern: str) -> str | None:
        """First VQA answer in ``category`` whose question matches ``pattern``."""
        rx = re.compile(pattern, re.I)
        for cat, q, a in self.vqa:
            if cat == category and rx.search(q or ""):
                return a
        return None


def _first(v) -> str | None:
    """raw_json fields are 1-element lists; empty string means absent."""
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if isinstance(v, str):
        v = v.strip()
    return v or None


def _parse_meta(s: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (s or "").split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip().lower()] = v.strip().rstrip(".")
    return out


@lru_cache(maxsize=1)
def _load() -> dict[str, AlpamayoClip]:
    """Every clip in the augmentation. ⛔ RAISES rather than returning `{}`.

    ⛔⛔ THE PRE-2026-09-06 BODY WAS `if not os.path.exists(RECORDS):
    return {}`. It turned a missing FILE into a negative FINDING, silently, on
    every downstream token. See `AlpamayoRecordsUnavailable`.

    ⚠️ `lru_cache` does not memoise exceptions, so a failing load
    re-raises on every call; a SUCCESSFUL load is cached, so a test that
    repoints `RECORDS_ENV` must call `_load.cache_clear()`.
    """
    import pandas as pd

    path = records_path()
    _LOAD_REPORT.clear()
    _LOAD_REPORT.update(records_path=path, records_available=False,
                        rows=0, clips=0, optional_opt_out=False,
                        row_json_failed=0, auto_labeling_json_failed=0,
                        box_json_failed=0, meta_action_unparsed=0)
    if not os.path.exists(path):
        if os.environ.get(RECORDS_OPTIONAL_ENV) == "1":
            _LOAD_REPORT["optional_opt_out"] = True
            warnings.warn(
                "%s=1 and %r is absent: the Alpamayo augmentation is EMPTY "
                "for this process. Every CoT token, box and meta_action will "
                "be missing. This is an OPT-IN, STAMPED empty "
                "(see load_report()), not a negative finding."
                % (RECORDS_OPTIONAL_ENV, path), RuntimeWarning, stacklevel=2)
            return {}
        raise AlpamayoRecordsUnavailable(
            "Alpamayo records parquet not found: %r. Set %s to the parquet "
            "(expected md5 %s), or set %s=1 to opt in to a STAMPED empty "
            "result. Returning an empty mapping here would make 'the file is "
            "missing' indistinguishable from 'this corpus carries no CoT'."
            % (path, RECORDS_ENV, RECORDS_MD5, RECORDS_OPTIONAL_ENV))
    try:
        df = pd.read_parquet(path)
    except Exception as exc:               # unreadable is not the same as empty
        raise AlpamayoRecordsUnavailable(
            "Alpamayo records parquet at %r could not be read: %s: %s"
            % (path, type(exc).__name__, exc)) from exc
    out: dict[str, AlpamayoClip] = {}

    def get(cid: str) -> AlpamayoClip:
        if cid not in out:
            out[cid] = AlpamayoClip(clip_id=cid)
        return out[cid]

    for row in df.itertuples(index=False):
        cid = row.clip_id
        try:
            d = json.loads(row.raw_json)
        except Exception:
            _LOAD_REPORT["row_json_failed"] += 1      # counted, not silent
            continue
        c = get(cid)
        task = row.task

        if task == "meta_action":
            c.meta_action = _parse_meta(_first(d.get("meta_action")))
            if not c.meta_action:
                _LOAD_REPORT["meta_action_unparsed"] += 1
            c.cot = _first(d.get("cot"))

        elif task == "auto_labeling":
            raw = _first(d.get("cot_auto_labeling"))
            try:
                a = json.loads(raw) if raw else {}
            except Exception:
                _LOAD_REPORT["auto_labeling_json_failed"] += 1
                a = {}
            c.chain_of_causation = a.get("chain_of_causation") or None
            c.components_analysis = a.get("critical_components_analysis") or None
            c.motion_analysis = a.get("ego_vehicle_motion_analysis") or None

        elif task == "vqa":
            ans = _first(d.get("answer"))
            if ans:
                c.vqa.append((row.vqa_category or "", row.question or "", ans))

        elif task == "grounding_via_vqa":
            raw = _first(d.get("box"))
            try:
                arr = json.loads(raw) if raw else []
            except Exception:
                _LOAD_REPORT["box_json_failed"] += 1
                arr = []
            for b in arr if isinstance(arr, list) else []:
                if isinstance(b, dict) and "bbox_2d" in b:
                    c.boxes.append((row.question or "", str(b.get("label", "?")),
                                    list(b["bbox_2d"])))

        elif task == "trajectory":
            for k, dst in (("ade_m", "ade_m"), ("fde_m", "fde_m")):
                v = d.get(k)
                if isinstance(v, list) and v:
                    setattr(c, dst, float(v[0]))
                elif isinstance(v, (int, float)):
                    setattr(c, dst, float(v))

    if not out:
        # ⛔ A readable file that yields NOTHING is still an unusable
        # source, and it is exactly as indistinguishable from a real absence
        # as a missing file is. Same verdict.
        raise AlpamayoRecordsUnavailable(
            "Alpamayo records parquet at %r was read (%d rows) but produced "
            "ZERO clips. Expected md5 %s (23,644 rows / 4,729 clips)."
            % (path, len(df), RECORDS_MD5))
    _LOAD_REPORT.update(records_available=True, rows=int(len(df)),
                        clips=len(out))
    # ⛔ A COUNTER NOTHING READS IS STILL A SILENT SWALLOW. The per-row
    # `except` branches above drop content without failing, so any nonzero
    # count is surfaced HERE rather than left for a caller that never asks.
    # ⭐ A WARNING IS NOT A THRESHOLD. MEASURED 2026-09-06: 0 of 23,644 rows
    # fail to parse on the canonical artifact, so any failure at all is an
    # anomaly FOR THAT FILE — but another source could legitimately carry a
    # few, and inventing a tolerance would be the uncalibrated-threshold
    # defect. So: never silent, never refused on a guessed number.
    swallowed = {k: v for k, v in _LOAD_REPORT.items()
                 if k.endswith(("_failed", "_unparsed")) and v}
    if swallowed:
        warnings.warn(
            "Alpamayo records at %r: %d rows -> %d clips, but content was "
            "DROPPED without failing: %s. These are counted, not tolerated: "
            "decide whether the build is still admissible (load_report())."
            % (path, len(df), len(out), swallowed),
            RuntimeWarning, stacklevel=2)
    return out


def available() -> set[str]:
    return set(_load())


def get(clip_id: str) -> AlpamayoClip | None:
    """The augmentation for one clip, or None. Never approximates."""
    return _load().get(clip_id)


def coverage() -> dict:
    """What the corpus actually holds — for a report that must not guess.

    ⛔ `records_available` and `records_path` are present on EVERY return,
    the empty one included, so a consumer can never read `clips: 0` as *"the
    corpus is empty"* when it means *"the file was not there"*. That empty
    return is now reachable only under the stamped opt-out.
    """
    d = _load()
    n = len(d)
    rep = load_report()
    stamp = {
        "records_available": bool(rep.get("records_available")),
        "records_path": rep.get("records_path"),
        "optional_opt_out": bool(rep.get("optional_opt_out")),
        "row_json_failed": rep.get("row_json_failed", 0),
    }
    if not n:
        return dict(stamp, clips=0)
    return {
        **stamp,
        "clips": n,
        "hours": round(n * 20.0 / 3600.0, 1),
        "with_meta_action": sum(1 for c in d.values() if c.meta_action),
        "with_cot": sum(1 for c in d.values() if c.cot),
        "with_chain_of_causation": sum(1 for c in d.values() if c.chain_of_causation),
        "with_boxes": sum(1 for c in d.values() if c.boxes),
        "with_vqa": sum(1 for c in d.values() if c.vqa),
        "with_trajectory": sum(1 for c in d.values() if c.ade_m is not None),
        "grounded_vru": sum(1 for c in d.values() if c.grounded("vru")),
        "grounded_vehicle": sum(1 for c in d.values() if c.grounded("vehicle")),
        "grounded_light": sum(1 for c in d.values() if c.grounded("traffic_light")),
    }
