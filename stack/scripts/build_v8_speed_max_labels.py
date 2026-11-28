"""Build the v8 ``speed_max_input`` block from the v7.2 labels we ALREADY HOLD.

⭐⭐ THE PI'S RULING, VERBATIM (2026-09-10): *"regarding to max speed as input,
stick to the labels we created in the data set with the logic of minimal speed
etc..."*

⇒ This is a **PURE TRANSFORM of fields already in the v7.2 release**. No sign
detector, no re-ask of Alpamayo, no new label generation, no re-selection of
clips. Every output record is one input record plus one added top-level key, in
the same order, over the same clip set -- ``--verify`` asserts exactly that,
because re-selecting episodes breaks cross-arm comparability and must be refused.

============================================================================
⛔⛔ WHAT THIS CHANNEL IS, DECLARED AND NEVER HIDDEN
============================================================================

``speed_max_input.v_max_ms`` **is** ``g_tac.goals.SPEED_BAND.v_hi_ms``, which is
the **maximum of the ego's OWN REALISED speed over [t0+2 s, +6 s]**
(``s2_geom_emit_v7.py:309-311``, ``v.max()`` over the band). Its provenance is
therefore ``ego-future``, and quantization does not launder it: MEASURED 5-fold
out-of-fold, clip-disjoint, n = 4,572, the raw ceiling is recoverable from the
pinned 8-step bin plus ``v0`` at **R^2 = 0.9702**
(``Research/2026-09-06-max-speed-input/raw/quantization_panel_train.txt``).

⭐ THE PI HAS DECIDED TO USE IT ANYWAY, AND THAT DECISION IS CONSISTENT with the
programme's existing position rather than a new exception: refcv5-v2 already
feeds an **oracle nav command** whose own ``config.json`` declares
``nav_cmd_derivation: "v7.2 nav_command token (oracle, provenance ego-future;
allow_oracle_nav=True)"``. The standing rule is that the nav command is an
**INPUT simulating the vehicle's nav system, not a training signal**; the max
speed is the same class of signal, standing in for a **speed-limit service**.

⛔ **THEREFORE THE BINDING REQUIREMENT IS DECLARATION, NOT REFUSAL.** Three
mechanisms carry it, and none of them is a comment:

1. every emitted block carries ``oracle: true``, ``provenance: "ego-future"``
   and a ``derivation`` string naming the source field and the window;
2. the blob is readable only through ``v7_labels.oracle_max_speed``, which
   refuses without ``allow_oracle_nav`` on the manifest;
3. the trainer stamps ``speed_max_derivation`` into ``config.json`` and
   **REFUSES TO START** without it (``refc_v3_train._assert_speed_max_stamp``).

⛔ No capability claim may be credited to this channel without that stamp beside
it. It is written into the pre-registration for the same reason.

============================================================================
⭐ THE LADDER -- WHICH FIELD IS THE CEILING, WHICH IS THE FLOOR, AND WHY
============================================================================

The pinned ladder (``max_speed_input.POSTED_LIMIT_STEPS_KMH``, unchanged here --
this builder does not get to re-pin it) is::

    km/h   20        30        50        70        80        100       120       130
    m/s    5.555556  8.333333  13.888889 19.444444 22.222222 27.777778 33.333333 36.111111

VALUES come from road law; MEMBERSHIP came from the corpus. ⛔ No step is a
quantile.

* **CEILING = ``v_hi_ms``**, snapped **UP** (``q = min{s : s >= v}``). This is
  what a posted limit *is* -- the smallest real sign that admits the speed
  observed -- and it is the only value the model-facing block encodes. Above the
  top step there is no larger real sign, so the value is the top step and
  ``over_ceiling`` is True: **a real limit can be exceeded**, and the ego-derived
  raw value cannot be (it *is* the max).
* **FLOOR = ``v_lo_ms``**, snapped **DOWN** (``f = max{s : s <= v}``). This is the
  PI's *"logic of minimal speed"*: the band's own minimum, on the same ladder,
  in the mirror direction. Below the bottom step there is no smaller real sign,
  so the floor is the bottom step and ``under_floor`` is True.

⭐⭐ **AND THE "30 km/h FLOOR" PREMISE IS ALREADY SETTLED -- DO NOT RE-DERIVE IT.**
The ladder's bottom step is **20 km/h**, and because :func:`snap_up` never returns
below its first step, **every clip receives a ceiling and none is null** --
including a fully stopped ego (``v_hi = 0.00 m/s`` snaps to 20 km/h). A synthetic
30 km/h floor was never needed. ``--verify`` prints the null count, which must be 0.

⚠️ **THE FLOOR IS RECORDED, NOT WIRED, AND THAT IS DELIBERATE.** The model-facing
block is ``MAX_SPEED_DIMS = 3`` = ``(value_norm, over_ceiling, valid)``; there is
no floor slot. Widening it would change the seam, break the parameter pin in
``stack/tests/test_max_speed_wiring.py``, and make refcv6 a **multi-lever** arm --
which is exactly what a single-lever pre-registration forbids. So the floor ships
in the *record* (where it stratifies a result and documents the band) and reaches
the *model* not at all. An arm that wants to feed it must say so and pre-register
it separately.

============================================================================
⛔ THE ROUNDING TRAP -- WHY NO ``v_max_bucket_ms`` IS EMITTED
============================================================================

MEASURED 2026-09-07 and recorded in ``refc_v3_train.enable_max_speed``'s own
docstring: a builder that shipped ``v_max_bucket_ms`` **rounded to 4 dp**
(``round(50/3.6, 4) = 13.8889`` for a step that is ``13.888888...``) shipped a
bucket **strictly GREATER than the step it names**. Feeding it back through the
ladder moved **2,631 of 4,572 clips (57.5 %) one step up** -- 50->70 on 1,593,
20->30 on 809, 100->120 on 229 -- and the build **verified against its own rounded
ladder**, read 0 % moved, and passed its own gate.

⇒ **This builder emits the bucket ONLY as an INTEGER km/h** (``v_max_bucket_kmh``
/ ``v_min_bucket_kmh``). There is no m/s bucket to round, so the defect has no
surface to live on, and the consumer's own content assertion
(``round(quantized_ms * 3.6) == v_max_bucket_kmh``,
``refc_v3_train.py:1574-1584``) is a comparison between an integer and an integer.
``stack/tests/test_v8_speed_max_builder.py`` pins the ABSENCE of an m/s bucket and
carries a **deliberate-regression arm that reintroduces the 4-dp rounding and must
go RED**.

⛔ And ``--verify`` scores this build against an **INDEPENDENTLY AUTHORED** ladder
(``verify_ladder_independent.py``) written in **exact rational arithmetic over the
km/h INTEGERS** -- never against this module's own derivation. A cross-check that
re-runs the producer's derivation measures determinism, not correctness.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tanitad.refs import max_speed_input as msi  # noqa: E402

#: The block's schema tag. ⭐ Spelled to match the v8 payload already pinned by
#: ``stack/tests/test_max_speed_input.py:150``
#: (``test_the_v8_label_payload_declares_units_and_reads``), whose docstring
#: says it is the block the v8 builder writes -- so the shipped record and the
#: test's literal agree rather than diverging by one word. ⚠️ No code path reads
#: this key; ``max_speed_input.artifact_meta`` separately declares the MODULE's
#: schema as ``tanitad.max_speed_input/1``, and the two describe different
#: objects (the label block vs the model-facing meta).
BLOCK_SCHEMA = "speed_max_input/1"

#: ⭐ THE PROVENANCE STRING THE RECORD CARRIES, and the same sentence the
#: trainer stamps into ``config.json``. It names the SOURCE FIELD, the WINDOW,
#: and the word ORACLE, because a reader who opens the artifact in isolation --
#: which is how artifacts are usually opened -- must not have to go and find the
#: run record to learn what the number is.
SPEED_MAX_DERIVATION = (
    "v7.2 g_tac.goals.SPEED_BAND.v_hi_ms (oracle, provenance ego-future: max of "
    "the ego's OWN REALISED speed over [t0+2 s, +6 s]; floor from .v_lo_ms; "
    "allow_oracle_nav=True). INPUT standing in for a speed-limit service, never "
    "a training signal.")

#: What the ceiling and floor are read from, spelled out per record.
SOURCE_FIELD = ("g_tac.goals.SPEED_BAND.v_hi_ms (ceiling, snapped UP) / "
                "g_tac.goals.SPEED_BAND.v_lo_ms (floor, snapped DOWN)")

#: ⛔ The window the value describes, in the record's own band units.
BAND_S = (2.0, 6.0)

#: ⛔⛔ THE RECORD MUST SAY WHICH RELEASE IT IS. The v7.2 records carry
#: ``release: "v7.2"``, and a blob whose FILENAME says v8 while its RECORDS say
#: v7.2 is the artifact-does-not-declare-itself failure again: ``release`` is
#: precisely the field a reader consults to learn what they hold, and a
#: filename is not a declaration (a file is renamed far more easily than a
#: record). ⚠️ ``schema_version`` and ``vocab`` are NOT touched -- they stay
#: ``s2-geom-v7`` / ``v7`` because ``load_v7_labels`` REFUSES anything else, and
#: the vocabulary genuinely did not change. Only the release did.
RELEASE = "v8.0"

#: What ``_provenance`` gains, so the record's own provenance block names the
#: new field rather than leaving it undescribed beside four that are described.
PROVENANCE_ENTRY = {
    "source": SOURCE_FIELD,
    "oracle": True,
    "provenance": "ego-future",
    "window_s": list(BAND_S),
    "derivation": SPEED_MAX_DERIVATION,
    "built_by": "stack/scripts/build_v8_speed_max_labels.py",
    "note": ("PURE TRANSFORM of fields already in the v7.2 release -- no sign "
             "detector, no new label generation, no re-selection of clips."),
}


class BuildRefused(SystemExit):
    """The build refuses rather than fabricating a ceiling."""


def snap_up(v_ms: float) -> tuple[int, bool]:
    """CEILING: ``min{k in ladder : k/3.6 >= v}``, as an INTEGER km/h.

    Delegates the comparison to :func:`max_speed_input.quantize_up` so the
    corpus is quantized by the pinned module and by nothing else, then converts
    the step back to its km/h integer. ⛔ The integer is the artifact; no m/s
    bucket is emitted (see the module docstring).
    """
    q_ms, over = msi.quantize_up(float(v_ms))
    return int(round(q_ms * 3.6)), bool(over)


def snap_down(v_ms: float) -> tuple[int, bool]:
    """FLOOR: ``max{k in ladder : k/3.6 <= v}``, as an INTEGER km/h.

    The mirror of :func:`snap_up`, and the PI's *"logic of minimal speed"*.
    Below the bottom step there is no smaller real sign, so the floor is the
    bottom step and ``under_floor`` is True -- exactly as ``over_ceiling``
    reports a clip above the top step rather than inventing a sign that exists
    nowhere.
    """
    if not (isinstance(v_ms, (int, float)) and math.isfinite(v_ms)):
        raise ValueError(f"v_ms must be a finite number, got {v_ms!r}")
    best = None
    for s_ms, k in zip(msi.POSTED_LIMIT_STEPS_MS, msi.POSTED_LIMIT_STEPS_KMH):
        if s_ms <= float(v_ms):
            best = k
    if best is None:
        return int(msi.POSTED_LIMIT_STEPS_KMH[0]), True
    return int(best), False


def build_block(rec: dict) -> dict:
    """The ``speed_max_input`` block for one v7.2 record.

    ⛔ Refuses a record with no ``SPEED_BAND`` or a non-finite bound rather than
    emitting a default. A fabricated ceiling is worse than a missing one: the
    trainer's refusal can see a missing block and cannot see a wrong number.
    """
    clip = rec.get("clip_id")
    sb = ((rec.get("g_tac") or {}).get("goals") or {}).get("SPEED_BAND")
    if not isinstance(sb, dict):
        raise BuildRefused(
            f"[v8-build] clip {clip!r} carries no `g_tac.goals.SPEED_BAND`. "
            f"This build is a PURE TRANSFORM of that field -- refusing to "
            f"fabricate a ceiling for a record that has none.")
    v_hi, v_lo = sb.get("v_hi_ms"), sb.get("v_lo_ms")
    for name, v in (("v_hi_ms", v_hi), ("v_lo_ms", v_lo)):
        if not (isinstance(v, (int, float)) and math.isfinite(v)):
            raise BuildRefused(
                f"[v8-build] clip {clip!r}: SPEED_BAND.{name} is {v!r}, not a "
                f"finite number. Refusing rather than coercing.")
    if float(v_lo) > float(v_hi):
        raise BuildRefused(
            f"[v8-build] clip {clip!r}: SPEED_BAND floor {v_lo} exceeds ceiling "
            f"{v_hi}. The band is inverted -- refusing rather than swapping "
            f"them silently.")
    k_hi, over = snap_up(float(v_hi))
    k_lo, under = snap_down(float(v_lo))
    return {
        "schema": BLOCK_SCHEMA,
        # ---- the CEILING: what the consumer reads -------------------------
        # ⛔ RAW m/s. `refc_v3_train.enable_max_speed` ships this value
        # unquantized and the ladder is applied ONCE, model-side, in
        # `max_speed_input.encode_block`.
        "v_max_ms": round(float(v_hi), 2),
        # ⛔⛔ UNITS ON THE WIRE, both spellings the reader accepts. A payload
        # that declares none is REFUSED by `read_max_speed_field` -- m/s vs
        # km/h vs mph is a 1.61x spread and this programme published a 396 g
        # anchor table from exactly that omission.
        "units": msi.CONTROL_UNITS,
        "control_units": msi.CONTROL_UNITS,
        # ⛔ INTEGER km/h ONLY. No `v_max_bucket_ms` exists to be rounded to
        # 4 dp; see the module docstring's 2,631/4,572 (57.5 %) measurement.
        "v_max_bucket_kmh": k_hi,
        "bucket_units": "km_h (INTEGER; no m/s bucket is shipped, by design)",
        "bucket_steps_kmh": list(msi.POSTED_LIMIT_STEPS_KMH),
        "over_ceiling": bool(over),
        # ---- the FLOOR: the PI's "logic of minimal speed" -----------------
        # ⚠️ RECORDED, NOT WIRED. `MAX_SPEED_DIMS` is 3 and carries no floor
        # slot; widening the seam would make refcv6 a multi-lever arm.
        "v_min_ms": round(float(v_lo), 2),
        "v_min_bucket_kmh": k_lo,
        "under_floor": bool(under),
        "band_width_ms": round(float(v_hi) - float(v_lo), 2),
        # the record's own held flag, carried through so a result can be
        # stratified by "was this a held band or an accelerating window?"
        # without re-opening the label blob.
        "held": bool(sb.get("held", False)),
        "band_s": list(sb.get("band_s") or BAND_S),
        # ---- the declaration ---------------------------------------------
        "oracle": True,
        "provenance": "ego-future",
        "derivation": SPEED_MAX_DERIVATION,
        "source_field": SOURCE_FIELD,
        "role": ("INPUT CHANNEL, user/nav-supplied at inference, like "
                 "nav_command. NEVER a training signal."),
    }


def _read(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()], md5


def build_split(src: Path, dst: Path) -> dict:
    """Transform one split. Returns the census that goes beside the blob."""
    recs, src_md5 = _read(src)
    if not recs:
        raise BuildRefused(f"[v8-build] {src} holds no records.")
    src_clips = [r.get("clip_id") for r in recs]

    out, buckets, floors = [], Counter(), Counter()
    n_over = n_under = n_null = 0
    # ⭐ SAME-BREATH POSITIVE CONTROLS. A zero anywhere below is a claim about
    # the CONTENT only because these two read non-zero through the same loader
    # in the same pass; a blob that could not be opened would fail both.
    ctl_clip = sum(1 for r in recs if r.get("clip_id"))
    ctl_gtac = sum(1 for r in recs if r.get("g_tac"))
    for r in recs:
        blk = build_block(r)
        if blk.get("v_max_ms") is None:
            n_null += 1
        n_over += int(blk["over_ceiling"])
        n_under += int(blk["under_floor"])
        buckets[blk["v_max_bucket_kmh"]] += 1
        floors[blk["v_min_bucket_kmh"]] += 1
        r2 = dict(r)
        r2["speed_max_input"] = blk
        # ⛔ the record declares WHICH RELEASE it is; a filename is not a
        # declaration. schema_version / vocab are deliberately untouched.
        r2["release"] = RELEASE
        prov = dict(r2.get("_provenance") or {})
        prov["speed_max_input"] = PROVENANCE_ENTRY
        r2["_provenance"] = prov
        out.append(r2)

    # ⛔ PARITY: one output record per input record, SAME clips, SAME order.
    # Anything that re-selects episodes breaks cross-arm comparability.
    if [r.get("clip_id") for r in out] != src_clips:
        raise BuildRefused(
            "[v8-build] ⛔ the output clip sequence differs from the input's. "
            "This build is a pure transform and must not re-select or reorder "
            "episodes -- parity is sacred.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dst, "wt", encoding="utf-8", newline="\n") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    _, dst_md5 = _read(dst)

    return {
        "src": str(src), "src_md5": src_md5,
        "dst": str(dst), "dst_md5": dst_md5,
        "release": RELEASE,
        "schema_version_UNCHANGED": recs[0].get("schema_version"),
        "vocab_UNCHANGED": recs[0].get("vocab"),
        "n_records": len(recs),
        "n_with_block": len(out),
        "coverage": round(len(out) / len(recs), 6),
        "control_clip_id_present": ctl_clip,
        "control_g_tac_present": ctl_gtac,
        "n_null_ceiling": n_null,
        "n_over_ceiling": n_over,
        "n_under_floor": n_under,
        "ceiling_bucket_hist_kmh": dict(sorted(buckets.items())),
        "floor_bucket_hist_kmh": dict(sorted(floors.items())),
        "ladder_kmh": list(msi.POSTED_LIMIT_STEPS_KMH),
        "ladder_ms": [repr(s) for s in msi.POSTED_LIMIT_STEPS_MS],
        "derivation": SPEED_MAX_DERIVATION,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-in", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--eval-in", required=True)
    ap.add_argument("--eval-out", required=True)
    ap.add_argument("--report", default=None,
                    help="write the census JSON here as well as to stdout")
    a = ap.parse_args(argv)

    rep = {
        "tool": "stack/scripts/build_v8_speed_max_labels.py",
        "block_schema": BLOCK_SCHEMA,
        "splits": {
            "train": build_split(Path(a.train_in), Path(a.train_out)),
            "eval": build_split(Path(a.eval_in), Path(a.eval_out)),
        },
    }
    txt = json.dumps(rep, indent=1)
    print(txt)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
