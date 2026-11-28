# PROPOSED addition to `CLAUDE.md` — traps preflight

**Status: ✅ APPLIED — commit `16d5ca550`. DO NOT APPLY AGAIN.**
Verified present at HEAD by content, attributed in-file as *"DE-C152 / class C82;
drafted by the DataFlyWheel, applied by the Master Mind on the PI's direct
authorisation 2026-08-29"*. The stale `Research Hub/Library/` → `Lab` line landed
in the same commit as a separately named change (0 occurrences remain at HEAD).

**How the authority resolved, recorded because the sequence is the point:** I
declined to apply this on a peer's request, and again when the same peer relayed
*"the PI approved it"* — a relayed approval is a report of authority, not the
authority, which is the identical gap one level up. It was applied by the Master
Mind **in their own session, on the PI's direct instruction to them**. That is a
sound path: they acted on authority they held, rather than on my behalf. The
history below is kept as written.

<details><summary>Original proposal text (superseded — kept for the record)</summary>

**Status at the time: ⏸ AWAITING THE PI. Not applied.**

The Master Mind offered me this edit ("your find, so yours to write"). I have
**not** applied it. `CLAUDE.md` is the instruction file that governs how every
agent in this programme behaves; a peer agent cannot authorise a change to it,
however well-reasoned the request — that authority is the PI's alone. The
finding itself is already banked where it belongs and needs no approval:
`Project Steering/RETRACTION_LOG.md` → **DE-C152 / class C82**.

**Sayed: if you want this in `CLAUDE.md`, say so and I will apply the text
below verbatim. If you would rather it live only in the retraction log, it
already does, and nothing is lost.**

---

## Proposed text (traps preflight, after the `df`/`free`/cgroup family)

> - ⛔ **AN ARTIFACT'S COST IS THE COST OF THE FILE ITS *CONSUMER* OPENS — NOT
>   THE ONE YOU FOUND ON DISK.** MEASURED 2026-08-29: I sized the B1 epcache by
>   loading `_epcache/.../ep_00000.pt` (`frames_u8 [199,9,256,256]` uint8 =
>   117.4 MB, **exactly** the file size), scaled it to 256×640 and published a
>   **1.38 TB capacity wall** — a format change, a geometry downgrade, a
>   sharding design and "a PI decision" all followed from it. Every number was
>   right; the **artifact** was wrong. The v7 trainer's `--v2-cache` reads
>   `*.v2ep.pt` (`tanitad/data/v2_dataset.py`, written by
>   `scripts/v2_compressed.py::build_compressed`), which stores **ENCODED**
>   frames — **34.0 MB/ep, 161 GB**, confirmed independently at ~36 MB/ep over
>   2,403 live episodes. **It fits with ~4× headroom; there was never a wall.**
>   ⇒ **Before pricing ANY derived artifact, open the CONSUMER'S LOADER and
>   price what IT reads; name the consumer and the loader file in the estimate,
>   or the estimate is inadmissible.** Same family as the `df` / Thor `free` /
>   cgroup `usage_in_bytes` / `step_s` traps with the object swapped: **a true
>   measurement quoted outside its scope reads exactly like an answer**, and it
>   is worse than no estimate because it manufactures decisions.
>   ⚠️ **Corollary — when you correct the artifact, RE-RUN the cost; never port
>   the old timing onto the new format.** Re-timing the real path changed the
>   plan more than the retraction did: **PNG encode is 70 % of the build**
>   (6.19 s decode+remap + 14.79 s encode), so the true wall-clock is
>   **3.4–4.6 h**, not the ~1.2 h a decode-only benchmark reported — and a 4 h
>   job was about to be started in a 1.5 h gap.
>   ⚠️ The codec is **load-bearing, not a speed knob**: `v2_dataset.py:325` and
>   `slice_v2_cache.py` **refuse to sub-frame a LOSSY cache**, so only a PNG
>   cache can be sliced to another geometry without a rebuild.

---

## Why this one is worth a traps slot rather than only a log entry

The existing `df` / `free` / cgroup / `step_s` entries all warn about **probes
that report the wrong scope**. This is the same error at the *planning* layer:
not a mis-read counter, but a correctly-read counter **about the wrong object**,
where the output is not a wrong diagnosis but a wrong *plan* — one that reached
a teammate as a decision requiring PI adjudication before it was caught.

</details>
