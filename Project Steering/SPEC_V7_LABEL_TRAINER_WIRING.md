# SPEC — wiring the v7.2 labels into the trainer

**Author** Master Mind · **Date** 2026-08-30 · **Status** BINDING (PI directive
2026-08-30: *"the trainer must use our last v7.2 labels"*) · **Implementation owner**
TrainingFlyWheel · **Register row** D-V72-WIRING.

## 1. The measured state — read from source, not inherited

| fact | evidence |
|---|---|
| the trainer is `stack/scripts/train_v6_staged.py` (7,529 lines) | ⚠️ **NOT** `stack/tanitad/train/`. My first probe there returned zero hits and would have read as *"no label path exists at all"*. **Absence at one location is not absence** — the second probe was `git ls-files`, which owns the fact. |
| `--s2-labels` is the trainer's ONLY label door | `:7315`, help at `:7317-7319` |
| it already accepts a **directory** carrying `clip_index.json` + `s2_labels_*.jsonl` | `:7317` |
| its loader refuses any record whose `schema_version != "s2-strategic-v1"` | `s2_labels.py:625-628` |
| `tanitad/data/v7_labels.py` is imported by **nothing except its own test** | `grep -rl v7_labels stack/ taniteval/` → the module, its test, pycache |

⇒ A fully built, fully tested reader for our schema is wired to nothing, while the
trainer's only label door opens onto a superseded corpus.

⭐ **Nobody chose that, and the phrasing matters.** It is *not* that "the trainer
expects different labels" — the trainer has no claim on a different label set. It is
that **the trainer cannot read ours at all**, which is a wiring defect on the trainer
side and never a reason to feed it older labels. ⚠️ Note also that pointing
`--s2-labels` at v7.2 **today** fails **loudly** at `s2_labels.py:625` rather than
mis-training silently. The guard already works; only the wiring is missing.

## 2. Routing: sniff the schema on the EXISTING flag — do not add a new one

Add a schema sniff to the `--s2-labels` path (dry-run `:4482`, train `:5164`) that
reads the first record's `schema_version` and dispatches:

* `s2-strategic-v1` → `load_s2_labels` — unchanged, byte-identical behaviour
* `s2-geom-v7` → `load_v7_labels` + the adapter in §3

⭐ **Why not a separate `--v7-labels` flag.** A second flag makes the label set an
*option*: every existing launch line stays valid while silently continuing to mean
"the superseded corpus". One door with a sniff makes the artifact self-describing —
the launch line names a path and the path chooses the reader — and it makes passing
both sets structurally impossible rather than merely discouraged.

⚠️ The sniff must read the **record**, never the filename. Path-based refusal already
exists (`_refuse_if_superseded`, `s2_labels.py:494`) and is not a substitute: a name
is a claim about a file, the `schema_version` field is the file.

## 3. ⛔ WHAT "DO NOT CONVERT" MEANS — and what it does NOT mean

The ruling stands: **never emit an `s2-strategic-v1` artifact from v7 labels.** That
downcast is lossy by construction — it discards the factored `tac_lat`/`tac_lon`,
`g_tac`'s `disputed` flags, the strata block and `lat_peak_m`: precisely what this
month produced.

⚠️ **It does NOT mean "keep away from the S2 code path", and reading it that way
would do real damage.** `S2WindowSupervision` (`s2_labels.py:705`) is not a schema —
it is the JOIN MACHINERY: the stable-id lookup, the validity-band test, the `n_stack`
frame offset, the per-token window census, and the **legacy-16-bit-id refusal**
(`:735-748`). Re-implementing that for v7 means re-implementing the collision guard,
and the census says the legacy ids **collide on 258 of 4,572 clips across 128 ids**.

⭐ And the sharp form of why that guard cannot be re-derived: a reimplementation does
not merely have to agree with `stable_episode_id` today — it has to reproduce the
legacy id's **collision structure** to be reconcilable at all. One clip wrong makes
the cross-check refuse the **whole** index. **A partial divergence is a total
refusal**, immediately, not by drift.

⇒ **Reuse the machinery. Replace only the rows.** The split is clean because v7 is a
strict superset on the fields the incumbent supervises:

| v7 field | destination | lossless? |
|---|---|---|
| `a_str.token` / `g_str.token` | existing `a_str` / `g_str` id | ✅ 1:1 |
| `a_str.args` / `g_str.args` | ⛔ **NOT 1:1 — a named DICT, not a vector** (§3.2) | needs an encoder |
| `t0_s` | the existing `_in_band` test | ✅ 1:1 |
| `bands` | ⛔ **NOT the same semantics — see §3.1** | three bands, not one |
| `tac_lat`, `tac_lon` | ⭐ **NEW** batch keys — factored, kept apart | new supervision |
| `tac_anchor` | ⭐ **NEW** key: the admissible predicted geometric goal point | new supervision |
| `audit` (incl. per-goal `disputed`) | ⛔ **NEVER a training input** (`v7_labels.py:117`) | audit only |
| `_oracle` | ⛔ only via `oracle_nav()`, stamped `allow_oracle_nav` in the manifest | gated |

⛔ **The factored pair must NOT be recombined into a single 5-way manoeuvre class.**
That lat+lon-mixing softmax is the programme's largest known defect and the whole
reason the factored labels exist.

### ⛔ 3.1 — v7 CARRIES THREE BANDS, ONE PER ABSTRACTION LAYER, AND THEY DIFFER ON EVERY RECORD

⚠️ **This corrects my own first draft, which called the band "same semantics".** MEASURED
on the v7 blob (301 records sampled, `s2_labels_v7.jsonl.gz` md5 `e22acf70…`):

```
bands.operative_s  [0.0,  2.0]
bands.tactical_s   [2.0,  6.0]
bands.strategic_s  [8.0, 30.0]      tactical_s != strategic_s on 301 of 301 records
```

The incumbent record has **one** `valid_window_s` and `S2WindowSupervision._in_band`
applies it to every family. v7 has **three**, and they differ on every record sampled.
⇒ **`_in_band` must be evaluated PER FAMILY**: the strategic pair against
`strategic_s`, the new factored-tactical keys against `tactical_s`. Applying one band
to all families would supervise the tactical heads over the strategic horizon —
silently, with no error and no count anomaly.

⚠️ **And the v7 record has NO `valid_window_s` key at all**, so
`rec.get("valid_window_s", band)` **silently falls back to the FILE-LEVEL default**.
That is the dangerous shape: not a crash, a plausible wrong answer. *(I initially
wrote that the key was present-with-`None` and would raise. It is simply absent and
the default fires — demonstrated, not assumed, which is why the claim survived
one round and this one did not.)*

⭐ **The three bands are not a formatting quirk — they are the hierarchy made
mechanical.** Operative validity is ~2 s, tactical ~4 s, strategic ~22 s. The nav
directive requires conditioning all three layers; the bands say each layer is valid
over a different horizon. Per-layer band evaluation therefore *is* the hierarchy, not
an implementation detail of it.

### ⛔ 3.2 — `args` IS A NAMED DICT, NOT THE INCUMBENT'S [8] VECTOR

`a_str` / `g_str` carry `{token, args, provenance, reason}` — the same four fields the
incumbent block has, which is why the token maps 1:1. But the payload differs:

```
"a_str": {"token": "PREPARE_TURN_R_FOLLOW_ROUTE",
          "args": {"within_m": 106.5, "by_time_s": 11.1},
          "provenance": "geometry", "reason": "106.5 m / 11.1s ahead, not yet begun"}
```

`_check_block` produces an `[8]` float vector plus its mask; v7's `args` is a **named
dict**. ⇒ the adapter needs an explicit arg encoder mapping named keys onto the
vector slots, and the **mask** must mark the slots the record did not name. Do not
assume positional agreement.

### ⭐ 3.3 — A SECOND, INDEPENDENT REASON NOT TO CONVERT

`load_s2_labels` requires a per-record `disjointness.situation_classifier_output_used:
false` stamp (5 references, incl. `assert_payload_disjoint`). **v7 records carry no
such stamp — 0 of 201 sampled.** `tanitad/data/v7_labels.py` has **zero** references
to it and loads all four artifacts (4,719 / 4,719 / 4,572 / 147).

⇒ converting to the old contract would have required **inventing** a disjointness
stamp our builder never emits — i.e. **asserting a property instead of recording
one**, which is precisely the failure the stamp exists to prevent. The vocabulary-loss
argument was the known reason not to convert; this is a second, independent one that
neither of us had found.

## 4. Two traps that will bite this wiring

⛔ **`_offs.append(max(ch // 3 - 1, 0))` (`s2_labels.py:734`) DERIVES the frame offset
from the CHANNEL COUNT.** It is correct only while `channels == 3 × n_stack`
(9 ch → 3 frames → offset 2). Nav conditioning is now mandatory on all three layers;
if nav ever enters as an **image channel** instead of a token, this constant silently
shifts the validity-band test and every supervised window moves — with no error.
⇒ **INVARIANT: nav enters as a TOKEN, never as an image channel**
(`SPEC_NAV_CONDITIONING_ALL_LAYERS.md`). If that ever changes, `_offs` must be fed
`n_stack` explicitly rather than deriving it. *Same class as DE-C152's derived
`HORIZON`: when a constant becomes derived, it silently changes the experiment.*

⚠️ **The manifest is load-bearing and must reach the run config.**
`LabelManifest.to_dict()` carries the md5 and `allow_oracle_nav`, and its own note
records that **six copies of this blob exist under three roots with differing md5s**.
A run whose config does not say which blob it read cannot be reproduced or audited.
⇒ write the manifest dict into the run config verbatim, as the S2 path already does
at `:4601`.

## 5. Artifacts — pin by md5, never by name

| role | path | md5 |
|---|---|---|
| train labels (4,572) | `labels/s2_labels_v7.2_train.jsonl.gz` | `0ff902130ce76886b8a925eceed9e3a5` |
| eval labels (147) | `labels/s2_labels_v7.2_eval.jsonl.gz` | `aa12c948f062181c3297265b51526ec5` |
| train episode join | `index/clip_index_v7.2_train.json` | round-trip verified, 0 id drift |
| eval episode join | `index/clip_index_v7.2_eval.json` | 147 records |

⚠️ These are the **schema-fixed rebuilds**; the earlier blobs were unloadable.

### ✅ 5.1 — RESOLVED: THE BLOBS ARE ON HF BY DESIGN (the probe log is kept below)

**MEASURED 2026-08-30, four locations plus the tool that owns the fact:**

| probe | result |
|---|---|
| `git ls-files` on my branch (`agent/arch-inf-20260803`) | **absent** |
| `find` over the whole repo working tree | **absent** (only the older `s2_labels_v7.jsonl.gz` and a validation sample) |
| the DataFlyWheel's worktree directory `.claude/worktrees/interesting-tharp-463cf3` | **absent** |
| `git ls-tree -r claude/interesting-tharp-463cf3` (their branch, tip `c29c659`) | **absent** |
| `find /home/nvidia` on **Thor**, where the trainer actually runs | **absent** |

⭐ **The EVAL6 cache, by contrast, verified immediately** —
`/home/nvidia/data/physicalai-b1-EVAL6-w120-256x640cyl` exists on Thor with exactly
**6** `.v2ep.pt` files, beside the B1 train cache. So this is not a general doubt
about the delivery; it is specifically the **label blobs and clip indexes** that are
not on any path the trainer can reach.

⛔ **Consequence: the wiring cannot be implemented as briefed.** "Point `--s2-labels`
at `labels/s2_labels_v7.2_train.jsonl.gz`" names a path that does not exist on Thor,
in the repo, or on the producing branch. A brief that says *just wire it up* would
send the implementer to discover this after starting.

⇒ **What is needed is the deliverable manifest, not another rebuild**: the artifact
and **where it lives** — repo path / Thor path / HF repo id. *An artifact in one
agent's context is not done* (operating standard, rule 3), and this is exactly the
failure that stranded TanitEval, REF-B v2's architecture and the pod ops bundle on
single disks. **✅ RESOLVED same day.** The blobs live on **HF `Sayood/tanitad-v7-training-corpus` (PRIVATE)** — `labels/`, `index/`, `splits/` — deliberately, because they are multi-MB data artifacts and the repo carries the CODE and the MANIFESTS that describe them. The repo-side code and records are **STAGED, not committed** (agents stage, never push), which is why `git ls-tree` on the producing branch shows a tip predating them. ⇒ **the five probes were the five places a REPO-RESIDENT artifact would be, and the artifact is not repo-resident.** ⭐ The correction worth keeping is not "look in more places" — it is that *"pushed and round-tripped"* names no address. **A remote is not a location; a repo id is.** That is the deliverable-manifest rule doing its job, and the reason the finding closed in one message instead of a day. ⚠️ Kept here rather than deleted because the probe log is what makes the resolution checkable.

## 6. The val round, and the two-root eval corpus

The 100-step val round consumes the **eval** side.

⛔ **THE EVAL CORPUS NOW LIVES IN TWO CACHE ROOTS, AND A LOADER POINTED AT ONE WILL
BE SILENTLY SHORT — the same shape we just fixed:**

| root | episodes |
|---|---|
| `physicalai-b1-w120-256x640cyl` | 141 |
| `physicalai-b1-EVAL6-w120-256x640cyl` | 6 (built 2026-08-30, 208 MB, 86 s) |

**The UNION is the eval set.** Read two roots. ⛔ **Do NOT symlink them together** —
the separation is the structural guarantee that the TRAIN cache is val40-clean, and a
symlink converts that guarantee back into a promise. Both roots are geometry-matched:
`requested_hfov 120.0 / achieved 120.0 / f_eff 305.5775 / observed_frac 1.0`.

⭐ The train cache was asserted unchanged **and** clean across that build — 4,713
episodes / 177,998,547,213 bytes / `_geometry.json` sha `f0d34b84`, all three
byte-identical, plus val40 members = 0 of 4,713 by digest. *Unchanged* only says the
build broke nothing; *clean* says it was right to begin with. Both are needed.

⛔ **The val40 overlap is now LIVE, not latent.** Our eval shares 6 clips with the
deployed val40, and per D-VAL40-NOLABELS those 6 are val40's **only** labelled clips
(34 of 40 carry none). ⇒ any **label-based** comparison between our eval score and a
val40 score would share its **entire** support — a reader comparing the two is
comparing the same six clips twice. Not a leak; a caveat that must travel with every
such comparison.

⚠️ The val machinery currently verifies its split by the `eval_split_v3` sha
(`ea8670e041c14ccb`). It will **refuse** the new artifact rather than silently load it
— the guard working. Repoint it to the v7.2 eval md5 above.

## 7. Verification — by CONTENT, never by exit code

1. `n_matched_episodes` must be **> 0** and reported; the trainer already refuses a
   zero join (`:5175`) — keep that refusal live on the v7 path.
2. Print the per-token **window census** for `a_str`/`g_str` **and** for the new
   `tac_lat`/`tac_lon` keys — **per family, never pooled**.
3. Assert `effective_mask` against presence (`v7_labels.py:357`). The suffix-collision
   defect that made an earlier mask/presence equality claim false is exactly why this
   assertion exists; do not skip it because it "should" hold.
4. The val loader must report **147**, and the report must name **both roots**. A run
   printing 141 is the defect recurring.
5. A `--dry-run` must print the manifest md5 and the joined counts before any GPU is
   spent — a preflight that fails in seconds instead of after the expensive part.
