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
| `str_action` | existing `a_str` key | ✅ 1:1 |
| `str_goal` | existing `g_str` key | ✅ 1:1 |
| `t0_s`, `bands` | the existing `_in_band` test | ✅ same semantics |
| `tac_lat`, `tac_lon` | ⭐ **NEW** batch keys — factored, kept apart | new supervision |
| `tac_anchor` | ⭐ **NEW** key: the admissible predicted geometric goal point | new supervision |
| `audit` (incl. per-goal `disputed`) | ⛔ **NEVER a training input** (`v7_labels.py:117`) | audit only |
| `_oracle` | ⛔ only via `oracle_nav()`, stamped `allow_oracle_nav` in the manifest | gated |

⛔ **The factored pair must NOT be recombined into a single 5-way manoeuvre class.**
That lat+lon-mixing softmax is the programme's largest known defect and the whole
reason the factored labels exist.

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

### ⛔ 5.1 — THE BLOCKER: I CANNOT FIND THESE ARTIFACTS ANYWHERE THE TRAINER RUNS

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
single disks. ⚠️ State this as **"not found at five probes"**, never as *"it was never
built"* — the EVAL6 result shows the work is real, and the missing thing may be a
path I have not looked at.

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
