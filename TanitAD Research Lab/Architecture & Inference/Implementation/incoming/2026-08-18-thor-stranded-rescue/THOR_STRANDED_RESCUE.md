# A11 — the Thor stranded-file rescue: **98 files banked, not 45** — and the drift instrument could only see 47 of them

**2026-08-18 · `tanitad-thor-wifi` (Jetson AGX Thor, `thor6`) → repo · 0 GPU · read-only pull**

> **Headline.** BACKLOG A11 was scoped from `pod_git_drift.py`, which **scans `*.py` and `*.sh` only**.
> A content-hash sweep of the same directories over **all** file types found **102 files present on
> Thor and absent from the repo by content** — more than double A11's 45. **98 are now in the repo,
> every one sha256-verified on both sides; 4 were deliberately left behind with reasons.**
>
> ⭐ **The single most consequential find is not a missing file, it is a WRONG one.** The repo's
> banked `thor_profile.py` (`…/incoming/2026-08-02-thor-deployment-profile/`) **cannot have produced
> its own co-banked `thor_profile.json`** — the JSON carries `"frame": "176x624 hfov 117.0"`, and the
> repo copy of the script **never assigns `out['frame']`**. The producing version existed only on
> Thor. See §4; this needs adjudication, not just filing.

---

## 1. What was actually there — MEASURED, and it is not 45

| evidence class | claim |
|---|---|
| **MEASURED** (`raw/drift_thor_after.json`, re-derived here) | `pod_git_drift.py` found **310** `HOST_ONLY` rows on Thor; **47** of them outside the third-party `alpasim/` checkout. **A11's "45" undercounts the `/home/nvidia` root by 2** — the tool's own `.txt` prints `21 /home/nvidia`, A11 records 19. The other seven directory counts in A11 are exact. |
| **MEASURED** (`raw/thor_sha256_inventory.txt`) | Widening from `.py`/`.sh` to every text-class file ≤3 MB in the same roots: **225 candidates, 123 already in the repo by content, 102 not.** |
| **MEASURED** (`raw/verify_sha256.txt`) | **98 pulled, 98/98 sha256-identical** to the source bytes. **0 mismatches.** |

**Why the instrument under-reported.** `stack/scripts/pod_git_drift.py:91` is `SUFFIXES = (".py", ".sh")`.
Every stranded `.json` result, `.md` note, `.log` transcript, `.yaml`, `.usda` and `.bak` is
**invisible to it by construction** — the same shape as its already-documented box→repo blind spot,
one axis over. That is an instrument finding, not a one-off: see §6.

⚠️ **Absence at one location is not absence — and here the trap was the converse, false PRESENCE.**
A basename probe of the repo reported `FINDINGS.md`, `S1_CLIMBOUT.md`, `hf_expected_train.json`,
`isp_report.json` and 20 more as "already banked". A **content** probe then showed that **10 of the 98
files pulled have a same-named repo file with different bytes** — `FINDINGS.md` among them.
**Presence of a name is not presence of the bytes**, which is exactly how §4 stayed hidden.

## 2. What was pulled — 98 files, 749,054 B

Method: `ssh -n 'cd /home/nvidia && tar -czf - <explicit file list>' > local.tgz`. **Nothing was
written on Thor** — tar streams to stdout. sha256 was computed on Thor *before* transfer and
re-computed on each repo copy; `raw/verify_sha256.txt` holds both.

⚠️ **The sha256 in `raw/` is the SOURCE-BYTE record and will not survive a round-trip through
checkout.** Staging emitted the usual `LF will be replaced by CRLF` warnings, so a later working copy
of these files will hash differently while the index blob is unchanged. That is the `CRLF_ONLY`
category the drift tool already knows about — **compare LF-normalised, not raw, when re-checking these.**

| bank path | n | bytes | what it is |
|---|---:|---:|---|
| `rescued/nurec_work/` | 37 | 397,307 | ⭐ **The NuRec/gsplat derivation chain.** `walk1/walk2.py` (msgpack tree walk — *this is where "volume.nurec is gzip+msgpack, not a blob" was established*), `rig.py`→`rig4.py` + `layout.py` (rig/pose/world-to-NRE frame derivation), `ckpt_probe/ckpt_modules.py` (the `.ckpt` pickle GLOBAL-opcode probe naming the PPISP code), `cfg_dump.py`, `isp_hunt/isp_identify/isp_decode/isp_quantile.py` + `ppisp_apply.py` + `affine_recheck.py` (the radiometric/CRF recovery, incl. the **quantile-matching** method adopted after paired-pixel conditioning returned a non-monotone curve), plus **11 `report.json`** scoring the quaternion-layout and sky/no-sky arms, `crf_ident.json`, `ppisp_scores.json`, and the scene's structural `x/*.usda` + `metadata.yaml` + `parsed_config.yaml`. |
| `rescued/home_nvidia_root/` | 28 | 77,797 | Thor bring-up and probe scripts: `prep_envs.sh` (**the two-venv build**), `install_trt.sh`, `alpasim_setup*.sh`, `pull_weights/pull_refc*.py`, `verify_val.py` (*"file count is not integrity"* — tar-over-ssh returns 0 on a dropped stream), the latency probes `thor_ksweep/thor_nvfp4/thor_bf16_roll/thor_trt_accuracy.py`, `thor_health.py`, `thor_continuity.py`, `rq_probe_fmt/rq_probe_layers.py`, plus **`thor_profile.py`** (§4) and 3 result JSONs with no repo copy. |
| `rescued/s1_climbout/` | 7 | 35,878 | `refc_s1_climbout_probe.py` (E-S1-0 runner variant), `show.py`/`show2.py` (the readers that render the S1 tables), and the 4 run logs `base/xl/objfam_base/objfam_xl.log`. |
| `rescued/rq_out/` | 18 | 34,737 | Rolling-shutter / renderer stream-E: `rs_regression_check.py` (**bit-wise proof the renderer edits changed nothing for existing callers**), `run_rsE{,2,3}.sh`, `rp_rs_check/report.json`, and 13 raw run logs. |
| `rescued/_s1_backup/` | 2 | 177,187 | ⭐ **`refc_pre.py` — the 115 KB REF-C model source as it stood BEFORE the S1 patch**, plus `refc_train.py.bak`. This is precisely the REF-B-v2 failure mode (an architecture living on one disk). |
| `rescued/parity_verify/` | 2 | 3,845 | `make_prefix.py` (the *contiguous sorted prefix* construction — the one non-full episode set `parity.check_uids(mode="subset")` admits, i.e. **not** a re-selection) and `verify_train_full.json`. |
| `rescued/lambda_findability/` | 2 | 12,291 | `run_lat.sh` + the Thor-side `refc_dump_latents.py` variant (see §4). |
| `rescued/nurec-gsplat/` | 2 | 10,012 | `supervise_t1.sh` (the unattended loop written **because Thor rebooted twice mid-run**) + a `FINDINGS.md` variant. |

## 3. What was deliberately **not** pulled — and why

| item | bytes | verdict |
|---|---:|---|
| `get-pip.py` | 2,230,427 | **DISPOSABLE** — third-party pypa bootstrap installer. Not our work, re-fetchable. |
| `nurec_work/x/rig_trajectories.json` | 2,155,480 | **REGENERABLE** — bulk per-frame rig poses **extracted from `volume.nurec`** by the rescued loader. sha256 recorded in `raw/thor_sha256_inventory.txt`. ⚠️ It *is* the direct input to `rig*.py`/`layout.py`, so those five scripts are **not runnable from the repo alone** — regenerate `x/` from the scene first. |
| `nurec_work/x/sequence_tracks.json` | 531,563 | **REGENERABLE** — same provenance. |
| `_s1_backup/refc.py.bak` | 115,231 | **DUPLICATE** — byte-identical (`sha256 4f51d321…`) to `refc_pre.py`, which was pulled. |
| 104 PNG renders, 46 `.npz`, 4+2 `.pt`, `.ckpt`, `.nurec`, `.msgpack` | **≈3.7 GB** | **OUTPUTS, not sources.** Reproducible by the rescued scripts from assets that live on HF. Their existence and sizes are recorded; the bytes are not repo material. |

**Judged by content, not extension**, per the brief: the 13 `rq_out/logs/*.log` and 4 `s1_climbout/*.log`
files were **kept** — they are the raw measurement transcripts, not noise — while `.png`/`.npz`
outputs of the same runs were dropped.

## 4. ⭐ The drift instrument scans the wrong direction to catch this — `thor_profile.py`

**MEASURED, decisive.** The repo banks a matched pair at
`…/incoming/2026-08-02-thor-deployment-profile/`: `thor_profile.py` (6,073 B) and `thor_profile.json`.

* The JSON's top-level keys include **`"frame": "176x624 hfov 117.0"`**.
* The repo's `thor_profile.py` **never assigns `out['frame']`**.
* Thor's copy (7,060 B, now `rescued/home_nvidia_root/thor_profile.py`) adds exactly that, through
  the trainer's own seam rather than a hand-built lookalike:

```python
from train_flagship_v4 import resolve_v2_frames
_cache_frame, _train_frame = resolve_v2_frames(_ns, cfg, label='thor_profile')
out['frame'] = f'{_train_frame.height}x{_train_frame.width} hfov {_train_frame.hfov_deg:.1f}'
cfg.speed_input = True
cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
```

Those four lines change **which model was profiled** (deployed v5f geometry, speed input on,
`action_dim=3`). ⇒ **The banked script is not the producer of the banked numbers.** This is the
C99/C102/C105 staleness class caught from the box→repo side for the first time — and note that
`pod_git_drift.py` classified this file `NAME_ONLY` ("weak evidence, not drift"), i.e. **it saw the
discrepancy and downgraded it.**

**Nine other same-named pairs** were found (`raw/rescue_manifest.json` → `same_named_repo_file`).
In eight the repo copy is the larger/later one and Thor's is a superseded working draft — including
`lambda_findability/refc_dump_latents.py` (11,629 B vs the repo's 30,089 B), which is the **already-known
C99 2.6× staleness, re-confirmed independently**. `nurec_work/stats.py` is a **name collision only** —
it is a msgpack scene-stats probe, unrelated to `stack/tanitad/{data,replay}/stats.py`.

⚠️ **UNVERIFIED and deliberately not acted on:** for `thor_profile.py` I have *not* overwritten the
banked copy. Direction is established by the JSON key, but which file should be canonical in that
package belongs to whoever owns that deliverable ⇒ filed as **BACKLOG A15**.

## 5. `*.mp4` — checked explicitly, and **no video rescue is needed**

The brief flags that `*.mp4` is gitignored and would need `git add -f`. **MEASURED:**

* **Zero** `.mp4` in any of the eight A11 directories.
* **26** `.mp4` elsewhere under `/home/nvidia` (≈250 MB): `cl_videos{,_hq,_hq2}`, `ol_videos{,_junction}`, `cutin_videos`, `scene2_videos`.
* **18 of 26 are already banked**, byte-for-byte. Verified by md5, not by name:
  `cl_videos_hq2/refc-base_with_objects.mp4` → `8c3892b0e8a3e581f25f002430696c6b`, identical to
  `Evaluation/Videos/alpasim-closedloop-thor-2026-08-03/refc-base_with_objects.mp4`;
  `ol_videos_junction/refc-base_openloop_empty_road.mp4` → `3e0a3fbd9eacb4a5b12bd3efdbdc1681`, identical
  to its `junction-7c72937c/` counterpart. The remaining 16 match by exact byte size.
* **The 8 not banked are `cl_videos/` and `cl_videos_hq/`** — the two earlier, lower-quality render
  passes **superseded by `cl_videos_hq2/`, which is the banked one.** ⇒ No `git add -f` required.
  If the PI wants the render-quality progression preserved, that is a separate ~80 MB decision.

## 5b. Beyond A11 — I looked outside the eight directories, because A11's scope came from the same blind instrument

A11's eight directories are wherever `pod_git_drift.py` happened to find a `.py` or `.sh`. **A second
content sweep** over the AlpaSim result directories it never had reason to name
(`ol_out*`, `cl_out*`, `cutin_out*`, `align_out`, `leadwork`, `lan_e0`, `trt_deploy`, `data`,
`epcache`, `backup`) hashed **187 more files**. MEASURED (`raw/thor_sha256_beyond_a11.txt`):

* ⭐ **Zero `.md`, `.py` or `.sh` stranded** — every analysis panel (`OL_PANEL_*.md`, 6 files, ~86 KB)
  and every script in those dirs is already in the repo by content. **The conclusions are safe.**
* **43 result JSONs are not in the repo by content.** **19 of them — the gates, summaries and scene-attach
  records, 58,293 B — are now banked** in `rescued_beyond_a11/`, 19/19 sha256-verified. Among them
  `align_out/gate_ctrl_offauto/alignment_gate.json` and `ol_out_junction/empty/openloop_summary.json`.
* **24 are NOT pulled**: per-window `rollouts_*.json` / `video_*_openloop.json` dumps, **9,550,490 B**,
  mostly `long_*` re-runs. Their full path + size + sha256 is recorded in
  `raw/beyond_a11_NOT_pulled.json` so the decision is reversible and nothing is lost from the record.
  ⇒ **New backlog item A14** — this is another stream's raw evidence and a 9.6 MB bank is that
  stream's call, not this rescue's.

## 6. Escalate — two instrument changes, one adjudication

1. ⛔ **`pod_git_drift.py:91` `SUFFIXES = (".py", ".sh")` is too narrow.** It missed **46 of the 102**
   stranded files here (**45 %** — 20 `.json`, 16 `.log`, 3 `.usda`, 2 `.yaml`, 2 `.txt`, 2 `.bak`,
   1 `.md`), including every result JSON and every run log. Widening to
   `.md .json .txt .yaml .jsonl .tsv .bak` with a size cap would have caught them.
   *This is a code change to a shared instrument and is NOT made here* — flagged for its owner.
2. ⚠️ **`NAME_ONLY` is not "weak evidence" when the basenames are ours.** It is how `thor_profile.py`
   escaped. A same-named file with different content in a *program-authored* directory should be
   reported at the same severity as `DRIFTED`.
3. ⭐ **Adjudicate `…/2026-08-02-thor-deployment-profile/thor_profile.py`** (§4) — the banked script
   does not produce the banked JSON. Filed as **BACKLOG A15**; the 24 un-pulled dumps of §5b as **A14**.

⚠️ **Index state at hand-off.** Per *stage, never push*, nothing was committed. The index also holds
**168 entries from sibling agents** (`CLAUDE.md`, `AGENT_OPERATING_STANDARD.md`, the `c106-adversarial`
and `latent-linear-ladder` packages). ⇒ Whoever commits must follow the `CLAUDE.md` hygiene rule and
say so in the message — this rescue's own 124 entries are the `2026-08-18-thor-stranded-rescue/` tree
plus `Project Steering/BACKLOG.md`.

**And the reason this accumulated at all:** the nightly checker had been pointed at four dead pods
since 2026-08-15. The rescue closes the backlog, not the hole.

## 7. Provenance

Everything under `rescued/` came from `tanitad-thor-wifi:/home/nvidia/…` on **2026-08-18 ~02:00 UTC**,
pulled read-only while `train_v6_staged.py` (PID 25477, stage S-W, run `v6F-SW-30k`) was training.
`rescued/home_nvidia_root/` mirrors `/home/nvidia/` itself; every other subdirectory mirrors the
same-named directory under `/home/nvidia/`. Per-file source path, byte count and sha256:
`raw/rescue_manifest.json`. The files date from **2026-08-02 → 2026-08-16** and were stranded because
the work that produced them banked its *conclusions* and its *polished* scripts (the
`stack/experiments/nurec-gsplat/` set) while the exploratory chain that got there stayed on the box.

**Trainer, before and after (MEASURED, `…/experiments/v6F-SW-30k/train_log.jsonl`) — it ADVANCED, which
is the check that counts; a live PID would not have been evidence:**

| | step | `step_s` | loss | UTC |
|---|---:|---:|---:|---|
| before the pull | 13,150 | 26.4716 | 2.2923 | 2026-08-18T01:55Z |
| after everything | **13,200** | **26.4707** | 2.0209 | 2026-08-18T02:08Z |

PID 25477 `kill -0` ALIVE at both ends. `step_s` moved **−0.0009 s (−0.003 %)** across the whole
rescue ⇒ **the pull cost the run nothing measurable.** Probes were `ssh -n` with pod-side computation
and opaque `MK…|…|MK` markers, so no filter could match its own command text.
