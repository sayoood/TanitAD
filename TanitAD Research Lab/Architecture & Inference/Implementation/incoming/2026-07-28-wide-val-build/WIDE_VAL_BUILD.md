# The 120° VAL split — the last blocker on the v5 run

*Agent: wide-val-build. Date 2026-07-27 (repo dates are ahead of the hub's narrative clock; this
file is dated to the real wall clock). Branch `agent/benchmarks-eval-20260721`, HEAD `7ca00c8`.*
**Status: IN PROGRESS — banked incrementally. See §9 for what is and is not done.**

---

## 0. Headline

*(filled in at the end — see §9 for live status)*

---

## 1. Priority 1 — the TRAIN build's final state

**In progress at time of writing.** Polled by **explicit PID only**
(`ps -o pid= -p 2924952,…,2924959`), never `pgrep -f` / `pkill -f`.

*(final count + FAILED identities go here)*

---

## 2. The val split's membership was NOT re-derived locally — it was already exported and committed

⭐ **This turned out to be an already-closed problem, and the check is worth stating because the
brief's constraint 3 assumed it was open.**

`split_clips()` runs on the **discovered** list, so deriving the split on a host lacking all 3,000
clips is an episode re-selection. It was not re-derived. Two independent artifacts already carry it:

| where | what it says | evidence class |
|---|---|---|
| `pod2:/workspace/wfov/paritysplit/parity_val_clips.txt` + `parity_split_meta.json` | 600 ids; `verified_val_key` **`0c5f7dac3b11`**, `keys_match_parity` **true** | **MEASURED** — written by `parity_split_export.py`, which **refuses to write** unless the host reproduces BOTH canonical keys |
| `stack/tanitad/data/parity_manifest.json` → `physicalai-val-0c5f7dac3b11.clip_membership` | `n_clips` **600**, `clip_id_sha256_sorted` **`0b176d2e…a68e`** | **MEASURED**, committed at HEAD `7ca00c8` |

**The two agree byte-for-byte** (digest `0b176d2e5cb49667d5009366817f948759724e69642e626a47362b93e31da68e`,
read back live from the pod through the shipped `parity.clip_membership_of()`), so
`verify_v2_membership` can run in its **strong `set-diff` mode**: the supplied list must first
reproduce the *committed* digest before it is trusted as the expectation. A self-consistent wrong
pair cannot verify.

⚠️ `ordered_equals_sorted` is **true** for this split, so the set proof is exactly as strong as an
ordered one here — a property of this corpus, not a general one.

⛔ **No skip indices are committed for `physicalai-val-0c5f7dac3b11`** (`skip_count: 0`,
`decode_failures: null`). A shortfall therefore cannot be verified as decode failures, and
`verify_v2_membership` degrades to the count test that must equal **0**. **The build must be
complete at 600 or it must not be registered.**

---

## 3. ⭐ What the preflight found: the reuse probe could never fire, and all 600 val clips are already on pod2

**MEASURED 2026-07-27 on pod2, before anything was built.**

`v2_compressed.py build` has a reuse branch whose comment states its purpose — *"a host that already
carries part of the raw corpus should not re-download it"* — and which had **never once fired**:

```python
mp4 = os.path.join(cam_dir, f"{cid}.mp4")            # <-- no such file has ever existed
```

Every artifact this corpus ships, in the zip and on disk, is named
`<clip_id>.camera_front_wide_120fov.mp4` / `….timestamps.parquet`. The probe spelled the name a
second time and spelled it differently.

| observation | value | how |
|---|---|---|
| `reused_local` across all 8 shards of the TRAIN build | **0** | the shards' own logs |
| mp4 files sitting in `cam_dir` | **761** | `ls` |
| `.timestamps.parquet` in `cam_dir` | **1,295** | `ls` |
| **parity VAL clips with BOTH mp4 and timestamps present** | ⭐ **600 / 600** | filename probe with the real suffix |
| parity TRAIN clips with both present | 161 / 2,400 | same probe |
| egomotion chunk zips already cached | **197 / 197** | `ls` |
| val chunks that would still need a download | **0 of 183** | per-chunk set diff |

⇒ **The val build needs no download at all.** The train build spent its wall-clock on ~374 GB of
egress it partly did not need; the val build would have repeated ~210 GB of it for 600 clips it
already had.

**The fix** (`stack/scripts/v2_compressed.py`, staged): one constant `CAM_NAME` used by BOTH the
chunk template and the reuse probe — the two spellings become one — the probe tries
`<cid>.camera_front_wide_120fov` and falls back to the bare `<cid>`, and `PAI_NO_LOCAL_REUSE=1`
forces the download path as the recovery route if a host's local copy is ever suspect.

⚠️ **`v2_compressed.py` belongs to `2026-07-27-geometry-configurable`** and was last touched by
`2026-07-28-wide-fov-build`. This is a defect fix in another stream's file and **needs their
review** — see §10.

**Test status: `cd stack && pytest -q` → 1,324 passed, 12 skipped, 0 failed** (dev box, after the
change). `tests/test_v2_compressed_real.py` → 2 passed / 5 skipped on the dev box (no torchvision);
its AST guard on the `fr`-rebinding defect class still runs and passes.

⚠️ **This changes the build's IO path, not its selection.** `--only-clips` still pins membership and
`verify_v2_membership` still proves it; reuse only decides where the bytes come from. A reused mp4
is also **never deleted** by the builder (that guard already existed).

---

## 4. Code shipped to pod2 — verified by a real `import`, not a `git log`

`origin/main` does not have the §9 parity code and neither did pod2:

```
pod2:/workspace/wfov/stack_head  ->  parity.py present, but
    has register_v2_geometry_sibling : False
    has assert_v2_splits_disjoint    : False
    has verify_v2_membership         : False
```

⭐ **pod2's `scripts/v2_compressed.py` md5 is `74e5d5065f73217af626f3d2bf12bae5`, byte-identical to
repo HEAD `7ca00c8`** — so the TRAIN build ran exactly the committed code, and my fix is the only
delta on the val path.

A 92,479-byte bundle (`v5ship.tgz`, md5 `e505cb67f3a8cc2570257590384e707f`) was `scp`'d and unpacked
over a copy of `stack_head` as **`pod2:/workspace/wfov/stack_v5`**, leaving the train build's stack
untouched. Verified by running it:

```
tanitad     : /workspace/wfov/stack_v5/tanitad/__init__.py
parity      : /workspace/wfov/stack_v5/tanitad/data/parity.py
  has verify_v2_membership         : True
  has register_v2_geometry_sibling : True
  has assert_v2_splits_disjoint    : True
val clip_membership n_clips : 600
val clip_membership digest  : 0b176d2e…a68e     <- matches the pod's own export
v2_compressed CAM_NAME      : camera_front_wide_120fov
val clips the FIXED probe finds locally: 600 / 600
```

---

## 5. The build

*(command, PIDs, logs, timings — filled in when it runs)*

---

## 6. Geometry verified on REAL DECODED FRAMES

*(filled in)*

---

## 7. Disjointness

*(filled in)*

---

## 8. Registration and the manifest entry as committed

*(filled in)*

---

## 9. Status / what is still open

*(filled in)*

---

## 10. Escalations

*(filled in)*

---

## 11. Deliverable manifest

*(filled in)*
