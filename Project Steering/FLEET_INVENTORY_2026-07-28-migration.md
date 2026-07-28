# Fleet inventory after the 2026-07-28 pod migration — what survived, what is still copying, what cannot be checked

**MEASURED 2026-07-28 17:04–17:10 UTC.** Requested by the PI: *"screen if we lost anything."*

## ⚠️ 0. The finding that governs every line below

**pod2's volume is ACTIVELY RESTORING, so "absent" does NOT mean "lost".**

| observation | value |
|---|---|
| 16:51 UTC | 9.2 GB |
| 17:04:06 UTC | 40.19 GB |
| 17:08:09 UTC | 67.00 GB |
| **measured rate** | **+26.81 GB / 243 s = 110.3 MB/s** |
| 17:09:29 UTC | 71 GB |

⛔ **I called this data loss before measuring the trend, and that was wrong.** A volume mid-migration
looks identical to an emptied one at a single instant. Two errors in that sequence, both mine:
1. **`ls -d` on the cache dirs "proved" they were present** — it tests the *name*, not the contents.
   The dirs were empty (0 files, 512 B).
2. **I then concluded the data was gone**, and proposed a datacenter-migration explanation that pod3
   immediately refuted (same `ca-mtl-1` cluster, 495 GB intact).
⇒ **Nothing below may be read as loss until the restore completes.** Estimated remaining, if pod2's
pre-migration size was ~349 GB: **~42 min at the measured rate.**

---

## 1. pod3 — `1682186f1e9b` · 495 GB · ✅ FULLY INTACT

**The irreplaceable material is here and verified by content, not by directory name:**

| corpus | episodes | skips | size | DONE |
|---|---|---|---|---|
| `physicalai-train-e438721ae894` | **2376** | **24** | 260 G | ✅ |
| `physicalai-val-0c5f7dac3b11` (poses view) | 600 | 0 | 4.6 M | – |
| `physicalai-val-heldout-79d4e3d2d4c6` | 44 | 0 | 4.9 G | ✅ |

**2376 + 24 is the canonical parity set** (key `e438721ae894`, skip-hash `f09e44db`) — so the wide
cache can be rebuilt from the *same* episodes with no re-selection, if it ever needs to be.

- **Checkpoints:** `/workspace/experiments` **30 GB** — the REF-A family, `dynenc-branchB`, plus
  `refc-diffusion-base-v21-30k` (the base under the whole E1 chain).
- **E1 chain artifacts:** `e1c` 2.3 G · `e1e` 3.6 G · `e1f` 1.8 G · `e1b` 1014 M · `e1a_e2a` 4.3 M.

🔴 **pod3 is currently a SINGLE POINT OF FAILURE.** Until pod2 finishes restoring, it is the only host
holding the parity corpus, the REF-C base checkpoint, and the E1 chain. Their summaries are committed
to the repo, but the **checkpoints and corpora are single-disk**, and HF backup is blocked by the
storage 403.

## 2. pod2 — `2e2e2e10613e` · 71 GB and rising · 🟡 RESTORE IN PROGRESS

Arrived so far: `/workspace/data` 59 G · `ckpts` 5.9 G · `archive` 3.1 G · `ckpt_step8500.pt` 3.0 G ·
`TanitAD` checkout 335 M.

**Not yet present — CHECK AGAIN WHEN THE RESTORE COMPLETES, do not treat as lost:**

| expected | status at 17:09 |
|---|---|
| `physicalai-train-e438721ae894-w120-256x640cyl` | dir exists, **0 entries** |
| `physicalai-val-0c5f7dac3b11-w120-256x640cyl` | dir exists, **0 entries** |
| `/workspace/smallval` (incl. arm A's `pw_A_old*.npz`) | dir exists, **0 entries** |
| `/workspace/experiments` | dir exists, **0 entries** |

## 3. eval — `b1d064888689` · 512 B · ✅ FRESH BY DESIGN

Genuinely empty (0 entries). **This is expected** — the PI terminated the broken eval pod and created
this one. Not a loss. A second idle A40 (46 GB) with a healthy disk (361 MB/s real `dd`).

## 4. pod1 — `38.147.83.18:34126` · 🔴 CANNOT BE INVENTORIED

```
root@38.147.83.18: Permission denied (publickey,password)
```
The host **answers** — this is an `authorized_keys` mismatch, not a stopped pod. `~/.ssh/tanitad_pod`
is rejected here while it authenticates on all three other hosts, and `~/.ssh/id_ed25519` does not
exist on the dev box.

🔴 **This is the only real gap in the inventory, and it is the highest-value one:** pod1 held
`flagship-v2corpus-30k` at **step 23,850/30,000 (79.5 %, ~2.9 GPU-days)** with `ckpt.pt` +
`ckpt_step{20000,15000,5000}.pt`. **Whether those survived is unknown.**

**To unblock:** add this dev box's public key to that pod —
`ssh-keygen -y -f ~/.ssh/tanitad_pod` — or supply the key file it expects.

---

## 5. Verdict

- ✅ **Nothing is confirmed lost.**
- ✅ **pod3 is intact**, and holds everything irreplaceable.
- 🟡 **pod2 is restoring at 110 MB/s**; re-inventory when it plateaus.
- 🔴 **pod1 is unverifiable** until key access is fixed — the one place a real loss could hide.

## 6. Recommended actions

1. **Fix pod1 key access** — it is the only unknown, and it guards ~2.9 GPU-days.
2. **Re-run this inventory when pod2's size plateaus** (watch `du -sb /workspace` stop growing).
3. **Do not rebuild the wide cache yet.** I proposed that when I believed it was lost; with the
   restore in flight it would be wasted GPU/disk and would risk racing the incoming data.
4. **Treat pod3 as precious** until pod2 completes: no destructive operations, no re-partitioning.
