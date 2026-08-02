# Pod shutdown 2026-08-02 — what each pod did, what is saved, what is lost

**Trigger: the PI reported $3.61 of remaining credit** and asked to save the valuable data, commit
it, document the work, and stop the pods. Across four A40s that is **under two hours of runway**,
so this is a shutdown under time pressure, not a planned wind-down.

⛔ **I cannot stop the pods.** `runpodctl` is present on the pods but **unauthenticated**
(`API key not found`), and there is no RunPod key in `Keys.txt`. Stopping is a **console action for
the PI**, like pod1's restart.

---

## The two pods the PI named

| console name | our alias | SSH | what it is |
|---|---|---|---|
| **extended_aquamarine_falcon** | **pod3** | `69.30.85.16:22079` | rollout-recovery + the k=20 arm |
| **productive_magenta_clownfish** | **eval** | `69.30.85.106:22073` | the eval/video pod |

⚠️ **Two more pods are also burning the same credit** and were rescued on the same pass:
**pod2** (`69.30.85.123:22091`, running v5e) and **newpod** (`69.30.85.48:22192`, running
v1arch-on-v2bal). ⇒ **$3.61 stops all four**, so treating only the two named ones as at-risk would
have stranded the other two.

---

## What each pod produced

### pod3 — *extended_aquamarine_falcon*

- ⭐ **The rollout-recovery verdict.** RR-20 (`--rollout-k 20`) vs RR-CTL (`--rollout-k 4`), both
  step 31999, same seed, one flag apart. **RR-20 wins**: ADE 0.424 → 0.348, paired ΔCI
  **[0.0613, 0.0906]** (separated), and the four families show it **erased the longitudinal speed
  bias (+0.9397 → −0.0092 m/s)** while **paying 2.2× in curvature**.
- The armed **k=20 from-scratch watcher** (the PI's directive), waiting on a corpus relay.
- 11 pod-only scripts including the `leakcheck_*` family behind C43/C64.

### eval — *productive_magenta_clownfish*

- ⭐ **C64 option A**: v1 **0.393** [0.307, 0.493] vs v2corpus **0.575** [0.429, 0.752], paired
  ΔCI **[−0.221, −0.145]** separated, n=418 over the 19 leak-free episodes.
- ⭐ **The first complete four-family panel** — tactical and strategic populated, not UNAVAILABLE.
- The 12 long v1-vs-v2corpus overlay videos.

### pod2 — v5e (still training)

v5 at **step ~4,350 / 30,000**, 12.0 s/step, stderr 0. Gateless after the held-out gate killed it
twice (C65/C66) and my own kill left it dead ~3.5 days (C67).

### newpod — v1arch-on-v2bal (still training)

The **corpus-only** arm the PI actually asked for (v1 flags, **no `--v2`**), at **step ~3,900**.
Also holds the completed **v2corpus 30k**.

---

## ✅ Saved — committed and pushed

**283 files** in `stack/experiments/pod-rescue-20260802/` (`2c90234`, `6d5f6d3`), plus earlier
`stack/experiments/pod-scripts-20260801/` (11 files, `11a02bf`).

The irreplaceable part is the **result JSONs** — the raw evidence behind every number quoted to the
PI, which cannot be regenerated once the GPUs stop:

| artifact | what it proves |
|---|---|
| `ab_v1-lf19_vs_v2corpus-lf19.json` | the C64 option A paired bootstrap |
| `fourfam_v1-lf19` / `fourfam_v2corpus-lf19` | the four binding families |
| `hier_v1-lf19` / `hier_v2corpus-lf19` | tactical κ + strategic route accuracy |
| `four_families_vs_floors.json`, `ctrv_readjudication.json` | cross-arm floors |
| `rr20.status` / `rrctl.status` / logs | the rollout-recovery verdict |
| every `train_log.jsonl` | the ONLY per-step history of v5, v1arch and v2corpus |
| every `config.json` | ⚠️ **required to LOAD a v2-family checkpoint at all** (`loaders.py` rebuilds the arch from it) |

---

## ⛔ NOT saved — the checkpoints

| checkpoint | size | where |
|---|---|---|
| `pod3:/workspace/rrft/ckpt.pt` (RR-20) | 3.3 GB | pod3 only |
| `pod3:/workspace/rrctl/ckpt.pt` (RR-CTL) | 3.3 GB | pod3 only |
| `pod2:…/flagship-v5-w120-30k/ckpt.pt` | 3.2 GB | pod2 only |
| `newpod:…/flagship-v1arch-v2bal-30k/ckpt.pt` | ~3.4 GB | newpod only |
| `newpod:…/flagship-v2corpus-30k/ckpt.pt` | 3.4 GB | newpod only |

**They cannot be moved in the remaining runway.** The dev-box relay runs ~1 MB/s (~1 h per
checkpoint, ~5 h for all five), pod-to-pod is blocked for these pairs, and HF has been
403-storage-full.

⭐ **STOP preserves them; TERMINATE destroys them.** A stopped RunPod pod keeps its volume. **If
these weights matter, stop — do not terminate.**

⚠️ **v1 and v2corpus survive regardless**: model-only copies were relayed to the eval pod earlier
and the v1 weights also exist on HF (`Sayood/tanitad-flagship-4b-phase0`). **RR-20, RR-CTL and v5
exist in exactly one place each.**

---

## What is lost by stopping now, stated honestly

1. **v5 dies at ~4,350 / 30,000** — 14 % of a 3.6-day run. It auto-resumes from `ckpt.pt` if the
   pod restarts.
2. **v1arch-on-v2bal dies at ~3,900 / 30,000** — same, resumable.
3. ⛔ **The k=20 from-scratch arm never starts.** Its watcher is armed and its corpus relay is
   incomplete (4,275 / 9,000 clips). **This is the experiment the PI explicitly asked for**, and
   with $3.61 there was never runway for a 30k run (~4 GPU-days) anyway. ⇒ the honest statement is
   **not "it was cancelled" but "it was never affordable at this credit level"**.

## Recommendation

**Stop all four, do not terminate.** Then the decision is a budget one: a single pod at ~$0.5/h is
~$12/day, and the queue that actually needs a GPU is short — the k=20 arm, milestone evals, and
four-family panels for REF-A/B/C. Everything else on the current backlog (the paper, the
DATA-vs-ARCH preflight, the sitclf L0 gold-set re-score, the REF-C plan) is **0-GPU** and can
proceed on the PI's own machine.
