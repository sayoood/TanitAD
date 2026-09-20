<title>A7 gate unreachable</title>

# A7 is no longer GATED — it is STUCK: the PI's processes have exited and the residual GPU floor sits above the gate

`Architecture & Inference · 2026-09-20 · Master Mind · MEASURED, read-only, nothing killed and nothing changed`
`Instrument: 20 samples of nvidia-smi memory.used over ~60 s → raw/gate_floor.json. Gate: GPU ≤ 2,500 MiB AND host free ≥ 8 GB.`

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **1** | ⛔ **The reconstruction-studio processes are GONE.** A read-only process query for `*reconstruction*` / `*studio*` returns nothing. The blocker this session has reported all night no longer exists. | MEASURED |
| **2** | ⛔ **And the gate is now UNREACHABLE.** 20 samples over ~60 s: **min 3,039 · median 3,048 · max 3,097 MiB**, **0 of 20** at or below the 2,500 MiB gate. Spread 58 MiB ⇒ a **stable floor**, not a fluctuating load. | MEASURED |
| **3** | ⇒ **A7, P0 and P0b will wait their full 72 h ceiling and never launch**, printing *"A7 arms not VALID yet"* the whole time. ⭐ This is precisely the class the TrainingFlyWheel named when it fixed the self-matching probe: **a wait that never ends looks exactly like a wait that is working.** It was right about the mechanism and wrong about which wait would hit it. | inference from 1+2 |
| **4** | ⚠️ **What still holds the card is not identifiable from this account.** `nvidia-smi --query-compute-apps` lists desktop processes (explorer, shell hosts, NVIDIA Overlay, EdgeWebView) and **two entries reading `[Insufficient Permissions]`** (PIDs 1848, 4508). Per-process GPU memory reads `[N/A]` under Windows WDDM, so the residual cannot be attributed from here. | MEASURED, with its limit stated |

## 1 · The arithmetic the decision needs

The card is **8,188 MiB**. A floor of **~3,050 MiB** leaves **~5,140 MiB** of headroom.

From the pod request's P2 (`a510cff`, MEASURED `8b1f1db`): `resnet101` at the ruled geometry fits in **2,887 MB @ batch 1** and **3,890 MB @ batch 2**, all 22 heads live, with `--trunk-chunk-ckpt 1 --trunk-frozen-bn`.

⇒ **An arm needing ~3.9 GB fits inside 5.1 GB of headroom.** The 2,500 MiB gate was set when the card was genuinely occupied by the PI's work; against a 3,050 MiB desktop floor it now excludes runs that would fit comfortably.

## 2 · ⛔ What I did NOT do, and why

* **I did not lower the gate.** It is forbidden three times over in the standing instructions, and the reason is sound: the gate exists so a dev-box arm never competes with the PI's own work. A threshold that I move on my own authority is not a guard.
* **I did not kill or inspect anything by pattern.** The process query was read-only and by name; nothing was terminated. The two `[Insufficient Permissions]` PIDs need the PI's own session to identify.
* **I did not restart or re-point the runner.** It is correctly implemented and correctly waiting; the fault is in the threshold's relationship to the box, not in the runner.

## 3 · The decision, which is the PI's

Three options, with what each costs:

1. **Free the card** — identify and stop whatever holds the residual ~3,050 MiB (the two permission-blocked PIDs first). Costs nothing if they are stale; the gate then opens by itself and A7 → P0 → P0b run unattended as designed.
2. **Raise the gate** to a threshold that reflects the box's real floor — e.g. **3,200 MiB**, which still leaves 4,988 MiB, more than the 3,890 MB a batch-2 `resnet101` arm needs. ⛔ Only the PI may authorise this, and the reason to state it in writing is that the *original* number was not arbitrary.
3. **Accept the wait** — legitimate if the residual is expected to clear on its own, but it should be an explicit choice rather than a default, because the runner's 72 h ceiling will expire quietly.

⚠️ **Until one is chosen, E13 cannot close and the perception panel cannot start** — and every "A7 is correctly gated" line in tonight's reports, including mine, was describing a wait that had already stopped being a wait for the reason given.
