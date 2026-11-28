<title>LEDGER B8 - tokenizers and discrete representations</title>

# LEDGER_B8 - tokenizers / discrete representations / visual bandwidth

`APPEND-ONLY. Opened 2026-09-19 (LAB-RUN-016): B8 had been scan-only since the charter (2026-08-31) and had no ledger. Transfer question: our latent geometry is a tokenizer question.`

---

## 2026-09-19-01 — One pooled token per view beats 3/6/12 for a WM-conditioned policy, and does NOT contradict our localisation ceiling

`HTML via summariser` · arXiv **2605.07931** (OneWM-VLA) · banked. MetaWorld MT50, H=30: tokens/view **1 → 53.13 %, 3 → 41.86, 6 → 33.85, 12 → 20.54**, 256 = OOM; FPS **4.81 vs 0.13** (1 vs 12). Pooling method matters: static average **72.8 %** vs adaptive (Max/Sum/Learn + learned fusion) **93.3 %** (LIBERO). Fisher ratio: **77 %** of discriminative signal kept. ⚠️ **No error bars**; the token sweep is **confounded with throughput**.
**Our position:** our MEASURED v6 ceiling (4×4 pool = 30°/bin, 2.1–7.8× too coarse) is about **localisation**; theirs is about **prediction for control**. ⇒ **Split the roles**: a pooled world token as the predictor's state, the full trunk for the geometry readout. FS19-6 (matched **steps**, not wall-clock).
Also scanned (09-18/19): Delta tokens `2604.04913` (held), Planning-in-8-Tokens `2603.05438`, token streaming `2605.09886`.
