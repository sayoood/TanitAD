# TELE_FIT_DEFS (fit kit of card PREREG_V8F_PROOF_ADD122, TELEGAP-T0, with note ADD122_N1)

Tier T0. Folder `F:/TanitAD-scratch/v8proof/telegap/fit/`. The tools of card ADD99 (`crop_fit.py`, `crop_dump.py`) are imported unchanged by path (`--kit-fit`, default `/workspace/v8proof/cropread/kit/fit`, fallback `../../cropread/fit` beside the kit). No result or image file was opened; every test runs on tiny synthetic tokens.

## 0. Clauses that could fail by construction on real data (read these first)

1. **Control t4, "the declared lines".** The card names two declared differences (number of windows, token count). The kit changes five module values of `crop_fit`: `N_WIN`, `N_TOK` (only for F), and three that the combination needs: `ARMS` (the arm names M, F, D2 are added so that fit folders and records carry the arm name), `TokenStore` (the combining store) and `build_samples` (wrapped, drops masked samples). Every record lists what changed (`module_values_changed`). If t4 is read as "only windows and token count", it fails on its wording; the lead must accept this list as the declared lines. D2 (two windows, nothing dropped) changes only the last three.
2. **Different sample sets across arms.** M and F drop every sample whose run holds a frame with mask false; D2 and the banked arm D drop nothing. M and F are therefore fitted on fewer samples than D2, and their dumps have NaN gaps (valid false) on the dropped rows. A run of 25 consecutive valid rows is needed by the reader (`closeread_fit.build_rows`, FIT_DEFS of ADD99), so one masked frame removes up to 25 scored rows around it, more than the masked frame. The comparison with D must use the common frame set (card t3); the dump alone does not do that. The drop share is unknown until t1 is logged.
3. **Control t5 (permuted F reads no better than 0.95 of the constant read).** Four windows give a larger head input (260 tokens, same 256-wide state); the kept epoch is the one with the lowest check loss on separately permuted check labels. If a permuted fit memorises, the kept epoch is early; the clause can still fail by chance. Nothing in the kit can change that.
4. **Control t4 range** (D2 seed 0 inside the range of the banked fits widened by 0.15 m): D2 reads the same tokens in the same order with the same seed, so it should reproduce the banked fit up to GPU nondeterminism; the banked fits of arm D were made on the fit and check clips of the pilot list at its own run root. If the pilot lists in `--lists-dir` are not the ones of those fits, t4 fails. A tiny CPU check in the tests shows D2 and crop_fit's own run on arm D give equal weights (atol 1e-6).
5. **Stop rule 10,800 s counts the waiting** for the slot (fewer than 3 GPU processes, more than 28 GB free). With another session on the pod the queue can stop before the last step; that is a stop of the queue, not a result. The wall time is per invocation of the script (a resume starts a new count).
6. **Page cache / memory.** A fit of F reads about 4 windows per frame, five frames per H5 sample; the token files of the two sources (about 17 GB tele plus the banked wide tokens) are memory-mapped and exceed what the 28 GB free gate promises as cache. Reads then come from disk and a fit can run slower than the estimate below. Container memory counted by the gate is the anon line, not the cache.

## 1. Assumptions about the unchanged tools that the tests could not settle

* `crop_fit.run_fit` takes lists from `<root>/lists/` and writes `<root>/fits/`. With the new root, `tele_fit.py` copies `train_final.txt` and `eval141.txt` from `--lists-dir` (default `/workspace/v8proof/cropread/lists`) into `<out-root>/lists/` once (sha256 compared when present). These two copies are new files under the new root; nothing is written under `/workspace/v8proof/cropread`.
* `crop_fit.run_fit` looks up `TokenStore`, `build_samples`, `N_TOK`, `N_WIN`, `ARMS` as module globals at call time (read in the source); the patch relies on it. `crop_dump.head_outputs` reads `CFT.N_TOK`, `CFT.WIDTH`, `CFT.HIST` the same way.
* The real width and token count (1024, 65) were replaced by 8 and 3 in the tests (the code reads `CFT.WIDTH`, `CFT.N_TOK_WIN` at call time); real-size arrays were not built.
* The dump is keyed by the present reader's rows: `cid`, `frame`, `clips[cid]` (from `crop_dump.dump_from_present`). The extra key `tele_dropped` is new; a reader that loads every key of the file reads it as an extra array, a reader that loads named keys ignores it. I did not read `closeread_fit.py` or the reader.
* Seed 2 has no present dump of its own (the pilot made seeds 0 and 1); the queue gives seed 2 the present dump of seed 0, as the queue of card ADD108 did. The dump file is still named `reader_s2_score.npz` and carries `seed` 2.
* `torch.load` of `head.pt` follows `crop_dump.py` (a dict of tensors and plain values).

## 2. Arms (token order inside a frame)

| arm | windows | n_win | n_tok | source |
|---|---|---|---|---|
| D2 | W32, W16 | 2 | 130 | wide file windows 0, 1 |
| M | W16, T8 | 2 | 130 | wide window 1, tele window 0 |
| F | W32, W16, T8, T4 | 4 | 260 | wide 0, 1, tele 0, 1 |

Combined arrays are a lazy view over memory maps (`CombinedTokens`); a slice is materialised only when asked, no combined token file is written. Mask: `<match-dir>/<clip>.npz` key `mask`, bool, length = frame count (refused otherwise). A frame f is dropped when any frame of its run has mask false: H5 run is f-4..f (clipped at 0), H1 run is f. Head: **H5** (the head of the fixed-window arm of card ADD99, `D-H5`). The permuted fit uses crop_fit's own switch; labels are permuted after the drop.

## 3. Outputs

* `<out-root>/fits/<arm>_H5_s<K>[_perm]/head.pt, FIT_RECORD.json` (crop_fit's own) and `TELE_FIT_RECORD.json`: arm, seed, n_win, n_tok, windows, `n_samples` {fit, sel} each with total, kept, dropped, sha256 of `crop_fit.py` as imported and of `crop_dump.py`, `module_values_changed`, `module_values_restored`, input folders, argv, seconds.
* `<out-root>/dump/<arm>_H5[_perm]/reader_s<K>_score.npz` (keys of `crop_dump` plus `tele_dropped`) and `reader_s<K>_meta.json` (adds `n_tele_dropped_rows`).
* Default `--out-root /workspace/v8proof/telegap/fit_out`. Queue logs and markers in `/workspace/v8proof/telegap/` and `fit_queue/`.

## 4. Run commands

Push this folder to the pod as `/workspace/v8proof/telegap/kit_fit/` (the queue finds its tools beside itself; `TG_KIT` overrides).

1. `DRY=1 bash /workspace/v8proof/telegap/kit_fit/pod_telegap_fit_queue.sh` (prints 8 jobs, 16 `CMD:` lines, creates nothing).
2. `nohup setsid bash /workspace/v8proof/telegap/kit_fit/pod_telegap_fit_queue.sh > /dev/null 2>&1 < /dev/null &` (needs `TELEGAP_ENC_DONE.txt`, no `TELEGAP_ENC_STOP.txt`). After a stop: `RESUME=1 ...`.
3. Single fit by hand: `python tele_fit.py --arm M --head H5 --seed 0 --wide-tok-dir /workspace/v8proof/cropread/enc_out/tok/D --tele-tok-dir /workspace/v8proof/telegap/enc_out/tok_T --match-dir /workspace/v8proof/telegap/enc_out/match` then `python tele_dump.py` with the same folders and `--present-file <present dump>`.

Order: D2 s0, F s0 permuted, M s0 s1 s2, F s0 s1 s2 (each fit then dump). Env: `TG_ROOT`, `TG_FIT_STOP_S` (10800), `TG_POLL_S` (60), `TG_OUT`, `TG_PY`, `TG_KIT`, `TG_CROPREAD`, `TG_CROPKIT`, `TG_PRESENT`, `TG_JOIN`, `TG_LISTS`, `DRY`, `RESUME`. A crop_fit record without a tele record, or a dump file without its meta file, stops the queue (look before going on).

## 5. Time and size (ESTIMATED)

`FIT_DEFS.md` of card ADD99 gives no per-fit time. The card's own estimate is about 1 pod-h for six fits and two control fits (F counted as two windows' worth twice over): with 4 two-window fits (D2, M x3) at x and 4 four-window fits (F x4) at 2x, 12x = 60 min, so **x about 5 min per two-window fit (D2, M) and about 10 min per F fit**, plus about 1 min per dump; the eight steps about 1.1 pod-h, inside the 10,800 s stop rule only if the slot waits stay short. Size: `head.pt` about 4 MB (0.9 M parameters), the dump a few MB, records a few KB; the whole queue well under 0.1 GB of new files (the 40 GB cap is not touched). Memory per batch: H5 batch of 256 x 5 frames x 260 tokens x 1024 x 2 bytes is about 0.7 GB for F (half of that for M and D2), plus the float copy on the device.
