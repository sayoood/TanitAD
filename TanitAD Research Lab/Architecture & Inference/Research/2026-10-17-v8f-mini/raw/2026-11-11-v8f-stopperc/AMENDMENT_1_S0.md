# AMENDMENT 1 to PREREG_V8F_STOPPERC: the `limit` T sees in the stage-0 reads is the causal bucket (Master Mind, 2026-11-11, before any S0-M or S0-P number)

Card: `PREREG_V8F_STOPPERC.md`, sha256 `70c75176e04dfb9284176aa15aa104df1a9204a43090440b3303c65e958c8877` (unchanged; this file amends it). Written after the S0-M smoke refused on its own assert (`A["limit"] == "logged"` read `causal`). No S0-M, S0-P or S0-C number existed at that point: the smoke ran 64 windows, banked no result and printed no statistic.

## The finding (MEASURED)
* All seven STOPBAL first-tick dumps were made with `--limit causal`: BAL-NULL_s0, _s1, _rep, BAL-G_s0, _s1, BAL-L_s0, _s1 (argv files `F:/TanitAD-scratch/stopbal_read/t1/argv_<tag>_m10_causal_heads.txt`, the token after `--limit`).
* In that mode `v8f_t1_eval.build_item` replaces the logged sidecar with `causal_bucket(v, j, CAUSAL_STEPS)`: the speed bin of the maximum finite speed over cache rows [j − 20, j], reading nothing after row j (read tree `stack/scripts/v8f_t1_eval.py:248-250`, `:261-273`, `CAUSAL_STEPS = 20` at `:69`).
* ⇒ In every STOPBAL-line T1-kin read (STOPDEC, STOPCUE on these arms, STOPORC, STOPHARD, STOPTAU, STOPBAL), T's `limit` input is **past speed only**. The ego-future band (`SPEED_BAND.v_hi_ms`, §2) is what T **trained** on, not what these reads fed it. §2's note and Q7 assumed the logged band at inference; for these reads it was not.

## What changes (before any number)
1. **S0-M.** The pass keeps the `off` format. Its comparator is T's **causal-limit** p(STOP_POINT) (the STOPBAL STOPDEC npz), not a "logged-limit" one. The read measures what the causal past-speed bucket adds to T's stop ranking. It still decides nothing in §4's table.
2. **S0-P.** E's `limit` and `limit_ok` are the **causal bucket on both sides**. On eval they are what `build_item(..., limit="causal")` gives (the max |diff| 0 control against the dumps' inputs stays). On the training side they come from the same `causal_bucket` with `CAUSAL_STEPS` 20 over the clip's past speed at the NOW row. The logged sidecar is never a probe input. E−limit drops both. The fraction of windows whose causal bucket is not ok is reported per side.
3. **S0-C.** Unchanged: it uses the dumps' own argv (causal), the same as STOPORC.

## What this does not change
* T trained on the logged ego-future band and is read with a causal bucket. That train/eval difference sits under every STOPBAL-line T1-kin number. It is the programme's existing design for these dumps (the `causal` mode exists so that the eval does not echo the ego's future speed, D-SPEEDBAND-EGO-ECHO-1). This card does not change it.
* §4's outcome table, §5 and §8's other constants.
