# FINISH REPORT ‚Äî produced automatically when the run wrote summary.json

Run: `D:/Projects/TanitAD/taniteval/results/bench/navsim_v2/navhard_two_stage/20260921T122714Z-navsim_v2-refcv4b_b1_v72_40k-859e25`

## The pre-registered bar (BAR-W7-1), as a literal comparison

```
run 20260921T122714Z-navsim_v2-refcv4b_b1_v72_40k-859e25  split navhard_two_stage  protocol EPDMS_v2_navhard_two_stage
tier T1-family ∑ loop s1 OPEN (PI ruling 2026-09-02) ∑ s2 UNRULED (2026-09-02 vocabulary) ∑ closed_loop False

arm    official two-stage EPDMS           S2-EPDMS-u    s1 mean    s2 mean
CV     0.114816                             0.338914   0.289588   0.329356
STOP   0.298532                             0.482827   0.563162   0.470232
ECHO   0.142938                             0.341280   0.386638   0.326801
A1     UNAVAILABLE                          0.384136   0.289588   0.374211
```

## Where A1 stands vs the floors (decomposition)

```
{
 "arm": "A1",
 "zero_attribution": {
  "stage_1": {
   "n_zero_score": 289,
   "n": 450
  },
  "stage_2": {
   "n_zero_score": 2493,
   "n": 5462
  }
 },
 "vs": {
  "STOP": -0.096021,
  "CV": 0.044855,
  "ECHO": 0.047409
 }
}
```

## Criteria gates (0 violations required)

### CV
```
rc=0
=== CV.json ===
tier: T1  (driving-performance tier)

```

### STOP
```
rc=0
=== STOP.json ===
tier: T1  (driving-performance tier)

```

### ECHO
```
rc=0
=== ECHO.json ===
tier: T1  (driving-performance tier)

```

### A1
```
rc=0
=== A1.json ===
tier: T1  (driving-performance tier)

```

## W5 report + failure gallery

index: `D:/Projects/TanitAD/taniteval/results/bench/navsim_v2/navhard_two_stage/20260921T122714Z-navsim_v2-refcv4b_b1_v72_40k-859e25/report/index.html` ‚Äî PRESENT

‚ö†Ô∏è RESULT.md has NOT been rewritten by this finisher ‚Äî the numbers, their evidence classes
and the verdict prose are a human/agent judgement, not a template fill. This file is the
input to that write-up, and every number in it comes from summary.json.
