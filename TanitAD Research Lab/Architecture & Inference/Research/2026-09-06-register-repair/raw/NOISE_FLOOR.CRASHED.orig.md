# run-to-run noise floor, from the identical-config warm-up stretch

An arm with `--withheld-bank-warmup N` rolls the FIXED bank for steps <= N, so
over that stretch it IS A0's configuration. Divergence there is nondeterminism.

## A1_pred vs A0_fixed — 9 identical-config rows (steps <= 450)
Traceback (most recent call last):
  File "C:\Users\Admin\run_wbank\noise_floor.py", line 106, in <module>
    raise SystemExit(main())
                     ~~~~^^
  File "C:\Users\Admin\run_wbank\noise_floor.py", line 85, in main
    print(f"non-zero differences: {nz}/{tot}"
    ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          f"{'  \u21d2 training is NOT deterministic' if nz else '  \u21d2 bit-identical'}")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python314\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u21d2' in position 29: character maps to <undefined>
