# trk_verify (G0-TRK instrument; no selection, no PDMS)
verdict: **PASS [PARTIAL/SMOKE MODE: not the real gate]**
- table sha256 3bda20804e094ea97c3cc5331090ae774a148cd9b73f74449ef55f08d75532dc; tokens 20/12146; parts 1
- **G0-TRK** d0 == table DAC: 1280/1280 (expected 777344), mismatches 0, pass=True
- corner rebuild vs NAVSIM stored corners: 20 tokens, max abs diff 0.0 m, pass=True
- **G0-TRK-DET** first 100 sorted tokens, compared 20/20: states bit-identical=True, d0=True, corner sha equal=True, pass=True
- **C-TRK-LIVE** share of proposals with max point displacement tracked vs raw > 0.01 m: 1.0000 (1280/1280); max 2.895 m; share d0 != d1: 0.0289 (37); pass=True
- tracked-corner array sha256 (float64 [20,64,41,4,2], ego frame, table order): `4f6674dea382d9217db10cb7b9670e439cf35d6a7fa453c7910d6dfd96792ed0`
- tracked xyh array sha256: `3da879e8b2734d75676f62bc035d1ed2c16a170cec0db1d9a1d4eaae12d17c77`
