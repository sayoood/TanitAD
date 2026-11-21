# trk_verify (G0-TRK instrument; no selection, no PDMS)
verdict: **PASS**
- table sha256 3bda20804e094ea97c3cc5331090ae774a148cd9b73f74449ef55f08d75532dc; tokens 12146/12146; parts 50
- **G0-TRK** d0 == table DAC: 777344/777344 (expected 777344), mismatches 0, pass=True
- corner rebuild vs NAVSIM stored corners: 20 tokens, max abs diff 0.0 m, pass=True
- **G0-TRK-DET** first 100 sorted tokens, compared 100/100: states bit-identical=True, d0=True, corner sha equal=True, pass=True
- **C-TRK-LIVE** share of proposals with max point displacement tracked vs raw > 0.01 m: 1.0000 (777344/777344); max 3.698 m; share d0 != d1: 0.0434 (33766); pass=True
- tracked-corner array sha256 (float64 [12146,64,41,4,2], ego frame, table order): `252ce5cd5b3964b772a2416f607cc976794355d89599e858903282c4f1e5d9dc`
- tracked xyh array sha256: `dea9724161bd1655d7cbc39b82ea90a0c4a4af6a2a0bcf4b5ba7eae817a083ce`
