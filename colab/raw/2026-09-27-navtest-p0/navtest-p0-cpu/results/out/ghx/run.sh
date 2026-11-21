
cd /content/p0
run() { env $2 MPLBACKEND=Agg /content/p0/venv_navsim/bin/python /content/p0/code/navtest/linux_run_v1.py score --label $1   --arm SEAM:/content/p0/bundle/ref/refe_sub200_ep016.npz --tokens /content/p0/bundle/tokens/A1_sub200_tokens.json   --out /content/p0/out/ghx --cache-name navtest --paths /content/p0/out/paths.json --windowspath-shim > /content/p0/out/ghx/$1.driver.log 2>&1; }
run refe_ghx_default ""
run refe_ghx_avx2 "OPENBLAS_CORETYPE=Haswell NPY_DISABLE_CPU_FEATURES='AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR'"
echo done > /content/p0/out/ghx/DONE
