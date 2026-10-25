"""HiCAP -- hierarchical contrastive action priors on a frozen VLM (paper: Paper/HiCAP/hicap.tex).

Modules: vocab (action tree, factored trajectory vocabulary), model (adapter, fusion, heads, towers),
losses (level-wide critics, curriculum), prune (R1-R6 masks, guards, conformal), decode (cascade S0-S4),
backbone (frozen-VLM tap wrapper; GPU/HF only). Nothing here has been trained.
"""
