"""
% of (model, outcome) pairs that differ in at least 1 context-pair (utility experiment).

Method (per model, all 5 in MODEL_LABELS):
  1. Load 1000-bootstrap Thurstonian mu fits per context. Convert each
     bootstrap sample's mu vector into a full ranking of the 50 outcomes
     (rank 1 = highest mu).
  2. For each (outcome, context-pair (A, B)):
       D_i = rank_A[i] - rank_B[i]   for i = 1..1000
       p   = 2 * min( P(D <= 0), P(D >= 0) )    # two-sided bootstrap p-value
  3. BH-FDR at alpha = 0.05 over each model's family of 500 = 50 outcomes
     x 10 context-pairs tests.
  4. Report:
       cells_pct    = % of 500 cells passing BH
       outcomes_pct = % of 50 outcomes with at least one BH-significant pair
     and average across the 5 models.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path("...")
BOOTSTRAP_CACHE = BASE / "bootstrap_mu_cache_N1000.npz"
CONTEXTS = ["neutral", "news", "reddit", "school", "vlog"]
ALPHA = 0.05
MODEL_LABELS = {
    "8b":      "Llama 3.1 8B Instruct",
    "70b":     "Llama 3.3 70B Instruct",
    "qwen":    "Qwen 3 30B MoE",
    "mistral": "Mistral Small 4",
    "claude":  "Claude Sonnet 4.6",
}


def bh_fdr(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg FDR control. Returns boolean mask of rejections."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p.astype(bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool)
    k = int(np.max(np.where(passed)[0]))
    return p <= ranked[k]


def load_bootstrap_ranks() -> dict:
    """dict[model_short][context] -> array (n_boot, n_outcomes), rank 1 = highest mu."""
    raw = np.load(BOOTSTRAP_CACHE, allow_pickle=True)
    out = {short: {} for short in MODEL_LABELS}
    for key in raw.files:
        short, ctx = key.split("__")
        mu = raw[key]
        ranks = np.empty_like(mu)
        for i in range(mu.shape[0]):
            ranks[i] = stats.rankdata(-mu[i], method="average")
        out[short][ctx] = ranks
    return out


def bootstrap_pvalue(ra: np.ndarray, rb: np.ndarray) -> float:
    """Two-sided empirical bootstrap p-value via paired-by-index difference distribution."""
    diffs = ra - rb
    n = diffs.size
    if n == 0:
        return 1.0
    p_le = (diffs <= 0).sum() / n
    p_ge = (diffs >= 0).sum() / n
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def main() -> None:
    ranks_by_model = load_bootstrap_ranks()

    rows = []
    for short, ranks_by_ctx in ranks_by_model.items():
        n_outcomes = ranks_by_ctx[CONTEXTS[0]].shape[1]
        for outcome in range(n_outcomes):
            for ca, cb in combinations(CONTEXTS, 2):
                p = bootstrap_pvalue(ranks_by_ctx[ca][:, outcome],
                                     ranks_by_ctx[cb][:, outcome])
                rows.append({"model_short": short,
                             "model": MODEL_LABELS[short],
                             "outcome": outcome,
                             "ctx_a": ca, "ctx_b": cb, "p": p})
    res = pd.DataFrame(rows)

    res["sig"] = False
    for short in MODEL_LABELS:
        idx = res.index[res["model_short"] == short]
        res.loc[idx, "sig"] = bh_fdr(res.loc[idx, "p"].to_numpy(), ALPHA)

    print(f"Bootstrap p-value (paired-by-index, two-sided), BH-FDR per model at alpha={ALPHA}")
    print(f"\n{'Model':<24} {'Cells sig (/500)':>20} {'Outcomes sig (/50)':>22}")
    print("-" * 70)
    cells_pcts, out_pcts = [], []
    for short in MODEL_LABELS:
        sub = res[res["model_short"] == short]
        n_cells = len(sub)
        n_sig_cells = int(sub["sig"].sum())
        outcome_any = sub.groupby("outcome")["sig"].any()
        n_pairs = len(outcome_any)
        n_sig = int(outcome_any.sum())
        cells_pcts.append(100 * n_sig_cells / n_cells)
        out_pcts.append(100 * n_sig / n_pairs)
        print(f"{MODEL_LABELS[short]:<24} {n_sig_cells:>5}/{n_cells} ({100*n_sig_cells/n_cells:>5.1f}%)"
              f"   {n_sig:>3}/{n_pairs} ({100*n_sig/n_pairs:>5.1f}%)")
    print("-" * 70)
    print(f"{'AVERAGE':<24} {'':>13} ({np.mean(cells_pcts):>5.1f}%)"
          f"   {'':>9} ({np.mean(out_pcts):>5.1f}%)")

    out_path = BASE.parent / "sig_utility_bootstrap_results.csv"
    res.to_csv(out_path, index=False)
    res.groupby(["model_short", "model", "outcome"])["sig"].any().reset_index() \
        .rename(columns={"sig": "sig_any_ctx_pair"}) \
        .to_csv(out_path.with_name("sig_utility_bootstrap_outcome_any.csv"), index=False)
    print(f"\nSaved per-cell:  {out_path}")
    print(f"Saved per-outcome: {out_path.with_name('sig_utility_bootstrap_outcome_any.csv')}")


if __name__ == "__main__":
    main()