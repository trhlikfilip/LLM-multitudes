"""
% of (country, trait) pairs that differ in at least 1 context-pair (country preferences).

Analog of utility_outcome_sig.py but using the natural sampling units (20 actual
experimental repeats per context) instead of bootstrap reps. Non-parametric
two-sample test on the rank distribution.

Method (per model, all 5 in MODEL_LABELS):
  1. Per (model, trait, context, repeat), score each country = sum of consistent-winner
     outcomes across 14 pairwise comparisons (+1 win, -1 loss, 0 tie).
  2. Rank countries within each (context, repeat). 20 per-repeat ranks per
     (model, trait, country, context).
  3. For each (model, trait, country, context-pair (A, B)):
       Mann-Whitney U on the 20 ranks in A vs the 20 in B (two-sided).
  4. BH-FDR at alpha = 0.05 over each model's family of 900 = 6 traits x 15
     countries x 10 context-pairs tests.
  5. Report:
       cells_pct = % of 900 cells passing BH
       pairs_pct = % of 90 (country, trait) pairs with at least one BH-significant pair
     and average across the 5 models.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path("...")
CONTEXTS = ["neutral", "news", "reddit", "school", "vlog"]
ALPHA = 0.05
MODEL_FILES = {
    "Llama 3.1 8B Instruct":  "Llama 3.1 8b chat/comparisons_all_llama-3.1-8b-instruct_t1.csv",
    "Llama 3.3 70B Instruct": "Llama 3.3 70b chat/comparisons_all_llama-3.3-70b-instruct_t1.csv",
    "Qwen 3 30B MoE":         "Qwen MoE Chat/comparisons_all_qwen3-30b-a3b-thinking-2507_t1.csv",
    "Mistral Small 4":        "Mistral 4 Small/comparisons_all_mistral-small-2603_t1.csv",
    "Claude Sonnet 4.6":      "Claude Sonnet 4.6/comparisons_all_anthropic.claude-sonnet-4-6_t1.csv",
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


def per_repeat_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (usecase, trait, repeat), score each country = wins - losses based on
    consistent_winner; ties contribute 0. Rank descending so rank 1 = most preferred.
    """
    parts = []
    for col_country, col_other in [("country_a", "country_b"), ("country_b", "country_a")]:
        x = df[["usecase", "trait", "repeat_index", col_country, col_other, "consistent_winner"]].copy()
        x = x.rename(columns={col_country: "country", col_other: "opponent"})
        x["delta"] = np.where(
            x["consistent_winner"] == x["country"], 1,
            np.where((x["consistent_winner"] != "TIE_OR_INCONSISTENT") &
                     (x["consistent_winner"] == x["opponent"]), -1, 0))
        parts.append(x[["usecase", "trait", "repeat_index", "country", "delta"]])
    long = pd.concat(parts, ignore_index=True)
    s = long.groupby(["usecase", "trait", "repeat_index", "country"], as_index=False)["delta"].sum()
    s = s.rename(columns={"delta": "score"})
    s["rank"] = s.groupby(["usecase", "trait", "repeat_index"])["score"] \
                 .rank(method="average", ascending=False)
    return s


def main() -> None:
    all_results = []
    for model_label, fname in MODEL_FILES.items():
        df = pd.read_csv(BASE / fname)
        ranks = per_repeat_ranks(df)
        traits = sorted(ranks["trait"].unique())
        countries = sorted(ranks["country"].unique())

        rows = []
        for trait in traits:
            for country in countries:
                sub = ranks[(ranks["trait"] == trait) & (ranks["country"] == country)]
                ctx = {c: sub[sub["usecase"] == c]["rank"].to_numpy() for c in CONTEXTS}
                for ca, cb in combinations(CONTEXTS, 2):
                    ra, rb = ctx[ca], ctx[cb]
                    if len(ra) == 0 or len(rb) == 0 or np.array_equal(ra, rb):
                        p = 1.0
                    else:
                        try:
                            p = stats.mannwhitneyu(ra, rb, alternative="two-sided").pvalue
                        except ValueError:
                            p = 1.0
                    rows.append({"model": model_label, "trait": trait, "country": country,
                                 "ctx_a": ca, "ctx_b": cb, "p": p})
        res = pd.DataFrame(rows)
        res["sig"] = bh_fdr(res["p"].to_numpy(), ALPHA)
        all_results.append(res)
    res_all = pd.concat(all_results, ignore_index=True)

    print(f"Mann-Whitney U on per-repeat ranks (N=20 per ctx), BH-FDR per model at alpha={ALPHA}")
    print(f"\n{'Model':<24} {'Cells sig (/900)':>20} {'(country, trait) sig (/90)':>30}")
    print("-" * 78)
    cells_pcts, pair_pcts = [], []
    for model_label in MODEL_FILES:
        sub = res_all[res_all["model"] == model_label]
        n_cells = len(sub)
        n_sig_cells = int(sub["sig"].sum())
        pair_any = sub.groupby(["trait", "country"])["sig"].any()
        n_pairs = len(pair_any)
        n_sig = int(pair_any.sum())
        cells_pcts.append(100 * n_sig_cells / n_cells)
        pair_pcts.append(100 * n_sig / n_pairs)
        print(f"{model_label:<24} {n_sig_cells:>5}/{n_cells} ({100*n_sig_cells/n_cells:>5.1f}%)"
              f"   {n_sig:>4}/{n_pairs} ({100*n_sig/n_pairs:>5.1f}%)")
    print("-" * 78)
    print(f"{'AVERAGE':<24} {'':>13} ({np.mean(cells_pcts):>5.1f}%)"
          f"   {'':>10} ({np.mean(pair_pcts):>5.1f}%)")

    out_path = Path("...")
    res_all.to_csv(out_path, index=False)
    res_all.groupby(["model", "trait", "country"])["sig"].any().reset_index() \
        .rename(columns={"sig": "sig_any_ctx_pair"}) \
        .to_csv(out_path.with_name("sig_country_per_repeat_pair_any.csv"), index=False)
    print(f"\nSaved per-cell:  {out_path}")
    print(f"Saved per-pair:  {out_path.with_name('sig_country_per_repeat_pair_any.csv')}")


if __name__ == "__main__":
    main()