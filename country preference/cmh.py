"""
Cochran-Mantel-Haenszel test (true 2x2) for context-pair preference shifts.

For every (trait, context-pair):
  - Filter TIE_OR_INCONSISTENT decisions (clean binary outcome).
  - For each country pair (stratum), build the 2x2 contingency table:
        rows = context (c1, c2)
        cols = winner (first_country, second_country)
  - Run stratified Mantel-Haenszel test of common-odds-ratio = 1
    (statsmodels.stats.contingency_tables.StratifiedTable.test_null_odds,
     two-sided, no continuity correction — the standard MH formulation).

Methodological choices and their justification:
  1. Tie filtering: vanilla CMH requires a binary outcome.  Collapsing ties
     into one of the two classes introduces an encoding-dependent bias
     (the rejection count differs depending on which class the tie is
     merged into).  Filtering is the only choice that preserves the test's
     standard 2x2 assumptions.
  2. Stratification by country pair: the natural unit of the design, makes
     within-item correlation explicit and respected by the test.
  3. No continuity correction: the standard Mantel-Haenszel test does not
     use continuity correction (Mantel & Haenszel 1959); statsmodels'
     test_null_odds matches this convention.
  4. Two-sided: tests common odds ratio != 1.  One-sided variants exist
     but two-sided is the standard reporting convention.

Usage:
    python cmh_2x2.py <comparisons.csv>

Outputs a (n_traits x n_context_pairs) grid of p-values and the total
count of cells with p<alpha (default x/60 for the standard 6 traits x
10 context-pairs setup).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import StratifiedTable

REQUIRED = ["usecase", "trait", "consistent_winner", "country_a", "country_b"]
PAPER_CODES = {"neutral": "N", "news": "W", "reddit": "R",
               "school": "S", "vlog": "V"}
ALPHA = 0.05


def load(csv_path):
    df = pd.read_csv(csv_path, usecols=REQUIRED)
    df["country_pair"] = df.apply(
        lambda r: tuple(sorted([r["country_a"], r["country_b"]])), axis=1)
    df["first_country"]  = df["country_pair"].apply(lambda p: p[0])
    df["second_country"] = df["country_pair"].apply(lambda p: p[1])
    df["is_tie"] = (df["consistent_winner"] == "TIE_OR_INCONSISTENT")
    df["first_wins"] = (df["consistent_winner"] == df["first_country"]).astype(np.int8)
    return df


def cmh_2x2(df, trait, c1, c2):
    """2x2 CMH stratified by country pair, ties filtered."""
    sub = df[(df["trait"] == trait)
             & (df["usecase"].isin([c1, c2]))
             & (~df["is_tie"])]
    tables = []
    for cp, g in sub.groupby("country_pair"):
        g1 = g[g["usecase"] == c1]
        g2 = g[g["usecase"] == c2]
        n1, n2 = len(g1), len(g2)
        if n1 == 0 or n2 == 0:
            continue
        k1 = int(g1["first_wins"].sum())
        k2 = int(g2["first_wins"].sum())
        # 2x2: rows = context (c1, c2), cols = (first wins, second wins)
        tables.append([[k1, n1 - k1], [k2, n2 - k2]])
    if not tables:
        return 1.0, 0
    arr = np.array(tables).transpose(1, 2, 0)        # statsmodels wants (2, 2, K)
    try:
        st = StratifiedTable(arr)
        # standard Mantel-Haenszel test, two-sided, no continuity correction
        return float(st.test_null_odds().pvalue), len(tables)
    except Exception:
        return 1.0, len(tables)


def fmt_p(p):
    if p < 0.005: return "<.005"
    s = f"{p:.2f}"
    return s.lstrip("0") if s.startswith("0.") else s


def run(csv_path, alpha=ALPHA):
    df = load(csv_path)
    contexts = sorted(df["usecase"].unique())
    traits   = sorted(df["trait"].unique())
    code_for = {c: PAPER_CODES[c] if c in PAPER_CODES else c[0].upper()
                for c in contexts}
    pairs = [(a, b) for i, a in enumerate(contexts) for b in contexts[i + 1:]]
    pair_codes = [code_for[a] + code_for[b] for a, b in pairs]

    print(f"File:   {csv_path}")
    print(f"Test:   Cochran-Mantel-Haenszel (2x2, ties filtered, two-sided, no continuity correction)")
    print(f"Strata: country pair")
    print(f"alpha = {alpha}\n")

    header = f"{'trait':<22}" + " ".join(f"{c:>6}" for c in pair_codes) + "   sig"
    print(header); print("-" * len(header))

    sig_total = 0
    n_strata_min, n_strata_max = 999, 0
    for t in traits:
        line = f"{t:<22}"; sig_row = 0
        for c1, c2 in pairs:
            p, n_strata = cmh_2x2(df, t, c1, c2)
            n_strata_min = min(n_strata_min, n_strata)
            n_strata_max = max(n_strata_max, n_strata)
            line += f" {fmt_p(p):>6}"
            if p < alpha: sig_row += 1
        line += f"   {sig_row}/{len(pairs)}"
        sig_total += sig_row
        print(line)
    print("-" * len(header))
    print(f"Total significant (p<{alpha}): {sig_total}/{len(traits)*len(pairs)}")
    print(f"Strata per cell: {n_strata_min}–{n_strata_max} country pairs (105 max)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()
    run(args.csv, alpha=args.alpha)