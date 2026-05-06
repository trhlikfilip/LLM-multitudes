"""Per-(model, domain) Spearman rho between context rankings."""
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

BASE = Path("...")
CONTEXTS = ["neutral", "news", "reddit", "school", "vlog"]
MODEL_DIRS = {"Llama-8B": "llama 8b", "Llama-70B": "Llama 70b",
              "Qwen": "Qwen", "Mistral": "Mistral 4", "Claude": "Claude Sonnet 4.6"}
DOMAIN_FULL = {"Animal": "Animal welfare and biodiversity",
               "Human Life": "Human life by region", "Self": "Self-preservation",
               "AI": "AI agency and power concentration",
               "Money": "Money anchors", "World": "World events"}


def load_mu(model_dir):
    df = pd.read_csv(BASE / model_dir / "utilities_all_all.csv")
    return {c: np.array([df[(df.usecase == c) & (df.outcome_idx == i)].mu.iloc[0]
                          for i in range(50)]) for c in CONTEXTS}


def per_domain_spearman():
    meta = pd.read_csv(BASE / "Llama 70b" / "utilities_all_all.csv") \
              .drop_duplicates("outcome_idx")
    cat_to_short = {full: short for short, full in DOMAIN_FULL.items()}
    dom = {short: sorted([int(r.outcome_idx) for r in meta.itertuples()
                           if cat_to_short.get(r.category) == short])
           for short in DOMAIN_FULL}
    rows = []
    for label, dir_ in MODEL_DIRS.items():
        mu = load_mu(dir_)
        for d, idxs in dom.items():
            ranks = {c: stats.rankdata(-mu[c][idxs], method="average") for c in CONTEXTS}
            rhos = [stats.spearmanr(ranks[a], ranks[b]).statistic
                    for a, b in combinations(CONTEXTS, 2)]
            rows.append({"model": label, "domain": d,
                         "rho_min": min(rhos), "rho_mean": np.mean(rhos)})
    return pd.DataFrame(rows)