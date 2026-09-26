"""
Stratified permutation tests for country preferences (RQ1 and RQ3).

For each comparison, keep parseable AB/BA-consistent decisions and sum
Pearson's chi-square statistics across country pairs. Each retained AB/BA
block contributes one decision. Permute condition labels within each pair
using equivalent hypergeometric draws, with p = (exceedances + 1) / (B + 1).

Apply Holm correction separately to the main context comparisons (300 for
five models), wording comparisons (30), and temperature comparisons (150).
Inference is conditional on retained blocks being exchangeable under the null.

Usage:
    python permutation_test.py /path/to/country_comparisons/*.parquet
    python permutation_test.py /path/to/country_comparisons/*.parquet \
        --controls /path/to/country_comparisons_ablations

Also accepts the original comparison CSVs. --controls reads alternative_prompts
and temperature_t{0,0.2,0.4,0.6,0.8} files, in CSV or Parquet format.
"""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

CONTEXTS = ["neutral", "news", "reddit", "school", "vlog"]
TRAITS = ["better_vibes", "beautiful_people", "cool_people",
          "interesting_culture", "democratic", "life_expectancy"]
REQUIRED = ["usecase", "trait", "repeat_index", "country_a", "country_b",
            "parsed_winner_ab", "parsed_winner_ba"]
MODEL_NAMES = {
    "llama-3.1-8b": "Llama-8B-Instruct",
    "llama-3.3-70b": "Llama-70B-Instruct",
    "qwen": "Qwen-3-30B-MoE",
    "mistral": "Mistral Small 4",
    "claude": "Claude Sonnet 4.6",
}
ALPHA = 0.05
PERMUTATIONS = 999999
SEED = 20260917


def load(path, model=None):
    """Read paired decisions and count wins for the first country in each pair."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        columns = [c for c in pq.read_schema(path).names if c in REQUIRED + ["model"]]
        df = pd.read_parquet(path, columns=columns)
    else:
        df = pd.read_csv(path, usecols=lambda c: c in REQUIRED + ["model"])
    if not set(REQUIRED).issubset(df.columns) or df.empty:
        raise ValueError(f"Missing paired-comparison columns or empty input: {path}")
    if "model" in df:
        if df["model"].nunique() != 1 or df["model"].isna().any():
            raise ValueError(f"Expected one model per file: {path}")
        file_model = str(df["model"].iloc[0])
        if model is not None and model != file_model:
            raise ValueError(f"Model label does not match {path}")
        model = file_model
    if model is None:
        model = next((name for token, name in MODEL_NAMES.items()
                      if token in str(path).lower()), None)
    if model is None:
        raise ValueError(f"Cannot identify model from {path}; use --model")

    df["trait"] = df["trait"].replace({"beutyful_people": "beautiful_people"})
    if df[REQUIRED[:5]].isna().any().any():
        raise ValueError(f"Missing comparison identifiers: {path}")
    if set(df["usecase"]) != set(CONTEXTS) or set(df["trait"]) != set(TRAITS):
        raise ValueError(f"Expected five contexts and six traits: {path}")
    df["lo"] = np.minimum(df["country_a"], df["country_b"])
    df["hi"] = np.maximum(df["country_a"], df["country_b"])
    countries = sorted(set(df["lo"]) | set(df["hi"]))
    pairs = list(combinations(countries, 2))
    if len(countries) != 15 or df["lo"].eq(df["hi"]).any():
        raise ValueError(f"Expected comparisons between 15 distinct countries: {path}")
    keys = ["usecase", "trait", "repeat_index"]
    if (len(df.groupby(keys[:2])) != len(CONTEXTS) * len(TRAITS) or
            df.duplicated(keys + ["lo", "hi"]).any() or
            not df.groupby(keys).size().eq(len(pairs)).all() or
            df.groupby(keys[:2])["repeat_index"].nunique().nunique() != 1):
        raise ValueError(f"Incomplete or duplicated pairwise repeats: {path}")

    valid_ab = df["parsed_winner_ab"].eq(df["country_a"]) | df["parsed_winner_ab"].eq(df["country_b"])
    valid_ba = df["parsed_winner_ba"].eq(df["country_a"]) | df["parsed_winner_ba"].eq(df["country_b"])
    kept = df[valid_ab & valid_ba & df["parsed_winner_ab"].eq(df["parsed_winner_ba"])].copy()
    kept["win"] = kept["parsed_winner_ab"].eq(kept["lo"]).astype(int)
    index = pd.MultiIndex.from_tuples(pairs, names=["lo", "hi"])
    groups = {}
    for context in CONTEXTS:
        for trait in TRAITS:
            sub = kept[(kept["usecase"] == context) & (kept["trait"] == trait)]
            groups[context, trait] = sub.groupby(["lo", "hi"])["win"] \
                .agg(["sum", "count"]).reindex(index, fill_value=0).to_numpy(dtype=np.int64)
    return model, groups, pairs


def permutation_test(a, b, permutations, seed):
    """Summed Pearson statistic, calibrated by within-pair label permutations."""
    k1, n1, k2, n2 = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    both = (n1 > 0) & (n2 > 0)
    effect = np.mean(abs(k1[both] / n1[both] - k2[both] / n2[both])) if both.any() else np.nan
    total, wins = n1 + n2, k1 + k2
    usable = both & (wins > 0) & (wins < total)
    # Use the less common pooled outcome to keep seeded draws label-invariant.
    swap = wins > total / 2
    k1, k2 = np.where(swap, n1 - k1, k1), np.where(swap, n2 - k2, k2)
    k1, n1, k2, n2 = [v[usable] for v in [k1, n1, k2, n2]]
    total, wins = n1 + n2, k1 + k2
    if not len(wins):
        return 1.0, 0.0, effect
    pooled = wins / total
    denominator = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    observed = float(((k1 / n1 - k2 / n2) ** 2 / denominator).sum())
    rng = np.random.default_rng(seed)
    exceedances = 0
    for start in range(0, permutations, 2048):
        size = min(2048, permutations - start)
        shuffled = rng.hypergeometric(wins, total - wins, n1, size=(size, len(wins)))
        statistics = ((shuffled / n1 - (wins - shuffled) / n2) ** 2 / denominator).sum(axis=1)
        exceedances += np.count_nonzero(statistics >= observed - max(1e-12, abs(observed) * 1e-12))
    return (exceedances + 1) / (permutations + 1), observed, effect


def compare(meta, a, b, permutations, seed):
    key = json.dumps(meta, sort_keys=True).encode()
    cell_seed = (seed + int.from_bytes(hashlib.sha256(key).digest()[:8], "little")) % (2 ** 63)
    p, statistic, effect = permutation_test(a, b, permutations, cell_seed)
    return {**meta, "p": p, "statistic": statistic,
            "mean_absolute_probability_difference": effect}


def run(paths, controls=None, model=None, permutations=PERMUTATIONS,
        seed=SEED, alpha=ALPHA, out=Path("permutation_results.csv")):
    datasets = [load(path, model) for path in paths]
    if len({name for name, _, _ in datasets}) != len(datasets):
        raise ValueError("Supply one main comparison file per model")
    control_data = []
    if controls:
        baseline = next((item for item in datasets if item[0] == "Llama-70B-Instruct"), None)
        if baseline is None:
            raise ValueError("RQ3 controls require the Llama-70B-Instruct main file")
        names = ["alternative_prompts"] + [f"temperature_t{t}" for t in ["0", "0.2", "0.4", "0.6", "0.8"]]
        for name in names:
            path = next((controls / (name + ext) for ext in [".parquet", ".csv"]
                         if (controls / (name + ext)).is_file()), None)
            if path is None:
                raise FileNotFoundError(f"Missing control: {controls / name}")
            _, groups, pairs = load(path, baseline[0])
            if pairs != baseline[2]:
                raise ValueError(f"Control country pairs differ from baseline: {path}")
            control_data.append((name, groups))

    rows = []
    for name, groups, _ in datasets:
        for trait in TRAITS:
            for ca, cb in combinations(CONTEXTS, 2):
                meta = dict(family="context", model=name, trait=trait, condition_a=ca, condition_b=cb)
                rows.append(compare(meta, groups[ca, trait], groups[cb, trait], permutations, seed))
        print(f"Context comparisons complete: {name}", flush=True)
    for name, groups in control_data:
        family = "wording" if name == "alternative_prompts" else "temperature"
        # Preserve the original seed key for t=0; Holm still includes all temperatures.
        seed_family = "temperature_zero" if name == "temperature_t0" else family
        for trait in TRAITS:
            for context in CONTEXTS:
                meta = dict(family=seed_family, model=baseline[0], trait=trait,
                            condition_a=context + "-main", condition_b=context + "-" + name)
                row = compare(meta, baseline[1][context, trait], groups[context, trait], permutations, seed)
                rows.append({**row, "family": family})
        print(f"Control comparisons complete: {name}", flush=True)

    results = pd.DataFrame(rows)
    for family, indices in results.groupby("family").groups.items():
        results.loc[indices, "p_holm"] = multipletests(results.loc[indices, "p"], method="holm")[1]
        results.loc[indices, "family_size"] = len(indices)
    results["family_size"] = results["family_size"].astype(int)
    results["significant"] = results["p_holm"] < alpha
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    for (family, name), group in results.groupby(["family", "model"]):
        summary = group.groupby("trait")["significant"].agg(["sum", "count"])
        print(f"\n{family}: {name} (Holm family: {int(group.family_size.iloc[0])})")
        print(summary.to_string())
        print(f"Total significant: {int(group.significant.sum())}/{len(group)}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="Main comparison files; use all five models for RQ1")
    ap.add_argument("--controls", type=Path, help="RQ3 wording and temperature file directory")
    ap.add_argument("--model", help="Model label for a CSV without a recognisable model name")
    ap.add_argument("--permutations", type=int, default=PERMUTATIONS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--out", type=Path, default=Path("permutation_results.csv"))
    args = ap.parse_args()
    if args.permutations < 1 or not 0 < args.alpha < 1:
        ap.error("Use a positive permutation count and 0 < alpha < 1")
    if args.model and len(args.files) != 1:
        ap.error("--model applies to one input file")
    run(args.files, args.controls, args.model, args.permutations, args.seed, args.alpha, args.out)
