"""
JS-divergence pipeline (data analysis + figure).

For each trait, build a 5x5 matrix of pairwise Jensen–Shannon divergence
between the 5 contexts. Word-count distributions are built from the model
essays, summed across the 5 models, and renormalised over each trait's
union vocabulary.

JSD is computed as scipy.spatial.distance.jensenshannon(...) ** 2 with
base=2, so values are bounded in [0, 1]: 0 = identical distributions,
1 = maximally divergent.

"""

import csv
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

csv.field_size_limit(sys.maxsize)
mpl.rcParams.update({
    "savefig.dpi": "figure",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------- Paths ----------------

BASE       = Path("...")
COUNTS_DIR = Path("...")
OUT        = Path("...")
COUNTS_DIR.mkdir(exist_ok=True, parents=True)
OUT.mkdir(exist_ok=True, parents=True)

MODEL_FILES = {
    "Claude Sonnet 4.6": BASE / "Claude Sonnet 4.6"  / "comparisons_all_anthropic.claude-sonnet-4-6_t1.csv",
    "Llama 3.1 8b":      BASE / "Llama 3.1 8b chat"  / "comparisons_all_llama-3.1-8b-instruct_t1.csv",
    "Llama 3.3 70b":     BASE / "Llama 3.3 70b chat" / "comparisons_all_llama-3.3-70b-instruct_t1.csv",
    "Mistral 4 Small":   BASE / "Mistral 4 Small"    / "comparisons_all_mistral-small-2603_t1.csv",
    "Qwen MoE":          BASE / "Qwen MoE Chat"      / "comparisons_all_qwen3-30b-a3b-thinking-2507_t1.csv",
}

# ---------------- Tokenisation ----------------

STOPWORDS = set(ENGLISH_STOP_WORDS)
MIN_LEN   = 3
TOKEN_RE  = re.compile(r"[a-z]+")

# ---------------- Layout ----------------

A4_LANDSCAPE = (11.69, 8.27)
DPI = 450

TRAITS = ["cool_people", "beutyful_people", "better_vibes",
          "interesting_culture", "democratic", "life_expectancy"]
TRAIT_LABELS = {
    "cool_people":         "cool people",
    "beutyful_people":     "beautiful people",
    "better_vibes":        "better vibes",
    "interesting_culture": "interesting culture",
    "democratic":          "democratic",
    "life_expectancy":     "life expectancy",
}
CONTEXTS = ["neutral", "news", "reddit", "school", "vlog"]


# ====================================================================
# STAGE 1: per-model word counts (cached)
# ====================================================================

def count_words_for_model(args):
    """Stream one model's CSV. Tokens: lowercase [a-z]+, len>=3, no stopwords."""
    model_name, path = args
    counters = defaultdict(Counter)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["trait"], row["usecase"])
            text = (row.get("raw_ab") or "") + " " + (row.get("raw_ba") or "")
            tokens = (t for t in TOKEN_RE.findall(text.lower())
                      if len(t) >= MIN_LEN and t not in STOPWORDS)
            counters[key].update(tokens)
    return model_name, dict(counters)


def load_or_build_counts():
    cache = COUNTS_DIR / "per_model_counts.pkl"
    if cache.exists():
        print(f"Loading cached counts from {cache.name}", flush=True)
        with open(cache, "rb") as f:
            return pickle.load(f)
    print(f"Streaming {len(MODEL_FILES)} model CSVs in parallel...", flush=True)
    model_counts = {}
    with ProcessPoolExecutor(max_workers=len(MODEL_FILES)) as ex:
        futs = {ex.submit(count_words_for_model, (n, p)): n for n, p in MODEL_FILES.items()}
        for fut in as_completed(futs):
            name, counts = fut.result()
            model_counts[name] = counts
            print(f"  done: {name}", flush=True)
    with open(cache, "wb") as f:
        pickle.dump(model_counts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached -> {cache.name}", flush=True)
    return model_counts


# ====================================================================
# STAGE 2: pairwise JS divergence per trait
# ====================================================================

def build_jsd(model_counts):
    cells    = sorted({cell for cnt in model_counts.values() for cell in cnt})
    traits   = sorted({t for t, _ in cells})
    contexts = sorted({c for _, c in cells})

    # Sum across models per (trait, context)
    totals = {}
    for trait in traits:
        for ctx in contexts:
            agg = Counter()
            for cells_for_model in model_counts.values():
                agg.update(cells_for_model.get((trait, ctx), {}))
            totals[(trait, ctx)] = agg

    out = {}
    for trait in traits:
        # Union vocab across the 5 contexts of this trait
        vocab = set()
        for ctx in contexts:
            vocab.update(totals[(trait, ctx)].keys())
        vocab = sorted(vocab)
        idx = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)

        # Probability distribution per context over the union vocab
        vecs = {}
        for ctx in contexts:
            v = np.zeros(V, dtype=np.float64)
            for w, c in totals[(trait, ctx)].items():
                v[idx[w]] = c
            s = v.sum()
            vecs[ctx] = v / s if s > 0 else v

        matrix = {}
        for a in contexts:
            matrix[a] = {}
            for b in contexts:
                # scipy returns sqrt(JSD) — square to get JSD in [0, 1] (base 2)
                d = jensenshannon(vecs[a], vecs[b], base=2)
                matrix[a][b] = float(d ** 2) if not np.isnan(d) else 0.0
        out[trait] = matrix
        print(f"  {trait:22s} | vocab={V:6d}")

    out["_meta"] = {
        "metric": "Jensen-Shannon divergence (squared, base 2) on probability "
                  "distributions over each trait's union vocabulary",
        "range":  "[0, 1]; 0 = identical, 1 = maximally divergent",
    }
    json_path = OUT / "js_divergence.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {json_path}")
    return out


# ====================================================================
# STAGE 3: figure
# ====================================================================

def _text_color(value, cmap_name, vmin, vmax):
    if value is None or np.isnan(value):
        return "black"
    norm = max(0.0, min(1.0, (value - vmin) / max(vmax - vmin, 1e-9)))
    rgba = mpl.colormaps[cmap_name](norm)
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "black" if lum > 0.55 else "white"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=DPI, facecolor="white",
                bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUT / f"{name}.pdf", facecolor="white",
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Wrote {name}.png + .pdf")


def fig_jsd(d):
    fig, axes = plt.subplots(2, 3, figsize=A4_LANDSCAPE, squeeze=False)

    # Shared colour scale across the 6 panels, set by the largest off-diagonal value
    off_diag = [d[trait][a][b] for trait in TRAITS
                for a in CONTEXTS for b in CONTEXTS if a != b]
    vmax = max(off_diag)

    cmap_obj = mpl.colormaps["YlOrRd"].copy()
    cmap_obj.set_bad("#e8e8e8")  # masked diagonal cells render grey

    for idx, trait in enumerate(TRAITS):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        M = np.array([[d[trait][a][b] for b in CONTEXTS] for a in CONTEXTS])
        Mplot = M.copy()
        np.fill_diagonal(Mplot, np.nan)
        ma = np.ma.masked_invalid(Mplot)
        ax.imshow(ma, cmap=cmap_obj, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(CONTEXTS)))
        ax.set_yticks(range(len(CONTEXTS)))
        ax.set_xticklabels(CONTEXTS, fontsize=9)
        ax.set_yticklabels(CONTEXTS, fontsize=9)
        ax.tick_params(top=False, bottom=False, left=False, right=False)
        ax.set_title(TRAIT_LABELS[trait], fontsize=11, pad=6)

        for i in range(len(CONTEXTS)):
            for j in range(len(CONTEXTS)):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=10, color="#888")
                else:
                    v = M[i, j]
                    color = _text_color(v, "YlOrRd", 0, vmax)
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=9, color=color)

    fig.subplots_adjust(left=0.05, right=0.97, top=0.96, bottom=0.06,
                        wspace=0.28, hspace=0.40)
    save(fig, "fig_jsd")


# ====================================================================
# Main
# ====================================================================

def main():
    json_path = OUT / "js_divergence.json"
    if json_path.exists():
        with open(json_path) as f:
            d = json.load(f)
        print(f"Loaded cached {json_path}")
    else:
        model_counts = load_or_build_counts()
        d = build_jsd(model_counts)
    fig_jsd(d)


if __name__ == "__main__":
    main()
