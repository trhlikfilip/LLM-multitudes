"""
Verdict pipeline (data analysis + figure) — standalone.

For each (model, trait, context) it computes:
  * verdict-marker rate  = (# verdict-lexicon tokens) / (# alphabetic tokens) × 100
  * mean first-verdict position  = mean over rows of (start_char_of_first_match / len(text)),
                                   computed per model then averaged across models

Output:
  context_metrics/verdict_metrics.json
  context_metrics/fig_verdict.png + .pdf
"""

import csv
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(sys.maxsize)
mpl.rcParams.update({
    "savefig.dpi": "figure",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------- Paths ----------------

BASE = Path("...")
OUT  = Path("...")
OUT.mkdir(exist_ok=True, parents=True)

MODEL_FILES = {
    "Claude Sonnet 4.6": BASE / "Claude Sonnet 4.6"   / "comparisons_all_anthropic.claude-sonnet-4-6_t1.csv",
    "Llama 3.1 8b":      BASE / "Llama 3.1 8b chat"   / "comparisons_all_llama-3.1-8b-instruct_t1.csv",
    "Llama 3.3 70b":     BASE / "Llama 3.3 70b chat"  / "comparisons_all_llama-3.3-70b-instruct_t1.csv",
    "Mistral 4 Small":   BASE / "Mistral 4 Small"     / "comparisons_all_mistral-small-2603_t1.csv",
    "Qwen MoE":          BASE / "Qwen MoE Chat"       / "comparisons_all_qwen3-30b-a3b-thinking-2507_t1.csv",
}

# ---------------- Lexicon ----------------

# D1: Verdict / outcome words
VERDICT = {
    # explicit verdict
    "winner", "wins", "won", "winning",
    "verdict",
    "victory", "victorious",
    # comparative outcomes
    "edges", "outperforms", "beats", "trumps", "surpasses",
    "exceeds", "outshines", "outranks", "outdoes",
    # preference
    "preferable", "preferred", "preferring", "preference",
    "favors", "favored", "favoring", "favor",
    # conclusion
    "concludes", "concluded", "concluding", "conclusion",
}

VERDICT_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in VERDICT) + r")\b")
TOKEN_RE   = re.compile(r"[a-z]+")  # denominator: lowercase alphabetic tokens

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
# DATA ANALYSIS
# ====================================================================

def process_model(args):
    """Single-pass scan over one model's CSV.

    Returns 4 dicts keyed by 'trait|context':
        n_total          - alphabetic-token count (denominator)
        verdict_hits     - # tokens in VERDICT lexicon
        verdict_pos_sum  - Σ over rows of (first-match-start / len(text))
        verdict_pos_n    - # rows where any verdict marker matched
    """
    model_name, path = args
    n_total         = defaultdict(int)
    verdict_hits    = defaultdict(int)
    verdict_pos_sum = defaultdict(float)
    verdict_pos_n   = defaultdict(int)

    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["trait"], row["usecase"])
            for side in ("raw_ab", "raw_ba"):
                text = row.get(side) or ""
                if not text:
                    continue
                low = text.lower()

                lc_tokens = TOKEN_RE.findall(low)
                tlen = len(lc_tokens)
                n_total[key] += tlen
                if tlen == 0:
                    continue

                # verdict-token count (lexicon membership in lowercase tokens)
                for t in lc_tokens:
                    if t in VERDICT:
                        verdict_hits[key] += 1

                # position of first verdict marker (0 = essay start, 1 = essay end)
                m = VERDICT_RE.search(low)
                if m:
                    verdict_pos_sum[key] += m.start() / max(len(low), 1)
                    verdict_pos_n[key]   += 1

    def s(d): return {f"{t}|{c}": v for (t, c), v in d.items()}
    return model_name, {
        "n_total":         s(n_total),
        "verdict_hits":    s(verdict_hits),
        "verdict_pos_sum": s(verdict_pos_sum),
        "verdict_pos_n":   s(verdict_pos_n),
    }


def run_analysis():
    print(f"VERDICT lexicon: {len(VERDICT)} tokens", flush=True)
    print(f"Streaming {len(MODEL_FILES)} model CSVs in parallel...", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=len(MODEL_FILES)) as ex:
        futs = {ex.submit(process_model, (n, p)): n for n, p in MODEL_FILES.items()}
        for fut in as_completed(futs):
            name, d = fut.result()
            results[name] = d
            print(f"  done: {name}", flush=True)

    out = {
        "_meta": {
            "lexicon": {"verdict_n": len(VERDICT)},
            "denominator":   "all alphabetic tokens [a-z]+ (no stopword/length filter)",
            "rate_unit":     "verdict_hits / n_total * 100  (= % of tokens that are verdict markers)",
            "position_unit": "first-match start char / text length, model-averaged",
        },
        "per_model": results,
    }
    json_path = OUT / "verdict_metrics.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {json_path}")
    return out


# ====================================================================
# AGGREGATION (per-cell matrices)
# ====================================================================

def matrix_rate(d):
    """% of tokens that are verdict markers, model-averaged."""
    models = list(d["per_model"].keys())
    M = np.zeros((len(TRAITS), len(CONTEXTS)))
    for i, t in enumerate(TRAITS):
        for j, c in enumerate(CONTEXTS):
            key = f"{t}|{c}"
            rates = []
            for m in models:
                num = d["per_model"][m]["verdict_hits"].get(key, 0)
                den = d["per_model"][m]["n_total"].get(key, 0)
                if den > 0:
                    rates.append(num / den * 100.0)
            M[i, j] = sum(rates) / len(rates) if rates else 0.0
    return M


def matrix_position(d):
    """Mean position of first verdict marker per model, then averaged across models."""
    models = list(d["per_model"].keys())
    M = np.zeros((len(TRAITS), len(CONTEXTS)))
    for i, t in enumerate(TRAITS):
        for j, c in enumerate(CONTEXTS):
            key = f"{t}|{c}"
            vals = []
            for m in models:
                ps = d["per_model"][m]["verdict_pos_sum"].get(key, 0.0)
                pn = d["per_model"][m]["verdict_pos_n"].get(key, 0)
                if pn > 0:
                    vals.append(ps / pn)
            M[i, j] = sum(vals) / len(vals) if vals else 0.0
    return M


# ====================================================================
# PLOTTING
# ====================================================================

def _text_color(value, cmap_name, vmin, vmax):
    if value is None or np.isnan(value):
        return "black"
    norm = max(0.0, min(1.0, (value - vmin) / max(vmax - vmin, 1e-9)))
    rgba = mpl.colormaps[cmap_name](norm)
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "black" if lum > 0.55 else "white"


def heatmap(ax, M, *, row_labels, col_labels, cmap, vmin=None, vmax=None,
            fmt="{:.2f}", fontsize=10, title=None, aspect="auto"):
    M = np.asarray(M, dtype=float)
    if vmin is None: vmin = float(np.nanmin(M))
    if vmax is None: vmax = float(np.nanmax(M))
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=fontsize)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=fontsize)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=fontsize, color=_text_color(v, cmap, vmin, vmax))
    if title:
        ax.set_title(title, fontsize=fontsize + 2, pad=6)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=DPI, facecolor="white",
                bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUT / f"{name}.pdf", facecolor="white",
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Wrote {name}.png + .pdf")


def fig_verdict(d):
    RATE = matrix_rate(d)
    POS  = matrix_position(d)
    n_v  = d["_meta"]["lexicon"]["verdict_n"]
    grid_aspect = len(CONTEXTS) / len(TRAITS)

    fig, axes = plt.subplots(1, 2, figsize=A4_LANDSCAPE)
    heatmap(axes[0], RATE,
            row_labels=[TRAIT_LABELS[t] for t in TRAITS], col_labels=CONTEXTS,
            cmap="YlOrRd", vmin=0, fmt="{:.2f}%",
            title=f"Verdict-marker rate  ({n_v} tokens; e.g. winner · wins · edges)",
            aspect=grid_aspect)
    heatmap(axes[1], POS,
            row_labels=[TRAIT_LABELS[t] for t in TRAITS], col_labels=CONTEXTS,
            cmap="RdYlGn_r", vmin=0, vmax=1, fmt="{:.2f}",
            title="Mean position of first verdict marker  (0 = early, 1 = late)",
            aspect=grid_aspect)
    fig.subplots_adjust(left=0.10, right=0.94, top=0.94, bottom=0.06, wspace=0.45)
    save(fig, "fig_verdict")


# ====================================================================
# Main
# ====================================================================

def main():
    json_path = OUT / "verdict_metrics.json"
    if json_path.exists():
        with open(json_path) as f:
            d = json.load(f)
        print(f"Loaded cached {json_path}")
    else:
        d = run_analysis()
    fig_verdict(d)


if __name__ == "__main__":
    main()