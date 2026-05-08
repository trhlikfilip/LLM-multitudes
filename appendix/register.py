"""
Re-stream all 5 model CSVs with comprehensive register-marker lexicons.

Counts target words DIRECTLY from raw lowercased text (no stopword filter,
no length filter), so modals like 'might'/'could'/'may' that the cached
counts dropped are now included.

Lexicons drawn from Hyland (2005) academic-hedging lists, Biber's stance
markers, and standard discourse-marker inventories.

Output: context_metrics/register_markers.json with:
  - "_meta"            : the lexicons + denominator definition
  - "hedges"           : {trait: {context: rate per 1000 tokens}}, model-averaged
  - "discourse_markers": same
  - "per_model"        : {model: {hedges: ..., discourse_markers: ...}}
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

BASE = Path("...")
OUT = Path("...")
OUT.mkdir(exist_ok=True, parents=True)

MODEL_FILES = {
    "Claude Sonnet 4.6": BASE / "Claude Sonnet 4.6" / "comparisons_all_anthropic.claude-sonnet-4-6_t1.csv",
    "Llama 3.1 8b":      BASE / "Llama 3.1 8b chat" / "comparisons_all_llama-3.1-8b-instruct_t1.csv",
    "Llama 3.3 70b":     BASE / "Llama 3.3 70b chat" / "comparisons_all_llama-3.3-70b-instruct_t1.csv",
    "Mistral 4 Small":   BASE / "Mistral 4 Small" / "comparisons_all_mistral-small-2603_t1.csv",
    "Qwen MoE":          BASE / "Qwen MoE Chat" / "comparisons_all_qwen3-30b-a3b-thinking-2507_t1.csv",
}

HEDGES = {
    # epistemic adverbs (probability/likelihood)
    "perhaps", "maybe", "possibly", "probably", "likely", "unlikely",
    "arguably", "presumably", "supposedly", "allegedly", "apparently",
    "seemingly", "ostensibly", "conceivably", "plausibly", "purportedly",
    # frequency hedges
    "generally", "often", "sometimes", "frequently", "usually", "typically",
    "occasionally", "rarely", "infrequently", "sporadically",
    # approximation / degree
    "approximately", "roughly", "somewhat", "fairly", "rather", "quite",
    "relatively", "partly", "partially", "mostly", "mainly", "largely",
    "nearly", "almost", "essentially", "virtually", "practically",
    # epistemic verbs (lemma + common inflections)
    "seems", "seem", "seemed", "seeming",
    "appears", "appear", "appeared", "appearing",
    "suggests", "suggest", "suggested", "suggesting",
    "indicates", "indicate", "indicated", "indicating",
    "implies", "imply", "implied", "implying",
    "supposes", "suppose", "supposed", "supposing",
    "believes", "believe", "believed", "believing",
    "thinks", "think", "thought", "thinking",
    "suspects", "suspect", "suspected", "suspecting",
    "assumes", "assume", "assumed", "assuming",
    "presumes", "presume", "presumed", "presuming",
    "wonders", "wonder", "wondered", "wondering",
    "reckons", "reckon", "reckoned", "reckoning",
    "tends", "tend", "tended", "tending",
    "leans", "lean", "leaned", "leaning",
    "estimates", "estimate", "estimated", "estimating",
    "speculates", "speculate", "speculated",
    # epistemic modals (often filtered as stopwords)
    "might", "may", "could", "would", "should", "can",
    "ought", "must",
    # adjective hedges
    "possible", "probable", "plausible", "conceivable",
    "doubtful", "uncertain", "debatable", "questionable",
    "tentative", "preliminary", "speculative",
    # quantifier/qualifier (kind/sort as hedges in 'kind of')
    "kind", "sort", "kinds", "sorts",
}

DISCOURSE = {
    # sequencing
    "first", "firstly", "second", "secondly", "third", "thirdly",
    "fourth", "fourthly", "fifth", "fifthly",
    "finally", "lastly", "subsequently", "previously",
    "initially", "afterwards",
    # contrast / opposition
    "however", "nevertheless", "nonetheless", "conversely",
    "alternatively", "otherwise", "instead",
    "although", "though", "despite", "regardless",
    "whereas", "contrarily", "contrastively",
    # causation / consequence
    "therefore", "thus", "consequently", "accordingly", "hence",
    "ergo", "ultimately",
    # addition / elaboration
    "moreover", "furthermore", "additionally", "similarly", "likewise",
    "particularly", "specifically", "namely", "especially", "indeed",
    "besides",
    # concession
    "admittedly", "granted", "naturally", "certainly", "undeniably",
    "obviously", "evidently", "clearly",
    # summary / framing
    "overall", "essentially", "fundamentally", "basically",
    "briefly", "summarily",
}

TOKEN_RE = re.compile(r"[a-z]+")

# ---------------- Layout (for plotting) ----------------

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


def process_model(args):
    model_name, path = args
    hedge_hits = defaultdict(int)
    disc_hits  = defaultdict(int)
    total_tokens = defaultdict(int)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["trait"], row["usecase"])
            text = (row.get("raw_ab") or "") + " " + (row.get("raw_ba") or "")
            tokens = TOKEN_RE.findall(text.lower())
            total_tokens[key] += len(tokens)
            for t in tokens:
                if t in HEDGES:
                    hedge_hits[key] += 1
                if t in DISCOURSE:
                    disc_hits[key] += 1
    return model_name, {
        "hedges":       {f"{t}|{c}": v for (t, c), v in hedge_hits.items()},
        "discourse":    {f"{t}|{c}": v for (t, c), v in disc_hits.items()},
        "total_tokens": {f"{t}|{c}": v for (t, c), v in total_tokens.items()},
    }


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
            fmt="{:.2f}%", fontsize=10, title=None, aspect="auto"):
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


def save_fig(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=DPI, facecolor="white",
                bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUT / f"{name}.pdf", facecolor="white",
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Wrote {name}.png + .pdf")


def fig_register_markers(out_dict):
    """Two-panel figure: Hedges + Discourse markers, model-averaged.

    register_markers.json stores rates per 1000 tokens; convert to % per 100 tokens
    (divide by 10) so the cell labels read like the rest of the figures.
    """
    n_h = out_dict["_meta"]["n_hedges"]
    n_d = out_dict["_meta"]["n_discourse"]
    fig, axes = plt.subplots(1, 2, figsize=A4_LANDSCAPE)
    grid_aspect = len(CONTEXTS) / len(TRAITS)
    panels = [
        ("hedges",            f"Hedges  ({n_h} tokens; e.g. perhaps · seems · might)"),
        ("discourse_markers", f"Discourse markers  ({n_d} tokens; e.g. however · therefore)"),
    ]
    for ax, (fam, title) in zip(axes, panels):
        M = np.array([[out_dict[fam][t][c] / 10.0 for c in CONTEXTS] for t in TRAITS])
        heatmap(ax, M,
                row_labels=[TRAIT_LABELS[t] for t in TRAITS], col_labels=CONTEXTS,
                cmap="YlOrRd", vmin=0, fmt="{:.2f}%", fontsize=10,
                title=title, aspect=grid_aspect)
    fig.subplots_adjust(left=0.13, right=0.94, top=0.94, bottom=0.06, wspace=0.45)
    save_fig(fig, "fig_register_markers")


def main():
    print(f"HEDGES lexicon: {len(HEDGES)} tokens", flush=True)
    print(f"DISCOURSE lexicon: {len(DISCOURSE)} tokens", flush=True)
    print("Streaming all 5 model CSVs (raw-text counting, no stopword filter)...", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=len(MODEL_FILES)) as ex:
        futs = {ex.submit(process_model, (n, p)): n for n, p in MODEL_FILES.items()}
        for fut in as_completed(futs):
            name, d = fut.result()
            results[name] = d
            print(f"  done: {name}", flush=True)

    cells_set = set()
    for d in results.values():
        cells_set.update(d["total_tokens"].keys())
    cells = sorted(cells_set)
    traits = sorted({c.split("|")[0] for c in cells})
    contexts = sorted({c.split("|")[1] for c in cells})

    out = {
        "_meta": {
            "hedges_lexicon":    sorted(HEDGES),
            "discourse_lexicon": sorted(DISCOURSE),
            "n_hedges":    len(HEDGES),
            "n_discourse": len(DISCOURSE),
            "denominator": "all alphabetic tokens [a-z]+ in raw_ab + raw_ba (lowercased; no stopword filter, no length filter)",
            "rate_unit":   "occurrences per 1000 tokens",
        },
        "hedges":            {t: {} for t in traits},
        "discourse_markers": {t: {} for t in traits},
        "per_model":         {m: {"hedges": {t: {} for t in traits},
                                  "discourse_markers": {t: {} for t in traits}}
                              for m in results},
    }

    for trait in traits:
        for ctx in contexts:
            key = f"{trait}|{ctx}"
            hedge_rates = []
            disc_rates  = []
            for model, d in results.items():
                tot = d["total_tokens"].get(key, 0) or 1
                hr = d["hedges"].get(key, 0) / tot * 1000
                dr = d["discourse"].get(key, 0) / tot * 1000
                hedge_rates.append(hr)
                disc_rates.append(dr)
                out["per_model"][model]["hedges"][trait][ctx] = hr
                out["per_model"][model]["discourse_markers"][trait][ctx] = dr
            out["hedges"][trait][ctx]            = sum(hedge_rates) / len(hedge_rates)
            out["discourse_markers"][trait][ctx] = sum(disc_rates)  / len(disc_rates)

    with open(OUT / "register_markers.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT / 'register_markers.json'}")

    fig_register_markers(out)

    # Quick summary
    for fam_key, label in (("hedges", "hedges"), ("discourse_markers", "discourse markers")):
        print(f"\n[{label}] per 1000 tokens, model-averaged")
        print(f"  trait                    " + "  ".join(f"{c:>8s}" for c in contexts))
        for trait in traits:
            print(f"  {trait:22s}" + "  ".join(
                f"{out[fam_key][trait][c]:8.2f}" for c in contexts))


if __name__ == "__main__":
    main()
