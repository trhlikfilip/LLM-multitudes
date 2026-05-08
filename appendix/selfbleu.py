"""
Self-BLEU pipeline (data analysis + figure) — standalone.

Streams the 5 model CSVs once, reservoir-samples up to 400 reasonings per
(model, trait, context, side), then for each (model, trait, context) cell:
  - draws random pairs of reasonings
  - computes sentence-level BLEU-4 between them
  - averages the BLEU scores

Higher self-BLEU = more boilerplate / templated reasoning.
Lower self-BLEU = more diverse phrasing.

Output:
  context_metrics/vader_samples.jsonl   (cached sample texts; reused if present)
  context_metrics/self_bleu.json        (per-cell BLEU scores)
  context_metrics/fig_self_bleu.png + .pdf
"""

import csv
import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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
    "Claude Sonnet 4.6": BASE / "Claude Sonnet 4.6"  / "comparisons_all_anthropic.claude-sonnet-4-6_t1.csv",
    "Llama 3.1 8b":      BASE / "Llama 3.1 8b chat"  / "comparisons_all_llama-3.1-8b-instruct_t1.csv",
    "Llama 3.3 70b":     BASE / "Llama 3.3 70b chat" / "comparisons_all_llama-3.3-70b-instruct_t1.csv",
    "Mistral 4 Small":   BASE / "Mistral 4 Small"    / "comparisons_all_mistral-small-2603_t1.csv",
    "Qwen MoE":          BASE / "Qwen MoE Chat"      / "comparisons_all_qwen3-30b-a3b-thinking-2507_t1.csv",
}

# ---------------- Knobs ----------------

SAMPLES_PER_CELL_PER_SIDE = 400
N_PAIRS_PER_MODEL_CELL    = 200
TOKEN_RE                  = re.compile(r"[a-z]+")

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
# STAGE 1: stream + reservoir-sample reasonings per (model, trait, ctx, side)
# ====================================================================

def sample_reasonings_for_model(args):
    """Reservoir-sample up to SAMPLES_PER_CELL_PER_SIDE reasonings per
    (trait, ctx, side) from one model's CSV."""
    model_name, path = args
    rng = random.Random(hash(model_name) & 0xFFFFFFFF)

    samples = defaultdict(list)
    seen    = defaultdict(int)

    def reservoir_add(key, text):
        seen[key] += 1
        bucket = samples[key]
        if len(bucket) < SAMPLES_PER_CELL_PER_SIDE:
            bucket.append(text)
        else:
            j = rng.randrange(seen[key])
            if j < SAMPLES_PER_CELL_PER_SIDE:
                bucket[j] = text

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trait = row["trait"]
            ctx   = row["usecase"]
            for side, key in (("ab", "raw_ab"), ("ba", "raw_ba")):
                text = row.get(key) or ""
                if text.strip():
                    reservoir_add((trait, ctx, side), text)

    flat = []
    for (trait, ctx, side), texts in samples.items():
        for text in texts:
            flat.append({
                "model": model_name, "trait": trait, "context": ctx,
                "side": side, "text": text,
            })
    return flat


def collect_samples():
    print(f"Streaming {len(MODEL_FILES)} model CSVs (reservoir sampling)...", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=len(MODEL_FILES)) as ex:
        futs = {ex.submit(sample_reasonings_for_model, (n, p)): n
                for n, p in MODEL_FILES.items()}
        for fut in as_completed(futs):
            recs = fut.result()
            print(f"  done: {recs[0]['model']}  | {len(recs)} samples", flush=True)
            out.extend(recs)
    samples_path = OUT / "vader_samples.jsonl"
    with open(samples_path, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {samples_path} ({len(out)} samples)")
    return out


# ====================================================================
# STAGE 2: BLEU over random pairs within each (model, trait, ctx) cell
# ====================================================================

def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def compute_self_bleu(records):
    print("Computing self-BLEU per (model, trait, context) cell...", flush=True)
    samples = defaultdict(list)
    for r in records:
        samples[(r["model"], r["trait"], r["context"])].append(tokenize(r["text"]))
    print(f"  loaded {sum(len(v) for v in samples.values())} reasonings across "
          f"{len(samples)} cells")

    sf  = SmoothingFunction().method1
    rng = random.Random(42)

    per_model = {}
    cells = sorted({(t, c) for (_m, t, c) in samples.keys()})

    for (model, trait, ctx), texts in samples.items():
        if len(texts) < 2:
            continue
        bleus = []
        for _ in range(N_PAIRS_PER_MODEL_CELL):
            i, j = rng.sample(range(len(texts)), 2)
            ref, cand = texts[i], texts[j]
            if not ref or not cand:
                continue
            bleus.append(sentence_bleu([ref], cand, smoothing_function=sf))
        per_model.setdefault(model, {}).setdefault(trait, {})[ctx] = (
            sum(bleus) / len(bleus) if bleus else 0.0
        )

    # Model-averaged
    model_avg = {}
    for trait, ctx in cells:
        vals = []
        for model in per_model:
            v = per_model[model].get(trait, {}).get(ctx)
            if v is not None:
                vals.append(v)
        if vals:
            model_avg.setdefault(trait, {})[ctx] = sum(vals) / len(vals)

    out = {
        "_meta": {
            "metric": "Mean sentence-level BLEU-4 (NLTK, smoothing method 1) "
                      "between random pairs of reasonings within each cell",
            "n_pairs_per_model_cell":   N_PAIRS_PER_MODEL_CELL,
            "samples_per_cell_per_side": SAMPLES_PER_CELL_PER_SIDE,
            "interpretation": "higher = more templated / boilerplate; "
                              "lower = more diverse phrasing",
        },
        "model_averaged": model_avg,
        "per_model":      per_model,
    }
    out_path = OUT / "self_bleu.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")

    # Quick text summary
    contexts_seen = sorted({c for t in model_avg for c in model_avg[t]})
    print("\nSelf-BLEU (model-averaged):")
    print(f'{"trait":24s}' + "".join(f"{c:>9s}" for c in contexts_seen))
    for t in sorted(model_avg):
        print(f"{t:24s}" + "".join(f"{model_avg[t][c]:9.3f}" for c in contexts_seen))

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


def fig_self_bleu(d):
    M = np.array([[d["model_averaged"][t][c] for c in CONTEXTS] for t in TRAITS])
    fig, ax = plt.subplots(figsize=(A4_LANDSCAPE[0] / 2, A4_LANDSCAPE[1]))
    grid_aspect = len(CONTEXTS) / len(TRAITS)
    vmin, vmax = 0.0, float(np.nanmax(M))
    cmap = "YlOrRd"
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect=grid_aspect)
    ax.set_xticks(range(len(CONTEXTS))); ax.set_xticklabels(CONTEXTS, fontsize=10)
    ax.set_yticks(range(len(TRAITS)));   ax.set_yticklabels(
        [TRAIT_LABELS[t] for t in TRAITS], fontsize=10)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=10, color=_text_color(v, cmap, vmin, vmax))
    ax.set_title("Self-BLEU within cell  (higher = more templated; lower = more diverse)",
                 fontsize=12, pad=6)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.92, bottom=0.06)
    save(fig, "fig_self_bleu")


# ====================================================================
# Main
# ====================================================================

def main():
    json_path    = OUT / "self_bleu.json"
    samples_path = OUT / "vader_samples.jsonl"

    if json_path.exists():
        with open(json_path) as f:
            d = json.load(f)
        print(f"Loaded cached {json_path}")
    else:
        if samples_path.exists():
            print(f"Loading existing samples from {samples_path}")
            records = []
            with open(samples_path) as f:
                for line in f:
                    records.append(json.loads(line))
        else:
            records = collect_samples()
        d = compute_self_bleu(records)
    fig_self_bleu(d)


if __name__ == "__main__":
    main()
