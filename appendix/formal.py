"""
Six independent components (per (model, trait, context)):

  1) Latinate / formal vocabulary   (lexicon match,  ~250 tokens)
  2) Nominalisation rate            (-tion/-sion/-ment/-ance/-ence/-ity/-ism/-ship/-ization)
  3) Passive voice                   (regex: "be/been/being + past participle" + variants)
  4) Polysyllabic / long-word rate   (token length >= 10 chars)
  5) Contraction rate (negative)     (regex: don't, isn't, you're, it's, ...)
  6) Citation / attribution           (regex: "according to", "research suggests", ...)

All rates expressed as % of alphabetic tokens (hits / n_alphabetic_tokens * 100),
matching fig_register_markers' convention.

Output: context_metrics/formal_register.json
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
OUT  = Path("...")
OUT.mkdir(exist_ok=True, parents=True)

MODEL_FILES = {
    "Claude Sonnet 4.6": BASE / "Claude Sonnet 4.6" / "comparisons_all_anthropic.claude-sonnet-4-6_t1.csv",
    "Llama 3.1 8b":      BASE / "Llama 3.1 8b chat" / "comparisons_all_llama-3.1-8b-instruct_t1.csv",
    "Llama 3.3 70b":     BASE / "Llama 3.3 70b chat" / "comparisons_all_llama-3.3-70b-instruct_t1.csv",
    "Mistral 4 Small":   BASE / "Mistral 4 Small" / "comparisons_all_mistral-small-2603_t1.csv",
    "Qwen MoE":          BASE / "Qwen MoE Chat" / "comparisons_all_qwen3-30b-a3b-thinking-2507_t1.csv",
}

# ============== Latinate / scholarly vocabulary lexicon ==============
# Drawn from: Hyland's academic vocabulary, Coxhead's Academic Word List (AWL),
# and standard formality lexica. Filtered to clearly formal-register words that
# usually have a casual Anglo-Saxon equivalent in everyday writing.
LATINATE_FORMAL = {
    # action verbs (casual ↔ formal: get/obtain, show/demonstrate, etc.)
    "obtain", "obtains", "obtained", "obtaining",
    "acquire", "acquires", "acquired", "acquiring",
    "purchase", "purchases", "purchased", "purchasing",
    "construct", "constructs", "constructed", "constructing", "construction",
    "establish", "establishes", "established", "establishing",
    "determine", "determines", "determined", "determining", "determination",
    "demonstrate", "demonstrates", "demonstrated", "demonstrating",
    "examine", "examines", "examined", "examining", "examination",
    "comprise", "comprises", "comprised", "comprising",
    "constitute", "constitutes", "constituted", "constituting",
    "facilitate", "facilitates", "facilitated", "facilitating",
    "encompass", "encompasses", "encompassed", "encompassing",
    "exemplify", "exemplifies", "exemplified", "exemplifying",
    "endeavor", "endeavors", "endeavored", "endeavoring",
    "endeavour", "endeavours", "endeavoured", "endeavouring",
    "utilize", "utilizes", "utilized", "utilizing",
    "utilise", "utilises", "utilised", "utilising",
    "ascertain", "ascertains", "ascertained",
    "elucidate", "elucidates", "elucidated",
    "delineate", "delineates", "delineated",
    "underscore", "underscores", "underscored",
    "denote", "denotes", "denoted",
    "discern", "discerns", "discerned",
    "deem", "deems", "deemed",
    "warrant", "warrants", "warranted",
    "illustrate", "illustrates", "illustrated", "illustrating",
    "incorporate", "incorporates", "incorporated", "incorporating",
    "implement", "implements", "implemented", "implementing",
    "investigate", "investigates", "investigated", "investigating",
    "evaluate", "evaluates", "evaluated", "evaluating",
    "assess", "assesses", "assessed", "assessing",
    "indicate", "indicates", "indicated", "indicating",
    "represent", "represents", "represented", "representing",
    "contribute", "contributes", "contributed", "contributing",
    "analyze", "analyzes", "analyzed", "analyzing",
    "analyse", "analyses", "analysed", "analysing",
    "interpret", "interprets", "interpreted", "interpreting",
    "comprehend", "comprehends", "comprehended",
    "perceive", "perceives", "perceived",
    "conceive", "conceives", "conceived",
    "assert", "asserts", "asserted", "asserting",
    "contend", "contends", "contended",
    "posit", "posits", "posited",
    "presume", "presumes", "presumed",
    "presuppose", "presupposes", "presupposed",
    "infer", "infers", "inferred",
    "deduce", "deduces", "deduced",
    "synthesize", "synthesizes", "synthesized",
    "synthesise", "synthesises", "synthesised",
    "exhibit", "exhibits", "exhibited", "exhibiting",
    "manifest", "manifests", "manifested",
    "yield", "yields", "yielded", "yielding",  # in formal "yields results"

    # adjectives (academic / scholarly)
    "additional", "significant", "substantial", "considerable",
    "extensive", "comprehensive", "exhaustive", "rigorous", "robust",
    "thorough", "systematic", "nuanced", "multifaceted",
    "heterogeneous", "homogeneous", "disparate", "discrete",
    "intrinsic", "extrinsic", "implicit", "explicit",
    "imperative", "paramount", "pertinent", "salient", "cogent",
    "prevalent", "ubiquitous", "extant", "nascent", "incumbent",
    "eminent", "imminent", "tantamount", "commensurate", "conducive",
    "judicious", "prudent", "holistic", "quintessential",
    "paradigmatic", "ostensible", "predominant", "feasible",
    "viable", "tenable", "untenable", "discernible",
    "indispensable", "negligible", "marginal", "fundamental",
    "rudimentary", "preliminary", "subsequent", "antecedent",
    "concurrent", "ulterior",

    # adverbs / connectives (formal)
    "additionally", "moreover", "furthermore", "consequently",
    "accordingly", "thereby", "therein", "thereafter", "hitherto",
    "henceforth", "wherein", "whereby", "ergo",
    "manifestly", "indubitably", "incontrovertibly", "demonstrably",
    "inherently", "intrinsically", "extrinsically", "fundamentally",
    "categorically", "unequivocally", "predominantly", "primarily",
    "concomitantly", "concurrently", "subsequently", "respectively",
    "notably", "particularly", "specifically",

    # nouns (academic / abstract)
    "implication", "implications", "ramification", "ramifications",
    "consideration", "considerations", "conjecture", "supposition",
    "presumption", "consensus", "discourse",
    "phenomenon", "phenomena", "criterion", "criteria",
    "paradigm", "paradigms", "methodology", "methodologies",
    "framework", "frameworks", "epitome", "exemplar", "exemplars",
    "synthesis", "antithesis", "thesis", "hypothesis", "hypotheses",
    "premise", "corollary", "axiom", "axioms",
    "tenet", "tenets", "doctrine", "doctrines",
    "exposition", "delineation", "elucidation", "depiction",
    "differentiation", "stratification", "categorization",
    "classification", "manifestation", "manifestations",
    "iteration", "iterations",
}

# ============== Nominalisation suffixes ==============
# A token is counted as a nominalisation if it ends with any of these AND has length >= 5
# (avoid catching short fragments like "tion" being a substring of just-the-suffix tokens).
NOM_SUFFIXES = ("tion", "sion", "ment", "ance", "ence", "ity",
                "ization", "isation", "ism", "ship")
NOM_MIN_LEN = 5

# ============== Long-word threshold ==============
LONG_WORD_MIN_LEN = 10

# ============== Passive voice patterns ==============
# Conservative: catches "be/been/being + V-ed|V-en + common irregulars",
# plus "has/have/had been + pp" and modal + be + pp.
PP_IRREGULAR = (
    r"made|seen|done|known|shown|given|found|chosen|taken|written|put|"
    r"cut|set|let|hit|read|spoken|broken|driven|risen|fallen|drawn|grown|"
    r"thrown|worn|forgotten|gotten|held|lost|paid|kept|left|brought|sought|"
    r"taught|caught|fought|dealt|meant|built|sent|spent"
)
PASSIVE_PATTERNS = [
    rf"\b(?:is|are|was|were|be|been|being)\s+(?:[a-z]+ed|[a-z]+en|{PP_IRREGULAR})\b",
    rf"\b(?:has|have|had)\s+been\s+(?:[a-z]+ed|[a-z]+en|{PP_IRREGULAR})\b",
    rf"\b(?:will|shall|can|could|may|might|should|would|must)\s+be\s+(?:[a-z]+ed|[a-z]+en|{PP_IRREGULAR})\b",
    rf"\bto\s+be\s+(?:[a-z]+ed|[a-z]+en|{PP_IRREGULAR})\b",
]
PASSIVE_RE = re.compile("|".join(PASSIVE_PATTERNS))

# ============== Contraction patterns ==============
CONTRACTION_PATTERNS = [
    r"\b(?:i|you|he|she|it|we|they|that|there|here|who|what|how|where|when|why)['’]s\b",
    r"\b(?:i|you|we|they)['’]ve\b",
    r"\b(?:i|you|he|she|we|they|it|that|there)['’]ll\b",
    r"\b(?:i|you|he|she|we|they)['’]d\b",
    r"\b(?:you|we|they|who)['’]re\b",
    r"\bi['’]m\b",
    r"\b(?:do|does|did|is|are|was|were|has|have|had|will|would|should|could|might|must|need|dare)n['’]t\b",
    r"\b(?:can|won|shan|ain)['’]t\b",
    r"\blet['’]s\b",
    r"\b(?:it|that|there|here|what|who|how|where|when)['’]s\b",
]
CONTRACTION_RE = re.compile("|".join(CONTRACTION_PATTERNS))

# ============== Citation / attribution patterns ==============
CITATION_PATTERNS = [
    r"\baccording\s+to\b",
    r"\bresearch\s+(?:suggests|shows|indicates|finds|demonstrates|reveals|has\s+shown)\b",
    r"\bstudies\s+(?:suggest|show|indicate|find|demonstrate|reveal|have\s+shown)\b",
    r"\bevidence\s+(?:suggests|shows|indicates|points\s+to|supports)\b",
    r"\bdata\s+(?:show|shows|suggest|suggests|indicate|indicates)\b",
    r"\bas\s+(?:noted|argued|shown|stated|observed|reported|established)\s+by\b",
    r"\bit\s+(?:has\s+been|is)\s+(?:argued|shown|noted|observed|established|known|recognised|recognized)\b",
    r"\bstatistics?\s+(?:show|shows|suggest|suggests|indicate|indicates)\b",
    r"\bsources?\s+(?:indicate|suggest|report|note)\b",
    r"\b(?:scholars|researchers|experts|analysts|academics)\s+(?:argue|suggest|claim|note|observe|maintain|hold)\b",
    r"\b(?:findings|results)\s+(?:show|shows|suggest|suggests|indicate|indicates|demonstrate)\b",
    r"\bwidely\s+(?:argued|believed|held|considered|recognized|recognised|accepted)\b",
    r"\bgenerally\s+(?:agreed|accepted|believed|held)\b",
]
CITATION_RE = re.compile("|".join(CITATION_PATTERNS))

# ============== Tokenisation ==============
TOKEN_RE = re.compile(r"[a-z]+")

# ============== Layout (for plotting) ==============
A4_PORTRAIT = (8.27, 11.69)
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


def is_nominalisation(tok):
    if len(tok) < NOM_MIN_LEN:
        return False
    return any(tok.endswith(s) for s in NOM_SUFFIXES)


def process_model(args):
    model_name, path = args
    n_total       = defaultdict(int)
    latinate_hits = defaultdict(int)
    nom_hits      = defaultdict(int)
    long_hits     = defaultdict(int)
    passive_hits  = defaultdict(int)
    contract_hits = defaultdict(int)
    citation_hits = defaultdict(int)

    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["trait"], row["usecase"])
            for side in ("raw_ab", "raw_ba"):
                text = row.get(side) or ""
                if not text:
                    continue
                low = text.lower()
                tokens = TOKEN_RE.findall(low)
                tlen = len(tokens)
                n_total[key] += tlen
                if tlen == 0:
                    continue

                # token-level passes
                for t in tokens:
                    if t in LATINATE_FORMAL:
                        latinate_hits[key] += 1
                    if is_nominalisation(t):
                        nom_hits[key] += 1
                    if len(t) >= LONG_WORD_MIN_LEN:
                        long_hits[key] += 1

                # regex on the original text (apostrophes preserved)
                passive_hits[key]  += len(PASSIVE_RE.findall(low))
                citation_hits[key] += len(CITATION_RE.findall(low))
                contract_hits[key] += len(CONTRACTION_RE.findall(text))  # case-preserve for ’/'

    def s(d): return {f"{t}|{c}": v for (t, c), v in d.items()}
    return model_name, {
        "n_total":       s(n_total),
        "latinate_hits": s(latinate_hits),
        "nom_hits":      s(nom_hits),
        "long_hits":     s(long_hits),
        "passive_hits":  s(passive_hits),
        "contract_hits": s(contract_hits),
        "citation_hits": s(citation_hits),
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
            fmt="{:.2f}%", fontsize=8, title=None, aspect="auto"):
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


def _matrix_from_formal(d, num_field):
    """Model-averaged 6×5 matrix of (num_field / n_total × 100)."""
    models = list(d["per_model"].keys())
    M = np.zeros((len(TRAITS), len(CONTEXTS)))
    for i, t in enumerate(TRAITS):
        for j, c in enumerate(CONTEXTS):
            key = f"{t}|{c}"
            rates = []
            for m in models:
                num = d["per_model"][m][num_field].get(key, 0)
                den = d["per_model"][m]["n_total"].get(key, 0)
                if den > 0:
                    rates.append(num / den * 100.0)
            M[i, j] = sum(rates) / len(rates) if rates else 0.0
    return M


def fig_formal_register(d):
    """6-panel figure (3 rows × 2 cols, A4 portrait): all formal-register components."""
    n_lat   = d["_meta"]["components"]["latinate_n"]
    long_th = d["_meta"]["components"]["long_word_min_len"]

    panels = [
        ("latinate_hits",
         f"Latinate vocabulary  ({n_lat} tokens; e.g. obtain · demonstrate)",
         "YlOrRd"),
        ("nom_hits",
         "Nominalisation  (-tion · -ment · -ity · -ism)",
         "YlOrRd"),
        ("passive_hits",
         "Passive voice  (e.g. is determined · has been shown)",
         "YlOrRd"),
        ("long_hits",
         f"Long-word rate  (tokens ≥ {long_th} chars)",
         "YlOrRd"),
        ("citation_hits",
         "Citation / attribution  (e.g. according to · research suggests)",
         "YlOrRd"),
        ("contract_hits",
         "Contractions  (negative marker — don't · it's · I'm)",
         "YlGnBu"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=A4_PORTRAIT, squeeze=False)
    grid_aspect = len(CONTEXTS) / len(TRAITS)
    for ax, (field, title, cmap) in zip(axes.flat, panels):
        M = _matrix_from_formal(d, field)
        heatmap(ax, M,
                row_labels=[TRAIT_LABELS[t] for t in TRAITS], col_labels=CONTEXTS,
                cmap=cmap, vmin=0, fmt="{:.2f}%", fontsize=8,
                title=title, aspect=grid_aspect)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.96, bottom=0.04,
                        wspace=0.65, hspace=0.55)
    save_fig(fig, "fig_formal_register")


def main():
    print(f"Lexicons: LATINATE={len(LATINATE_FORMAL)}, "
          f"NOM_SUFFIXES={len(NOM_SUFFIXES)}, "
          f"PASSIVE={len(PASSIVE_PATTERNS)}, "
          f"CONTRACTION={len(CONTRACTION_PATTERNS)}, "
          f"CITATION={len(CITATION_PATTERNS)}", flush=True)
    print(f"Long-word threshold: >= {LONG_WORD_MIN_LEN} chars", flush=True)
    print("Streaming all 5 model CSVs (formal-register sweep)...", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=len(MODEL_FILES)) as ex:
        futs = {ex.submit(process_model, (n, p)): n for n, p in MODEL_FILES.items()}
        for fut in as_completed(futs):
            name, d = fut.result()
            results[name] = d
            print(f"  done: {name}", flush=True)

    out = {
        "_meta": {
            "components": {
                "latinate_n":       len(LATINATE_FORMAL),
                "nom_suffixes_n":   len(NOM_SUFFIXES),
                "passive_n":        len(PASSIVE_PATTERNS),
                "contraction_n":    len(CONTRACTION_PATTERNS),
                "citation_n":       len(CITATION_PATTERNS),
                "long_word_min_len": LONG_WORD_MIN_LEN,
                "nom_min_len":       NOM_MIN_LEN,
                "nom_suffix_list":   list(NOM_SUFFIXES),
            },
            "denominator": "all alphabetic tokens [a-z]+",
            "rate_unit":   "% of all alphabetic tokens (hits / n_total * 100)",
            "polarity":    "contraction rate is a negative formality marker; all others positive",
        },
        "per_model": {m: d for m, d in results.items()},
    }

    with open(OUT / "formal_register.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT / 'formal_register.json'}")

    fig_formal_register(out)

    # Quick text summary (model-averaged % of tokens) for each component
    cells = sorted({k for d in results.values() for k in d["n_total"]})
    traits = sorted({k.split("|")[0] for k in cells})
    contexts = sorted({k.split("|")[1] for k in cells})
    fields = [("latinate_hits", "Latinate"),
              ("nom_hits",      "Nominalisation"),
              ("long_hits",     "Long-word"),
              ("passive_hits",  "Passive"),
              ("citation_hits", "Citation"),
              ("contract_hits", "Contraction")]
    for f_key, label in fields:
        print(f"\n[{label}]  % of tokens, model-averaged")
        print("  trait                    " + "  ".join(f"{c:>8s}" for c in contexts))
        for t in traits:
            row = f"  {t:22s}"
            for c in contexts:
                key = f"{t}|{c}"
                rates = []
                for m in results:
                    num = results[m][f_key].get(key, 0)
                    den = results[m]["n_total"].get(key, 0)
                    if den > 0:
                        rates.append(num / den * 100.0)
                row += f"  {sum(rates)/len(rates):8.3f}"
            print(row)


if __name__ == "__main__":
    main()
