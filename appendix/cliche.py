"""
Phase 4 — extra reasoning-text experiments (one streaming pass over the full corpus).

Computes per (model, trait, context):

  A1 boosters / certainty markers      (lexicon match)
  A2 personal pronouns                  (lexicon match: 1sg, 1pl, 2nd, 3sg, 3pl)
  C1 concrete-vs-abstract               (capitalised + digit token rate; abstract-lexicon rate)
  C2 comparative-construction density   (regex match)
  C3 cliché / formulaic-phrase rate     (regex match)
  D1 verdict position & rate            (verdict-lexicon match + position of first hit)
  G1 stereotype-trope rate              (lexicon match across ~20 countries)
  H1 meta / disclaimer rate             (regex match)

All rates are converted to PERCENT-OF-TOKENS at viz time (denominator = total
alphabetic tokens, exactly like fig_register_markers).

Output: context_metrics/phase4_metrics.json
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

# ---------------- Lexicons ----------------

# A1: Boosters (Hyland 2005; Biber stance markers; ~110 tokens)
BOOSTERS = {
    "definitely", "certainly", "absolutely", "undoubtedly", "undeniably",
    "indisputably", "unquestionably", "surely", "indeed", "decidedly",
    "plainly", "manifestly", "evidently", "obviously", "clearly", "naturally",
    "very", "extremely", "highly", "exceptionally", "exceedingly", "immensely",
    "profoundly", "remarkably", "strikingly", "overwhelmingly", "completely",
    "totally", "utterly", "entirely", "wholly", "fully", "perfectly",
    "thoroughly", "vastly", "tremendously", "incredibly", "particularly",
    "shows", "show", "showed", "showing",
    "demonstrates", "demonstrate", "demonstrated", "demonstrating",
    "proves", "prove", "proved", "proven", "proving",
    "establishes", "establish", "established", "establishing",
    "confirms", "confirm", "confirmed", "confirming",
    "verifies", "verify", "verified", "verifying",
    "ensures", "ensure", "ensured", "ensuring",
    "guarantees", "guarantee", "guaranteed", "guaranteeing",
    "truly", "really", "actually", "literally", "genuinely",
    "absolute", "certain", "sure", "true", "real", "actual",
    "evident", "obvious", "clear", "manifest", "incontestable",
    "unmistakable", "incontrovertible",
    "exactly", "precisely",
    "must",
}

# A2: Pronouns by category
PRON_1SG = {"i", "me", "my", "mine", "myself"}
PRON_1PL = {"we", "us", "our", "ours", "ourselves"}
PRON_2   = {"you", "your", "yours", "yourself", "yourselves"}
PRON_3SG = {"he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "it", "its", "itself"}
PRON_3PL = {"they", "them", "their", "theirs", "themselves"}

# C1: Abstract-framing lexicon (broad social/cultural placeholders)
ABSTRACT = {
    "people", "society", "societies", "culture", "cultures",
    "lifestyle", "lifestyles", "atmosphere", "atmospheres",
    "ambiance", "ambience", "vibe", "vibes", "energy", "energies",
    "charm", "essence", "character", "spirit", "soul",
    "feel", "feeling", "feelings",
    "tradition", "traditions", "values", "value",
    "ideal", "ideals", "principle", "principles",
    "moral", "morals", "ethic", "ethics",
    "outlook", "mentality", "mindset", "attitude", "attitudes",
    "approach", "approaches",
    "way", "ways", "manner", "manners", "fashion", "style", "styles",
    "nature", "tone",
    "diversity", "uniqueness", "individuality",
    "experience", "experiences",
    "quality", "qualities", "aspect", "aspects",
    "factor", "factors", "element", "elements",
    "context", "contexts", "background", "backgrounds",
    "warmth", "comfort", "coziness",
    "community", "communities", "heritage",
    "identity", "identities",
    "philosophy", "philosophies",
    "thing", "things", "stuff",
}

# D1: Verdict / outcome words
VERDICT = {
    "winner", "wins", "won", "winning",
    "verdict",
    "victory", "victorious",
    "edges", "outperforms", "beats", "trumps", "surpasses",
    "exceeds", "outshines", "outranks", "outdoes",
    "preferable", "preferred", "preferring", "preference",
    "favors", "favored", "favoring", "favor",
    "concludes", "concluded", "concluding", "conclusion",
}

# G1: Stereotype lexicon — strong country-evoking tokens
STEREOTYPES = {
    # Japan
    "sushi", "anime", "manga", "samurai", "ninja", "geisha", "kabuki",
    "kimono", "ramen", "tokyo", "kyoto", "fuji", "sake", "haiku",
    "origami", "shinto", "wasabi", "bento",
    # Canada
    "maple", "hockey", "beaver", "moose", "hortons", "poutine",
    # France
    "baguette", "croissant", "eiffel", "champagne", "louvre",
    "bordeaux", "burgundy", "brie", "camembert", "macaron",
    # Italy
    "pizza", "pasta", "spaghetti", "vatican", "venice", "florence",
    "gelato", "colosseum", "tuscany", "espresso", "lasagna", "tiramisu",
    # USA
    "hollywood", "broadway", "manhattan", "yankee", "yankees",
    # UK
    "thames", "shakespeare", "windsor", "buckingham",
    # Germany
    "oktoberfest", "berlin", "bavaria", "bratwurst", "autobahn",
    "schnitzel", "pretzel",
    # India
    "bollywood", "taj", "mahal", "ganges", "saree", "sari", "naan",
    "tandoori", "delhi", "mumbai", "diwali", "monsoon", "yoga",
    # China
    "pagoda", "jade", "silk", "beijing", "shanghai", "ming",
    "dynasty", "wok", "dumpling", "dumplings",
    # Brazil
    "samba", "carnival", "carnaval", "rio", "favela", "favelas",
    "caipirinha", "bossa", "feijoada", "amazon",
    # Mexico
    "taco", "tacos", "burrito", "tequila", "mariachi", "aztec", "aztecs",
    "guacamole", "siesta", "fiesta", "salsa",
    # Australia
    "kangaroo", "koala", "outback", "sydney", "vegemite", "didgeridoo",
    "boomerang", "aussie",
    # Russia
    "vodka", "kremlin", "tsar", "moscow", "siberia", "borscht",
    # Egypt
    "pyramid", "pyramids", "pharaoh", "pharaohs", "nile", "sphinx", "cairo",
    # Greece
    "olympus", "parthenon", "gyros", "feta", "mediterranean",
    # Spain
    "flamenco", "bullfight", "tapas", "paella",
    # Turkey
    "kebab", "ottoman", "istanbul", "bosphorus", "baklava",
    # Argentina
    "tango", "gaucho", "asado",
    # Ireland
    "guinness", "leprechaun", "shamrock",
    # Switzerland
    "alps", "fondue", "cuckoo", "zurich", "geneva",
    # Nordic
    "viking", "ikea", "abba", "stockholm", "fjord", "fjords",
    # Nigeria
    "lagos", "yoruba", "igbo", "afrobeats", "jollof", "afrobeat",
    # Kenya
    "safari", "nairobi", "masai", "maasai", "kilimanjaro",
    # Czechia
    "prague", "pilsner", "kafka",
    # Indonesia
    "bali", "jakarta", "batik", "gamelan", "rendang",
    # South Korea
    "kpop", "kimchi", "seoul", "kdrama", "samsung", "hyundai",
    # South Africa
    "apartheid", "mandela",
    # Other
    "hookah", "kabuki",
}

# ---------------- Regex patterns ----------------

# C2: Comparative constructions
COMPARATIVE_PATTERNS = [
    r"\b[a-z]{3,}er\s+than\b",
    r"\bmore\s+[a-z]+\s+than\b",
    r"\bless\s+[a-z]+\s+than\b",
    r"\bfewer\s+[a-z]+\s+than\b",
    r"\bsuperior\s+to\b",
    r"\binferior\s+to\b",
    r"\bcompared\s+(?:to|with)\b",
    r"\bin\s+contrast\b",
    r"\bin\s+comparison\b",
    r"\bversus\b",
    r"\bwhereas\b",
    r"\bunlike\b",
    r"\brather\s+than\b",
    r"\bopposed\s+to\b",
    r"\bedges?\s+out\b",
    r"\boutperforms?\b",
    r"\bsurpass(?:es|ed)?\b",
    r"\bexceeds?\b",
]
COMPARATIVE_RE = re.compile("|".join(COMPARATIVE_PATTERNS))

# C3: Country-essay clichés (multi-word + single-word formulaic adjectives)
CLICHE_PATTERNS = [
    r"\brich\s+tapestry",
    r"\brich\s+heritage",
    r"\brich\s+history",
    r"\brich\s+cultural",
    r"\bmelting\s+pot",
    r"\bworld[- ]class",
    r"\boff\s+the\s+beaten\s+path",
    r"\bmust[- ]see",
    r"\bmust[- ]visit",
    r"\bland\s+of\s+contrasts?",
    r"\bvibrant\s+(?:culture|atmosphere|nightlife|community|cities)",
    r"\bdeep[- ]rooted",
    r"\bstunning\s+landscapes?",
    r"\bbreathtaking\s+(?:views?|landscapes?|scenery)",
    r"\bancient\s+traditions?",
    r"\b(?:unique|perfect|harmonious)\s+blend",
    r"\bhidden\s+gems?",
    r"\bfood\s+paradise",
    r"\bsecond\s+to\s+none",
    r"\btruly\s+unforgettable",
    r"\blike\s+no\s+other",
    r"\bcomes\s+alive",
    r"\btreasure\s+trove",
    r"\bwarm\s+hospitality",
    r"\bwarm\s+and\s+welcoming",
    r"\bsteeped\s+in",
    r"\bnestled\s+(?:in|between|amongst|among)",
    r"\bbustling\s+(?:streets|markets|cities|metropolis)",
    r"\bone[- ]of[- ]a[- ]kind",
    r"\bawe[- ]inspiring",
    r"\beye[- ]opening",
    r"\bjaw[- ]dropping",
    r"\bvibrant\b",
    r"\btapestry\b",
    r"\bboasts\b",
    r"\brenowned\b",
    r"\bcaptivating\b",
    r"\benchanting\b",
    r"\bmesmerizing\b",
    r"\bquintessentially?\b",
    r"\bcosmopolitan\b",
    r"\bpicturesque\b",
]
CLICHE_RE = re.compile("|".join(CLICHE_PATTERNS))

# H1: Meta / disclaimer language
META_PATTERNS = [
    r"\bas\s+an?\s+ai\b",
    r"\bas\s+a\s+language\s+model\b",
    r"\bas\s+an?\s+assistant\b",
    r"\bi['’]?m\s+an?\s+ai\b",
    r"\bi\s+am\s+an?\s+ai\b",
    r"\bi\s+cannot\b",
    r"\bi\s+can['’]?t\b",
    r"\bi\s+am\s+unable\b",
    r"\bi\s+don['’]?t\s+have\b",
    r"\bi\s+lack\b",
    r"\bsubjective\b",
    r"\bsubjectivity\b",
    r"\bopinions?\b",
    r"\bpersonal\s+(?:opinion|preference|view|taste)\b",
    r"\bdepend(?:s|ing)?\s+on\b",
    r"\bworth\s+noting\b",
    r"\bworth\s+mentioning\b",
    r"\bimportant\s+to\s+note\b",
    r"\bmany\s+factors\b",
    r"\bvarious\s+factors\b",
    r"\bnumerous\s+factors\b",
    r"\bcomplex\s+(?:issue|topic|question|matter)\b",
    r"\bno\s+(?:clear|definitive|easy|simple|single|right)\s+answer\b",
    r"\bmatter\s+of\s+(?:opinion|taste|preference)\b",
    r"\bhard\s+to\s+(?:say|determine|choose|decide)\b",
    r"\bdifficult\s+to\s+(?:say|determine|choose|decide)\b",
    r"\bto\s+be\s+fair\b",
    r"\beveryone\s+is\s+different\b",
    r"\bvaries\s+(?:by|between|from)\b",
]
META_RE = re.compile("|".join(META_PATTERNS))

# D1: verdict regex used for position-finding
VERDICT_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in VERDICT) + r")\b")

TOKEN_RE      = re.compile(r"[a-z]+")
TOKEN_CASE_RE = re.compile(r"[A-Za-z]+")
DIGIT_RE      = re.compile(r"\d+")

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


# ---------------- Per-model worker ----------------

def process_model(args):
    model_name, path = args
    n_total       = defaultdict(int)
    n_chars       = defaultdict(int)
    boost_hits    = defaultdict(int)
    pron_hits     = {k: defaultdict(int) for k in ("1sg", "1pl", "2", "3sg", "3pl")}
    abstract_hits = defaultdict(int)
    cap_tokens    = defaultdict(int)
    digit_tokens  = defaultdict(int)
    verdict_hits  = defaultdict(int)
    cliche_hits   = defaultdict(int)
    comp_hits     = defaultdict(int)
    meta_hits     = defaultdict(int)
    stereo_hits   = defaultdict(int)
    verdict_pos_sum = defaultdict(float)
    verdict_pos_n   = defaultdict(int)
    n_rows          = defaultdict(int)

    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row["trait"], row["usecase"])
            n_rows[key] += 2
            for side in ("raw_ab", "raw_ba"):
                text = row.get(side) or ""
                if not text:
                    continue
                low = text.lower()
                n_chars[key] += len(text)

                lc_tokens = TOKEN_RE.findall(low)
                tlen = len(lc_tokens)
                n_total[key] += tlen
                if tlen == 0:
                    continue

                for tok in TOKEN_CASE_RE.findall(text):
                    if tok[0].isupper():
                        cap_tokens[key] += 1
                digit_tokens[key] += len(DIGIT_RE.findall(text))

                for t in lc_tokens:
                    if t in BOOSTERS:    boost_hits[key]     += 1
                    if t in ABSTRACT:    abstract_hits[key]  += 1
                    if t in VERDICT:     verdict_hits[key]   += 1
                    if t in STEREOTYPES: stereo_hits[key]    += 1
                    if t in PRON_1SG:    pron_hits["1sg"][key] += 1
                    elif t in PRON_1PL:  pron_hits["1pl"][key] += 1
                    elif t in PRON_2:    pron_hits["2"][key]   += 1
                    elif t in PRON_3SG:  pron_hits["3sg"][key] += 1
                    elif t in PRON_3PL:  pron_hits["3pl"][key] += 1

                cliche_hits[key] += len(CLICHE_RE.findall(low))
                comp_hits[key]   += len(COMPARATIVE_RE.findall(low))
                meta_hits[key]   += len(META_RE.findall(low))

                m = VERDICT_RE.search(low)
                if m:
                    verdict_pos_sum[key] += m.start() / max(len(low), 1)
                    verdict_pos_n[key]   += 1

    def s(d): return {f"{t}|{c}": v for (t, c), v in d.items()}
    return model_name, {
        "n_total":       s(n_total),
        "n_chars":       s(n_chars),
        "n_rows":        s(n_rows),
        "boost_hits":    s(boost_hits),
        "pron_1sg_hits": s(pron_hits["1sg"]),
        "pron_1pl_hits": s(pron_hits["1pl"]),
        "pron_2_hits":   s(pron_hits["2"]),
        "pron_3sg_hits": s(pron_hits["3sg"]),
        "pron_3pl_hits": s(pron_hits["3pl"]),
        "abstract_hits": s(abstract_hits),
        "cap_tokens":    s(cap_tokens),
        "digit_tokens":  s(digit_tokens),
        "verdict_hits":  s(verdict_hits),
        "cliche_hits":   s(cliche_hits),
        "comp_hits":     s(comp_hits),
        "meta_hits":     s(meta_hits),
        "stereo_hits":   s(stereo_hits),
        "verdict_pos_sum": s(verdict_pos_sum),
        "verdict_pos_n":   s(verdict_pos_n),
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


def _matrix_from_phase4(d, num_field, denom_field="n_total", scale=100.0):
    """Model-averaged 6×5 matrix of (num_field / denom_field × scale)."""
    models = list(d["per_model"].keys())
    M = np.zeros((len(TRAITS), len(CONTEXTS)))
    for i, t in enumerate(TRAITS):
        for j, c in enumerate(CONTEXTS):
            key = f"{t}|{c}"
            rates = []
            for m in models:
                num = d["per_model"][m][num_field].get(key, 0)
                den = d["per_model"][m][denom_field].get(key, 0)
                if den > 0:
                    rates.append(num / den * scale)
            M[i, j] = sum(rates) / len(rates) if rates else 0.0
    return M


def fig_cliche(d):
    M = _matrix_from_phase4(d, "cliche_hits")
    n_c = d["_meta"]["lexicons"]["cliche_patterns_n"]
    fig, ax = plt.subplots(figsize=(A4_LANDSCAPE[0] / 2, A4_LANDSCAPE[1]))
    grid_aspect = len(CONTEXTS) / len(TRAITS)
    heatmap(ax, M,
            row_labels=[TRAIT_LABELS[t] for t in TRAITS], col_labels=CONTEXTS,
            cmap="YlOrRd", vmin=0, fmt="{:.2f}%", fontsize=10,
            title=f"Cliché / formulaic-phrase rate  ({n_c} patterns; "
                  "e.g. rich tapestry · vibrant · melting pot)",
            aspect=grid_aspect)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.92, bottom=0.06)
    save_fig(fig, "fig_cliche")


def main():
    print(f"Lexicons: BOOSTERS={len(BOOSTERS)}, ABSTRACT={len(ABSTRACT)}, "
          f"VERDICT={len(VERDICT)}, STEREOTYPES={len(STEREOTYPES)}", flush=True)
    print(f"Regex patterns: CLICHE={len(CLICHE_PATTERNS)}, "
          f"COMP={len(COMPARATIVE_PATTERNS)}, META={len(META_PATTERNS)}", flush=True)
    print("Streaming all 5 model CSVs (one pass for all phase-4 metrics)...", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=len(MODEL_FILES)) as ex:
        futs = {ex.submit(process_model, (n, p)): n for n, p in MODEL_FILES.items()}
        for fut in as_completed(futs):
            name, d = fut.result()
            results[name] = d
            print(f"  done: {name}", flush=True)

    out = {
        "_meta": {
            "lexicons": {
                "boosters_n":        len(BOOSTERS),
                "abstract_n":        len(ABSTRACT),
                "verdict_n":         len(VERDICT),
                "stereotypes_n":     len(STEREOTYPES),
                "cliche_patterns_n": len(CLICHE_PATTERNS),
                "comp_patterns_n":   len(COMPARATIVE_PATTERNS),
                "meta_patterns_n":   len(META_PATTERNS),
            },
            "denominator": "all alphabetic tokens [a-z]+ (no stopword filter, no length filter)",
            "rate_unit":   "rates are stored as % of all alphabetic tokens (hits / total * 100)",
        },
        "per_model": {},
    }

    for model, d in results.items():
        out["per_model"][model] = d

    with open(OUT / "phase4_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT / 'phase4_metrics.json'}")

    fig_cliche(out)


if __name__ == "__main__":
    main()