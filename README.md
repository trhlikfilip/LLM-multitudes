# LLM-Multitudes

Code accompanying the NeurIPS 2026 submission **"LLMs Contain Multitudes: How Deployment Context Reshapes Model-Level Preferences and Values"**.

We test whether reported LLM-level preferences and values survive when the surrounding deployment context changes. Across five widely-used LLMs and over 1B generated tokens, deployment context (e.g., writing a Reddit post, a news article, a school essay) produces variation far larger than prompt paraphrasing or temperature shifts.

## Resources

- **Dataset (1B+ tokens, parsed votes, per-context vote matrices, fitted Thurstonian utilities, reasoning traces):** [`LLM-multitudes-neurips-2026/LLM-Multitudes`](https://huggingface.co/datasets/LLM-multitudes-neurips-2026/LLM-Multitudes) on Hugging Face.

If you only want to reproduce the statistical analyses, downloading the Hugging Face dataset is sufficient; the audit scripts below are only needed to regenerate raw model outputs.

## Repository structure

```

├── country preference/                 # Preference Elicitation (Section 4)
│ ├── audit.py                          # OpenRouter pairwise elicitation
│ ├── cmh.py                            # Cochran-Mantel-Haenszel test (RQ1)
│ └── country_preference_sig.py         # BH-FDR Mann-Whitney rank test (RQ1)
├── utility elicitation/                # Utility Elicitation (Section 5)
│ ├── audit_utility.py                  # elicitation + Thurstonian fit
│ ├── utility_per_domain_spearman.py    # Per-(model, domain) Spearman rho (RQ5)
│ └── utiltity_outcome_sig.py           # Bootstrap rank test, BH-FDR (RQ4)
└── appendix/                           # Reasoning Analysis (Appendix C)
├── jsd.py                              # JS divergence between contexts
├── register.py                         # Hedges + discourse markers
├── verdict.py                          # Verdict-marker rate + position
├── cliche.py                           # Cliché / formulaic-phrase rate
├── selfbleu.py                         # Self-BLEU within cell (templating)
└── formal.py                           # Formal-register components
```

## Setup

```bash
pip install openai numpy pandas scipy statsmodels torch tqdm
export OPENROUTER_API_KEY=...
# optional, recommended by OpenRouter:
export OPENROUTER_SITE_URL=...
export OPENROUTER_APP_NAME=...
```

Open-weight models (Llama 3.1-8B, Llama 3.3-70B, Qwen-3-30B-MoE, Mistral Small 4) are routed through OpenRouter. Claude Sonnet 4.6 is served via AWS Bedrock through the global cross-region inference profile. Full model identifiers are listed in Appendix E.1 of the paper.

The four analysis scripts (`cmh.py`, `country_preference_sig.py`, `utility_per_domain_spearman.py`, `utiltity_outcome_sig.py`) each contain a `BASE = Path("...")` placeholder pointing to the elicitation outputs. Set this to a local copy of the Hugging Face dataset (or to the directory the audit scripts wrote into). The bootstrap rank test additionally expects `bootstrap_mu_cache_N1000.npz` (a precomputed cache of 1000 Thurstonian fits per (model, context)) inside `BASE`.

## Reproducing the main experiments

### Experiment 1: Country preferences (Section 4)

15 countries, 6 traits, 5 deployment contexts, 20 repeats per pair, AB/BA counterbalanced. Total: 126,000 prompts per model.

```bash
cd "country preference"
python audit.py                                            # elicitation
python cmh.py comparisons_all_<tag>.csv                    # decision-level CMH (RQ1)
python country_preference_sig.py                           # rank-level BH-FDR (RQ1)
```

`audit.py` writes `comparisons_all_<tag>.csv` plus per-context splits and a `checkpoint_rows_all_<tag>.jsonl` resume log; an interrupted run can be restarted by re-invoking the same command.

### Experiment 2: Utility elicitation (Section 5)

50 outcomes across 6 domains, 5 deployment contexts, 10 repeats per pair, AB/BA counterbalanced. Total: 122,500 votes per model. The audit script also fits per-context Thurstonian-Mosteller utilities (anchored at the "no change" outcome, mu = 0).

```bash
cd "utility elicitation"
python audit_utility.py                                    # elicitation + Thurstonian fit
python utility_per_domain_spearman.py                      # per-domain stability (RQ5)
python utiltity_outcome_sig.py                             # bootstrap rank test (RQ4)
```

`audit_utility.py` is similarly crash-safe and supports `--fresh-start`, `--usecase <name>`, and `--repeats <n>`.

## Sampling configuration

All experiments use `temperature=1.0`, `max_tokens=768`, `top_p=1.0`. The exact context-induction lines, system messages, and counterbalancing scheme are documented in Section 3 (Figure 2) and Appendix E.1 of the paper.

### Appendix: Linguistic-style metrics

Six standalone scripts in `code/appendix/` reproduce the supplementary heatmaps
on the reasoning text of Experiment 1 (JS divergence, hedges, verdict markers,
clichés, self-BLEU, formal-register components). Each is end-to-end (corpus
scan → JSON → figure); set the `BASE` and `OUT` `Path("...")` placeholders at
the top of each file to your local copy of the country-preference CSVs.

## Citation

Anonymized for NeurIPS review.
