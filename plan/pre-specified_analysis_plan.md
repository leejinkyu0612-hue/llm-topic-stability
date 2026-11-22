# Pre-specified Analysis Plan (사전계획) — fixed before the main experiment

- Study: Run-to-run stability and configuration sensitivity of two-stage LLM topic extraction compared with LDA, NMF and BERTopic on scientific abstracts
- Author: Jinkyu Lee (HappyAI)
- Status: written and committed to GitHub before any main-experiment API call. Pilot runs (≤200 documents, tag `pilot`) were executed only to test the code; they are excluded from all analyses.
- Wording: this is a time-stamped, pre-specified plan in a public Git repository, not a registration with a formal registry.

## 1. Research questions
- RQ1 (validity): How closely do the document partitions produced by each method agree with the arXiv primary categories?
- RQ2 (repeatability): How similar are LLM partitions across identical repeated runs at temperature 0, and how does this compare with seed-to-seed agreement of LDA, NMF and BERTopic?
- RQ3 (configuration sensitivity): How much do prompt paraphrase, generation sample, document order and temperature change the partition relative to identical repetition?
- RQ4 (stage decomposition): Is run-to-run variation mainly introduced in the topic generation stage or in the assignment stage?
- RQ5 (practical cost): What are the monetary cost, wall-clock time and invalid-output rate per run?

## 2. Hypotheses
- H1: Mean NMI with the reference categories is higher for each LLM pipeline than for LDA, on both corpora (one-sided expectation, tested two-sided).
- H2 (non-directional comparison): Mean pairwise ARI of identical LLM runs at temperature 0 is below 1.00; its level relative to baseline seed stability is reported without a directional hypothesis because prior reports conflict (Pham et al., 2024 vs. Reiss, 2023).
- H3: For each perturbation factor, mean ARI between perturbed and base runs is lower than mean pairwise ARI among base runs.
- H4: Mean pairwise ARI of assignment-only repetitions (topic list fixed) is higher than that of full-pipeline repetitions.

## 3. Data
- arXiv API metadata (CC0 1.0). CS8: cs.CV, cs.CL, cs.CR, cs.RO, cs.DB, cs.NI, cs.SE, cs.DS, submitted in 2024; QBIO6: q-bio.NC, q-bio.PE, q-bio.BM, q-bio.GN, q-bio.MN, q-bio.QM, submitted in 2023–2024. 100 documents per primary category (seed 20261122). Collected 2026-11-22; collection log in `data/collection_log.json`.
- Input text for all methods: title + first 120 words of the abstract.
- Exclusion criteria: none at document level (all sampled documents retained). A document whose LLM assignment is missing or outside the topic list after one retry is labelled -1 ("unassigned") and kept as its own cluster in all metrics.

## 4. Methods and conditions
- LLMs via OpenRouter: openai/gpt-4o-mini, google/gemini-2.5-flash-lite, mistralai/mistral-small-3.2-24b-instruct, deepseek/deepseek-v3.2. Max tokens 1500 (generation), 1200 (assignment).
- Pipeline: generation on G=60 random documents (title + 50 words) asking for exactly K topics; assignment in batches of 20; K = number of reference categories (8 or 6).
- Baselines (seeds 1–10): LDA (α=0.1, η=0.01, 100 iterations; fixed after a pilot showed near-chance agreement with sklearn defaults, before any main run), NMF (TF-IDF, random init), BERTopic (all-MiniLM-L6-v2, UMAP 5-d, k-means K).
- Conditions: base (T=0, prompt P0, generation seed 0, order seed 0) × 5 repetitions, both corpora, 4 LLMs; on CS8 for gpt-4o-mini and mistral-small-3.2: temp (T=0.7) × 3, prompt (P1, P2, P3), gensample (seeds 1–3), order (seeds 1–3), fixedtopics (topics from base r1, assignment repeated 3×); on QBIO6 for gpt-4o-mini: prompt (P1–P3), gensample (seeds 1–3).
- Budget cap: US$3.00 total API spend; if exceeded, remaining conditions are dropped in reverse order of the list above and reported.

## 5. Outcomes (primary and secondary)
- Primary: NMI (RQ1); mean pairwise ARI (RQ2–RQ4).
- Secondary: ARI and P1 versus reference; pairwise NMI; prevalence deviation (mean and maximum absolute difference in topic shares after Hungarian alignment, percentage points); unassigned rate; topics used; cost (US$); wall time (s); tokens.
- Exploratory (labelled as such): label-name semantic stability across LLM runs; corpus × method interaction; cross-listed (multi-category) documents.

## 6. Analysis
- 95% CIs by bootstrap over runs (2,000 resamples, seed 20261122) for mean pairwise agreement; percentile bootstrap over runs for per-run metrics.
- Method comparisons on per-run NMI: Mann–Whitney U with Holm correction within corpus; effect size Cliff's delta.
- H3/H4: difference of means with bootstrap CI; supported if the CI excludes 0 in the predicted direction.
- Thresholds: "high agreement" reported at ARI ≥ 0.7, 0.8 and 0.9 (sensitivity); no single cut-off determines conclusions.
- No causal claims beyond the manipulated factors; results are conditional on model versions, providers and date (2026-11).

## 7. Deviations
Any change after this commit will be committed separately with a reason and reported in the paper as a deviation; any analysis not listed here will be labelled exploratory.
