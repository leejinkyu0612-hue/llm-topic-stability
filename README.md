# LLM topic extraction: run-to-run stability and configuration sensitivity

Code, prompts, configuration, logs and processed data for the study
"Run-to-run stability of two-stage LLM topic extraction compared with LDA, NMF and BERTopic on scientific abstracts" (Jinkyu Lee, HappyAI).

## Contents
- `plan/pre-specified_analysis_plan.md` – analysis plan committed before the main experiment (see commit history for time stamps).
- `code/` – data collection (`collect_arxiv.py`), LLM pipeline (`llm_topics.py`), baselines (`baselines.py`), analysis (`analysis.py`), runner (`run_all.py`), figures.
- `data/` – arXiv identifiers, titles, abstracts and primary categories of the two corpora (CS8, QBIO6). arXiv metadata is distributed under CC0 1.0 (https://info.arxiv.org/help/api/tou.html).
- `runs/` – one JSON file per run (topic list, document assignments, settings).
- `logs/api_calls.jsonl` – one line per API call (model, provider, temperature, tokens, cost, latency, raw response). No API keys are stored.
- `results/` – computed metrics.

## Reproduce
```
pip install scikit-learn bertopic sentence-transformers umap-learn scipy numpy
python code/collect_arxiv.py data          # optional: re-collect (results may differ as arXiv changes)
set OPENROUTER_KEY_FILE=path\to\key.txt    # file containing an OpenRouter key
python code/run_all.py --part base
python code/run_all.py --part perturb
python code/baselines.py --corpus CS8 ; python code/baselines.py --corpus QBIO6
python code/analysis.py --tag main
```
LLM outputs from hosted APIs are not bit-for-bit reproducible; the logged responses in `logs/` allow the reported metrics to be recomputed exactly.

## License
Code: MIT. Data: arXiv metadata (CC0 1.0); derived files CC0 1.0.
