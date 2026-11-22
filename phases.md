# Weighted H-Index — Project Plan

**Goal:** Implement a context-aware, weighted h-index that weights citations by the relevance of the cited article to the citing article (assessed via an LLM on article abstracts), adapting Zhu et al. (2015) Influence-Primed H-index. Validate against conventional h-index and existing alternatives.

**Repo:** `weighted-h-index`  
**Language:** Python (managed with `uv`)  
**Dataset source:** OpenAlex API (free, abstracts + citation edges, no key required)  
**Primary LLM:** Qwen3.7 Flash (Alibaba) — cheapest Chinese provider found (~$0.03 in / $0.13 out per 1M tokens), OpenAI-compatible endpoint  
**Fallback LLM:** DeepSeek V4 Flash ($0.22/$0.66 off-peak, 2× at peak)  
**Local model (later):** Ollama (Qwen3-4B/8B)  
**Documentation:** LaTeX white paper (`article` class, modular sections, `references.bib`, compiled with `latexmk`)

---

## Core idea

Since full article text is not always available, rely only on **abstracts**. Replace Zhu et al.'s in-body mention count `c` with an LLM-derived relevance weight `w(p_i → p_j) ∈ [0,1]` computed from the pair of citing/cited abstracts:

- Weighted citation count: `wcp(p_j) = Σ_{p_i cites p_j} w(p_i → p_j)`
- Weighted h-index: largest `h` such that ≥ h of the author's papers have `wcp ≥ h`

This preserves Zhu et al.'s formal definition but substitutes context-aware abstract-pair weights for raw mention counts.

---

## Phase 0 — Scaffold (complete)

- [x] `git init`
- [x] Directory structure: `src/`, `src/llm/`, `data/`, `notebooks/`, `docs/whitepaper/`, `tests/`, `.github/workflows/`
- [x] `README.md` — project overview, install, usage, model/config notes
- [x] `LICENSE` — MIT
- [x] `.gitignore` — Python + data + LaTeX artifacts
- [x] `pyproject.toml` — project metadata + dependencies (httpx, pydantic, typer, pandas, numpy, scipy, python-dotenv)
- [x] `.python-version` (3.11) + `uv` virtual environment (`.venv`)
- [x] `phases.md` — this file
- [x] Whitepaper LaTeX skeleton: `main.tex`, `abstract.tex`, `introduction.tex`, `related_work.tex`, `methodology.tex`, `experiments.tex`, `results.tex`, `conclusion.tex`, `references.bib` (incl. Zhu et al. 2015, asi.23179), `latexmkrc`
- [x] `.github/workflows/ci.yml` — lint + test on push/PR

---

## Phase 1 — Data pipeline (OpenAlex) (complete)

- [x] `src/data.py` — `Paper`, `CitationEdge`, `ResearcherCorpus` dataclasses with JSONL save/load
- [x] `OpenAlexClient` with polite-pool `mailto` header and retry/backoff on HTTP 429
- [x] `fetch_author_works` / `fetch_citing_works` (cursor-paginated)
- [x] `build_researcher_corpus`: author's oeuvre + all citing works, edges citing→cited, abstracts from OpenAlex inverted index
- [x] CLI `fetch` command — builds corpus and saves to `data/corpus.jsonl`
- [x] CLI `compute` command — conventional / hip / weighted h-index from saved corpus (weighted uses unit edge weights until Phase 2 LLM scoring)
- [x] `tests/test_index.py`, `tests/test_data.py` — 5 tests passing
- [x] Ruff lint clean


---

## Phase 2 — LLM integration (API) (complete)

- [x] `src/llm/types.py` — `RelevanceResponse` shared type (broke circular import)
- [x] `src/llm/client.py` — async OpenAI-compatible client with caching + cost tracking
- [x] `src/llm/cache.py` — SQLite-backed response cache keyed by abstract-pair hash
- [x] `src/llm/tracker.py` — cost/token tracking with per-provider pricing estimates
- [x] `src/llm/providers.py` — Qwen/DeepSeek/Kimi/GLM config, `load_dotenv()` on import
- [x] `src/llm/prompts.py` — relevance prompt + `clamp_weight`
- [x] `src/scoring.py` — `CorpusScorer`: concurrency control, rate delay, threshold ablation
- [x] CLI `compute` — `--use-llm`, `--provider`, `--concurrency`, `--threshold`, `--cache-db`, `--cost-report`
- [x] `tests/test_cache.py`, `tests/test_tracker.py`, `tests/test_scoring.py` — 14 tests passing
- [x] Ruff lint clean

---

## Phase 3 — Evaluation (current)

- [ ] Enhance `src/evaluation.py` — Spearman correlation, precision@N, threshold ablation
- [ ] Comparison harness: weighted vs conventional vs hip rankings
- [ ] Notebooks: result visualization (correlation plots, precision@N tables)
- [ ] Run evaluation on sampled OpenAlex researchers

---

## Phase 4 — Local model

- [ ] Ollama backend (`src/llm/ollama.py`), offline mode
- [ ] Parity tests vs API results
- [ ] Config switch: `LLM_BACKEND=api|ollama`

---

## Phase 5 — White paper (LaTeX)

- [ ] Draft: Abstract, Introduction, Related Work (Zhu et al. hip-index), Methodology, Experiments, Results, Conclusion
- [ ] Compile to PDF via `latexmk`
- [ ] Tables/figures for correlation and precision@N results

---

## Phase 6 — Polish

- [ ] Packaging (pip installable via uv)
- [ ] README completion
- [ ] CI workflow (`.github/workflows/ci.yml`): lint, typecheck, tests
- [ ] Release

---

## Repo structure

```
weighted-h-index/
├── README.md
├── LICENSE
├── .gitignore
├── phases.md
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── index.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── providers.py
│   │   └── prompts.py
│   ├── evaluation.py
│   └── cli.py
├── data/              # gitignored
├── notebooks/
├── docs/whitepaper/
│   ├── main.tex
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── methodology.tex
│   ├── experiments.tex
│   ├── results.tex
│   ├── conclusion.tex
│   ├── references.bib
│   └── latexmkrc
├── tests/
└── .github/workflows/ci.yml
```
