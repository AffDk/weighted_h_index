# Weighted H-Index

A context-aware, weighted h-index that weights citations by the relevance of the cited article to the citing article, assessed via an LLM on article abstracts. Adapts Zhu et al. (2015) Influence-Primed H-index.

## Idea

Conventional h-index treats all citations equally. The weighted h-index assigns each citation a weight `w ∈ [0,1]` based on how much the citing article relies on the cited article (computed from citing + cited abstracts by an LLM):

- Weighted citation count: `wcp(p_j) = Σ w(p_i → p_j)`
- Weighted h-index: largest `h` such that ≥ h papers have `wcp ≥ h`

## Repo status

Phases 0–1 complete (see `phases.md` for the full plan):
- Phase 0 — scaffold
- Phase 1 — OpenAlex data pipeline

## Install (uv)

This project uses [uv](https://github.com/astral-sh/uv) for environment and package management.

```bash
# Create virtual environment and install package + dev tools
uv venv
uv pip install -e . pytest ruff

# Activate (PowerShell)
.venv\Scripts\activate

# Or run via uv without activating
uv run python -m src.cli --help
```

## Safe API key setup

Never commit real API keys to git. The repo uses a `.env` file (gitignored) loaded automatically via `python-dotenv`.

### 1. Get an API key

**Primary (recommended) — Gemini via Google AI Studio**:
1. Sign in at [Google AI Studio](https://aistudio.google.com).
2. Open the [API keys page](https://aistudio.google.com/app/apikey).
3. Create an API key.

### 2. Create your local `.env`

```powershell
# Copy the template
Copy-Item .env.example .env
```

Then edit `.env` and paste your key:

```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=sk-your_actual_key_here
```

Only one provider key is strictly required. `.env` is listed in `.gitignore` and will never be committed.

### 3. Verify

```powershell
.venv\Scripts\python scripts/test_gemini.py
```

The test never prints the key. It calls the Gemini API once with cache disabled and reports failures with the key redacted.

If you get `Missing API key env var`, your `.env` key isn't loaded — check the filename and spelling.
If you get `401`, the API key is invalid or not enabled for the Gemini API.
If you get `403`, the key is authenticated but lacks permission or billing is blocked.
If you get `404`, the model name is wrong or not available in your region.

## Usage

```bash
weighted-h-index fetch --author <openalex-id>
weighted-h-index compute --author <id> --method weighted
```

Example (OpenAlex author ID for Hirsch):

```bash
weighted-h-index fetch --author A5103253126
weighted-h-index compute --author A5103253126 --method weighted
```

## LLM configuration

Set `LLM_PROVIDER=gemini` (default), plus `GEMINI_API_KEY`.

| Provider | Env var | Get key at |
|---|---|---|
| Gemini | `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey |

Optional overrides: `GEMINI_BASE_URL`, `GEMINI_MODEL`.

## Citation

Zhu et al. (2015), "Measuring Academic Influence: Not All Citations Are Equal", JAIST 66(2). DOI: 10.1002/asi.23179
