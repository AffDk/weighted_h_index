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

**Primary (recommended) — Qwen via Alibaba Cloud DashScope** (cheapest Chinese provider):
1. Sign up at [Alibaba Cloud](https://account.alibabacloud.com/login/login_new.htm).
2. Go to the [DashScope API Keys page](https://help.aliyun.com/zh/dashscope/developer-reference/api-details).
3. Create an API key. It looks like `sk-...`.

**Fallback — DeepSeek:**
1. Sign up at [DeepSeek Platform](https://platform.deepseek.com).
2. Go to [API Keys](https://platform.deepseek.com/api_keys).
3. Create a key. It looks like `sk-...`.

Other supported providers (Kimi/Moonshot, Zhipu/GLM) follow the same pattern; see `src/llm/providers.py` for the env var names.

### 2. Create your local `.env`

```powershell
# Copy the template
Copy-Item .env.example .env
```

Then edit `.env` and paste your key(s):

```ini
LLM_PROVIDER=qwen
DASHSCOPE_API_KEY=sk-your_actual_key_here
```

Only one provider key is strictly required; set `LLM_PROVIDER` to match. `.env` is listed in `.gitignore` and will never be committed.

### 3. Verify

```powershell
.venv\Scripts\python -c "from src.llm.providers import get_client; c = get_client(); print('client ready:', c.model)"
```

If you get `Missing API key env var`, your `.env` key isn't loaded — check the filename and spelling.

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

Set `LLM_PROVIDER=qwen` (default), `deepseek`, `kimi`, or `glm`, plus the matching API key env var:

| Provider | Env var | Get key at |
|---|---|---|
| Qwen (DashScope) | `DASHSCOPE_API_KEY` | Alibaba Cloud DashScope console |
| DeepSeek | `DEEPSEEK_API_KEY` | https://platform.deepseek.com/api_keys |
| Kimi (Moonshot) | `MOONSHOT_API_KEY` | Moonshot platform |
| GLM (Zhipu) | `ZHIPU_API_KEY` | Zhipu open platform |

## Citation

Zhu et al. (2015), "Measuring Academic Influence: Not All Citations Are Equal", JAIST 66(2). DOI: 10.1002/asi.23179
