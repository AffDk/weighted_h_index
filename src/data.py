"""OpenAlex data fetcher: works (abstracts) and citation edges.

Builds a researcher subgraph: an author's oeuvre plus all works that cite it,
so each citation edge has both a citing and a cited abstract available for
LLM relevance scoring.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

OPENALEX_BASE = "https://api.openalex.org"
DEFAULT_PER_PAGE = 200
MAX_RETRIES = 5
BACKOFF_BASE = 1.5


@dataclass
class Paper:
    openalex_id: str
    title: str
    abstract: str
    publication_year: int | None
    cited_by_count: int
    is_author_work: bool = False


@dataclass
class CitationEdge:
    citing_paper_id: str
    cited_paper_id: str


@dataclass
class ResearcherCorpus:
    researcher_id: str
    papers: dict[str, Paper] = field(default_factory=dict)
    edges: list[CitationEdge] = field(default_factory=list)

    def add_paper(self, paper: Paper) -> None:
        self.papers[paper.openalex_id] = paper

    def add_edge(self, citing_id: str, cited_id: str) -> None:
        self.edges.append(CitationEdge(citing_id, cited_id))

    def cited_papers(self) -> list[Paper]:
        return [p for p in self.papers.values() if p.is_author_work]

    def citing_papers(self) -> list[Paper]:
        return [p for p in self.papers.values() if not p.is_author_work]

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for p in self.papers.values():
            rec = asdict(p)
            rec["type"] = "paper"
            records.append(rec)
        for e in self.edges:
            records.append({"type": "edge", **asdict(e)})
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, researcher_id: str, path: str | Path) -> ResearcherCorpus:
        corpus = cls(researcher_id=researcher_id)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") == "paper":
                    p = Paper(
                        openalex_id=rec["openalex_id"],
                        title=rec["title"],
                        abstract=rec["abstract"],
                        publication_year=rec.get("publication_year"),
                        cited_by_count=rec.get("cited_by_count", 0),
                        is_author_work=rec.get("is_author_work", False),
                    )
                    corpus.add_paper(p)
                elif rec.get("type") == "edge":
                    corpus.add_edge(rec["citing_paper_id"], rec["cited_paper_id"])
        return corpus


class OpenAlexClient:
    """Thin client with polite-pool email and retry/backoff on 429."""

    def __init__(self, mailto: str | None = None, timeout: float = 30.0) -> None:
        self.headers: dict[str, str] = {}
        if mailto:
            self.headers["mailto"] = mailto
        self._client = httpx.Client(timeout=timeout, headers=self.headers)

    def _get(self, url: str, params: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.get(url, params=params)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("retry-after", BACKOFF_BASE ** attempt))
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_BASE ** attempt)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def close(self) -> None:
        self._client.close()


def _select_fields() -> str:
    return (
        "id,title,display_name,publication_year,authorships,"
        "cited_by_count,abstract_inverted_index,ids"
    )


def abstract_from_inverted_index(inverted: dict | None) -> str:
    """Reconstruct abstract text from OpenAlex inverted index."""
    if not inverted:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for pos in positions:
            words.append((pos, word))
    words.sort(key=lambda x: x[0])
    return " ".join(w for _, w in words)


def _work_to_paper(work: dict, is_author_work: bool = False) -> Paper:
    raw_id = work.get("id", "")
    openalex_id = raw_id.split("/")[-1] if raw_id else ""
    return Paper(
        openalex_id=openalex_id,
        title=work.get("title", "") or "",
        abstract=abstract_from_inverted_index(work.get("abstract_inverted_index")),
        publication_year=work.get("publication_year"),
        cited_by_count=work.get("cited_by_count", 0) or 0,
        is_author_work=is_author_work,
    )


def fetch_author_works(
    client: OpenAlexClient,
    openalex_id: str,
    per_page: int = DEFAULT_PER_PAGE,
) -> list[Paper]:
    """Fetch all works for an author, paginating through the cursor API."""
    url = f"{OPENALEX_BASE}/works"
    params = {
        "filter": f"author.id:{openalex_id}",
        "per-page": per_page,
        "select": _select_fields(),
    }
    papers: list[Paper] = []
    cursor: str | None = "*"
    while cursor:
        params["cursor"] = cursor
        data = client._get(url, params)
        for work in data.get("results", []):
            papers.append(_work_to_paper(work, is_author_work=True))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    return papers


def fetch_citing_works(
    client: OpenAlexClient,
    work_id: str,
    per_page: int = DEFAULT_PER_PAGE,
) -> list[Paper]:
    """Fetch works that cite a given work (incoming citation edges)."""
    url = f"{OPENALEX_BASE}/works"
    params = {
        "filter": f"cites:{work_id}",
        "per-page": per_page,
        "select": _select_fields(),
    }
    papers: list[Paper] = []
    cursor: str | None = "*"
    while cursor:
        params["cursor"] = cursor
        data = client._get(url, params)
        for work in data.get("results", []):
            papers.append(_work_to_paper(work, is_author_work=False))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    return papers


def build_researcher_corpus(
    client: OpenAlexClient,
    researcher_id: str,
    max_citing_per_work: int | None = None,
) -> ResearcherCorpus:
    """Build a researcher subgraph: author's works + all citing works.

    Returns a corpus where each author work is flagged is_author_work=True and
    each citing work is is_author_work=False, with edges citing->cited.
    """
    corpus = ResearcherCorpus(researcher_id=researcher_id)

    author_works = fetch_author_works(client, researcher_id)
    for p in author_works:
        corpus.add_paper(p)

    for work in author_works:
        citing = fetch_citing_works(client, work.openalex_id)
        if max_citing_per_work is not None:
            citing = citing[:max_citing_per_work]
        for p in citing:
            corpus.add_paper(p)
            corpus.add_edge(p.openalex_id, work.openalex_id)

    return corpus


def save_corpus(corpus: ResearcherCorpus, path: str | Path) -> None:
    corpus.to_jsonl(path)


def load_corpus(researcher_id: str, path: str | Path) -> ResearcherCorpus:
    return ResearcherCorpus.from_jsonl(researcher_id, path)
