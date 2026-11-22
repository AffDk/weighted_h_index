"""Relevance prompt templates for citation weighting."""

SYSTEM_PROMPT = (
    "You are an expert scientific reviewer evaluating citation relevance. "
    "Given a citing article abstract and a cited article abstract, estimate "
    "how heavily the citing article relies on the cited article."
)

USER_TEMPLATE = """Task: Rate citation relevance from 0.0 to 1.0.

1.0 = The citing article heavily relies on the cited article (core idea, method, or problem).
0.0 = The citation is marginal or courtesy (little relevance to the citing article's contribution).

Return ONLY valid JSON: {"weight": <float in [0,1]>, "reasoning": "<one sentence>"}

CITING ARTICLE ABSTRACT:
{citing_abstract}

CITED ARTICLE ABSTRACT:
{cited_abstract}
"""


def build_prompt(citing_abstract: str, cited_abstract: str) -> str:
    return USER_TEMPLATE.format(
        citing_abstract=citing_abstract,
        cited_abstract=cited_abstract,
    )


def clamp_weight(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
