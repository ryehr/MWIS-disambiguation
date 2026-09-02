"""Prompt sources for the English / Chinese / Japanese experiments.

The primary source is *parallel*: the same passages, professionally translated
into each language, so a cross-language comparison of embedding rate or KL
divergence is not confounded by topic.  FLORES itself is a gated repo on the
Hub; `facebook/belebele` is ungated and carries the FLORES passages verbatim in
its `flores_passage` field, aligned across languages by `link`.  488 passages
are parallel across English, Chinese and Japanese.

Wikipedia supplies unrelated, effectively unlimited prompts, used to build the
larger corpora the steganalysis detectors are trained on.

A passage is a statement, not an instruction, so it is wrapped in a per-language
instruction template: the passage sets the topic, the template makes the model
answer in that language.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LANGS", "TEMPLATES", "Prompt", "flores_prompts", "wikipedia_prompts", "SOURCES"]

LANGS = {
    "en": {"belebele": "eng_Latn", "wiki": "20231101.en"},
    "zh": {"belebele": "zho_Hans", "wiki": "20231101.zh"},
    "ja": {"belebele": "jpn_Jpan", "wiki": "20231101.ja"},
}

TEMPLATES = {
    "en": "Write a short paragraph about the following topic: {topic}",
    "zh": "请围绕以下主题写一段短文：{topic}",
    "ja": "次のトピックについて短い文章を書いてください：{topic}",
}


@dataclass
class Prompt:
    lang: str
    text: str        # the instruction handed to the model
    topic: str       # the underlying passage
    key: str         # identical across languages for parallel prompts


def _truncate(s: str, max_chars: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= max_chars else s[:max_chars].rstrip() + "…"


def flores_prompts(lang: str, n: int = 200, max_topic_chars: int = 300) -> list[Prompt]:
    """Parallel FLORES passages via belebele.  Equal `key` == equal content."""
    from datasets import load_dataset

    ds = load_dataset("facebook/belebele", LANGS[lang]["belebele"], split="test")
    seen: dict[str, str] = {}
    for row in ds:
        seen.setdefault(row["link"], row["flores_passage"].strip())
    out = []
    for link in sorted(seen)[:n]:          # sorted: same order in every language
        topic = _truncate(seen[link], max_topic_chars)
        out.append(Prompt(lang, TEMPLATES[lang].format(topic=topic), topic, f"flores-{link}"))
    return out


def wikipedia_prompts(lang: str, n: int = 200, min_body_chars: int = 200) -> list[Prompt]:
    """Streamed Wikipedia titles as topics.  Not parallel, but unlimited."""
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", LANGS[lang]["wiki"], split="train", streaming=True)
    out = []
    for row in ds:
        if len(out) >= n:
            break
        if len(row["text"].strip()) < min_body_chars:
            continue
        topic = row["title"].strip()
        out.append(Prompt(lang, TEMPLATES[lang].format(topic=topic), topic, f"wiki-{row['id']}"))
    del ds
    return out


SOURCES = {"flores": flores_prompts, "wikipedia": wikipedia_prompts}
