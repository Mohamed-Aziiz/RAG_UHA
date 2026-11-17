"""Jeu de questions annotées (`eval/questions.jsonl`, une question par ligne)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from rag_uha.types import Answer, Lang


@dataclass
class EvalQuestion:
    id: str
    question: str
    lang: Lang
    type: Literal["factual", "multi_hop"]
    reference_answer: str
    gold_docs: list[str]  # doc_id attendus ; vide si hors corpus
    cross_lingual: bool = False  # question dans une langue, documents attendus dans l'autre
    answerable: bool = True  # False = hors corpus : la bonne réponse est un refus
    notes: str = ""

    @property
    def category(self) -> str:
        if not self.answerable:
            return "hors_corpus"
        if self.cross_lingual:
            return "inter_langues"
        return self.type


@dataclass
class RunRecord:
    question: EvalQuestion
    answer: Answer

    def to_dict(self) -> dict:
        return {"question": asdict(self.question), "answer": self.answer.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> RunRecord:
        return cls(EvalQuestion(**d["question"]), Answer.from_dict(d["answer"]))


def load_questions(path: Path) -> list[EvalQuestion]:
    qs: list[EvalQuestion] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.lstrip().startswith("//"):
                qs.append(EvalQuestion(**json.loads(line)))
    ids = [q.id for q in qs]
    if len(set(ids)) != len(ids):
        raise ValueError("Identifiants de questions en double")
    return qs


def validate_questions(qs: list[EvalQuestion], known_docs: set[str] | None = None) -> list[str]:
    """Renvoie la liste des problèmes (vide = OK)."""
    issues = []
    for q in qs:
        if q.answerable and not q.gold_docs:
            issues.append(f"{q.id}: answerable sans gold_docs")
        if not q.answerable and q.gold_docs:
            issues.append(f"{q.id}: hors corpus avec gold_docs")
        if not q.reference_answer.strip():
            issues.append(f"{q.id}: reference_answer vide")
        if known_docs is not None:
            for d in q.gold_docs:
                if d not in known_docs:
                    issues.append(f"{q.id}: gold_doc inconnu {d!r}")
    return issues


def summarize(qs: list[EvalQuestion]) -> dict:
    return {
        "n": len(qs),
        "by_category": dict(Counter(q.category for q in qs)),
        "by_lang": dict(Counter(q.lang for q in qs)),
        "multi_doc": sum(1 for q in qs if len(q.gold_docs) > 1),
    }
