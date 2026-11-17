"""Prompts communs aux deux pipelines.

Le RAG et le GraphRAG utilisent le MÊME prompt de réponse : seule la façon de construire le
contexte diffère. C'est une condition de la comparaison équitable.
"""

from __future__ import annotations

from rag_uha.types import RetrievedChunk

ANSWER_SYSTEM = """You are the documentation assistant of the Université de Haute-Alsace (UHA).
Answer the user's question using ONLY the information in the provided context.

Rules:
- Answer in the SAME LANGUAGE as the question (French question -> French answer, English question -> English answer), even if the context is in another language.
- After each factual statement, cite its source in square brackets using the document id shown in the context, e.g. [master-data-science].
- Be concise and factual. Do not add information that is not in the context.
- If the context does not contain the answer, say so explicitly in the language of the question (e.g. « Je ne trouve pas cette information dans les documents disponibles. » / "I cannot find this information in the available documents.") and do not guess."""

REFUSAL_MARKERS = (
    "je ne trouve pas",
    "ne figure pas",
    "pas d'information",
    "pas cette information",
    "aucune information",
    "i cannot find",
    "i can't find",
    "not find this information",
    "no information",
    "not contain",
    "does not mention",
    "ne mentionne pas",
)


def format_context(contexts: list[RetrievedChunk]) -> str:
    parts = []
    for rc in contexts:
        parts.append(
            f"[source: {rc.chunk.doc_id}] (chunk {rc.chunk.id}, lang={rc.chunk.lang})\n{rc.chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def answer_prompt(question: str, context_block: str) -> str:
    return f"""---Context---
{context_block}

---Question---
{question}

---Answer---"""


def is_refusal(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in REFUSAL_MARKERS)


# Guidance des types d'entités pour l'extraction LightRAG (langue pivot : anglais).
UHA_ENTITY_TYPES_GUIDANCE = """Classify each entity using one of the following types. If no type fits, use `Other`.
Always write entity names in their official form, with the acronym in parentheses when one exists
(e.g. "Université de Haute-Alsace (UHA)", "Faculté des Sciences et Techniques (FST)"), and use the
SAME name for the same entity whether the text is in French or in English.

- Formation: a degree program or training path (Licence, Master, BUT, Doctorat, DU...)
- Diplome: a diploma or qualification awarded
- Composante: a faculty, school, institute or department of the university (FST, IUT, ENSCM...)
- Laboratoire: a research laboratory or team
- Service: an administrative or support service (scolarité, relations internationales, bibliothèque...)
- Personne: a person or a role (président, directeur, responsable de formation...)
- Cours: a course, teaching unit (UE) or module
- Lieu: a campus, building, city or address
- Procedure: a rule, regulation, procedure or requirement (admission, inscription, examens...)
- Date: a date, deadline, period or academic calendar item
- Organisation: an external organisation (ministère, CROUS, entreprise, université partenaire...)"""
