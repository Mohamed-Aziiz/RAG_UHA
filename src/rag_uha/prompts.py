"""Prompts communs aux deux pipelines.

Le RAG et le GraphRAG utilisent le MÊME prompt de réponse : seule la façon de construire le
contexte diffère. C'est une condition de la comparaison équitable.
"""

from __future__ import annotations

import re

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


_FR = {
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "du",
    "de",
    "et",
    "est",
    "sont",
    "je",
    "j",
    "on",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "comment",
    "combien",
    "pourquoi",
    "ou",
    "qui",
    "que",
    "qu",
    "quoi",
    "est-ce",
    "ce",
    "c",
    "mon",
    "ma",
    "mes",
    "il",
    "elle",
    "y",
    "a",
    "pour",
    "avec",
    "dans",
    "sur",
    "pas",
    "peut",
    "faut",
    "l",
    "d",
    "au",
    "aux",
}
_EN = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "do",
    "does",
    "did",
    "can",
    "could",
    "what",
    "which",
    "who",
    "how",
    "where",
    "when",
    "why",
    "i",
    "my",
    "you",
    "of",
    "and",
    "in",
    "to",
    "for",
    "with",
    "on",
    "at",
    "there",
    "if",
    "get",
    "any",
    "many",
    "much",
}


def question_language(question: str) -> str:
    """'fr' ou 'en', par comptage de mots-outils. Les accents tranchent en cas d'egalite.

    Le prompt systeme demande deja de repondre dans la langue de la question, mais
    gpt-oss l'ignore quand le contexte est surtout en francais (3 questions anglaises sur 8
    recevaient une reponse en francais). On detecte donc la langue dans le code et on
    l'impose en fin de prompt, la ou le modele obeit le mieux.
    """
    mots = re.findall(r"[a-zà-ÿ'-]+", question.lower())
    fr = sum(w in _FR for w in mots)
    en = sum(w in _EN for w in mots)
    if fr == en:
        return "fr" if re.search(r"[éèêàùçôîâ]", question.lower()) else "en"
    return "fr" if fr > en else "en"


def answer_prompt(question: str, context_block: str) -> str:
    langue = "French" if question_language(question) == "fr" else "English"
    return f"""---Context---
{context_block}

---Question---
{question}

---Answer---
(Write your answer in {langue}, whatever the language of the context.)"""


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
