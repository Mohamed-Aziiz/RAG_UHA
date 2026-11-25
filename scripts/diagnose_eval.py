"""Diagnostic question par question : POURQUOI une réponse est mal notée ?

    uv run python scripts/diagnose_eval.py --run rag
    uv run python scripts/diagnose_eval.py --run graphrag
    uv run python scripts/diagnose_eval.py --run rag --show q07     # détail d'une question

Pour chaque question répondable, on extrait de la réponse de référence ses « faits clés »
(adresses e-mail, numéros, nombres, années, montants, noms propres et mots rares), puis on
regarde où ils se trouvent :

- dans les passages récupérés ?  → sinon, la RECHERCHE a échoué ;
- dans la réponse générée ?      → sinon, la GÉNÉRATION a échoué (fait disponible mais non repris) ;
- si les faits sont dans la réponse mais que answer_correctness reste bas → c'est la MÉTRIQUE.

Le script mesure aussi l'effet des citations [doc_id] sur la partie « similarité » de
answer_correctness (25 % du score) : similarité réponse/référence avec et sans les citations.

Sortie : un tableau à l'écran + eval/results/diagnostic_<run>.csv.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from rag_uha.config import load_settings
from rag_uha.prompts import is_refusal

STOP = set(
    [
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "du",
        "de",
        "d",
        "l",
        "et",
        "ou",
        "en",
        "au",
        "aux",
        "a",
        "à",
        "pour",
        "par",
        "sur",
        "dans",
        "avec",
        "sans",
        "est",
        "sont",
        "ce",
        "cette",
        "ces",
        "qui",
        "que",
        "quoi",
        "dont",
        "il",
        "elle",
        "ils",
        "elles",
        "on",
        "vous",
        "nous",
        "je",
        "tu",
        "se",
        "sa",
        "son",
        "ses",
        "leur",
        "leurs",
        "plus",
        "moins",
        "tres",
        "très",
        "peut",
        "etre",
        "être",
        "avoir",
        "fait",
        "faire",
        "aussi",
        "ainsi",
        "comme",
        "mais",
        "donc",
        "car",
        "si",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "and",
        "or",
        "is",
        "are",
        "be",
        "by",
        "at",
        "from",
        "as",
        "it",
        "this",
        "that",
        "these",
        "those",
        "can",
        "you",
        "your",
        "their",
        "there",
        "which",
        "what",
        "who",
        "how",
        "when",
        "where",
        "yes",
        "non",
        "oui",
        "not",
        "no",
    ]
)
CITATION = re.compile(r"[\[【][^\]】\n]{3,200}[\]】]")


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t)


def key_facts(reference: str) -> list[str]:
    """Faits vérifiables de la référence : e-mails, téléphones, nombres, mots porteurs de sens."""
    facts: list[str] = []
    facts += re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", reference)
    facts += [
        re.sub(r"\D", "", p) for p in re.findall(r"(?:\+33\s?|0)\d(?:[\s.]?\d{2}){4}", reference)
    ]
    rest = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+|(?:\+33\s?|0)\d(?:[\s.]?\d{2}){4}", " ", reference)
    facts += re.findall(r"\b\d+(?:[.,]\d+)?\b", rest)
    for w in re.findall(r"[A-Za-zÀ-ÿ][\wÀ-ÿ'’-]{3,}", rest):
        n = norm(w).strip("'’-")
        if n not in STOP and len(n) >= 5:
            facts.append(w)
    seen, out = set(), []
    for f in facts:
        k = norm(f)
        if k and k not in seen:
            seen.add(k)
            out.append(f)
    return out


def present(fact: str, text_norm: str, digits: str) -> bool:
    f = norm(fact)
    if f.isdigit() and len(f) >= 9:  # téléphone : comparer les chiffres seuls
        return f in digits
    if len(f) >= 7 and not f.isdigit():  # mot long : tolérer pluriel / accord (racine)
        f = f[: max(6, len(f) - 2)]
    return f in text_norm


def coverage(facts: list[str], text: str) -> tuple[float, list[str]]:
    if not facts:
        return float("nan"), []
    tn, dg = norm(text), re.sub(r"\D", "", text)
    missing = [f for f in facts if not present(f, tn, dg)]
    return 1 - len(missing) / len(facts), missing


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--show", default=None, help="id d'une question à afficher en détail")
    p.add_argument("--no-embed", action="store_true", help="ne pas calculer l'effet des citations")
    args = p.parse_args()
    s = load_settings(args.config)

    runs = load_jsonl(s.paths.runs_dir / f"{args.run}.jsonl")
    scores = {r["id"]: r for r in load_jsonl(s.paths.runs_dir / f"{args.run}_ragas.jsonl")}
    if not runs:
        raise SystemExit(f"Run {args.run} introuvable dans {s.paths.runs_dir}")

    emb = None
    if not args.no_embed:
        from rag_uha.embeddings import get_embedder

        emb = get_embedder(s)

    rows = []
    for r in runs:
        q, a = r["question"], r["answer"]
        qid = q["id"]
        if not q.get("answerable", True):
            continue
        ctx_list = [c["chunk"] for c in sorted(a["contexts"], key=lambda c: c["rank"])]
        ctx_text = (
            "\n".join(c["text"] for c in ctx_list)
            + "\n"
            + a.get("extra", {}).get("graph_context", "")
        )
        docs = list(dict.fromkeys(c["doc_id"] for c in ctx_list))
        gold = q["gold_docs"]
        gold_rank = next((i for i, d in enumerate(docs, 1) if d in gold), None)
        facts = key_facts(q["reference_answer"])
        ctx_cov, ctx_missing = coverage(facts, ctx_text)
        ans_clean = CITATION.sub("", a["text"]).strip()
        ans_cov, ans_missing = coverage(facts, ans_clean)
        refused = is_refusal(a["text"])
        sc = scores.get(qid, {})
        corr = sc.get("answer_correctness", float("nan"))

        if refused:
            cause = (
                "refus à tort (info dans le contexte)"
                if ctx_cov >= 0.5
                else "refus : info absente du contexte"
            )
        elif ctx_cov < 0.5:
            cause = "RECHERCHE : info absente du contexte"
        elif ans_cov < 0.5:
            cause = "GÉNÉRATION : info dans le contexte, pas dans la réponse"
        elif corr == corr and corr < 0.6:
            cause = "MÉTRIQUE : réponse contient les faits mais notée bas"
        else:
            cause = "OK"

        rows.append(
            {
                "id": qid,
                "cat": "inter_langues" if q.get("cross_lingual") else q["type"],
                "lang": q["lang"],
                "gold_rank": gold_rank,
                "ctx_cov": round(ctx_cov, 2),
                "ans_cov": round(ans_cov, 2),
                "refus": refused,
                "mots_rep": len(ans_clean.split()),
                "mots_ref": len(q["reference_answer"].split()),
                "n_citations": len(CITATION.findall(a["text"])),
                "correctness": corr,
                "relevancy": sc.get("answer_relevancy", float("nan")),
                "faithfulness": sc.get("faithfulness", float("nan")),
                "cause": cause,
                "_answer": a["text"],
                "_answer_clean": ans_clean,
                "_reference": q["reference_answer"],
                "_question": q["question"],
                "_facts": facts,
                "_ctx_missing": ctx_missing,
                "_ans_missing": ans_missing,
                "_docs": docs,
                "_gold": gold,
            }
        )

    df = pd.DataFrame(rows)

    if emb is not None and len(df):
        ref_v = emb.embed_documents(df["_reference"].tolist())
        raw_v = emb.embed_documents(df["_answer"].tolist())
        cln_v = emb.embed_documents(df["_answer_clean"].tolist())
        df["sim_avec_citations"] = (ref_v * raw_v).sum(1).round(3)
        df["sim_sans_citations"] = (ref_v * cln_v).sum(1).round(3)

    if args.show:
        row = df[df.id == args.show]
        if row.empty:
            raise SystemExit(f"{args.show} : introuvable (ou question hors corpus)")
        r = row.iloc[0].to_dict()
        print(f"=== {r['id']} ({r['cat']}, {r['lang']}) — {r['_question']}")
        print(f"Référence : {r['_reference']}")
        print(f"Faits clés : {r['_facts']}")
        print(f"Documents attendus : {r['_gold']}")
        print(f"Documents récupérés : {r['_docs']}  (rang du bon : {r['gold_rank']})")
        print(f"Faits absents du CONTEXTE : {r['_ctx_missing']}")
        print(f"Faits absents de la RÉPONSE : {r['_ans_missing']}")
        print(f"\nRéponse :\n{r['_answer']}\n")
        print({k: r[k] for k in ("correctness", "relevancy", "faithfulness", "cause")})
        return

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    cols = [
        "id",
        "cat",
        "gold_rank",
        "ctx_cov",
        "ans_cov",
        "refus",
        "mots_rep",
        "mots_ref",
        "n_citations",
        "correctness",
        "relevancy",
        "cause",
    ]
    print(df[cols].to_string(index=False))

    print("\n=== Causes (questions répondables) ===")
    for cause, n in Counter(df["cause"]).most_common():
        sub = df[df.cause == cause]
        print(f"{n:>3}  {cause:<58} correctness moyen {sub['correctness'].mean():.2f}")

    print("\n=== Indicateurs ===")
    print(f"Bon document dans le contexte (Hit) : {df.gold_rank.notna().mean():.0%}")
    print(f"Faits de la référence présents dans le contexte : {df.ctx_cov.mean():.0%}")
    print(f"Faits de la référence présents dans la réponse  : {df.ans_cov.mean():.0%}")
    print(
        f"Longueur moyenne : réponse {df.mots_rep.mean():.0f} mots, référence {df.mots_ref.mean():.0f} mots"
    )
    print(f"Citations par réponse : {df.n_citations.mean():.1f}")
    if "sim_avec_citations" in df:
        print(
            f"Similarité réponse/référence : avec citations {df.sim_avec_citations.mean():.3f}, "
            f"sans citations {df.sim_sans_citations.mean():.3f}"
        )

    out = s.paths.results_dir / f"diagnostic_{args.run}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[
        [c for c in df.columns if not c.startswith("_")] + ["_question", "_reference", "_answer"]
    ].to_csv(out, index=False, encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
