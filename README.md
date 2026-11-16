# Assistant UHA — RAG classique vs GraphRAG

> Projet du module *[nom du module]* — Master 1 Data Science, Université de Haute-Alsace — novembre 2026.
> Auteur : Mohamed Aziz.

Assistant intelligent pour exploiter les ressources documentaires de l'UHA (formations, règlements,
vie étudiante), en français et en anglais. Deux architectures sont implémentées et comparées
expérimentalement : un **RAG classique** (LlamaIndex + ChromaDB), qui sert de référence, et un
**GraphRAG** (LightRAG, graphe de connaissances NetworkX).

<!-- Les sections marquées (à compléter) sont remplies au fil du projet. -->

## 1. Description et objectifs

(à compléter) Problème étudié, demande de l'encadrant, les 4 axes de comparaison :
pertinence des réponses, fiabilité (hallucinations, fidélité aux sources), qualité de la
récupération, performances (latence, tokens, temps et coût d'indexation).

## 2. Données

(à compléter) Provenance (pages du site de l'UHA, scraping), volume (documents, tokens, chunks),
répartition FR / EN et par thème, formats (`.txt`, `.md`, `.html`, `.pdf`), étapes de préparation.
La liste des documents est dans `data/raw/documents.csv` (sur le Drive).

## 3. Approche

(à compléter) Schéma d'architecture, socle commun aux deux pipelines (chunks, embeddings, LLM de
génération, prompt), RAG classique, GraphRAG.

## 4. Modèles utilisés

| Rôle | Modèle | Fournisseur | Pourquoi |
| --- | --- | --- | --- |
| Indexation du graphe | `openai/gpt-oss-20b` | Groq | gros volume, quota séparé |
| Génération des réponses | `openai/gpt-oss-120b` | Groq | qualité, commun aux deux pipelines |
| Juge (RAGAS) | `qwen/qwen3.8-27b` | Groq | différent du générateur |
| Embeddings | `@cf/baai/bge-m3` | Cloudflare Workers AI | multilingue FR / EN, API gratuite |

## 5. Installation (Windows, PowerShell)

Prérequis : [Python 3.11](https://www.python.org/downloads/), [uv](https://docs.astral.sh/uv/), Git.

```powershell
git clone https://github.com/Mohamed-Aziiz/RAG_UHA.git
cd RAG_UHA
uv python pin 3.11
uv sync                          # crée .venv et installe tout (sans PyTorch : embeddings par API)
uv run nbstripout --install      # nettoie les sorties des notebooks avant chaque commit
Copy-Item .env.example .env      # puis remplir GROQ_API_KEY, CF_ACCOUNT_ID, CF_API_TOKEN
uv run python scripts/check_setup.py          # vérifie la config et les clés
uv run python scripts/check_setup.py --ping   # fait un appel réel à Groq et à Cloudflare
```

Clés nécessaires (gratuites) : [Groq](https://console.groq.com/keys) et
[Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) (identifiant de compte + jeton API).

## 6. Lancer la démo

```powershell
# Récupérer les artefacts (index ChromaDB et graphe LightRAG) depuis le Drive, puis :
uv run streamlit run app/streamlit_app.py
```

(à compléter) Lien du Drive, `scripts/unpack_artifacts.py`.

## 7. Reproduire les expériences

```powershell
uv run python scripts/build_chunks.py          # data/raw -> data/chunks/chunks.jsonl
uv run python scripts/build_vector_index.py    # chunks -> ChromaDB
uv run python scripts/build_graph.py --limit 3 # test d'indexation du graphe sur 3 documents
uv run python scripts/build_graph.py           # indexation complète (reprise automatique)
uv run python scripts/run_eval.py --pipeline rag
uv run python scripts/run_eval.py --pipeline graphrag
uv run python scripts/judge.py                 # RAGAS
uv run python scripts/make_report.py           # eval/results/
```

(à compléter) Durée et coût attendus de chaque étape.

## 8. Protocole d'évaluation

(à compléter) 25 questions annotées (`eval/questions.jsonl`), métriques par axe, juge.

## 9. Résultats

(à compléter) Tableau comparatif, tableaux par type de question et par langue, coût d'indexation,
exemples commentés, statistiques du graphe.

## 10. Analyse et limites

(à compléter)

## 11. Pistes

(à compléter)

## 12. Structure du dépôt

```text
configs/config.yaml        toute la configuration (modèles, chemins, paramètres)
.env.example               clés API attendues (le .env réel n'est jamais commité)
src/rag_uha/               code de l'application
  config.py, types.py      socle : configuration et types partagés
  llm.py, embeddings.py    accès aux API (Groq / Ollama, Cloudflare)
  loaders.py, chunking.py  chargement des documents et découpage commun
  indexing/                construction des index (ChromaDB, LightRAG)
  pipelines/               RAGPipeline, GraphRAGPipeline (même interface)
  eval/                    runner, métriques, RAGAS, tableau comparatif
scripts/                   une étape = un script
app/                       interface Streamlit
notebooks/                 prototypage et figures (importent rag_uha)
eval/                      questions annotées et résultats légers
data/                      données et artefacts (ignoré par Git, sur le Drive)
```

## Références

- LightRAG : https://github.com/HKUDS/LightRAG
- LlamaIndex : https://docs.llamaindex.ai/
- RAGAS : https://docs.ragas.io/
- BGE-M3 : https://huggingface.co/BAAI/bge-m3
