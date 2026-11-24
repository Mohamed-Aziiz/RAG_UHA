# Passation du 23 novembre (nuit) — à lire avant de reprendre

## En bref

1. **La cause des mauvaises réponses était l'extraction des PDF**, pas la recherche. `pypdf` sortait le
   texte dans l'ordre du fichier, pas l'ordre de lecture : sur la plaquette, les titres « LICENCES
   PROFESSIONNELLES » et « MASTERS » ressortaient collés en tête de page, détachés de leurs listes. Le
   texte indexé ne disait plus qu'une formation était un master. Remplacé par **PyMuPDF** (`sort=True`).
2. Le RAG a maintenant une **recherche hybride** vectoriel + BM25 (entrelacement).
3. Le prompt **impose la langue de la question** (3 questions anglaises sur 8 recevaient une réponse en français).
4. **Le graphe LightRAG doit être reconstruit** sur les nouveaux chunks (voir « À faire »).
5. Les **deux runs sont à refaire** : le prompt a changé après le dernier run RAG, et le run GraphRAG
   date de l'ancien graphe.
6. **24 nov. : GraphRAG, `kg_chunk_pick_method: WEIGHT`.** Avec « uha » dans la question, le GraphRAG
   ne trouvait plus le master IMDS (voir « Le mot « UHA » » plus bas).

## Ce qui a changé

| Fichier | Changement | Pourquoi |
| --- | --- | --- |
| `src/rag_uha/loaders.py` | PDF lus avec PyMuPDF (`get_text(sort=True)`) au lieu de `pypdf.extract_text()` | ordre de lecture cassé (cf. ci-dessus). Le mode `layout` de pypdf remet l'ordre mais coupe les mots (« infor matique ») |
| `pyproject.toml`, `uv.lock` | `pymupdf` ajouté | idem — faire `uv sync` |
| `src/rag_uha/lexical.py` (nouveau) | BM25 Okapi maison + RRF | les termes rares et exacts (« ingénierie mathématique ») ne remontent pas par embeddings sur un chunk-liste |
| `src/rag_uha/pipelines/rag.py` | fusion vectoriel + BM25 ; reranker optionnel | `fusion: interleave`, `bm25_slots: 2` : 1er vectoriel, 2 meilleurs BM25, puis suite du vectoriel |
| `src/rag_uha/rerank.py` (nouveau) | reranker cross-encoder Cloudflare (`bge-reranker-base`) | **désactivé** : sur ce corpus il met en tête des passages hors sujet, et il n'est pas multilingue |
| `src/rag_uha/prompts.py` | `question_language()` + langue imposée en fin de prompt ; `import re` ajouté le 24 nov. | testé juste sur 32/32 questions. Commun aux deux pipelines. Sans l'import, les deux pipelines plantaient (`NameError`) |
| `configs/config.yaml`, `config.py`, `indexing/graph_index.py` | `graphrag.kg_chunk_pick_method: WEIGHT` (24 nov.) | voir « Le mot « UHA » ». Réglé à la requête, pas besoin de reconstruire le graphe |
| `scripts/diag_graphrag.py`, `diag_motscles.py`, `diag_passages.py` (nouveaux) | diagnostics étape par étape d'une requête LightRAG | reproduisent l'analyse du mot « UHA » ; aucun ou très peu d'appels LLM |
| `src/rag_uha/pipelines/graphrag.py` | repli des `sources` seulement si aucun passage | sinon chaque réponse listait 96 à 145 documents sur 162 |
| `src/rag_uha/viz.py` | HTML du sous-graphe écrit en UTF-8 | plantage `cp1252` sous Windows (le bouton « Afficher le sous-graphe ») |
| `configs/config.yaml` | `llm_query` / `llm_judge` sur OpenRouter ; prix OpenRouter ; `graphrag.max_total_tokens: 20000` ; bloc `rag` (hybride) | à 5000, LightRAG ne remontait **aucun** passage sur 23/25 questions |
| `eval/questions.jsonl` | les 25 questions de Lisa (champ `type` normalisé) | `cross_lingual` / `out_of_corpus` n'étaient pas des valeurs valides de `type` |
| `tests/` | fixture `questions_example.jsonl` ; reranker et BM25 coupés dans la fixture RAG | les tests dépendaient du fichier de questions réel ; tests hors ligne. **27 passed** |
| `scripts/pack_artifacts.py` | part `raw` ajoutée | les documents sources n'étaient ni sur GitHub ni dans les zips |
| `app/`, `.streamlit/` | démo refaite : thème sombre (bleu nuit), une couleur par méthode prise au logo (bleu = RAG, cyan = GraphRAG), Archivo + Literata, citations `[doc-id]` remplacées par des numéros renvoyant aux pages uha.fr, phrase de comparaison sous les réponses, résultats gardés en session | logo dans `app/assets/` : `logo_uha.svg` (couleurs d'origine, pour rapport/slides) et `logo_uha_sombre.svg` (affiché par l'appli). Pris sur uha.fr, sans le bloc République française |

## À faire, dans l'ordre

```powershell
uv sync
uv run pytest -q
# 1. Graphe sur les nouveaux chunks. Le cache LLM évite de repayer les 156 documents .md inchangés.
#    LightRAG reprend selon l'id des chunks : sans repartir d'un dossier vide, il garderait l'ancien texte.
ren data\artifacts\lightrag lightrag_v1_pypdf
mkdir data\artifacts\lightrag
copy data\artifacts\lightrag_v1_pypdf\kv_store_llm_response_cache.json data\artifacts\lightrag\
uv run python scripts/build_graph.py
# 2. Les deux runs
ren data\runs\rag.jsonl rag_v2b_prompt_v1.jsonl
ren data\runs\graphrag.jsonl graphrag_v1_pypdf.jsonl
uv run python scripts/run_eval.py --pipeline rag
uv run python scripts/run_eval.py --pipeline graphrag
```

Si le graphe a déjà été reconstruit pendant la nuit, `data\artifacts\lightrag\metadata.json` le dit
(`created_at`, `n_entities`). **Attention** : grâce au cache, le coût de ce rebuild sera sous-évalué ; le
coût d'indexation de référence reste celui du build complet : **1 026 186 tokens, 330 appels, 45,5 min**.

Ensuite : juge RAGAS, `make_report.py`, README §9-§10, slides.

**Le juge est suspect** : le test sur 2 questions a donné `answer_relevancy = 0.0` aux deux et des `NaN`.
Un 0,0 pile sur une réponse correcte est une panne, pas un score — à diagnostiquer avant de s'y fier.

## Le mot « UHA » (24 nov.)

Question test : « est ce que uha propose une formation ingenieurie mathematique et data science ».
Sans « uha » les deux pipelines répondent ; avec « uha » le RAG d'Aziz (vectoriel seul) et le GraphRAG
échouent, notre RAG hybride répond.

- **RAG vectoriel** : la question entière devient un vecteur ; « UHA », présent partout, le tire vers
  les pages génériques. **BM25 neutralise** ce mot (IDF quasi nul) : c'est pour ça que le RAG hybride répond.
- **GraphRAG, étape 1 (mots-clés)** : le LLM met **seulement `UHA`** dans les mots-clés bas niveau
  (recherche d'entités). L'entité UHA a 542 relations : 122 des 138 relations retenues la touchent.
- **GraphRAG, étape 2 (choix des passages)** : même avec les bons mots-clés, l'entité
  « parcours Ingénierie mathématique et data sciences » est trouvée mais son passage disparaît, car
  LightRAG classe les passages liés aux entités par similarité avec **la question brute**
  (`kg_chunk_pick_method: VECTOR`). Avec `WEIGHT` (ordre des entités) : passage du master au rang 1,
  et pas de régression sur la question sans « uha » (rangs 1 et 2). Détail : `diag_passages.py`.
- Piste non appliquée : faire passer les mots-clés présents dans tout le corpus en haut niveau
  (UHA en haut niveau + `WEIGHT` : 42 relations au lieu de 138, master aux rangs 1 et 5).
- **À valider sur les 25 questions** : `uv run python scripts/run_eval.py --pipeline graphrag --tag weight`.

**Autre défaut vu dans les logs** : `Embedding func: ... bound to a different event loop` à chaque
requête. `graphrag.py` crée une boucle asyncio par question ; LightRAG relance ses workers
d'embedding à chaque fois. Ça se rattrape, mais ça ajoute de la latence. À corriger (une boucle
persistante).

## Résultats provisoires (25 questions, chunks 1200/100)

| | hit@4 | MRR | factuelles | multi-sauts | inter-langues | refus à tort | tokens/q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAG de départ (pypdf, vectoriel) | 0,727 | 0,659 | 0,80 | 0,88 | 0,25* | 5 | 3 414 |
| RAG final (PyMuPDF + BM25) | **0,773** | 0,659 | **0,90** | **1,00** | 0,00 | 5 | 3 332 |
| GraphRAG (**ancien graphe**) | 0,864 | 0,674 | 0,90 | 0,88 | **0,75** | 3 | 13 514 |

\* q11 : bon document retrouvé, mais réponse « I cannot find » — un hit sans réponse.

Relecture manuelle du RAG final (22 répondables) : 13 justes, 3 partielles, 2 fausses, 4 refus.

## Constats pour le rapport

- **hit@4 au niveau document surestime** : bon document, mauvais passage. C'était la cause de 3 refus sur 5 (RAG) et 3 sur 3 (GraphRAG).
- **Inter-langues : 0 réponse correcte sur 4 pour le RAG, quelle que soit la config** ; 0,75 pour le GraphRAG. Le graphe est construit avec `summary_language: English` et la consigne d'un même nom d'entité en FR et en EN : il sert de pivot entre les langues. BM25 ne peut rien entre langues.
- **Budget de contexte du GraphRAG** : sans passage remonté sur 23/25 questions à 5 000 tokens, 7/25 à 10 000, 0/25 à 20 000 (`graphrag_bug5000`, `graphrag_ctx10000`, `graphrag`). Le RAG n'a pas cette dépendance.
- **Contrôle d'équité** : RAG à 12 passages au lieu de 4 (`rag_topk12`) → hit@4 inchangé pour 2,8× le coût. L'avance du GraphRAG ne vient pas de la taille de son contexte.
- **Non-déterminisme** : `rag_v3_bm25x1` et `rag_v3_bis` ont des contextes identiques et des refus différents (q03). OpenRouter route vers des fournisseurs différents ; ±1 refus est du bruit. Seul le hit@4 est fiable à l'unité.
- **Corpus périmé** : q06 répond « portes ouvertes le 10 février 2024 » (référence : 16 janvier 2027), q01 cite un guichet de 2022 — le scraping a ramené d'anciennes actualités.
- **Détection des refus par mots-clés** : q15 est comptée comme un refus alors que sa première moitié est juste.
- **Latence** : sur une question inédite, GraphRAG 6,8 s contre 0,5 s pour le RAG. Les ~1 s mesurées sur les runs viennent du cache LLM de LightRAG (questions déjà posées).

## Fichiers de `data/runs/`

| Fichier | Contenu |
| --- | --- |
| `rag_v1_pypdf.jsonl` | RAG de départ (pypdf, vectoriel seul, top_k 4) — **référence** |
| `rag_topk12.jsonl` | contrôle : 12 passages |
| `rag_rerank.jsonl` | reranker `bge-reranker-base` (anciens chunks) |
| `rag_v2_bm25x2.jsonl` | PyMuPDF + BM25, 2 places, ancien ordre d'entrelacement |
| `rag_v3_bm25x1.jsonl`, `rag_v3_bis.jsonl` | BM25 1 place ; mêmes contextes, réponses différentes (non-déterminisme) |
| `rag.jsonl` | RAG final, **avant** le correctif de langue — à refaire |
| `graphrag_bug5000.jsonl`, `graphrag_ctx10000.jsonl` | étude du budget de contexte |
| `graphrag.jsonl` | GraphRAG 20 000 tokens, **ancien graphe** — à refaire |
| `rag_ragas.jsonl` | test du juge sur 2 questions (suspect) |

Zips v2 (Drive) : `raw`, `chunks`, `index` — et `graph`, `runs` une fois le graphe reconstruit.
