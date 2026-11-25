## Tableau comparatif principal

| axe | métrique | rag_v2 | graphrag_v2 |
|---|---|---|---|
| Pertinence | Answer relevancy (RAGAS) ↑ | 0.850 | 0.809 |
| Pertinence | Answer correctness vs référence (RAGAS) ↑ | 0.801 | 0.725 |
| Fiabilité | Faithfulness / fidélité aux sources (RAGAS) ↑ | 0.937 | 0.966 |
| Fiabilité | Refus corrects sur questions hors corpus ↑ | 1.000 | 1.000 |
| Fiabilité | Refus à tort sur questions répondables ↓ | 0.045 | 0.000 |
| Récupération | Hit@k (niveau document) ↑ | 0.955 | 1.000 |
| Récupération | MRR (niveau document) ↑ | 0.769 | 0.761 |
| Récupération | Rappel des documents attendus @k ↑ | 0.924 | 0.947 |
| Performances | Latence médiane par question (s) ↓ | 0.900 | 1.887 |
| Performances | Latence p95 (s) ↓ | 1.482 | 2.759 |
| Performances | Latence de recherche médiane (s) ↓ | 0.118 | 0.892 |
| Performances | Tokens LLM par question ↓ | 3,628 | 14,929 |
| Performances | Appels LLM par question ↓ | 1.000 | 2.000 |
| Performances | Coût estimé pour 1 000 questions ($) ↓ | 0.123 | 0.483 |


## Par type de question

| pipeline | category | n_questions | hit_at_k | mrr | faithfulness | answer_correctness | latency_median_s | tokens_mean |
|---|---|---|---|---|---|---|---|---|
| rag_v2 | factual | 11 | 1.000 | 0.864 | 0.984 | 0.850 | 0.900 | 3,666 |
| rag_v2 | multi_hop | 7.000 | 0.857 | 0.690 | 0.917 | 0.747 | 1.047 | 3,714 |
| rag_v2 | inter_langues | 4.000 | 1.000 | 0.646 | 0.833 | 0.739 | 0.770 | 3,422 |
| rag_v2 | hors_corpus | 3.000 | — | — | — | — | 0.514 | 3,564 |
| graphrag_v2 | factual | 11 | 1.000 | 0.833 | 0.960 | 0.719 | 1.887 | 15,093 |
| graphrag_v2 | multi_hop | 7.000 | 1.000 | 0.702 | 0.958 | 0.658 | 2.269 | 14,802 |
| graphrag_v2 | inter_langues | 4.000 | 1.000 | 0.667 | 1.000 | 0.844 | 2.059 | 14,490 |
| graphrag_v2 | hors_corpus | 3.000 | — | — | — | — | 1.356 | 15,205 |


## Par langue de la question

| pipeline | lang | n_questions | hit_at_k | mrr | faithfulness | answer_correctness | latency_median_s | tokens_mean |
|---|---|---|---|---|---|---|---|---|
| rag_v2 | fr | 16 | 0.929 | 0.756 | 0.926 | 0.796 | 0.885 | 3,614 |
| rag_v2 | en | 9.000 | 1.000 | 0.792 | 0.952 | 0.809 | 0.900 | 3,654 |
| graphrag_v2 | fr | 16 | 1.000 | 0.768 | 0.967 | 0.756 | 1.868 | 15,110 |
| graphrag_v2 | en | 9.000 | 1.000 | 0.750 | 0.964 | 0.680 | 1.924 | 14,606 |


## Coût d'indexation

| index | rag | graphrag |
|---|---|---|
| modèle LLM d'indexation | — (aucun) | openai/gpt-oss-120b |
| tokens LLM | 0 | 531345 |
| appels LLM | 0 | 142 |
| coût LLM estimé ($) | 0.000 | 0.040 |
| durée (s) | 7.600 | 895 |
| entités | — | 1129 |
| relations | — | 1110 |
| taille sur disque (Mo) | 4.300 | 26 |


## Figures

![fig_quality](fig_quality.png)
![fig_perf](fig_perf.png)
![fig_by_category](fig_by_category.png)


## Exemples où les pipelines divergent (à commenter)

### n07 (multi_hop, fr) — Je veux devenir prof d'anglais : quel master préparer à l'UHA, dans quelle faculté, et quand passe-t-on le concours ?

*Référence :* Le master MEEF (Métiers de l'enseignement, de l'éducation et de la formation) spécialité Anglais, à la FLSH. Le concours de recrutement se passe pendant le M2.

**rag_v2** (correctness 0.51, sources fr-formation-1-domaines-de-formation-diplomes-masters, plaquette-offre-de-formation-uha, fr-formation-1-domaines-de-formation-diplomes-droit-economie-gestion, fr-formation-1-domaines-de-formation-diplomes-arts-lettres-et-langues) :
> Le master à préparer est le **Master Anglais** proposé dans le domaine « Arts, lettres et civilisations », qui dépend de la **Faculté des Lettres, Langues et Sciences Humaines (FLSH)** de l’UHA[plaquette-offre-de-formation-uha].  
Le contexte ne précise pas la date ou le moment où le concours de professeur d’anglais doit être passé. Je ne trouve pas cette information dans les documents disponibles.

**graphrag_v2** (correctness 0.86, sources fr-formation-1-domaines-de-formation-diplomes-masters, plaquette-offre-de-formation-uha, fr-formation-1-domaines-de-formation-diplomes-arts-lettres-et-langues, fr-formation-1-domaines-de-formation-diplomes-metiers-de-l-enseignement) :
> Le master à préparer est le **Master MEEF – spécialité Anglais**, proposé par la **Faculté des Lettres, Langues et Sciences Humaines (FLSH)** de l’UHA ; le concours de recrutement des professeurs s’ouvre aux étudiants inscrits en **2ᵉ année de master (M2)** et se déroule pendant cette année [fr-formation-1-domaines-de-formation-diplomes-metiers-de-l-enseignement] .

### n23 (hors_corpus, fr) — Quel est le mot de passe du Wi-Fi de l'UHA ?

*Référence :* Je ne sais pas : les documents citent les réseaux (eduroam et UHA) mais ne donnent aucun mot de passe.

**rag_v2** (correctness 0.55, sources fr-formation-1-accompagnement-a-la-reussite-1-numerique, fr-formation-1-orientation-decouverte-des-formations-1-accompagnement-a, livret-de-rentree-etudiant-uha, infographie-elan) :
> Je ne trouve pas cette information dans les documents disponibles.

**graphrag_v2** (correctness 0.67, sources fr-formation-1-accompagnement-a-la-reussite-1-numerique, livret-de-rentree-etudiant-uha, fr-formation-1-accompagnement-a-la-reussite-1-langues-certifications-aut, infographie-elan) :
> Je ne trouve pas cette information dans les documents disponibles.

### n12 (factual, fr) — Depuis quand l'IUT de Mulhouse propose-t-il des formations trinationales, et avec quels partenaires ?

*Référence :* Depuis 1997, en partenariat avec la Berufsakademie de Lörrach et la Fachhochschule Beider Basel.

**rag_v2** (correctness 0.92, sources plaquette-uha-formations-bi-trinationales-1e4600, fr-formation-1-facultes-iut-ecoles-d-ingenieurs-iutm-institut-universita, fr-formation-1-formations-bi-trinationales-transfrontalieres) :
> L’IUT de Mulhouse propose des formations trinationales depuis 1997, en partenariat avec la Berufsakademie de Lörrach (Allemagne) et la Fachhochschule Beider Basel (Suisse)[fr-formation-1-facultes-iut-ecoles-d-ingenieurs-iutm-institut-universita].

**graphrag_v2** (correctness 0.92, sources plaquette-uha-formations-bi-trinationales-1e4600, fr-formation-1-accompagnement-a-la-reussite-1-elan-eveil-a-la-liberte-et, fr-formation-1-facultes-iut-ecoles-d-ingenieurs-iutm-institut-universita, fr-formation-1-accompagnement-a-la-reussite-1-learning-center) :
> L’IUT de Mulhouse propose des formations trinationales depuis 1997, en partenariat avec la Berufsakademie de Lörrach (Allemagne) et la Fachhochschule Beider Basel (Suisse)[fr-formation-1-facultes-iut-ecoles-d-ingenieurs-iutm-institut-universita::0].

### n10 (multi_hop, fr) — Si je pars un semestre dans une université partenaire, est-ce que mes notes comptent à l'UHA, et qui je contacte ?

*Référence :* Oui : les résultats obtenus dans l'université partenaire sont reconnus par l'UHA, transcrits en notes françaises, et les crédits ECTS sont attribués automatiquement. Contact : international@uha.fr.

**rag_v2** (correctness 0.83, sources fr-formation-1-accompagnement-a-la-reussite-1-mobilites-internationales, fr-formation-1-domaines-de-formation-diplomes-sciences-humaines-et-socia, livret-de-rentree-etudiant-uha) :
> Oui, les résultats d’examens obtenus dans l’université partenaire sont reconnus par l’UHA, transcrits en notes françaises et les crédits ECTS correspondants sont automatiquement attribués ; vous pouvez donc valider votre semestre à l’étranger [fr-formation-1-accompagnement-a-la-reussite-1-mobilites-internationales].  
Pour toute question, il faut contacter le service international à l’adresse international@uha.fr [fr-formation-1-accompagnement-a-la-reussite-1-mobilites-internationales].

**graphrag_v2** (correctness nan, sources fr-formation-1-accompagnement-a-la-reussite-1-mobilites-internationales, infographie-elan, livret-de-rentree-etudiant-uha, fr-formation-1-facultes-iut-ecoles-d-ingenieurs-iutc-institut-universita) :
> Oui, les notes obtenues à l’étranger sont reconnues, transcrites en notes françaises et les crédits ECTS sont attribués par l’UHA ; vous pouvez vous renseigner auprès du service international à l’adresse international@uha.fr [fr-formation-1-accompagnement-a-la-reussite-1-mobilites-internationales].

### n19 (inter_langues, fr) — Quels sont les départements de la FSESJ ?

*Référence :* Cinq départements : Droit ; Politique, économie et société ; Hautes études de management ; Histoire et patrimoine ; Information / communication.

**rag_v2** (correctness nan, sources fr-formation-1-facultes-iut-ecoles-d-ingenieurs-fsesj-faculte-des-scienc, fr-formation-1-facultes-iut-ecoles-d-ingenieurs-flsh-faculte-des-lettres, en-education-and-training-university-components, fr-formation-1-facultes-iut-ecoles-d-ingenieurs) :
> Les départements de la FSESJ sont : Droit ; Politique, Économie et Société ; Hautes Études de Management ; Histoire et Patrimoine ; et Information‑Communication [en-education-and-training-university-components].

**graphrag_v2** (correctness 0.96, sources fr-formation-1-facultes-iut-ecoles-d-ingenieurs-fsesj-faculte-des-scienc, livret-de-rentree-etudiant-uha, en-education-and-training-university-components, fr-formation-1-facultes-iut-ecoles-d-ingenieurs-iutc-institut-universita) :
> La FSESJ comprend cinq départements : le département de droit, le département de politique, économie et société, le département d’études supérieures de management, le département d’histoire et patrimoine, et le département d’information/communication [en-education-and-training-university-components].

### n22 (inter_langues, en) — How long does the RobotHub Académie project last, and how is it funded?

*Référence :* 38 months, from April 2023 to September 2026. It is co-funded by European INTERREG funds, with a total budget of 3.6 million euros, including 1.5 million from the ERDF (FEDER).

**rag_v2** (correctness nan, sources fr-formation-1-grands-projets-de-formation-robothub-academie, en-education-and-training-information-and-orientation, en-education-and-training-university-components, rapport-activite-demouha-2022-2026-1e4600) :
> The RobotHub Académie project runs for 38 months, from April 2023 to September 2026 [fr-formation-1-grands-projets-de-formation-robothub-academie]. It is co‑financed by European INTERREG funds with a total budget of €3.6 million, of which €1.5 million comes from the FEDER programme [fr-formation-1-grands-projets-de-formation-robothub-academie].

**graphrag_v2** (correctness 0.94, sources fr-formation-1-grands-projets-de-formation-robothub-academie, rapport-activite-demouha-2022-2026-1e4600, fr-formation-1-accompagnement-a-la-reussite-1, fr-formation-1-grands-projets-de-formation-demouha) :
> The RobotHub Académie project runs for 38 months, from April 2023 until September 2026 [fr-formation-1-grands-projets-de-formation-robothub-academie]. It is co‑financed by the European INTERREG programme with a total budget of €3.6 million, of which €1.5 million comes from the FEDER fund [fr-formation-1-grands-projets-de-formation-robothub-academie].
