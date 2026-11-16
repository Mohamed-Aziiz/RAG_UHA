"""Vérifie l'installation : config lisible, clés présentes, dossiers de données.

uv run python scripts/check_setup.py
uv run python scripts/check_setup.py --config configs/config.yaml --ping
"""

from __future__ import annotations

import argparse
import os
import sys

from rag_uha.config import load_settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--ping", action="store_true", help="fait un appel réel à Groq et Cloudflare"
    )
    args = parser.parse_args()

    s = load_settings(args.config)
    ok = True
    print(f"Config   : {s.config_path}")
    print(f"App      : {s.app.name} | pipeline par défaut : {s.pipeline}")
    for role in ("indexing", "query", "judge"):
        llm = s.llm(role)
        print(f"LLM {role:<8}: {llm.provider:<6} {llm.model}")
    print(f"Embeddings: {s.embeddings.provider} {s.embeddings.model} (dim {s.embeddings.dim})")

    # Clés : on vérifie seulement la présence, jamais on ne les affiche.
    env_names = {s.llm_indexing.api_key_env, s.llm_query.api_key_env, s.llm_judge.api_key_env}
    env_names |= {s.embeddings.account_id_env, s.embeddings.api_token_env}
    for name in sorted(n for n in env_names if n):
        present = bool(os.environ.get(name))
        ok &= present
        print(f"  {'OK ' if present else 'MANQUE'} {name}")

    # Dossiers de données (créés s'ils n'existent pas : ils sont ignorés par Git)
    for p in (
        s.paths.raw_dir,
        s.paths.chunks_file.parent,
        s.paths.chroma_dir.parent,
        s.paths.runs_dir,
        s.paths.results_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)
    print(f"Données  : {s.paths.raw_dir}")

    if args.ping:
        import httpx
        from openai import OpenAI

        llm = s.llm_query
        client = OpenAI(base_url=llm.base_url, api_key=llm.api_key)
        r = client.chat.completions.create(
            model=llm.model,
            max_tokens=20,
            temperature=0,
            messages=[{"role": "user", "content": "Réponds juste : OK"}],
            extra_body=llm.extra or None,
        )
        print(f"Groq     : {r.choices[0].message.content!r} ({r.usage.total_tokens} tokens)")

        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{s.embeddings.account_id}"
            f"/ai/run/{s.embeddings.model}"
        )
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {s.embeddings.api_token}"},
            json={"text": ["Université de Haute-Alsace", "University of Upper Alsace"]},
            timeout=60,
        )
        resp.raise_for_status()
        vectors = resp.json()["result"]["data"]
        print(f"Cloudflare: {len(vectors)} vecteurs de dimension {len(vectors[0])}")

    print("\nTout est en place." if ok else "\nIl manque des clés : complète .env puis relance.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
