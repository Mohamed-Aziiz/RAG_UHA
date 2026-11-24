"""Zippe les artefacts (chunks, index ChromaDB, graphe LightRAG, runs) pour le Drive, avec un manifest.

uv run python scripts/pack_artifacts.py --version v1 --out "G:/Mon Drive/RAG_UHA/artifacts"
uv run python scripts/pack_artifacts.py --version v1 --only graph
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from rag_uha.config import load_settings

PARTS = {
    "raw": lambda s: s.paths.raw_dir,  # documents sources : sans eux, impossible de re-decouper
    "chunks": lambda s: s.paths.chunks_file.parent,
    "index": lambda s: s.paths.chroma_dir,
    "graph": lambda s: s.paths.lightrag_dir,
    "runs": lambda s: s.paths.runs_dir,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1")
    p.add_argument("--out", default="data/artifacts_zip")
    p.add_argument("--only", nargs="+", default=list(PARTS))
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    s = load_settings(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"

    for name in args.only:
        src = PARTS[name](s)
        if not src.exists():
            print(f"(absent : {src})")
            continue
        manifest = {
            "artifact": name,
            "version": args.version,
            "git_commit": commit,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "embedding_model": s.embeddings.model,
            "indexing_model": s.llm_indexing.model,
            "chunking": s.chunking.model_dump(),
            "source_dir": str(src),
        }
        (src / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        target = out / f"{name}_{args.version}"
        if target.with_suffix(".zip").exists():
            raise SystemExit(
                f"{target}.zip existe déjà : incrémente --version, on n'écrase jamais un zip."
            )
        shutil.make_archive(str(target), "zip", root_dir=src)
        print(f"→ {target}.zip")


if __name__ == "__main__":
    main()
