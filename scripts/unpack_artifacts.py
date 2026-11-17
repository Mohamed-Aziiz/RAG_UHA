"""Récupère les artefacts zippés (Drive) dans data/ : rien à recalculer.

uv run python scripts/unpack_artifacts.py --from "G:/Mon Drive/RAG_UHA/artifacts" --version v1
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rag_uha.config import load_settings
from scripts.pack_artifacts import PARTS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="src", default="data/artifacts_zip")
    p.add_argument("--version", default="v1")
    p.add_argument("--only", nargs="+", default=list(PARTS))
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    s = load_settings(args.config)

    for name in args.only:
        zip_path = Path(args.src) / f"{name}_{args.version}.zip"
        if not zip_path.exists():
            print(f"(absent : {zip_path})")
            continue
        dest = PARTS[name](s)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(zip_path), str(dest))
        m = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
        warn = ""
        if m.get("embedding_model") != s.embeddings.model:
            warn = (
                f"  ATTENTION : embeddings {m.get('embedding_model')} ≠ config {s.embeddings.model}"
            )
        print(f"→ {dest}  ({m.get('version')}, commit {m.get('git_commit')}){warn}")


if __name__ == "__main__":
    main()
