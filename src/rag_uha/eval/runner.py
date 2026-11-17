"""Exécute un pipeline sur le jeu de questions et écrit un JSONL (reprise si interrompu)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from rag_uha.eval.dataset import EvalQuestion, RunRecord
from rag_uha.pipelines.base import QAPipeline

log = logging.getLogger(__name__)


def load_run(path: Path) -> list[RunRecord]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [RunRecord.from_dict(json.loads(line)) for line in f if line.strip()]


def run_pipeline(
    pipeline: QAPipeline,
    questions: list[EvalQuestion],
    out_file: Path,
    pause_s: float = 0.0,
    max_retries: int = 2,
) -> list[RunRecord]:
    """Une ligne JSON par question ; les questions déjà présentes dans `out_file` sont sautées."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = {r.question.id: r for r in load_run(out_file)}
    records = list(done.values())
    todo = [q for q in questions if q.id not in done]
    log.info("%s : %d questions à traiter (%d déjà faites)", pipeline.name, len(todo), len(done))

    with out_file.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, start=1):
            for attempt in range(1, max_retries + 2):
                try:
                    ans = pipeline.answer(q.question)
                    break
                except Exception as exc:
                    if attempt > max_retries:
                        raise
                    log.warning("%s : échec (%s), nouvel essai dans 30s", q.id, type(exc).__name__)
                    time.sleep(30)
            rec = RunRecord(q, ans)
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            records.append(rec)
            log.info(
                "[%d/%d] %s  %.1fs  %d tokens  sources=%s",
                i,
                len(todo),
                q.id,
                ans.latency_s,
                ans.usage.total_tokens,
                ",".join(ans.sources[:3]),
            )
            if pause_s:
                time.sleep(pause_s)
    order = {q.id: k for k, q in enumerate(questions)}
    return sorted(records, key=lambda r: order.get(r.question.id, 10**6))
