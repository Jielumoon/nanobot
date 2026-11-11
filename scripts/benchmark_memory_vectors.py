"""Benchmark the real SQLite vector scan without a provider or user workspace.

Run from the repository root: python -m scripts.benchmark_memory_vectors
Timing excludes setup, Markdown sync and API latency. tracemalloc reports peak
Python allocations during one query, not SQLite's native cache or total RSS.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import random
import statistics
import tempfile
import time
import tracemalloc
from array import array
from pathlib import Path

from nanobot.agent.memory_documents import Claim, MemoryDocuments, encode_claim, iso, utc_now
from nanobot.agent.memory_index import MemoryIndex


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("参数必须为正整数")
    return result


def benchmark(count: int, dimensions: int, repeats: int, limit: int) -> dict[str, object]:
    rng = random.Random(20261111)

    def vector() -> array[float]:
        # Positive samples exercise top-K selection on every row, not just misses.
        values = [rng.random() for _ in range(dimensions)]
        norm = math.sqrt(sum(value * value for value in values))
        return array("f", (value / norm for value in values))

    with tempfile.TemporaryDirectory(prefix="nanobot-vectors-") as directory:
        workspace = Path(directory)
        page = workspace / "memory/bank/facts.md"
        page.parent.mkdir(parents=True)
        with page.open("w", encoding="utf-8") as handle:
            for number in range(count):
                claim = Claim(
                    claim_id=f"c_{number:012x}", text=f"基准事实 {number}", kind="fact",
                )
                handle.write(encode_claim(claim) + "\n")
        index = MemoryIndex(MemoryDocuments(workspace))
        index.sync()
        fingerprint = json.dumps(["benchmark", "synthetic", dimensions])
        pending = index.pending_vectors(fingerprint, count)
        for start in range(0, count, 32):
            batch = pending[start:start + 32]
            index.save_vectors(fingerprint, batch, [vector() for _ in batch])
        del pending
        query = (fingerprint, vector())
        now = iso(utc_now())

        def search() -> int:
            return len(index.search(
                "", now=now, limit=limit, min_score=0, exclude=(), include_history=False,
                semantic=query, deduplicate=True,
            ))

        found = search()
        samples: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            search()
            samples.append((time.perf_counter() - start) * 1000)
        gc.collect()
        tracemalloc.start()
        search()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "python": platform.python_version(), "platform": platform.platform(),
            "rows": count, "dimensions": dimensions, "topK": limit, "results": found,
            "queryMedianMs": round(statistics.median(samples), 2),
            "queryMaxMs": round(max(samples), 2), "querySamplesMs": samples,
            "queryPythonPeakKiB": round(peak / 1024, 2),
            "rawVectorMiB": round(count * dimensions * array("f").itemsize / 1024 ** 2, 2),
            "databaseMiB": round(index.db_path.stat().st_size / 1024 ** 2, 2),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="测量记忆向量检索耗时及查询时 Python 分配峰值")
    parser.add_argument("--count", type=positive_int, default=10_000, help="向量条数")
    parser.add_argument("--dimensions", type=positive_int, default=512, help="向量维度")
    parser.add_argument("--repeats", type=positive_int, default=5, help="计时次数")
    parser.add_argument("--limit", type=positive_int, default=5, help="返回条数")
    args = parser.parse_args()
    print(json.dumps(
        benchmark(args.count, args.dimensions, args.repeats, args.limit),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
