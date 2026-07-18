"""Golden-set loaders and (later) the builder. M1 ships the JSONL reader."""

from rageval.datasets.golden import GoldenRecord, load_golden

__all__ = ["GoldenRecord", "load_golden"]
