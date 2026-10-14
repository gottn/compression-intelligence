"""File and cache I/O shared by the experiments. Every function here reads or writes disk."""
from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from cilab.pairs import model_slug
from cilab.text import cut_to_bytes


def load_corpora(data_dir: Path, names: Iterable[str], max_bytes: int | None) -> dict[str, str]:
    """{name: text} from data_dir/<name>.txt, each cut to max_bytes UTF-8 bytes if given."""
    return {n: cut_to_bytes((data_dir / f"{n}.txt").read_text(encoding="utf-8"), max_bytes) for n in names}


def hf_snapshot_dir(model_id: str) -> Path:
    """Local HF-cache snapshot folder of a downloaded model (no network)."""
    return Path(hf_hub_download(model_id, "tokenizer_config.json", local_files_only=True)).parent


def read_tokenizer_config(model_id: str) -> dict[str, object]:
    """The model's tokenizer_config.json from the HF cache, {} if the file is absent."""
    path = hf_snapshot_dir(model_id) / "tokenizer_config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def existing_keys(csv_path: Path, key_cols: Sequence[str]) -> set[tuple[str, ...]]:
    """Set of key tuples already present in a results CSV (empty if it does not exist)."""
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        return {tuple(r[k] for k in key_cols) for r in csv.DictReader(f)}


def append_rows(csv_path: Path, rows: Iterable[Mapping[str, object]], cols: Sequence[str]) -> None:
    """Append rows to a CSV with columns `cols`, writing the header if the file is new."""
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def token_bits_path(results_dir: Path, model_id: str, condition: str, corpus: str) -> Path:
    return results_dir / "tokens" / f"{model_slug(model_id)}__{condition}__{corpus}.npy"


def load_token_bits(results_dir: Path, model_id: str, condition: str, corpus: str) -> np.ndarray:
    """Saved per-token bits (stored float32) of one run, as float64."""
    return np.load(token_bits_path(results_dir, model_id, condition, corpus)).astype(np.float64)
