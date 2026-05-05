from __future__ import annotations

import contextlib
import fcntl
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import RESULTS_ROOT, SHARED_MANIFEST_FILENAME, SUPPORTED_CATEGORIES

ANNOTATOR_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_annotator_id(annotator_id: str) -> str:
    annotator_id = annotator_id.strip()
    if not annotator_id:
        raise ValueError("annotator_id must not be empty.")
    if not ANNOTATOR_ID_RE.fullmatch(annotator_id):
        raise ValueError(
            "annotator_id must be filesystem-safe and contain only letters, digits, _ or -."
        )
    return annotator_id


def manifest_results_dir(category: str) -> Path:
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")
    return RESULTS_ROOT / category / Path(SHARED_MANIFEST_FILENAME).stem


def results_path_for_annotator(category: str, annotator_id: str) -> Path:
    safe_id = validate_annotator_id(annotator_id)
    return manifest_results_dir(category) / f"{safe_id}.jsonl"


def _lock_path_for_result(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


@contextlib.contextmanager
def _file_lock(lock_path: Path, *, exclusive: bool) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(lock_file.fileno(), mode)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_result_rows(category: str, annotator_id: str) -> list[dict[str, Any]]:
    path = results_path_for_annotator(category, annotator_id)
    lock_path = _lock_path_for_result(path)
    with _file_lock(lock_path, exclusive=False):
        if not path.exists():
            return []

        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
        return rows


def load_result_rows_for_categories(
    annotator_id: str,
    categories: Iterable[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category in categories:
        rows.extend(load_result_rows(category, annotator_id))
    return rows


def completed_item_ids(category: str, annotator_id: str) -> set[str]:
    return {str(row["item_id"]) for row in load_result_rows(category, annotator_id)}


def append_result_row(category: str, annotator_id: str, row: dict[str, Any]) -> Path:
    path = results_path_for_annotator(category, annotator_id)
    lock_path = _lock_path_for_result(path)
    with _file_lock(lock_path, exclusive=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return path


def available_result_paths(
    *,
    categories: Iterable[str] | None = None,
    annotator_id: str | None = None,
) -> list[Path]:
    selected_categories = list(categories) if categories is not None else list(SUPPORTED_CATEGORIES)
    out: list[Path] = []
    for category in selected_categories:
        directory = manifest_results_dir(category)
        if not directory.exists():
            continue
        if annotator_id is None:
            out.extend(sorted(directory.glob("*.jsonl")))
        else:
            path = results_path_for_annotator(category, annotator_id)
            if path.exists():
                out.append(path)
    return out


def reset_results_for_annotator(annotator_id: str, categories: Iterable[str]) -> list[Path]:
    removed: list[Path] = []
    for category in categories:
        result_path = results_path_for_annotator(category, annotator_id)
        lock_path = _lock_path_for_result(result_path)
        with _file_lock(lock_path, exclusive=True):
            if result_path.exists():
                result_path.unlink()
                removed.append(result_path)

            agreement_path = manifest_results_dir(category) / "agreement_summary.json"
            if agreement_path.exists():
                agreement_path.unlink()
                removed.append(agreement_path)

    return removed
