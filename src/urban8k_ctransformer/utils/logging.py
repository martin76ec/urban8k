"""Logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import IO


class _TeeStream:
    """Mirror writes to a list of streams (used for stdout/stderr capture)."""

    def __init__(self, *streams: IO[str]) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

    def isatty(self) -> bool:
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)

    def fileno(self) -> int:
        # tqdm may call fileno(); delegate to the first real terminal stream.
        for s in self.streams:
            try:
                return s.fileno()
            except (OSError, ValueError):
                continue
        raise OSError("no underlying terminal stream")


def attach_file_log(path: str | Path) -> Path:
    """Mirror stdout and stderr to ``path`` (append mode) and return the path.

    Idempotent: calling twice with the same path does not add duplicate tees.
    The original streams remain active, so console output is preserved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.open("a", encoding="utf-8", buffering=1)

    if not isinstance(sys.stdout, _TeeStream):
        sys.stdout = _TeeStream(sys.stdout, file)
        sys.stderr = _TeeStream(sys.stderr, file)
    elif file not in sys.stdout.streams:
        stdout_tee: _TeeStream = sys.stdout
        stderr_tee: _TeeStream = sys.stderr  # type: ignore[assignment]
        stdout_tee.streams = (*stdout_tee.streams, file)
        stderr_tee.streams = (*stderr_tee.streams, file)
    return path


def get_logger(name: str = "urban8k", log_file: str | Path | None = None) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: logger name (hierarchical, e.g. ``urban8k.train``).
        log_file: optional path to a file that also receives every record.
            The file is appended to (not overwritten), so it survives resumes.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    if log_file is not None:
        log_path = Path(log_file)
        # Avoid duplicate file handlers for the same logger.
        already = any(
            isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).resolve() == log_path.resolve()
            for h in logger.handlers
        )
        if not already:
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            )
            logger.addHandler(file_handler)

    return logger
