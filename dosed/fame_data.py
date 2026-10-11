from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_ROOT = ROOT / "data" / "processed_baselines" / "fame_unified_page_natural_channel_sampling"
EXPERIMENT_ROOT = Path(os.environ.get("FAME_EXPERIMENT_ROOT", DEFAULT_EXPERIMENT_ROOT)).resolve()
RUNS_ROOT = Path(os.environ.get("FAME_RUNS_ROOT", str(EXPERIMENT_ROOT / "runs"))).resolve()
DATA_ROOT = EXPERIMENT_ROOT / "seed200_continuous"
SPLIT_PATH = EXPERIMENT_ROOT / "splits" / "subject_splits.json"
TARGET_CHANNELS = ("F3-A2", "F4-A1", "C3-A2", "C4-A1")
CHANNEL_SLUGS = {channel: channel.replace("-", "").lower() for channel in TARGET_CHANNELS}
SLUG_CHANNELS = {slug: channel for channel, slug in CHANNEL_SLUGS.items()}
FS = 200
PAGE_SECONDS = 30
PAGE_SAMPLES = FS * PAGE_SECONDS
SLEEP_STAGES = {"N1", "N2", "N3", "REM"}


def record_id(subject_id: str, channel: str) -> str:
    return f"{CHANNEL_SLUGS[channel]}__{subject_id}"


def parse_record_id(value: str) -> tuple[str, str]:
    slug, subject_id = value.split("__", 1)
    return subject_id, SLUG_CHANNELS[slug]


def load_subject_split(split_seed: int, fold: int) -> dict[str, list[str]]:
    payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    run_id = f"seed{split_seed}_fold{fold}"
    row = next(item for item in payload["runs"] if item["run_id"] == run_id)
    return {name: list(row[name]) for name in ("train", "validation", "test")}


def expand_records(subject_ids: list[str]) -> list[str]:
    return [record_id(subject_id, channel) for channel in TARGET_CHANNELS for subject_id in subject_ids]


def subject_dir(subject_id: str) -> Path:
    return DATA_ROOT / subject_id


def signal_path(subject_id: str, channel: str) -> Path:
    return subject_dir(subject_id) / f"{channel.replace('-', '')}_200hz.npy"


def load_events(subject_id: str, task: str, channel: str) -> list[dict]:
    events = json.loads((subject_dir(subject_id) / "events.json").read_text(encoding="utf-8"))
    return [row for row in events if row["task"] == task and row["channel"] == channel]


def load_stage_labels(subject_id: str) -> np.ndarray:
    return np.load(subject_dir(subject_id) / "stage_labels.npy", mmap_mode="r")


def n2_pages(subject_id: str) -> np.ndarray:
    return np.load(subject_dir(subject_id) / "n2_pages.npy", mmap_mode="r")


def selected_stage_pages(subject_id: str, channel: str, stage_scope: str = "n2") -> np.ndarray:
    complete = complete_pages(subject_id, channel)
    if stage_scope == "n2":
        # Preserve the original FAME N2 training page selection exactly.
        pages = np.asarray(n2_pages(subject_id), dtype=np.int64)
        return pages[pages < complete]
    if stage_scope != "nrem_n1n2n3":
        raise ValueError(f"unknown stage scope {stage_scope!r}")
    labels = np.asarray(load_stage_labels(subject_id))
    pages = np.flatnonzero(np.isin(labels, ("N1", "N2", "N3"))).astype(np.int64)
    # SEED's context cannot be supplied at the first and last record pages.
    return pages[(pages > 0) & (pages < complete - 1)]


def complete_pages(subject_id: str, channel: str) -> int:
    signal = np.load(signal_path(subject_id, channel), mmap_mode="r")
    return min(int(signal.size // PAGE_SAMPLES), int(load_stage_labels(subject_id).size))


def page_overlaps_event(page_index: int, event: dict) -> bool:
    left = page_index * PAGE_SECONDS
    right = left + PAGE_SECONDS
    return float(event["end_seconds"]) > left and float(event["start_seconds"]) < right


def valid_n2_events(subject_id: str, task: str, channel: str) -> list[dict]:
    return valid_stage_events(subject_id, task, channel, "n2")


def valid_stage_events(subject_id: str, task: str, channel: str, stage_scope: str) -> list[dict]:
    allowed = set(int(value) for value in selected_stage_pages(subject_id, channel, stage_scope))
    result = []
    for event in load_events(subject_id, task, channel):
        start = float(event["start_seconds"])
        end = float(event["end_seconds"])
        duration = end - start
        overlap = 0.0
        first = max(0, int(np.floor(start / PAGE_SECONDS)))
        last = max(first, int(np.floor(max(start, end - 1e-12) / PAGE_SECONDS)))
        for page in range(first, last + 1):
            if page in allowed:
                overlap += max(0.0, min(end, (page + 1) * PAGE_SECONDS) - max(start, page * PAGE_SECONDS))
        if duration > 0 and overlap / duration >= 0.5:
            result.append(event)
    return result


def valid_stage_events_any_overlap(subject_id: str, task: str, channel: str, stage_scope: str) -> list[dict]:
    """Return events with any positive-duration overlap on a selected page.

    This is the V3 unified page-label rule. The legacy 50% rule above remains
    available so historical experiments can be reproduced unchanged.
    """
    allowed = set(int(value) for value in selected_stage_pages(subject_id, channel, stage_scope))
    result = []
    for event in load_events(subject_id, task, channel):
        start = float(event["start_seconds"])
        end = float(event["end_seconds"])
        if end <= start:
            continue
        first = max(0, int(np.floor(start / PAGE_SECONDS)))
        last = max(first, int(np.floor(np.nextafter(end, -np.inf) / PAGE_SECONDS)))
        if any(page in allowed and page_overlaps_event(page, event) for page in range(first, last + 1)):
            result.append(event)
    return result


def compute_seed_global_std(record_ids: list[str]) -> float:
    """Match SEED: sleep samples, record-wise absolute p99 removal, pooled variance."""
    total_samples = 0
    sum_x = 0.0
    sum_x2 = 0.0
    for value in record_ids:
        subject_id, channel = parse_record_id(value)
        signal = np.load(signal_path(subject_id, channel), mmap_mode="r")
        labels = load_stage_labels(subject_id)
        pages = np.flatnonzero(np.isin(labels, tuple(SLEEP_STAGES)))
        pages = pages[pages < signal.size // PAGE_SAMPLES]
        if pages.size == 0:
            continue
        sleep_signal = np.concatenate(
            [np.asarray(signal[page * PAGE_SAMPLES : (page + 1) * PAGE_SAMPLES], dtype=np.float64) for page in pages]
        )
        threshold = np.percentile(np.abs(sleep_signal), 99)
        retained = sleep_signal[np.abs(sleep_signal) <= threshold]
        total_samples += retained.size
        sum_x += float(np.sum(retained, dtype=np.float64))
        sum_x2 += float(np.sum(retained * retained, dtype=np.float64))
    if total_samples == 0:
        raise ValueError("no sleep samples available for global standard deviation")
    variance = sum_x2 / total_samples - (sum_x / total_samples) ** 2
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError(f"invalid global variance {variance}")
    return float(np.sqrt(variance))


