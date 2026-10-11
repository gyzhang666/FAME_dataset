from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "FAME"
SOURCE = Path(os.environ.get("FAME_SOURCE_ROOT", str(SOURCE))).resolve()
EXPERIMENT_ROOT = Path(
    os.environ.get(
        "FAME_EXPERIMENT_ROOT",
        str(ROOT / "data" / "processed_baselines" / "fame_unified_page_natural_channel_sampling"),
    )
).resolve()
OUTPUT = EXPERIMENT_ROOT / "seed200_continuous"
REPORTS = EXPERIMENT_ROOT / "reports"
TARGET_FS = 200
LOW_HZ = 0.1
HIGH_HZ = 35.0
FILTER_ORDER = 3
TARGET_CHANNELS = ("F3-A2", "F4-A1", "C3-A2", "C4-A1")
EVENT_LABELS = {"Spindle": "spindle", "K-complex": "kcomplex"}
STAGE_LABELS = {"n1": "N1", "n2": "N2", "n3": "N3", "rem": "REM", "w": "W", "a": "A"}
SUBJECTS = np.asarray(
    [f"01-01-{index:04d}" for index in range(1, 21)]
    + [f"01-02-{index:04d}" for index in range(1, 21)]
)


def _read_text(handle, width: int) -> str:
    return handle.read(width).decode("latin-1", errors="replace").strip()


def read_edf_header(path: Path) -> dict:
    """Read EDF header fields needed for audit and signal scaling."""
    with path.open("rb") as handle:
        version, patient, recording = (_read_text(handle, 8), _read_text(handle, 80), _read_text(handle, 80))
        start_date, start_time = _read_text(handle, 8), _read_text(handle, 8)
        header_bytes = int(_read_text(handle, 8) or 0)
        reserved = _read_text(handle, 44)
        records, record_duration = int(_read_text(handle, 8) or -1), float(_read_text(handle, 8) or 0)
        signal_count = int(_read_text(handle, 4) or 0)

        def many(width):
            return [_read_text(handle, width) for _ in range(signal_count)]

        labels, transducers, units = many(16), many(80), many(8)
        physical_min, physical_max = [float(v) for v in many(8)], [float(v) for v in many(8)]
        digital_min, digital_max = [int(v) for v in many(8)], [int(v) for v in many(8)]
        prefilters, samples_per_record = many(80), [int(v) for v in many(8)]
        _read_text(handle, 32 * signal_count)
    channels = []
    for i, label in enumerate(labels):
        samples = samples_per_record[i]
        channels.append({"label": label, "transducer": transducers[i], "unit": units[i],
                         "physical_min": physical_min[i], "physical_max": physical_max[i],
                         "digital_min": digital_min[i], "digital_max": digital_max[i],
                         "prefilter": prefilters[i], "samples_per_record": samples,
                         "sampling_rate": samples / record_duration if record_duration else None})
    return {"version": version, "patient": patient, "recording": recording, "start_date": start_date,
            "start_time": start_time, "header_bytes": header_bytes, "reserved": reserved,
            "data_records": records, "record_duration": record_duration,
            "duration_seconds": records * record_duration if records >= 0 else None,
            "signal_count": signal_count, "channels": channels}


class EDFSignalReader:
    """Memory-mapped, read-only signal access for ordinary int16 EDF files."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.header = read_edf_header(self.path)
        if self.path.suffix.casefold() != ".edf" or self.header["data_records"] < 0:
            raise ValueError("Only ordinary EDF with a known data-record count is supported")
        self.samples_per_record = [int(ch["samples_per_record"]) for ch in self.header["channels"]]
        self.total_samples_per_record = sum(self.samples_per_record)
        expected = self.header["header_bytes"] + self.header["data_records"] * self.total_samples_per_record * 2
        if self.path.stat().st_size < expected:
            raise ValueError(f"EDF is truncated: expected at least {expected} bytes")
        self._records = np.memmap(self.path, dtype="<i2", mode="r", offset=self.header["header_bytes"],
                                  shape=(self.header["data_records"], self.total_samples_per_record))

    @property
    def labels(self):
        return [ch["label"] for ch in self.header["channels"]]

    def read_digital(self, raw_label: str) -> np.ndarray:
        try:
            index = self.labels.index(raw_label)
        except ValueError as exc:
            raise KeyError(f"Channel {raw_label!r} not found in {self.path.name}") from exc
        offset, count = sum(self.samples_per_record[:index]), self.samples_per_record[index]
        return np.asarray(self._records[:, offset:offset + count]).reshape(-1).copy()

    def read_physical(self, raw_label: str, dtype=np.float64) -> tuple[np.ndarray, dict]:
        index = self.labels.index(raw_label)
        channel = self.header["channels"][index]
        digital = self.read_digital(raw_label)
        dmin, dmax = channel["digital_min"], channel["digital_max"]
        pmin, pmax = channel["physical_min"], channel["physical_max"]
        if dmax == dmin:
            raise ValueError(f"Invalid digital range for {raw_label}")
        physical = (digital.astype(dtype) - dmin) * ((pmax - pmin) / (dmax - dmin)) + pmin
        quality = {"digital_min_count": int(np.count_nonzero(digital == dmin)),
                   "digital_max_count": int(np.count_nonzero(digital == dmax)),
                   "rail_fraction": float(np.mean((digital == dmin) | (digital == dmax))),
                   "digital_min": dmin, "digital_max": dmax, "physical_min": pmin,
                   "physical_max": pmax, "unit": channel["unit"],
                   "sampling_rate": channel["sampling_rate"], "sample_count": int(digital.size)}
        return physical, quality


def _canonical_channel(value: str) -> str | None:
    compact = value.strip().upper().replace("EEG", "").replace(" ", "").replace("_", "").replace("-", "")
    aliases = {"F3A2": "F3-A2", "F4A1": "F4-A1", "C3A2": "C3-A2", "C4A1": "C4-A1",
               "F3M2": "F3-A2", "F4M1": "F4-A1", "C3M2": "C3-A2", "C4M1": "C4-A1"}
    return aliases.get(compact)


def _subject_paths(subject_id: str) -> tuple[Path, Path, Path]:
    group = "part1" if subject_id.startswith("01-01-") else "part2"
    return (SOURCE / group / f"{subject_id} PSG.edf", SOURCE / group / f"{subject_id}-annotation.csv",
            SOURCE / "state" / f"{subject_id}-state.csv")


def _audit_stage_datetime(value: str) -> tuple[datetime, str]:
    for pattern, precision in (("%Y/%m/%d %H:%M", "minute"), ("%d/%m/%Y %H:%M:%S", "second"),
                               ("%Y/%m/%d %H:%M:%S", "second")):
        try:
            return datetime.strptime(value, pattern), precision
        except ValueError:
            pass
    raise ValueError(f"unsupported staging datetime {value!r}")


def _audit_annotations(path: Path, duration_seconds: float) -> tuple[Counter, Counter, list[str]]:
    task_counts, channel_counts, errors = Counter(), Counter(), []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != ["channel", "start_time", "stop_time", "label"]:
            return task_counts, channel_counts, [f"unexpected annotation columns: {reader.fieldnames}"]
        seen = set()
        for row_number, row in enumerate(reader, start=2):
            try:
                channel = _canonical_channel(row["channel"])
                task = EVENT_LABELS.get(row["label"].strip())
                start, stop = float(row["start_time"]), float(row["stop_time"])
                if channel is None or task is None or not (0 <= start < stop <= duration_seconds + 1e-6):
                    raise ValueError(f"invalid event: {row}")
                key = (channel, task, start, stop)
                if key in seen:
                    raise ValueError("duplicate event")
                seen.add(key)
                task_counts[task] += 1
                channel_counts[f"{channel}:{task}"] += 1
            except Exception as exc:
                errors.append(f"row {row_number}: {type(exc).__name__}: {exc}")
    return task_counts, channel_counts, errors


def _audit_stages(path: Path, edf_start: datetime, duration_seconds: float):
    errors = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 6:
        return Counter(), 0, 0, 0.0, None, ["staging file has fewer than six rows"]
    declarations = [r for r in rows if r[:2] == ["Signal Type", "Discret"]]
    rates = [r for r in rows if len(r) >= 2 and r[0] == "Rate [s]"]
    if len(declarations) != 1:
        errors.append(f"expected one Signal Type,Discret row, found {len(declarations)}")
    if len(rates) != 1 or rates[0][1] != "30":
        errors.append(f"staging rate is not exactly one 30-second declaration: {rates}")
    try:
        header = next(i for i, row in enumerate(rows) if row[:2] == ["Time", "Value"])
    except StopIteration:
        return Counter(), 0, 0, 0.0, None, errors + ["missing Time,Value staging header"]
    data_rows = [row for row in rows[header + 1:] if any(cell.strip() for cell in row)]
    usable = min(len(data_rows), int(duration_seconds // 30))
    counts, first, precision0, offset = Counter(), None, None, None
    for index, row in enumerate(data_rows):
        if len(row) < 2:
            errors.append(f"stage row {header + index + 2}: too few columns")
            continue
        stage = STAGE_LABELS.get(row[1].strip().casefold())
        if stage is None:
            errors.append(f"unknown stage {row[1]!r}")
        elif index < usable:
            counts[stage] += 1
        try:
            observed, precision = _audit_stage_datetime(row[0].strip())
            if first is None:
                first, precision0 = observed, precision
                if precision == "second":
                    offset = (observed - edf_start).total_seconds()
                    if abs(offset) >= 60:
                        errors.append(f"staging start differs from EDF start by {offset} seconds")
            elif precision != precision0:
                errors.append("staging datetime precision changes within file")
            expected = (first + timedelta(seconds=30 * index) if precision0 == "second"
                        else (edf_start + timedelta(seconds=30 * index)).replace(second=0, microsecond=0))
            if observed != expected:
                errors.append(f"stage row {header + index + 2}: timestamp cadence mismatch")
        except ValueError as exc:
            errors.append(f"stage row {header + index + 2}: {exc}")
    return counts, len(data_rows), usable, usable * 30.0, offset, errors


def _build_subject_splits() -> dict:
    """Create the fixed three-seed, five-fold subject split manifest."""
    rows, runs = [], []
    for seed in range(3):
        tested, validated = [], []
        permuted = np.random.RandomState(seed=seed).permutation(SUBJECTS)
        folds = [np.sort(permuted[index * 8:(index + 1) * 8]) for index in range(5)]
        for fold in range(5):
            test = folds[fold].tolist()
            validation = folds[(fold + 1) % 5].tolist()
            train = np.sort(np.asarray([s for s in SUBJECTS if s not in set(test + validation)])).tolist()
            assert len(train) == 24 and len(validation) == 8 and len(test) == 8
            assert not (set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test))
            run_id = f"seed{seed}_fold{fold}"
            runs.append({"run_id": run_id, "split_seed": seed, "fold": fold,
                         "train": train, "validation": validation, "test": test})
            for subset, ids in (("train", train), ("validation", validation), ("test", test)):
                rows.extend({"run_id": run_id, "split_seed": seed, "fold": fold,
                             "subset": subset, "subject_id": subject} for subject in ids)
            tested.extend(test)
            validated.extend(validation)
        assert sorted(tested) == sorted(SUBJECTS.tolist())
        assert sorted(validated) == sorted(SUBJECTS.tolist())
    split_root = EXPERIMENT_ROOT / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "PASS", "experiment_id": "fame_unified_page_natural_channel_sampling",
        "subject_count": 40, "split_seeds": 3, "folds_per_seed": 5,
        "runs": runs, "invariants": {"subject_level_only": True,
        "train_validation_test_counts": [24, 8, 8],
        "each_subject_tested_once_per_seed": True,
        "each_subject_validated_once_per_seed": True,
        "channels_expanded_only_after_subject_split": True},
    }
    split_json = split_root / "subject_splits.json"
    if split_json.is_file() and json.loads(split_json.read_text(encoding="utf-8")) != summary:
        raise RuntimeError(f"existing split manifest differs from deterministic splits: {split_json}")
    if not split_json.exists():
        _json(split_json, summary)
    split_csv = split_root / "subject_splits.csv"
    if not split_csv.exists():
        with split_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run_id", "split_seed", "fold", "subset", "subject_id"])
            writer.writeheader()
            writer.writerows(rows)
    return summary


def run_audit(compute_hashes: bool = False) -> dict:
    report_root = EXPERIMENT_ROOT / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    records, errors, expected_paths = [], [], set()
    for subject_id in SUBJECTS.tolist():
        edf_path, annotation_path, staging_path = _subject_paths(subject_id)
        expected_paths.update((edf_path, annotation_path, staging_path))
        missing = [str(p.resolve()) for p in (edf_path, annotation_path, staging_path) if not p.is_file()]
        if missing:
            errors.extend({"subject_id": subject_id, "error": f"missing file: {p}"} for p in missing)
            continue
        try:
            header = read_edf_header(edf_path)
            start = datetime.strptime(f"{header['start_date']} {header['start_time']}", "%d.%m.%y %H.%M.%S")
            duration = float(header["duration_seconds"])
            mapped = {}
            for channel in header["channels"]:
                canonical = _canonical_channel(channel["label"])
                if canonical:
                    if canonical in mapped:
                        raise ValueError(f"duplicate target channel alias {canonical}")
                    mapped[canonical] = channel
            if set(mapped) != set(TARGET_CHANNELS):
                raise ValueError(f"target channels mismatch: {sorted(mapped)}")
            task_counts, channel_counts, event_errors = _audit_annotations(annotation_path, duration)
            stages, stage_rows, usable, coverage, offset, stage_errors = _audit_stages(staging_path, start, duration)
            subject_errors = event_errors + stage_errors
            errors.extend({"subject_id": subject_id, "error": error} for error in subject_errors)
            records.append({"subject_id": subject_id, "edf_path": str(edf_path.resolve()),
                            "annotation_path": str(annotation_path.resolve()), "staging_path": str(staging_path.resolve()),
                            "edf_start": start.isoformat(), "edf_duration_seconds": duration,
                            "target_sampling_rates_hz": sorted({float(mapped[c]["sampling_rate"]) for c in TARGET_CHANNELS}),
                            "target_units": sorted({str(mapped[c]["unit"]).replace("µ", "u") for c in TARGET_CHANNELS}),
                            "annotation_task_counts": dict(sorted(task_counts.items())),
                            "annotation_channel_task_counts": dict(sorted(channel_counts.items())),
                            "stage_counts": dict(sorted(stages.items())), "stage_row_count": stage_rows,
                            "usable_complete_stage_row_count": usable,
                            "discarded_stage_rows_outside_edf": stage_rows - usable,
                            "stage_coverage_seconds": coverage,
                            "staging_timestamp_start_offset_seconds": offset,
                            "unscored_tail_seconds": duration - coverage,
                            "status": "PASS" if not subject_errors else "FAIL",
                            "source_sha256": {"edf": _sha256(edf_path), "annotation": _sha256(annotation_path),
                                              "staging": _sha256(staging_path)} if compute_hashes else None})
        except Exception as exc:
            errors.append({"subject_id": subject_id, "error": f"{type(exc).__name__}: {exc}"})
    discovered = {p for p in SOURCE.rglob("*") if p.is_file()}
    unexpected = sorted(str(p.resolve()) for p in discovered - expected_paths)
    errors.extend({"subject_id": "DATASET", "error": f"unexpected file: {p}"} for p in unexpected)
    result = {"experiment_id": "fame_unified_page_natural_channel_sampling",
              "source_root": str(SOURCE.resolve()), "source_mutated": False, "expected_subjects": 40,
              "audited_subjects": len(records), "passed_subjects": sum(x["status"] == "PASS" for x in records),
              "failed_subjects": sum(x["status"] != "PASS" for x in records), "unexpected_files": unexpected,
              "hashes_computed": compute_hashes,
              "status": "PASS" if len(records) == 40 and not errors else "FAIL", "records": records, "errors": errors}
    _json(report_root / "fame_upload_audit.json", result)
    with (report_root / "fame_upload_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["subject_id", "status", "edf_start", "edf_duration_seconds", "target_sampling_rates_hz",
                  "target_units", "stage_row_count", "usable_complete_stage_row_count",
                  "discarded_stage_rows_outside_edf", "stage_coverage_seconds",
                  "staging_timestamp_start_offset_seconds", "unscored_tail_seconds", "annotation_task_counts", "stage_counts"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return result


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _processing_version() -> str:
    payload = {
        "experiment_id": "fame_unified_page_natural_channel_sampling",
        "target_fs": TARGET_FS,
        "bandpass_hz": [LOW_HZ, HIGH_HZ],
        "filter_order": FILTER_ORDER,
        "filter_implementation": "scipy.signal.butter_ba_plus_filtfilt_default_padding",
        "resampling": "scipy.signal.resample_poly",
        "stage_alignment": "row_index_times_30_seconds_from_edf_sample_zero",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _raw_channel_map(reader: EDFSignalReader) -> dict[str, str]:
    result = {}
    for raw in reader.labels:
        canonical = _canonical_channel(raw)
        if canonical is None:
            continue
        if canonical in result:
            raise ValueError(f"duplicate target alias {canonical}: {result[canonical]!r}, {raw!r}")
        result[canonical] = raw
    if set(result) != set(TARGET_CHANNELS):
        raise ValueError(f"target channel mismatch: {sorted(result)}")
    return result


def _to_uv(signal: np.ndarray, unit: str) -> np.ndarray:
    normalized = unit.strip().replace("µ", "u").casefold()
    factors = {"uv": 1.0, "mv": 1_000.0, "v": 1_000_000.0}
    if normalized not in factors:
        raise ValueError(f"unsupported EEG unit {unit!r}")
    return np.asarray(signal, dtype=np.float64) * factors[normalized]


def _stats(signal: np.ndarray) -> dict:
    finite = np.isfinite(signal)
    if not np.all(finite):
        return {"sample_count": int(signal.size), "finite_fraction": float(np.mean(finite))}
    absolute = np.abs(signal)
    differences = np.diff(signal)
    return {
        "sample_count": int(signal.size),
        "finite_fraction": 1.0,
        "mean_uv": float(np.mean(signal)),
        "std_uv": float(np.std(signal)),
        "min_uv": float(np.min(signal)),
        "max_uv": float(np.max(signal)),
        "abs_p99_uv": float(np.percentile(absolute, 99)),
        "abs_p999_uv": float(np.percentile(absolute, 99.9)),
        "zero_difference_fraction": float(np.mean(differences == 0)) if differences.size else 1.0,
    }


def _seed_preprocess(signal_uv: np.ndarray, source_fs: float) -> np.ndarray:
    rounded_fs = int(round(source_fs))
    if not math.isclose(source_fs, rounded_fs, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"non-integer source sampling rate {source_fs}; the SEED MASS interpolation branch "
            "must be reviewed before processing this record"
        )
    if not (0 < LOW_HZ < HIGH_HZ < rounded_fs / 2):
        raise ValueError(f"invalid bandpass for source fs {source_fs}")
    b, a = butter(FILTER_ORDER, (LOW_HZ / (0.5 * rounded_fs), HIGH_HZ / (0.5 * rounded_fs)), btype="band")
    filtered = filtfilt(b, a, signal_uv)
    if rounded_fs == TARGET_FS:
        output = filtered
    else:
        ratio = Fraction(TARGET_FS, rounded_fs)
        output = resample_poly(filtered, ratio.numerator, ratio.denominator)
    if not np.all(np.isfinite(output)):
        raise ValueError("non-finite value after filtering/resampling")
    return np.asarray(output, dtype=np.float32)


def _events(path: Path, duration_seconds: float) -> list[dict]:
    result = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != ["channel", "start_time", "stop_time", "label"]:
            raise ValueError(f"unexpected annotation columns: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            channel = _canonical_channel(row["channel"])
            task = EVENT_LABELS.get(row["label"].strip())
            start = float(row["start_time"])
            end = float(row["stop_time"])
            if channel is None or task is None or not (0 <= start < end <= duration_seconds + 1e-6):
                raise ValueError(f"invalid event at {path.name}:{row_number}: {row}")
            result.append(
                {
                    "event_id": f"{path.stem}:{row_number}",
                    "channel": channel,
                    "task": task,
                    "raw_label": row["label"].strip(),
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": end - start,
                    "confidence": 1.0,
                    "source_row": row_number,
                }
            )
    return result


def _stages(path: Path, duration_seconds: float) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header_index = next(index for index, row in enumerate(rows) if row[:2] == ["Time", "Value"])
    data_rows = [row for row in rows[header_index + 1 :] if any(cell.strip() for cell in row)]
    usable_rows = min(len(data_rows), int(duration_seconds // 30))
    result = []
    for index, row in enumerate(data_rows[:usable_rows]):
        stage = STAGE_LABELS.get(row[1].strip().casefold())
        if stage is None:
            raise ValueError(f"unknown stage at {path.name}:{header_index + index + 2}: {row[1]!r}")
        result.append(
            {
                "page_index": index,
                "start_seconds": index * 30.0,
                "end_seconds": (index + 1) * 30.0,
                "stage": stage,
                "source_time": row[0].strip(),
                "source_row": header_index + index + 2,
            }
        )
    return result


def _write_stage_csv(path: Path, stages: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["page_index", "start_seconds", "end_seconds", "stage", "source_time", "source_row"],
        )
        writer.writeheader()
        writer.writerows(stages)


def run_prepare(limit: int | None = None, check_only: bool = False,
                audit: dict | None = None) -> dict:
    if audit is None:
        audit = run_audit(compute_hashes=True)
    if audit.get("status") != "PASS" or not audit.get("hashes_computed"):
        raise ValueError("the frozen FAME audit is not PASS with source hashes")
    audit_by_subject = {row["subject_id"]: row for row in audit["records"]}
    subjects = SUBJECTS.tolist()
    if limit is not None:
        subjects = subjects[:limit]
    estimate_bytes = 0
    for subject_id in subjects:
        edf_path, _, _ = _subject_paths(subject_id)
        duration = float(read_edf_header(edf_path)["duration_seconds"])
        estimate_bytes += int(duration * TARGET_FS * len(TARGET_CHANNELS) * 4)
    if check_only:
        return {
            "status": "CHECK_PASS",
            "subjects": len(subjects),
            "estimated_signal_output_bytes": estimate_bytes,
            "processing_version": _processing_version(),
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    processed, skipped, failures = [], [], []
    for subject_id in subjects:
        target = OUTPUT / subject_id
        metadata_path = target / "metadata.json"
        source_hashes = audit_by_subject[subject_id]["source_sha256"]
        if metadata_path.is_file():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "COMPLETE"
                and existing.get("processing_version") == _processing_version()
                and existing.get("source_sha256") == source_hashes
            ):
                skipped.append(subject_id)
                continue
            failures.append({"subject_id": subject_id, "error": "existing output does not match frozen inputs/version; refusing overwrite"})
            continue
        if target.exists():
            failures.append({"subject_id": subject_id, "error": "partial target exists; refusing automatic removal"})
            continue
        target.mkdir(parents=True)
        try:
            edf_path, annotation_path, staging_path = _subject_paths(subject_id)
            reader = EDFSignalReader(edf_path)
            duration = float(reader.header["duration_seconds"])
            raw_channels = _raw_channel_map(reader)
            events = _events(annotation_path, duration)
            stages = _stages(staging_path, duration)
            channel_metadata = {}
            for channel in TARGET_CHANNELS:
                raw_signal, raw_quality = reader.read_physical(raw_channels[channel])
                signal_uv = _to_uv(raw_signal, raw_quality["unit"])
                processed_signal = _seed_preprocess(signal_uv, float(raw_quality["sampling_rate"]))
                signal_path = target / f"{channel.replace('-', '')}_200hz.npy"
                np.save(signal_path, processed_signal, allow_pickle=False)
                channel_metadata[channel] = {
                    "raw_label": raw_channels[channel],
                    "source_sampling_rate_hz": float(raw_quality["sampling_rate"]),
                    "source_unit": raw_quality["unit"],
                    "source_quality": raw_quality,
                    "raw_uv_stats": _stats(signal_uv),
                    "filtered_resampled_uv_stats": _stats(processed_signal),
                    "output_sampling_rate_hz": TARGET_FS,
                    "output_sample_count": int(processed_signal.size),
                    "signal_path": str(signal_path.resolve()),
                    "signal_sha256": _sha256(signal_path),
                }
            _json(target / "events.json", events)
            _write_stage_csv(target / "stages_30s.csv", stages)
            stage_labels = np.asarray([row["stage"] for row in stages], dtype="<U3")
            np.save(target / "stage_labels.npy", stage_labels, allow_pickle=False)
            np.save(target / "n2_pages.npy", np.flatnonzero(stage_labels == "N2").astype(np.int32), allow_pickle=False)
            event_counts = Counter(f"{row['channel']}:{row['task']}" for row in events)
            stage_counts = Counter(row["stage"] for row in stages)
            metadata = {
                "status": "COMPLETE",
                "experiment_id": "fame_unified_page_natural_channel_sampling",
                "processing_version": _processing_version(),
                "subject_id": subject_id,
                "source_sha256": source_hashes,
                "source_paths": {
                    "edf": str(edf_path.resolve()),
                    "annotation": str(annotation_path.resolve()),
                    "staging": str(staging_path.resolve()),
                },
                "edf_duration_seconds": duration,
                "preprocessing": {
                    "bandpass_hz": [LOW_HZ, HIGH_HZ],
                    "butterworth_order": FILTER_ORDER,
                    "phase": "zero_phase_scipy_filtfilt",
                    "target_sampling_rate_hz": TARGET_FS,
                    "resampling": "scipy_resample_poly",
                    "normalization": "deferred_to_each_fold_using_train_plus_validation",
                    "model_frame_seconds": 30,
                },
                "channels": channel_metadata,
                "event_counts": dict(sorted(event_counts.items())),
                "stage_counts": dict(sorted(stage_counts.items())),
                "usable_complete_stage_pages": len(stages),
                "n2_page_count": int(np.count_nonzero(stage_labels == "N2")),
            }
            _json(metadata_path, metadata)
            processed.append(subject_id)
        except Exception as exc:
            failures.append({"subject_id": subject_id, "error": f"{type(exc).__name__}: {exc}"})

    status = "COMPLETE" if not failures and len(processed) + len(skipped) == len(subjects) else "PARTIAL"
    summary = {
        "status": status,
        "experiment_id": "fame_unified_page_natural_channel_sampling",
        "processing_version": _processing_version(),
        "requested_subjects": len(subjects),
        "processed": processed,
        "skipped": skipped,
        "failures": failures,
        "output_root": str(OUTPUT.resolve()),
        "source_mutated": False,
    }
    _json(REPORTS / "preprocessing_summary.json", summary)
    _build_subject_splits()
    return summary


def run_normalization() -> dict:
    import fame_data as data

    output = EXPERIMENT_ROOT / "normalization"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_seed in range(3):
        for fold in range(5):
            split = data.load_subject_split(split_seed, fold)
            record_ids = data.expand_records(split["train"] + split["validation"])
            path = output / f"seed{split_seed}_fold{fold}.json"
            if path.is_file():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("record_ids") != record_ids:
                    raise RuntimeError(f"existing normalization inputs differ: {path}")
                global_std, status = float(existing["global_std_uv"]), "SKIPPED_EXISTING"
            else:
                global_std = data.compute_seed_global_std(record_ids)
                payload = {"status": "COMPLETE", "experiment_id": "fame_unified_page_natural_channel_sampling",
                           "split_seed": split_seed, "fold": fold,
                           "subjects": sorted(split["train"] + split["validation"]), "record_ids": record_ids,
                           "global_std_uv": global_std,
                           "method": "SEED paper: all sleep-stage samples from train+validation; per-record absolute p99 excluded",
                           "test_subjects_excluded": sorted(split["test"])}
                _json(path, payload)
                status = "COMPUTED"
            rows.append({"split_seed": split_seed, "fold": fold, "global_std_uv": global_std, "status": status})
    summary = {"status": "COMPLETE", "fold_statistics": rows}
    _json(output / "normalization_summary.json", summary)
    return summary


def run_validation() -> dict:
    import fame_data as data

    failures, warnings, records = [], [], []
    task_counts, stage_counts = Counter(), Counter()
    for subject_id in SUBJECTS.tolist():
        folder = data.subject_dir(subject_id)
        metadata_path = folder / "metadata.json"
        if not metadata_path.is_file():
            failures.append({"subject_id": subject_id, "error": "missing metadata"})
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "COMPLETE":
            failures.append({"subject_id": subject_id, "error": "metadata status is not COMPLETE"})
        labels = np.load(folder / "stage_labels.npy")
        n2 = np.load(folder / "n2_pages.npy")
        if np.any(n2 < 0) or np.any(n2 >= labels.size) or np.any(labels[n2] != "N2"):
            failures.append({"subject_id": subject_id, "error": "invalid n2 page index"})
        if int(np.count_nonzero(labels == "N2")) != int(n2.size):
            failures.append({"subject_id": subject_id, "error": "n2 count mismatch"})
        stage_counts.update(str(value) for value in labels)
        for event in json.loads((folder / "events.json").read_text(encoding="utf-8")):
            task_counts[f"{event['channel']}:{event['task']}"] += 1
        for channel in TARGET_CHANNELS:
            info = metadata["channels"].get(channel)
            if info is None:
                failures.append({"subject_id": subject_id, "channel": channel, "error": "missing channel metadata"})
                continue
            signal = np.load(Path(info["signal_path"]), mmap_mode="r")
            expected = int(round(float(metadata["edf_duration_seconds"]) * TARGET_FS))
            if signal.ndim != 1 or signal.size != expected:
                failures.append({"subject_id": subject_id, "channel": channel,
                                 "error": f"signal shape {signal.shape} != ({expected},)"})
            probe = np.asarray(signal[::max(1, signal.size // 100_000)])
            if not np.all(np.isfinite(probe)):
                failures.append({"subject_id": subject_id, "channel": channel, "error": "non-finite signal probe"})
            rail = float(info["source_quality"]["rail_fraction"])
            if rail >= 0.001:
                warnings.append({"subject_id": subject_id, "channel": channel,
                                 "warning": "source rail fraction >=0.1%", "value": rail})
            records.append({"subject_id": subject_id, "channel": channel, "samples": int(signal.size),
                            "filtered_std_uv": float(info["filtered_resampled_uv_stats"]["std_uv"]),
                            "source_rail_fraction": rail})
    result = {"status": "PASS" if not failures and len(records) == 160 else "FAIL",
              "experiment_id": "fame_unified_page_natural_channel_sampling",
              "subject_count": len({row["subject_id"] for row in records}), "channel_record_count": len(records),
              "sampling_rate_hz": TARGET_FS, "frame_seconds": 30, "target_samples_per_frame": 6000,
              "event_counts": dict(sorted(task_counts.items())), "stage_counts": dict(sorted(stage_counts.items())),
              "filtered_std_uv_range": [min((r["filtered_std_uv"] for r in records), default=None),
                                        max((r["filtered_std_uv"] for r in records), default=None)],
              "warnings": warnings, "failures": failures}
    _json(REPORTS / "preprocessing_validation.json", result)
    return result


def run(stage: str = "all", limit: int | None = None, check_only: bool = False,
        compute_hashes: bool = True) -> dict:
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive subject count")
    if stage == "audit":
        return run_audit(compute_hashes=compute_hashes)
    if stage == "prepare":
        return run_prepare(limit=limit, check_only=check_only)
    if stage == "splits":
        return _build_subject_splits()
    if stage == "normalize":
        return run_normalization()
    if stage == "validate":
        return run_validation()
    if stage != "all":
        raise ValueError(f"unknown preprocessing stage {stage!r}")
    audit = run_audit(compute_hashes=True)
    if audit["status"] != "PASS":
        raise RuntimeError(f"FAME audit failed; see {REPORTS / 'fame_upload_audit.json'}")
    prepared = run_prepare(limit=limit, check_only=check_only, audit=audit)
    if prepared["status"] not in ("COMPLETE", "CHECK_PASS"):
        raise RuntimeError(f"FAME preparation failed; see {REPORTS / 'preprocessing_summary.json'}")
    if check_only:
        return {"status": "CHECK_PASS", "audit": audit["status"], "prepare": prepared["status"]}
    if limit is not None:
        return {"status": "PARTIAL_PREPARATION", "audit": audit["status"], "prepare": prepared["status"],
                "normalization": "SKIPPED_FOR_PARTIAL_COHORT", "validation": "SKIPPED_FOR_PARTIAL_COHORT"}
    normalized = run_normalization()
    if normalized["status"] != "COMPLETE":
        raise RuntimeError("normalization did not complete")
    validated = run_validation()
    if validated["status"] != "PASS":
        raise RuntimeError(f"FAME preprocessing validation failed; see {REPORTS / 'preprocessing_validation.json'}")
    return {"status": "COMPLETE", "audit": audit["status"], "prepare": prepared["status"],
            "normalization": normalized["status"], "validation": validated["status"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", choices=["audit", "splits", "prepare", "normalize", "validate", "all"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-hashes", action="store_true", help="Audit without source hashes; preparation requires hashes")
    args = parser.parse_args()
    result = run(stage=args.stage, limit=args.limit, check_only=args.check_only, compute_hashes=not args.no_hashes)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] in {"CHECK_PASS", "COMPLETE", "PARTIAL_PREPARATION"} else 1)
