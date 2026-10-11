"""fame_kc.py: Defines the FAME class that manipulates the FAME database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import time

import numpy as np
import pyedflib

from sleeprnn.common import constants
from sleeprnn.common import pkeys
from sleeprnn.data import utils
from sleeprnn.data.dataset import Dataset
from sleeprnn.data.dataset import KEY_EEG, KEY_MARKS
from sleeprnn.data.dataset import KEY_N2_PAGES, KEY_ALL_PAGES, KEY_HYPNOGRAM

PATH_FAME = os.path.abspath(
    os.environ.get(
        "FAME_DATA_ROOT",
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "FAME"),
    )
)
PATH_PART_1 = "part1"
PATH_PART_2 = "part2"
PATH_STATES = "state"

KEY_FILE_EEG = "file_eeg"
KEY_FILE_STATES = "file_states"
KEY_FILE_MARKS = "file_marks"
KEY_CHANNEL = "channel"

TARGET_CHANNELS = ("F3-A2", "F4-A1", "C3-A2", "C4-A1")
CHANNEL_SLUGS = {
    "F3-A2": "f3a2",
    "F4-A1": "f4a1",
    "C3-A2": "c3a2",
    "C4-A1": "c4a1",
}
SLUG_CHANNELS = {value: key for key, value in CHANNEL_SLUGS.items()}
SUBJECT_IDS = ["01-01-%04d" % index for index in range(1, 21)] + [
    "01-02-%04d" % index for index in range(1, 21)
]
STAGE_LABELS = {
    "n1": "N1",
    "n2": "N2",
    "n3": "N3",
    "rem": "REM",
    "w": "W",
    "a": "A",
}
EVENT_LABELS = {"Spindle": constants.SPINDLE, "K-complex": constants.KCOMPLEX}


class FameKC(Dataset):
    """This is a class to manipulate the FAME data EEG dataset.
    For K-complex events

    Expected directory tree inside PATH_FAME:

    PATH_FAME
    |__ part1
        |__ 01-01-0001 PSG.edf
        |__ 01-01-0001-annotation.csv
        |__ ...
    |__ part2
        |__ 01-02-0001 PSG.edf
        |__ 01-02-0001-annotation.csv
        |__ ...
    |__ state
        |__ 01-01-0001-state.csv
        |__ 01-02-0001-state.csv
        |__ ...
    """

    def __init__(self, params=None, load_checkpoint=False, verbose=True, **kwargs):
        """Constructor"""
        self.stage_scope = kwargs.get("stage_scope", "nrem_n1n2n3")
        if self.stage_scope not in ("n2", "nrem_n1n2n3"):
            raise ValueError("Unsupported FAME stage scope: %s" % self.stage_scope)

        # FAME parameters
        params = {} if params is None else params.copy()
        params[pkeys.FS] = 200
        params[pkeys.PAGE_DURATION] = 30
        all_ids = self._expand_records(SUBJECT_IDS)

        super(FameKC, self).__init__(
            dataset_dir=kwargs.get("dataset_dir", PATH_FAME),
            load_checkpoint=load_checkpoint,
            dataset_name="%s_%s" % (constants.FAME_KC_NAME, self.stage_scope),
            all_ids=all_ids,
            event_name=constants.KCOMPLEX,
            hypnogram_sleep_labels=["N1", "N2", "N3", "REM"],
            hypnogram_page_duration=30,
            n_experts=1,
            default_expert=1,
            default_page_subset=constants.N2_RECORD,
            params=params,
            verbose=verbose,
        )

        self.filt_signal_dict = {}
        self.exists_cache = False

    def cv_split(self, n_folds, fold_id, seed=0, subject_ids=None):
        """Subject-wise 5-fold cross-validation split."""
        if n_folds != 5:
            raise ValueError("%d folds are not supported, choose 5" % n_folds)
        if fold_id >= n_folds:
            raise ValueError("fold id %s invalid for %d folds" % (fold_id, n_folds))
        if seed != 0:
            raise ValueError("FAME uses split seed 0")

        subjects = np.asarray(SUBJECT_IDS)
        permuted = np.random.RandomState(seed=seed).permutation(subjects)
        test_folds = [
            np.sort(permuted[index * 8 : (index + 1) * 8])
            for index in range(n_folds)
        ]
        test_subjects = test_folds[fold_id]
        val_subjects = test_folds[(fold_id + 1) % n_folds]
        held_out = set(test_subjects) | set(val_subjects)
        train_subjects = np.sort(
            np.asarray([subject for subject in subjects if subject not in held_out])
        )
        train_ids = self._expand_records(train_subjects)
        val_ids = self._expand_records(val_subjects)
        test_ids = self._expand_records(test_subjects)
        return train_ids, val_ids, test_ids

    def _load_from_source(self):
        """Loads the data from files and transforms it appropriately."""
        data_paths = self._get_file_paths()
        data = {}
        n_data = len(data_paths)
        start = time.time()
        for i, record_id in enumerate(data_paths.keys()):
            print("\nLoading ID %s" % record_id)
            path_dict = data_paths[record_id]

            # Read data
            signal = self._read_eeg(path_dict[KEY_FILE_EEG], path_dict[KEY_CHANNEL])
            hypnogram = self._read_states(path_dict[KEY_FILE_STATES], signal.size)
            signal, hypnogram = self._fix_signal_and_states(signal, hypnogram)
            all_pages, selected_pages = self._hypnogram_selections(hypnogram)
            marks_1 = self._read_marks(
                path_dict[KEY_FILE_MARKS], path_dict[KEY_CHANNEL], signal.size
            )

            print("Selected pages: %d" % selected_pages.shape[0])
            print("Whole-night pages: %d" % all_pages.shape[0])
            print("Hypnogram pages: %d" % hypnogram.shape[0])
            print("Marks KC from E1: %d" % marks_1.shape[0])

            # Save data
            ind_dict = {
                KEY_EEG: signal,
                KEY_N2_PAGES: selected_pages,
                KEY_ALL_PAGES: all_pages,
                "%s_1" % KEY_MARKS: marks_1,
                KEY_HYPNOGRAM: hypnogram,
            }
            data[record_id] = ind_dict
            print(
                "Loaded ID %s (%03d/%03d ready). Time elapsed: %1.4f [s]"
                % (record_id, i + 1, n_data, time.time() - start)
            )
        print("%d records have been read." % len(data))
        return data

    def _get_file_paths(self):
        """Returns a list of dicts containing paths to load the database."""
        data_paths = {}
        for record_id in self.all_ids:
            subject_id, channel = self._parse_record_id(record_id)
            group = PATH_PART_1 if subject_id.startswith("01-01-") else PATH_PART_2
            path_eeg_file = os.path.join(
                self.dataset_dir, group, "%s PSG.edf" % subject_id
            )
            path_marks_file = os.path.join(
                self.dataset_dir, group, "%s-annotation.csv" % subject_id
            )
            path_states_file = os.path.join(
                self.dataset_dir, PATH_STATES, "%s-state.csv" % subject_id
            )
            ind_dict = {
                KEY_FILE_EEG: path_eeg_file,
                KEY_FILE_STATES: path_states_file,
                KEY_FILE_MARKS: path_marks_file,
                KEY_CHANNEL: channel,
            }
            for key in (KEY_FILE_EEG, KEY_FILE_STATES, KEY_FILE_MARKS):
                if not os.path.isfile(ind_dict[key]):
                    print("File not found: %s" % ind_dict[key])
            data_paths[record_id] = ind_dict
        print("%d records in %s dataset." % (len(data_paths), self.dataset_name))
        print("Subject IDs: %s" % SUBJECT_IDS)
        return data_paths

    def _read_eeg(self, path_eeg_file, channel):
        """Loads signal from 'path_eeg_file', does filtering and resampling."""
        with pyedflib.EdfReader(path_eeg_file) as file:
            channel_names = file.getSignalLabels()
            channel_map = self._channel_map(channel_names)
            channel_to_extract = channel_names.index(channel_map[channel])
            signal = file.readSignal(channel_to_extract)
            fs_old = file.samplefrequency(channel_to_extract)
            unit = file.getPhysicalDimension(channel_to_extract)
            print("Channel extracted: %s" % file.getLabel(channel_to_extract))

        signal = self._to_microvolts(signal, unit)
        fs_old_round = int(np.round(fs_old))
        if not np.isclose(fs_old, fs_old_round, rtol=0, atol=1e-9):
            raise ValueError("Non-integer source sampling rate %s" % fs_old)

        signal = utils.broad_filter(signal, fs_old_round, lowcut=0.1, highcut=35)
        if self.fs != fs_old_round:
            print("Resampling from %d Hz to required %d Hz" % (fs_old_round, self.fs))
            signal = utils.resample_signal(
                signal, fs_old=fs_old_round, fs_new=self.fs
            )
        else:
            print("Signal already at required %d Hz" % self.fs)
        if not np.all(np.isfinite(signal)):
            raise ValueError("Non-finite value after filtering/resampling")
        return signal.astype(np.float32)

    def _read_marks(self, path_marks_file, channel, signal_size):
        """Loads K-complex annotations from 'path_marks_file'."""
        marks_time = []
        with open(path_marks_file, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            expected = ["channel", "start_time", "stop_time", "label"]
            if list(reader.fieldnames or []) != expected:
                raise ValueError("Unexpected annotation columns: %s" % reader.fieldnames)
            for row_number, row in enumerate(reader, start=2):
                row_channel = self._canonical_channel(row["channel"])
                event_name = EVENT_LABELS.get(row["label"].strip())
                start = float(row["start_time"])
                end = float(row["stop_time"])
                if row_channel is None or event_name is None:
                    raise ValueError(
                        "Invalid annotation at %s:%d" % (path_marks_file, row_number)
                    )
                if not (0 <= start < end):
                    raise ValueError(
                        "Invalid event time at %s:%d" % (path_marks_file, row_number)
                    )
                if row_channel == channel and event_name == self.event_name:
                    marks_time.append([start, end])
        if not marks_time:
            return np.empty((0, 2), dtype=np.int32)
        marks = np.round(np.asarray(marks_time) * self.fs).astype(np.int32)
        # Keep overlapping events and clip them to the retained signal.
        # return utils.filter_stamps(marks, 0, signal_size - 1)
        marks = utils.filter_stamps(marks, 0, signal_size - 1)
        return np.clip(marks, 0, signal_size - 1)

    def _read_states(self, path_states_file, signal_size):
        """Loads the 30-second hypnogram from 'path_states_file'."""
        with open(path_states_file, "r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.reader(file))
        header_index = next(
            index for index, row in enumerate(rows) if row[:2] == ["Time", "Value"]
        )
        data_rows = [
            row
            for row in rows[header_index + 1 :]
            if any(cell.strip() for cell in row)
        ]
        usable_rows = min(len(data_rows), signal_size // self.page_size)
        hypnogram = []
        for index, row in enumerate(data_rows[:usable_rows]):
            stage = STAGE_LABELS.get(row[1].strip().casefold())
            if stage is None:
                raise ValueError(
                    "Unknown stage at %s:%d" % (
                        path_states_file,
                        header_index + index + 2,
                    )
                )
            hypnogram.append(stage)
        return np.asarray(hypnogram)

    def _fix_signal_and_states(self, signal, hypnogram):
        n_pages_valid = min(signal.size // self.page_size, hypnogram.size)
        n_samples_valid = n_pages_valid * self.page_size
        signal = signal[:n_samples_valid]
        hypnogram = hypnogram[:n_pages_valid]
        return signal, hypnogram

    def _hypnogram_selections(self, hypnogram):
        total_pages = hypnogram.size
        all_pages = np.arange(1, max(1, total_pages - 1), dtype=np.int32)
        if self.stage_scope == "n2":
            selected_labels = ["N2"]
        else:
            selected_labels = ["N1", "N2", "N3"]
        selected_pages = all_pages[
            np.isin(hypnogram[all_pages], selected_labels)
        ]
        return all_pages, selected_pages

    @staticmethod
    def _canonical_channel(value):
        compact = (
            value.strip()
            .upper()
            .replace("EEG", "")
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )
        aliases = {
            "F3A2": "F3-A2",
            "F4A1": "F4-A1",
            "C3A2": "C3-A2",
            "C4A1": "C4-A1",
            "F3M2": "F3-A2",
            "F4M1": "F4-A1",
            "C3M2": "C3-A2",
            "C4M1": "C4-A1",
        }
        return aliases.get(compact)

    @classmethod
    def _channel_map(cls, channel_names):
        result = {}
        for raw_name in channel_names:
            canonical = cls._canonical_channel(raw_name)
            if canonical is None:
                continue
            if canonical in result:
                raise ValueError("Duplicate target channel %s" % canonical)
            result[canonical] = raw_name
        if set(result) != set(TARGET_CHANNELS):
            raise ValueError("Target channel mismatch: %s" % sorted(result))
        return result

    @staticmethod
    def _to_microvolts(signal, unit):
        normalized = unit.strip().replace("µ", "u").casefold()
        factors = {"uv": 1.0, "mv": 1000.0, "v": 1000000.0}
        if normalized not in factors:
            raise ValueError("Unsupported EEG unit %s" % unit)
        return np.asarray(signal, dtype=np.float64) * factors[normalized]

    @staticmethod
    def _record_id(subject_id, channel):
        return "%s__%s" % (CHANNEL_SLUGS[channel], subject_id)

    @staticmethod
    def _parse_record_id(record_id):
        slug, subject_id = record_id.split("__", 1)
        return subject_id, SLUG_CHANNELS[slug]

    @classmethod
    def _expand_records(cls, subject_ids):
        return [
            cls._record_id(subject_id, channel)
            for channel in TARGET_CHANNELS
            for subject_id in subject_ids
        ]

    def create_signal_cache(self, highcut=4):
        signals = self.get_signals(normalize_clip=False)
        for k, record_id in enumerate(self.all_ids):
            filt_signal = utils.filter_iir_lowpass(signals[k], self.fs, highcut=highcut)
            filt_signal = filt_signal.astype(np.float32)
            self.filt_signal_dict[record_id] = filt_signal
        self.exists_cache = True

    def delete_signal_cache(self):
        self.filt_signal_dict = {}
        self.exists_cache = False

    def get_subject_filt_signal(self, record_id):
        if self.exists_cache:
            signal = self.filt_signal_dict[record_id]
        else:
            signal = None
        return signal

    def get_subset_filt_signals(self, record_id_list):
        if self.exists_cache:
            subset_signals = [
                self.get_subject_filt_signal(record_id) for record_id in record_id_list
            ]
        else:
            subset_signals = None
        return subset_signals

    def get_filt_signals(self):
        if self.exists_cache:
            subset_signals = [
                self.get_subject_filt_signal(record_id) for record_id in self.all_ids
            ]
        else:
            subset_signals = None
        return subset_signals

    def exists_filt_signal_cache(self):
        return self.exists_cache
