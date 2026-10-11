
import os
import time
import numpy as np
import pandas as pd
import mne

# ─────────────────────────────────────────────────────────────
# 常量配置
# ─────────────────────────────────────────────────────────────
TARGET_SFREQ   = 256
EPOCH_LEN_S    = 30
EPOCH_LEN_PT   = TARGET_SFREQ * EPOCH_LEN_S
AMP            = 1_000_000
FRAG_THRESHOLD = 0.25

VALID_STAGES = {'N1', 'N2', 'N3'}
CHANNELS = ['C3A2', 'C4A1', 'F3A2', 'F4A1']
CH_CSV_TO_EDF = {
    'C3-A2': 'C3A2',
    'C4-A1': 'C4A1',
    'F3-A2': 'F3A2',
    'F4-A1': 'F4A1',
}

BIO_DIR = './data/FAME/FAME_bio'
ANA_DIR = './data/FAME/FAME_ana'
OUT_DIR = './SpindleData/FAME'


# ─────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────

def parse_state_csv(path: str) -> list:

    df = pd.read_csv(path, skiprows=4)
    result = []
    for _, row in df.iterrows():
        if row['Value'] in VALID_STAGES:
            result.append((float(row['Onset_sec']), float(row['Duration_sec'])))
    return result


def parse_spindle_csv(path: str) -> dict:

    df = pd.read_csv(path)
    spindles = {ch: [] for ch in CHANNELS}
    for _, row in df.iterrows():
        ch_edf = CH_CSV_TO_EDF.get(row['channel'])
        if ch_edf is not None:
            spindles[ch_edf].append((float(row['start_time']), float(row['stop_time'])))
    return spindles


def extract_channel_data(ch_data: np.ndarray, epochs: list,
                          ch_spindles: list) -> tuple:
    all_epochs = []
    all_labels = []

    for onset, duration in epochs:
        start_idx = int(round(onset * TARGET_SFREQ))
        end_idx   = start_idx + EPOCH_LEN_PT
        epoch_end = onset + duration


        if end_idx > len(ch_data):
            continue

        epoch_label = np.zeros(EPOCH_LEN_PT, dtype=np.float32)
        fragments   = []

        for sp_start, sp_stop in ch_spindles:
            if sp_stop <= onset or sp_start >= epoch_end:
                continue


            rel_start = max(0, int(round((sp_start - onset) * TARGET_SFREQ)))
            rel_end   = min(EPOCH_LEN_PT - 1, int(round((sp_stop - onset) * TARGET_SFREQ)))


            full_dur   = sp_stop - sp_start
            frag_dur   = (rel_end - rel_start) / TARGET_SFREQ
            frag_ratio = frag_dur / full_dur if full_dur > 0 else 0.0
            fragments.append(frag_ratio)

            epoch_label[rel_start:rel_end + 1] = 1.0


        if len(fragments) == 0:
            continue
        if len(fragments) == 1 and fragments[0] < FRAG_THRESHOLD:
            continue

        epoch_signal = (ch_data[start_idx:end_idx] * AMP).astype(np.float32)
        all_epochs.append(epoch_signal)
        all_labels.append(epoch_label)

    if len(all_epochs) == 0:
        return (np.array([], dtype=np.float32).reshape(0, EPOCH_LEN_PT),
                np.array([], dtype=np.float32).reshape(0, EPOCH_LEN_PT))

    return (np.array(all_epochs, dtype=np.float32),
            np.array(all_labels, dtype=np.float32))


def process_subject(subject_id: str):
    t0 = time.time()
    print(f'\n{"="*60}')
    print(f'Processing subject: {subject_id}')

    edf_path     = os.path.join(BIO_DIR, f'{subject_id}-PSG.edf')
    state_path   = os.path.join(ANA_DIR, f'{subject_id}-state.csv')
    spindle_path = os.path.join(ANA_DIR, f'{subject_id}-spindle.csv')


    for p, desc in [(edf_path, 'EDF'), (state_path, 'State CSV'), (spindle_path, 'Spindle CSV')]:
        if not os.path.exists(p):
            print(f'  WARNING: {desc} not found: {p}, skipping subject')
            return


    raw = mne.io.read_raw_edf(edf_path, include=CHANNELS, preload=True, verbose=False)
    print(f'  Loaded channels: {raw.ch_names}, sfreq={raw.info["sfreq"]:.0f}Hz')


    if raw.info['sfreq'] != TARGET_SFREQ:
        print(f'  Resampling {raw.info["sfreq"]:.0f}Hz -> {TARGET_SFREQ}Hz')
        raw.resample(TARGET_SFREQ, verbose=False)


    states         = parse_state_csv(state_path)
    spindles_by_ch = parse_spindle_csv(spindle_path)
    print(f'  N1/N2/N3 epochs: {len(states)}')

    os.makedirs(OUT_DIR, exist_ok=True)


    for ch in CHANNELS:
        if ch not in raw.ch_names:
            print(f'  WARNING: Channel {ch} not in EDF, skipping')
            continue

        virtual_id  = f'{subject_id}_{ch}'
        ch_data     = raw.get_data(picks=[ch])[0]
        ch_spindles = spindles_by_ch[ch]

        data, label = extract_channel_data(ch_data, states, ch_spindles)

        if data.shape[0] == 0:
            print(f'  {ch}: WARNING - no valid epochs!')
            continue

        np.save(os.path.join(OUT_DIR, f'epoch_{virtual_id}'), data)
        np.save(os.path.join(OUT_DIR, f'label_{virtual_id}'), label)

        n_with_spindle = int(np.sum(np.any(label > 0, axis=1)))
        print(f'  {ch}: {data.shape[0]} epochs ({n_with_spindle} with spindles), shape={data.shape}')

    del raw
    print(f'  Time: {time.time() - t0:.1f}s')

def main():
    subjects = sorted([
        f.replace('-PSG.edf', '')
        for f in os.listdir(BIO_DIR)
        if f.endswith('-PSG.edf')
    ])
    print(f'Found {len(subjects)} subjects')
    print(f'Output: {os.path.abspath(OUT_DIR)}')
    print(f'Config: sfreq={TARGET_SFREQ}Hz, epoch={EPOCH_LEN_S}s={EPOCH_LEN_PT}pts, '
          f'stages={VALID_STAGES}, frag_threshold={FRAG_THRESHOLD}')

    total_start = time.time()
    for subject_id in subjects:
        try:
            process_subject(subject_id)
        except Exception as e:
            import traceback
            print(f'[ERROR] {subject_id} 处理失败: {e}')
            traceback.print_exc()

    print(f'\n{"="*60}')
    print(f'All done! Total: {time.time() - total_start:.1f}s')


if __name__ == '__main__':
    mne.set_config('MNE_USE_CUDA', 'false')
    main()
