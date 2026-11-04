''' Calendar bookkeeping for evaluating on concatenated monthly .npy files.

    The files are 3-hourly (SAMPLES_PER_DAY steps/day) and get concatenated
    month after month. Index conventions:
      - raw index:     position in that concatenation, before any history
                       padding (the indices of the `dates` list).
      - dataset index: what npyDataWndHist.__getitem__ takes, i.e.
                       raw index - padding, where padding is the history window
                       length in steps (see hist_window).
    Months may be missing on disk; build_date_index records where the
    concatenation jumps over one, so evaluation dates whose history window would
    splice across that jump can be skipped.
'''
import os
import numpy as np
import pandas as pd

SAMPLES_PER_DAY = 8    # source data is 3-hourly; matches npyDataWndHist.SAMPLES_PER_DAY

SEASON_OF_MONTH = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
                   6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}


def hist_window(hist_duration_days, hist_freq_days=1):
    """(n_hist, padding) for a history window, mirroring npyDataWndHist: n_hist
    wind snapshots spaced hist_freq_days apart, and padding = offset between raw
    and dataset index (= hist_duration_days * SAMPLES_PER_DAY). The snapshots
    actually reach back only padding - hist_freq_days * SAMPLES_PER_DAY steps
    (lags 0 .. hist_duration_days - hist_freq_days days)."""
    if hist_duration_days % hist_freq_days != 0:
        raise ValueError('hist_duration_days must be a multiple of hist_freq_days')
    n_hist = hist_duration_days // hist_freq_days
    return n_hist, n_hist * hist_freq_days * SAMPLES_PER_DAY


def available_file_list(years, x_dir, f_dir=None, x_prefix='wavemean', f_prefix='forcing'):
    """(file_list, ym_list) for the months in `years` whose wave and forcing .npy
    files both exist on disk. f_dir defaults to x_dir (OPTION 3 keeps waveparts_*
    and forcing_* in different directories)."""
    f_dir = x_dir if f_dir is None else f_dir
    file_list, ym_list = [], []
    for y in years:
        for m in range(1, 13):
            xfile = os.path.join(x_dir, f'{x_prefix}_{y}{m:02d}.npy')
            ffile = os.path.join(f_dir, f'{f_prefix}_{y}{m:02d}.npy')
            if os.path.exists(xfile) and os.path.exists(ffile):
                file_list.append((xfile, ffile))
                ym_list.append((y, m))
            else:
                print(f'[skip] {y}-{m:02d}: preprocessed files not found')
    return file_list, ym_list


def previous_month(year, month):
    return (year - 1, 12) if month == 1 else (year, month - 1)


def months_file_list(test_months, x_dir, f_dir=None, x_prefix='wavemean', f_prefix='forcing',
                     prepend_previous=True):
    """(file_list, ym_list) for test_months, preceded by the month before the
    first one when prepend_previous, so the first test days have a full
    wind-history window. That extra month is left out, with a note, if its files
    are missing; a missing test month raises."""
    f_dir = x_dir if f_dir is None else f_dir
    paths = lambda y, m: (os.path.join(x_dir, f'{x_prefix}_{y}{m:02d}.npy'),
                          os.path.join(f_dir, f'{f_prefix}_{y}{m:02d}.npy'))
    ym_list = list(test_months)
    if prepend_previous:
        prev = previous_month(*ym_list[0])
        if all(os.path.exists(p) for p in paths(*prev)):
            ym_list.insert(0, prev)
        else:
            print(f'[note] {prev[0]}-{prev[1]:02d} not found; the first test days have no history and are skipped')
    file_list = [paths(y, m) for y, m in ym_list]
    missing = [p for pair in file_list for p in pair if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f'Missing test files: {missing}')
    return file_list, ym_list


def file_lengths(file_list):
    """Steps per file, without building a dataset. Mirrors
    MultiFileNpyData.__init__ (every step of every file is used)."""
    return [len(np.load(xfile, mmap_mode='r')) for xfile, _ in file_list]


def build_date_index(ym_list, file_lengths):
    """Per-raw-index calendar timestamp, plus the set of raw indices where the
    concatenated file list jumps over a real calendar gap (a missing month)."""
    dates, gap_before = [], set()
    raw_idx = 0
    prev_ym = None
    for (y, m), flen in zip(ym_list, file_lengths):
        if prev_ym is not None:
            py, pm = prev_ym
            expected = (py + 1, 1) if pm == 12 else (py, pm + 1)
            if (y, m) != expected:
                gap_before.add(raw_idx)
        month_start = pd.Timestamp(year=y, month=m, day=1)
        dates.extend(month_start + pd.Timedelta(hours=3 * k) for k in range(flen))
        raw_idx += flen
        prev_ym = (y, m)
    # Keep as a plain list of pd.Timestamp (not np.array) -- we only ever index it
    # with a single int, and np.array() would silently coerce to datetime64, whose
    # scalars lack .date()/.year/.month.
    return dates, gap_before


def window_crosses_gap(raw_idx, padding, gap_before):
    """True if the history window [raw_idx-padding, raw_idx] would splice across
    a calendar gap, which would feed the model stale/unrelated 'history'."""
    return any(raw_idx - padding < g <= raw_idx for g in gap_before)


def select_eval_indices(total_raw, paddings, stride_days, gap_before=(), start=0):
    """Raw indices to evaluate: every stride_days days, starting at `start` or
    once the longest history window fits, whichever is later (so a prepended
    history-only month is skipped), skipping any date whose window (for any of
    `paddings`) crosses a calendar gap. Shared across models with different
    history lengths so they're all scored on the same dates."""
    paddings = list(paddings)
    stride = stride_days * SAMPLES_PER_DAY
    return [r for r in range(max(max(paddings), start), total_raw, stride)
            if not any(window_crosses_gap(r, p, gap_before) for p in paddings)]


def dates_to_raw_indices(wanted, dates, padding=0):
    """Raw indices of the timestamps in `wanted`, looked up in `dates` (from
    build_date_index). Timestamps not on the 3-hourly grid or not covered by the
    files are skipped, as are those whose history window would reach before the
    first file (raw < padding); each skip is printed."""
    pos = {d: i for i, d in enumerate(dates)}
    out = []
    for w in wanted:
        r = pos.get(pd.Timestamp(w))
        if r is None:
            print(f'[skip] {w}: not in the loaded files')
        elif r < padding:
            print(f'[skip] {w}: history window reaches before the first file')
        else:
            out.append(r)
    return out


def select_sample_indices(dates, padding, every_n, sample_days=None, sample_hour=0, keep_months=None):
    """Raw indices analysis/paper_figure/sample.py samples, given `dates` (from
    build_date_index): every every_n steps from the first step with a full history
    window, or, if sample_days is set, those days of every month at sample_hour
    (days past a month's end skipped). With keep_months, only dates in those
    (year, month)s are kept, e.g. to drop a month loaded only for its history.
    Striding starts at the first kept month's first step."""
    months = sorted({(d.year, d.month) for d in dates}) if keep_months is None else sorted(keep_months)
    if sample_days is None:
        first = pd.Timestamp(year=months[0][0], month=months[0][1], day=1)
        start = max(padding, next((i for i, d in enumerate(dates) if d >= first), len(dates)))
        raw = range(start, len(dates), every_n)
        return [r for r in raw if (dates[r].year, dates[r].month) in set(months)]
    wanted = [pd.Timestamp(year=y, month=m, day=d, hour=sample_hour) for y, m in months
              for d in sample_days if d <= pd.Timestamp(year=y, month=m, day=1).days_in_month]
    return dates_to_raw_indices(wanted, dates, padding)


def write_dates_csv(path, raw_indices, dates, padding):
    """Save the dataset index -> date mapping of the sampled steps."""
    pd.DataFrame({'index': [to_dataset_idx(r, padding) for r in raw_indices],
                  'date': [dates[r] for r in raw_indices]}).to_csv(path, index=False)


def to_dataset_idx(raw_idx, padding):
    """Raw index -> the index npyDataWndHist.__getitem__ expects."""
    return raw_idx - padding
