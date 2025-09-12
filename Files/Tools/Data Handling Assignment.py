# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:22:14 2025

@author: jadesimone
"""

# -*- coding: utf-8 -*-
"""
Data generator + Excel writer per student name.

Requirements:
    pip install numpy xlsxwriter
"""

import numpy as np
import math
from pathlib import Path
import xlsxwriter


# ---------- Utilities ----------

def nearest_correlation(A, max_iters=100, tol=1e-10):
    """
    Higham (2002) projection to the nearest correlation matrix.
    Ensures symmetric, unit diagonal, positive semidefinite.
    """
    # Symmetrize
    A = (A + A.T) / 2.0
    # Force diagonal to 1
    np.fill_diagonal(A, 1.0)

    Y = A.copy()
    delta_S = np.zeros_like(A)

    for _ in range(max_iters):
        R = Y - delta_S
        # Project R to PSD via eigenvalue thresholding
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals[eigvals < 0] = 0
        X = (eigvecs * eigvals) @ eigvecs.T

        delta_S = X - R
        Y = X.copy()
        # Force unit diagonal and symmetry
        np.fill_diagonal(Y, 1.0)
        Y = (Y + Y.T) / 2.0

        # Convergence check
        off_diag = Y - A
        np.fill_diagonal(off_diag, 0.0)
        if np.linalg.norm(off_diag, ord='fro') < tol:
            break

    # Final safety: clip to [-1, 1] and unit diagonal
    Y = np.clip(Y, -1.0, 1.0)
    np.fill_diagonal(Y, 1.0)
    return Y


def cholesky_psd(C):
    """
    Cholesky for PSD matrix (jitter if needed).
    Returns lower-triangular L such that L @ L.T ≈ C.
    """
    jitter = 1e-10
    for _ in range(10):
        try:
            return np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            # Add small jitter to diagonal
            C = C + np.eye(C.shape[0]) * jitter
            jitter *= 10
    # Final fallback via eigen-decomp
    vals, vecs = np.linalg.eigh(C)
    vals[vals < 0] = 0
    C_psd = (vecs * vals) @ vecs.T
    return np.linalg.cholesky(C_psd + 1e-12 * np.eye(C.shape[0]))


def standardize_cols(X):
    """
    Column-wise standardization to mean 0, sd 1 (guarding against zero-variance).
    """
    X = X.astype(float)
    means = X.mean(axis=0)
    sds = X.std(axis=0, ddof=1)
    sds[sds == 0] = 1.0
    return (X - means) / sds


def scale_and_clip_column(col, sd, mean, vmin, vmax):
    """
    Given standardized column, scale to desired sd/mean then clip to [vmin, vmax].
    """
    col = col * sd + mean
    col = np.clip(col, vmin, vmax)
    return col


def rowwise_pearson(x, y):
    """
    Pearson correlation between two length-5 vectors (row-wise usage).
    Returns scalar correlation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x.mean()
    ym = y.mean()
    xv = x - xm
    yv = y - ym
    num = np.sum(xv * yv)
    den = math.sqrt(np.sum(xv * xv) * np.sum(yv * yv))
    if den == 0:
        return 0.0
    return float(num / den)


def _nudge_constant_5(arr, rng):
    """If all 5 values are identical, nudge one entry by ±1 within [1,9]."""
    arr = arr.astype(int, copy=True)
    if np.all(arr == arr[0]):
        j = rng.integers(0, 5)        # choose which of the 5 to change
        v = arr[j]
        if v <= 1:
            arr[j] = 2                # keep within bounds
        elif v >= 9:
            arr[j] = 8
        else:
            arr[j] = v + rng.choice([-1, 1])  # random ±1
    return arr

def variance(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 0.0
    m = arr.mean()
    return float(np.mean((arr - m) ** 2))


# ---------- Step 1: loop setup ----------

student_names = ["Student1", "Student2", "Student3"]


# ---------- Fixed target correlation matrix (7x7) ----------

# Order: Age, Sex, Aptitude, Con, Integrity, Sat, Perf
target_correlation_matrix = np.array([
    [1.00,  0.00,  0.17,  0.15,  0.18,  0.11,  0.02],
    [0.00,  1.00, -0.03, -0.05, -0.10,  0.07,  0.00],
    [0.17, -0.03,  1.00,  0.09,  0.10,  0.14,  0.42],
    [0.15, -0.05,  0.09,  1.00,  0.30,  0.13,  0.35],
    [0.18, -0.10,  0.10,  0.30,  1.00,  0.21,  0.25],
    [0.11,  0.07,  0.14,  0.13,  0.21,  1.00,  0.20],
    [0.02,  0.00,  0.42,  0.35,  0.25,  0.20,  1.00],
], dtype=float)


# ---------- Column scaling parameters ----------
# index 0..6 -> Age, Sex, Aptitude, Con, Integrity, Sat, Perf
scale_specs = [
    dict(sd=13.0,   mean=44.0, vmin=21.0, vmax=67.0),  # Age
    dict(sd=0.2,    mean=0.5,  vmin=0.0,  vmax=1.0),   # Sex
    dict(sd=15.0,   mean=100,  vmin=50.0, vmax=150.0), # Aptitude
    dict(sd=1.75,   mean=4.0,  vmin=1.0,  vmax=7.0),   # Con
    dict(sd=1.75,   mean=4.0,  vmin=1.0,  vmax=7.0),   # Integrity
    dict(sd=1.25,   mean=3.0,  vmin=1.0,  vmax=5.0),   # Satisfaction
    dict(sd=17.5,   mean=50.0, vmin=1.0,  vmax=99.0),  # Performance
]


# ---------- Main per-student loop ----------

rng = np.random.default_rng()  # seedless by spec

for student in student_names:

    # ----- Step 2: Correlated base data (matrix-only work) -----

    # 2.2 Randomize off-diagonals ~ N(mean, .07), clipped [-1,1], keep diag=1, symmetrize
    rnd = target_correlation_matrix.copy()
    n = rnd.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            mu = target_correlation_matrix[i, j]
            val = rng.normal(mu, 0.07)
            val = float(np.clip(val, -1.0, 1.0))
            rnd[i, j] = val
            rnd[j, i] = val
    np.fill_diagonal(rnd, 1.0)

    # Project to nearest correlation matrix to ensure Cholesky succeeds
    randomized_correlation_matrix = nearest_correlation(rnd)

    # 2.3 Cholesky
    cholesky_correlation_matrix = cholesky_psd(randomized_correlation_matrix)

    # 2.4 Random 1000x7 uniform data
    random_data = rng.uniform(low=0.0, high=1.0, size=(1000, 7))

    # 2.5 Correlate via Cholesky (note: follows instruction; typical practice uses normals)
    correlated_data = random_data @ cholesky_correlation_matrix.T  # (1000x7) * (7x7)^T

    # 2.6 Standardize columns
    standardized_data = standardize_cols(correlated_data)

    # 2.7 Scale columns and clip to bounds
    scaled_data = np.zeros_like(standardized_data)
    for j in range(7):
        spec = scale_specs[j]
        scaled_data[:, j] = scale_and_clip_column(
            standardized_data[:, j],
            sd=spec["sd"],
            mean=spec["mean"],
            vmin=spec["vmin"],
            vmax=spec["vmax"],
        )

    # Make Sex (column index 1) binary 0/1 by rounding
    scaled_data[:, 1] = np.rint(scaled_data[:, 1]).astype(int)

    # 2.10 analysis_dataset 1000x50
    analysis_dataset = np.zeros((1000, 50), dtype=float)

    # 2.11 Populate columns in required order

    # Col1: uniform [0,1]
    analysis_dataset[:, 0] = rng.uniform(0.0, 1.0, size=1000)

    # Col2: IDs 1..1000
    analysis_dataset[:, 1] = np.arange(1, 1001, dtype=int)

    # Cols 3-7: uniform integers 1..9
    analysis_dataset[:, 2:7] = rng.integers(1, 10, size=(1000, 5))

    # Col8: Age integer ~ N(loc=scaled_data[:,0], sd=7.5), clipped 21..67
    age = np.rint(rng.normal(loc=scaled_data[:, 0], scale=7.5))
    age = np.clip(age, 21, 67).astype(int)
    analysis_dataset[:, 7] = age

    # Col9: Sex (exactly from scaled_data col2)
    analysis_dataset[:, 8] = scaled_data[:, 1]

    # Cols 10-13: Aptitude 4 items int ~ N(mean=scaled col3, sd=10), clip 50..150
    apt_mean = scaled_data[:, 2]
    for k, col in enumerate(range(9, 13)):
        vals = np.rint(rng.normal(loc=apt_mean, scale=10.0))
        vals = np.clip(vals, 50, 150).astype(int)
        analysis_dataset[:, col] = vals

    # Conscientiousness indices in analysis_dataset (1-based spec -> 0-based here):
    # Cols 14,15,17,19,20,22,23,24 (direct)
    # Cols 18,21 are reverse (will compute below)
    con_mean = scaled_data[:, 3]
    direct_con_cols = [13, 14, 16, 18, 19, 21, 22, 23]  # zero-based
    for col in direct_con_cols:
        vals = np.rint(rng.normal(loc=con_mean, scale=1.5))
        vals = np.clip(vals, 1, 7).astype(int)
        analysis_dataset[:, col] = vals

    # Col16: if col1 in [0.955, 0.9625] => random int 1..7 else 2
    mask_16 = (analysis_dataset[:, 0] >= 0.955) & (analysis_dataset[:, 0] < 0.9625)
    col16 = np.full(1000, 2, dtype=int)
    col16[mask_16] = rng.integers(1, 8, size=mask_16.sum())
    analysis_dataset[:, 15] = col16  # column 16 (0-based 15)

    # Conscientiousness reverse items: store RAW values now (reversing moved to Step 3)
    for col in [17, 20]:  # zero-based for 18 and 21
        vals = np.rint(rng.normal(loc=con_mean, scale=1.5))
        vals = np.clip(vals, 1, 7).astype(int)
        analysis_dataset[:, col] = vals  # <-- no 8 - vals here

    # Integrity:
    # Direct: Cols 25,27,29,30,32,35 (0-based [24,26,28,29,31,34])
    # Reverse: Cols 26,28,33,34 (0-based [25,27,32,33]) = 8 - value
    int_mean = scaled_data[:, 4]
    direct_int_cols = [24, 26, 28, 29, 31, 34]
    for col in direct_int_cols:
        vals = np.rint(rng.normal(loc=int_mean, scale=1.5))
        vals = np.clip(vals, 1, 7).astype(int)
        analysis_dataset[:, col] = vals

    rev_int_cols = [25, 27, 32, 33]
    for col in rev_int_cols:
        vals = np.rint(rng.normal(loc=int_mean, scale=1.5))
        vals = np.clip(vals, 1, 7).astype(int)
        analysis_dataset[:, col] = 8 - vals

    # Col31: if col1 in [0.9625, 0.97) => random int 1..7 else 5
    mask_31 = (analysis_dataset[:, 0] >= 0.9625) & (analysis_dataset[:, 0] < 0.97)
    col31 = np.full(1000, 5, dtype=int)
    col31[mask_31] = rng.integers(1, 8, size=mask_31.sum())
    analysis_dataset[:, 30] = col31  # 0-based index 30

    # Satisfaction:
    # Direct: Cols 36,37,40 (0-based [35,36,39]) sd=1.25, clip 1..5
    # Reverse: Cols 38,39 (0-based [37,38]) = 6 - value
    sat_mean = scaled_data[:, 5]
    direct_sat_cols = [35, 36, 39]
    for col in direct_sat_cols:
        vals = np.rint(rng.normal(loc=sat_mean, scale=1.25))
        vals = np.clip(vals, 1, 5).astype(int)
        analysis_dataset[:, col] = vals

    rev_sat_cols = [37, 38]
    for col in rev_sat_cols:
        vals = np.rint(rng.normal(loc=sat_mean, scale=1.25))
        vals = np.clip(vals, 1, 5).astype(int)
        analysis_dataset[:, col] = 6 - vals

    # Performance Cols 41..43 (0-based [40,41,42]) int ~ N(mean=perf, sd=12.5), clip 1..99
    perf_mean = scaled_data[:, 6]
    for col in [40, 41, 42]:
        vals = np.rint(rng.normal(loc=perf_mean, scale=12.5))
        vals = np.clip(vals, 1, 99).astype(int)
        analysis_dataset[:, col] = vals

    # SE follow-ups Cols 44..48 depend on Cols 3..7
    # Each col j (44..48) ~ int N(mean = analysis_dataset[:, j-41?], sd=2), clip 1..9,
    # then if col1 in [0.985,1.0] set value = 10 - value
    # Mapping: 44<-3, 45<-4, 46<-5, 47<-6, 48<-7
    mask_flip = (analysis_dataset[:, 0] >= 0.985) & (analysis_dataset[:, 0] <= 1.0)
    for out_col, src_col in zip([43, 44, 45, 46, 47], [2, 3, 4, 5, 6]):
        mu = analysis_dataset[:, src_col]
        vals = np.rint(rng.normal(loc=mu, scale=0.5))
        vals = np.clip(vals, 1, 9).astype(int)
        flip_vals = 10 - vals
        vals = np.where(mask_flip, flip_vals, vals)
        analysis_dataset[:, out_col] = vals
        
    for i in range(1000):
        # SE1..SE5 (cols 3..7 -> indices 2:7)
        analysis_dataset[i, 2:7]   = _nudge_constant_5(analysis_dataset[i, 2:7], rng)
        # SE6..SE10 (cols 44..48 -> indices 43:48)
        analysis_dataset[i, 43:48] = _nudge_constant_5(analysis_dataset[i, 43:48], rng)

    # Col49: Time — if col1 in [0.94,0.955) => int 30..70 else 148..259
    mask_time = (analysis_dataset[:, 0] >= 0.94) & (analysis_dataset[:, 0] < 0.955)
    time_vals = np.empty(1000, dtype=int)
    time_vals[mask_time] = rng.integers(30, 71, size=mask_time.sum())
    time_vals[~mask_time] = rng.integers(148, 260, size=(~mask_time).sum())
    analysis_dataset[:, 48] = time_vals  # 0-based 48 is column 49

    # Col50: correlation between SE1..SE5 (cols 3..7 -> 0-based 2..6) and SE6..SE10 (cols 44..48 -> 0-based 43..47)
    corr_vals = np.zeros(1000, dtype=float)
    for i in range(1000):
        left = analysis_dataset[i, 2:7]
        right = analysis_dataset[i, 43:48]
        corr_vals[i] = rowwise_pearson(left, right)
    analysis_dataset[:, 49] = corr_vals

    # 2.12 A: if col1 in [0.97,0.9775) => set cols 15,17,18,19,20,21,22,23,24 = col14
    mask_A = (analysis_dataset[:, 0] >= 0.97) & (analysis_dataset[:, 0] < 0.9775)
    cols_A = [14, 16, 17, 18, 19, 20, 21, 22, 23]  # zero-based (15,17,18,19,20,21,22,23,24)
    for col in cols_A:
        analysis_dataset[:, col] = np.where(mask_A, analysis_dataset[:, 13], analysis_dataset[:, col])

    # 2.12 B: if col1 in [0.9775,0.985) => set cols 26,27,28,29,30,32,33,34,35 = col25
    mask_B = (analysis_dataset[:, 0] >= 0.9775) & (analysis_dataset[:, 0] < 0.985)
    cols_B = [25, 26, 27, 28, 29, 31, 32, 33, 34]  # zero-based (26..30,32..35)
    for col in cols_B:
        analysis_dataset[:, col] = np.where(mask_B, analysis_dataset[:, 24], analysis_dataset[:, col])

    # ---------- Step 3: Key file data ----------

    # 3.1 analyzed_data 1000x10
    analyzed_data = np.zeros((1000, 10), dtype=object)  # last col is string

    # Col1: ID 1..1000
    analyzed_data[:, 0] = np.arange(1, 1001, dtype=int)

    # Col2: Age (from analysis_dataset col8 -> 0-based 7)
    analyzed_data[:, 1] = analysis_dataset[:, 7].astype(int)

    # Col3: Sex (col9 -> 8)
    analyzed_data[:, 2] = analysis_dataset[:, 8]

    # Col4: Aptitude = mean cols 10..13 -> 0-based 9..12
    analyzed_data[:, 3] = np.mean(analysis_dataset[:, 9:13], axis=1)

    # Col5: Conscientiousness (reverse at key-time to mirror Integrity approach)
    con_combo = np.column_stack([
        analysis_dataset[:, 13],              # 14
        analysis_dataset[:, 14],              # 15
        analysis_dataset[:, 16],              # 17
        8 - analysis_dataset[:, 17],          # 18 (R)
        analysis_dataset[:, 18],              # 19
        analysis_dataset[:, 19],              # 20
        8 - analysis_dataset[:, 20],          # 21 (R)
        analysis_dataset[:, 21],              # 22
        analysis_dataset[:, 22],              # 23
        analysis_dataset[:, 23],              # 24
    ])
    analyzed_data[:, 4] = con_combo.mean(axis=1)

    # Col6: Integrity (explicit reverse where needed)
    # column 25, 8 - 26, 27, 8 - 28, 29, 30, 32, 8 - 33, 8 - 34, 35
    int_combo = np.column_stack([
        analysis_dataset[:, 24],
        8 - analysis_dataset[:, 25],
        analysis_dataset[:, 26],
        8 - analysis_dataset[:, 27],
        analysis_dataset[:, 28],
        analysis_dataset[:, 29],
        analysis_dataset[:, 31],
        8 - analysis_dataset[:, 32],
        8 - analysis_dataset[:, 33],
        analysis_dataset[:, 34],
    ])
    analyzed_data[:, 5] = int_combo.mean(axis=1)

    # Col7: Satisfaction (6 - 38, 6 - 39 for reverses)
    sat_combo = np.column_stack([
        analysis_dataset[:, 35],
        analysis_dataset[:, 36],
        6 - analysis_dataset[:, 37],
        6 - analysis_dataset[:, 38],
        analysis_dataset[:, 39],
    ])
    analyzed_data[:, 6] = sat_combo.mean(axis=1)

    # Col8: Performance = mean cols 41..43 -> 0-based 40..42
    analyzed_data[:, 7] = np.mean(analysis_dataset[:, 40:43], axis=1)

    # Col9: Screen = 0 default
    analyzed_data[:, 8] = 0

    # Col10: Reason = ""
    analyzed_data[:, 9] = ""

    # 3.2 Screens
    for i in range(1000):
        # A. Time (col49 -> 48) < 75 => code 1, "Time"
        if analysis_dataset[i, 48] < 75:
            analyzed_data[i, 8] = 1
            analyzed_data[i, 9] = "Time"

        # B. Iitem #1 (col16 -> 15) != 2 => code 2
        if analysis_dataset[i, 15] != 2:
            analyzed_data[i, 8] = 2
            analyzed_data[i, 9] = "Instructed item #1"

        # C. Iitem #2 (col31 -> 30) != 5 => code 2
        if analysis_dataset[i, 30] != 5:
            analyzed_data[i, 8] = 2
            analyzed_data[i, 9] = "Instructed item #2"

        # D. Variance of Con items (14,15,17,18,19,20,21,22,23,24) zero => code 3
        con_idxs = [13, 14, 16, 17, 18, 19, 20, 21, 22, 23]
        if variance(analysis_dataset[i, con_idxs]) == 0.0:
            analyzed_data[i, 8] = 3
            analyzed_data[i, 9] = "Same responses to all conscientiousness items"

        # E. Variance of Integrity items (26,27,28,29,30,32,33,34,35,36) zero => code 3
        # (Note: includes 36 per specification)
        int_idxs = [24, 25, 26, 27, 28, 29, 31, 32, 33, 34]
        if variance(analysis_dataset[i, int_idxs]) == 0.0:
            analyzed_data[i, 8] = 3
            analyzed_data[i, 9] = "Same responses to all integrity items"

        # F. SE correlation (col50 -> 49) negative => code 4
        if analysis_dataset[i, 49] < 0:
            analyzed_data[i, 8] = 4
            analyzed_data[i, 9] = "Inconsistent responses to self-efficacy items"

    # ---------- Excel Writing (XlsxWriter) ----------

    outdir = Path(".")
    # 13,17: Student Data workbook
    data_wb_path = outdir / f"{student} Data File.xlsx"
    wb = xlsxwriter.Workbook(str(data_wb_path))

    # Formats
    bold = wb.add_format({"bold": True})
    normal = wb.add_format({})
    bold_center = wb.add_format({"bold": True, "align": "center"})
    grey_header = wb.add_format({"bold": True, "bg_color": "#DDDDDD", "border": 1})
    grey_fill = wb.add_format({"bg_color": "#DDDDDD"})
    black_fill = wb.add_format({"bg_color": "#000000"})
    unlocked = wb.add_format({"locked": False})
    border = wb.add_format({"border": 1})
    # For the Instructions sheet text wrapping
    wrap = wb.add_format({"text_wrap": True})

    # 14: Instructions sheet
    wsI = wb.add_worksheet("Instructions")

    # Column headers A1..D1
    wsI.write("A1", "Variable Name", bold)
    wsI.write("B1", "Range", bold)
    wsI.write("C1", "Description", bold)
    wsI.write("D1", "Notes", bold)

    # A2..A11
    instr_rows = [
        ("A2", "ID"), ("A3", "SE1 to SE10"), ("A4", "Age"), ("A5", "Apt1 to Apt4"),
        ("A6", "Con1 to Con10"), ("A7", "Int1 to Int10"), ("A8", "Sat1 to Sat5"),
        ("A9", "Perf1 to Perf3"), ("A10", "Iitem1 to Iitem2"), ("A11", "Time")
    ]
    for cell, val in instr_rows:
        wsI.write(cell, val, normal)

    # B2..B11
    range_vals = [
        ("B2", "1 to 1000"), ("B3", "1 to 9"), ("B4", "21 to 67"), ("B5", "50 to 150"),
        ("B6", "1 to 7"), ("B7", "1 to 7"), ("B8", "1 to 5"), ("B9", "1 to 99"),
        ("B10", "1 to 7"), ("B11", "0 to ∞")
    ]
    for cell, val in range_vals:
        wsI.write(cell, val, normal)

    # C2..C11
    desc_vals = [
        ("C2", "Id number"), ("C3", "Self-efficacy indicators 1 to 10"),
        ("C4", "Age (in years)"), ("C5", "Aptitude indicators 1 to 4"),
        ("C6", "Conscientiousness indicators 1 to 10"), ("C7", "Integrity indicators 1 to 10"),
        ("C8", "Satisfaction indicators 1 to 5"), ("C9", "Performance indicators 1 to 10"),
        ("C10", "Instructed items 1 and 2"), ("C11", "Survey competion time (in seconds)")
    ]
    for cell, val in desc_vals:
        wsI.write(cell, val, normal)

    # D2, D6, D7, D8, D10
    wsI.write("D2", "ID", normal)
    wsI.write("D6", "(R) indicates a reverse-scored item", normal)
    wsI.write("D7", "(R) indicates a reverse-scored item", normal)
    wsI.write("D8", "(R) indicates a reverse-scored item", normal)
    wsI.write("D10", "Correct answers are 2 for Iitem1 and 5 for Iitem2", normal)

    # A13.. (block of instructions)
    wsI.write("A13", "Instructions", bold)
    wsI.write("A14", "Your assignment is to use the data from the worksheet entitled Starting Data to populate the worksheet entitled Analysis Data.", normal)
    wsI.write("A16", "Your first task is to create scale scores for Age, Aptitude, Conscientiousness, Integrity, Satisfaction, and Performance for each ID number. ", normal)
    wsI.write("A17", "For scales with a single item, you can simply transfer the values into the corresponding column of Analysis Data.", normal)
    wsI.write("A18", "For scales with multiple indicators, you should average the values of each indicator before transfering the average value into the corresponding column of Analysis Data.", normal)
    wsI.write("A19", "For scales including reverse-scored items, you will need to ensure all indicators are scored in te same direction (positively) before computing the average. ", normal)
    wsI.write("A21", "Your second task is to improve the quality of your data by computing a number of data screening techniques.", normal)
    wsI.write("A22", "Populate the Analysis Data column labeled Screen with the following values:", normal)
    wsI.write("A23", "0: The ID number was not screened using any technique", normal)
    wsI.write("A24", "1: The ID number was screened because they spent less than 2 seconds per item on the survey (there were 37 items on this survey)", normal)
    wsI.write("A25", "2: The ID number was screened because they indicated an incorrect response to one of the instructed items", normal)
    wsI.write("A26", "3: The ID number was screened because they indicated the same response to all ten conscientiousness items or all ten integrity items", normal)
    wsI.write("A27", "4: The ID number was screened because their first five SE item responses were not positively correlated with their second five SE item responses", normal)
    wsI.set_column("A:A", 110, wrap)
    wsI.set_column("B:D", 35)

    # 15: Starting Data sheet
    wsS = wb.add_worksheet("Starting Data")

    # Row 1 headers across columns A..AV (48 columns, bold)
    labels = [
        "ID#","SE1","SE2","SE3","SE4","SE5","Age","Sex","Apt1","Apt2","Apt3","Apt4",
        "Con1","Con2","Iitem1","Con3","Con4(R)","Con5","Con6","Con7(R)","Con8","Con9","Con10",
        "Int1","Int2(R)","Int3","Int4(R)","Int5","Int6","Iitem2","Int7","Int8(R)","Int9(R)","Int10",
        "Sat1","Sat2","Sat3(R)","Sat4(R)","Sat5","Perf1","Perf2","Perf3","SE6","SE7","SE8","SE9","SE10","Time"
    ]
    wsS.write_row(0, 0, labels, bold)
    
    # Data: analysis_dataset columns 2..49 (0-based 1..48) → rows 2..1001, columns A..AV
    data_block = analysis_dataset[:, 1:49]  # shape (1000, 48)
    for r in range(1000):
        # write a single row of 48 values starting at column 0 (A), row 1+r (2..1001 in Excel)
        wsS.write_row(1 + r, 0, data_block[r, :].tolist())
    
    # Freeze header row
    wsS.freeze_panes(1, 0)

    # 16: Analysis Data sheet
    wsA = wb.add_worksheet("Analysis Data")

    headers = ["ID","Age","Sex","Aptitude","Conscientiousness","Integrity","Satisfaction","Performance","Screen"]
    for j, h in enumerate(headers):
        wsA.write(0, j, h, grey_header)

    # A2..A1001: IDs
    for i in range(1000):
        wsA.write(1 + i, 0, i + 1, grey_fill)

    # Shade J1..J1002 black (J is column index 9)
    for r in range(0, 1002):
        wsA.write(r, 9, "", black_fill)

    # Shade A1002..I1002 black
    for c in range(0, 9):
        wsA.write(1001, c, "", black_fill)

    # Protect sheet: only B2..I1001 editable
    wsA.protect("begolis", {'select_locked_cells': True, 'select_unlocked_cells': True})
    # Lock everything by default; then unlock B2..I1001
    wsA.set_column(1, 8, 12, None)  # widths
    for r in range(1, 1001):
        wsA.set_row(r, None)
        wsA.write_blank(r, 1, None, unlocked)  # ensure cell exists and unlocked
        for c in range(1, 9):
            wsA.write_blank(r, c, None, unlocked)

    # (We leave values empty for students to fill in.)

    wb.close()

    # 3.3 Key workbook (clean + workbook-local formats)
    key_wb_path = outdir / f"{student} Key.xlsx"
    kwb = xlsxwriter.Workbook(str(key_wb_path))
    wsf = kwb.add_worksheet("Final Dataset")
    
    # >>> DO NOT reuse formats from wb. Create new ones on kwb:
    bold_key = kwb.add_format({"bold": True})
    
    # Headers A1..J1
    key_headers = [
        "ID","Age","Sex","Aptitude","Conscientiousness",
        "Integrity","Satisfaction","Performance","Screen","Reason"
    ]
    for j, h in enumerate(key_headers):
        wsf.write(0, j, h, bold_key)
    
    # Helpers for safe typing
    def _is_number(x):
        return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)
    
    def _to_py(x):
        # numpy scalar -> Python scalar
        return x.item() if isinstance(x, np.generic) else x
    
    # Write 1000 rows, starting at A2 (row index 1)
    for i in range(1000):
        # numeric cols 0..8
        for col in range(0, 9):
            v = _to_py(analyzed_data[i, col])
            if _is_number(v):
                # cast bools to ints for Excel
                if isinstance(v, bool):
                    v = int(v)
                wsf.write_number(1 + i, col, float(v))
            else:
                wsf.write_blank(1 + i, col, None)
    
        # text col 9 (Reason)
        v = _to_py(analyzed_data[i, 9])
        wsf.write_string(1 + i, 9, "" if v is None else str(v))
    
    kwb.close()

print("\nAll student files created.")
