# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 07:30:39 2026

@author: tamph1
"""


# -*- coding: utf-8 -*
"""
credit_scoring_v5.py
Created on Thu May 28 16:25:31 2026
@author: tamph1

CREDIT SCORING — VARIABLE SELECTION PIPELINE v5
EXPORT: PARQUET BROTLI

OPTIMIZED v5:
  - MIN_BINS = 2 (v4=3): cho phép binning tối thiểu 2 bin,
    tăng khả năng giữ biến có ít giá trị unique.
  - Fix merge_interval_bins: max_merge_iter dùng initial_len
    được capture trước vòng lặp, tránh tái tính len(groups)
    đã giảm → giới hạn tight và đúng về lý thuyết.
  - Giữ nguyên toàn bộ fix từ v2, v3, v4.
"""
#
import gc
import pyarrow as pa
import pyarrow.parquet as pq

import sys
import os
import re
import numpy as np
import pandas as pd

# THAM SỐ NGƯỠNG TỐI ƯU 

IV_THRESHOLD               = 0.1       # Ngưỡng giữ lại tín hiệu từ "Weak" đến "Strong"
CORR_THRESHOLD             = 0.89      # Kiểm soát chặt chẽ đa cộng tuyến cặp (Pairwise)
VIF_THRESHOLD              = 5        # Ngưỡng an toàn tối đa cho ổn định hệ số hồi quy
MIN_BIN_SIZE               = 0.05       # Tối thiểu 5% tổng quan sát để đảm bảo tính đại diện
MAX_BINS                   = 10         # Giới hạn để duy trì tính đơn giản và tránh Overfitting
MIN_BINS                   = 2          # Bắt buộc phân tách được đặc trưng
DATE_COL                   = "NGAY_BC"
TRAIN_LABEL                = "train"
TEST_LABEL                 = "test"
SPLIT_COL                  = "train_test"
MONOTONIC_ENFORCE          = True       # Bắt buộc để đảm bảo tính giải thích (Logic kinh doanh)
PSI_THRESHOLD              = 0.1       # Ngưỡng "Green Zone" - Biến số ổn định tuyệt đối
ADAPTIVE_SMOOTH            = True       # Tự điều chỉnh WoE cho các bin hiếm (Rare bins)
MIN_SMOOTH_EPS             = 0.05     # Tránh lỗi chia cho 0 trong tính toán WoE/IV
MAX_SMOOTH_EPS             = 0.3       # Giới hạn mức độ can thiệp làm phẳng dữ liệu
MIN_BIN_COUNT              = 50         # Số lượng mẫu tối thiểu tuyệt đối trong 1 bin
MIN_EVENTS_PER_BIN         = 10         # Đảm bảo đủ mẫu Default để tính toán WoE tin cậy
MIN_NON_EVENTS_PER_BIN     = 10         # Đảm bảo đủ mẫu Non-Default
IV_STABILITY_N_BOOTSTRAP   = 50        # Số lần tái mẫu để kiểm định độ ổn định IV
IV_STABILITY_CV_THRESHOLD  = 0.1       # Hệ số biến thiên IV tối đa giữa các lần Bootstrap
IV_STABILITY_CHECK         = True       # Kích hoạt kiểm tra tính ổn định trước khi chọn biến


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_FILE  = r"D:\model\credit_data.parquet"

TARGET_COL  = "NX"
ID_COLS = ["BILLING_ACCTNUM", "CIF" ,"NGAY_BC"]

OUTPUT_FILE_RAW                = r"D:\model\credit_data_woe_raw.parquet"
OUTPUT_FILE_WOE                = r"D:\model\credit_data_woe.parquet"
OUTPUT_FOLDER_RAW_XLSX_BY_DATE = r"D:\model\credit_data_woe"

MAPPING_FOLDER = r"D:\model\mapping"

NUMERIC_DTYPES = [
    np.int8, np.int16, np.int32,
    np.int64, np.float16, np.float32, np.float64
]


# ============================================================
# SPECIAL MIXED NUMERIC VARIABLES
# ============================================================
# Với các biến này:
# - Giá trị đặc biệt được giữ nguyên thành 01 bin riêng.
## - Các giá trị còn lại được xử lý như float và binning numeric.
SPECIAL_FLOAT_BINNING = {
    "LTV_CIF": "tín chấp",
}


SPECIAL_CATEGORICAL_COLUMNS = {
#'PL_BDS_DS': "tín chấp",
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def _normalize_special_value(value):
    """
    Chuẩn hóa một giá trị đặc biệt để phục vụ so sánh.

    Quy tắc:
    - None, NaN và pd.NA được giữ là missing dưới dạng None.
    - Các giá trị còn lại được chuyển thành chuỗi.
    - Loại bỏ khoảng trắng đầu và cuối.
    - Chuyển về chữ thường.
    """
    if value is None:
        return None

    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        is_missing = False

    if isinstance(is_missing, (bool, np.bool_)) and is_missing:
        return None

    return str(value).strip().lower()



def _is_special_value(series, special_value):
    """
    Nhận diện giá trị đặc biệt trong Series.

    So sánh không phân biệt chữ hoa, chữ thường và khoảng trắng
    đầu/cuối. Missing value không được chuyển thành chuỗi "nan",
    "none" hoặc "<NA>".

    Parameters
    ----------
    series : pd.Series
        Series cần kiểm tra.
    special_value : object
        Giá trị đặc biệt cần nhận diện. Nếu là None hoặc missing,
        kết quả sẽ là toàn bộ False.

    Returns
    -------
    pd.Series
        Series boolean giữ nguyên index của đầu vào.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series phải là một pandas Series.")

    result_false = pd.Series(False, index=series.index, dtype=bool)

    normalized_special = _normalize_special_value(special_value)
    if normalized_special is None:
        return result_false

    normalized_series = series.map(_normalize_special_value)

    return (
        normalized_series
        .eq(normalized_special)
        .fillna(False)
        .astype(bool)
    )



def make_box_table(headers, rows, aligns=None):
    aligns = aligns or ["left"] * len(headers)
    rows   = [["" if v is None else str(v) for v in r] for r in rows]

    widths = [
        max(len(str(h)), max([len(r[i]) for r in rows], default=0))
        for i, h in enumerate(headers)
    ]

    def fmt(t, w, a):
        if a == "right":  return t.rjust(w)
        if a == "center": return t.center(w)
        return t.ljust(w)

    top  = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    sep  = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    bot  = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"
    head = "│" + "│".join(
        f" {fmt(str(h), w, 'center')} " for h, w in zip(headers, widths)
    ) + "│"
    body = [
        "│" + "│".join(
            f" {fmt(c, widths[i], aligns[i])} " for i, c in enumerate(r)
        ) + "│"
        for r in rows
    ]

    return "\n".join([top, head, sep] + body + [bot])


def print_bin_table(stats_df, iv_total):
    if stats_df is None or stats_df.empty:
        print("(Không có bảng bin khả dụng cho biến này.)")
        return

    rows = [
        [
            r["Bin"],
            f"{r['SizePct'] * 100:6.2f}%",
            f"{r['WOE']: .6f}",
            f"{r['IV']: .6f}"
        ]
        for _, r in stats_df.iterrows()
    ]
    rows.append(["IV TOTAL", "", "", f"{iv_total: .6f}"])

    print(
        make_box_table(
            ["Bin", "Size (%)", "WOE", "IV riêng"],
            rows,
            ["left", "right", "right", "right"]
        )
    )


def format_interval_label(left, right):
    return f"({left:.6g}, {right:.6g}]"

def _prepare_parquet_chunk(chunk):
    """
    Chuẩn hóa chunk trước khi ghi Parquet:
    - Object/category -> string để PyArrow không phải suy luận mixed type.
    - Giữ nguyên numeric/datetime/bool.
    """
    out = chunk.copy()

    obj_cols = out.select_dtypes(include=["object", "category"]).columns
    for c in obj_cols:
        out[c] = out[c].astype("string")

    return out


def export_parquet_brotli(
    df,
    output_file,
    chunk_size=50000,
    compression="brotli"
):
    """
    Export Parquet dạng chunk để tránh ArrowMemoryError.

    Khác với pandas.to_parquet():
    - Không convert toàn bộ DataFrame sang Arrow Table một lần.
    - Ghi từng chunk nhỏ qua pyarrow.parquet.ParquetWriter.
    - Giảm peak RAM đáng kể.
    """
    print("=" * 60)
    print("EXPORT PARQUET CHUNKED")
    print("=" * 60)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        os.remove(output_file)

    n_rows = df.shape[0]
    n_cols = df.shape[1]

    print(f"File       : {output_file}")
    print(f"Shape      : {df.shape}")
    print(f"Chunk size : {chunk_size:,}")
    print(f"Compression: {compression}")
    print()

    writer = None
    schema = None
    n_chunks = int(np.ceil(n_rows / max(chunk_size, 1)))

    try:
        if n_rows == 0:
            empty_df = _prepare_parquet_chunk(df)
            table = pa.Table.from_pandas(
                empty_df,
                preserve_index=False,
                nthreads=1,
                safe=False
            )
            pq.write_table(
                table,
                output_file,
                compression=compression
            )
            print("Đã xuất DataFrame rỗng.")
            return

        for i, start in enumerate(range(0, n_rows, chunk_size), start=1):
            end = min(start + chunk_size, n_rows)

            chunk = df.iloc[start:end]
            chunk = _prepare_parquet_chunk(chunk)

            if writer is None:
                table = pa.Table.from_pandas(
                    chunk,
                    preserve_index=False,
                    nthreads=1,
                    safe=False
                )
                schema = table.schema
                writer = pq.ParquetWriter(
                    output_file,
                    schema=schema,
                    compression=compression,
                    use_dictionary=True
                )
            else:
                table = pa.Table.from_pandas(
                    chunk,
                    schema=schema,
                    preserve_index=False,
                    nthreads=1,
                    safe=False
                )

            writer.write_table(table, row_group_size=len(chunk))

            print(
                f"  ✓ Chunk {i}/{n_chunks} | "
                f"rows={start:,}-{end-1:,} | "
                f"shape={chunk.shape}"
            )

            del chunk, table
            gc.collect()

    finally:
        if writer is not None:
            writer.close()

    print()
    print(f"Đã xuất: {output_file}")

    if os.path.exists(output_file):
        print(f"Dung lượng: {os.path.getsize(output_file) / 1024**2:.2f} MB")

    print("=" * 60)
    print()
    
    

def format_billing_acctnum(df, col="BILLING_ACCTNUM"):
    if col not in df.columns:
        print(f"CẢNH BÁO: Không tìm thấy cột '{col}' để format.")
        return df

    df_out = df.copy()
    df_out[col] = (
        df_out[col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .apply(lambda x: x.zfill(13) if len(x) <= 13 and x != "" else x)
    )
    return df_out


# ============================================================
# ADAPTIVE SMOOTHING
# ============================================================

def compute_smooth_eps(
    count,
    total,
    min_eps=MIN_SMOOTH_EPS,
    max_eps=MAX_SMOOTH_EPS
):
    """
    Epsilon adaptive theo kích thước bin.
    Bin nhỏ → eps lớn (smooth mạnh).
    Bin lớn → eps nhỏ (ít bias).
    Bảo vệ chia zero: max(total, 1).
    """
    ratio = count / max(total, 1)
    eps   = max_eps - (max_eps - min_eps) * min(ratio / 0.5, 1.0)
    return float(np.clip(eps, min_eps, max_eps))


# ============================================================
# MONOTONICITY CHECK
# ============================================================

def check_monotonic_woe(stats_df):
    """Kiểm tra WOE đơn điệu tăng hoặc giảm."""
    if stats_df is None or len(stats_df) < 2:
        return True

    woe_vals = stats_df["WOE"].tolist()
    diffs    = [woe_vals[i + 1] - woe_vals[i] for i in range(len(woe_vals) - 1)]
    pos      = sum(1 for d in diffs if d >  1e-8)
    neg      = sum(1 for d in diffs if d < -1e-8)

    return pos == 0 or neg == 0


def check_monotonic_event_rate(stats_df):
    """
    Kiểm tra event rate đơn điệu.
    Phát hiện bin nhiễu sau adaptive smoothing.
    """
    if stats_df is None or len(stats_df) < 2:
        return True

    counts = stats_df["Count"].tolist()
    events = stats_df["Events"].tolist()
    er     = [e / max(c, 1) for e, c in zip(events, counts)]
    diffs  = [er[i + 1] - er[i] for i in range(len(er) - 1)]
    pos    = sum(1 for d in diffs if d >  1e-6)
    neg    = sum(1 for d in diffs if d < -1e-6)

    return pos == 0 or neg == 0


# ============================================================
# MONOTONICITY ENFORCEMENT — giữ nguyên từ v4 (Fix C)
# ============================================================

def _parse_interval_label(label_str):
    """Parse label (left, right] → (float, float) hoặc None."""
    try:
        m = re.match(
            r"^[\(\[]\s*(-?[\d.eE+\-inf]+)\s*,\s*(-?[\d.eE+\-inf]+)\s*[\)\]]$",
            str(label_str).strip()
        )
        if m:
            return float(m.group(1)), float(m.group(2))
    except Exception:
        pass
    return None


def _enforce_monotonic_one_direction(
    bin_series,
    y,
    bin_rules,
    is_increasing,
    min_bin_size=MIN_BIN_SIZE,
    min_bins=MIN_BINS
):
    """
    Enforce đơn điệu theo một hướng.

    Returns
    -------
    (labels, order, bin_rules_new, iv_achieved)
    """
    MAX_ITER       = 50
    iteration      = 0
    current_rules  = sorted(bin_rules, key=lambda z: z["left"])
    current_labels = bin_series.copy()
    last_iv        = -np.inf

    while iteration < MAX_ITER:
        iteration += 1
        current_order = [r["label"] for r in current_rules]

        try:
            stats, iv, wmap, _ = compute_woe_iv(
                current_labels, y, bin_order=current_order
            )
            last_iv = iv
        except Exception:
            break

        if stats is None or len(stats) < 2:
            break

        woe_vals  = stats["WOE"].tolist()
        diffs     = [
            woe_vals[i + 1] - woe_vals[i]
            for i in range(len(woe_vals) - 1)
        ]
        pos_count = sum(1 for d in diffs if d >  1e-8)
        neg_count = sum(1 for d in diffs if d < -1e-8)

        if is_increasing and neg_count == 0:
            break
        if not is_increasing and pos_count == 0:
            break

        if len(current_rules) <= min_bins:
            break

        size_pct = stats["SizePct"].tolist()
        worst_i  = None
        worst_v  = float("inf")

        for i, d in enumerate(diffs):
            violation = (
                (is_increasing     and d < -1e-8)
                or (not is_increasing and d >  1e-8)
            )
            if violation:
                sz = min(size_pct[i], size_pct[i + 1])
                if sz < worst_v:
                    worst_v = sz
                    worst_i = i

        if worst_i is None:
            break

        lo = current_rules[worst_i]
        hi = current_rules[worst_i + 1]

        merged_left  = min(lo["left"],  hi["left"])
        merged_right = max(lo["right"], hi["right"])
        merged_label = format_interval_label(merged_left, merged_right)

        merged_rule = {
            "left":  merged_left,
            "right": merged_right,
            "label": merged_label
        }

        current_rules = (
            current_rules[:worst_i]
            + [merged_rule]
            + current_rules[worst_i + 2:]
        )

        label_map = {r["label"]: r["label"] for r in current_rules}
        label_map[lo["label"]] = merged_label
        label_map[hi["label"]] = merged_label

        current_labels = current_labels.map(
            lambda lbl: label_map.get(str(lbl), str(lbl))
        ).astype(object)

        if len(current_rules) < min_bins:
            break

    final_order = [r["label"] for r in current_rules]
    return current_labels, final_order, current_rules, last_iv


def enforce_monotonic_bins_numeric(
    bin_series,
    y,
    bin_rules,
    min_bin_size=MIN_BIN_SIZE,
    min_bins=MIN_BINS
):
    """
    Thử cả 2 hướng, chọn IV cao hơn.
    Trả về (labels, order, rules, iv_best) — Fix C từ v4.
    """
    labels_inc, order_inc, rules_inc, _ = _enforce_monotonic_one_direction(
        bin_series, y, bin_rules,
        is_increasing=True,
        min_bin_size=min_bin_size,
        min_bins=min_bins
    )

    labels_dec, order_dec, rules_dec, _ = _enforce_monotonic_one_direction(
        bin_series, y, bin_rules,
        is_increasing=False,
        min_bin_size=min_bin_size,
        min_bins=min_bins
    )

    iv_inc = -np.inf
    iv_dec = -np.inf

    if len(order_inc) >= min_bins:
        try:
            _, iv_inc, _, _ = compute_woe_iv(
                labels_inc, y, bin_order=order_inc
            )
        except Exception:
            iv_inc = -np.inf

    if len(order_dec) >= min_bins:
        try:
            _, iv_dec, _, _ = compute_woe_iv(
                labels_dec, y, bin_order=order_dec
            )
        except Exception:
            iv_dec = -np.inf

    if iv_inc >= iv_dec and len(order_inc) >= min_bins:
        return labels_inc, order_inc, rules_inc, iv_inc
    elif len(order_dec) >= min_bins:
        return labels_dec, order_dec, rules_dec, iv_dec
    else:
        final_order = [
            r["label"] for r in sorted(bin_rules, key=lambda z: z["left"])
        ]
        try:
            _, iv_orig, _, _ = compute_woe_iv(
                bin_series, y, bin_order=final_order
            )
        except Exception:
            iv_orig = -np.inf
        return bin_series.copy(), final_order, bin_rules, iv_orig


# ============================================================
# PSI COMPUTATION
# ============================================================

def compute_psi(expected_series, actual_series, bins_order):
    """PSI giữa train và test. Không âm."""
    def get_dist(s, order):
        counts = pd.Series(s).value_counts()
        total  = max(len(s), 1)
        return {b: counts.get(b, 0) / total for b in order}

    exp_dist = get_dist(expected_series, bins_order)
    act_dist = get_dist(actual_series,   bins_order)

    psi = 0.0
    for b in bins_order:
        e = max(exp_dist.get(b, 0.0), 1e-8)
        a = max(act_dist.get(b, 0.0), 1e-8)
        psi += (a - e) * np.log(a / e)

    return float(max(psi, 0.0))


# ============================================================
# IV STABILITY CHECK — v4 Fix B
# ============================================================

def check_iv_stability(
    x_series,
    y_series,
    bin_rules,
    kind,
    category_to_bin=None,
    cat_info=None,
    fallback_bin=None,
    n_bootstrap=IV_STABILITY_N_BOOTSTRAP,
    cv_threshold=IV_STABILITY_CV_THRESHOLD,
    random_state=42
):
    """
    Bootstrap IV stability check.
    Fix B: index reset về 0..n-1 trước khi truyền vào
    assign_numeric_bins_for_test.
    """
    rng   = np.random.default_rng(random_state)
    n     = len(y_series)
    x_arr = x_series.values
    y_arr = y_series.values

    iv_list = []

    for _ in range(n_bootstrap):
        boot_idx = rng.integers(0, n, size=n)

        x_boot = pd.Series(
            x_arr[boot_idx],
            index=np.arange(n),
            dtype=x_series.dtype
        )
        y_boot = pd.Series(
            y_arr[boot_idx],
            index=np.arange(n),
            dtype=int
        )

        if y_boot.nunique() < 2:
            continue

        try:
            if kind == "num":
                special_value_boot = SPECIAL_FLOAT_BINNING.get(x_series.name)
                special_mask_boot = _is_special_value(x_boot, special_value_boot)
                
                x_boot_num = pd.to_numeric(
                    x_boot.astype(str)
                          .str.strip()
                          .str.replace(",", "", regex=False)
                          .mask(special_mask_boot, np.nan),
                    errors="coerce"
                )
                
                median_boot = float(x_boot_num.median()) if x_boot_num.notna().any() else 0.0

                labels_boot = assign_numeric_bins_for_test(
                    x_boot,
                    bin_rules,
                    median_boot,
                    special_value=special_value_boot
                )
            else:
                if category_to_bin is None or cat_info is None:
                    continue
                fb = fallback_bin or (
                    list(category_to_bin.values())[0]
                    if category_to_bin
                    else "Unknown"
                )
                labels_boot, _ = assign_categorical_bins_for_test(
                    x_boot, category_to_bin, cat_info, fb
                )

            _, iv_b, _, _ = compute_woe_iv(labels_boot, y_boot)
            if iv_b >= 0:
                iv_list.append(iv_b)

        except Exception:
            continue

    if len(iv_list) < 3:
        return np.nan, np.nan, np.nan, True

    mean_iv = float(np.mean(iv_list))
    std_iv  = float(np.std(iv_list))
    cv      = std_iv / max(mean_iv, 1e-8)
    stable  = cv < cv_threshold

    return mean_iv, std_iv, cv, stable


# ============================================================
# LOAD, VALIDATE, TRAIN/TEST SPLIT
# ============================================================
def detect_numeric_like_columns_on_train(
    df_train,
    target_col=TARGET_COL,
    id_cols=ID_COLS,
    split_col=SPLIT_COL,
    parse_ratio=0.95
):
    """
    Xác định cột numeric-like chỉ trên TRAIN.

    Thứ tự ưu tiên:
    1. Cột trong SPECIAL_CATEGORICAL_COLUMNS luôn là categorical.
    2. Cột trong SPECIAL_FLOAT_BINNING được kiểm tra numeric-like
       sau khi loại special value.
    3. Các cột object/category còn lại áp dụng parse_ratio thông thường.
    """
    numeric_like_cols = []
    parse_summary = []

    excluded_cols = set(
        [target_col, split_col] + list(id_cols)
    )

    for c in df_train.columns:
        if c in excluded_cols:
            continue

        s = df_train[c]

        # Các cột cấu hình categorical tuyệt đối không được chuyển numeric.
        if c in SPECIAL_CATEGORICAL_COLUMNS:
            special_value = SPECIAL_CATEGORICAL_COLUMNS[c]
            special_mask = _is_special_value(
                s,
                special_value
            )

            parse_summary.append({
                "column": c,
                "non_missing_train": int(s.notna().sum()),
                "special_value": special_value,
                "special_count_train": int(special_mask.sum()),
                "parsed_train": 0,
                "parse_ratio_train": np.nan,
                "selected_convert": False,
                "reason": "Cột được cấu hình bắt buộc categorical"
            })
            continue

        if s.dtype not in ["object", "category"]:
            continue

        nonmiss = s.notna()

        special_value = SPECIAL_FLOAT_BINNING.get(c)
        special_mask = _is_special_value(
            s,
            special_value
        )

        check_mask = nonmiss & (~special_mask)

        if check_mask.sum() == 0:
            parse_summary.append({
                "column": c,
                "non_missing_train": int(nonmiss.sum()),
                "special_value": (
                    special_value
                    if special_value is not None
                    else ""
                ),
                "special_count_train": int(special_mask.sum()),
                "parsed_train": 0,
                "parse_ratio_train": np.nan,
                "selected_convert": False,
                "reason": (
                    "Không có giá trị numeric để parse "
                    "sau khi loại special"
                )
            })
            continue

        parsed = pd.to_numeric(
            s.loc[check_mask]
             .astype(str)
             .str.strip()
             .str.replace(",", "", regex=False),
            errors="coerce"
        )

        ratio = (
            parsed.notna().sum()
            / check_mask.sum()
        )
        selected = ratio >= parse_ratio

        if selected:
            numeric_like_cols.append(c)

        parse_summary.append({
            "column": c,
            "non_missing_train": int(nonmiss.sum()),
            "special_value": (
                special_value
                if special_value is not None
                else ""
            ),
            "special_count_train": int(special_mask.sum()),
            "parsed_train": int(parsed.notna().sum()),
            "parse_ratio_train": float(ratio),
            "selected_convert": bool(selected),
            "reason": (
                f"parse_ratio_train >= {parse_ratio}"
                if selected
                else f"parse_ratio_train < {parse_ratio}"
            )
        })

    return numeric_like_cols, pd.DataFrame(parse_summary)
def apply_numeric_like_conversion(
    df,
    numeric_like_cols
):
    """
    Áp dụng schema numeric-like học từ TRAIN cho TRAIN và TEST.

    Các cột thuộc SPECIAL_CATEGORICAL_COLUMNS luôn được giữ nguyên,
    kể cả khi vô tình xuất hiện trong numeric_like_cols.
    """
    out = df.copy()

    for c in numeric_like_cols:
        if c not in out.columns:
            continue

        # Lớp bảo vệ thứ hai cho categorical đặc biệt.
        if c in SPECIAL_CATEGORICAL_COLUMNS:
            continue

        special_value = SPECIAL_FLOAT_BINNING.get(c)

        if special_value is None:
            out[c] = pd.to_numeric(
                out[c]
                .astype(str)
                .str.strip()
                .str.replace(",", "", regex=False)
                .mask(out[c].isna(), np.nan),
                errors="coerce"
            )
        else:
            special_mask = _is_special_value(
                out[c],
                special_value
            )

            converted = pd.to_numeric(
                out[c]
                .astype(str)
                .str.strip()
                .str.replace(",", "", regex=False)
                .mask(out[c].isna(), np.nan)
                .mask(special_mask, np.nan),
                errors="coerce"
            )

            out[c] = converted.astype(object)
            out.loc[special_mask, c] = special_value

    return out



def load_and_validate(input_file=INPUT_FILE, target_col=TARGET_COL):
    print("=" * 60)
    print("ĐỌC FILE & VALIDATE DỮ LIỆU")
    print("=" * 60)

    try:
        df_raw = pd.read_parquet(input_file)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file '{input_file}'.")
        return None
    except Exception as exc:
        print(f"LỖI: Không thể đọc file. Chi tiết: {exc}")
        return None

    print(f"Đọc file thành công: {input_file}")
    print(f"Shape ban đầu: {df_raw.shape}")

    if target_col not in df_raw.columns:
        print(f"LỖI: Không tìm thấy cột target '{target_col}'.")
        print(f"Cột thực tế: {list(df_raw.columns)}")
        return None

    nmiss = df_raw[target_col].isna().sum()
    if nmiss > 0:
        df_raw = df_raw.dropna(subset=[target_col]).copy()
        print(f"Đã loại {nmiss} dòng thiếu target.")

    vals = sorted(pd.Series(df_raw[target_col].dropna().unique()).tolist())
    if set(vals) != {0, 1}:
        print("LỖI: Target không nhị phân 0/1.")
        print(df_raw[target_col].value_counts(dropna=False).to_string())
        return None

    df_raw[target_col] = df_raw[target_col].astype(int)
    print("Phân phối target:")
    print(df_raw[target_col].value_counts().sort_index().to_string())
    print()

    return df_raw

def validate_existing_train_test_split(
    df,
    split_col=SPLIT_COL,
    train_label=TRAIN_LABEL,
    test_label=TEST_LABEL,
    target_col=TARGET_COL
):
    print("=" * 60)
    print("VALIDATE CỘT TRAIN / TEST CÓ SẴN")
    print("=" * 60)

    if split_col not in df.columns:
        print(f"LỖI: Không tìm thấy cột split '{split_col}'.")
        return None

    if target_col not in df.columns:
        print(f"LỖI: Không tìm thấy cột target '{target_col}'.")
        return None

    df_out = df.copy()

    # Kiểm tra missing trước khi astype(str),
    # vì astype(str) sẽ biến NaN thành chuỗi "nan"
    n_missing = int(df_out[split_col].isna().sum())
    if n_missing > 0:
        print(f"LỖI: Cột '{split_col}' có {n_missing} giá trị missing.")
        return None

    # Chuẩn hóa label chuẩn để so sánh nhất quán
    train_label_norm = str(train_label).strip().lower()
    test_label_norm  = str(test_label).strip().lower()

    # Chuẩn hóa nhẹ dữ liệu thực tế để tránh lỗi khoảng trắng / chữ hoa
    df_out[split_col] = (
        df_out[split_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Kiểm tra giá trị rỗng sau khi strip
    n_blank = int((df_out[split_col] == "").sum())
    if n_blank > 0:
        print(f"LỖI: Cột '{split_col}' có {n_blank} giá trị rỗng sau khi strip.")
        return None

    valid_labels = {train_label_norm, test_label_norm}
    actual_labels = set(df_out[split_col].unique())

    invalid_labels = actual_labels - valid_labels
    if invalid_labels:
        print(f"LỖI: Cột '{split_col}' có nhãn không hợp lệ:")
        print(sorted(invalid_labels))
        print(f"Nhãn hợp lệ yêu cầu: {sorted(valid_labels)}")
        return None

    print("Phân phối train/test:")
    print(df_out[split_col].value_counts(dropna=False).to_string())
    print()

    print("Cross-tab train/test x target:")
    print(
        pd.crosstab(
            df_out[split_col],
            df_out[target_col],
            margins=True
        ).to_string()
    )
    print()

    if train_label_norm not in actual_labels:
        print(f"LỖI: Không có dòng '{train_label_norm}'.")
        return None

    if test_label_norm not in actual_labels:
        print(f"CẢNH BÁO: Không có dòng '{test_label_norm}'. Pipeline chỉ chạy train.")
        print()

    return df_out



# ============================================================
# STEP 1 — SPLIT FEATURES
# ============================================================
def split_features(
    df,
    target_col=TARGET_COL,
    id_cols=ID_COLS,
    split_col=SPLIT_COL
):
    """
    Tách biến numeric và categorical.

    Ưu tiên:
    - SPECIAL_FLOAT_BINNING luôn thuộc numeric/mixed numeric.
    - SPECIAL_CATEGORICAL_COLUMNS luôn thuộc categorical.
    - Không cho một cột xuất hiện đồng thời trong cả hai nhóm.
    """
    print("=" * 60)
    print("BƯỚC 1: TÁCH BIẾN")
    print("=" * 60)

    drop_cols = [
        target_col
    ] + [
        c for c in id_cols
        if c in df.columns
    ]

    if split_col in df.columns:
        drop_cols.append(split_col)

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    numeric_cols = list(
        X.select_dtypes(
            include=NUMERIC_DTYPES
        ).columns
    )

    cat_cols = list(
        X.select_dtypes(
            include=["object", "category", "string"]
        ).columns
    )

    # Mixed numeric luôn thuộc nhóm numeric.
    for c in SPECIAL_FLOAT_BINNING:
        if c not in X.columns:
            continue

        if c not in numeric_cols:
            numeric_cols.append(c)

        if c in cat_cols:
            cat_cols.remove(c)

    # Categorical đặc biệt luôn thuộc nhóm categorical,
    # kể cả khi dữ liệu hiện tại chỉ chứa chuỗi dạng số
    # hoặc dtype đã bị suy luận thành numeric.
    for c in SPECIAL_CATEGORICAL_COLUMNS:
        if c not in X.columns:
            continue

        if c in numeric_cols:
            numeric_cols.remove(c)

        if c not in cat_cols:
            cat_cols.append(c)

    numeric_cols = list(dict.fromkeys(numeric_cols))
    cat_cols = list(dict.fromkeys(cat_cols))

    overlap = set(numeric_cols) & set(cat_cols)

    if overlap:
        raise ValueError(
            "Các cột xuất hiện đồng thời trong X_num và X_cat: "
            f"{sorted(overlap)}"
        )

    X_num = X[numeric_cols].copy()
    X_cat = X[cat_cols].copy()

    print(f"Số dòng     : {df.shape[0]}")
    print(
        f"Numeric     : {X_num.shape[1]} — "
        f"{list(X_num.columns)}"
    )
    print(
        f"Categorical : {X_cat.shape[1]} — "
        f"{list(X_cat.columns)}"
    )
    print()

    out = {
        "X_num": X_num,
        "X_cat": X_cat,
        "X_all": X.copy(),
        "y": y,
        "id_data": df[
            [
                c for c in id_cols
                if c in df.columns
            ]
        ].copy()
    }

    if split_col in df.columns:
        out["train_test"] = df[split_col]

    return out


# ============================================================
# WOE / IV FUNCTIONS
# ============================================================

def compute_woe_iv(
    bin_series,
    y,
    bin_order=None,
    adaptive_smooth=ADAPTIVE_SMOOTH,
    min_eps=MIN_SMOOTH_EPS,
    max_eps=MAX_SMOOTH_EPS
):
    """
    Tính WOE và IV.
    - Adaptive smoothing per-bin (Fix v2).
    - te_adj/tn_adj tính lại → EventDist tổng = 1 (Fix v2).
    - IV không âm (Fix v2).
    """
    tmp = pd.DataFrame({
        "Bin": bin_series.astype(object),
        "y":   y.astype(int)
    })

    if bin_order is not None:
        tmp["Bin"] = pd.Categorical(
            tmp["Bin"], categories=bin_order, ordered=True
        )

    g = (
        tmp.groupby("Bin", observed=True, sort=True)["y"]
           .agg(Count="count", Events="sum")
           .reset_index()
    )

    g["Bin"]       = g["Bin"].astype(str)
    g["NonEvents"] = g["Count"] - g["Events"]

    te = int(g["Events"].sum())
    tn = int(g["NonEvents"].sum())

    if te == 0 or tn == 0:
        raise ZeroDivisionError(
            "Target phải có cả event=1 và non-event=0."
        )

    total_count = int(g["Count"].sum())
    smoothed    = bool(((g["Events"] == 0) | (g["NonEvents"] == 0)).any())

    ev_adj  = []
    nev_adj = []

    for _, row in g.iterrows():
        cnt = int(row["Count"])
        e   = int(row["Events"])
        ne  = int(row["NonEvents"])

        if e == 0 or ne == 0:
            eps = (
                compute_smooth_eps(cnt, total_count, min_eps, max_eps)
                if adaptive_smooth else 0.5
            )
            e_adj  = float(e)  + eps if e  == 0 else float(e)
            ne_adj = float(ne) + eps if ne == 0 else float(ne)
        else:
            e_adj  = float(e)
            ne_adj = float(ne)

        ev_adj.append(e_adj)
        nev_adj.append(ne_adj)

    te_adj = sum(ev_adj)
    tn_adj = sum(nev_adj)

    g["EventDist"]    = [e  / te_adj for e  in ev_adj]
    g["NonEventDist"] = [ne / tn_adj for ne in nev_adj]
    g["WOE"]          = np.log(
        np.array(g["EventDist"]) / np.array(g["NonEventDist"])
    )
    g["IV"]      = (
        (np.array(g["EventDist"]) - np.array(g["NonEventDist"]))
        * np.array(g["WOE"])
    )
    g["SizePct"] = g["Count"] / len(y)

    iv = float(g["IV"].sum())
    if abs(iv) < 1e-10:
        iv = 0.0
    elif iv < 0:
        print(
            f"CẢNH BÁO: IV âm ({iv:.6f}); kiểm tra binning/smoothing."
        )

    return (
        g[["Bin", "Count", "Events", "NonEvents", "SizePct", "WOE", "IV"]],
        iv,
        dict(zip(g["Bin"], g["WOE"])),
        smoothed
    )


def merge_interval_bins(
    bin_series,
    y,
    min_bin_size=MIN_BIN_SIZE,
    min_bins=MIN_BINS,
    min_events=MIN_EVENTS_PER_BIN,
    min_non_events=MIN_NON_EVENTS_PER_BIN
):
    """
    [v5 Fix] Merge bin nhỏ vào bin liền kề.

    Fix v5: max_merge_iter dùng initial_n_groups (captured trước
    vòng lặp) thay vì len(groups) động — giới hạn tight và đúng
    về lý thuyết.

    Mỗi lần merge giảm len(groups) đi 1, nên tối đa cần
    (initial_n_groups - min_bins) lần merge. Thêm buffer nhỏ (+10)
    để an toàn với edge cases.

    Thứ tự ưu tiên merge (Fix D từ v4):
    1. Bin vi phạm min_events hoặc min_non_events.
    2. Bin vi phạm min_count.
    """
    min_count = max(
        int(np.ceil(min_bin_size * len(y))),
        MIN_BIN_COUNT
    )

    tmp_df = pd.DataFrame({"bin": bin_series, "y": y.values})
    agg    = tmp_df.groupby("bin", observed=True)["y"].agg(
        ["count", "sum"]
    )

    groups = []
    for c in bin_series.cat.categories:
        if c not in agg.index or int(agg.loc[c, "count"]) == 0:
            continue
        cnt = int(agg.loc[c, "count"])
        ev  = int(agg.loc[c, "sum"])
        groups.append({
            "left":       float(c.left),
            "right":      float(c.right),
            "count":      cnt,
            "events":     ev,
            "non_events": cnt - ev,
            "members":    [c]
        })

    # [v5 Fix] Capture initial length TRƯỚC vòng lặp
    initial_n_groups = len(groups)
    # Tối đa cần merge (initial - min_bins) lần, +10 buffer
    max_merge_iter   = max(initial_n_groups - min_bins, 0) + 10
    merge_iter       = 0

    while len(groups) > min_bins and merge_iter < max_merge_iter:
        merge_iter += 1

        violators_ev = [
            (k, gr) for k, gr in enumerate(groups)
            if gr["events"] < min_events
            or gr["non_events"] < min_non_events
        ]
        violators_cnt = [
            (k, gr) for k, gr in enumerate(groups)
            if gr["count"] < min_count
        ]

        if violators_ev:
            i = min(violators_ev, key=lambda x: x[1]["count"])[0]
        elif violators_cnt:
            i = min(violators_cnt, key=lambda x: x[1]["count"])[0]
        else:
            break

        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:
            j = (
                i - 1
                if groups[i - 1]["count"] <= groups[i + 1]["count"]
                else i + 1
            )

        lo, hi = sorted([i, j])
        merged = {
            "left":       min(groups[lo]["left"],       groups[hi]["left"]),
            "right":      max(groups[lo]["right"],      groups[hi]["right"]),
            "count":      groups[lo]["count"]      + groups[hi]["count"],
            "events":     groups[lo]["events"]     + groups[hi]["events"],
            "non_events": groups[lo]["non_events"] + groups[hi]["non_events"],
            "members":    groups[lo]["members"]    + groups[hi]["members"]
        }
        groups = groups[:lo] + [merged] + groups[hi + 1:]

    mapping   = {}
    order     = []
    bin_rules = []

    for gr in groups:
        lab = format_interval_label(gr["left"], gr["right"])
        order.append(lab)
        bin_rules.append({
            "left":  gr["left"],
            "right": gr["right"],
            "label": lab
        })
        for m in gr["members"]:
            mapping[m] = lab

    return bin_series.map(mapping).astype(object), order, bin_rules


def build_numeric_candidate(
    x,
    y,
    n_bins,
    min_bin_size=MIN_BIN_SIZE,
    min_bins=MIN_BINS,
    monotonic_enforce=MONOTONIC_ENFORCE,
    special_value=None
):
    """
    Tạo candidate binning numeric.

    Nếu special_value khác None, ví dụ LTV = 'tín chấp':
    - special_value được giữ nguyên thành 01 bin riêng.
    - phần còn lại được convert float và binning numeric.
    - monotonicity chỉ enforce trên phần numeric, không ép lên bin special.
    """
    x_obj = x.copy()

    special_mask = _is_special_value(x_obj, special_value)
    numeric_mask = ~special_mask

    x_num_all = pd.to_numeric(
        x_obj.astype(str)
             .str.strip()
             .str.replace(",", "", regex=False)
             .mask(special_mask, np.nan),
        errors="coerce"
    )
    bad_numeric_mask = (
        x_obj.notna()
        & (~special_mask)
        & x_num_all.isna()
    )
    
    if bad_numeric_mask.any():
        print(
            f"CẢNH BÁO: {getattr(x, 'name', '')} có "
            f"{int(bad_numeric_mask.sum())} giá trị không parse được numeric "
            f"và không phải special_value='{special_value}'. "
            "Các giá trị này sẽ được fill median."
        )    
    median_value = (
        float(x_num_all.loc[numeric_mask].median())
        if x_num_all.loc[numeric_mask].notna().any()
        else 0.0
    )

    x_num_all = x_num_all.fillna(median_value)

    x_num = x_num_all.loc[numeric_mask]
    y_num = y.loc[numeric_mask]

    if len(x_num) == 0:
        return None

    if x_num.nunique(dropna=True) < min_bins:
        return None

    q = min(n_bins, x_num.nunique(dropna=True))
    binned = None

    try:
        binned = pd.qcut(x_num, q=q, duplicates="drop")
        if len(binned.cat.categories) < min_bins:
            binned = None
    except Exception:
        binned = None

    if binned is None:
        try:
            binned = pd.cut(x_num, bins=q, duplicates="drop")
            if len(binned.cat.categories) < min_bins:
                return None
        except Exception:
            return None

    labels_num, order_num, bin_rules = merge_interval_bins(
        binned,
        y_num,
        min_bin_size=min_bin_size,
        min_bins=min_bins,
        min_events=MIN_EVENTS_PER_BIN,
        min_non_events=MIN_NON_EVENTS_PER_BIN
    )

    if len(order_num) < min_bins:
        return None

    try:
        stats_num, iv_num, wmap_num, sm_num = compute_woe_iv(
            labels_num,
            y_num,
            bin_order=order_num
        )
    except Exception:
        return None

    if not check_monotonic_event_rate(stats_num):
        print(
            f"      [v5] Event rate numeric-part không đơn điệu "
            f"tại n_bins={n_bins}."
        )

    if (
        monotonic_enforce
        and not check_monotonic_woe(stats_num)
        and len(order_num) > min_bins
    ):
        try:
            labels_m, order_m, rules_m, iv_m = enforce_monotonic_bins_numeric(
                labels_num,
                y_num,
                bin_rules,
                min_bin_size=min_bin_size,
                min_bins=min_bins
            )

            if len(order_m) >= min_bins and iv_m >= 0:
                labels_num, order_num, bin_rules = labels_m, order_m, rules_m
        except Exception:
            pass

    labels_all = pd.Series(index=x.index, dtype=object)
    labels_all.loc[numeric_mask] = labels_num.astype(str)

    order_all = list(order_num)

    special_label = str(special_value) if special_value is not None else None
    special_seen_train = bool(special_mask.any())
    
    if special_value is not None and special_seen_train:
        labels_all.loc[special_mask] = special_label
        order_all.append(special_label)

    try:
        stats, iv, wmap, sm = compute_woe_iv(
            labels_all,
            y,
            bin_order=order_all
        )
    except Exception:
        return None
    
    # ========================================================
    # Special bin exemption:
    # - min_size thực tế vẫn tính trên toàn bộ bin, gồm cả "tín chấp".
    # - min_size_select dùng để xét chọn biến, loại riêng special bin.
    #   Nhờ vậy "tín chấp" được giữ dù nhỏ hơn MIN_BIN_SIZE.
    # ========================================================
    min_size_actual = float(stats["SizePct"].min())
    
    special_size_pct = None
    if special_value is not None and special_seen_train and special_label is not None:
        special_rows = stats[stats["Bin"].astype(str) == str(special_label)]
        if not special_rows.empty:
            special_size_pct = float(special_rows["SizePct"].iloc[0])
    
        stats_for_select = stats[stats["Bin"].astype(str) != str(special_label)]
    
        if stats_for_select.empty:
            min_size_select = min_size_actual
        else:
            min_size_select = float(stats_for_select["SizePct"].min())
    else:
        min_size_select = min_size_actual
    
    return {
        "binned":             labels_all,
        "stats":              stats,
        "iv":                 iv,
        "woe_map":            wmap,
        "n_bins":             len(order_all),
        "min_size":           min_size_actual,
        "min_size_select":    min_size_select,
        "special_size_pct":   special_size_pct,
        "smoothed":           sm,
        "bin_rules":          bin_rules,
        "median":             median_value,
        "special_value":      special_value,
        "special_label":      special_label,
        "special_seen_train": special_seen_train
    }

def preprocess_categorical_series(
    x,
    min_bin_size=MIN_BIN_SIZE,
    return_info=False,
    special_value=None
):
    """
    Chuẩn hóa categorical và xử lý missing.

    Nếu có special_value:
    - Nhận diện không phân biệt hoa/thường và khoảng trắng thừa.
    - Chuẩn hóa special value về một label thống nhất.
    - Không biến missing thành chuỗi "nan" hoặc "none".
    - Lưu đầy đủ metadata để áp dụng cho TEST.
    """
    x_obj = x.astype(object).copy()

    normalized_special = _normalize_special_value(
        special_value
    )

    special_label = (
        str(special_value).strip()
        if normalized_special is not None
        else None
    )

    special_mask = _is_special_value(
        x_obj,
        special_value
    )

    if special_label is not None:
        x_obj.loc[special_mask] = special_label

    missing_rate = float(x_obj.isna().mean())

    if missing_rate >= min_bin_size:
        out = (
            x_obj
            .where(x_obj.notna(), "Missing")
            .astype(str)
        )

        info = {
            "missing_policy": "missing_as_category",
            "fill_value": "Missing"
        }
    else:
        # Không ưu tiên special value làm mode để fill missing.
        mode = x_obj.loc[~special_mask].mode(
            dropna=True
        )

        if mode.empty:
            mode = x_obj.mode(dropna=True)

        fill_value = (
            mode.iloc[0]
            if not mode.empty
            else "Missing"
        )

        out = (
            x_obj
            .fillna(fill_value)
            .astype(str)
        )

        info = {
            "missing_policy": "fill_mode",
            "fill_value": str(fill_value)
        }

    info.update({
        "special_value": special_value,
        "special_label": special_label,
        "special_seen_train": bool(special_mask.any())
    })

    if return_info:
        return out, info

    return out


def _try_categorical_merge_both_sides(groups, min_bins):
    """
    [v4 Fix A] Thử merge bin nhỏ nhất vào cả hai phía.
    Trả về (groups_new, changed: bool).
    """
    i = int(np.argmin([gr["count"] for gr in groups]))

    candidates = []

    if i > 0:
        gl     = [gr.copy() for gr in groups]
        lo, hi = i - 1, i
        tot    = gl[lo]["count"] + gl[hi]["count"]
        ml     = {
            "cats":     gl[lo]["cats"] + gl[hi]["cats"],
            "count":    tot,
            "mean_woe": (
                gl[lo]["mean_woe"] * gl[lo]["count"]
                + gl[hi]["mean_woe"] * gl[hi]["count"]
            ) / tot
        }
        candidates.append(gl[:lo] + [ml] + gl[hi + 1:])

    if i < len(groups) - 1:
        gr_list = [gr.copy() for gr in groups]
        lo, hi  = i, i + 1
        tot     = gr_list[lo]["count"] + gr_list[hi]["count"]
        mr      = {
            "cats":     gr_list[lo]["cats"] + gr_list[hi]["cats"],
            "count":    tot,
            "mean_woe": (
                gr_list[lo]["mean_woe"] * gr_list[lo]["count"]
                + gr_list[hi]["mean_woe"] * gr_list[hi]["count"]
            ) / tot
        }
        candidates.append(gr_list[:lo] + [mr] + gr_list[hi + 1:])

    if not candidates:
        return groups, False

    def woe_range(g):
        woes = [gr["mean_woe"] for gr in g]
        return (max(woes) - min(woes)) if len(woes) > 1 else 0.0

    best = max(candidates, key=woe_range)
    return best, True


def build_categorical_candidate(
    x,
    y,
    n_bins,
    min_bin_size=MIN_BIN_SIZE,
    min_bins=MIN_BINS
):
    """
    Tạo candidate categorical binning.

    Đối với cột thuộc SPECIAL_CATEGORICAL_COLUMNS:
    - Special value được giữ thành bin riêng.
    - Special bin không tham gia phân nhóm hoặc merge.
    - Special bin được miễn điều kiện MIN_BIN_SIZE.
    - Các bin categorical còn lại vẫn phải đáp ứng MIN_BIN_SIZE.
    - Nếu special không xuất hiện trong TRAIN thì không tạo WOE
      cho special từ TEST.
    """
    special_value = SPECIAL_CATEGORICAL_COLUMNS.get(
        x.name
    )

    x_cat, cat_info = preprocess_categorical_series(
        x,
        min_bin_size=min_bin_size,
        return_info=True,
        special_value=special_value
    )

    special_label = cat_info["special_label"]
    special_seen_train = cat_info["special_seen_train"]

    if special_label is not None:
        special_mask = x_cat.eq(special_label)
    else:
        special_mask = pd.Series(
            False,
            index=x_cat.index,
            dtype=bool
        )

    regular_mask = ~special_mask
    x_regular = x_cat.loc[regular_mask]
    y_regular = y.loc[regular_mask]

    # Nếu special đã xuất hiện thì tổng số bin tối thiểu có thể gồm:
    # 1 special bin + các regular bin còn lại.
    min_regular_bins = max(
        min_bins - int(special_seen_train),
        1
    )

    if (
        x_regular.nunique(dropna=True)
        < min_regular_bins
    ):
        return None

    try:
        single, _, single_map, _ = compute_woe_iv(
            x_regular,
            y_regular,
            bin_order=sorted(x_regular.unique())
        )
    except Exception:
        return None

    info = single[
        ["Bin", "Count"]
    ].rename(
        columns={"Bin": "Category"}
    )

    info["WOE_SINGLE"] = info["Category"].map(
        single_map
    )

    info = info.sort_values(
        ["WOE_SINGLE", "Category"]
    ).reset_index(drop=True)

    max_regular_bins = max(
        n_bins - int(special_seen_train),
        min_regular_bins
    )

    g = min(
        max_regular_bins,
        len(info)
    )

    if g < min_regular_bins:
        return None

    total = info["Count"].sum()

    if total <= 0:
        return None

    info["midpoint"] = (
        info["Count"].cumsum()
        - info["Count"] / 2.0
    )

    info["group_id"] = (
        np.floor(
            info["midpoint"] / total * g
        )
        .astype(int)
        .clip(0, g - 1)
    )

    groups = []

    for _, part in info.groupby(
        "group_id",
        sort=True
    ):
        groups.append({
            "cats": part["Category"].tolist(),
            "count": int(part["Count"].sum()),
            "mean_woe": float(
                np.average(
                    part["WOE_SINGLE"],
                    weights=part["Count"]
                )
            )
        })

    min_count = max(
        int(np.ceil(min_bin_size * len(y))),
        MIN_BIN_COUNT
    )

    initial_n_groups = len(groups)

    max_merge_iter = (
        max(
            initial_n_groups - min_regular_bins,
            0
        )
        + 10
    )

    merge_iter = 0

    # Chỉ merge các regular groups.
    # Special bin chưa được đưa vào groups nên không thể bị merge.
    while (
        len(groups) > min_regular_bins
        and min(
            gr["count"]
            for gr in groups
        ) < min_count
        and merge_iter < max_merge_iter
    ):
        merge_iter += 1

        groups, changed = (
            _try_categorical_merge_both_sides(
                groups,
                min_regular_bins
            )
        )

        if not changed:
            break

        if min(
            gr["count"]
            for gr in groups
        ) >= min_count:
            break

    mapping = {}
    order = []

    for group in groups:
        label = "+".join(
            sorted(group["cats"])
        )

        order.append(label)

        for category in group["cats"]:
            mapping[category] = label

    # Chỉ bổ sung special bin sau khi regular grouping hoàn tất.
    if special_seen_train:
        mapping[special_label] = special_label
        order.append(special_label)

    labels = x_cat.map(mapping).astype(object)

    if labels.isna().any():
        return None

    if len(order) < min_bins:
        return None

    try:
        stats, iv, wmap, smoothed = compute_woe_iv(
            labels,
            y,
            bin_order=order
        )
    except Exception:
        return None

    # Fallback chỉ lấy từ regular category, không dùng special
    # làm fallback cho category mới.
    fallback_category = (
        info.sort_values(
            "Count",
            ascending=False
        )
        .iloc[0]["Category"]
    )

    fallback_bin = mapping.get(
        fallback_category
    )

    min_size_actual = float(
        stats["SizePct"].min()
    )

    special_size_pct = None

    if special_seen_train:
        special_rows = stats[
            stats["Bin"].astype(str)
            == str(special_label)
        ]

        if not special_rows.empty:
            special_size_pct = float(
                special_rows["SizePct"].iloc[0]
            )

        # Loại special bin khỏi điều kiện MIN_BIN_SIZE.
        stats_for_select = stats[
            stats["Bin"].astype(str)
            != str(special_label)
        ]

        if stats_for_select.empty:
            min_size_select = min_size_actual
        else:
            min_size_select = float(
                stats_for_select["SizePct"].min()
            )
    else:
        min_size_select = min_size_actual

    return {
        "binned": labels,
        "stats": stats,
        "iv": iv,
        "woe_map": wmap,
        "n_bins": len(order),

        # Kích thước bin nhỏ nhất thực tế, có gồm special bin.
        "min_size": min_size_actual,

        # Giá trị dùng để xét chọn biến, không gồm special bin.
        "min_size_select": min_size_select,

        "special_size_pct": special_size_pct,
        "smoothed": smoothed,
        "category_to_bin": mapping,
        "cat_info": cat_info,
        "fallback_category": fallback_category,
        "fallback_bin": fallback_bin,
        "special_value": special_value,
        "special_label": special_label,
        "special_seen_train": special_seen_train
    }


def rejection_reason(
    cand,
    iv_threshold=IV_THRESHOLD,
    min_bin_size=MIN_BIN_SIZE
):
    if cand is None:
        return (
            "Biến chỉ có 1 giá trị duy nhất hoặc "
            "không tạo được tối thiểu 2 bin hợp lệ"
        )

    min_size_eval = cand.get("min_size_select", cand["min_size"])

    iv_fail   = cand["iv"] < iv_threshold
    size_fail = min_size_eval < min_bin_size

    special_note = ""
    if cand.get("special_value") is not None and cand.get("special_seen_train", False):
        special_note = (
            f"; special_bin='{cand.get('special_label')}' "
            f"size={cand.get('special_size_pct', np.nan) * 100:.2f}% "
            "được miễn MIN_BIN_SIZE"
        )

    if iv_fail and size_fail:
        return (
            f"Cả hai: IV={cand['iv']:.6f}<{iv_threshold} "
            f"và min_size_xét_chọn={min_size_eval*100:.2f}%"
            f"<{min_bin_size*100:.0f}%"
            f"{special_note}"
        )

    if iv_fail:
        return f"IV={cand['iv']:.6f} < {iv_threshold}{special_note}"

    if size_fail:
        return (
            f"min_size_xét_chọn={min_size_eval*100:.2f}% "
            f"< {min_bin_size*100:.0f}%"
            f"{special_note}"
        )

    return "Không có n_bins nào thỏa đồng thời IV và min_bin_size"



# ============================================================
# STEP 2 — AUTO BINNING AND VARIABLE SELECTION ON TRAIN ONLY
# ============================================================

def auto_bin_select(
    df1,
    iv_threshold=IV_THRESHOLD,
    min_bin_size=MIN_BIN_SIZE,
    max_bins=MAX_BINS,
    min_bins=MIN_BINS
):
    """
    Tự động binning và chọn biến — CHỈ trên TRAIN.
    Thử TẤT CẢ n_bins (Fix 1 v3), chọn IV cao nhất.
    IV stability bootstrap (Fix 4 v3 + Fix B v4).
    """
    print("=" * 60)
    print("BƯỚC 2: AUTO BINNING & CHỌN BIẾN TRÊN TRAIN (v5)")
    print("=" * 60)
    print()

    X_num = df1["X_num"]
    X_cat = df1["X_cat"]
    X_all = df1["X_all"]
    y     = df1["y"]

    selected     = []
    rejected     = []
    stats        = {}
    wmaps        = {}
    ivs          = {}
    reasons      = {}
    xb           = pd.DataFrame(index=y.index)
    opt_bins     = {}

    min_sizes        = {}
    actual_min_sizes = {}
    special_sizes    = {}
    smooth_vars      = []

    binning_meta = {}
    iv_stability = {}

    feature_list = (
        [(c, "num") for c in X_num.columns]
        + [(c, "cat") for c in X_cat.columns]
    )

    for col, kind in feature_list:
        best      = None
        all_valid = []

        for n in range(max_bins, min_bins - 1, -1):
            if kind == "num":
                cand = build_numeric_candidate(
                    X_num[col],
                    y,
                    n,
                    min_bin_size,
                    min_bins,
                    special_value=SPECIAL_FLOAT_BINNING.get(col)
                )

            else:
                cand = build_categorical_candidate(
                    X_cat[col], y, n, min_bin_size, min_bins
                )

            if cand is None:
                continue

            if best is None or cand["iv"] > best["iv"]:
                best = cand

            cand_min_size_select = cand.get("min_size_select", cand["min_size"])
            
            if (
                cand_min_size_select >= min_bin_size
                and cand["iv"] >= iv_threshold
            ):
                all_valid.append(cand)


        chosen = max(all_valid, key=lambda c: c["iv"]) if all_valid else None
        final  = chosen if chosen is not None else best

        if chosen is not None:
            selected.append(col)
            xb[col] = chosen["binned"]


            if kind == "num":
                fallback_bin = (
                    chosen["bin_rules"][0]["label"]
                    if chosen.get("bin_rules")
                    else None
                )
            
                binning_meta[col] = {
                    "kind": "num",
                    "bin_rules": chosen["bin_rules"],
                    "median": chosen["median"],
                    "woe_map": chosen["woe_map"],
                    "special_value": chosen.get("special_value"),
                    "special_label": chosen.get("special_label"),
                    "special_seen_train": chosen.get(
                        "special_seen_train",
                        False
                    ),
                    "fallback_bin": fallback_bin
                }

            else:
                binning_meta[col] = {
                    "kind":              "cat",
                    "category_to_bin":   chosen["category_to_bin"],
                    "cat_info":          chosen["cat_info"],
                    "fallback_category": chosen["fallback_category"],
                    "fallback_bin":      chosen["fallback_bin"],
                    "woe_map":           chosen["woe_map"]
                }
        else:
            rejected.append(col)
            reasons[col] = rejection_reason(final, iv_threshold, min_bin_size)

        if final is not None:
            stats[col]     = final["stats"]
            wmaps[col]     = final["woe_map"]
            ivs[col]       = final["iv"]
            opt_bins[col]  = final["n_bins"]
            
            min_sizes[col]        = final.get("min_size_select", final["min_size"])
            actual_min_sizes[col] = final["min_size"]
            special_sizes[col]    = final.get("special_size_pct")

            if final["smoothed"]:
                smooth_vars.append(col)

    # ---- Print selected ----
    print(f"✅ BIẾN ĐƯỢC CHỌN: [{len(selected)} biến]")
    print("-" * 60)


    if selected:
        for col in selected:
            print(f"Biến: {col}")
    
            msg = (
                f"IV={ivs[col]:.6f}>={iv_threshold} | "
          f"bins={opt_bins[col]} | "
                f"min_size_xét_chọn={min_sizes[col]*100:.2f}%"
                f">={min_bin_size*100:.0f}%"
            )
    
            if special_sizes.get(col) is not None:
                msg += (
                    f" | min_size_thực_tế={actual_min_sizes[col]*100:.2f}%"
                    f" | special_bin_size={special_sizes[col]*100:.2f}%"
                    " [special bin miễn MIN_BIN_SIZE]"
                )
    
            print(msg)
    
            if col in smooth_vars:
                print("Ghi chú: Adaptive smoothing áp dụng.")
    
            print_bin_table(stats[col], ivs[col])
            print()
    else:
        print("Không có biến nào thỏa điều kiện.\n")


    # ---- Print rejected ----
    print(f"❌ BIẾN BỊ LOẠI: [{len(rejected)} biến]")
    print("-" * 60)

    if rejected:
        for col in rejected:
            print(f"Biến: {col} | Lý do: {reasons[col]}")
            if col in smooth_vars:
                print("Ghi chú: Adaptive smoothing áp dụng.")
            print_bin_table(stats.get(col), ivs.get(col, 0.0))
            print()
    else:
        print("Không có biến nào bị loại ở Bước 2.\n")

    if not selected:
        print(
            "CẢNH BÁO: Tất cả bị loại. "
            "Đề xuất giảm IV_THRESHOLD/MIN_BIN_SIZE.\n"
        )

    # ---- IV Stability Check ----
    if IV_STABILITY_CHECK and selected:
        print("=" * 60)
        print("BƯỚC 2.5: IV STABILITY CHECK (BOOTSTRAP) — v5")
        print("=" * 60)
        print(
            f"Bootstrap n={IV_STABILITY_N_BOOTSTRAP} | "
            f"CV threshold={IV_STABILITY_CV_THRESHOLD:.0%}\n"
        )

        stability_rows = []

        for col in selected:
            meta  = binning_meta[col]
            kind  = meta["kind"]
            x_col = X_num[col] if kind == "num" else X_cat[col]

            mean_iv, std_iv, cv, stable = check_iv_stability(
                x_series=x_col,
                y_series=y,
                bin_rules=meta.get("bin_rules"),
                kind=kind,
                category_to_bin=meta.get("category_to_bin"),
                cat_info=meta.get("cat_info"),
                fallback_bin=meta.get("fallback_bin"),
                n_bootstrap=IV_STABILITY_N_BOOTSTRAP,
                cv_threshold=IV_STABILITY_CV_THRESHOLD
            )

            iv_stability[col] = (mean_iv, std_iv, cv, stable)
            status = "✅ ỔN ĐỊNH" if stable else "⚠️ KHÔNG ỔN ĐỊNH"

            stability_rows.append([
                col,
                f"{ivs[col]:.6f}",
                f"{mean_iv:.6f}" if not np.isnan(mean_iv) else "N/A",
                f"{std_iv:.6f}"  if not np.isnan(std_iv)  else "N/A",
                f"{cv:.2%}"      if not np.isnan(cv)       else "N/A",
                status
            ])

        print(
            make_box_table(
                ["Biến", "IV train", "Mean IV",
                 "Std IV", "CV", "Trạng thái"],
                stability_rows,
                ["left", "right", "right", "right", "right", "left"]
            )
        )
        print()

        unstable = [c for c in selected if not iv_stability[c][3]]
        if unstable:
            print(
                f"CẢNH BÁO: {len(unstable)} biến "
                f"CV>={IV_STABILITY_CV_THRESHOLD:.0%}:"
            )
            for v in unstable:
                print(f"  - {v}")
            print(
                "Ghi chú: Không tự động loại. "
                "Cần business judgment.\n"
            )
        else:
            print(
                f"Tất cả biến CV<{IV_STABILITY_CV_THRESHOLD:.0%}: "
                "ổn định.\n"
            )

    out = {
        "selected_vars":     selected,
        "bin_stats":         {k: stats[k] for k in selected},
        "woe_maps":          {k: wmaps[k] for k in selected},
        "X_binned":          (
            xb[selected] if selected
            else pd.DataFrame(index=y.index)
        ),
        "X_raw_selected":    (
            X_all[selected].copy() if selected
            else pd.DataFrame(index=y.index)
        ),
        "y":                 y,
        "iv_scores":         {k: ivs[k] for k in selected},
        "rejected_vars":     rejected,
        "rejection_reasons": reasons,
        "binning_meta":      binning_meta,
        "iv_stability":      iv_stability
    }

    if "id_data" in df1:
        out["id_data"] = df1["id_data"]
    if "train_test" in df1:
        out["train_test"] = df1["train_test"]

    return out


# ============================================================
# STEP 3 — WOE TRANSFORMATION ON TRAIN
# ============================================================

def woe_transform(df2):
    print("=" * 60)
    print("BƯỚC 3: WOE TRANSFORMATION - TRAIN")
    print("=" * 60)

    df3 = pd.DataFrame(index=df2["y"].index)

    if "id_data" in df2:
        for c in df2["id_data"].columns:
            df3[c] = df2["id_data"][c]

    for c in df2["selected_vars"]:
        bin_col = f"{c}_bin"
        woe_col = f"{c}_bin_woe"

        df3[c]       = df2["X_raw_selected"][c]
        df3[bin_col] = df2["X_binned"][c].astype(str)
        df3[woe_col] = df3[bin_col].map(df2["woe_maps"][c]).astype(float)

        if df3[woe_col].isna().any():
            n_na = int(df3[woe_col].isna().sum())
            print(
                f"CẢNH BÁO: {c} có {n_na} bin không map được WOE."
            )

    df3[TARGET_COL] = df2["y"].astype(int)

    if "train_test" in df2:
        df3[SPLIT_COL] = df2["train_test"].astype(str)

    print(f"Cột df3: {list(df3.columns)}")
    print(f"Shape: {df3.shape}")
    print(df3.head().to_string(index=True))
    print()

    return df3


# ============================================================
# STEP 4 — MULTICOLLINEARITY ON TRAIN ONLY
# ============================================================

def compute_vif_matrix(X):
    """VIF bằng nghịch đảo correlation matrix."""
    cols = list(X.columns)

    if len(cols) == 0:
        return pd.DataFrame(columns=["Variable", "VIF"])
    if len(cols) == 1:
        return pd.DataFrame({"Variable": cols, "VIF": [1.0]})

    corr = (
        X.astype(float)
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0.0)
         .corr()
         .fillna(0.0)
         .values
    )

    try:
        vals = np.diag(np.linalg.inv(corr))
        vals = np.where(np.isfinite(vals), vals, 999.0)
    except Exception:
        vals = np.full(len(cols), 999.0)

    return pd.DataFrame({
        "Variable": cols,
        "VIF":      np.maximum(vals, 1.0)
    })


def handle_multicollinearity(
    df3,
    iv_scores,
    corr_threshold=CORR_THRESHOLD,
    vif_threshold=VIF_THRESHOLD
):
    """Loại multicollinear — CHỈ TRAIN."""
    print("=" * 60)
    print("BƯỚC 4: XỬ LÝ MULTICOLLINEARITY TRÊN TRAIN")
    print("=" * 60)
    print()

    WOE_SUFFIX = "_bin_woe"
    BIN_SUFFIX = "_bin"

    woe_cols = [c for c in df3.columns if c.endswith(WOE_SUFFIX)]

    def woe_to_var(woe_col):
        return (
            woe_col[:-len(WOE_SUFFIX)]
            if woe_col.endswith(WOE_SUFFIX)
            else woe_col
        )

    remaining   = woe_cols.copy()
    removed     = []
    corr_logs   = []
    vif_logs    = []
    global_iter = 0

    while global_iter < 100:
        global_iter  += 1
        corr_changed  = False
        vif_changed   = False

        while len(remaining) > 1:
            corr = df3[remaining].corr().fillna(0.0)

            if not np.allclose(corr.values, corr.values.T, atol=1e-10):
                raise ValueError("Correlation matrix không symmetric.")

            upper = corr.abs().where(
                np.triu(np.ones(corr.shape, dtype=bool), k=1)
            )
            viol = upper.stack(dropna=True)
            viol = viol[viol >= corr_threshold]

            if viol.empty:
                break

            a, b   = viol.sort_values(ascending=False).index[0]
            r      = float(corr.loc[a, b])
            var_a  = woe_to_var(a)
            var_b  = woe_to_var(b)
            iva    = iv_scores.get(var_a, -np.inf)
            ivb    = iv_scores.get(var_b, -np.inf)

            if iva < ivb:
                rem, keep = a, b
                cmp = f"IV({var_a})={iva:.6f}<IV({var_b})={ivb:.6f}"
            elif ivb < iva:
                rem, keep = b, a
                cmp = f"IV({var_b})={ivb:.6f}<IV({var_a})={iva:.6f}"
            else:
                rem  = max(a, b)
                keep = min(a, b)
                cmp  = f"IV tie; alphabet loại {woe_to_var(rem)}"

            rem_var  = woe_to_var(rem)
            keep_var = woe_to_var(keep)

            remaining.remove(rem)
            removed.append([rem_var, f"Corr cao với {keep_var}"])
            corr_changed = True
            corr_logs.append(
                f"[{global_iter}] Loại {rem_var} "
                f"|r({var_a},{var_b})|={abs(r):.4f}>={corr_threshold} | "
                f"{cmp} | Còn: {len(remaining)}"
            )

        while len(remaining) > 1:
            vif = compute_vif_matrix(df3[remaining])
            mv  = float(vif["VIF"].max())

            if mv < vif_threshold:
                break

            cand = vif[vif["VIF"] == mv].copy()
            cand["var"] = cand["Variable"].apply(woe_to_var)
            cand["IV"]  = cand["var"].map(iv_scores)

            rem = str(
                cand.sort_values(
                    ["VIF", "IV", "var"],
                    ascending=[False, True, True]
                ).iloc[0]["Variable"]
            )
            rem_var = woe_to_var(rem)

            remaining.remove(rem)
            removed.append([rem_var, f"VIF={mv:.4f}>={vif_threshold}"])
            vif_changed = True
            vif_logs.append(
                f"[{global_iter}] Loại {rem_var} "
                f"VIF={mv:.4f}>={vif_threshold} | Còn: {len(remaining)}"
            )

        if not corr_changed and not vif_changed:
            break

    if global_iter >= 100:
        print("CẢNH BÁO: Đạt 100 vòng lặp; kiểm tra CORR/VIF.")

    print("--- 4.1 CORRELATION ---")
    print(
        "\n".join(corr_logs)
        if corr_logs
        else f"Không có cặp |r|>={corr_threshold}."
    )
    print()

    print("--- 4.2 VIF ---")
    print(
        "\n".join(vif_logs)
        if vif_logs
        else f"Không có VIF>={vif_threshold}."
    )
    if len(remaining) == 1:
        print("Ghi chú: 1 biến → VIF=1.0.")

    print()
    print("--- 4.3 HỘI TỤ ---")
    print(f"✅ Hội tụ sau {global_iter} vòng.\n")

    final_woe_cols = sorted(
        remaining,
        key=lambda c: iv_scores.get(woe_to_var(c), -np.inf),
        reverse=True
    )
    final_vars = [woe_to_var(c) for c in final_woe_cols]

    if not final_woe_cols:
        print("CẢNH BÁO: Tất cả bị loại ở Bước 4.")

    print(f"✅ BIẾN CUỐI CÙNG: [{len(final_vars)}]")
    print("-" * 60)

    print("Correlation Matrix:")
    print(
        df3[final_woe_cols].corr().fillna(0.0).to_string()
        if final_woe_cols else "(Không còn biến.)"
    )
    print()

    print("VIF:")
    fvif = (
        compute_vif_matrix(df3[final_woe_cols])
        if final_woe_cols
        else pd.DataFrame(columns=["Variable", "VIF"])
    )
    vif_rows = [
        [woe_to_var(r["Variable"]), r["Variable"], f"{r['VIF']:.6f}"]
        for _, r in fvif.iterrows()
    ]
    print(
        make_box_table(
            ["Biến gốc", "Cột WOE", "VIF"],
            vif_rows,
            ["left", "left", "right"]
        )
    )
    print()

    print(f"❌ LOẠI Ở BƯỚC 4: [{len(removed)}]")
    print("-" * 60)
    print(
        make_box_table(["Biến", "Lý do"], removed, ["left", "left"])
        if removed else "Không có biến bị loại."
    )
    print()

    output_cols = []
    for var in final_vars:
        for suf in ["", BIN_SUFFIX, WOE_SUFFIX]:
            col = var + suf
            if col in df3.columns:
                output_cols.append(col)

    id_exist      = [c for c in ID_COLS if c in df3.columns]
    final_out_cols = id_exist + output_cols + [TARGET_COL]

    if SPLIT_COL in df3.columns:
        final_out_cols.append(SPLIT_COL)

    return df3[final_out_cols].copy()


# ============================================================
# PSI STABILITY CHECK
# ============================================================

def check_woe_stability(
    df4_train,
    df4_test,
    iv_scores,
    psi_threshold=PSI_THRESHOLD
):
    """PSI train vs test. Chỉ cảnh báo, không tự động loại."""
    print("=" * 60)
    print("BƯỚC 4.5: PSI STABILITY CHECK (TRAIN vs TEST)")
    print("=" * 60)

    WOE_SUFFIX = "_bin_woe"
    BIN_SUFFIX = "_bin"

    common_woe = [
        c for c in df4_train.columns
        if c.endswith(WOE_SUFFIX) and c in df4_test.columns
    ]

    if not common_woe:
        print("Không có cột WOE chung.")
        print()
        return

    psi_rows = []

    for woe_col in common_woe:
        var     = woe_col[:-len(WOE_SUFFIX)]
        bin_col = f"{var}{BIN_SUFFIX}"

        if (
            bin_col not in df4_train.columns
            or bin_col not in df4_test.columns
        ):
            continue

        train_bins = df4_train[bin_col].astype(str).tolist()
        test_bins  = df4_test[bin_col].astype(str).tolist()
        all_bins   = sorted(set(train_bins + test_bins))

        psi_val = compute_psi(train_bins, test_bins, all_bins)
        status  = (
            "⚠️ MẤT ỔN ĐỊNH"
            if psi_val > psi_threshold
            else "✅ ỔN ĐỊNH"
        )
        psi_rows.append([var, f"{psi_val:.6f}", status])

    if psi_rows:
        print(
            make_box_table(
                ["Biến", "PSI", "Trạng thái"],
                psi_rows,
                ["left", "right", "left"]
            )
        )
    else:
        print("Không đủ dữ liệu tính PSI.")

    print()
    unstable = [r[0] for r in psi_rows if "MẤT ỔN ĐỊNH" in r[2]]

    if unstable:
        print(f"CẢNH BÁO: {len(unstable)} biến PSI>{psi_threshold}:")
        for v in unstable:
            print(f"  - {v}")
    else:
        print(f"Tất cả PSI<={psi_threshold}: ổn định.")

    print()


# ============================================================
# TRANSFORM TEST USING TRAIN BINNING / WOE
# ============================================================

def get_final_vars_from_df4(
    df4,
    target_col=TARGET_COL,
    split_col=SPLIT_COL
):
    WOE_SUFFIX = "_bin_woe"
    return [
        c[:-len(WOE_SUFFIX)]
        for c in df4.columns
        if c.endswith(WOE_SUFFIX)
    ]


def assign_numeric_bins_for_test(
    x,
    bin_rules,
    median_value,
    special_value=None,
    special_label=None,
    special_seen_train=True,
    fallback_bin=None
):
    """
    Gán numeric bin cho TEST theo quy tắc học từ TRAIN.

    Quy tắc:
    - Special value đã xuất hiện trong TRAIN được gán special_label.
    - Special value chưa xuất hiện trong TRAIN được gán fallback_bin.
    - Không tạo WOE mới hoặc tự gán WOE bằng 0 trên TEST.
    - Giá trị numeric ngoài khoảng được gán interval gần nhất.
    - Giá trị không parse được numeric dùng median học từ TRAIN.
    """
    special_mask = _is_special_value(
        x,
        special_value
    )

    if (
        special_label is None
        and special_value is not None
    ):
        special_label = str(special_value).strip()

    rules = sorted(
        bin_rules,
        key=lambda rule: rule["left"]
    )

    if fallback_bin is None and rules:
        fallback_bin = rules[0]["label"]

    x_num = pd.to_numeric(
        x.astype(str)
         .str.strip()
         .str.replace(",", "", regex=False)
         .mask(x.isna(), np.nan)
         .mask(special_mask, np.nan),
        errors="coerce"
    ).fillna(median_value)

    out = pd.Series(
        index=x.index,
        dtype=object
    )

    if special_value is not None:
        if special_seen_train:
            out.loc[special_mask] = special_label
        else:
            out.loc[special_mask] = fallback_bin

            if special_mask.any():
                print(
                    f"CẢNH BÁO: {x.name or '<không tên>'} có "
                    f"{int(special_mask.sum())} dòng "
                    f"special_value='{special_value}' trong TEST "
                    "nhưng TRAIN không có. "
                    f"Áp dụng fallback_bin='{fallback_bin}', "
                    "không học hoặc tự gán WOE từ TEST."
                )


    numeric_mask = ~special_mask

    for rule in rules:
        interval_mask = (
            numeric_mask
            & (x_num > rule["left"])
            & (x_num <= rule["right"])
        )

        out.loc[interval_mask] = rule["label"]

    unassigned_mask = out.isna()

    if unassigned_mask.any() and rules:
        for idx in out.index[unassigned_mask]:
            value = float(x_num.loc[idx])
            nearest_rule = min(
                rules,
                key=lambda rule: min(
                    abs(value - rule["left"]),
                    abs(value - rule["right"])
                )
            )

            out.loc[idx] = nearest_rule["label"]

    return out.astype(str)





def assign_categorical_bins_for_test(
    x,
    category_to_bin,
    cat_info,
    fallback_bin
):
    """
    Gán categorical bin cho TEST theo quy tắc học từ TRAIN.

    - Không fit lại binning trên TEST.
    - Chuẩn hóa special value giống TRAIN.
    - Special value đã thấy trong TRAIN được gán đúng special bin.
    - Special value chỉ xuất hiện trong TEST được đưa vào fallback bin.
    - Không tự học hoặc tự gán WOE mới từ TEST.
    """
    x_cat = x.astype(object).copy()

    special_value = cat_info.get(
        "special_value"
    )
    special_label = cat_info.get(
        "special_label"
    )
    special_seen_train = bool(
        cat_info.get(
            "special_seen_train",
            False
        )
    )

    special_mask = _is_special_value(
        x_cat,
        special_value
    )

    if special_label is not None:
        x_cat.loc[special_mask] = special_label

    if (
        cat_info["missing_policy"]
        == "missing_as_category"
    ):
        x_cat = (
            x_cat
            .where(x_cat.notna(), "Missing")
            .astype(str)
        )
    else:
        x_cat = (
            x_cat
            .fillna(cat_info["fill_value"])
            .astype(str)
        )

    out = x_cat.map(category_to_bin)
    unseen_count = int(out.isna().sum())

    # Special chỉ xuất hiện trong TEST:
    # không học WOE từ TEST và không tự gán WOE tùy ý.
    if (
        special_value is not None
        and special_mask.any()
        and not special_seen_train
    ):
        print(
            f"CẢNH BÁO: {x.name or '<không tên>'} có "
            f"{int(special_mask.sum())} dòng "
            f"special_value='{special_value}' trong TEST "
            "nhưng TRAIN không có. "
            f"Áp dụng fallback_bin='{fallback_bin}', "
            "không học WOE từ TEST."
        )

    if unseen_count > 0:
        out = out.fillna(fallback_bin)

    return out.astype(str), unseen_count



def transform_test_with_train_binning(
    df_test,
    df2,
    final_vars,
    target_col=TARGET_COL,
    split_col=SPLIT_COL
):
    print("=" * 60)
    print("TRANSFORM WOE CHO TEST THEO BINNING TỪ TRAIN")
    print("=" * 60)

    df4_test = pd.DataFrame(index=df_test.index)

    for c in ID_COLS:
        if c in df_test.columns:
            df4_test[c] = df_test[c]

    unseen_logs = []

    for var in final_vars:
        bin_col = f"{var}_bin"
        woe_col = f"{var}_bin_woe"
        meta    = df2["binning_meta"][var]

        df4_test[var] = df_test[var]

        if meta["kind"] == "num":
            df4_test[bin_col] = assign_numeric_bins_for_test(
                df_test[var],
                meta["bin_rules"],
                meta["median"],
                special_value=meta.get("special_value"),
                special_label=meta.get("special_label"),
                special_seen_train=meta.get(
                    "special_seen_train",
                    False
                ),
                fallback_bin=meta.get("fallback_bin")
            )

        else:
            df4_test[bin_col], unseen_count = assign_categorical_bins_for_test(
                df_test[var],
                meta["category_to_bin"],
                meta["cat_info"],
                meta["fallback_bin"]
            )

            if unseen_count > 0:
                unseen_logs.append(
                    f"- {var}: {unseen_count} dòng category mới → "
                    f"fallback={meta['fallback_bin']}"
                )

        # Map WOE áp dụng chung cho cả numeric và categorical
        woe_map_use = dict(meta["woe_map"])


        df4_test[woe_col] = (
            df4_test[bin_col].map(woe_map_use).astype(float)
        )

        if df4_test[woe_col].isna().any():
            n_na = int(df4_test[woe_col].isna().sum())
            print(f"CẢNH BÁO: {var} có {n_na} dòng không map WOE.")

    df4_test[target_col] = df_test[target_col].astype(int)

    if split_col in df_test.columns:
        df4_test[split_col] = df_test[split_col].astype(str)

    print(f"Rows: {df4_test.shape[0]} | Vars: {len(final_vars)}")
    print()

    if unseen_logs:
        print("Category mới trong test:")
        print("\n".join(unseen_logs))
        print()
    else:
        print("Không phát hiện category mới.")
        print()

    print(f"Cột df4_test: {list(df4_test.columns)}")
    print()

    return df4_test


# ============================================================
# SUMMARY AND EXPORT FUNCTIONS
# ============================================================

def print_pipeline_summary(df1, df2, df4):
    WOE_SUFFIX     = "_bin_woe"
    nnum           = df1["X_num"].shape[1]
    ncat           = df1["X_cat"].shape[1]
    final_woe_cols = [c for c in df4.columns if c.endswith(WOE_SUFFIX)]
    final_vars     = [c[:-len(WOE_SUFFIX)] for c in final_woe_cols]

    print("=" * 60)
    print("TỔNG KẾT PIPELINE v5")
    print("=" * 60)
    print(f"Input  : {nnum+ncat} ({nnum} num, {ncat} cat)")
    print(f"Bước 2 : {len(df2['selected_vars'])} biến")
    print(f"Bước 4 : {len(final_vars)} biến")
    print()

    if SPLIT_COL in df4.columns:
        print(df4[SPLIT_COL].value_counts(dropna=False).to_string())
        print()

    print("BIẾN CUỐI:")
    for c in final_vars:
        iv_val = df2["iv_scores"].get(c, np.nan)
        stab   = df2.get("iv_stability", {}).get(c)
        s_str  = ""
        if stab is not None:
            _, _, cv, stable = stab
            if not np.isnan(cv):
                s_str = f" | CV={cv:.1%} ({'✅' if stable else '⚠️'})"
        print(f"  - {c} | IV={iv_val:.6f}{s_str}")

    print()
    print("CỘT df4:")
    for c in df4.columns:
        print(f"  - {c}")
    print(f"\nShape df4: {df4.shape}")
    print("=" * 60)
    print()


def create_df4_v2_from_df4(
    df4,
    target_col=TARGET_COL,
    split_col=SPLIT_COL
):
    WOE_SUFFIX    = "_bin_woe"
    id_exist      = [c for c in ID_COLS if c in df4.columns]
    woe_cols      = [c for c in df4.columns if c.endswith(WOE_SUFFIX)]
    final_cols    = id_exist + woe_cols

    if target_col in df4.columns:
        final_cols.append(target_col)
    else:
        print(f"CẢNH BÁO: Không tìm thấy '{target_col}'.")

    if split_col in df4.columns:
        final_cols.append(split_col)

    final_cols = list(dict.fromkeys(final_cols))
    df4_v2     = df4[final_cols].copy()
    rename_map = {c: c[:-len(WOE_SUFFIX)] for c in woe_cols}
    df4_v2     = df4_v2.rename(columns=rename_map)

    print("=" * 60)
    print("TẠO df4_v2")
    print("=" * 60)
    for old, new in rename_map.items():
        print(f"  {old} → {new}")
    print(f"\nShape df4={df4.shape} | df4_v2={df4_v2.shape}")

    if split_col in df4_v2.columns:
        print(df4_v2[split_col].value_counts(dropna=False).to_string())

    print(f"\nCột df4_v2: {list(df4_v2.columns)}")
    print("=" * 60)
    print()

    return df4_v2


def safe_filename(name):
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))


def export_woe_mapping_files_from_df4(df4, output_folder=MAPPING_FOLDER):
    WOE_SUFFIX = "_bin_woe"
    BIN_SUFFIX = "_bin"

    print("=" * 60)
    print("EXPORT MAPPING WOE")
    print("=" * 60)

    os.makedirs(output_folder, exist_ok=True)
    woe_cols       = [c for c in df4.columns if c.endswith(WOE_SUFFIX)]
    exported_files = []

    if not woe_cols:
        print("Không có *_bin_woe để export.")
        print(f"Files=0 | Folder={output_folder}")
        print()
        return exported_files

    for woe_col in woe_cols:
        var         = woe_col[:-len(WOE_SUFFIX)]
        bin_col     = f"{var}{BIN_SUFFIX}"
        has_raw     = var     in df4.columns
        has_bin     = bin_col in df4.columns
        has_woe     = woe_col in df4.columns

        if not has_woe:
            print(f"SKIP {var}: thiếu {woe_col}")
            continue
        
        if var in SPECIAL_FLOAT_BINNING:
            if not has_bin:
                print(f"SKIP {var}: mixed numeric nhưng thiếu {bin_col}")
                continue
            export_cols = [bin_col, woe_col]
            var_type    = "mixed_numeric_special"
        
        elif has_raw and pd.api.types.is_numeric_dtype(df4[var]):
            if not has_bin:
                print(f"SKIP {var}: numeric nhưng thiếu {bin_col}")
                continue
            export_cols = [bin_col, woe_col]
            var_type    = "numeric"
        
        elif has_raw:
            export_cols = (
                [var, bin_col, woe_col] if has_bin
                else [var, woe_col]
            )
            var_type = "categorical"
        else:
            if not has_bin:
                print(
                    f"SKIP {var}: không có cột gốc và thiếu {bin_col}"
                )
                continue
            print(f"WARN {var}: không có cột gốc, xuất fallback.")
            export_cols = [bin_col, woe_col]
            var_type    = "unknown"

        miss = [c for c in export_cols if c not in df4.columns]
        if miss:
            print(f"SKIP {var}: thiếu {miss}")
            continue

        mapping_df = (
            df4[export_cols]
            .drop_duplicates()
            .sort_values(by=export_cols)
            .reset_index(drop=True)
        )

        out_file = os.path.join(
            output_folder, f"{safe_filename(var)}_woe.xlsx"
        )
        mapping_df.to_excel(out_file, index=False, engine="openpyxl")
        exported_files.append(out_file)
        print(
            f"✓ {out_file} | {var_type} | rows={mapping_df.shape[0]}"
        )

    print(f"\nFiles={len(exported_files)} | Folder={output_folder}")
    print("=" * 60)
    print()

    return exported_files

def export_df4_to_excel_by_ngay_bc(
    df4,
    output_dir=OUTPUT_FOLDER_RAW_XLSX_BY_DATE,
    date_col="NGAY_BC",
    file_prefix="credit_data_woe_raw",
    split_col=SPLIT_COL,
    test_label=TEST_LABEL,
    id_col="BILLING_ACCTNUM"
):
    print("=" * 60)
    print("EXPORT df4 THEO NGÀY BÁO CÁO")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    if date_col not in df4.columns:
        print(f"LỖI: Thiếu cột ngày báo cáo '{date_col}'.")
        return []

    if split_col not in df4.columns:
        print(f"LỖI: Thiếu cột split '{split_col}'.")
        return []

    df_export = df4.copy()

    df_export[date_col] = pd.to_datetime(
        df_export[date_col],
        errors="coerce",
        dayfirst=True
    )

    n_invalid = int(df_export[date_col].isna().sum())

    if n_invalid > 0:
        print(f"LỖI: {n_invalid} dòng ngày báo cáo lỗi.")
        print(
            df4.loc[df_export[date_col].isna(), date_col]
               .head(10)
               .to_string(index=False)
        )
        return []

    before = df_export.shape

    df_export = df_export[
        df_export[split_col] == test_label
    ].copy()

    after = df_export.shape

    print(f"Filter: {split_col} == '{test_label}'")
    print(f"Before={before} | After={after}")
    print()

    if df_export.empty:
        print("CẢNH BÁO: Không có dòng test để export.")
        print(f"Files=0 | Folder={output_dir}")
        print()
        return []

    if id_col in df_export.columns:
        df_export = format_billing_acctnum(
            df=df_export,
            col=id_col
        )

    list_ngay = sorted(df_export[date_col].dt.date.unique())
    exported_files = []

    for ngay in list_ngay:
        df_sub = df_export[
            df_export[date_col].dt.date == ngay
        ].copy()

        df_sub[date_col] = df_sub[date_col].dt.strftime("%d/%m/%Y")

        fname = f"{file_prefix}_{ngay.strftime('%d_%m_%Y')}.xlsx"
        fpath = os.path.join(output_dir, fname)

        df_sub.to_excel(fpath, index=False, engine="openpyxl")
        exported_files.append(fpath)

        print(
            f"✓ {fpath} | "
            f"{ngay.strftime('%d/%m/%Y')} | "
            f"rows={df_sub.shape[0]}"
        )

        del df_sub

    print(f"\nFiles={len(exported_files)} | Folder={output_dir}")
    print("=" * 60)
    print()

    return exported_files

# ============================================================
# MAIN PIPELINE
# ============================================================

df = load_and_validate(INPUT_FILE, TARGET_COL)
if df is None:
    sys.exit(0)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns="Unnamed: 0")

df = format_billing_acctnum(df=df, col="BILLING_ACCTNUM")


df = validate_existing_train_test_split(
    df,
    split_col=SPLIT_COL,
    train_label=TRAIN_LABEL,
    test_label=TEST_LABEL,
    target_col=TARGET_COL
)

if df is None:
    sys.exit(0)


df_train = df[df[SPLIT_COL] == TRAIN_LABEL].copy()
df_test  = df[df[SPLIT_COL] == TEST_LABEL].copy()

train_target_vals = set(df_train[TARGET_COL].dropna().unique())
if train_target_vals != {0, 1}:
    print("LỖI: Target trong TRAIN không có đủ cả 0 và 1.")
    print(df_train[TARGET_COL].value_counts(dropna=False).to_string())
    sys.exit(0)

if not df_test.empty:
    test_target_vals = set(df_test[TARGET_COL].dropna().unique())
    if not test_target_vals.issubset({0, 1}):
        print("LỖI: Target trong TEST có giá trị ngoài 0/1.")
        print(df_test[TARGET_COL].value_counts(dropna=False).to_string())
        sys.exit(0)
        
        


# ============================================================
# ÉP KIỂU NUMERIC-LIKE — FIT TRAIN, APPLY TEST
# ============================================================

print("=" * 60)
print("ÉP KIỂU NUMERIC-LIKE — FIT TRAIN, APPLY TEST")
print("=" * 60)

numeric_like_cols, numeric_parse_summary = detect_numeric_like_columns_on_train(
    df_train=df_train,
    target_col=TARGET_COL,
    id_cols=ID_COLS,
    split_col=SPLIT_COL,
    parse_ratio=0.95
)

if numeric_like_cols:
    print("Cột numeric-like:")
    for c in numeric_like_cols:
        print(f"  - {c}")
else:
    print("Không phát hiện cột numeric-like.")

print()
if numeric_parse_summary.empty:
    print("(Không có cột cần kiểm tra.)")
else:
    print(numeric_parse_summary.to_string(index=False))
print()

df_train = apply_numeric_like_conversion(
    df=df_train, numeric_like_cols=numeric_like_cols
)
df_test = apply_numeric_like_conversion(
    df=df_test, numeric_like_cols=numeric_like_cols
)
df = pd.concat([df_train, df_test], axis=0).sort_index()

print(
    f"Shape: df={df.shape} | "
    f"train={df_train.shape} | "
    f"test={df_test.shape}"
)
print("=" * 60)
print()

print("=" * 60)
print("VALIDATE SAU SPLIT")
print("=" * 60)
print(
    f"df={df.shape} | "
    f"train={df_train.shape} | "
    f"test={df_test.shape}"
)
print()

if df_train.empty:
    print("LỖI: df_train rỗng.")
    sys.exit(0)

if df_test.empty:
    print("CẢNH BÁO: df_test rỗng. Chỉ chạy train.")
    print()

# ============================================================
# BƯỚC 1 → 4: FIT TRÊN TRAIN
# ============================================================

df1 = split_features(
    df_train, target_col=TARGET_COL, split_col=SPLIT_COL
)
df2 = auto_bin_select(df1)

_fallback = pd.concat(
    [
        df1["id_data"],
        pd.DataFrame({
            TARGET_COL: df1["y"],
            SPLIT_COL:  df1.get(
                "train_test",
                pd.Series(
                    [TRAIN_LABEL] * len(df1["y"]),
                    index=df1["y"].index
                )
            )
        })
    ],
    axis=1
)

df3       = woe_transform(df2) if df2["selected_vars"] else _fallback
df4_train = (
    handle_multicollinearity(df3, iv_scores=df2["iv_scores"])
    if df2["selected_vars"]
    else _fallback
)

# ============================================================
# TRANSFORM TEST
# ============================================================

final_vars = get_final_vars_from_df4(
    df4_train, target_col=TARGET_COL, split_col=SPLIT_COL
)

if df_test.empty:
    df4_test = pd.DataFrame(columns=df4_train.columns)
else:
    df4_test = transform_test_with_train_binning(
        df_test=df_test,
        df2=df2,
        final_vars=final_vars,
        target_col=TARGET_COL,
        split_col=SPLIT_COL
    )

df4 = pd.concat([df4_train, df4_test], axis=0).sort_index()

# ============================================================
# PSI CHECK
# ============================================================

if not df_test.empty and not df4_test.empty:
    check_woe_stability(
        df4_train, df4_test,
        df2["iv_scores"],
        psi_threshold=PSI_THRESHOLD
    )

# ============================================================
# SUMMARY
# ============================================================

print_pipeline_summary(df1, df2, df4)

# ============================================================
# EXPORT
# ============================================================

print("=" * 60)
print("EXPORT")
print("=" * 60)

export_parquet_brotli(df=df4, output_file=OUTPUT_FILE_RAW)

exported_xlsx = export_df4_to_excel_by_ngay_bc(
    df4=df4,
    output_dir=OUTPUT_FOLDER_RAW_XLSX_BY_DATE,
    date_col="NGAY_BC",
    file_prefix="credit_data_woe_raw"
)


exported_mapping = export_woe_mapping_files_from_df4(
    df4=df4_train,
    output_folder=MAPPING_FOLDER
)


df4_v2 = create_df4_v2_from_df4(
    df4, target_col=TARGET_COL, split_col=SPLIT_COL
)
export_parquet_brotli(df=df4_v2, output_file=OUTPUT_FILE_WOE)

print("=" * 60)
print("HOÀN TẤT")
print("=" * 60)
print(f"Raw parquet  : {OUTPUT_FILE_RAW}")
print(f"WOE parquet  : {OUTPUT_FILE_WOE}")
print(f"XLSX folder  : {OUTPUT_FOLDER_RAW_XLSX_BY_DATE}")
print(f"Mapping      : {MAPPING_FOLDER}")
print(f"XLSX files   : {len(exported_xlsx)}")
print(f"Mapping files: {len(exported_mapping)}")
print("=" * 60)
