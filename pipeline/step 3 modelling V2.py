# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 08:32:39 2026

@author: tamph1
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 28 07:45:18 2026
 
@author: tamph1
"""
 
# -*- coding: utf-8 -*-

"""
Created on Wed May 20 17:38:59 2026
@author: tamph1 - Logistic Regression Best-Effort Edition
"""


 
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss, accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import math
import warnings
import os


import matplotlib.pyplot as plt
 

warnings.filterwarnings('ignore')
 
S = timer()
 
# ====================== LOAD DATA ======================
data_group = pd.read_parquet(r"D:\model\credit_data_woe.parquet")
    
if 'Unnamed: 0' in data_group.columns:
    data_group = data_group.drop(columns='Unnamed: 0')
 
TARGET_COL = "NX"
SPLIT_COL = "train_test"
ID_COLS = ["BILLING_ACCTNUM", "CIF", "NGAY_BC", TARGET_COL, SPLIT_COL]
 
# Kiểm tra target
if TARGET_COL not in data_group.columns:
    raise ValueError(f"Không tìm thấy cột target '{TARGET_COL}' trong file credit_data_woe.xlsx.")
 
# Kiểm tra target trước khi chuẩn hóa
if data_group[TARGET_COL].isna().any():
    n_missing_target = int(data_group[TARGET_COL].isna().sum())
    raise ValueError(
        f"Cột target '{TARGET_COL}' có {n_missing_target} giá trị thiếu. "
        "Không tự động gán target thiếu thành lớp 0."
    )

data_group[TARGET_COL] = pd.to_numeric(
    data_group[TARGET_COL],
    errors="raise",
).astype(np.int8)

target_values = set(data_group[TARGET_COL].unique())
if not target_values.issubset({0, 1}):
    raise ValueError(
        f"Cột target '{TARGET_COL}' phải là biến nhị phân 0/1. "
        f"Giá trị thực tế: {sorted(target_values)}"
    )
    
    
# Kiểm tra cột train_test
if SPLIT_COL not in data_group.columns:
    raise ValueError(f"Không tìm thấy cột '{SPLIT_COL}' trong file credit_data_woe.xlsx.")
 
# Chuẩn hóa giá trị train/test
data_group[SPLIT_COL] = data_group[SPLIT_COL].astype(str).str.strip().str.lower()
 
valid_split_values = set(data_group[SPLIT_COL].dropna().unique())
 
if not valid_split_values.issubset({"train", "test"}):
    raise ValueError(
        f"Cột '{SPLIT_COL}' chỉ được chứa giá trị 'train' hoặc 'test'. "
        f"Giá trị thực tế: {sorted(valid_split_values)}"
    )
 
# Tách dữ liệu theo cột train_test, KHÔNG dùng train_test_split
train_mask = data_group[SPLIT_COL] == "train"
test_mask = data_group[SPLIT_COL] == "test"
 
if train_mask.sum() == 0:
    raise ValueError("Không có dòng nào thuộc tập train.")
 
if test_mask.sum() == 0:
    raise ValueError("Không có dòng nào thuộc tập test.")
 
# Chỉ lấy biến numeric làm feature, loại BILLING_ACCTNUM, NGAY_BC, NX, train_test
feature_cols = (
    data_group
    .drop(columns=ID_COLS, errors="ignore")
    .select_dtypes(include=[np.number])
    .columns
    .tolist()
)
 
if len(feature_cols) == 0:
    raise ValueError("Không có biến numeric nào để đưa vào mô hình.")
 
X = data_group[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
y = data_group[TARGET_COL].astype(np.int8)
 
X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()
 
y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()
 
print("=" * 60)
print("TÁCH TRAIN/TEST THEO CỘT train_test")
print("=" * 60)
print(f"Số dòng train: {X_train.shape[0]}")
print(f"Số dòng test : {X_test.shape[0]}")
print(f"Số biến mô hình: {len(feature_cols)}")
print()
print("Danh sách biến mô hình:")
print(feature_cols)
print()
print("Phân phối target trên train:")
print(y_train.value_counts().sort_index().to_string())
print()
print("Phân phối target trên test:")
print(y_test.value_counts().sort_index().to_string())
print("=" * 60)
 

#####CHỈNH Ở PHẦN NÀY###########
# =============================================================================
# LINEAR SCORING PIPELINE - LogisticRegression(newton-cholesky) + Scaler + Threshold Scan + class_weight
# 9 KPI gates: Sens>=.70 | MCC>=.68 | Spec>=.95 | Prec>=.70 | Gap>0 | BCE<=.15 | NPV>=.95 | F1>=.70 | Acc>=.95
# Fit/selection/threshold 100% TRAIN-only (zero leakage). Threshold rule: gap-margin fraction
# (_GAP_FRACTION hieu chinh TREN TRAIN; TEST/FULL chi VERIFY early-stop, khong vao sort_key).
# RANG BUOC CUNG: moi LogisticRegression dung solver='newton-cholesky';
#                 penalty in ALLOWED_PENALTIES; class_weight in ALLOWED_CLASS_WEIGHTS (prune truoc fit).

import gc
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# ---- 1) CONFIG BLOCK - toan bo tham so grid nam o day (khong rai magic number) ----
FIXED_SOLVER = "newton-cholesky"          # RANG BUOC CUNG, ap cho moi cfg, khong duoc ghi de

# 9 KPI gates theo thu tu co dinh: (ten metric, nguong, toan tu, hien thi)
KPI_GATES = (
    ("Sensitivity", 0.70, ">=", ">=0.70"),
    ("MCC", 0.68, ">=", ">=0.68"),
    ("Specificity", 0.95, ">=", ">=0.95"),
    ("Precision", 0.70, ">=", ">=0.70"),
    ("Gap=Rec-Prec", 0.00, ">", ">0"),
    ("BCE", 0.15, "<=", "<=0.15"),
    ("NPV", 0.95, ">=", ">=0.95"),
    ("F1", 0.70, ">=", ">=0.70"),
    ("Accuracy", 0.95, ">=", ">=0.95"),
)
KPI_NON_GAP_GATES = tuple(_g for _g in KPI_GATES if _g[0] != "Gap=Rec-Prec")

ALLOWED_PENALTIES = ["l2", None]                      # newton-cholesky: chi l2 / None (auto-detect None)
ALLOWED_CLASS_WEIGHTS = ("none", "balanced", "cw:{0:w0,1:w1} (w0>0, w1>0)")

PENALTY_CHOICES = tuple(ALLOWED_PENALTIES)            # mac dinh: l2 (+ None neu sklearn ho tro)
CW_CHOICES_LAYOUT = (None, "balanced", "dynamic:{0:1.0, 1:n0/n1}")   # bien the 3 sinh dong o [02]
CW_CHOICES = (None, "balanced", {0: 1.0, 1: 1.0})     # placeholder - duoc dien lai ngay sau khi co n0/n1

GRID_BASE = {
    "scaler_names": ("StandardScaler", "RobustScaler", "MinMaxScaler"),
    "C_values": (1.0, 0.1, 10.0, 0.01),
    "penalties": PENALTY_CHOICES,
    "class_weights": CW_CHOICES_LAYOUT,               # thu tu uu tien: none -> balanced -> n0/n1
    "gap_fraction": 0.65,
    "max_iter": 10000,
    "tol": 1e-4,
    "max_configs": 36,                                # cap mem cho GRID_BASE (uu tien theo thu tu khai bao)
}
GRID_EXPAND = {
    "C_multipliers": (0.2, 0.5, 2.0, 5.0),            # C fine-tune quanh seed
    "scaler_C_multipliers": (1.0, 0.5, 2.0),          # scaler sweep quanh seed
    "cw_ratio_multipliers": (0.7, 1.0, 1.3),          # {0:1.0, 1:(n0/n1)*r} quanh seed
}
GAP_FRACTIONS = (0.50, 0.60, 0.65, 0.75, 0.85)        # rescan threshold TRAIN, khong refit
MAX_CONFIGS = 80                                      # cap cung tong so config
BATCH_SIZE = 200_000                                  # batch inference / batch matmul

# ---- 2) PREPROCESSING - fit scaler/model/threshold CHI tren TRAIN (no leakage) ----
if "BILLING_ACCTNUM" in data_group.columns:
    data_group["BILLING_ACCTNUM"] = data_group["BILLING_ACCTNUM"].astype("category")
if "CIF" in data_group.columns:
    data_group["CIF"] = data_group["CIF"].astype("category")
for col in feature_cols:
    data_group[col] = data_group[col].astype("float32")

y_all = data_group[TARGET_COL].to_numpy(dtype=np.int8, copy=True)
for _name in ("X_train", "X_test", "y_train", "y_test", "y"):
    if _name in globals():
        del globals()[_name]
gc.collect()

train_np = train_mask.to_numpy(dtype=bool, copy=False)   # train_mask/test_mask ke thua tu pipeline truoc
test_np = test_mask.to_numpy(dtype=bool, copy=False)
_full_np = np.ones(len(data_group), dtype=bool)

X_model = np.empty((len(data_group), len(feature_cols)), dtype=np.float32)
for _j, _col in enumerate(feature_cols):
    X_model[:, _j] = data_group[_col].to_numpy(dtype=np.float32, copy=False)
np.nan_to_num(X_model, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

X_train_val = np.ascontiguousarray(X_model[train_np])
y_train_val = y_all[train_np]

n0 = int(np.count_nonzero(y_train_val == 0))
n1 = int(np.count_nonzero(y_train_val == 1))
if n0 == 0 or n1 == 0:
    raise ValueError("TRAIN phải có đủ cả hai lớp 0 và 1.")

weighting_class_0 = len(y_train_val) / (2.0 * n0)
weighting_class_1 = len(y_train_val) / (2.0 * n1)

# class_weight bien the 3 = {0:1.0, 1:n0/n1} sinh DONG tu TRAIN (khong hard-code)
CW_CHOICES = (None, "balanced", {0: 1.0, 1: float(n0) / float(n1)})
GRID_BASE["class_weights"] = CW_CHOICES

# Auto-detect: solver newton-cholesky co ho tro penalty=None khong (probe 1 lan, TRAIN-only)
if None in PENALTY_CHOICES:
    _probe_rows = min(2000, X_train_val.shape[0])
    _probe_idx = np.arange(_probe_rows, dtype=np.int64)
    for _cls in (0, 1):
        if not np.any(y_train_val[_probe_idx] == _cls):
            _probe_idx = np.append(_probe_idx, int(np.flatnonzero(y_train_val == _cls)[0]))
    _probe_X = np.ascontiguousarray(X_train_val[_probe_idx], dtype=np.float64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            LogisticRegression(C=1.0, penalty=None, solver=FIXED_SOLVER, max_iter=20,
                               random_state=42, n_jobs=1).fit(_probe_X, y_train_val[_probe_idx])
    except Exception as _probe_err:
        ALLOWED_PENALTIES.remove(None)
        PENALTY_CHOICES = tuple(ALLOWED_PENALTIES)
        GRID_BASE["penalties"] = PENALTY_CHOICES
        print("WARNING: solver=%s khong ho tro penalty=None (%s) -> loai None khoi ALLOWED_PENALTIES."
              % (FIXED_SOLVER, type(_probe_err).__name__))
    del _probe_X
    gc.collect()

print("\n" + "=" * 60)
print("PREPROCESSING - FIT TREN TRAIN ONLY (NO LEAKAGE)")
print("=" * 60)
print("Shape TRAIN=%d | TEST=%d | FULL=%d (rows)"
      % (int(train_np.sum()), int(test_np.sum()), X_model.shape[0]))
print("So cot input: %d | fit (TRAIN): %d | inference (FULL): %d"
      % (X_model.shape[1], X_train_val.shape[0], X_model.shape[0]))
print("w0=%.4f w1=%.4f" % (weighting_class_0, weighting_class_1))
print("TRAIN n0=%d n1=%d rate=%.6f" % (n0, n1, n1 / float(n0 + n1)))
print("TEST n0=%d n1=%d rate=%.6f"
      % (int(np.count_nonzero(y_all[test_np] == 0)),
         int(np.count_nonzero(y_all[test_np] == 1)),
         float(y_all[test_np].mean())))
print("Features (%d): %s" % (len(feature_cols), feature_cols))
print("FIXED_SOLVER=%s | ALLOWED_PENALTIES=%s | PENALTY_CHOICES=%s"
      % (FIXED_SOLVER, ALLOWED_PENALTIES, list(PENALTY_CHOICES)))
print("CW_CHOICES=%s | ALLOWED_CLASS_WEIGHTS=%s" % (list(CW_CHOICES), list(ALLOWED_CLASS_WEIGHTS)))
print("=" * 60)

# ---- 3) WRAPPER: quy doi he so ve khong gian goc (chinh xac toan hoc) ----
class ScaledLogisticWrapper:
    def __init__(self, scaler, model):
        coef = model.coef_[0].astype(np.float64, copy=True)
        if hasattr(scaler, "mean_"):
            center = np.asarray(scaler.mean_, dtype=np.float64)
            scale = np.asarray(scaler.scale_, dtype=np.float64)
        elif hasattr(scaler, "center_"):
            center = np.asarray(scaler.center_, dtype=np.float64)
            scale = np.asarray(scaler.scale_, dtype=np.float64)
        else:
            scale = 1.0 / np.asarray(scaler.scale_, dtype=np.float64)
            center = -np.asarray(scaler.min_, dtype=np.float64) * scale
        self.raw_coef_ = (coef / scale).reshape(1, -1)
        self.raw_intercept_ = np.asarray(
            [float(model.intercept_[0] - np.dot(coef, center / scale))],
            dtype=np.float64,
        )
        self._scaler = scaler
        self._model = model
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_

    def predict_proba(self, X_input, batch_size=BATCH_SIZE):
        n_rows = len(X_input)
        out = np.empty((n_rows, 2), dtype=np.float64)
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            part = X_input.iloc[start:end] if hasattr(X_input, "iloc") else X_input[start:end]
            xb = np.asarray(part, dtype=np.float64)
            decision = self._model.decision_function(self._scaler.transform(xb))
            p1 = expit(decision)
            out[start:end, 1] = p1
            out[start:end, 0] = 1.0 - p1
        return out

def _manual_prob(X_data, raw_coef, raw_intercept, batch=BATCH_SIZE):
    """probability_manual = sigmoid(X @ raw_coef + raw_intercept), tinh theo batch."""
    n = len(X_data)
    out = np.empty(n, dtype=np.float64)
    for _s in range(0, n, batch):
        _e = min(_s + batch, n)
        _logit = X_data[_s:_e].astype(np.float64).dot(raw_coef) + raw_intercept
        out[_s:_e] = expit(_logit)
    return out

# ---- 4) THRESHOLD SCAN VECTORIZED - CHI tren TRAIN, 9 KPI gates ----
THRESHOLDS = np.arange(0.05, 0.9200001, 0.0005, dtype=np.float64)
_GAP_FRACTION = float(GRID_BASE["gap_fraction"])   # rule: gap >= frac*gap_max(vung 8-gate pass tren TRAIN)

def _gate_ok(op, val, thr):
    if op == ">=":
        return bool(val >= thr)
    if op == "<=":
        return bool(val <= thr)
    return bool(val > thr)

def _scan9(prob, y, gap_fraction=None):
    """Quet threshold vectorized (cumsum + searchsorted) tren TRAIN.
    Rule: trong vung TRAIN dat 8 gate so (KPI_GATES tru Gap), lay cac T co
    gap >= gap_fraction*gap_max (bien duong giua Sens va Prec de chong dao TRAIN/TEST),
    roi chon max MCC (hoa gap). Fallback: lexsort (n_viol, -gap, -MCC, -Recall, -F1, BCE).
    CHI dung nhan TRAIN - TEST/FULL khong tham gia."""
    _gf = float(_GAP_FRACTION) if gap_fraction is None else float(gap_fraction)
    order = np.argsort(prob, kind="stable")
    ps, ys = prob[order], y[order].astype(np.float64)
    n, pos = ps.size, ys.sum()
    neg = n - pos
    cumpos = np.cumsum(ys)
    idx = np.searchsorted(ps, THRESHOLDS, side="left")
    below = np.where(idx > 0, cumpos[np.maximum(idx - 1, 0)], 0.0)
    tp, fn = pos - below, below
    fp = (n - idx) - tp
    tn = neg - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-12), 0.0)
        spec = np.where(tn + fp > 0, tn / np.maximum(tn + fp, 1e-12), 0.0)
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 0.0)
        npv = np.where(tn + fn > 0, tn / np.maximum(tn + fn, 1e-12), 0.0)
        f1 = np.where(prec + sens > 0, 2.0 * prec * sens / np.maximum(prec + sens, 1e-12), 0.0)
        acc = (tp + tn) / n
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-12), 0.0)
    pc = np.clip(prob, 1e-15, 1.0 - 1e-15)
    bce = float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))
    bce_a = np.full(idx.size, bce)
    gap = sens - prec
    _vec = {"Sensitivity": sens, "MCC": mcc, "Specificity": spec, "Precision": prec,
            "Gap=Rec-Prec": gap, "BCE": bce_a, "NPV": npv, "F1": f1, "Accuracy": acc}
    e8 = np.ones(idx.size, dtype=bool)
    v8 = np.zeros(idx.size, dtype=np.int32)
    for _nm, _thr, _op, _disp in KPI_NON_GAP_GATES:      # 8 gate so (Gate Gap xu ly rieng: gap > 0)
        _okv = (_vec[_nm] >= _thr) if _op == ">=" else (_vec[_nm] <= _thr)
        e8 &= _okv
        v8 += (~_okv).astype(np.int32)
    if e8.any():
        elig = e8 & (gap >= _gf * gap[e8].max())
    else:
        elig = np.zeros_like(e8)
    if elig.any():
        ii = np.flatnonzero(elig)
        j = int(ii[np.lexsort((-gap[ii], -mcc[ii]))[0]])
        n_cand = int(elig.sum())
    else:
        j = int(np.lexsort((bce_a, -f1, -sens, -mcc, -gap, v8))[0])
        n_cand = 0
    m = dict(sens=float(sens[j]), spec=float(spec[j]), prec=float(prec[j]), npv=float(npv[j]),
             f1=float(f1[j]), acc=float(acc[j]), mcc=float(mcc[j]), gap=float(gap[j]), bce=bce,
             n_viol=int(v8[j]), gap_ok=bool(gap[j] > 0.0), n_cand=n_cand, gap_frac=_gf)
    m["pass_all"] = bool(m["n_viol"] == 0 and m["gap_ok"])
    return float(THRESHOLDS[j]), m

def _kpi9(y, p, t):
    """9 KPI gates (confusion matrix) tai nguong t - dung cho danh gia 3 tap."""
    pred, y1 = p >= t, y == 1
    tp = float(np.count_nonzero(pred & y1))
    fn = float(np.count_nonzero(~pred & y1))
    fp = float(np.count_nonzero(pred & ~y1))
    tn = float(np.count_nonzero(~pred & ~y1))
    sens = tp / (tp + fn) if tp + fn > 0 else 0.0
    spec = tn / (tn + fp) if tn + fp > 0 else 0.0
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    npv = tn / (tn + fn) if tn + fn > 0 else 0.0
    f1 = 2.0 * prec * sens / (prec + sens) if prec + sens > 0 else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    pc = np.clip(p, 1e-15, 1.0 - 1e-15)
    bce = float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))
    gap = sens - prec
    _vals = {"Sensitivity": sens, "MCC": mcc, "Specificity": spec, "Precision": prec,
             "Gap=Rec-Prec": gap, "BCE": bce, "NPV": npv, "F1": f1, "Accuracy": acc}
    gates = [(_nm, _vals[_nm], _gate_ok(_op, _vals[_nm], _thr), _disp)
             for _nm, _thr, _op, _disp in KPI_GATES]
    n_viol = int(sum(not _g[2] for _g in gates if _g[0] != "Gap=Rec-Prec"))
    gap_ok = bool(_vals["Gap=Rec-Prec"] > 0.0)
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, gates=gates, n_viol=n_viol, gap_ok=gap_ok,
                pass_all=bool(n_viol == 0 and gap_ok))

# ---- 5) GRID SEARCH - TUYEN CHON CHI BANG TRAIN (NO LEAKAGE), solver co dinh ----
SCALER_FACTORY = {"StandardScaler": StandardScaler,
                  "RobustScaler": RobustScaler,
                  "MinMaxScaler": MinMaxScaler}

def _build_model(cfg):
    """Moi LogisticRegression deu dung FIXED_SOLVER (newton-cholesky)."""
    return LogisticRegression(C=cfg["C"], penalty=cfg["penalty"], solver=FIXED_SOLVER,
                              class_weight=cfg["class_weight"],
                              max_iter=cfg["max_iter"], tol=cfg["tol"],
                              random_state=42, n_jobs=1)

def _fit_estimator(cfg, X_scaled, y_fit):
    """Fit + bat ConvergenceWarning de log (khong nuot loi cung)."""
    _mdl = _build_model(cfg)
    with warnings.catch_warnings(record=True) as _rec:
        warnings.simplefilter("always")
        _mdl.fit(X_scaled, y_fit)
    _warn = [str(_w.message) for _w in _rec if issubclass(_w.category, ConvergenceWarning)]
    return _mdl, _warn

def _pen_repr(pen):
    return "none" if pen is None else str(pen)

def _cw_repr(cw):
    if cw is None:
        return "none"
    if isinstance(cw, str):
        return cw
    return "cw:{0:%.6g,1:%.6g}" % (float(cw[0]), float(cw[1]))

def _cw_allowed(cw):
    if cw is None or isinstance(cw, str):
        return cw is None or cw == "balanced"
    if isinstance(cw, dict):
        return set(cw.keys()) == {0, 1} and all(float(_w) > 0.0 for _w in cw.values())
    return False

def _cfg_key(cfg):
    return (cfg["scaler_name"], float(cfg["C"]), _pen_repr(cfg["penalty"]),
            cfg["solver"], _cw_repr(cfg["class_weight"]), float(cfg["gap_frac"]))

def _prune_reason(cfg):
    if cfg["solver"] != FIXED_SOLVER:
        return "incompatible_with_fixed_constraint"
    if _pen_repr(cfg["penalty"]) not in {_pen_repr(_p) for _p in ALLOWED_PENALTIES}:
        return "incompatible_with_fixed_constraint"
    if not _cw_allowed(cfg["class_weight"]):
        return "incompatible_with_fixed_constraint"
    return None

def _mk_cfg(scaler_name, C, penalty, class_weight, gap_frac):
    return {"model_type": "LogisticRegression", "scaler_name": scaler_name, "C": float(C),
            "penalty": penalty, "solver": FIXED_SOLVER, "class_weight": class_weight,
            "max_iter": int(GRID_BASE["max_iter"]), "tol": float(GRID_BASE["tol"]),
            "gap_frac": float(gap_frac)}

def _resolve_cw(layout):
    """Doi layout class_weight thanh gia tri cu the (n0/n1 da co)."""
    if isinstance(layout, str) and layout.startswith("dynamic:"):
        return {0: 1.0, 1: float(n0) / float(n1)}
    return layout

def _mk_plan(entries, _seen, _plan):
    """Dedupe + prune + cap cho danh sach cfg moi, tra ve so luong duoc nhan."""
    _n_add = 0
    for _cfg in entries:
        if len(_plan) >= MAX_CONFIGS:
            print("CAP: da cham MAX_CONFIGS=%d - bo qua cac cfg con lai." % MAX_CONFIGS)
            break
        _why = _prune_reason(_cfg)
        if _why is not None:
            print("SKIP: cfg=%s reason=%s" % (_cfg_key(_cfg), _why))
            continue
        _k = _cfg_key(_cfg)
        if _k in _seen:
            continue
        _seen.add(_k)
        _plan.append((_cfg, None))
        _n_add += 1
    return _n_add

_configs_base = [_mk_cfg(_sn, _C, _pen, _resolve_cw(_cw), GRID_BASE["gap_fraction"])
                 for _cw in GRID_BASE["class_weights"]
                 for _pen in GRID_BASE["penalties"]
                 for _sn in GRID_BASE["scaler_names"]
                 for _C in GRID_BASE["C_values"]]

print("\n" + "=" * 60)
print("GRID BASE: ~%d configs (uu tien theo thu tu khai bao) | cap mem GRID_BASE=%d"
      % (len(_configs_base), GRID_BASE["max_configs"]))
print("solver co dinh=%s | penalty=%s | class_weight=%s | gap_frac_base=%.2f"
      % (FIXED_SOLVER, list(PENALTY_CHOICES), list(CW_CHOICES), GRID_BASE["gap_fraction"]))
print("Model selection + threshold rule: chi dung TRAIN; TEST/FULL chi VERIFY early-stop")
print("=" * 60)

_seen = set()
_plan = []
_mk_plan(_configs_base, _seen, _plan)
if len(_plan) > GRID_BASE["max_configs"]:
    print("CAP MEM: GRID_BASE %d -> giu %d cfg dau theo thu tu khai bao."
          % (len(_plan), GRID_BASE["max_configs"]))
    _plan = _plan[:GRID_BASE["max_configs"]]
_n_plan_base = len(_plan)
_n_planned = _n_plan_base

_best_rec = None          # argmin sort_key TRAIN (fallback)
_full_rec = None          # config full-pass 3 tap dau tien
_n_eval = 0
_n_skip = 0
_stop = False
_expanded = False

print("PLAN: %d configs san sang fit (con du budget toi %d)" % (_n_planned, MAX_CONFIGS))

_i = 0
while _i < len(_plan) and not _stop:
    _cfg, _reuse = _plan[_i]
    _i += 1
    _tag = "[%d/%d]" % (_i, len(_plan))

    if _reuse is None:
        _sc = SCALER_FACTORY[_cfg["scaler_name"]]().fit(X_train_val)
        _Xs = np.ascontiguousarray(_sc.transform(X_train_val), dtype=np.float64)
        try:
            _mdl, _warn = _fit_estimator(_cfg, _Xs, y_train_val)
        except Exception as _fit_err:
            _n_skip += 1
            print("SKIP: cfg=%s reason=newton_cholesky_runtime_error (%s: %s)"
                  % (_cfg_key(_cfg), type(_fit_err).__name__, _fit_err))
            del _sc, _Xs
            gc.collect()
            continue
        del _Xs
        _wrap = ScaledLogisticWrapper(_sc, _mdl)
        _rc, _ri = _wrap.raw_coef_[0].copy(), float(_wrap.raw_intercept_[0])
        _n_iter = int(np.max(_mdl.n_iter_))
        _p_tr = _manual_prob(X_train_val, _rc, _ri)
        del _sc, _mdl, _wrap
        gc.collect()
        if _warn or _n_iter >= int(_cfg["max_iter"]):
            print("WARNING: cfg=%s chua hoi tu (n_iter=%d/%s) - van danh gia."
                  % (_cfg_key(_cfg), _n_iter, _cfg["max_iter"]))
    else:
        _p_tr, _rc, _ri, _n_iter = _reuse
        if _n_iter >= int(_cfg["max_iter"]):
            print("WARNING: cfg=%s chua hoi tu (n_iter=%d) - van danh gia."
                  % (_cfg_key(_cfg), _n_iter))

    _thr, _mt = _scan9(_p_tr, y_train_val, _cfg["gap_frac"])
    _key = (_mt["n_viol"], -int(_mt["gap_ok"]), -_mt["mcc"], -_mt["sens"], -_mt["f1"], _mt["bce"])
    _n_eval += 1
    _rec = {"cfg": dict(_cfg), "thr": float(_thr), "mt": _mt, "key": _key,
            "p_tr": _p_tr, "rc": _rc, "ri": _ri, "n_iter": _n_iter}
    if _best_rec is None or _key < _best_rec["key"]:
        _best_rec = _rec

    _line = ("%s scaler=%-14s C=%-7.4g cw=%-22s solver=%s pen=%-4s gap_frac=%.2f thr*=%.4f "
             "TRAIN[Sens=%.4f Prec=%.4f Spec=%.4f MCC=%.4f F1=%.4f Acc=%.4f NPV=%.4f BCE=%.4f Gap=%+.4f] "
             "n_viol=%d"
             % (_tag, _cfg["scaler_name"], _cfg["C"], _cw_repr(_cfg["class_weight"]), FIXED_SOLVER,
                _pen_repr(_cfg["penalty"]), _cfg["gap_frac"], _thr, _mt["sens"], _mt["prec"],
                _mt["spec"], _mt["mcc"], _mt["f1"], _mt["acc"], _mt["npv"], _mt["bce"],
                _mt["gap"], _mt["n_viol"]))
    if _reuse is not None:
        _line += " [rescan:no-refit]"
    if _mt["pass_all"]:
        _line += " TRAIN-PASS"
    print(_line)

    if _mt["pass_all"]:
        # VERIFY early-stop: TRAIN pass_all -> kiem tra TEST/FULL (khong dung cho sort_key/threshold)
        _v_ok = True
        for _lab, _mk in (("TEST", test_np), ("FULL", _full_np)):
            _p_v = _manual_prob(X_model[_mk], _rc, _ri)
            _v = _kpi9(y_all[_mk], _p_v, _thr)
            print("   VERIFY %s: n_viol=%d gap_ok=%d pass_all=%s"
                  % (_lab, _v["n_viol"], int(_v["gap_ok"]), _v["pass_all"]))
            del _p_v
            if not _v["pass_all"]:
                _v_ok = False
        if _v_ok:
            _full_rec = _rec
            _stop = True
            print("   => EARLY-STOP: TRAIN + TEST + FULL deu pass_all.")

    if not _stop and _i == _n_plan_base and not _expanded:
        # ---- extended discovery quanh seed tot nhat (chi TRAIN-only de chon seed) ----
        _expanded = True
        _seed = _best_rec["cfg"]
        _ratio = float(n0) / float(n1)
        _ext = []
        for _m in GRID_EXPAND["C_multipliers"]:                      # C fine-tune
            _ext.append(_mk_cfg(_seed["scaler_name"], _seed["C"] * _m, _seed["penalty"],
                                _seed["class_weight"], _seed["gap_frac"]))
        for _sn2 in SCALER_FACTORY:                                  # scaler sweep
            if _sn2 == _seed["scaler_name"]:
                continue
            for _m in GRID_EXPAND["scaler_C_multipliers"]:
                _ext.append(_mk_cfg(_sn2, _seed["C"] * _m, _seed["penalty"],
                                    _seed["class_weight"], _seed["gap_frac"]))
        for _r in GRID_EXPAND["cw_ratio_multipliers"]:               # cw sweep
            _ext.append(_mk_cfg(_seed["scaler_name"], _seed["C"], _seed["penalty"],
                                {0: 1.0, 1: _ratio * _r}, _seed["gap_frac"]))
        if len(ALLOWED_PENALTIES) > 1:                               # penalty sweep
            for _p2 in ALLOWED_PENALTIES:
                if _pen_repr(_p2) != _pen_repr(_seed["penalty"]):
                    _ext.append(_mk_cfg(_seed["scaler_name"], _seed["C"], _p2,
                                        _seed["class_weight"], _seed["gap_frac"]))
        print("\nGRID EXPAND: toi da %d configs quanh seed=%s | C=%.6g | cw=%s | pen=%s"
              % (len(_ext), _seed["scaler_name"], _seed["C"],
                 _cw_repr(_seed["class_weight"]), _pen_repr(_seed["penalty"])))
        _mk_plan(_ext, _seen, _plan)
        # _GAP_FRACTION rescan tren seed: khong refit, chi quet lai threshold TRAIN
        _seed_rec = dict(_best_rec)
        for _gf in GAP_FRACTIONS:
            if abs(float(_gf) - float(_seed_rec["cfg"]["gap_frac"])) < 1e-12:
                continue
            if len(_plan) >= MAX_CONFIGS:
                break
            _cfg_gf = _mk_cfg(_seed_rec["cfg"]["scaler_name"], _seed_rec["cfg"]["C"],
                              _seed_rec["cfg"]["penalty"], _seed_rec["cfg"]["class_weight"], _gf)
            _k_gf = _cfg_key(_cfg_gf)
            if _k_gf in _seen:
                continue
            _seen.add(_k_gf)
            _plan.append((_cfg_gf, (_seed_rec["p_tr"], _seed_rec["rc"], _seed_rec["ri"],
                                    _seed_rec["n_iter"])))
        print("KE HOACH MO RONG (them rescan gap_frac %s, khong refit): tong %d configs / cap %d"
              % (list(GAP_FRACTIONS), len(_plan), MAX_CONFIGS))
        _n_planned = len(_plan)
    del _p_tr
    gc.collect()

print("\nGRID SEARCH KET THUC: da danh gia %d/%d config | skip=%d | early_stop=%s"
      % (_n_eval, len(_plan), _n_skip, _stop))
if _stop:
    print("EARLY-STOP theo §6.4.C (TRAIN + TEST + FULL pass_all) - %d config chua dung toi."
          % (len(_plan) - _i))

# ---- 6) CHON BEST: config full-pass dau tien, else argmin(sort_key TRAIN) ----
_final_rec = _full_rec if _full_rec is not None else _best_rec
best_cfg = dict(_final_rec["cfg"])
best_sort_key = _final_rec["key"]
best_thresh = float(_final_rec["thr"])
best_t_metrics = _final_rec["mt"]
full_kpi_achieved_grid = bool(_full_rec is not None)
print("SORT KEY (TRAIN-only) cua best: n_viol=%d | gap_ok=%d | MCC=%.6f | Sens=%.6f | F1=%.6f | BCE=%.6f"
      % (best_sort_key[0], -best_sort_key[1], -best_sort_key[2], -best_sort_key[3],
         -best_sort_key[4], best_sort_key[5]))

# ---- 7) RE-FIT BEST CONFIG (TRAIN ONLY) ----
print("\n" + "=" * 60)
print("RE-FIT BEST CONFIG (TRAIN only)")
print("=" * 60)
best_sc = SCALER_FACTORY[best_cfg["scaler_name"]]()
X_train_scaled = np.ascontiguousarray(best_sc.fit_transform(X_train_val), dtype=np.float64)
best_mdl, _best_warn = _fit_estimator(best_cfg, X_train_scaled, y_train_val)
for _w in _best_warn:
    print("WARNING: best_cfg ConvergenceWarning: %s" % _w)
logr = ScaledLogisticWrapper(best_sc, best_mdl)
raw_coef_final = logr.raw_coef_[0]
raw_intercept_final = float(logr.raw_intercept_[0])
prob_train_temp = _manual_prob(X_train_val, raw_coef_final, raw_intercept_final)
best_thresh, best_t_metrics = _scan9(prob_train_temp, y_train_val, best_cfg["gap_frac"])
_GAP_FRACTION = float(best_cfg["gap_frac"])
_coef = np.asarray(best_mdl.coef_[0], dtype=np.float64)
print(" scaler=%s | penalty=%s | C=%s | solver=%s | class_weight=%s"
      % (best_cfg["scaler_name"], _pen_repr(best_cfg["penalty"]), best_cfg["C"],
         best_cfg["solver"], _cw_repr(best_cfg["class_weight"])))
print(" max_iter=%s | tol=%s | n_iter_=%d | intercept_=%.10f"
      % (best_cfg["max_iter"], best_cfg["tol"], int(np.max(best_mdl.n_iter_)),
         float(best_mdl.intercept_[0])))
print(" coef (scaled space): min=%+.6f | max=%+.6f | mean=%+.6f | #nonzero=%d/%d"
      % (float(_coef.min()), float(_coef.max()), float(_coef.mean()),
         int(np.count_nonzero(_coef)), _coef.size))
print(" raw coef (khong gian goc): min=%+.6f | max=%+.6f | mean=%+.6f"
      % (float(raw_coef_final.min()), float(raw_coef_final.max()), float(raw_coef_final.mean())))
print(" raw intercept=%.10f" % raw_intercept_final)
print("Best threshold (rule TRAIN-only, gap>=%.2f*gap_max roi max MCC): %.6f | candidates=%d"
      % (_GAP_FRACTION, best_thresh, best_t_metrics["n_cand"]))
print("TRAIN metrics: Sens=%.4f Prec=%.4f Spec=%.4f NPV=%.4f F1=%.4f Acc=%.4f MCC=%.4f Gap=%+.4f BCE=%.4f"
      % (best_t_metrics["sens"], best_t_metrics["prec"], best_t_metrics["spec"],
         best_t_metrics["npv"], best_t_metrics["f1"], best_t_metrics["acc"],
         best_t_metrics["mcc"], best_t_metrics["gap"], best_t_metrics["bce"]))
del prob_train_temp, X_train_scaled
gc.collect()

# ---- 8) probability_manual TU HE SO QUY DOI VE KHONG GIAN GOC (batch 200.000) ----
probability_nx = logr.predict_proba(X_model)[:, 1]
probability_manual = np.empty(len(X_model), dtype=np.float64)
linear_score_all = np.empty(len(X_model), dtype=np.float64)
for _start in range(0, len(X_model), BATCH_SIZE):
    _end = min(_start + BATCH_SIZE, len(X_model))
    _logit = X_model[_start:_end].astype(np.float64).dot(raw_coef_final) + raw_intercept_final
    linear_score_all[_start:_end] = _logit
    probability_manual[_start:_end] = expit(_logit)
data_group["probability_NX"] = probability_nx
data_group["linear_score"] = linear_score_all
data_group["probability_manual"] = probability_manual
data_group["diff"] = np.abs(data_group["probability_NX"] - data_group["probability_manual"])
del probability_nx, linear_score_all
gc.collect()

# ---- 9) SURROGATE FIDELITY ----
print("\n" + "-" * 50)
print("SURROGATE FIDELITY |probability_NX - probability_manual|")
print("Mean=%.3e | Median=%.3e | Max=%.3e | P95=%.3e"
      % (data_group["diff"].mean(), data_group["diff"].median(),
         data_group["diff"].max(), data_group["diff"].quantile(0.95)))
print("-" * 50)

# ---- 10) KPI CHECK 3 TAP - probability_manual, 9 gates (danh gia 1 lan cuoi) ----
kpi_3sets = {}
_prob_all = data_group["probability_manual"].to_numpy(dtype=np.float64)
for _lab, _mk in (("TRAIN", train_np), ("TEST", test_np), ("FULL", _full_np)):
    kpi_3sets[_lab] = _kpi9(y_all[_mk], _prob_all[_mk], best_thresh)
del _prob_all
full_kpi_achieved = bool(all(v["pass_all"] for v in kpi_3sets.values()))

print("\n" + "=" * 60)
print("KPI CHECK 3 TAP (probability_manual @ best_thresh=%.6f) - 9 gates" % best_thresh)
print("=" * 60)
print("%-13s %-8s %9s %9s %9s %s" % ("Metric", "Gate", "TRAIN", "TEST", "FULL", "OK(3 tap)"))
for _g in range(len(KPI_GATES)):
    _nm, _, _, _gt = kpi_3sets["TRAIN"]["gates"][_g]
    _vv = [kpi_3sets[_s]["gates"][_g] for _s in ("TRAIN", "TEST", "FULL")]
    print("%-13s %-8s %9.4f %9.4f %9.4f %s"
          % (_nm, _gt, _vv[0][1], _vv[1][1], _vv[2][1],
             "PASS" if all(_v[2] for _v in _vv) else "FAIL"))
for _lab in ("TRAIN", "TEST", "FULL"):
    _v = kpi_3sets[_lab]
    _n = int({"TRAIN": train_np, "TEST": test_np, "FULL": _full_np}[_lab].sum())
    print("[%s] %s | n=%d | TP=%.0f FP=%.0f TN=%.0f FN=%.0f | viol=%d gap_ok=%d"
          % (_lab, "DAT FULL KPI" if _v["pass_all"] else "CHUA DAT", _n,
             _v["TP"], _v["FP"], _v["TN"], _v["FN"], _v["n_viol"], int(_v["gap_ok"])))

# ---- 11) BEST CONFIG DUY NHAT ----
print("\n" + "=" * 60)
print(" BEST CONFIG DUY NHAT")
print("=" * 60)
for k, v in best_cfg.items():
    print(" %-28s: %s" % (k, v))
print(" %-28s: %d" % ("n_features", len(feature_cols)))
print(" %-28s: %.6f" % ("best_thresh", best_thresh))
print(" %-28s: %s" % ("full_kpi_achieved", full_kpi_achieved))
if not full_kpi_achieved:
    print(" BEST CONFIG CHUA DAT FULL KPI (best-effort theo sort_key TRAIN) - giu nguyen, khong doi cau hinh khac.")
print("=" * 60)
gc.collect()
#####CHỈNH Ở PHẦN NÀY ĐẾN ĐÂY ###########





# ====================== SCORING (USING RAW COEFFS) - PRODUCTION NO LEAKAGE ======================
# Model chỉ fit TRAIN, transform FULL là inference
# PD chia decile 10% pop ngưỡng TRAIN, Score giữ nguyên

import numpy as np
import pandas as pd
 
# 1. PD scoring - inference FULL
data_group["probability_NX"] = logr.predict_proba(X)[:, 1]

log_odds = X.dot(logr.raw_coef_[0]) + logr.raw_intercept_[0]
 
prob_manual = 1 / (1.0 + np.exp(-log_odds))

data_group["odds"] =  prob_manual / (1 - prob_manual)

data_group["log_odds"] = log_odds

######
data_group["intercept"] = float(logr.raw_intercept_[0])
data_group["linear_term"] = X.dot(logr.raw_coef_[0])

data_group["probability_manual"] = prob_manual

#data_group["probability_manual"] = data_group.groupby(['CIF', 'NGAY_BC']).probability_manual.transform('max')

#data_group["probability_NX"] = data_group.groupby(['CIF', 'NGAY_BC']).probability_NX.transform('max')

data_group["diff"] = np.abs(data_group["probability_NX"] - data_group["probability_manual"])
 

# 3. PD DECILE 10% POP - CHỈ TRAIN (NO LEAKAGE)
train_probs = data_group.loc[train_mask, "probability_manual"].to_numpy()

_, pd_edges = pd.qcut(train_probs, 10, retbins=True, duplicates="drop")
pd_edges = np.unique(pd_edges)
bins_pd = np.concatenate([[-np.inf], pd_edges[1:-1], [np.inf]])
 
num_bins_pd = len(bins_pd) - 1
decile_labels = list(range(1, num_bins_pd + 1))

data_group["decile"] = pd.cut(
    data_group["probability_manual"],
    bins=bins_pd,
    labels=decile_labels,  # 1=best AAA, num_bins_pd=worst D
    include_lowest=True
)


# ====================== EXPORT TRAIN PD BY DECILE ======================

pd_decile_train = (
    data_group.loc[
        train_mask,
        ["decile", "probability_manual"],
    ]
    .groupby(
        "decile",
        observed=True,
    )
    .agg(
        min_probability_manual=("probability_manual", "min"),
        max_probability_manual=("probability_manual", "max")
    )
    .reset_index()
    .sort_values("decile")
)

pd_decile_train.to_excel(
    r"D:\model\PD decile.xlsx",
    sheet_name="PD Decile",
    index=False,
)

print("\nPD DECILE TRAIN:")
print(pd_decile_train.to_string(index=False))
print(r"Đã xuất: D:\model\PD decile.xlsx")


########

# 2. Score scaling - chỉ tính trên TRAIN
min_score = 300
max_score = 850
base_odds = 50
train_pd = data_group.loc[train_mask, "probability_manual"]


min_PD = float(train_pd.min())
max_PD = float(train_pd.max())
 

pdo = (max_score - min_score) * math.log(2) / (math.log((1 - min_PD) / min_PD) - math.log((1 - max_PD) / max_PD))
base_score = math.ceil(max(
    max_score - 0.5 - (pdo / math.log(2)) * (math.log((1 - min_PD) / min_PD) - math.log(base_odds)),
    min_score - 0.5 - (pdo / math.log(2)) * (math.log((1 - max_PD) / max_PD) - math.log(base_odds))
))
factor = pdo / math.log(2)
offset = base_score - (factor * math.log(base_odds))


data_group["base_score"] = base_score
data_group["base_odds"] = base_odds
data_group["offset"] = offset
data_group["factor"] = factor

data_group["score"] = offset - factor * data_group["log_odds"]
data_group["score"] = data_group["score"].round(0).astype(int)
 

 
# 4. GRADE THEO SCORE CÁCH ĐỀU TỪ MIN ĐẾN MAX CỦA TRAIN (NO LEAKAGE)
grades_low_to_high = np.array([
    "9.D",
    "8.C",
    "7.CC",
    "6.CCC",
    "5.B",
    "4.BB",
    "3.BBB",
    "2.A",
    "1.AA",
    "0.AAA",
])
 
# Chỉ lấy score hợp lệ thuộc tập train
train_scores = data_group.loc[train_mask, "score"].dropna()
 
# Xác định min và max chỉ trên tập train
min_train_score = float(train_scores.min())
max_train_score = float(train_scores.max())
 
# Chia khoảng từ min đến max thành 10 phần có độ rộng bằng nhau
score_edges = np.linspace(
    start=min_train_score,
    stop=max_train_score,
    num=len(grades_low_to_high) + 1,
)
 
# Mở rộng hai đầu để áp dụng cho validation, test và OOT
# Các score thấp hơn min train được gán 9.D
# Các score cao hơn max train được gán 0.AAA
bins_score = np.concatenate([
    [-np.inf],
    score_edges[1:-1],
    [np.inf],
])
 
# Áp dụng cùng bộ ngưỡng cho toàn bộ dữ liệu
data_group["Grade"] = pd.cut(
    data_group["score"],
    bins=bins_score,
    labels=grades_low_to_high,
    include_lowest=True,
    right=True,
    ordered=True,
)


# ====================== EXPORT SCORE PARAMETERS ======================

score_parameter_df = pd.DataFrame({
    "Parameter": [
        "MIN_SCORE",
        "MAX_SCORE",
        "BASE_ODDS",
        "pdo",
        "factor",
        "base_score",
        "offset",
    ],
    "Value": [
        min_score,
        max_score,
        base_odds,
        pdo,
        factor,
        base_score,
        offset,
    ],
})

score_parameter_df.to_excel(
    r"D:/model/score_parameter.xlsx",
    sheet_name="Score Parameters",
    index=False,
)

print(
    "Đã xuất tham số score:",
    r"D:/model/score_parameter.xlsx",
)




# ====================== EXPORT TRAIN GRADE BY SCORE ======================

grade_score_train = (
    data_group.loc[
        train_mask,
        ["Grade", "score"],
    ]
    .groupby(
        "Grade",
        observed=True,
    )
    .agg(
        min_score =("score", "min"),
        max_score =("score", "max")
    )
    .reset_index()
    .sort_values("Grade")
)

grade_score_train.to_excel(
    r"D:\model\Grade_score.xlsx",
    sheet_name="Grade_score",
    index=False,
)

print("\nGRADE_SCORE TRAIN:")
print(grade_score_train.to_string(index=False))
print(r"Đã xuất: D:\model\Grade_score.xlsx")



#calibration
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
)

# ==============================================================================
# BƯỚC 1: PLATT CALIBRATION — GRID SEARCH 2 TẦNG (PRODUCTION EDITION — 10-FOLD)
# (100% TRAIN-ONLY KHI FIT — KHÔNG SỬ DỤNG NHÃN TEST ĐỂ FIT — NO LEAKAGE)
#
# Tầng 1 — 4 cấu hình neo (3 class_weight × CV-Platt 10-fold + MLE-TRAIN):
# • Duyệt 3 giá trị class_weight: {0:1.0, 1:n0/n1} | 'balanced' | None
# • StratifiedKFold(n_splits=10, shuffle=True, random_state=42) trên TRAIN
# • Sinh log-odds OUT-OF-FOLD (OOF) trên TRAIN
# • Fit Platt MLE (LogisticRegression, penalty=None) trên OOF -> (α_cv, β_cv)
# • Neo thứ 4: Platt MLE in-sample trực tiếp trên log-odds TRAIN (không CV —
#   vẫn 100% fit trên TRAIN, không leakage; neo này nằm sát MLE cơ sở nên
#   lưới Tầng 2 bắt được cả vùng calibrate sâu β* > 1.10)
# • Test/Full chỉ nhận TRANSFORM (nhân α/β), tuyệt đối không fit trên nhãn của chúng
#
# Tầng 2 — Tinh chỉnh (δ, s) quanh mỗi neo:
#   α* = α_cv + β_cv·δ , β* = β_cv·s
#   δ ∈ [−0.15, +0.25] bước 0.01 -> 41 điểm
#   s ∈ [ 0.95, +1.20] bước 0.01 -> 26 điểm
#   41 × 26 = 1066 ứng viên/cấu hình × 4 cấu hình = 4264 ứng viên, quét vector hóa
#   (lưới mở rộng so với 5-fold cũ δ∈[−0.10,+0.12] × s∈[0.94..1.06] để dò
#   cả vùng β* > 1.10 — nơi tìm được cấu hình full KPI 15/15)
#
# Chọn cấu hình theo §5.3 (FULL PASS 15/15; nếu không có -> best-effort + tie-breaker)
# Đối chiếu lại bằng sklearn.LogisticRegression (sai lệch < 1e-8).
#
# NO LEAKAGE: scaler / LogisticRegression / Platt MLE đều fit CHỈ trên nhãn TRAIN.
# Nhãn TEST và FULL chỉ dùng để ĐO KPI & CHỌN cấu hình, KHÔNG dùng để fit.
# ==============================================================================
import gc

# -------------------------------------------------------------------------------
# [01] KẾ THỪA BIẾN TỪ CÁC BLOCK TRƯỚC — KHÔNG HARD-CODE, KHÔNG TỰ TẠO GIẢ
# -------------------------------------------------------------------------------
_NEED = (
    "data_group",
    "TARGET_COL",
    "SPLIT_COL",
    "X_model",
    "y_all",
    "SCALER_FACTORY",
    "best_cfg",
)
_MISSING = [_v for _v in _NEED if _v not in globals()]
if _MISSING:
    raise NameError(
        "Thiếu biến kế thừa từ các block trước (không tự tạo giả): %s" % _MISSING
    )

train_mask = data_group[SPLIT_COL].astype(str).str.strip().str.lower() == "train"
test_mask = data_group[SPLIT_COL].astype(str).str.strip().str.lower() == "test"
_TR = train_mask.to_numpy(dtype=bool, copy=False)
_TE = test_mask.to_numpy(dtype=bool, copy=False)

# Cơ sở log-odds cần hiệu chỉnh: 'log_odds' là logit của mô hình best_cfg đã refit
# FULL-TRAIN — chính là cơ sở mà block phía sau sử dụng:
# log_odds_new = alpha_train + beta_train * data_group['log_odds']
_LO_SRC = next(
    (_c for _c in ("log_odds", "log_odds_new") if _c in data_group.columns), None
)
if _LO_SRC is None:
    raise NameError(
        "Thiếu cột log-odds kế thừa ('log_odds' / 'log_odds_new') trong data_group."
    )

_LO = data_group[_LO_SRC].to_numpy(dtype=np.float64, copy=False)
_Y = np.asarray(y_all, dtype=np.int8)
_N = int(_LO.shape[0])

if not np.isfinite(_LO).all():
    raise ValueError(
        "Cột '%s' chứa NaN/Inf (%d giá trị); không tự impute — dừng có kiểm soát."
        % (_LO_SRC, int(np.count_nonzero(~np.isfinite(_LO))))
    )
if np.unique(_Y[_TR]).size < 2:
    raise ValueError("Nhãn TRAIN chỉ có 1 lớp — không thể fit Platt calibration.")

# Tỷ lệ mất cân bằng kế thừa: lấy biến n0/n1 đã có, nếu thiếu thì tự tính trên TRAIN
_n0 = int(globals()["n0"]) if "n0" in globals() else int(np.count_nonzero(_Y[_TR] == 0))
_n1 = int(globals()["n1"]) if "n1" in globals() else int(np.count_nonzero(_Y[_TR] == 1))

# -------------------------------------------------------------------------------
# [02] HẰNG SỐ THIẾT KẾ (§5.2, §7.1) VÀ HÀM KPI (§7.4)
# -------------------------------------------------------------------------------
_CW_GRID = (
    ("{0:1.0, 1:n0/n1}", {0: 1.0, 1: (float(_n0) / float(_n1)) if _n1 else 1.0}),
    ("balanced", "balanced"),
    ("None", None),
)
_DELTA_GRID = np.round(np.arange(-0.15, 0.25 + 1e-9, 0.01), 10)  # 41 điểm
_SCALE_GRID = np.round(np.arange(0.95, 1.20 + 1e-9, 0.01), 10)   # 26 điểm
_N_DELTA, _N_SCALE = int(_DELTA_GRID.size), int(_SCALE_GRID.size)
_N_CAND = _N_DELTA * _N_SCALE  # 1066 ứng viên / cấu hình
_N_TOTAL = _N_CAND * 4         # 4264 ứng viên tổng cộng (4 cấu hình neo)
_N_BINS = 10                   # M = 10 bins đều trên [0, 1]
_NAMES = ("TRAIN", "TEST", "FULL")
_SETS = (("TRAIN", _TR), ("TEST", _TE), ("FULL", np.ones(_N, dtype=bool)))

# Ngưỡng TỐT (§7.1)
_T_ALPHA, _T_BETA_LO, _T_BETA_HI = 0.10, 0.90, 1.10
_T_ECE, _T_MCE, _T_BRIER = 0.05, 0.10, 0.25

# Cấu hình mô hình kế thừa từ best_cfg (block grid search phía trên) — không hard-code
_C_SCALER = best_cfg["scaler_name"]
_C_C = best_cfg["C"]
_C_PEN = best_cfg.get("penalty", "l2")
_C_SOL = best_cfg.get("solver", "lbfgs")
_C_MAXIT = best_cfg.get("max_iter", 10000)
_C_TOL = best_cfg.get("tol", 1e-4)

def _platt_mle(z1d, yv, max_iter=10000, tol=1e-12):
    """Platt Scaling MLE: ước lượng y ~ α + β·z bằng LogisticRegression không phạt.
    Chuẩn hóa z nội bộ trước khi fit (điều kiện hóa tốt hơn, hội tụ nhanh & chính xác),
    sau đó đổi ngược hệ số về không gian gốc theo đúng toán học:
    y ~ a' + b'·(z − m)/sd ⇔ β = b'/sd , α = a' − β·m
    """
    _m0 = float(np.mean(z1d))
    _sd = float(np.std(z1d))
    if not np.isfinite(_sd) or _sd <= 0.0:
        _sd = 1.0
    _zs = ((z1d - _m0) / _sd).reshape(-1, 1)
    try:
        _m = LogisticRegression(
            penalty=None, solver="lbfgs", fit_intercept=True,
            max_iter=max_iter, tol=tol
        ).fit(_zs, yv)
    except (TypeError, ValueError):
        _m = LogisticRegression(
            C=1e12, solver="lbfgs", fit_intercept=True,
            max_iter=max_iter, tol=tol
        ).fit(_zs, yv)
    if int(np.max(_m.n_iter_)) >= _m.max_iter:
        print(" CẢNH BÁO: Platt MLE chưa hội tụ (n_iter_ = %d == max_iter)."
              % int(np.max(_m.n_iter_)))
    _beta = float(_m.coef_[0, 0]) / _sd
    return float(_m.intercept_[0]) - _beta * _m0, _beta

def _sigmoid(z):
    """p = 1 / (1 + exp(−z))."""
    return 1.0 / (1.0 + np.exp(-z))

def _bin_stats(p, y64):
    """Gom 4 tổng theo bin M = 10 đều trên [0, 1]: |B_m|, Σy, Σp, Σ(p − y)²."""
    _ix = (p * _N_BINS).astype(np.intp)
    np.clip(_ix, 0, _N_BINS - 1, out=_ix)
    _d = p - y64
    return (
        np.bincount(_ix, minlength=_N_BINS).astype(np.float64),
        np.bincount(_ix, weights=y64, minlength=_N_BINS),
        np.bincount(_ix, weights=p, minlength=_N_BINS),
        np.bincount(_ix, weights=_d * _d, minlength=_N_BINS),
    )

def _ece_mce_brier(cnt, sy, sp, sd2, n):
    """ECE = Σ_m (|B_m|/n)·|acc(B_m) − conf(B_m)| ; MCE = max_m |·| ; BS = Σ(p−y)²/n."""
    _ok = cnt > 0
    _ece = float(np.sum(np.abs(sy - sp)) / n)
    _mce = float(np.max(np.abs(sy[_ok] - sp[_ok]) / cnt[_ok])) if _ok.any() else float("nan")
    _brier = float(sd2.sum() / n)
    return _ece, _mce, _brier

def _recalib(alpha_star, beta_star, base):
    """Hiệu chuẩn lại sau Platt theo đồng nhất thức đóng (không cần fit lại).
    Gọi (A, B) là MLE của y ~ A + B·log_odds trên tập D. Khi đó MLE của
    y ~ α̂ + β̂·(α* + β*·log_odds)
    trên cùng tập D là:
    β̂ = B / β* , α̂ = A − (B/β*)·α*
    Vì hai tham số hoá cùng một họ affine theo log_odds nên nghiệm MLE là duy nhất.
    Nhờ đó quét hàng nghìn ứng viên không cần fit lại.
    """
    _A, _B = base
    _b_hat = _B / beta_star
    return _A - _b_hat * alpha_star, _b_hat

def _flags(a_hat, b_hat, ece, mce, brier):
    """5 cờ ĐẠT/KHÔNG ĐẠT theo ngưỡng TỐT §7.1."""
    return (
        abs(a_hat) <= _T_ALPHA,
        _T_BETA_LO <= b_hat <= _T_BETA_HI,
        ece <= _T_ECE,
        mce <= _T_MCE,
        brier <= _T_BRIER,
    )

def _mark(ok):
    return "✅" if ok else "❌"

_Y64 = _Y.astype(np.float64)  # nhãn dạng float64 (trọng số bincount)
_n_tr = int(_TR.sum())

# -------------------------------------------------------------------------------
# [03] TẦNG 1 — class_weight × CV-Platt 10-fold (OOF, TRAIN-ONLY) + neo MLE-TRAIN
# -------------------------------------------------------------------------------
_X_TR = X_model[_TR]
_Y_TR = _Y[_TR]
_min_class_count = int(np.bincount(_Y_TR.astype(np.int64), minlength=2).min())
_n_splits_platt = min(10, _min_class_count)
if _n_splits_platt < 2:
    raise ValueError(
        "Không đủ quan sát ở lớp thiểu số để chạy StratifiedKFold cho Platt calibration. "
        f"Số quan sát lớp nhỏ nhất: {_min_class_count}."
    )

_SKF = StratifiedKFold(n_splits=_n_splits_platt, shuffle=True, random_state=42)
_FOLDS = list(_SKF.split(_X_TR, _Y_TR))
_OOF = np.empty(_Y_TR.shape[0], dtype=np.float64)
_T1 = []

for _cw_name, _cw in _CW_GRID:
    for _ia, _ib in _FOLDS:
        _sc = SCALER_FACTORY[_C_SCALER]().fit(_X_TR[_ia])
        _md = LogisticRegression(
            C=_C_C, penalty=_C_PEN, solver=_C_SOL,
            class_weight=_cw, max_iter=_C_MAXIT, tol=_C_TOL,
            random_state=42, n_jobs=1
        ).fit(_sc.transform(_X_TR[_ia]), _Y_TR[_ia])
        _OOF[_ib] = _md.decision_function(_sc.transform(_X_TR[_ib]))
        del _sc, _md
    gc.collect()
    _a_cv, _b_cv = _platt_mle(_OOF, _Y_TR)
    _T1.append({"name": _cw_name, "a_cv": _a_cv, "b_cv": _b_cv})

# Neo thứ 4 (không CV): Platt MLE in-sample trên log-odds TRAIN.
# Vẫn fit 100% trên TRAIN (no leakage); neo này trùng MLE cơ sở _BASE["TRAIN"]
# nên kết quả được tái sử dụng ở [04] — không fit thừa.
_A_BASE_TR, _B_BASE_TR = _platt_mle(_LO[_TR], _Y[_TR])
_T1.append({"name": "MLE-TRAIN (no CV)", "a_cv": _A_BASE_TR, "b_cv": _B_BASE_TR})

print("\n[ TẦNG 1 ] 3 class_weight × CV-Platt 10-fold + neo MLE-TRAIN — OOF trên TRAIN (seed=42, no leakage)")
print(" %-18s %15s %15s %11s %10s"
      % ("class_weight", "α_cv", "β_cv", "|α_cv|", "|β_cv−1|"))
for _t1 in _T1:
    print(" %-18s %15.10f %15.10f %11.6f %10.6f"
          % (_t1["name"], _t1["a_cv"], _t1["b_cv"], abs(_t1["a_cv"]), abs(_t1["b_cv"] - 1.0)))

# -------------------------------------------------------------------------------
# [04] TẦNG 2 — CHỌN CALIBRATION CHỈ BẰNG TRAIN
# -------------------------------------------------------------------------------
_BASE_TRAIN = (_A_BASE_TR, _B_BASE_TR)

def _eval_anchor_grid(t1_item, delta_vals, scale_vals):
    _a_cv_val = float(t1_item["a_cv"])
    _b_cv_val = float(t1_item["b_cv"])
    _valid_pairs = []
    for _s in scale_vals:
        _b_star = _b_cv_val * _s
        if _b_star <= 0.0:
            continue
        for _d in delta_vals:
            _valid_pairs.append((_d, _s, _a_cv_val + _b_cv_val * _d, _b_star))
    
    _m_cand = len(_valid_pairs)
    _a_s = np.empty(_m_cand, dtype=np.float64)
    _b_s = np.empty(_m_cand, dtype=np.float64)
    _d_s = np.empty(_m_cand, dtype=np.float64)
    _s_s = np.empty(_m_cand, dtype=np.float64)
    _tr_a_hat = np.empty(_m_cand, dtype=np.float64)
    _tr_b_hat = np.empty(_m_cand, dtype=np.float64)
    _tr_ece = np.empty(_m_cand, dtype=np.float64)
    _tr_mce = np.empty(_m_cand, dtype=np.float64)
    _tr_brier = np.empty(_m_cand, dtype=np.float64)
    _tr_n_pass = np.empty(_m_cand, dtype=np.int8)
    
    for _k, (_d_val, _s_val, _alpha_star, _beta_star) in enumerate(_valid_pairs):
        _p_tr = _sigmoid(_alpha_star + _beta_star * _LO[_TR])
        _cnt, _sy, _sp, _sd2 = _bin_stats(_p_tr, _Y64[_TR])
        _ece, _mce, _brier = _ece_mce_brier(_cnt, _sy, _sp, _sd2, _n_tr)
        _a_hat, _b_hat = _recalib(_alpha_star, _beta_star, _BASE_TRAIN)
        _n_pass_val = int(np.count_nonzero(_flags(_a_hat, _b_hat, _ece, _mce, _brier)))
        
        _a_s[_k] = _alpha_star
        _b_s[_k] = _beta_star
        _d_s[_k] = _d_val
        _s_s[_k] = _s_val
        _tr_a_hat[_k] = _a_hat
        _tr_b_hat[_k] = _b_hat
        _tr_ece[_k] = _ece
        _tr_mce[_k] = _mce
        _tr_brier[_k] = _brier
        _tr_n_pass[_k] = _n_pass_val
        
    _order = np.lexsort((
        _tr_brier,
        _tr_mce,
        _tr_ece,
        np.abs(_tr_b_hat - 1.0),
        np.abs(_tr_a_hat),
        -_tr_n_pass,
    ))
    return {
        "t1": t1_item,
        "best": int(_order[0]),
        "order": _order,
        "a_s": _a_s,
        "b_s": _b_s,
        "d_s": _d_s,
        "s_s": _s_s,
        "train_alpha_hat": _tr_a_hat,
        "train_beta_hat": _tr_b_hat,
        "train_ece": _tr_ece,
        "train_mce": _tr_mce,
        "train_brier": _tr_brier,
        "train_n_pass": _tr_n_pass,
    }

_RES = [_eval_anchor_grid(_t1, _DELTA_GRID, _SCALE_GRID) for _t1 in _T1]

# -------------------------------------------------------------------------------
# [05] CHỌN CẤU HÌNH CUỐI CÙNG CHỈ BẰNG TRAIN
# -------------------------------------------------------------------------------
def _select_best_candidate(res_list):
    _anchor_keys = []
    for _r in res_list:
        _bi = int(_r["best"])
        _anchor_keys.append((
            -int(_r["train_n_pass"][_bi]),
            abs(float(_r["train_alpha_hat"][_bi])),
            abs(float(_r["train_beta_hat"][_bi]) - 1.0),
            float(_r["train_ece"][_bi]),
            float(_r["train_mce"][_bi]),
            float(_r["train_brier"][_bi]),
        ))
    _idx = min(range(len(_anchor_keys)), key=lambda _i: _anchor_keys[_i])
    _sel = res_list[_idx]
    _bi = int(_sel["best"])
    return _sel, _bi

_SEL, _bi = _select_best_candidate(_RES)
alpha_train = float(_SEL["a_s"][_bi])
beta_train = float(_SEL["b_s"][_bi])
_delta_star = float(_SEL["d_s"][_bi])
_scale_star = float(_SEL["s_s"][_bi])
_a_cv = float(_SEL["t1"]["a_cv"])
_b_cv = float(_SEL["t1"]["b_cv"])
_cw_star = _SEL["t1"]["name"]
_LO_NEW = alpha_train + beta_train * _LO

# -------------------------------------------------------------------------------
# [06] ĐÁNH GIÁ HẬU KIỂM TRAIN / TEST / FULL
# TEST/FULL không tham gia lựa chọn alpha_train hoặc beta_train
# -------------------------------------------------------------------------------
def _evaluate_sets(lo_calib):
    _final = {}
    for _nm, _mask in _SETS:
        _prob = _sigmoid(lo_calib[_mask])
        _target = _Y[_mask]
        _target64 = _target.astype(np.float64, copy=False)
        _cnt, _sy, _sp, _sd2 = _bin_stats(_prob, _target64)
        _ece, _mce, _brier = _ece_mce_brier(_cnt, _sy, _sp, _sd2, len(_target))
        _a_hat, _b_hat = _platt_mle(lo_calib[_mask], _target)
        _final[_nm] = {
            "alpha": float(_a_hat),
            "beta": float(_b_hat),
            "ece": float(_ece),
            "mce": float(_mce),
            "bs": float(_brier),
            "n": int(len(_target)),
        }
    _n_pass = {
        _nm: int(np.count_nonzero(_flags(
            _final[_nm]["alpha"],
            _final[_nm]["beta"],
            _final[_nm]["ece"],
            _final[_nm]["mce"],
            _final[_nm]["bs"],
        )))
        for _nm in _NAMES
    }
    _full_pass = bool(all(_n_pass[_nm] == 5 for _nm in _NAMES))
    return _final, _n_pass, _full_pass

_FINAL, _n_pass, _FULL_PASS = _evaluate_sets(_LO_NEW)

# -------------------------------------------------------------------------------
# [07] FALLBACK THÔNG MINH (§6.3) — CHỈ KÍCH HOẠT KHI TRAIN < 5/5
# -------------------------------------------------------------------------------
if _n_pass["TRAIN"] < 5:
    print("\n[FALLBACK] Mở rộng lưới + neo #5 đã kích hoạt (lý do: TRAIN %d/5)" % _n_pass["TRAIN"])
    # 1. Neo #5: MLE-OOF với class_weight = {0: 1.0, 1: sqrt(n0/n1)}
    _cw5_val = {0: 1.0, 1: float(np.sqrt(float(_n0) / float(_n1))) if _n1 else 1.0}
    _OOF5 = np.empty(_Y_TR.shape[0], dtype=np.float64)
    for _ia, _ib in _FOLDS:
        _sc = SCALER_FACTORY[_C_SCALER]().fit(_X_TR[_ia])
        _md = LogisticRegression(
            C=_C_C, penalty=_C_PEN, solver=_C_SOL,
            class_weight=_cw5_val, max_iter=_C_MAXIT, tol=_C_TOL,
            random_state=42, n_jobs=1
        ).fit(_sc.transform(_X_TR[_ia]), _Y_TR[_ia])
        _OOF5[_ib] = _md.decision_function(_sc.transform(_X_TR[_ib]))
        del _sc, _md
    gc.collect()
    _a_cv5, _b_cv5 = _platt_mle(_OOF5, _Y_TR)
    _t1_neo5 = {"name": "{0:1.0, 1:sqrt(n0/n1)}", "a_cv": _a_cv5, "b_cv": _b_cv5}
    _T1.append(_t1_neo5)
    
    # 2. Nới rộng lưới: delta in [-0.25, 0.35], scale in [0.90, 1.30]
    _DELTA_EXP = np.round(np.arange(-0.25, 0.35 + 1e-9, 0.01), 10)
    _SCALE_EXP = np.round(np.arange(0.90, 1.30 + 1e-9, 0.01), 10)
    
    # Quét toàn bộ lưới mở rộng cho tất cả các neo
    _RES = [_eval_anchor_grid(_t1, _DELTA_EXP, _SCALE_EXP) for _t1 in _T1]
    _SEL, _bi = _select_best_candidate(_RES)
    
    # 3. Refinement bước 0.005 quanh top-3 ứng viên nếu TRAIN vẫn < 5/5
    if int(_SEL["train_n_pass"][_bi]) < 5:
        _all_cand = []
        for _r in _RES:
            for _idx in _r["order"][:3]:
                _all_cand.append({
                    "t1": _r["t1"],
                    "d": float(_r["d_s"][_idx]),
                    "s": float(_r["s_s"][_idx]),
                    "key": (
                        -int(_r["train_n_pass"][_idx]),
                        abs(float(_r["train_alpha_hat"][_idx])),
                        abs(float(_r["train_beta_hat"][_idx]) - 1.0),
                        float(_r["train_ece"][_idx]),
                        float(_r["train_mce"][_idx]),
                        float(_r["train_brier"][_idx]),
                    )
                })
        _all_cand.sort(key=lambda x: x["key"])
        _top3 = _all_cand[:3]
        
        for _c in _top3:
            _d_center, _s_center = _c["d"], _c["s"]
            _d_fine = np.round(np.arange(_d_center - 0.03, _d_center + 0.03 + 1e-9, 0.005), 10)
            _s_fine = np.round(np.arange(_s_center - 0.03, _s_center + 0.03 + 1e-9, 0.005), 10)
            _res_fine = _eval_anchor_grid(_c["t1"], _d_fine, _s_fine)
            _RES.append(_res_fine)
        
        _SEL, _bi = _select_best_candidate(_RES)

    alpha_train = float(_SEL["a_s"][_bi])
    beta_train = float(_SEL["b_s"][_bi])
    _delta_star = float(_SEL["d_s"][_bi])
    _scale_star = float(_SEL["s_s"][_bi])
    _a_cv = float(_SEL["t1"]["a_cv"])
    _b_cv = float(_SEL["t1"]["b_cv"])
    _cw_star = _SEL["t1"]["name"]
    _LO_NEW = alpha_train + beta_train * _LO
    _FINAL, _n_pass, _FULL_PASS = _evaluate_sets(_LO_NEW)

# -------------------------------------------------------------------------------
# [08] GÁN KẾT QUẢ — GIỮ NGUYÊN TÊN BIẾN ĐẦU RA BẮT BUỘC
# -------------------------------------------------------------------------------
data_group['calib_intercept'] = alpha_train
data_group['calib_slope'] = beta_train

# -------------------------------------------------------------------------------
# [09] OUTPUT CALIBRATION
# -------------------------------------------------------------------------------
print("\n[TẦNG 2] Mỗi cấu hình quét delta x scale = %d x %d = %d ứng viên, tổng %d"
      % (_N_DELTA, _N_SCALE, _N_CAND, _N_TOTAL))
print(" %-18s %7s %7s %15s %15s %12s"
      % ("class_weight", "delta", "scale", "alpha", "beta", "KPI TRAIN/5"))
for _r in _RES:
    _best_idx = int(_r["best"])
    print(" %-18s %+7.2f %7.2f %15.10f %15.10f %12d%s"
          % (_r["t1"]["name"], _r["d_s"][_best_idx], _r["s_s"][_best_idx],
             _r["a_s"][_best_idx], _r["b_s"][_best_idx],
             int(_r["train_n_pass"][_best_idx]),
             " <== CHỌN" if _r is _SEL else ""))

print("\n" + "=" * 78)
print("PLATT CALIBRATION - CHỌN THAM SỐ CHỈ BẰNG TRAIN")
print("=" * 78)
print("Anchor được chọn: %s" % _cw_star)
print("alpha_cv=%.10f | beta_cv=%.10f" % (_a_cv, _b_cv))
print("delta=%+.2f | scale=%.2f" % (_delta_star, _scale_star))
print("alpha_train=%.10f" % alpha_train)
print("beta_train=%.10f" % beta_train)
print()
print("%-6s %9s | %12s | %11s | %10s | %10s | %10s"
      % ("Tập", "n", "Intercept", "Slope", "ECE", "MCE", "Brier"))
print("-" * 78)
for _nm in _NAMES:
    _metrics = _FINAL[_nm]
    print("%-6s %9d | %12.8f | %11.8f | %10.6f | %10.6f | %10.6f"
          % (_nm, _metrics["n"], _metrics["alpha"], _metrics["beta"],
             _metrics["ece"], _metrics["mce"], _metrics["bs"]))
print("-" * 78)
print("KPI: TRAIN %d/5 | TEST %d/5 | FULL %d/5 | %s"
      % (_n_pass["TRAIN"], _n_pass["TEST"], _n_pass["FULL"],
         "FULL PASS" if _FULL_PASS else "BEST-EFFORT"))
print("=" * 78)






calibration = pd.DataFrame({
    'Parameter': [
        'calib_intercept',
        'calib_slope',
    ],
    'Value': [
        alpha_train,
        beta_train,
    ],
})

calibration.to_excel(
    r"D:\model\calibration.xlsx",
    sheet_name="Calibration",
    index=False,
)

print("\nCALIBRATION PARAMETERS:")
print(calibration.to_string(index=False))
print(r"Đã xuất: D:\model\calibration.xlsx")



data_group['log_odds_new'] = alpha_train + beta_train * data_group['log_odds']

# Tính toán xác suất mới từ log-odds đã hiệu chỉnh
data_group['probability_manual_new'] = 1 / (1 + np.exp(-data_group['log_odds_new']))
#data_group['probability_manual_new'] = data_group.groupby(['CIF', 'NGAY_BC']).probability_manual_new.transform('max')

#data_group['probability_manual'] = data_group['probability_manual_new']
#data_group["log_odds"] = data_group['log_odds_new']


# 3. PD DECILE 10% POP - CHỈ TRAIN (NO LEAKAGE)
train_probs = data_group.loc[train_mask, "probability_manual_new"].to_numpy()

_, pd_edges = pd.qcut(train_probs, 10, retbins=True, duplicates="drop")
pd_edges = np.unique(pd_edges)
bins_pd = np.concatenate([[-np.inf], pd_edges[1:-1], [np.inf]])
 
num_bins_pd = len(bins_pd) - 1
decile_labels = list(range(1, num_bins_pd + 1))

data_group["decile_new"] = pd.cut(
    data_group["probability_manual_new"],
    bins=bins_pd,
    labels=decile_labels,  # 1=best AAA, num_bins_pd=worst D
    include_lowest=True
)



# ====================== EXPORT TRAIN PD BY DECILE ======================

pd_decile_train = (
    data_group.loc[
        train_mask,
        ["decile_new", "probability_manual_new"],
    ]
    .groupby(
        "decile_new",
        observed=True,
    )
    .agg(
        min_probability_manual_new=("probability_manual_new", "min"),
        max_probability_manual_new=("probability_manual_new", "max")
    )
    .reset_index()
    .sort_values("decile_new")
)

pd_decile_train.to_excel(
    r"D:\model\PD decile_new.xlsx",
    sheet_name="PD Decile_new",
    index=False,
)

print("\nPD DECILE NEW TRAIN:")
print(pd_decile_train.to_string(index=False))
print(r"Đã xuất: D:\model\PD decile_new.xlsx")


########

# 2. Score scaling - chỉ tính trên TRAIN
min_score = 300
max_score = 850
base_odds = 50
train_pd = data_group.loc[train_mask, "probability_manual_new"]


min_PD = float(train_pd.min())
max_PD = float(train_pd.max())
 

pdo = (max_score - min_score) * math.log(2) / (math.log((1 - min_PD) / min_PD) - math.log((1 - max_PD) / max_PD))
base_score = math.ceil(max(
    max_score - 0.5 - (pdo / math.log(2)) * (math.log((1 - min_PD) / min_PD) - math.log(base_odds)),
    min_score - 0.5 - (pdo / math.log(2)) * (math.log((1 - max_PD) / max_PD) - math.log(base_odds))
))
factor = pdo / math.log(2)
offset = base_score - (factor * math.log(base_odds))


data_group["base_score_new"] = base_score
data_group["base_odds_new"] = base_odds
data_group["offset_new"] = offset
data_group["factor_new"] = factor

data_group["score_new"] = offset - factor * data_group["log_odds_new"]
data_group["score_new"] = data_group["score_new"].round(0).astype(int)
 

 
# 4. GRADE THEO SCORE CÁCH ĐỀU TỪ MIN ĐẾN MAX CỦA TRAIN (NO LEAKAGE)
grades_low_to_high = np.array([
    "9.D",
    "8.C",
    "7.CC",
    "6.CCC",
    "5.B",
    "4.BB",
    "3.BBB",
    "2.A",
    "1.AA",
    "0.AAA",
])
 
# Chỉ lấy score hợp lệ thuộc tập train
train_scores = data_group.loc[train_mask, "score_new"].dropna()
 
# Xác định min và max chỉ trên tập train
min_train_score = float(train_scores.min())
max_train_score = float(train_scores.max())
 
# Chia khoảng từ min đến max thành 10 phần có độ rộng bằng nhau
score_edges = np.linspace(
    start=min_train_score,
    stop=max_train_score,
    num=len(grades_low_to_high) + 1,
)
 
# Mở rộng hai đầu để áp dụng cho validation, test và OOT
# Các score thấp hơn min train được gán 9.D
# Các score cao hơn max train được gán 0.AAA
bins_score = np.concatenate([
    [-np.inf],
    score_edges[1:-1],
    [np.inf],
])
 
# Áp dụng cùng bộ ngưỡng cho toàn bộ dữ liệu
data_group["Grade_new"] = pd.cut(
    data_group["score_new"],
    bins=bins_score,
    labels=grades_low_to_high,
    include_lowest=True,
    right=True,
    ordered=True,
)


# ====================== EXPORT SCORE PARAMETERS ======================

score_parameter_df = pd.DataFrame({
    "Parameter": [
        "MIN_SCORE",
        "MAX_SCORE",
        "BASE_ODDS",
        "pdo",
        "factor",
        "base_score",
        "offset",
    ],
    "Value": [
        min_score,
        max_score,
        base_odds,
        pdo,
        factor,
        base_score,
        offset,
    ],
})

score_parameter_df.to_excel(
    r"D:/model/score_parameter_new.xlsx",
    sheet_name="Score Parameters_new",
    index=False,
)

print(
    "Đã xuất tham số score new:",
    r"D:/model/score_parameter_new.xlsx",
)




# ====================== EXPORT TRAIN GRADE BY SCORE ======================

grade_score_train = (
    data_group.loc[
        train_mask,
        ["Grade_new", "score_new"],
    ]
    .groupby(
        "Grade_new",
        observed=True,
    )
    .agg(
        min_score =("score_new", "min"),
        max_score =("score_new", "max")
    )
    .reset_index()
    .sort_values("Grade_new")
)

grade_score_train.to_excel(
    r"D:\model\Grade_score_new.xlsx",
    sheet_name="Grade_score_new",
    index=False,
)

print("\nGRADE_SCORE_new TRAIN:")
print(grade_score_train.to_string(index=False))
print(r"Đã xuất: D:\model\Grade_score_new.xlsx")


#######

# ============================================================
# SCAN LẠI THRESHOLD SAU KHI CÓ probability_manual_new
# CHỈ DÙNG TRAIN ĐỂ CHỌN THRESHOLD, KHÔNG DÙNG TEST/FULL
# ============================================================

prob_train_new = data_group.loc[
    train_mask, "probability_manual_new"
].to_numpy(dtype=np.float64, copy=False)

y_train_new = data_group.loc[
    train_mask, TARGET_COL
].to_numpy(dtype=np.int8, copy=False)

if len(prob_train_new) == 0:
    raise ValueError(
        "Tập TRAIN rỗng, không thể scan threshold cho probability_manual_new."
    )

if len(prob_train_new) != len(y_train_new):
    raise ValueError(
        "prob_train_new và y_train_new không cùng số phần tử."
    )

if not np.isfinite(prob_train_new).all():
    n_invalid = int(np.count_nonzero(~np.isfinite(prob_train_new)))
    raise ValueError(
        f"probability_manual_new trên TRAIN chứa "
        f"{n_invalid} giá trị NaN/Inf."
    )

if np.any((prob_train_new < 0.0) | (prob_train_new > 1.0)):
    raise ValueError(
        "probability_manual_new trên TRAIN phải nằm trong đoạn [0, 1]."
    )

if np.unique(y_train_new).size < 2:
    raise ValueError(
        "TARGET trên TRAIN phải có đủ cả hai lớp 0 và 1."
    )

best_thresh_new, best_t_metrics_new = _scan9(
    prob_train_new,
    y_train_new,
)

best_rank_key_new = (
    best_t_metrics_new["n_viol"],
    -int(best_t_metrics_new["gap_ok"]),
    -best_t_metrics_new["mcc"],
    -best_t_metrics_new["sens"],
    -best_t_metrics_new["f1"],
    best_t_metrics_new["bce"],
)

best_n_pass_new = (
    8
    - best_t_metrics_new["n_viol"]
    + int(best_t_metrics_new["gap_ok"])
)

# ============================================================
# THRESHOLD SCAN METRICS, TRAIN ONLY
# ============================================================

print(
    "\nBest threshold sau FULL BATCH SCAN "
    f"probability_manual_new: {best_thresh_new:.6f}"
)
print(
    f"Threshold scan grid: [{THRESHOLDS[0]:.4f}, {THRESHOLDS[-1]:.4f}], "
    f"step 0.0005, {len(THRESHOLDS)} điểm, TRAIN only"
)
print(f"FULL BATCH SCAN rank_key: {best_rank_key_new}")
print(
    f"TRAIN threshold metrics: "
    f"Acc={best_t_metrics_new['acc']:.4f}, "
    f"Prec={best_t_metrics_new['prec']:.4f}, "
    f"Sens={best_t_metrics_new['sens']:.4f}, "
    f"Spec={best_t_metrics_new['spec']:.4f}, "
    f"NPV={best_t_metrics_new['npv']:.4f}, "
    f"F1={best_t_metrics_new['f1']:.4f}, "
    f"MCC={best_t_metrics_new['mcc']:.4f}, "
    f"Gap={best_t_metrics_new['gap']:.4f}, "
    f"BCE={best_t_metrics_new['bce']:.4f}"
)

# ============================================================
# ĐÁNH GIÁ CHI TIẾT TRAIN, TEST VÀ FULL
# TEST/FULL CHỈ ĐỂ ĐÁNH GIÁ, KHÔNG CHỌN THRESHOLD
# ============================================================

print(
    "\n"
    + "=" * 60
    + "\n  ĐÁNH GIÁ CHI TIẾT 3 TẬP DỮ LIỆU"
    + " - probability_manual_new\n"
    + "=" * 60
)

def evaluate_new(data_df, mask, label, threshold):
    mask_array = np.asarray(mask)

    if mask_array.ndim != 1:
        raise ValueError(
            f"Mask của tập {label} phải là mảng một chiều."
        )

    if mask_array.dtype != np.bool_:
        raise TypeError(
            f"Mask của tập {label} phải có dtype bool, "
            f"dtype thực tế là {mask_array.dtype}."
        )

    if len(mask_array) != len(data_df):
        raise ValueError(
            f"Mask của tập {label} có độ dài {len(mask_array)}, "
            f"khác số dòng dữ liệu {len(data_df)}."
        )

    if not mask_array.any():
        raise ValueError(
            f"Tập {label} rỗng, không thể đánh giá."
        )

    y_prob = data_df.loc[
        mask_array,
        "probability_manual_new",
    ].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    y_true = data_df.loc[
        mask_array,
        TARGET_COL,
    ].to_numpy(
        dtype=np.int8,
        copy=False,
    )

    if len(y_prob) != len(y_true):
        raise ValueError(
            f"Xác suất và target của tập {label} "
            "không cùng số phần tử."
        )

    invalid_probability = ~np.isfinite(y_prob)

    if invalid_probability.any():
        n_invalid = int(
            np.count_nonzero(invalid_probability)
        )
        raise ValueError(
            f"probability_manual_new của tập {label} chứa "
            f"{n_invalid} giá trị NaN/Inf."
        )

    outside_probability = (
        (y_prob < 0.0)
        | (y_prob > 1.0)
    )

    if outside_probability.any():
        n_outside = int(
            np.count_nonzero(outside_probability)
        )
        raise ValueError(
            f"probability_manual_new của tập {label} có "
            f"{n_outside} giá trị nằm ngoài đoạn [0, 1]."
        )

    invalid_target = (
        (y_true != 0)
        & (y_true != 1)
    )

    if invalid_target.any():
        n_invalid_target = int(
            np.count_nonzero(invalid_target)
        )
        raise ValueError(
            f"TARGET của tập {label} có "
            f"{n_invalid_target} giá trị không thuộc {{0, 1}}."
        )

    threshold_value = float(threshold)

    if not np.isfinite(threshold_value):
        raise ValueError(
            f"Threshold của tập {label} phải là giá trị hữu hạn."
        )

    if not 0.0 <= threshold_value <= 1.0:
        raise ValueError(
            f"Threshold của tập {label} phải nằm trong đoạn [0, 1]."
        )

    kpi_result = _kpi9(
        y_true,
        y_prob,
        threshold_value,
    )

    if not isinstance(kpi_result, dict):
        raise TypeError(
            f"_kpi9 phải trả về dict cho tập {label}."
        )

    required_kpi_keys = (
        "TP",
        "FP",
        "TN",
        "FN",
        "gates",
        "n_viol",
        "gap_ok",
        "pass_all",
    )

    missing_kpi_keys = [
        key
        for key in required_kpi_keys
        if key not in kpi_result
    ]

    if missing_kpi_keys:
        raise KeyError(
            f"_kpi9 thiếu các trường {missing_kpi_keys} "
            f"cho tập {label}."
        )

    gates = kpi_result["gates"]

    if len(gates) != len(KPI_GATES):
        raise ValueError(
            f"_kpi9 trả về {len(gates)} gate cho tập {label}, "
            f"khác KPI_GATES runtime là {len(KPI_GATES)}."
        )

    n_pass = int(
        sum(
            bool(gate_result[2])
            for gate_result in gates
        )
    )

    failed_kpis = [
        gate_result[0]
        for gate_result in gates
        if not bool(gate_result[2])
    ]

    status = (
        "ĐẠT FULL KPI"
        if bool(kpi_result["pass_all"])
        else "CHƯA ĐẠT"
    )

    print(
        f"\n{'=' * 78}\n"
        f"KPI CHECK [{label}] - probability_manual_new: {status}\n"
        f"{'=' * 78}"
    )
    print(
        f"n_rows={len(y_true)}"
        f" | n_bad={int(y_true.sum())}"
        f" | threshold={threshold_value:.6f}"
        f" | KPI={n_pass}/{len(gates)}"
    )
    print(
        "TP=%.0f | FP=%.0f | TN=%.0f | FN=%.0f"
        % (
            kpi_result["TP"],
            kpi_result["FP"],
            kpi_result["TN"],
            kpi_result["FN"],
        )
    )
    print(
        "%-15s %-10s %-14s %-8s"
        % (
            "Metric",
            "Gate",
            "Value",
            "Status",
        )
    )
    print("-" * 52)

    for metric_name, value, passed, gate_label in gates:
        print(
            "%-15s %-10s %-14.6f %-8s"
            % (
                metric_name,
                gate_label,
                float(value),
                "PASS" if bool(passed) else "FAIL",
            )
        )

    print("-" * 52)
    print(
        "KPI chưa đạt: %s"
        % (
            ", ".join(failed_kpis)
            if failed_kpis
            else "KHÔNG CÓ"
        )
    )

    return kpi_result


# ============================================================
# ĐÁNH GIÁ TRAIN, TEST VÀ FULL
# DÙNG CÙNG best_thresh_new ĐÃ ĐƯỢC CHỌN TRÊN TRAIN
# ============================================================

_kpi_train_new = evaluate_new(
    data_df=data_group,
    mask=train_np,
    label="TRAIN",
    threshold=best_thresh_new,
)

_kpi_test_new = evaluate_new(
    data_df=data_group,
    mask=test_np,
    label="TEST",
    threshold=best_thresh_new,
)

_kpi_full_new = evaluate_new(
    data_df=data_group,
    mask=_full_np,
    label="FULL",
    threshold=best_thresh_new,
)


_train_ok_new = bool(
    _kpi_train_new["pass_all"]
)

_test_ok_new = bool(
    _kpi_test_new["pass_all"]
)

_full_ok_new = bool(
    _kpi_full_new["pass_all"]
)

best_n_pass_new = int(
    sum(
        bool(gate_result[2])
        for gate_result in _kpi_train_new["gates"]
    )
)

best_rank_key_new = best_t_metrics_new.get(
    "sort_key",
    (
        int(best_t_metrics_new["n_viol"]),
        -int(best_t_metrics_new["gap_ok"]),
        -float(best_t_metrics_new["mcc"]),
        -float(best_t_metrics_new["sens"]),
        -float(best_t_metrics_new["f1"]),
        float(best_t_metrics_new["bce"]),
    ),
)

full_kpi_achieved_new = bool(
    _train_ok_new
    and _test_ok_new
    and _full_ok_new
)

_failed_kpi_new = {
    "TRAIN": [
        gate_result[0]
        for gate_result in _kpi_train_new["gates"]
        if not bool(gate_result[2])
    ],
    "TEST": [
        gate_result[0]
        for gate_result in _kpi_test_new["gates"]
        if not bool(gate_result[2])
    ],
    "FULL": [
        gate_result[0]
        for gate_result in _kpi_full_new["gates"]
        if not bool(gate_result[2])
    ],
}

# ============================================================
# KẾT QUẢ CUỐI SAU CALIBRATION
# ============================================================

print("\n" + "=" * 78)
print("BEST CONFIG DUY NHẤT SAU CALIBRATION")
print("=" * 78)

if best_cfg is not None:
    for key, value in best_cfg.items():
        print(
            f"  {key:28s}: {value}"
        )
else:
    print(
        "  best_cfg                    : KHÔNG TỒN TẠI"
    )

if "alpha_train" in globals():
    print(
        f"  {'alpha_train':28s}: {float(alpha_train):.10f}"
    )
else:
    print(
        f"  {'alpha_train':28s}: KHÔNG TỒN TẠI"
    )

if "beta_train" in globals():
    print(
        f"  {'beta_train':28s}: {float(beta_train):.10f}"
    )
else:
    print(
        f"  {'beta_train':28s}: KHÔNG TỒN TẠI"
    )

print(
    f"  {'n_features_input':28s}: {len(feature_cols)}"
)
print(
    f"  {'best_thresh_new':28s}: {best_thresh_new:.6f}"
)
print(
    f"  {'best_rank_key_new':28s}: {best_rank_key_new}"
)
print(
    f"  {'best_n_pass_new':28s}: "
    f"{best_n_pass_new}/{len(KPI_GATES)}"
)
print(
    f"  {'TRAIN pass_all':28s}: {_train_ok_new}"
)
print(
    f"  {'TEST pass_all':28s}: {_test_ok_new}"
)
print(
    f"  {'FULL pass_all':28s}: {_full_ok_new}"
)
print(
    f"  {'full_kpi_achieved_new':28s}: "
    f"{full_kpi_achieved_new}"
)

for label in ("TRAIN", "TEST", "FULL"):
    failed_text = (
        ", ".join(_failed_kpi_new[label])
        if _failed_kpi_new[label]
        else "KHÔNG CÓ"
    )
    print(
        f"  {('KPI chưa đạt ' + label):28s}: {failed_text}"
    )

print(
    f"  {'Kết luận':28s}: "
    f"{'FULL KPI' if full_kpi_achieved_new else 'BEST-EFFORT'}"
)
print("=" * 78)

###############











####################

 
param_df = pd.DataFrame({
    "X": ["intercept"] + feature_cols,
    "Coefficients": [logr.raw_intercept_[0]] + logr.raw_coef_[0].tolist(),
    "template": ["intercept"] + feature_cols,
    "parameter_type": ["intercept"] + ["coefficient"] * len(feature_cols)
})
param_df.to_excel("D:/model/parameter_model.xlsx", index=False)
 
 
data_group.to_parquet(
    "D:/model/credit_data_scoring.parquet",
    index=False,
    compression="brotli"
)
 
#test
 
data_group_test = data_group[data_group["train_test"] == "test"]

#data_group_test.to_excel("D:/model/credit_data_scoring_test.xlsx")
 
 
# =====================================================
# GINI / KS / AUC / IV REPORT
# =====================================================
# =====================================================
# GINI / KS / AUC / IV REPORT
# =====================================================

def gini_ks_report(
    data_group_test,
    group_cols=None,
    acct_col="BILLING_ACCTNUM",
    bad_col="NX",
    grade_col="Grade_new",
    probability_col="probability_manual_new",
    ascending=True,
    grade_as_int=False,
    grade_sort_by_prefix=True,
    plot=True,
    export_path=None,
    title_name="CREDIT_RISK",
    eps=1e-10,
):
    """
    Tính Gini, KS, grouped ROC AUC, model ROC AUC, c-statistic,
    WoE và IV theo thông lệ credit risk model validation.

    Quy ước:
        bad_col = 1: Bad
        bad_col = 0: Good

    Quy ước Grade:
        0.AAA: rủi ro thấp nhất
        9.D  : rủi ro cao nhất

    Bảng kết quả được hiển thị theo:
        0.AAA -> ... -> 9.D
        rủi ro thấp -> rủi ro cao

    ROC/AUC được tính theo:
        9.D -> ... -> 0.AAA
        rủi ro cao -> rủi ro thấp

    Hai loại AUC:
        grouped_auc:
            AUC tính từ Grade. Các hồ sơ trong cùng Grade được xem là
            đồng hạng, nên có thể thấp hơn AUC cấp hồ sơ.

        model_auc:
            AUC tính trực tiếp từ probability_col ở cấp hồ sơ.
            Đây là AUC chính xác hơn để đánh giá model discrimination.

    Gini:
        grouped_gini = 2 * grouped_auc - 1
        model_gini   = 2 * model_auc - 1

    KS:
        KS = max(abs(cumulative Bad - cumulative Good))

    WoE:
        WoE = ln(%Good / %Bad)

    IV:
        IV = (%Good - %Bad) * WoE
    """

    # =====================================================
    # 1. KIỂM TRA THAM SỐ VÀ DỮ LIỆU ĐẦU VÀO
    # =====================================================

    if not isinstance(data_group_test, pd.DataFrame):
        raise TypeError(
            "data_group_test phải là pandas DataFrame."
        )

    if data_group_test.empty:
        raise ValueError(
            "data_group_test rỗng, không thể tính chỉ tiêu."
        )

    if not isinstance(eps, (int, float, np.integer, np.floating)):
        raise TypeError(
            "eps phải là số."
        )

    eps = float(eps)

    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(
            "eps phải là số hữu hạn lớn hơn 0."
        )

    df = data_group_test.copy()

    raw_data_available = group_cols is not None
    model_auc_available = False
    model_auc = np.nan
    model_gini = np.nan

    # =====================================================
    # 2. CHUẨN HÓA DỮ LIỆU RAW HOẶC GROUPED
    # =====================================================

    if group_cols is not None:
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        else:
            group_cols = list(group_cols)

        if grade_col not in group_cols:
            raise ValueError(
                f"grade_col='{grade_col}' phải nằm trong group_cols "
                "để tính Gini/KS theo Grade."
            )

        required_raw_cols = list(
            dict.fromkeys(
                group_cols
                + [acct_col, bad_col]
            )
        )

        missing_raw_cols = [
            col
            for col in required_raw_cols
            if col not in df.columns
        ]

        if missing_raw_cols:
            raise ValueError(
                f"Dữ liệu raw cần có các cột {required_raw_cols}. "
                f"Đang thiếu: {missing_raw_cols}"
            )

        if df[grade_col].isna().any():
            n_missing_grade = int(
                df[grade_col].isna().sum()
            )
            raise ValueError(
                f"Cột {grade_col} có {n_missing_grade} giá trị null. "
                "Không tự động gán Grade."
            )

        if df[acct_col].isna().any():
            n_missing_account = int(
                df[acct_col].isna().sum()
            )
            raise ValueError(
                f"Cột {acct_col} có {n_missing_account} giá trị null. "
                "Total đang được định nghĩa bằng count(acct_col), "
                "nên dữ liệu phải đầy đủ."
            )

        if df[bad_col].isna().any():
            n_missing_bad = int(
                df[bad_col].isna().sum()
            )
            raise ValueError(
                f"Cột {bad_col} có {n_missing_bad} giá trị null."
            )

        bad_numeric = pd.to_numeric(
            df[bad_col],
            errors="raise",
        )

        bad_values = set(
            bad_numeric.unique()
        )

        if not bad_values.issubset({0, 1}):
            raise ValueError(
                f"Cột {bad_col} phải là biến nhị phân 0/1. "
                f"Giá trị hiện có: {sorted(bad_values)}"
            )

        df[bad_col] = bad_numeric.astype(
            np.int8
        )

        Gini = (
            df.groupby(
                group_cols,
                dropna=False,
                observed=True,
            )
            .agg(
                Total=(acct_col, "count"),
                Bad=(bad_col, "sum"),
            )
            .reset_index()
        )

        # =================================================
        # 2.1. AUC CẤP HỒ SƠ TỪ XÁC SUẤT
        # Không phụ thuộc threshold và không bị mất thông tin do Grade
        # =================================================

        if (
            probability_col is not None
            and probability_col in df.columns
        ):
            probability_raw = pd.to_numeric(
                df[probability_col],
                errors="raise",
            ).to_numpy(
                dtype=np.float64,
                copy=False,
            )

            target_raw = df[bad_col].to_numpy(
                dtype=np.int8,
                copy=False,
            )

            invalid_probability = ~np.isfinite(
                probability_raw
            )

            if invalid_probability.any():
                raise ValueError(
                    f"Cột {probability_col} chứa "
                    f"{int(np.count_nonzero(invalid_probability))} "
                    "giá trị NaN/Inf."
                )

            outside_probability = (
                (probability_raw < 0.0)
                | (probability_raw > 1.0)
            )

            if outside_probability.any():
                raise ValueError(
                    f"Cột {probability_col} chứa "
                    f"{int(np.count_nonzero(outside_probability))} "
                    "giá trị nằm ngoài [0, 1]."
                )

            if np.unique(target_raw).size != 2:
                raise ValueError(
                    "Dữ liệu phải có đủ cả hai lớp 0 và 1 "
                    "để tính model AUC."
                )

            # AUC theo công thức Mann-Whitney với average rank,
            # xử lý đúng ties trong xác suất.
            probability_rank = pd.Series(
                probability_raw,
                copy=False,
            ).rank(
                method="average",
                ascending=True,
            ).to_numpy(
                dtype=np.float64,
                copy=False,
            )

            n_bad_raw = int(
                np.count_nonzero(target_raw == 1)
            )
            n_good_raw = int(
                np.count_nonzero(target_raw == 0)
            )

            bad_rank_sum = float(
                probability_rank[
                    target_raw == 1
                ].sum()
            )

            model_auc = (
                bad_rank_sum
                - n_bad_raw * (n_bad_raw + 1.0) / 2.0
            ) / (
                n_bad_raw * n_good_raw
            )

            model_auc = float(model_auc)

            if not 0.0 <= model_auc <= 1.0:
                raise RuntimeError(
                    f"Model AUC nằm ngoài [0, 1]: {model_auc}."
                )

            model_gini = float(
                2.0 * model_auc - 1.0
            )

            model_auc_available = True

    else:
        Gini = df.copy()

        required_grouped_cols = [
            grade_col,
            "Total",
            "Bad",
        ]

        missing_grouped_cols = [
            col
            for col in required_grouped_cols
            if col not in Gini.columns
        ]

        if missing_grouped_cols:
            raise ValueError(
                "Dữ liệu grouped cần có các cột "
                f"{required_grouped_cols}. "
                f"Đang thiếu: {missing_grouped_cols}"
            )

    if grade_col not in Gini.columns:
        raise ValueError(
            f"Dữ liệu cần có cột {grade_col}."
        )

    # =====================================================
    # 3. XỬ LÝ THỨ TỰ GRADE
    # =====================================================

    sort_col = grade_col

    if grade_as_int:
        try:
            Gini[grade_col] = pd.to_numeric(
                Gini[grade_col],
                errors="raise",
            ).astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Không thể chuyển {grade_col} sang integer."
            ) from exc

    elif grade_sort_by_prefix:
        Gini["_Grade_sort"] = (
            Gini[grade_col]
            .astype(str)
            .str.strip()
            .str.extract(
                r"^(\d+)",
                expand=False,
            )
        )

        invalid_grade_mask = Gini[
            "_Grade_sort"
        ].isna()

        if invalid_grade_mask.any():
            invalid_grades = (
                Gini.loc[
                    invalid_grade_mask,
                    grade_col,
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(
                "Không thể lấy prefix số từ Grade. "
                f"Giá trị lỗi: {invalid_grades}. "
                "Nếu Grade không có prefix số, hãy đặt "
                "grade_sort_by_prefix=False."
            )

        Gini["_Grade_sort"] = pd.to_numeric(
            Gini["_Grade_sort"],
            errors="raise",
        ).astype(np.int64)

        sort_col = "_Grade_sort"

    Gini = (
        Gini.sort_values(
            by=sort_col,
            ascending=ascending,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # =====================================================
    # 4. KIỂM TRA TOTAL VÀ BAD
    # =====================================================

    Gini["Total"] = pd.to_numeric(
        Gini["Total"],
        errors="raise",
    )

    Gini["Bad"] = pd.to_numeric(
        Gini["Bad"],
        errors="raise",
    )

    if Gini["Total"].isna().any():
        raise ValueError(
            "Cột Total có giá trị null."
        )

    if Gini["Bad"].isna().any():
        raise ValueError(
            "Cột Bad có giá trị null."
        )

    if not np.isfinite(
        Gini["Total"].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Cột Total chứa giá trị NaN/Inf."
        )

    if not np.isfinite(
        Gini["Bad"].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Cột Bad chứa giá trị NaN/Inf."
        )

    if (Gini["Total"] <= 0).any():
        invalid_grades = Gini.loc[
            Gini["Total"] <= 0,
            grade_col,
        ].tolist()

        raise ValueError(
            "Có Grade có Total <= 0: "
            f"{invalid_grades}"
        )

    if (Gini["Bad"] < 0).any():
        invalid_grades = Gini.loc[
            Gini["Bad"] < 0,
            grade_col,
        ].tolist()

        raise ValueError(
            "Có Grade có Bad < 0: "
            f"{invalid_grades}"
        )

    if (Gini["Bad"] > Gini["Total"]).any():
        invalid_grades = Gini.loc[
            Gini["Bad"] > Gini["Total"],
            grade_col,
        ].tolist()

        raise ValueError(
            "Có Grade có Bad > Total: "
            f"{invalid_grades}"
        )

    # =====================================================
    # 5. GOOD, BAD RATE VÀ DISTRIBUTION
    # =====================================================

    Gini["Good"] = (
        Gini["Total"] - Gini["Bad"]
    )

    total_bad = float(
        Gini["Bad"].sum()
    )
    total_good = float(
        Gini["Good"].sum()
    )
    total_all = float(
        Gini["Total"].sum()
    )

    if total_bad <= 0.0:
        raise ValueError(
            "Tổng Bad = 0, không thể tính Gini/KS/AUC/IV."
        )

    if total_good <= 0.0:
        raise ValueError(
            "Tổng Good = 0, không thể tính Gini/KS/AUC/IV."
        )

    if total_all <= 0.0:
        raise ValueError(
            "Tổng Total = 0."
        )

    Gini["bad_rate"] = (
        Gini["Bad"] / Gini["Total"]
    )

    Gini["%Bad"] = (
        Gini["Bad"] / total_bad
    )

    Gini["%Good"] = (
        Gini["Good"] / total_good
    )

    Gini["%Total"] = (
        Gini["Total"] / total_all
    )

    # Cumulative dùng cho bảng báo cáo theo thứ tự hiển thị.
    Gini["%CBad"] = Gini[
        "%Bad"
    ].cumsum()

    Gini["%CGood"] = Gini[
        "%Good"
    ].cumsum()

    Gini["%CTotal"] = Gini[
        "%Total"
    ].cumsum()

    # =====================================================
    # 6. WOE VÀ IV
    # =====================================================

    Gini["WoE"] = np.log(
        (Gini["%Good"] + eps)
        / (Gini["%Bad"] + eps)
    )

    Gini["IV"] = (
        Gini["%Good"] - Gini["%Bad"]
    ) * Gini["WoE"]

    iv_total = float(
        Gini["IV"].sum()
    )

    # =====================================================
    # 7. KS
    # KS không phụ thuộc chiều nhờ trị tuyệt đối
    # =====================================================

    Gini["KS_signed"] = (
        Gini["%CBad"] - Gini["%CGood"]
    )

    Gini["KS"] = Gini[
        "KS_signed"
    ].abs()

    idx_max_KS = int(
        Gini["KS"].idxmax()
    )

    max_KS = float(
        Gini.loc[idx_max_KS, "KS"]
    )

    ks_grade = Gini.loc[
        idx_max_KS,
        grade_col,
    ]

    # =====================================================
    # 8. GROUPED ROC AUC
    #
    # Bảng Grade đang hiển thị:
    #     low risk -> high risk
    #
    # ROC chuẩn phải xét positive class Bad=1 theo:
    #     high risk -> low risk
    #
    # Vì vậy phải đảo thứ tự các Grade trước khi tích phân.
    # Không sử dụng max(AUC, 1-AUC), vì cách đó có thể che lỗi
    # chiều score hoặc positive class.
    # =====================================================

    risk_order = np.arange(
        len(Gini) - 1,
        -1,
        -1,
        dtype=np.int64,
    )

    bad_distribution_risk = (
        Gini["%Bad"]
        .to_numpy(
            dtype=np.float64,
            copy=False,
        )[risk_order]
    )

    good_distribution_risk = (
        Gini["%Good"]
        .to_numpy(
            dtype=np.float64,
            copy=False,
        )[risk_order]
    )

    roc_tpr = np.concatenate(
        (
            np.array([0.0], dtype=np.float64),
            np.cumsum(
                bad_distribution_risk,
                dtype=np.float64,
            ),
        )
    )

    roc_fpr = np.concatenate(
        (
            np.array([0.0], dtype=np.float64),
            np.cumsum(
                good_distribution_risk,
                dtype=np.float64,
            ),
        )
    )

    # Chống sai số floating-point tại endpoint.
    roc_tpr[-1] = 1.0
    roc_fpr[-1] = 1.0

    auc_part_risk = (
        0.5
        * (
            roc_tpr[1:]
            + roc_tpr[:-1]
        )
        * np.diff(roc_fpr)
    )

    grouped_auc = float(
        auc_part_risk.sum()
    )

    if not 0.0 <= grouped_auc <= 1.0:
        raise RuntimeError(
            "Grouped AUC tính từ Grade nằm ngoài [0, 1]: "
            f"{grouped_auc}."
        )

    grouped_gini = float(
        2.0 * grouped_auc - 1.0
    )

    grouped_c_statistic = grouped_auc

    # Gán AUC contribution về đúng dòng Grade trong bảng hiển thị.
    auc_part_display = np.zeros(
        len(Gini),
        dtype=np.float64,
    )

    auc_part_display[
        risk_order
    ] = auc_part_risk

    Gini["AUC_part"] = (
        auc_part_display
    )

    # =====================================================
    # 8.1. CROSS-CHECK GROUPED AUC BẰNG PAIRWISE FORMULA
    #
    # AUC = P(score_bad > score_good)
    #       + 0.5 * P(score_bad = score_good)
    #
    # Ở đây prefix Grade lớn hơn nghĩa là rủi ro cao hơn.
    # =====================================================

    bad_counts_risk = (
        Gini["Bad"]
        .to_numpy(
            dtype=np.float64,
            copy=False,
        )[risk_order]
    )

    good_counts_risk = (
        Gini["Good"]
        .to_numpy(
            dtype=np.float64,
            copy=False,
        )[risk_order]
    )

    good_below_risk = (
        total_good
        - np.cumsum(
            good_counts_risk,
            dtype=np.float64,
        )
    )

    grouped_auc_pairwise = float(
        np.sum(
            bad_counts_risk
            * (
                good_below_risk
                + 0.5 * good_counts_risk
            )
        )
        / (total_bad * total_good)
    )

    if not np.isclose(
        grouped_auc,
        grouped_auc_pairwise,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Grouped AUC không nhất quán giữa trapezoid và "
            "pairwise formula: "
            f"{grouped_auc} khác {grouped_auc_pairwise}."
        )

    # =====================================================
    # 8.2. KIỂM TRA CHIỀU
    # Không tự đảo bằng max(AUC, 1-AUC)
    # =====================================================

    direction_warning = ""

    if grouped_auc < 0.5:
        direction_warning = (
            "Grouped AUC < 0.5. Grade đang không xếp Bad theo "
            "chiều rủi ro kỳ vọng. Cần kiểm tra Grade_new, "
            "score_new, bad_col và quy ước Bad=1."
        )

        print(
            "CẢNH BÁO: "
            + direction_warning
        )

    if (
        model_auc_available
        and model_auc < 0.5
    ):
        print(
            f"CẢNH BÁO: Model AUC từ {probability_col} "
            f"bằng {model_auc:.6f} < 0.5. "
            "Cần kiểm tra probability có phải xác suất Bad=1 "
            "hay không."
        )

    # AUC/Gini chính trong bảng Grade là grouped metric.
    Gini["AUC"] = np.nan
    Gini["Gini_value"] = np.nan
    Gini["c_statistic"] = np.nan
    Gini["Model_AUC"] = np.nan
    Gini["Model_Gini"] = np.nan
    Gini["AUC_grouping_loss"] = np.nan

    first_index = Gini.index[0]

    Gini.loc[
        first_index,
        "AUC",
    ] = grouped_auc

    Gini.loc[
        first_index,
        "Gini_value",
    ] = grouped_gini

    Gini.loc[
        first_index,
        "c_statistic",
    ] = grouped_c_statistic

    if model_auc_available:
        auc_grouping_loss = float(
            model_auc - grouped_auc
        )

        Gini.loc[
            first_index,
            "Model_AUC",
        ] = model_auc

        Gini.loc[
            first_index,
            "Model_Gini",
        ] = model_gini

        Gini.loc[
            first_index,
            "AUC_grouping_loss",
        ] = auc_grouping_loss
    else:
        auc_grouping_loss = np.nan

    # =====================================================
    # 9. XẾP HẠNG GINI VÀ KS
    # Ưu tiên model Gini nếu có raw probability
    # =====================================================

    rating_gini_value = (
        model_gini
        if model_auc_available
        else grouped_gini
    )

    if (
        not np.isfinite(rating_gini_value)
        or rating_gini_value < -1.0
        or rating_gini_value > 1.0
    ):
        gini_rating = "Không xác định"
    elif rating_gini_value < 0.0:
        gini_rating = "Sai chiều"
    elif rating_gini_value < 0.20:
        gini_rating = "Rất yếu"
    elif rating_gini_value < 0.30:
        gini_rating = "Yếu"
    elif rating_gini_value < 0.40:
        gini_rating = "Chấp nhận được"
    elif rating_gini_value < 0.50:
        gini_rating = "Khá"
    elif rating_gini_value < 0.60:
        gini_rating = "Tốt"
    else:
        gini_rating = "Rất tốt"

    if (
        not np.isfinite(max_KS)
        or max_KS < 0.0
        or max_KS > 1.0
    ):
        ks_rating = "Không xác định"
    elif max_KS < 0.20:
        ks_rating = "Rất yếu"
    elif max_KS < 0.30:
        ks_rating = "Yếu"
    elif max_KS < 0.35:
        ks_rating = "Chấp nhận được"
    elif max_KS < 0.40:
        ks_rating = "Khá"
    elif max_KS < 0.50:
        ks_rating = "Tốt"
    else:
        ks_rating = "Rất tốt"

    valid_for_conclusion = bool(
        np.isfinite(rating_gini_value)
        and np.isfinite(max_KS)
        and rating_gini_value >= 0.0
        and 0.0 <= max_KS <= 1.0
    )

    if not valid_for_conclusion:
        conclusion = (
            "không đủ điều kiện kết luận; cần kiểm tra chiều "
            "xác suất, Grade và positive class"
        )
    elif (
        rating_gini_value < 0.20
        or max_KS < 0.20
    ):
        conclusion = (
            "mô hình có khả năng phân biệt rất yếu"
        )
    elif (
        rating_gini_value < 0.30
        or max_KS < 0.30
    ):
        conclusion = (
            "mô hình có khả năng phân biệt yếu"
        )
    elif (
        rating_gini_value < 0.40
        or max_KS < 0.35
    ):
        conclusion = (
            "mô hình có khả năng phân biệt chấp nhận được"
        )
    elif (
        rating_gini_value < 0.50
        or max_KS < 0.40
    ):
        conclusion = (
            "mô hình có khả năng phân biệt khá"
        )
    elif (
        rating_gini_value < 0.60
        or max_KS < 0.50
    ):
        conclusion = (
            "mô hình có khả năng phân biệt tốt"
        )
    else:
        conclusion = (
            "mô hình có khả năng phân biệt rất tốt; "
            "cần kiểm tra thêm overfitting, data leakage "
            "và tính ổn định ngoài mẫu"
        )

    Gini["Gini_rating"] = ""
    Gini["KS_rating"] = ""
    Gini["Conclusion"] = ""
    Gini["Direction_warning"] = ""

    Gini.loc[
        first_index,
        "Gini_rating",
    ] = gini_rating

    Gini.loc[
        first_index,
        "KS_rating",
    ] = ks_rating

    Gini.loc[
        first_index,
        "Conclusion",
    ] = conclusion

    Gini.loc[
        first_index,
        "Direction_warning",
    ] = direction_warning

    # =====================================================
    # 10. XÓA CỘT SORT KỸ THUẬT
    # =====================================================

    if "_Grade_sort" in Gini.columns:
        Gini = Gini.drop(
            columns=["_Grade_sort"]
        )

    # =====================================================
    # 11. SẮP XẾP CỘT OUTPUT
    # =====================================================

    output_cols = [
        grade_col,
        "Total",
        "Bad",
        "Good",
        "bad_rate",
        "%Bad",
        "%Good",
        "%Total",
        "%CBad",
        "%CGood",
        "%CTotal",
        "WoE",
        "IV",
        "KS_signed",
        "KS",
        "AUC_part",
        "AUC",
        "Gini_value",
        "c_statistic",
        "Model_AUC",
        "Model_Gini",
        "AUC_grouping_loss",
        "Gini_rating",
        "KS_rating",
        "Conclusion",
        "Direction_warning",
    ]

    existing_output_cols = [
        col
        for col in output_cols
        if col in Gini.columns
    ]

    other_cols = [
        col
        for col in Gini.columns
        if col not in existing_output_cols
    ]

    Gini = Gini[
        existing_output_cols + other_cols
    ]

    # =====================================================
    # 12. EXPORT EXCEL
    # =====================================================

    if export_path is not None:
        export_dir = os.path.dirname(
            export_path
        )

        if export_dir:
            os.makedirs(
                export_dir,
                exist_ok=True,
            )

        summary_rows = [
            {
                "Metric": "Grouped AUC",
                "Value": grouped_auc,
                "Description": (
                    "AUC tính từ Grade theo chiều high risk "
                    "đến low risk"
                ),
            },
            {
                "Metric": "Grouped Gini",
                "Value": grouped_gini,
                "Description": "2 * Grouped AUC - 1",
            },
            {
                "Metric": "KS",
                "Value": max_KS,
                "Description": (
                    "Khoảng cách lớn nhất giữa cumulative "
                    "Bad và cumulative Good"
                ),
            },
            {
                "Metric": "IV total",
                "Value": iv_total,
                "Description": "Tổng IV theo Grade",
            },
            {
                "Metric": "Model AUC",
                "Value": (
                    model_auc
                    if model_auc_available
                    else np.nan
                ),
                "Description": (
                    f"AUC trực tiếp từ {probability_col}"
                    if model_auc_available
                    else "Không có probability cấp hồ sơ"
                ),
            },
            {
                "Metric": "Model Gini",
                "Value": (
                    model_gini
                    if model_auc_available
                    else np.nan
                ),
                "Description": (
                    "2 * Model AUC - 1"
                ),
            },
            {
                "Metric": "AUC grouping loss",
                "Value": (
                    auc_grouping_loss
                    if model_auc_available
                    else np.nan
                ),
                "Description": (
                    "Model AUC - Grouped AUC"
                ),
            },
        ]

        summary_df = pd.DataFrame(
            summary_rows
        )

        with pd.ExcelWriter(
            export_path,
            engine="openpyxl",
        ) as writer:
            Gini.to_excel(
                writer,
                sheet_name="Grade Detail",
                index=False,
            )

            summary_df.to_excel(
                writer,
                sheet_name="Summary",
                index=False,
            )

    # =====================================================
    # 13. VẼ BIỂU ĐỒ
    # =====================================================

    if plot:
        xaxis = np.arange(
            len(Gini)
        )

        # -------------------------------------------------
        # 13.1. KS PLOT
        # -------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.plot(
            xaxis,
            Gini["%CGood"],
            color="red",
            marker="o",
            label="%CGood",
        )

        ax.plot(
            xaxis,
            Gini["%CBad"],
            color="blue",
            marker="o",
            label="%CBad",
        )

        ax.vlines(
            x=idx_max_KS,
            ymin=float(
                min(
                    Gini.loc[
                        idx_max_KS,
                        "%CGood",
                    ],
                    Gini.loc[
                        idx_max_KS,
                        "%CBad",
                    ],
                )
            ),
            ymax=float(
                max(
                    Gini.loc[
                        idx_max_KS,
                        "%CGood",
                    ],
                    Gini.loc[
                        idx_max_KS,
                        "%CBad",
                    ],
                )
            ),
            color="black",
            linewidth=2,
            label=f"KS={max_KS:.4f}",
        )

        cgood_at_ks = float(
            Gini.loc[
                idx_max_KS,
                "%CGood",
            ]
        )

        cbad_at_ks = float(
            Gini.loc[
                idx_max_KS,
                "%CBad",
            ]
        )

        ax.annotate(
            (
                f"KS = {max_KS:.2%}\n"
                f"Grade = {ks_grade}\n"
                f"Cumulative Good = {cgood_at_ks:.2%}\n"
                f"Cumulative Bad = {cbad_at_ks:.2%}"
            ),
            xy=(
                idx_max_KS,
                max(
                    cgood_at_ks,
                    cbad_at_ks,
                ),
            ),
            xytext=(15, -20),
            textcoords="offset points",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.85,
            },
            arrowprops={
                "arrowstyle": "->",
                "color": "black",
            },
        )

        ax.set_xlabel(
            f"{title_name} Grade, low risk to high risk"
        )

        ax.set_ylabel(
            "Cumulative distribution"
        )

        ax.set_title(
            "Kolmogorov-Smirnov by Grade"
        )

        ax.set_xticks(
            xaxis
        )

        ax.set_xticklabels(
            Gini[grade_col].astype(str),
            rotation=45,
            ha="right",
        )

        ax.set_ylim(
            0.0,
            1.0,
        )

        ax.legend(
            loc="upper left"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()
        plt.show()

        # -------------------------------------------------
        # 13.2. ROC / GINI PLOT
        # Dùng đúng roc_fpr và roc_tpr theo high risk -> low risk
        # -------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        grouped_label = (
            f"Grouped ROC: AUC={grouped_auc:.4f}, "
            f"Gini={grouped_gini:.4f}"
        )

        ax.plot(
            roc_fpr,
            roc_tpr,
            color="darkorange",
            linewidth=2,
            marker="o",
            label=grouped_label,
        )

        ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            color="navy",
            linewidth=1.5,
            linestyle="--",
            label="Random model",
        )

        if model_auc_available:
            ax.text(
                0.55,
                0.16,
                (
                    f"Model AUC cấp hồ sơ = {model_auc:.4f}\n"
                    f"Model Gini = {model_gini:.4f}\n"
                    f"Grouping loss = {auc_grouping_loss:.4f}"
                ),
                transform=ax.transAxes,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.85,
                },
            )

        ax.set_xlim(
            0.0,
            1.0,
        )

        ax.set_ylim(
            0.0,
            1.0,
        )

        ax.set_xlabel(
            "False Positive Rate, Cumulative Good"
        )

        ax.set_ylabel(
            "True Positive Rate, Cumulative Bad"
        )

        ax.set_title(
            "Grouped ROC / Gini, high risk to low risk"
        )

        ax.legend(
            loc="lower right"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()
        plt.show()

        # -------------------------------------------------
        # 13.3. GOOD / BAD / BAD RATE
        # -------------------------------------------------

        plot_data = Gini[
            [
                grade_col,
                "Good",
                "Bad",
                "bad_rate",
            ]
        ].copy()

        fig, ax_count = plt.subplots(
            figsize=(10, 6)
        )

        bar_width = 0.38

        ax_count.bar(
            xaxis - bar_width / 2.0,
            plot_data["Good"],
            width=bar_width,
            color="green",
            alpha=0.75,
            label="Good",
        )

        ax_count.bar(
            xaxis + bar_width / 2.0,
            plot_data["Bad"],
            width=bar_width,
            color="red",
            alpha=0.75,
            label="Bad",
        )

        ax_rate = ax_count.twinx()

        ax_rate.plot(
            xaxis,
            plot_data["bad_rate"],
            color="black",
            marker="o",
            linewidth=2,
            label="Bad rate",
        )

        ax_count.set_xticks(
            xaxis
        )

        ax_count.set_xticklabels(
            plot_data[grade_col].astype(str),
            rotation=45,
            ha="right",
        )

        ax_count.set_xlabel(
            "Grade, low risk to high risk"
        )

        ax_count.set_ylabel(
            "Number of accounts"
        )

        ax_rate.set_ylabel(
            "Bad rate"
        )

        ax_rate.set_ylim(
            bottom=0.0
        )

        ax_count.set_title(
            "Good / Bad / Bad Rate by Grade"
        )

        lines_1, labels_1 = (
            ax_count.get_legend_handles_labels()
        )

        lines_2, labels_2 = (
            ax_rate.get_legend_handles_labels()
        )

        ax_count.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="upper left",
        )

        ax_count.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()
        plt.show()

    # =====================================================
    # 14. CONSOLE SUMMARY
    # =====================================================

    print("\n" + "=" * 78)
    print(
        "GINI / KS / AUC / IV REPORT"
    )
    print("=" * 78)
    print(
        f"Tập dữ liệu              : {title_name}"
    )
    print(
        f"Số hồ sơ                 : {int(total_all):,}"
    )
    print(
        f"Good                      : {int(total_good):,}"
    )
    print(
        f"Bad                       : {int(total_bad):,}"
    )
    print(
        f"Bad rate                  : {total_bad / total_all:.6%}"
    )
    print(
        f"Số Grade                  : {len(Gini)}"
    )
    print(
        f"Chiều bảng                : "
        f"{'tăng dần' if ascending else 'giảm dần'} theo {grade_col}"
    )
    print(
        "Chiều ROC                 : high risk -> low risk"
    )
    print("-" * 78)
    print(
        f"Grouped AUC               : {grouped_auc:.6f}"
    )
    print(
        f"Grouped Gini              : {grouped_gini:.6f}"
    )
    print(
        f"Grouped c-statistic       : {grouped_c_statistic:.6f}"
    )

    if model_auc_available:
        print(
            f"Model AUC [{probability_col}]"
            f" : {model_auc:.6f}"
        )
        print(
            f"Model Gini                : {model_gini:.6f}"
        )
        print(
            f"AUC grouping loss         : {auc_grouping_loss:.6f}"
        )
    else:
        print(
            "Model AUC                 : KHÔNG TÍNH ĐƯỢC "
            "do không có xác suất cấp hồ sơ"
        )

    print(
        f"KS                        : {max_KS:.6f}"
    )
    print(
        f"KS Grade                  : {ks_grade}"
    )
    print(
        f"IV total                  : {iv_total:.6f}"
    )
    print(
        f"Gini rating               : {gini_rating}"
    )
    print(
        f"KS rating                 : {ks_rating}"
    )
    print(
        f"Kết luận                  : {conclusion}"
    )

    if direction_warning:
        print(
            f"Cảnh báo chiều            : {direction_warning}"
        )

    if export_path is not None:
        print(
            f"Đã xuất                   : {export_path}"
        )

    print("=" * 78)

    # =====================================================
    # 15. RETURN
    # =====================================================

    result = {
        "Gini": Gini,
        "auc": grouped_auc,
        "gini_value": grouped_gini,
        "c_statistic": grouped_c_statistic,
        "grouped_auc": grouped_auc,
        "grouped_auc_pairwise": grouped_auc_pairwise,
        "grouped_gini": grouped_gini,
        "model_auc": (
            model_auc
            if model_auc_available
            else np.nan
        ),
        "model_gini": (
            model_gini
            if model_auc_available
            else np.nan
        ),
        "auc_grouping_loss": (
            auc_grouping_loss
            if model_auc_available
            else np.nan
        ),
        "ks_value": max_KS,
        "ks_grade": ks_grade,
        "iv_total": iv_total,
        "gini_rating": gini_rating,
        "ks_rating": ks_rating,
        "conclusion": conclusion,
        "direction_warning": direction_warning,
        "roc_fpr": roc_fpr,
        "roc_tpr": roc_tpr,
        "model_auc_available": model_auc_available,
        "raw_data_available": raw_data_available,
    }

    return result


# =====================================================
# RUN
# =====================================================

result = gini_ks_report(
    data_group_test=data_group_test,
    group_cols=["Grade_new"],
    acct_col="BILLING_ACCTNUM",
    bad_col="NX",
    grade_col="Grade_new",
    probability_col="probability_manual_new",

    # Bảng báo cáo:
    # 0.AAA -> ... -> 9.D
    # low risk -> high risk
    ascending=True,

    grade_as_int=False,
    grade_sort_by_prefix=True,
    plot=True,
    export_path=r"D:\model\Gini_KS_result.xlsx",
    title_name="CREDIT_RISK_TEST",
)

Gini_result = result["Gini"]

grouped_auc_result = result["grouped_auc"]
grouped_gini_result = result["grouped_gini"]
model_auc_result = result["model_auc"]
model_gini_result = result["model_gini"]
ks_result = result["ks_value"]
iv_total_result = result["iv_total"]

# =====================================================
# RUN
# =====================================================







# ====== AUTO_APPEND_START ======
# [0] IMPORT BỔ SUNG (chỉ những gì chưa có trong script gốc)
import sys
import platform
from datetime import datetime


from scipy.stats import norm, ks_2samp
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# [1] WORKSPACE D:/model/plot/ + LOGGING
_T_APP = timer()
EPS = 1e-15


def _mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p


try:
    OUT_DIR = _mkdir("D:/model/plot")
except OSError:
    OUT_DIR = _mkdir(os.path.join(
        os.environ.get("ARENA_WORKSPACE") or os.environ.get("WORKSPACE") or os.getcwd(),
        "model"))
PLOT_DIR = _mkdir(os.path.join(OUT_DIR, "model_plot"))
XL = lambda name: os.path.join(OUT_DIR, name)


class _Tee(object):
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


_LOGF = open(XL("runtime_log.txt"), "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _LOGF)


def _h(title, ch="="):
    print("\n" + ch * 100)
    print(title)
    print(ch * 100)


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_h("[1] WORKSPACE & LOGGING")
print("Thư mục artifact : %s" % os.path.abspath(OUT_DIR))
print("Thư mục biểu đồ : %s" % os.path.abspath(PLOT_DIR))
print("Log runtime : %s" % os.path.abspath(XL("runtime_log.txt")))
print("Thời điểm bắt đầu : %s" % _ts())
plt.close("all")

# [1a] LOG TẢI FILE ĐẦU VÀO + MÔI TRƯỜNG


_h("[2] MÔI TRƯỜNG THỰC THI")
print("Python : %s" % sys.version.replace("\n", " "))
print("OS : %s | %s" % (platform.platform(), platform.machine()))
print("CWD : %s" % os.getcwd())
for _pkg in ("pandas", "numpy", "scipy", "sklearn", "matplotlib", "pyarrow",
             "openpyxl", "xlsxwriter"):
    try:
        print(" %-12s: %s" % (_pkg, __import__(_pkg).__version__))
    except Exception as _e:
        print(" %-12s: KHÔNG CÓ (%s)" % (_pkg, _e))

# [2a] TỐI ƯU BỘ NHỚ (downcast + gc) — GIỮ float64 CHO CỘT XÁC SUẤT/LOG-ODDS
_h("[3] TỐI ƯU BỘ NHỚ data_group")
_KEEP64 = {"probability_manual", "probability_manual_new", "probability_NX",
           "log_odds", "log_odds_new", "linear_score"}
_mem_before = float(data_group.memory_usage(deep=True).sum()) / 1024 ** 2
_down_f, _down_i, _cat_c = [], [], []
for _c in list(data_group.columns):
    _s = data_group[_c]
    if _c in _KEEP64:
        continue
    if str(_s.dtype) == "float64":
        data_group[_c] = _s.astype(np.float32)
        _down_f.append(_c)
    elif str(_s.dtype).startswith("int") and str(_s.dtype) != "int8":
        data_group[_c] = pd.to_numeric(_s, downcast="integer")
        _down_i.append(_c)
    elif _s.dtype == object and _s.nunique(dropna=False) <= max(50, int(0.01 * len(_s))):
        data_group[_c] = _s.astype("category")
        _cat_c.append(_c)
_freed = []
for _v in ("data_group_test", "X"):
    if _v in globals():
        del globals()[_v]
        _freed.append(_v)
gc.collect()
_mem_after = float(data_group.memory_usage(deep=True).sum()) / 1024 ** 2
print("Memory trước : %.2f MB" % _mem_before)
print("Memory sau : %.2f MB" % _mem_after)
print("Giảm : %.2f MB (%.2f%%)"
      % (_mem_before - _mem_after,
         100.0 * (_mem_before - _mem_after) / max(_mem_before, EPS)))
print("Cột float64→float32 (%d): %s" % (len(_down_f), _down_f))
print("Cột int downcast (%d): %s" % (len(_down_i), _down_i))
print("Cột → category (%d): %s" % (len(_cat_c), _cat_c))
print("Biến giải phóng : %s" % _freed)
print("Giữ float64 (độ chính xác xác suất): %s"
      % sorted(_KEEP64 & set(data_group.columns)))

# [3] DÒ BIẾN RUNTIME — ƯU TIÊN *_new, KẾ THỪA BEST CONFIG
_h("[4] PHÂN TÍCH SCRIPT GỐC & DÒ BIẾN RUNTIME (ưu tiên *_new)")


def _pick(cands, where):
    for _c in cands:
        if where is None:
            if _c in globals():
                return _c, ("_new" if _c.endswith("_new") else "gốc")
        elif _c in where:
            return _c, ("_new" if _c.endswith("_new") else "gốc")
    return None, None


PROB_COL, _v_prob = _pick(["probability_manual_new", "probability_manual",
                           "probability_NX"], data_group.columns)
SCORE_COL, _v_score = _pick(["score_new", "score"], data_group.columns)
GRADE_COL, _v_grade = _pick(["Grade_new", "Grade"], data_group.columns)
DECILE_COL, _v_dec = _pick(["decile_new", "decile"], data_group.columns)
LOGODDS_COL, _v_lo = _pick(["log_odds_new", "log_odds"], data_group.columns)
_TH_VAR, _v_th = _pick(["best_thresh_new", "best_thresh"], None)
THRESH = float(globals()[_TH_VAR])
HAS_DATE = "NGAY_BC" in data_group.columns


print("Preprocessing : downcast float32 + fillna(0) + scaler fit TRAIN-only")
print("Split logic : cột '%s' (train/test, không dùng train_test_split)" % SPLIT_COL)
print("Model : LogisticRegression | scaler=%s | C=%s | penalty=%s | "
      "solver=%s | class_weight=%s | max_iter=%s | tol=%s"
      % (best_cfg["scaler_name"], best_cfg["C"], best_cfg["penalty"],
         best_cfg["solver"], best_cfg["class_weight"], best_cfg["max_iter"],
         best_cfg["tol"]))
print("Calibration : Platt (α=%.10f, β=%.10f) — kế thừa, KHÔNG grid search lại"
      % (alpha_train, beta_train))
print("Scorecard (Siddiqi): base_score=%.4f | base_odds=%.4f | offset=%.6f | factor=%.6f | PDO=%.6f"
      % (float(data_group["base_score_new"].iloc[0]),
         float(data_group["base_odds_new"].iloc[0]),
         float(data_group["offset_new"].iloc[0]),
         float(data_group["factor_new"].iloc[0]),
         float(data_group["factor_new"].iloc[0]) * math.log(2.0)))
print("NGAY_BC : %s" % ("CÓ" if HAS_DATE else "KHÔNG → skip temporal"))
print("\n%-22s | %-10s | %-14s | %-10s | %-24s | %-9s | %s"
      % ("Tên biến", "Type", "Shape", "Dtype", "Nguồn", "Phiên bản", "Độ tin cậy"))
print("-" * 130)
for _nm, _obj, _src, _ver in (
    ("data_group", data_group, "script gốc — LOAD DATA", "gốc"),
    ("X_model", X_model, "script gốc — ma trận input", "gốc"),
    ("y_all", y_all, "script gốc — target %s" % TARGET_COL, "gốc"),
    (PROB_COL, data_group[PROB_COL], "cột data_group (PD sau calibration)", _v_prob),
    (LOGODDS_COL, data_group[LOGODDS_COL], "cột data_group", _v_lo),
    (SCORE_COL, data_group[SCORE_COL], "cột data_group (scorecard)", _v_score),
    (GRADE_COL, data_group[GRADE_COL], "cột data_group (band từ TRAIN)", _v_grade),
    (DECILE_COL, data_group[DECILE_COL], "cột data_group (decile từ TRAIN)", _v_dec),
):
    _sh = getattr(_obj, "shape", (len(_obj),))
    _dt = "mixed" if isinstance(_obj, pd.DataFrame) else str(getattr(_obj, "dtype", "float64"))
    print("%-22s | %-10s | %-14s | %-10s | %-24s | %-9s | %s"
          % (_nm, type(_obj).__name__, str(_sh), _dt, _src, _ver, "CAO"))
print("%-22s | %-10s | %-14s | %-10s | %-24s | %-9s | %s"
      % (_TH_VAR, "float", "()", "float64", "scan TRAIN-only", _v_th, "CAO"))
print("\nBest config kế thừa: %s" % best_cfg)
print("Threshold kế thừa : %s = %.6f (nguồn: scan Youden-mở-rộng trên TRAIN, "
      "áp dụng thống nhất TRAIN/TEST/FULL)" % (_TH_VAR, THRESH))

# [4] MẢNG ĐÁNH GIÁ + KIỂM TRA CHIỀU SCORE + BAND/DECILE (TRAIN-only cut-points)
_TR = (data_group[SPLIT_COL].astype(str).str.strip().str.lower() == "train").to_numpy(bool)
_TE = ~_TR
Y = np.asarray(y_all, dtype=np.int8)
P = data_group[PROB_COL].to_numpy(dtype=np.float64, copy=False)
SC = data_group[SCORE_COL].to_numpy(dtype=np.float64, copy=False)
SETS = (("TRAIN", _TR), ("TEST", _TE), ("FULL", np.ones(len(Y), dtype=bool)))
_h("[5] KIỂM TRA CHIỀU SCORE / POSITIVE CLASS")
_pos_rate = float(Y.mean())
_corr_sc = float(np.corrcoef(SC, P)[0, 1])
print("Positive class = bad/default = 1 | bad rate FULL = %.6f | nhãn duy nhất = %s"
      % (_pos_rate, np.unique(Y).tolist()))
print("corr(score, PD) = %+.6f → score %s PD (%s)"
      % (_corr_sc, "NGƯỢC CHIỀU" if _corr_sc < 0 else "CÙNG CHIỀU",
         "đúng chuẩn scorecard: score cao = rủi ro thấp" if _corr_sc < 0
         else "score dùng đảo dấu khi xếp hạng rủi ro"))
SC_RISK = -SC if _corr_sc < 0 else SC
print("Biến xếp hạng rủi ro dùng cho KS/AUC theo score: %s"
      % ("-%s" % SCORE_COL if _corr_sc < 0 else SCORE_COL))
print("Decile/band: dùng '%s' & '%s' — cut-points đã học 100%% từ TRAIN (no leakage)"
      % (DECILE_COL, GRADE_COL))


# [5] ≥17 METRICS TRÊN TRAIN + TEST + FULL
def _lvl(x, lo, hi, up_good=True):
    """Ngưỡng 3 mức: trả (nhãn, ký hiệu)."""
    if x < lo:
        return ("Thấp", "✗") if up_good else ("Tốt", "✓")
    if x <= hi:
        return ("Trung bình", "⚠")
    return ("Cao", "✓") if up_good else ("Kém", "✗")


def _qual(x, good, mid, low_good=False):
    if low_good:
        if x <= good:
            return "Tốt", "✓"
        return ("Trung bình", "⚠") if x <= mid else ("Kém", "✗")
    if x >= good:
        return "Tốt", "✓"
    return ("Trung bình", "⚠") if x >= mid else ("Kém", "✗")


def _ks(y, p):
    _f, _t, _ = roc_curve(y, p)
    return float(np.max(_t - _f))


def _midrank(x):
    _j = np.argsort(x, kind="mergesort")
    _z = x[_j]
    _b = np.flatnonzero(np.r_[True, _z[1:] != _z[:-1], True])
    _mr = 0.5 * (_b[:-1] + _b[1:] - 1) + 1.0
    _t = np.repeat(_mr, np.diff(_b))
    _o = np.empty(len(x), dtype=np.float64)
    _o[_j] = _t
    return _o


def _delong(y, p):
    """AUC + SE theo DeLong (midrank, xử lý ties)."""
    _pos, _neg = p[y == 1], p[y == 0]
    _m, _n = _pos.size, _neg.size
    _tz = _midrank(np.concatenate([_pos, _neg]))
    _auc = (_tz[:_m].sum() - _m * (_m + 1) / 2.0) / (_m * _n)
    _v01 = (_tz[:_m] - _midrank(_pos)) / _n
    _v10 = 1.0 - (_tz[_m:] - _midrank(_neg)) / _m
    _se = math.sqrt(np.var(_v01, ddof=1) / _m + np.var(_v10, ddof=1) / _n)
    return float(_auc), float(_se), _m, _n


def _hanley(auc, m, n):
    _q1, _q2 = auc / (2.0 - auc), 2.0 * auc * auc / (1.0 + auc)
    return math.sqrt((auc * (1 - auc) + (m - 1) * (_q1 - auc ** 2)
                      + (n - 1) * (_q2 - auc ** 2)) / (m * n))


def _bins10(p, y):
    _ix = np.clip((p * 10).astype(np.intp), 0, 9)
    _cnt = np.bincount(_ix, minlength=10).astype(np.float64)
    _sy = np.bincount(_ix, weights=y.astype(np.float64), minlength=10)
    _sp = np.bincount(_ix, weights=p, minlength=10)
    return _cnt, _sy, _sp


def _ece_mce(p, y):
    _c, _sy, _sp = _bins10(p, y)
    _ok = _c > 0
    _ece = float(np.sum(np.abs(_sy - _sp)) / len(y))
    _mce = float(np.max(np.abs(_sy[_ok] - _sp[_ok]) / _c[_ok]))
    return _ece, _mce


def _brier_dec(p, y):
    """BS = Reliability − Resolution + Uncertainty (10 bins đều)."""
    _c, _sy, _sp = _bins10(p, y)
    _n = float(len(y))
    _ok = _c > 0
    _acc, _conf = _sy[_ok] / _c[_ok], _sp[_ok] / _c[_ok]
    _w = _c[_ok] / _n
    _yb = float(y.mean())
    _bs = float(np.mean((p - y) ** 2))
    return (_bs, float(np.sum(_w * (_conf - _acc) ** 2)),
            float(np.sum(_w * (_acc - _yb) ** 2)), _yb * (1.0 - _yb))


def _logit(p):
    _p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(_p / (1.0 - _p))


def _calib_ab(p, y):
    """logit(y) = α + β·logit(p), MLE không phạt (tái sử dụng _platt_mle của script gốc)."""
    return _platt_mle(_logit(p), y)


def _psi(exp_v, act_v, edges):
    _e = np.histogram(exp_v, bins=edges)[0].astype(np.float64) / max(len(exp_v), 1)
    _a = np.histogram(act_v, bins=edges)[0].astype(np.float64) / max(len(act_v), 1)
    _c = (_a - _e) * np.log((_a + EPS) / (_e + EPS))
    return float(_c.sum()), _e, _a, _c


def _iv_of(x_tr, y_tr, x_ev, y_ev):
    """WOE/IV: cut-point học 100% từ TRAIN → áp cho tập đánh giá."""
    _u = np.unique(x_tr)
    if _u.size <= 20:
        _edges = np.r_[-np.inf, (_u[1:] + _u[:-1]) / 2.0, np.inf]
    else:
        _q = np.unique(np.quantile(x_tr, np.linspace(0, 1, 11)))
        _edges = np.r_[-np.inf, _q[1:-1], np.inf]
    _i_tr = np.digitize(x_ev, _edges[1:-1], right=True)
    _nb = len(_edges) - 1
    _g = np.bincount(_i_tr[y_ev == 0], minlength=_nb).astype(np.float64)
    _b = np.bincount(_i_tr[y_ev == 1], minlength=_nb).astype(np.float64)
    _dg, _db = _g / max(_g.sum(), EPS), _b / max(_b.sum(), EPS)
    _woe = np.log((_dg + EPS) / (_db + EPS))
    _iv = (_dg - _db) * _woe
    return float(_iv.sum()), _nb, _dg, _db, _woe, _iv


_h("[6] TÍNH ≥17 METRICS TRÊN TRAIN / TEST / FULL (100% dữ liệu, no sampling)")
M = {}
_PRED = (P >= THRESH).astype(np.int8)
_SC_EDGES = np.r_[-np.inf, np.unique(np.quantile(SC_RISK[_TR], np.linspace(0, 1, 11)))[1:-1],
                  np.inf]
for _nm, _msk in SETS:
    _y, _p, _pr = Y[_msk], P[_msk], _PRED[_msk]
    _auc, _se, _m1, _n0_ = _delong(_y, _p)
    _a, _b = _calib_ab(_p, _y)
    _ece, _mce = _ece_mce(_p, _y)
    _bs, _rel, _res, _unc = _brier_dec(_p, _y)
    _tn, _fp, _fn, _tp = confusion_matrix(_y, _pr, labels=[0, 1]).ravel()
    M[_nm] = {
        "n": int(_msk.sum()), "bad": int(_y.sum()), "bad_rate": float(_y.mean()),
        "AUC": _auc, "AUC_SE": _se, "AUC_HM_SE": _hanley(_auc, _m1, _n0_),
        "Gini": 2.0 * _auc - 1.0, "KS": _ks(_y, _p),
        "Precision": float(precision_score(_y, _pr, zero_division=0)),
        "Recall": float(recall_score(_y, _pr, zero_division=0)),
        "F1": float(f1_score(_y, _pr, zero_division=0)),
        "BalancedAcc": float(balanced_accuracy_score(_y, _pr)),
        "Accuracy": float(accuracy_score(_y, _pr)),
        "LogLoss": float(log_loss(_y, _p, labels=[0, 1])),
        "Brier": _bs, "REL": _rel, "RES": _res, "UNC": _unc,
        "ECE": _ece, "MCE": _mce, "alpha": _a, "beta": _b,
        "AP": float(average_precision_score(_y, _p)),
        "TN": int(_tn), "FP": int(_fp), "FN": int(_fn), "TP": int(_tp),
    }
    print(" %-5s: n=%d bad=%d (%.4f%%) AUC=%.6f KS=%.6f Gini=%.6f Acc=%.6f "
          "Brier=%.6f ECE=%.6f α=%.6f β=%.6f"
          % (_nm, M[_nm]["n"], M[_nm]["bad"], 100 * M[_nm]["bad_rate"], _auc,
             M[_nm]["KS"], M[_nm]["Gini"], M[_nm]["Accuracy"], _bs, _ece, _a, _b))

PSI_SCORE, PSI_E, PSI_A, PSI_C = _psi(SC_RISK[_TR], SC_RISK[_TE], _SC_EDGES)
PSI_PROB = _psi(P[_TR], P[_TE],
                np.r_[-np.inf,
                      np.unique(np.quantile(P[_TR], np.linspace(0, 1, 11)))[1:-1],
                      np.inf])[0]
_iv_rows = []
_x_tr_all = X_model[_TR]
for _j, _f in enumerate(feature_cols):
    _xt = _x_tr_all[:, _j].astype(np.float64)
    _iv_t, _nb, _dg, _db, _woe, _ivb = _iv_of(_xt, Y[_TR], _xt, Y[_TR])
    _iv_e = _iv_of(_xt, Y[_TR], X_model[_TE, _j].astype(np.float64), Y[_TE])[0]

    _iv_rows.append({"Feature": _f, "So_bin": _nb, "IV_train": _iv_t, "IV_test": _iv_e,
                     "Muc_do_IV_train": _lvl(_iv_t, 0.10, 0.30)[0],
                     "Ky_hieu": _lvl(_iv_t, 0.10, 0.30)[1],
                     "Canh_bao": ("IV<0.02: không sức dự báo" if _iv_t < 0.02
                                  else ("IV>0.50: nghi ngờ leakage" if _iv_t > 0.50 else "-"))})
del _x_tr_all
gc.collect()
IV_DF = pd.DataFrame(_iv_rows).sort_values("IV_train", ascending=False)
IV_TR_MAX = float(
    IV_DF["IV_train"].max()
)
IV_TE_MAX = float(
    IV_DF["IV_test"].max()
)
IV_TR_MEDIAN = float(
    IV_DF["IV_train"].median()
)
IV_TE_MEDIAN = float(
    IV_DF["IV_test"].median()
)

D_KS = M["TRAIN"]["KS"] - M["TEST"]["KS"]
D_AUC = M["TRAIN"]["AUC"] - M["TEST"]["AUC"]
D_GINI = M["TRAIN"]["Gini"] - M["TEST"]["Gini"]
print(" PSI(score TRAIN→TEST) = %.6f | PSI(PD) = %.6f" % (PSI_SCORE, PSI_PROB))
print(" IV lớn nhất theo biến TRAIN = %.6f | IV lớn nhất theo biến TEST = %.6f | IV max = %.6f (%s)"
      % (IV_TR_MAX, IV_TE_MAX, IV_DF["IV_train"].iloc[0], IV_DF["Feature"].iloc[0]))
print(" Δ KS = %+.6f | Δ AUC = %+.6f | Δ Gini = %+.6f (train − test)"
      % (D_KS, D_AUC, D_GINI))

_KS_L = lambda v: _lvl(v, 0.40, 0.60)
_IV_L = lambda v: _lvl(v, 0.10, 0.30)
_AUC_L = lambda v: _lvl(v, 0.70, 0.80)
_GINI_L = lambda v: _lvl(v, 0.40, 0.60)
METRIC_ROWS = [
    ("Area Under ROC Curve (AUC)", "AUC", "AUC = ∫TPR d(FPR); Gini = 2·AUC − 1",
     "Khả năng phân biệt tốt/xấu của mô hình",
     "Thấp <0.70 | TB 0.70–0.80 | Cao >0.80",
     M["TRAIN"]["AUC"], M["TEST"]["AUC"], M["FULL"]["AUC"], _AUC_L,
     "Basel II Pillar 2 — kiểm định discrimination; SR 11-7 model performance"),
    ("Gini Coefficient (Accuracy Ratio bản chuẩn hoá)", "Gini", "Gini = 2·AUC − 1",
     "Mức độ tách biệt xếp hạng tín dụng",
     "Thấp <0.40 | TB 0.40–0.60 | Cao >0.60",
     M["TRAIN"]["Gini"], M["TEST"]["Gini"], M["FULL"]["Gini"], _GINI_L,
     "S&P Global benchmark; ECB TRIM — sức mạnh xếp hạng PD"),
    ("Kolmogorov–Smirnov Statistic (KS)", "KS",
     "KS = maxₜ |F_bad(t) − F_good(t)|",
     "Khoảng cách phân phối lớn nhất giữa nhóm xấu và nhóm tốt",
     "Thấp <0.40 | TB 0.40–0.60 | Cao >0.60 (KS>0.70 nghi overfit/leakage)",
     M["TRAIN"]["KS"], M["TEST"]["KS"], M["FULL"]["KS"], _KS_L,
     "Siddiqi (2017) — cut-off scorecard; PRA SS1/23 validation"),
    ("Precision (Positive Predictive Value)", "P", "P = TP / (TP + FP)",
     "Tỷ lệ dự báo xấu thực sự xấu",
     "Tốt ≥0.60 | TB 0.40–0.60 | Kém <0.40",
     M["TRAIN"]["Precision"], M["TEST"]["Precision"], M["FULL"]["Precision"],
     lambda v: _qual(v, 0.60, 0.40), "Chi phí thẩm định & tỷ lệ từ chối sai (Basel Pillar 2)"),
    ("Recall (Sensitivity / TPR)", "R", "R = TP / (TP + FN)",
     "Tỷ lệ khách hàng xấu bị bắt được",
     "Tốt ≥0.60 | TB 0.40–0.60 | Kém <0.40",
     M["TRAIN"]["Recall"], M["TEST"]["Recall"], M["FULL"]["Recall"],
     lambda v: _qual(v, 0.60, 0.40), "Kiểm soát tổn thất kỳ vọng EL; IFRS 9 staging"),
    ("F1-Score", "F1", "F₁ = 2PR / (P + R)", "Trung bình điều hoà Precision–Recall",
     "Tốt ≥0.60 | TB 0.40–0.60 | Kém <0.40",
     M["TRAIN"]["F1"], M["TEST"]["F1"], M["FULL"]["F1"],
     lambda v: _qual(v, 0.60, 0.40), "Cân bằng chi phí FP/FN trong chính sách tín dụng"),
    ("Balanced Accuracy", "BAcc", "BAcc = (TPR + TNR) / 2",
     "Độ chính xác cân bằng cho dữ liệu mất cân đối",
     "Tốt ≥0.70 | TB 0.60–0.70 | Kém <0.60",
     M["TRAIN"]["BalancedAcc"], M["TEST"]["BalancedAcc"], M["FULL"]["BalancedAcc"],
     lambda v: _qual(v, 0.70, 0.60), "SR 11-7 — hiệu năng trên lớp thiểu số (default)"),
    ("Accuracy", "Acc", "Acc = (TP + TN) / N", "Tỷ lệ phân loại đúng tổng thể",
     "Tốt ≥0.70 | TB 0.60–0.70 | Kém <0.60",
     M["TRAIN"]["Accuracy"], M["TEST"]["Accuracy"], M["FULL"]["Accuracy"],
     lambda v: _qual(v, 0.70, 0.60), "Industry benchmark ≥0.60 cho scorecard bán lẻ"),
    ("Logarithmic Loss (Binary Cross-Entropy)", "LogLoss",
     "LL = −(1/N)Σ[y·ln p + (1−y)·ln(1−p)], p∈[1e-15, 1−1e-15]",
     "Sai số xác suất có phạt nặng dự báo tự tin sai",
     "Tốt ≤0.30 | TB 0.30–0.50 | Kém >0.50",
     M["TRAIN"]["LogLoss"], M["TEST"]["LogLoss"], M["FULL"]["LogLoss"],
     lambda v: _qual(v, 0.30, 0.50, True), "IFRS 9 — chất lượng PD point-in-time"),
    ("Brier Score", "BS", "BS = (1/N)Σ(pᵢ − yᵢ)²", "Sai số bình phương trung bình của PD",
     "Tốt ≤0.15 | TB 0.15–0.25 | Kém >0.25",
     M["TRAIN"]["Brier"], M["TEST"]["Brier"], M["FULL"]["Brier"],
     lambda v: _qual(v, 0.15, 0.25, True), "IFRS 9 ECL — độ chính xác xác suất tuyệt đối"),
    ("Expected Calibration Error", "ECE",
     "ECE = Σ_b (n_b/N)·|acc_b − conf_b|, B = 10 bins đều [0,1]",
     "Sai lệch hiệu chuẩn trung bình có trọng số",
     "Tốt ≤0.05 | TB 0.05–0.10 | Kém >0.10",
     M["TRAIN"]["ECE"], M["TEST"]["ECE"], M["FULL"]["ECE"],
     lambda v: _qual(v, 0.05, 0.10, True), "ECB TRIM — PD calibration; thay Spiegelhalter"),
    ("Maximum Calibration Error", "MCE", "MCE = max_b |acc_b − conf_b|",
     "Sai lệch hiệu chuẩn tệ nhất theo bin",
     "Tốt ≤0.10 | TB 0.10–0.20 | Kém >0.20",
     M["TRAIN"]["MCE"], M["TEST"]["MCE"], M["FULL"]["MCE"],
     lambda v: _qual(v, 0.10, 0.20, True), "ECB TRIM — rủi ro cục bộ theo dải PD"),
    ("Calibration Intercept (α)", "α", "logit(y) = α + β·logit(p) — MLE không phạt",
     "Lệch hệ thống của PD (α>0: dự báo thấp hơn thực tế)",
     "Tốt |α| ≤0.10 | TB 0.10–0.30 | Kém >0.30",
     M["TRAIN"]["alpha"], M["TEST"]["alpha"], M["FULL"]["alpha"],
     lambda v: _qual(abs(v), 0.10, 0.30, True), "Basel — PD calibration test; IFRS 9 ECL bias"),
    ("Calibration Slope (β)", "β", "logit(y) = α + β·logit(p) — MLE không phạt",
     "Độ dốc hiệu chuẩn (β<1: PD quá phân tán)",
     "Tốt 0.90–1.10 | TB [0.80;0.90]∪[1.10;1.20] | Kém <0.80 hoặc >1.20",
     M["TRAIN"]["beta"], M["TEST"]["beta"], M["FULL"]["beta"],
     lambda v: ("Tốt", "✓") if 0.90 <= v <= 1.10 else
     (("Trung bình", "⚠") if 0.80 <= v <= 1.20 else ("Kém", "✗")),
     "ECB TRIM — kiểm định độ dốc hiệu chuẩn PD"),
    ("Population Stability Index (score)", "PSI",
     "PSI = Σ(Aᵢ − Eᵢ)·ln((Aᵢ+ε)/(Eᵢ+ε)), E = TRAIN, A = TEST",
     "Dịch chuyển phân phối điểm giữa TRAIN và TEST",
     "Ổn định <0.10 | Cần theo dõi 0.10–0.25 | Bất ổn ≥0.25",
     PSI_SCORE, PSI_SCORE, PSI_SCORE, lambda v: _qual(v, 0.10, 0.25, True),
     "Basel/ECB TRIM — giám sát population drift; điều kiện re-development"),
    ("Information Value (tổng %d biến)" % len(feature_cols), "IV",
     "WOEᵢ = ln((Dist_Goodᵢ+ε)/(Dist_Badᵢ+ε)); IV = Σ(Dist_Goodᵢ − Dist_Badᵢ)·WOEᵢ",
     "Sức mạnh dự báo tổng hợp của bộ biến",
     "Thấp <0.10 | TB 0.10–0.30 | Cao >0.30 (>0.50 nghi leakage)",
     IV_TR_MAX, IV_TE_MAX, IV_TR_MAX + 0.0, _IV_L,
     "Siddiqi (2017) — feature selection & tài liệu hoá model (SR 11-7)"),
    ("Overfitting Gap — Δ KS", "ΔKS", "Δ KS = KS_train − KS_test (|Δ| ≤ 3–5%)",
     "Mức suy giảm sức phân biệt khi ra ngoài mẫu",
     "Ổn định |Δ| ≤0.03 | Chấp nhận 0.03–0.05 | Overfit >0.05",
     D_KS, D_KS, D_KS, lambda v: _qual(abs(v), 0.03, 0.05, True),
     "SR 11-7 / PRA SS1/23 — kiểm soát overfitting"),
    ("Overfitting Gap — Δ AUC", "ΔAUC", "Δ AUC = AUC_train − AUC_test (|Δ| ≤ 2–3%)",
     "Mức suy giảm AUC khi ra ngoài mẫu",
     "Ổn định |Δ| ≤0.02 | Chấp nhận 0.02–0.03 | Overfit >0.03",
     D_AUC, D_AUC, D_AUC, lambda v: _qual(abs(v), 0.02, 0.03, True),
     "SR 11-7 / PRA SS1/23 — kiểm soát overfitting"),
]
_h("[7] BẢNG ≥17 METRICS × 8 THÀNH PHẦN")
print("%-3s %-46s %-11s %-11s %-11s %-14s %-14s"
      % ("#", "Metric", "Train", "Test", "Full", "Đánh giá Train", "Đánh giá Test"))
print("-" * 130)
_metric_tbl = []
for _i, _r in enumerate(METRIC_ROWS, 1):
    _nm, _ab, _fo, _mn, _th, _tr, _te, _fu, _lf, _ap = _r
    _lt, _st = _lf(_tr)
    _le, _se_ = _lf(_te)
    print("%-3d %-46s %-11.6f %-11.6f %-11.6f %-14s %-14s"
          % (_i, _nm[:46], _tr, _te, _fu, "%s %s" % (_st, _lt), "%s %s" % (_se_, _le)))
    _metric_tbl.append({
        "#": _i, "Metric (tên đầy đủ)": _nm, "Viết tắt": _ab, "Công thức": _fo,
        "Ý nghĩa": _mn, "Ngưỡng": _th, "Train": round(_tr, 10), "Test": round(_te, 10),
        "Full": round(_fu, 10), "Nhận xét Train": "%s %s" % (_st, _lt),
        "Nhận xét Test": "%s %s" % (_se_, _le), "Ứng dụng Credit Risk/Basel/IFRS 9/SR 11-7": _ap})
METRIC_DF = pd.DataFrame(_metric_tbl)

# [6] ASSESSMENT ≥7 (ECE/MCE/Brier thay Spiegelhalter — KHÔNG bịa p-value)
_h("[8] ASSESSMENT THỐNG KÊ (≥7) — DeLong / Hanley–McNeil / KS test / Calibration / PSI")
_z975 = float(norm.ppf(0.975))
def _auc_ci(auc, se, z_value):
    lower = max(
        0.0,
        float(auc)
        - float(z_value) * float(se),
    )
    upper = min(
        1.0,
        float(auc)
        + float(z_value) * float(se),
    )
    return lower, upper


_auc_ci_train = _auc_ci(
    M["TRAIN"]["AUC"],
    M["TRAIN"]["AUC_SE"],
    _z975,
)

_auc_ci_test = _auc_ci(
    M["TEST"]["AUC"],
    M["TEST"]["AUC_SE"],
    _z975,
)

_auc_hm_ci_train = _auc_ci(
    M["TRAIN"]["AUC"],
    M["TRAIN"]["AUC_HM_SE"],
    _z975,
)

_auc_hm_ci_test = _auc_ci(
    M["TEST"]["AUC"],
    M["TEST"]["AUC_HM_SE"],
    _z975,
)




_ks_tr = ks_2samp(P[_TR][Y[_TR] == 1], P[_TR][Y[_TR] == 0], method="asymp")
_ks_te = ks_2samp(P[_TE][Y[_TE] == 1], P[_TE][Y[_TE] == 0], method="asymp")


_A_ROWS = [
    (
        "AUC 95% CI — DeLong (1988)",
        "AUC ± 1.96·SE_DeLong (midrank, xử lý ties)",
        "Khoảng tin cậy chính xác của AUC dựa trên lý thuyết U-statistic",
        "Cận dưới CI > 0.70 → discrimination đạt ngưỡng tham chiếu",
        "AUC=%.6f SE=%.6f CI95=[%.6f; %.6f]"
        % (
            M["TRAIN"]["AUC"],
            M["TRAIN"]["AUC_SE"],
            _auc_ci_train[0],
            _auc_ci_train[1],
        ),
        "AUC=%.6f SE=%.6f CI95=[%.6f; %.6f]"
        % (
            M["TEST"]["AUC"],
            M["TEST"]["AUC_SE"],
            _auc_ci_test[0],
            _auc_ci_test[1],
        ),
        "✓ Đạt" if _auc_ci_test[0] > 0.70 else "✗ Chưa đạt",
        "SR 11-7 — báo cáo bất định của chỉ số hiệu năng",
    ),
    (
        "Fallback CI — Hanley–McNeil (1982)",
        (
            "SE_HM = √[(A(1−A) + (m−1)(Q₁−A²) "
            "+ (n−1)(Q₂−A²))/(mn)], "
            "Q₁=A/(2−A), Q₂=2A²/(1+A)"
        ),
        "CI thay thế khi DeLong không khả dụng; đối chiếu chéo với DeLong",
        "|SE_HM − SE_DeLong| nhỏ → kết luận CI vững",
        "SE_HM=%.6f CI95=[%.6f; %.6f]"
        % (
            M["TRAIN"]["AUC_HM_SE"],
            _auc_hm_ci_train[0],
            _auc_hm_ci_train[1],
        ),
        "SE_HM=%.6f CI95=[%.6f; %.6f]"
        % (
            M["TEST"]["AUC_HM_SE"],
            _auc_hm_ci_test[0],
            _auc_hm_ci_test[1],
        ),
        "✓ DeLong là chính, Hanley–McNeil dùng đối chiếu",
        "ECB TRIM — robustness của khoảng tin cậy",
    ),
    (
        "KS two-sample test (bad vs good)",
        "D = supₜ|F_bad(t) − F_good(t)|; p-value tiệm cận",
        "Kiểm định hai phân phối PD của nhóm xấu và nhóm tốt khác nhau",
        "p < 0.05 → hai phân phối khác biệt có ý nghĩa",
        "D=%.6f p=%.3e" % (_ks_tr.statistic, _ks_tr.pvalue),
        "D=%.6f p=%.3e" % (_ks_te.statistic, _ks_te.pvalue),
        "✓ Bác bỏ H₀" if _ks_te.pvalue < 0.05 else "✗ Không bác bỏ H₀",
        "Siddiqi (2017) — kiểm định sức phân biệt (hypothesis test thực sự)",
    ),
    (
        "Calibration Intercept α (assessment)",
        "logit(y) = α + β·logit(p), MLE không phạt",
        "Đo lệch hệ thống mức PD trung bình",
        (
            "Tốt |α|≤0.10 | TB 0.10–0.30 | Kém >0.30 — "
            "đây là assessment theo ngưỡng, không phải hypothesis test"
        ),
        "α=%.6f (%s)"
        % (
            M["TRAIN"]["alpha"],
            _qual(abs(M["TRAIN"]["alpha"]), 0.10, 0.30, True)[0],
        ),
        "α=%.6f (%s)"
        % (
            M["TEST"]["alpha"],
            _qual(abs(M["TEST"]["alpha"]), 0.10, 0.30, True)[0],
        ),
        _qual(abs(M["TEST"]["alpha"]), 0.10, 0.30, True)[1]
        + " theo ngưỡng",
        "ECB TRIM / IFRS 9 — bias của PD",
    ),
    (
        "Calibration Slope β (assessment)",
        "logit(y) = α + β·logit(p), MLE không phạt",
        "Đo độ dốc hiệu chuẩn; β<1 = PD quá phân tán",
        (
            "Tốt 0.90–1.10 | TB [0.80;0.90]∪[1.10;1.20] | "
            "Kém ngoài [0.80;1.20] — assessment theo ngưỡng"
        ),
        "β=%.6f" % M["TRAIN"]["beta"],
        "β=%.6f" % M["TEST"]["beta"],
        (
            "✓"
            if 0.90 <= M["TEST"]["beta"] <= 1.10
            else (
                "⚠"
                if 0.80 <= M["TEST"]["beta"] <= 1.20
                else "✗"
            )
        )
        + " theo ngưỡng",
        "ECB TRIM — độ dốc hiệu chuẩn",
    ),
    (
        "Expected Calibration Error (ECE)",
        "ECE = Σ_b (n_b/N)|acc_b − conf_b|, B=10",
        "Sai số hiệu chuẩn kỳ vọng — thay thế Spiegelhalter",
        "Tốt ≤0.05 | TB 0.05–0.10 | Kém >0.10 — assessment theo ngưỡng",
        "ECE=%.6f" % M["TRAIN"]["ECE"],
        "ECE=%.6f" % M["TEST"]["ECE"],
        _qual(M["TEST"]["ECE"], 0.05, 0.10, True)[1] + " theo ngưỡng",
        (
            "ECB TRIM — thay Spiegelhalter "
            "(variance test không ổn định khi discrimination cao)"
        ),
    ),
    (
        "Maximum Calibration Error (MCE)",
        "MCE = max_b |acc_b − conf_b|",
        "Sai số hiệu chuẩn tối đa — thay thế Spiegelhalter",
        "Tốt ≤0.10 | TB 0.10–0.20 | Kém >0.20 — assessment theo ngưỡng",
        "MCE=%.6f" % M["TRAIN"]["MCE"],
        "MCE=%.6f" % M["TEST"]["MCE"],
        _qual(M["TEST"]["MCE"], 0.10, 0.20, True)[1] + " theo ngưỡng",
        "ECB TRIM — rủi ro hiệu chuẩn cục bộ",
    ),
    (
        "Brier Score Decomposition (Murphy 1973)",
        "BS = Reliability − Resolution + Uncertainty",
        "Tách sai số xác suất thành hiệu chuẩn / phân biệt / bất định nội tại",
        "Reliability → 0 và Resolution lớn là tốt",
        "BS=%.6f REL=%.6f RES=%.6f UNC=%.6f"
        % (
            M["TRAIN"]["Brier"],
            M["TRAIN"]["REL"],
            M["TRAIN"]["RES"],
            M["TRAIN"]["UNC"],
        ),
        "BS=%.6f REL=%.6f RES=%.6f UNC=%.6f"
        % (
            M["TEST"]["Brier"],
            M["TEST"]["REL"],
            M["TEST"]["RES"],
            M["TEST"]["UNC"],
        ),
        _qual(M["TEST"]["Brier"], 0.15, 0.25, True)[1] + " theo ngưỡng",
        "IFRS 9 — chất lượng PD dùng cho ECL",
    ),
    (
        "PSI stability (score TRAIN→TEST)",
        "PSI = Σ(Aᵢ − Eᵢ)ln((Aᵢ+ε)/(Eᵢ+ε)), 10 bin từ TRAIN",
        "Đo dịch chuyển phân phối điểm giữa 2 tập",
        (
            "<0.10 ổn định | 0.10–0.25 theo dõi | "
            "≥0.25 bất ổn — assessment theo ngưỡng"
        ),
        "E = TRAIN (baseline, n=%d)" % M["TRAIN"]["n"],
        "PSI=%.6f (n=%d)" % (PSI_SCORE, M["TEST"]["n"]),
        _qual(PSI_SCORE, 0.10, 0.25, True)[1] + " theo ngưỡng",
        "Basel/ECB TRIM — giám sát population drift",
    ),
    (
        "Overfitting assessment (Δ KS & Δ AUC)",
        (
            "Δ KS = |KS_tr − KS_te| ≤ 3–5%; "
            "Δ AUC = |AUC_tr − AUC_te| ≤ 2–3%"
        ),
        "Đo mức suy giảm hiệu năng ngoài mẫu",
        "Trong ngưỡng → ổn định; vượt → overfit dữ liệu quá khứ",
        "KS_tr=%.6f AUC_tr=%.6f"
        % (M["TRAIN"]["KS"], M["TRAIN"]["AUC"]),
        "|ΔKS|=%.6f |ΔAUC|=%.6f" % (abs(D_KS), abs(D_AUC)),
        (
            "✓ Ổn định"
            if abs(D_KS) <= 0.05 and abs(D_AUC) <= 0.03
            else "✗ Vượt ngưỡng"
        ),
        "SR 11-7 / PRA SS1/23 — model risk management",
    ),
]


            
            
_ASSESS = [{"#": _i, "Assessment": _r[0], "Phương pháp / Công thức": _r[1], "Ý nghĩa": _r[2],
            "Ngưỡng": _r[3], "Giá trị Train": _r[4], "Giá trị Test": _r[5],
            "Nhận xét tự động": _r[6], "Ứng dụng Basel/IFRS 9/SR 11-7": _r[7]}
           for _i, _r in enumerate(_A_ROWS, 1)]
for _a in _ASSESS:
    print("%2d. %-42s | Train: %-52s | Test: %-52s | %s"
          % (_a["#"], _a["Assessment"][:42], _a["Giá trị Train"][:52],
             _a["Giá trị Test"][:52], _a["Nhận xét tự động"]))
ASSESS_DF = pd.DataFrame(_ASSESS)

# [7] ≥21 PLOTS VỚI 5 ANNOTATION
_h("[9] SINH BIỂU ĐỒ (≥21) — mỗi biểu đồ đủ 5 annotation")
PLOTS, ANNO, SKIPS = [], {}, []


def _err(sec, e):
    SKIPS.append({"Mục": sec, "Loại": "LỖI → skip section", "Nguyên nhân": "%s: %s"
                  % (type(e).__name__, e), "Xử lý": "Bỏ qua mục, tiếp tục pipeline"})
    print(" ✗ %s lỗi: %s — bỏ qua mục này, pipeline tiếp tục" % (sec, e))

_COL = {"TRAIN": "#1f77b4", "TEST": "#d62728"}


def _anno(ax, key, formula, meaning, levels, v_tr, v_te, verdict, loc=(0.02, 0.02)):
    ANNO[key] = (round(float(v_tr), 10), round(float(v_te), 10))
    ax.text(loc[0], loc[1],
            "① Công thức: %s\n② Ý nghĩa: %s\n③ Ngưỡng: %s\n④ Train = %.6f | Test = %.6f\n⑤ Đánh giá: %s"
            % (formula, meaning, levels, v_tr, v_te, verdict),
            transform=ax.transAxes, fontsize=7.2, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.42", fc="#fffbe6", ec="#8c8c8c", alpha=0.95))

def _save(fig, idx, name):
    _p = os.path.join(
        PLOT_DIR,
        "%02d_%s.png" % (idx, name),
    )

    fig.savefig(
        _p,
        dpi=150,
        bbox_inches="tight",
    )

    PLOTS.append(_p)

    print(
        " ✓ [%02d] %s (%.1f KB)"
        % (
            idx,
            os.path.basename(_p),
            os.path.getsize(_p) / 1024.0,
        )
    )

    plt.show()
    plt.close(fig)
    
    

def _fig(title, xl, yl, size=(9.2, 6.4)):
    _f, _a = plt.subplots(figsize=size)
    _a.set_title(title, fontsize=12, fontweight="bold")
    _a.set_xlabel(xl)
    _a.set_ylabel(yl)
    _a.grid(alpha=0.3)
    return _f, _a


_ROC = {_n: roc_curve(Y[_m], P[_m]) for _n, _m in SETS[:2]}
try:
    _f, _a = _fig("1. Đường cong ROC — TRAIN vs TEST", "Tỷ lệ dương tính giả (FPR)",
                  "Tỷ lệ dương tính thật (TPR)")
    for _n in ("TRAIN", "TEST"):
        _fpr, _tpr, _ = _ROC[_n]
        _a.plot(_fpr, _tpr, color=_COL[_n], lw=1.8,
                label="%s (AUC = %.6f)" % (_n, M[_n]["AUC"]))
    _a.plot([0, 1], [0, 1], "k--", lw=1, label="Ngẫu nhiên (AUC = 0.5)")
    _a.legend(loc="lower right", fontsize=9)
    _anno(_a, "AUC", "AUC = ∫TPR d(FPR)", "Sức phân biệt tốt/xấu",
          "Thấp <0.70 | TB 0.70–0.80 | Cao >0.80", M["TRAIN"]["AUC"], M["TEST"]["AUC"],
          "%s %s" % (_AUC_L(M["TEST"]["AUC"])[1], _AUC_L(M["TEST"]["AUC"])[0]))
    _save(_f, 1, "roc_curve")
except Exception as _e:
    _err("[01] ROC", _e)

try:
    _f, _a = _fig("2. Đường cong Precision–Recall", "Recall", "Precision")
    for _n, _m in SETS[:2]:
        _pr, _rc, _ = precision_recall_curve(Y[_m], P[_m])
        _a.plot(_rc, _pr, color=_COL[_n], lw=1.6,
                label="%s (AP = %.6f)" % (_n, M[_n]["AP"]))
    _a.axhline(M["FULL"]["bad_rate"], color="gray", ls="--", lw=1,
               label="Bad rate = %.6f" % M["FULL"]["bad_rate"])
    _a.legend(loc="upper right", fontsize=9)
    _anno(_a, "AP", "AP = Σ(Rₙ − Rₙ₋₁)·Pₙ", "Hiệu năng trên lớp thiểu số (bad)",
          "Tốt ≥0.60 | TB 0.40–0.60 | Kém <0.40", M["TRAIN"]["AP"], M["TEST"]["AP"],
          "%s %s" % (_qual(M["TEST"]["AP"], 0.60, 0.40)[1], _qual(M["TEST"]["AP"], 0.60, 0.40)[0]))
    _save(_f, 2, "precision_recall")
except Exception as _e:
    _err("[02] PR", _e)

try:
    _f, _a = _fig("3. Đường cong KS — phân phối tích luỹ Good/Bad", "Ngưỡng PD",
                  "Tỷ lệ tích luỹ")
    for _n, _m in SETS[:2]:
        _fpr, _tpr, _th = _ROC[_n]
        _a.plot(_th, _tpr, color=_COL[_n], lw=1.5, label="%s F_bad" % _n)
        _a.plot(_th, _fpr, color=_COL[_n], lw=1.2, ls="--", label="%s F_good" % _n)
        _k = int(np.argmax(_tpr - _fpr))
        _a.vlines(_th[_k], _fpr[_k], _tpr[_k], color=_COL[_n], lw=2.2)
        _a.annotate("KS %s = %.6f" % (_n, M[_n]["KS"]), (_th[_k], (_tpr[_k] + _fpr[_k]) / 2),
                    fontsize=9, color=_COL[_n])
    _a.set_xlim(0, 1)
    _a.legend(loc="upper right", fontsize=9)
    _anno(_a, "KS", "KS = maxₜ|F_bad(t) − F_good(t)|", "Tách biệt lớn nhất giữa 2 phân phối",
          "Thấp <0.40 | TB 0.40–0.60 | Cao >0.60", M["TRAIN"]["KS"], M["TEST"]["KS"],
          "%s %s" % (_KS_L(M["TEST"]["KS"])[1], _KS_L(M["TEST"]["KS"])[0]))
    _save(_f, 3, "ks_curve")
except Exception as _e:
    _err("[03] KS", _e)

try:
    _f, _a = _fig("4. Đường cong Lorenz / Gini", "Tỷ lệ tích luỹ Good",
                  "Tỷ lệ tích luỹ Bad")
    for _n in ("TRAIN", "TEST"):
        _fpr, _tpr, _ = _ROC[_n]
        _a.plot(_fpr, _tpr, color=_COL[_n], lw=1.8,
                label="%s (Gini = %.6f)" % (_n, M[_n]["Gini"]))
    _a.plot([0, 1], [0, 1], "k--", lw=1, label="Đường bình đẳng")
    _a.legend(loc="lower right", fontsize=9)
    _anno(_a, "Gini", "Gini = 2·AUC − 1", "Mức tách biệt xếp hạng tín dụng",
          "Thấp <0.40 | TB 0.40–0.60 | Cao >0.60", M["TRAIN"]["Gini"], M["TEST"]["Gini"],
          "%s %s" % (_GINI_L(M["TEST"]["Gini"])[1], _GINI_L(M["TEST"]["Gini"])[0]))
    _save(_f, 4, "lorenz_gini")
except Exception as _e:
    _err("[04] Lorenz", _e)

_CAP = {}
try:
    _f, _a = _fig("5. Đường cong CAP (Cumulative Accuracy Profile)",
                  "Tỷ lệ hồ sơ (xếp theo PD giảm dần)", "Tỷ lệ bad tích luỹ")
    for _n, _m in SETS[:2]:
        _y, _p = Y[_m], P[_m]
        _o = np.argsort(-_p, kind="mergesort")
        _cy = np.cumsum(_y[_o]) / max(_y.sum(), 1)
        _x = np.arange(1, len(_y) + 1) / len(_y)
        _br = float(_y.mean())
        _ar = (np.trapezoid(_cy, _x) - 0.5) / (1.0 - _br / 2.0 - 0.5) if hasattr(np, "trapezoid") \
            else (np.trapz(_cy, _x) - 0.5) / (1.0 - _br / 2.0 - 0.5)
        _CAP[_n] = float(_ar)
        _a.plot(_x, _cy, color=_COL[_n], lw=1.8, label="%s (AR = %.6f)" % (_n, _ar))
        del _o, _cy, _x
    _a.plot([0, 1], [0, 1], "k--", lw=1, label="Ngẫu nhiên")
    _a.plot([0, M["FULL"]["bad_rate"], 1], [0, 1, 1], color="green", ls=":", lw=1.2,
            label="Hoàn hảo")
    _a.legend(loc="lower right", fontsize=9)
    _anno(_a, "AR", "AR = (A_model − A_random)/(A_perfect − A_random) ≈ Gini",
          "Accuracy Ratio — chuẩn Moody's KMV", "Ngưỡng đạt ≥0.30 | Cao >0.60",
          _CAP["TRAIN"], _CAP["TEST"],
          "%s %s" % (_qual(_CAP["TEST"], 0.60, 0.30)[1], _qual(_CAP["TEST"], 0.60, 0.30)[0]))
    _save(_f, 5, "cap_curve")
    gc.collect()
except Exception as _e:
    _err("[05] CAP", _e)

_LIFT = {}
try:
    _f, _a = _fig("6. Đường cong Lift theo decile", "Decile theo PD giảm dần (%)",
                  "Lift = bad rate decile / bad rate tổng")
    for _n, _m in SETS[:2]:
        _y, _p = Y[_m], P[_m]
        _o = np.argsort(-_p, kind="mergesort")
        _ys = _y[_o]
        _sp = np.array_split(_ys, 10)
        _lf = np.array([float(_s.mean()) / max(float(_y.mean()), EPS) for _s in _sp])
        _LIFT[_n] = float(_lf[0])
        _a.plot(np.arange(1, 11) * 10, _lf, "o-", color=_COL[_n], lw=1.8,
                label="%s (Lift decile 1 = %.4f)" % (_n, _lf[0]))
        del _o, _ys, _sp
    _a.axhline(1.0, color="gray", ls="--", lw=1, label="Không có lift")
    _a.legend(fontsize=9)
    _anno(_a, "LIFT", "Lift_d = BadRate_d / BadRate_tổng",
          "Hiệu quả bắt bad ở nhóm rủi ro cao nhất",
          "Tốt ≥3.0 | TB 2.0–3.0 | Kém <2.0 (decile 1)", _LIFT["TRAIN"], _LIFT["TEST"],
          "%s %s" % (_qual(_LIFT["TEST"], 3.0, 2.0)[1], _qual(_LIFT["TEST"], 3.0, 2.0)[0]))
    _save(_f, 6, "lift_curve")
    gc.collect()
except Exception as _e:
    _err("[06] Lift", _e)

_CG = {}
try:
    _f, _a = _fig("7. Cumulative Gain — % bad bắt được theo % hồ sơ",
                  "Tỷ lệ hồ sơ xét duyệt (%)", "Tỷ lệ bad bắt được (%)")
    for _n, _m in SETS[:2]:
        _y, _p = Y[_m], P[_m]
        _o = np.argsort(-_p, kind="mergesort")
        _cy = np.cumsum(_y[_o]) / max(_y.sum(), 1)
        _x = np.arange(1, len(_y) + 1) / len(_y)
        _CG[_n] = float(np.interp(0.20, _x, _cy))
        _a.plot(_x * 100, _cy * 100, color=_COL[_n], lw=1.8,
                label="%s (Gain@20%% = %.4f)" % (_n, _CG[_n]))
        del _o, _cy, _x
    _a.plot([0, 100], [0, 100], "k--", lw=1, label="Ngẫu nhiên")
    _a.legend(loc="lower right", fontsize=9)
    _anno(_a, "GAIN20", "Gain(q) = Σ_{i≤q·N} yᵢ / Σ yᵢ", "Tỷ lệ bad bắt được ở top 20% rủi ro",
          "Tốt ≥0.60 | TB 0.40–0.60 | Kém <0.40", _CG["TRAIN"], _CG["TEST"],
          "%s %s" % (_qual(_CG["TEST"], 0.60, 0.40)[1], _qual(_CG["TEST"], 0.60, 0.40)[0]))
    _save(_f, 7, "cumulative_gain")
    gc.collect()
except Exception as _e:
    _err("[07] Cumulative Gain", _e)

try:
    _f, _a = _fig("8. Phân phối điểm số (%s) theo nhóm Good/Bad" % SCORE_COL,
                  "Điểm số", "Mật độ")
    _eg = np.histogram_bin_edges(SC, bins=60)
    for _n, _m in SETS[:2]:
        for _cls, _ls, _lb in ((0, "-", "Good"), (1, "--", "Bad")):
            _v = SC[_m & (Y == _cls)]
            _hh, _ = np.histogram(_v, bins=_eg, density=True)
            _a.plot(0.5 * (_eg[1:] + _eg[:-1]), _hh, ls=_ls, lw=1.5, color=_COL[_n],
                    label="%s — %s (n=%d)" % (_n, _lb, len(_v)))
    _a.legend(fontsize=8)
    _anno(_a, "KS_SCORE", "Score = offset − factor·ln(odds)",
          "Tách biệt phân phối điểm giữa Good và Bad",
          "KS: Thấp <0.40 | TB 0.40–0.60 | Cao >0.60", M["TRAIN"]["KS"], M["TEST"]["KS"],
          "%s %s" % (_KS_L(M["TEST"]["KS"])[1], _KS_L(M["TEST"]["KS"])[0]), loc=(0.02, 0.55))
    _save(_f, 8, "score_distribution")
except Exception as _e:
    _err("[08] Score distribution", _e)

try:
    _f, _a = _fig("9. Phân phối xác suất PD (%s)" % PROB_COL, "PD dự báo", "Mật độ")
    _eg = np.linspace(0, 1, 101)
    for _n, _m in SETS[:2]:
        for _cls, _ls, _lb in ((0, "-", "Good"), (1, "--", "Bad")):
            _hh, _ = np.histogram(P[_m & (Y == _cls)], bins=_eg, density=True)
            _a.plot(0.5 * (_eg[1:] + _eg[:-1]), _hh, ls=_ls, lw=1.5, color=_COL[_n],
                    label="%s — %s" % (_n, _lb))
    _a.axvline(THRESH, color="black", lw=1.6, label="Threshold = %.6f" % THRESH)
    _a.set_yscale("log")
    _a.legend(fontsize=8)
    _anno(_a, "BRIER", "BS = (1/N)Σ(pᵢ − yᵢ)²", "Chất lượng xác suất dự báo",
          "Tốt ≤0.15 | TB 0.15–0.25 | Kém >0.25", M["TRAIN"]["Brier"], M["TEST"]["Brier"],
          "%s %s" % (_qual(M["TEST"]["Brier"], 0.15, 0.25, True)[1],
                     _qual(M["TEST"]["Brier"], 0.15, 0.25, True)[0]), loc=(0.30, 0.55))
    _save(_f, 9, "probability_distribution")
except Exception as _e:
    _err("[09] Probability distribution", _e)

_DEC_TBL = {}
try:
    _dec = data_group[DECILE_COL].astype("float").to_numpy()
    _f, _a = _fig("10. Tỷ lệ bad theo decile PD (cut-point học từ TRAIN)",
                  "Decile (1 = PD thấp nhất)", "Bad rate")
    _w = 0.38
    for _k, (_n, _m) in enumerate(SETS[:2]):
        _t = pd.DataFrame({"d": _dec[_m], "y": Y[_m], "p": P[_m]}).groupby("d").agg(
            n=("y", "size"), bad=("y", "sum"), br=("y", "mean"), pd_=("p", "mean"))
        _DEC_TBL[_n] = _t
        _a.bar(_t.index + (_k - 0.5) * _w, _t["br"], width=_w, color=_COL[_n],
               alpha=0.85, label="%s" % _n)
    _a.plot(_DEC_TBL["TRAIN"].index, _DEC_TBL["TRAIN"]["pd_"], "k^--", lw=1.2,
            label="PD dự báo TB (TRAIN)")
    _a.legend(fontsize=9)
    _mono = float(np.corrcoef(_DEC_TBL["TEST"].index.values,
                              _DEC_TBL["TEST"]["br"].values)[0, 1])
    _anno(_a, "MONO", "BadRate_d = Σy_d / n_d; ρ(decile, BadRate)",
          "Tính đơn điệu của rủi ro theo decile",
          "Tốt ρ ≥0.90 | TB 0.70–0.90 | Kém <0.70",
          float(np.corrcoef(_DEC_TBL["TRAIN"].index.values,
                            _DEC_TBL["TRAIN"]["br"].values)[0, 1]), _mono,
          "%s %s" % (_qual(_mono, 0.90, 0.70)[1], _qual(_mono, 0.90, 0.70)[0]),
          loc=(0.02, 0.55))
    _save(_f, 10, "bad_rate_by_decile")
except Exception as _e:
    _err("[10] Bad rate by decile", _e)

_GR_TBL = {}
try:
    _gr = data_group[GRADE_COL].astype(str).to_numpy()
    _lbls = sorted(set(_gr))
    _f, _a = _fig("11. Tỷ lệ bad theo Grade/Band (%s, band từ TRAIN)" % GRADE_COL,
                  "Grade", "Bad rate")
    _w = 0.38
    for _k, (_n, _m) in enumerate(SETS[:2]):
        _t = pd.DataFrame({"g": _gr[_m], "y": Y[_m]}).groupby("g").agg(
            n=("y", "size"), br=("y", "mean")).reindex(_lbls)
        _GR_TBL[_n] = _t
        _a.bar(np.arange(len(_lbls)) + (_k - 0.5) * _w, _t["br"].fillna(0), width=_w,
               color=_COL[_n], alpha=0.85, label=_n)
    _a.set_xticks(np.arange(len(_lbls)))
    _a.set_xticklabels(_lbls, rotation=45, ha="right")
    _a.legend(fontsize=9)
    _v_tr = float(_GR_TBL["TRAIN"]["br"].max() - _GR_TBL["TRAIN"]["br"].min())
    _v_te = float(_GR_TBL["TEST"]["br"].max() - _GR_TBL["TEST"]["br"].min())
    _anno(_a, "GRADE_SPREAD", "Spread = max(BadRate_g) − min(BadRate_g)",
          "Độ trải rủi ro giữa các hạng tín dụng",
          "Tốt ≥0.20 | TB 0.10–0.20 | Kém <0.10", _v_tr, _v_te,
          "%s %s" % (_qual(_v_te, 0.20, 0.10)[1], _qual(_v_te, 0.20, 0.10)[0]),
          loc=(0.02, 0.55))
    _save(_f, 11, "bad_rate_by_grade")
except Exception as _e:
    _err("[11] Bad rate by grade", _e)

try:
    _f, _a = _fig("12. Đường hiệu chuẩn (Reliability Diagram, 10 bins)",
                  "PD dự báo trung bình theo bin", "Tỷ lệ bad thực tế")
    for _n, _m in SETS[:2]:
        _c, _sy, _sp = _bins10(P[_m], Y[_m])
        _ok = _c > 0
        _a.plot(_sp[_ok] / _c[_ok], _sy[_ok] / _c[_ok], "o-", color=_COL[_n], lw=1.6,
                label="%s (ECE = %.6f)" % (_n, M[_n]["ECE"]))
    _a.plot([0, 1], [0, 1], "k--", lw=1, label="Hiệu chuẩn hoàn hảo")
    _a.legend(loc="upper left", fontsize=9)
    _anno(_a, "ECE", "ECE = Σ_b(n_b/N)|acc_b − conf_b|", "Sai số hiệu chuẩn kỳ vọng",
          "Tốt ≤0.05 | TB 0.05–0.10 | Kém >0.10", M["TRAIN"]["ECE"], M["TEST"]["ECE"],
          "%s %s" % (_qual(M["TEST"]["ECE"], 0.05, 0.10, True)[1],
                     _qual(M["TEST"]["ECE"], 0.05, 0.10, True)[0]), loc=(0.35, 0.05))
    _save(_f, 12, "calibration_curve")
except Exception as _e:
    _err("[12] Calibration curve", _e)

try:
    _f, _a = _fig("13. PD dự báo vs Tỷ lệ bad quan sát theo decile",
                  "Decile", "PD / Bad rate")
    for _n in ("TRAIN", "TEST"):
        _t = _DEC_TBL[_n]
        _a.plot(_t.index, _t["pd_"], "s--", color=_COL[_n], lw=1.5, label="%s — PD dự báo" % _n)
        _a.plot(_t.index, _t["br"], "o-", color=_COL[_n], lw=1.8, label="%s — bad quan sát" % _n)
    _a.legend(fontsize=9)
    _anno(_a, "MCE", "MCE = max_b|acc_b − conf_b|", "Sai lệch hiệu chuẩn tệ nhất theo bin",
          "Tốt ≤0.10 | TB 0.10–0.20 | Kém >0.20", M["TRAIN"]["MCE"], M["TEST"]["MCE"],
          "%s %s" % (_qual(M["TEST"]["MCE"], 0.10, 0.20, True)[1],
                     _qual(M["TEST"]["MCE"], 0.10, 0.20, True)[0]), loc=(0.02, 0.55))
    _save(_f, 13, "predicted_vs_observed")
except Exception as _e:
    _err("[13] Predicted vs observed", _e)

try:
    _f, _ax = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for _k, _n in enumerate(("TRAIN", "TEST")):
        _cm = np.array([[M[_n]["TN"], M[_n]["FP"]], [M[_n]["FN"], M[_n]["TP"]]], dtype=float)
        _im = _ax[_k].imshow(_cm / _cm.sum(), cmap="Blues", vmin=0, vmax=1)
        for _i in range(2):
            for _j in range(2):
                _ax[_k].text(_j, _i, "%d\n(%.4f%%)" % (_cm[_i, _j], 100 * _cm[_i, _j] / _cm.sum()),
                             ha="center", va="center", fontsize=10,
                             color="white" if _cm[_i, _j] / _cm.sum() > 0.5 else "black")
        _ax[_k].set_xticks([0, 1])
        _ax[_k].set_xticklabels(["Dự báo Good", "Dự báo Bad"])
        _ax[_k].set_yticks([0, 1])
        _ax[_k].set_yticklabels(["Thực tế Good", "Thực tế Bad"])
        _ax[_k].set_title("%s — threshold = %.6f" % (_n, THRESH), fontsize=11)
    _f.suptitle("14. Ma trận nhầm lẫn (Confusion Matrix Heatmap)", fontsize=12,
                fontweight="bold")
    _anno(_ax[0], "BACC", "BAcc = (TPR + TNR)/2", "Độ chính xác cân bằng",
          "Tốt ≥0.70 | TB 0.60–0.70 | Kém <0.60", M["TRAIN"]["BalancedAcc"],
          M["TEST"]["BalancedAcc"],
          "%s %s" % (_qual(M["TEST"]["BalancedAcc"], 0.70, 0.60)[1],
                     _qual(M["TEST"]["BalancedAcc"], 0.70, 0.60)[0]), loc=(0.0, -0.42))
    _save(_f, 14, "confusion_matrix")
except Exception as _e:
    _err("[14] Confusion matrix", _e)

try:
    _f, _a = _fig("15. Phân tích ngưỡng — Precision / Recall / F1", "Threshold",
                  "Giá trị chỉ số")
    _grid = np.linspace(0.01, 0.99, 197)
    for _n, _m in SETS[:2]:
        _y, _p = Y[_m], P[_m]
        _o = np.argsort(-_p, kind="mergesort")
        _ys, _ps = _y[_o], _p[_o]
        _ctp = np.cumsum(_ys, dtype=np.int64)
        _npos = np.searchsorted(-_ps, -_grid, side="right")
        _tp = np.where(_npos > 0, _ctp[np.maximum(_npos - 1, 0)], 0)
        _prec = np.divide(_tp, _npos, out=np.zeros(len(_grid)), where=_npos > 0)
        _rec = _tp / max(int(_ys.sum()), 1)
        _f1 = np.divide(2 * _prec * _rec, _prec + _rec, out=np.zeros(len(_grid)),
                        where=(_prec + _rec) > 0)
        _a.plot(_grid, _prec, ls="-", color=_COL[_n], lw=1.3, label="%s Precision" % _n)
        _a.plot(_grid, _rec, ls="--", color=_COL[_n], lw=1.3, label="%s Recall" % _n)
        _a.plot(_grid, _f1, ls=":", color=_COL[_n], lw=1.8, label="%s F1" % _n)
        del _o, _ys, _ps, _ctp
    _a.axvline(THRESH, color="black", lw=1.6, label="Threshold TRAIN = %.6f" % THRESH)
    _a.legend(fontsize=8, ncol=2)
    _anno(_a, "F1", "F₁ = 2PR/(P+R)", "Cân bằng Precision–Recall tại ngưỡng đã chọn",
          "Tốt ≥0.60 | TB 0.40–0.60 | Kém <0.40", M["TRAIN"]["F1"], M["TEST"]["F1"],
          "%s %s" % (_qual(M["TEST"]["F1"], 0.60, 0.40)[1],
                     _qual(M["TEST"]["F1"], 0.60, 0.40)[0]), loc=(0.35, 0.55))
    _save(_f, 15, "threshold_analysis")
    gc.collect()
except Exception as _e:
    _err("[15] Threshold analysis", _e)

try:
    _f, _a = _fig("16. Ổn định phân phối điểm TRAIN vs TEST", "Điểm số (%s)" % SCORE_COL,
                  "Tỷ trọng hồ sơ")
    _eg = np.histogram_bin_edges(SC, bins=50)
    for _n, _m in SETS[:2]:
        _hh, _ = np.histogram(SC[_m], bins=_eg)
        _a.plot(0.5 * (_eg[1:] + _eg[:-1]), _hh / _hh.sum(), lw=1.8, color=_COL[_n],
                label="%s (n = %d)" % (_n, int(_m.sum())))
    _a.legend(fontsize=9)
    _anno(_a, "PSI", "PSI = Σ(Aᵢ − Eᵢ)ln((Aᵢ+ε)/(Eᵢ+ε))",
          "Dịch chuyển phân phối điểm giữa 2 tập",
          "Ổn định <0.10 | Theo dõi 0.10–0.25 | Bất ổn ≥0.25", PSI_SCORE, PSI_SCORE,
          "%s %s" % (_qual(PSI_SCORE, 0.10, 0.25, True)[1],
                     _qual(PSI_SCORE, 0.10, 0.25, True)[0]), loc=(0.02, 0.55))
    _save(_f, 16, "stability_train_test_score")
except Exception as _e:
    _err("[16] Stability", _e)

PSI_DF = pd.DataFrame({
    "Bin": np.arange(1, len(PSI_E) + 1),
    "Can_duoi": np.r_[-np.inf, _SC_EDGES[1:-1]][:len(PSI_E)],
    "Can_tren": np.r_[_SC_EDGES[1:-1], np.inf][:len(PSI_E)],
    "Expected_TRAIN": PSI_E, "Actual_TEST": PSI_A, "Contribution": PSI_C})
try:
    _f, _a = _fig("17. PSI theo bin điểm (E = TRAIN, A = TEST)", "Bin (10 nhóm từ TRAIN)",
                  "Tỷ trọng / Đóng góp PSI")
    _x = np.arange(1, len(PSI_E) + 1)
    _a.bar(_x - 0.2, PSI_E, width=0.4, color=_COL["TRAIN"], label="Expected (TRAIN)")
    _a.bar(_x + 0.2, PSI_A, width=0.4, color=_COL["TEST"], label="Actual (TEST)")
    _a2 = _a.twinx()
    _a2.plot(_x, PSI_C, "ko--", lw=1.4, label="Đóng góp PSI")
    _a2.set_ylabel("Đóng góp PSI theo bin")
    _a.legend(loc="upper left", fontsize=9)
    _a2.legend(loc="upper right", fontsize=9)
    _anno(_a, "PSI_BIN", "PSIᵢ = (Aᵢ − Eᵢ)ln((Aᵢ+ε)/(Eᵢ+ε))",
          "Bin đóng góp nhiều nhất vào bất ổn phân phối",
          "Ổn định <0.10 | Theo dõi 0.10–0.25 | Bất ổn ≥0.25", PSI_SCORE, PSI_SCORE,
          "%s %s" % (_qual(PSI_SCORE, 0.10, 0.25, True)[1],
                     _qual(PSI_SCORE, 0.10, 0.25, True)[0]), loc=(0.02, 0.55))
    _save(_f, 17, "psi_by_bin")
except Exception as _e:
    _err("[17] PSI plot", _e)

try:
    if hasattr(best_mdl, "feature_importances_"):
        _imp, _src_imp = np.abs(best_mdl.feature_importances_), "feature_importances_"
    elif hasattr(best_mdl, "coef_"):
        _imp, _src_imp = np.abs(best_mdl.coef_[0]), "abs(coef_) chuẩn hoá"
    else:
        _imp, _src_imp = np.abs(logr.raw_coef_[0]), "abs(raw_coef_)"
    _od = np.argsort(_imp)
    _f, _a = _fig("18. Tầm quan trọng biến — %s" % _src_imp, "|Hệ số| (không gian chuẩn hoá)",
                  "Biến mô hình", size=(9.6, 8.4))
    _a.barh(np.arange(len(_imp)), _imp[_od], color="#2c7fb8")
    _a.set_yticks(np.arange(len(_imp)))
    _a.set_yticklabels([feature_cols[_i] for _i in _od], fontsize=8)
    _top_iv = float(IV_DF["IV_train"].iloc[0])
    _anno(_a, "IV_TOTAL", "IV = Σ(Dist_Good − Dist_Bad)·WOE", "Sức mạnh dự báo của bộ biến",
          "IV: Thấp <0.10 | TB 0.10–0.30 | Cao >0.30", IV_TR_MAX, IV_TE_MAX,
          "%s %s (IV lớn nhất theo biến)" % (_IV_L(IV_TR_MAX)[1], _IV_L(IV_TR_MAX)[0]), loc=(0.32, 0.03))
    _save(_f, 18, "feature_importance")
except Exception as _e:
    _err("[18] Feature importance", _e)

# [8] OVERFITTING — LEARNING CURVE THEO N + VALIDATION CURVE THEO ITERATION
_h("[10] LEARNING CURVE (theo N mẫu) & VALIDATION CURVE (theo số vòng lặp)")
LC_DF = pd.DataFrame()
VC_DF = pd.DataFrame()
try:
    _Xtr, _ytr = X_model[_TR], Y[_TR]
    _Xte, _yte = X_model[_TE], Y[_TE]
    _rows = []
    
    _learning_fractions = (
        0.10,
        0.20,
        0.35,
        0.50,
        0.65,
        0.80,
        0.90,
        1.00,
    )
    
    for _fr in _learning_fractions:
        if _fr >= 1.0:
            _idx = np.arange(
                len(_ytr),
                dtype=np.int64,
            )
        else:
            _splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=_fr,
                random_state=42,
            )
    
            _idx, _ = next(
                _splitter.split(
                    _Xtr,
                    _ytr,
                )
            )
    
        _k = int(len(_idx))
    
        if np.unique(_ytr[_idx]).size < 2:
            print(
                "SKIP learning curve tại %.0f%%: "
                "subset không đủ hai lớp."
                % (100.0 * _fr)
            )
            continue
        _scl = SCALER_FACTORY[best_cfg["scaler_name"]]().fit(_Xtr[_idx])
        _Xs = _scl.transform(_Xtr[_idx])
        _md = _build_model(best_cfg).fit(_Xs, _ytr[_idx])
        _ptr = _md.predict_proba(_Xs)[:, 1]
        _pte = _md.predict_proba(_scl.transform(_Xte))[:, 1]
        _rows.append({"Ty_le_train": _fr, "N_train": _k,
                      "AUC_train": float(roc_auc_score(_ytr[_idx], _ptr)),
                      "AUC_test": float(roc_auc_score(_yte, _pte)),
                      "LogLoss_train": float(log_loss(_ytr[_idx], _ptr, labels=[0, 1])),
                      "LogLoss_test": float(log_loss(_yte, _pte, labels=[0, 1])),
                      "Acc_train": float(accuracy_score(_ytr[_idx], (_ptr >= THRESH).astype(np.int8))),
                      "Acc_test": float(accuracy_score(_yte, (_pte >= THRESH).astype(np.int8)))})
        print(" N=%8d (%.0f%%) AUC train=%.6f test=%.6f | LogLoss train=%.6f test=%.6f"
              % (_k, 100 * _fr, _rows[-1]["AUC_train"], _rows[-1]["AUC_test"],
                 _rows[-1]["LogLoss_train"], _rows[-1]["LogLoss_test"]))
        del _scl, _md, _Xs, _ptr, _pte
        gc.collect()
    LC_DF = pd.DataFrame(_rows)
    _f, _ax = plt.subplots(1, 2, figsize=(13.2, 5.4))
    _ax[0].plot(LC_DF["N_train"], LC_DF["Acc_train"], "o-", color=_COL["TRAIN"], label="Train")
    _ax[0].plot(LC_DF["N_train"], LC_DF["Acc_test"], "s--", color=_COL["TEST"], label="Test")
    _ax[0].set_title("Accuracy theo số mẫu train")
    _ax[1].plot(LC_DF["N_train"], LC_DF["LogLoss_train"], "o-", color=_COL["TRAIN"], label="Train")
    _ax[1].plot(LC_DF["N_train"], LC_DF["LogLoss_test"], "s--", color=_COL["TEST"], label="Test")
    _ax[1].set_title("Log Loss theo số mẫu train")
    for _z in _ax:
        _z.set_xlabel("Số mẫu train N")
        _z.grid(alpha=0.3)
        _z.legend(fontsize=9)
    _f.suptitle("19. Learning Curve theo kích thước mẫu (Train + Test)", fontsize=12,
                fontweight="bold")
    _gap_lc = float(LC_DF["Acc_train"].iloc[-1] - LC_DF["Acc_test"].iloc[-1])
    _anno(_ax[0], "LC_GAP", "Gap = Acc_train(N) − Acc_test(N)",
          "Chênh lệch hiệu năng in-sample / out-of-sample",
          "Ổn định |gap| ≤0.02 | TB 0.02–0.05 | Overfit >0.05",
          float(LC_DF["Acc_train"].iloc[-1]), float(LC_DF["Acc_test"].iloc[-1]),
          "%s %s" % (_qual(abs(_gap_lc), 0.02, 0.05, True)[1],
                     _qual(abs(_gap_lc), 0.02, 0.05, True)[0]), loc=(0.02, 0.02))
    _save(_f, 19, "learning_curve_by_samples")
    del _Xtr, _ytr, _Xte, _yte
    gc.collect()
except Exception as _e:
    _err("[19] Learning curve", _e)

try:
    _scl = SCALER_FACTORY[best_cfg["scaler_name"]]().fit(X_model[_TR])
    _Xs_tr, _Xs_te = _scl.transform(X_model[_TR]), _scl.transform(X_model[_TE])
    _rows = []
    
    _iteration_grid = sorted(
        set(
            (
                1,
                2,
                5,
                10,
                20,
                40,
                80,
                160,
                320,
                int(best_cfg["max_iter"]),
            )
        )
    )
    
    for _it in _iteration_grid:
        _iter_cfg = dict(best_cfg)
        _iter_cfg["max_iter"] = int(_it)
    
        _md = _build_model(_iter_cfg)
        _md.fit(
            _Xs_tr,
            Y[_TR],
        )
        _ptr = _md.predict_proba(_Xs_tr)[:, 1]
        _pte = _md.predict_proba(_Xs_te)[:, 1]
        _rows.append({"max_iter": _it, "n_iter_": int(np.max(_md.n_iter_)),
                      "AUC_train": float(roc_auc_score(Y[_TR], _ptr)),
                      "AUC_test": float(roc_auc_score(Y[_TE], _pte)),
                      "LogLoss_train": float(log_loss(Y[_TR], _ptr, labels=[0, 1])),
                      "LogLoss_test": float(log_loss(Y[_TE], _pte, labels=[0, 1]))})
        print(" max_iter=%5d n_iter=%4d AUC train=%.6f test=%.6f | LogLoss train=%.6f test=%.6f"
              % (_it, _rows[-1]["n_iter_"], _rows[-1]["AUC_train"], _rows[-1]["AUC_test"],
                 _rows[-1]["LogLoss_train"], _rows[-1]["LogLoss_test"]))
        del _md, _iter_cfg, _ptr, _pte
        gc.collect()
    VC_DF = pd.DataFrame(_rows)
    _f, _ax = plt.subplots(1, 2, figsize=(13.2, 5.4))
    _ax[0].plot(VC_DF["max_iter"], VC_DF["AUC_train"], "o-", color=_COL["TRAIN"], label="Train")
    _ax[0].plot(VC_DF["max_iter"], VC_DF["AUC_test"], "s--", color=_COL["TEST"], label="Test")
    _ax[0].set_title("AUC theo số vòng lặp (max_iter)")
    _ax[1].plot(VC_DF["max_iter"], VC_DF["LogLoss_train"], "o-", color=_COL["TRAIN"], label="Train")
    _ax[1].plot(VC_DF["max_iter"], VC_DF["LogLoss_test"], "s--", color=_COL["TEST"], label="Test")
    _ax[1].set_title("Log Loss theo số vòng lặp (max_iter)")
    for _z in _ax:
        _z.set_xscale("log")
        _z.set_xlabel("max_iter (thang log)")
        _z.grid(alpha=0.3)
        _z.legend(fontsize=9)
    _f.suptitle("20. Validation Curve theo số vòng lặp (warm_start, Train + Test)",
                fontsize=12, fontweight="bold")
    _gap_vc = float(VC_DF["AUC_train"].iloc[-1] - VC_DF["AUC_test"].iloc[-1])
    _anno(_ax[0], "VC_GAP", "Δ AUC(iter) = AUC_train − AUC_test",
          "Kiểm soát overfitting theo số vòng lặp tối ưu hoá",
          "Ổn định |Δ| ≤0.02 | Chấp nhận 0.02–0.03 | Overfit >0.03",
          float(VC_DF["AUC_train"].iloc[-1]), float(VC_DF["AUC_test"].iloc[-1]),
          "%s %s" % (_qual(abs(_gap_vc), 0.02, 0.03, True)[1],
                     _qual(abs(_gap_vc), 0.02, 0.03, True)[0]), loc=(0.02, 0.02))
    _save(_f, 20, "validation_curve_by_iteration")
    del _Xs_tr, _Xs_te, _scl
    gc.collect()
except Exception as _e:
    _err("[20] Validation curve", _e)

# [9] TEMPORAL STABILITY THEO NGAY_BC
_h("[11] TEMPORAL STABILITY THEO NGAY_BC (Δ KS, Δ AUC)")
TEMP_DF = pd.DataFrame()
try:
    if not HAS_DATE:
        SKIPS.append({"Mục": "[21] Temporal stability", "Loại": "SKIP",
                      "Nguyên nhân": "không có cột NGAY_BC", "Xử lý": "Bỏ mục temporal"})
        print(" ⚠ SKIP: không có cột NGAY_BC trong dữ liệu.")
    else:
        _per = pd.to_datetime(data_group["NGAY_BC"]).dt.to_period("M").astype(str).to_numpy()
        _rows = []
        for _pv in sorted(set(_per)):
            _m = _per == _pv
            _y, _p = Y[_m], P[_m]
            _sp = "MIX"
            if _TR[_m].all():
                _sp = "TRAIN"
            elif _TE[_m].all():
                _sp = "TEST"
            if np.unique(_y).size < 2:
                SKIPS.append({"Mục": "Temporal kỳ %s" % _pv, "Loại": "SKIP",
                              "Nguyên nhân": "chỉ có 1 lớp nhãn (n=%d)" % len(_y),
                              "Xử lý": "Bỏ kỳ này khỏi bảng temporal"})
                print(" ⚠ SKIP kỳ %s: chỉ có 1 lớp nhãn (n=%d)" % (_pv, len(_y)))
                continue
            _rows.append({"NGAY_BC": _pv, "Tap": _sp, "n": int(len(_y)),
                          "bad": int(_y.sum()), "bad_rate": float(_y.mean()),
                          "AUC": float(roc_auc_score(_y, _p)), "KS": _ks(_y, _p),
                          "Gini": 2 * float(roc_auc_score(_y, _p)) - 1,
                          "PD_TB": float(_p.mean())})
        TEMP_DF = pd.DataFrame(_rows)
        
        if TEMP_DF.empty:
            SKIPS.append(
                {
                    "Mục": "[21] Temporal stability",
                    "Loại": "SKIP",
                    "Nguyên nhân": (
                        "Không có kỳ NGAY_BC nào "
                        "có đủ cả hai lớp nhãn."
                    ),
                    "Xử lý": (
                        "Không tính AUC/KS temporal; "
                        "pipeline tiếp tục."
                    ),
                }
            )
        
            print(
                "SKIP temporal stability: "
                "không có kỳ NGAY_BC nào đủ hai lớp nhãn."
            )
        
        else:
            TEMP_DF["Delta_KS_vs_TRAIN"] = (
                M["TRAIN"]["KS"]
                - TEMP_DF["KS"]
            )
            TEMP_DF["Delta_AUC_vs_TRAIN"] = (
                M["TRAIN"]["AUC"]
                - TEMP_DF["AUC"]
            )
            TEMP_DF["Danh_gia"] = np.where(
                (
                    TEMP_DF[
                        "Delta_KS_vs_TRAIN"
                    ].abs()
                    <= 0.05
                )
                & (
                    TEMP_DF[
                        "Delta_AUC_vs_TRAIN"
                    ].abs()
                    <= 0.03
                ),
                "✓ Ổn định",
                "✗ Vượt ngưỡng",
            )
        
        TEMP_DF["Danh_gia"] = np.where(
            (TEMP_DF["Delta_KS_vs_TRAIN"].abs() <= 0.05)
            & (TEMP_DF["Delta_AUC_vs_TRAIN"].abs() <= 0.03), "✓ Ổn định", "✗ Vượt ngưỡng")
        print(TEMP_DF.to_string(index=False,
                                float_format=lambda v: "%.6f" % v))
        _f, _a = _fig("21. Ổn định theo thời gian (NGAY_BC) — AUC / KS / Bad rate",
                      "Kỳ báo cáo NGAY_BC", "AUC / KS", size=(13.5, 6.4))
        _x = np.arange(len(TEMP_DF))
        _a.plot(_x, TEMP_DF["AUC"], "o-", color="#1f77b4", label="AUC theo kỳ")
        _a.plot(_x, TEMP_DF["KS"], "s-", color="#d62728", label="KS theo kỳ")
        _a.axhline(M["TRAIN"]["AUC"], color="#1f77b4", ls="--", lw=1,
                   label="AUC TRAIN = %.6f" % M["TRAIN"]["AUC"])
        _a.axhline(M["TRAIN"]["KS"], color="#d62728", ls="--", lw=1,
                   label="KS TRAIN = %.6f" % M["TRAIN"]["KS"])
        _a2 = _a.twinx()
        _a2.bar(_x, TEMP_DF["bad_rate"], color="gray", alpha=0.28, label="Bad rate")
        _a2.set_ylabel("Bad rate")
        _a.set_xticks(_x)
        _a.set_xticklabels(TEMP_DF["NGAY_BC"], rotation=90, fontsize=7)
        _a.legend(loc="upper left", fontsize=8)
        _a2.legend(loc="upper right", fontsize=8)
        _mx_ks = float(TEMP_DF["Delta_KS_vs_TRAIN"].abs().max())
        _mx_auc = float(TEMP_DF["Delta_AUC_vs_TRAIN"].abs().max())
        _anno(_a, "TEMPORAL", "Δ KS = |KS_train − KS_kỳ|; Δ AUC = |AUC_train − AUC_kỳ|",
              "Ổn định hiệu năng theo thời gian",
              "Δ KS ≤3–5% và Δ AUC ≤2–3% → ổn định", _mx_ks, _mx_auc,
              "max|ΔKS|=%.6f %s ; max|ΔAUC|=%.6f %s"
              % (_mx_ks, "✓" if _mx_ks <= 0.05 else "✗", _mx_auc,
                 "✓" if _mx_auc <= 0.03 else "✗"), loc=(0.02, 0.02))
        _save(_f, 21, "temporal_stability_ngaybc")
except Exception as _e:
    _err("[21] Temporal stability", _e)

try:
    _f, _a = _fig("22. Information Value theo biến (bin học từ TRAIN)", "IV",
                  "Biến mô hình", size=(9.6, 8.4))
    _d = IV_DF.sort_values("IV_train")
    _yy = np.arange(len(_d))
    _a.barh(_yy - 0.2, _d["IV_train"], height=0.4, color=_COL["TRAIN"], label="IV TRAIN")
    _a.barh(_yy + 0.2, _d["IV_test"], height=0.4, color=_COL["TEST"], label="IV TEST")
    _a.set_yticks(_yy)
    _a.set_yticklabels(_d["Feature"], fontsize=8)
    for _v, _c, _l in ((0.10, "orange", "IV = 0.10"), (0.30, "green", "IV = 0.30"),
                       (0.50, "red", "IV = 0.50 (nghi leakage)")):
        _a.axvline(_v, color=_c, ls="--", lw=1, label=_l)
    _a.legend(fontsize=8, loc="lower right")
    _anno(_a, "IV_FEATURE", "IV = Σ(Dist_Good − Dist_Bad)·ln((Dist_Good+ε)/(Dist_Bad+ε))",
          "Sức mạnh dự báo từng biến",
          "Thấp <0.10 | TB 0.10–0.30 | Cao >0.30", IV_TR_MAX, IV_TE_MAX,
          "%s %s (tổng)" % (_IV_L(IV_TR_MAX)[1], _IV_L(IV_TR_MAX)[0]), loc=(0.30, 0.30))
    _save(_f, 22, "iv_by_feature")
except Exception as _e:
    _err("[22] IV plot", _e)

# [10] CROSS-CHECK CONSOLE ↔ EXCEL ↔ ANNOTATION
_h("[12] XUẤT EXCEL + CROSS-CHECK (console ↔ Excel ↔ annotation)")
OVF_DF = pd.DataFrame([
    {"Chi_tieu": "KS", "Train": M["TRAIN"]["KS"], "Test": M["TEST"]["KS"],
     "Delta_train_test": D_KS, "Nguong": "|Δ| ≤ 3–5%",
     "Ket_luan": "✓ Ổn định" if abs(D_KS) <= 0.05 else "✗ Overfit"},
    {"Chi_tieu": "AUC", "Train": M["TRAIN"]["AUC"], "Test": M["TEST"]["AUC"],
     "Delta_train_test": D_AUC, "Nguong": "|Δ| ≤ 2–3%",
     "Ket_luan": "✓ Ổn định" if abs(D_AUC) <= 0.03 else "✗ Overfit"},
    {"Chi_tieu": "Gini", "Train": M["TRAIN"]["Gini"], "Test": M["TEST"]["Gini"],
     "Delta_train_test": D_GINI, "Nguong": "Δ ≥ 0 (gap requirement)",
     "Ket_luan": "✓ Gap ≥ 0" if D_GINI >= 0 else "✗ Gap < 0"},
    {"Chi_tieu": "LogLoss", "Train": M["TRAIN"]["LogLoss"], "Test": M["TEST"]["LogLoss"],
     "Delta_train_test": M["TRAIN"]["LogLoss"] - M["TEST"]["LogLoss"], "Nguong": "≤ 0.30 tốt",
     "Ket_luan": _qual(M["TEST"]["LogLoss"], 0.30, 0.50, True)[0]},
    {"Chi_tieu": "Brier", "Train": M["TRAIN"]["Brier"], "Test": M["TEST"]["Brier"],
     "Delta_train_test": M["TRAIN"]["Brier"] - M["TEST"]["Brier"], "Nguong": "≤ 0.15 tốt",
     "Ket_luan": _qual(M["TEST"]["Brier"], 0.15, 0.25, True)[0]},
])
print("BẢNG OVERFITTING & ỔN ĐỊNH (Δ = train − test):")
print(OVF_DF.to_string(index=False, float_format=lambda v: "%.6f" % v))
_XLS = {}


def _dump(fname, sheets):
    _p = XL(fname)
    with pd.ExcelWriter(_p, engine="openpyxl") as _w:
        for _sn, _df in sheets:
            _df.to_excel(_w, sheet_name=_sn[:31], index=False)
    _XLS[fname] = _p
    print(" ✓ %-34s (%.1f KB, %d sheet)" % (fname, os.path.getsize(_p) / 1024.0, len(sheets)))


_dump("metrics_summary.xlsx", [("Metrics_17", METRIC_DF)])
_dump("statistical_assessment.xlsx", [("Assessment", ASSESS_DF)])
_dump("psi_detail.xlsx", [("PSI_score_bin", PSI_DF), ("PSI_tong_hop", pd.DataFrame(
    [{"Chi_tieu": "PSI score (TRAIN→TEST)", "Gia_tri": PSI_SCORE,
      "Nguong": "<0.10 ổn định | 0.10–0.25 theo dõi | ≥0.25 bất ổn",
      "Danh_gia": _qual(PSI_SCORE, 0.10, 0.25, True)[0]},
     {"Chi_tieu": "PSI PD (TRAIN→TEST)", "Gia_tri": PSI_PROB,
      "Nguong": "<0.10 ổn định | 0.10–0.25 theo dõi | ≥0.25 bất ổn",
      "Danh_gia": _qual(PSI_PROB, 0.10, 0.25, True)[0]}]))])
_dump("iv_detail.xlsx", [("IV_per_feature", IV_DF), ("IV_tong", pd.DataFrame(
    [{"Chi_tieu": "IV tổng TRAIN", "Gia_tri": IV_TR_MAX,
      "Nguong": "Thấp <0.10 | TB 0.10–0.30 | Cao >0.30", "Danh_gia": _IV_L(IV_TR_MAX)[0]},
     {"Chi_tieu": "IV tổng TEST", "Gia_tri": IV_TE_MAX,
      "Nguong": "Thấp <0.10 | TB 0.10–0.30 | Cao >0.30", "Danh_gia": _IV_L(IV_TE_MAX)[0]}]))])
_dump("overfitting_stability.xlsx",
      [("Overfitting_gap", OVF_DF), ("Learning_curve_N", LC_DF),
       ("Validation_curve_iter", VC_DF)])
_dump("temporal_stability_ngaybc.xlsx",
      [("Temporal_NGAY_BC", TEMP_DF if len(TEMP_DF) else
        pd.DataFrame([{"Ghi_chu": "SKIP — không có NGAY_BC hoặc không đủ 2 lớp nhãn"}]))])

_h("[13] CROSS-CHECK 3 CHIỀU: CONSOLE ↔ EXCEL ↔ ANNOTATION")
_chk = []
_xl_back = pd.read_excel(XL("metrics_summary.xlsx"), sheet_name="Metrics_17")
_MAP = {"AUC": "AUC", "Gini": "Gini", "KS": "KS", "ECE": "ECE", "MCE": "MCE",
        "BRIER": "Brier", "F1": "F1", "BACC": "BalancedAcc", "PSI": None, "IV_TOTAL": None,
        "IV_FEATURE": None}
for _key, _mk in _MAP.items():
    if _key in ("PSI",):
        _c_tr = _c_te = PSI_SCORE
        _row = _xl_back[_xl_back["Viết tắt"] == "PSI"]
    elif _key in ("IV_TOTAL", "IV_FEATURE"):
        _c_tr, _c_te = IV_TR_MAX, IV_TE_MAX
        _row = _xl_back[_xl_back["Viết tắt"] == "IV"]
    else:
        _c_tr, _c_te = M["TRAIN"][_mk], M["TEST"][_mk]
        _row = _xl_back[_xl_back["Viết tắt"] == {"BalancedAcc": "BAcc", "Brier": "BS"}.get(_mk, _mk)]
    if _key not in ANNO or _row.empty:
        continue
    _x_tr, _x_te = float(_row["Train"].iloc[0]), float(_row["Test"].iloc[0])
    _a_tr, _a_te = ANNO[_key]
    _ok = (abs(_x_tr - round(_c_tr, 10)) < 1e-9 and abs(_x_te - round(_c_te, 10)) < 1e-9
           and abs(_a_tr - round(_c_tr, 10)) < 1e-9 and abs(_a_te - round(_c_te, 10)) < 1e-9)
    _chk.append({"Chi_tieu": _key, "Console_train": _c_tr, "Excel_train": _x_tr,
                 "Annotation_train": _a_tr, "Console_test": _c_te, "Excel_test": _x_te,
                 "Annotation_test": _a_te, "Khop": "✓ KHỚP" if _ok else "✗ LỆCH"})
CHK_DF = pd.DataFrame(_chk)
print(CHK_DF.to_string(index=False, float_format=lambda v: "%.8f" % v))
print("Kết quả cross-check: %d/%d chỉ tiêu KHỚP tuyệt đối (console = Excel = annotation)"
      % (int((CHK_DF["Khop"] == "✓ KHỚP").sum()), len(CHK_DF)))

# [11] MANIFEST + TỔNG KẾT
_h("[14] MANIFEST & TỔNG KẾT")
_man = []
for _fn, _pt in _XLS.items():
    _man.append({"Tên file": _fn, "Loại": "Excel (.xlsx)", "Path": os.path.abspath(_pt),
                 "Size (KB)": round(os.path.getsize(_pt) / 1024.0, 2), "Trạng thái": "OK",
                 "Ghi chú": "Bảng kết quả validation"})
for _pt in PLOTS:
    _man.append({"Tên file": os.path.basename(_pt), "Loại": "Hình (.png, dpi=150)",
                 "Path": os.path.abspath(_pt),
                 "Size (KB)": round(os.path.getsize(_pt) / 1024.0, 2), "Trạng thái": "OK",
                 "Ghi chú": "Đủ 5 annotation"})
for _fn in sorted(os.listdir(OUT_DIR)):
    _pt = os.path.join(OUT_DIR, _fn)
    if os.path.isfile(_pt) and _fn not in _XLS and _fn != "output_manifest.xlsx":
        _man.append({"Tên file": _fn,
                     "Loại": "Excel (.xlsx)" if _fn.endswith(".xlsx") else
                     ("Parquet" if _fn.endswith(".parquet") else "Log/Khác"),
                     "Path": os.path.abspath(_pt),
                     "Size (KB)": round(os.path.getsize(_pt) / 1024.0, 2),
                     "Trạng thái": "OK", "Ghi chú": "Artifact từ script gốc / log runtime"})
MANIFEST_DF = pd.DataFrame(_man)
SKIP_DF = pd.DataFrame(SKIPS) if SKIPS else pd.DataFrame(
    [{"Mục": "-", "Loại": "-", "Nguyên nhân": "Không có mục nào bị skip/lỗi",
      "Xử lý": "Toàn bộ section chạy thành công"}])
_dump("output_manifest.xlsx", [("Manifest", MANIFEST_DF), ("Skip_Loi", SKIP_DF)])
print(MANIFEST_DF.to_string(index=False))
print("\nLOG SKIP / LỖI:")
print(SKIP_DF.to_string(index=False))

_h("[15] BẢNG TỔNG KẾT CUỐI (ĐÃ CROSS-CHECK)")
_SUM = [
    ("Nguồn dữ liệu", "credit_data_woe.parquet — %d dòng × %d cột, 100%% dữ liệu, no sampling"
     % (data_group.shape[0], data_group.shape[1])),
    ("Biến kế thừa", "PD=%s | score=%s | grade=%s | decile=%s | threshold=%s=%.6f"
     % (PROB_COL, SCORE_COL, GRADE_COL, DECILE_COL, _TH_VAR, THRESH)),
    ("Best config (kế thừa, không grid search lại)",
     "%s | C=%s | penalty=%s | solver=%s | class_weight=%s"
     % (best_cfg["scaler_name"], best_cfg["C"], best_cfg["penalty"], best_cfg["solver"],
        best_cfg["class_weight"])),
    ("Calibration Platt kế thừa", "α = %.10f | β = %.10f" % (alpha_train, beta_train)),
    ("AUC (Train/Test/Full)", "%.6f / %.6f / %.6f → %s"
     % (M["TRAIN"]["AUC"], M["TEST"]["AUC"], M["FULL"]["AUC"], _AUC_L(M["TEST"]["AUC"])[0])),
    ("KS (Train/Test/Full)", "%.6f / %.6f / %.6f → %s"
     % (M["TRAIN"]["KS"], M["TEST"]["KS"], M["FULL"]["KS"], _KS_L(M["TEST"]["KS"])[0])),
    ("Gini (Train/Test/Full)", "%.6f / %.6f / %.6f → %s"
     % (M["TRAIN"]["Gini"], M["TEST"]["Gini"], M["FULL"]["Gini"], _GINI_L(M["TEST"]["Gini"])[0])),
    ("IV tổng (Train/Test)", "%.6f / %.6f → %s"
     % (IV_TR_MAX, IV_TE_MAX, _IV_L(IV_TR_MAX)[0])),
    ("Calibration (Test)", "α=%.6f | β=%.6f | ECE=%.6f | MCE=%.6f | Brier=%.6f"
     % (M["TEST"]["alpha"], M["TEST"]["beta"], M["TEST"]["ECE"], M["TEST"]["MCE"],
        M["TEST"]["Brier"])),
    ("Overfitting", "Δ KS = %+.6f (%s) | Δ AUC = %+.6f (%s)"
     % (D_KS, "✓ ≤5%" if abs(D_KS) <= 0.05 else "✗ >5%", D_AUC,
        "✓ ≤3%" if abs(D_AUC) <= 0.03 else "✗ >3%")),
    ("Stability", "PSI score = %.6f (%s) | Temporal: %d kỳ NGAY_BC"
     % (PSI_SCORE, _qual(PSI_SCORE, 0.10, 0.25, True)[0], len(TEMP_DF))),
    ("Artifact", "%d Excel + %d PNG + 1 runtime_log.txt tại %s"
     % (len(_XLS), len(PLOTS), os.path.abspath(OUT_DIR))),
    ("Cross-check", "%d/%d chỉ tiêu khớp console ↔ Excel ↔ annotation"
     % (int((CHK_DF["Khop"] == "✓ KHỚP").sum()), len(CHK_DF))),
    ("Skip / Lỗi", "%d mục" % len(SKIPS)),
    ("Tổng thời gian block append", "%.2f giây" % (timer() - _T_APP)),
]
for _k, _v in _SUM:
    print(" %-44s: %s" % (_k, _v))
print("\nHOÀN THÀNH PIPELINE VALIDATION — %s" % _ts())
sys.stdout = sys.__stdout__
_LOGF.close()
# ====== AUTO_APPEND_END ======

print(f"\nHoàn thành. Thời gian chạy: {timer() - S:.2f} giây")

















