# -*- coding: utf-8 -*-
"""
Sinh dữ liệu GIẢ LẬP (synthetic) cho kiểm thử kỹ thuật và minh họa giao diện.

CẢNH BÁO — đọc trước khi dùng:
- Mọi dữ liệu do module này sinh ra là hoàn toàn giả lập bằng bộ sinh số ngẫu
  nhiên có hạt giống cố định; không bắt nguồn từ bất kỳ khách hàng, hồ sơ vay
  hay tệp dữ liệu ngân hàng nào. Kết quả mô hình trên dữ liệu này chỉ chứng
  minh đường ống kỹ thuật chạy được, KHÔNG phải kết quả thực nghiệm và không
  được trình bày như hiệu năng mô hình thật.
- Riêng cột train_test trong dữ liệu giả lập được gán đơn giản theo mốc tháng
  (các tháng đầu = train, các tháng cuối = test) duy nhất để có dữ liệu kiểm
  thử đạt điều kiện đầu vào của Step 2/Step 3. Đây KHÔNG phải khối lọc TT14 /
  chia OOT 30% của quy trình gốc (khối đó hiện vắng mặt trong Step 1 và thuộc
  về chủ sản phẩm). Lộ trình chạy dữ liệu thật của ứng dụng không dùng module
  này và không tự gán train_test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NHAN_GIA_LAP = "DỮ LIỆU GIẢ LẬP MINH HỌA — KHÔNG PHẢI DỮ LIỆU THẬT"

_THOIHAN = ["NGANHAN", "TRUNGHAN", "DAIHAN"]
_RUI_RO_THOIHAN = {"NGANHAN": 0.0, "TRUNGHAN": 0.30, "DAIHAN": 0.55}
_CIB = ["KHCN", "SME", "CIB"]
_RUI_RO_CIB = {"KHCN": 0.10, "SME": 0.35, "CIB": -0.10}


def _cot_phu(rng: np.random.Generator, n: int) -> dict:
    """Các cột phân loại nhiễu (không mang tín hiệu nhãn) để Step 2 có biến bị loại bởi IV."""
    def chon(ds, p=None):
        return rng.choice(ds, size=n, p=p)
    return {
        "CAP_1": chon(["CAP1_A", "CAP1_B", "CAP1_C"]),
        "CAP_2": chon(["CAP2_A", "CAP2_B"]),
        "DM": chon(["DM_A", "DM_B", "DM_C", "DM_D"]),
        "DVKD_OLD": chon([f"DVKD_{i:02d}" for i in range(1, 9)]),
        "TDVKD": chon([f"DONVI_{i:02d}" for i in range(1, 9)]),
        "CAP_PD": chon(["TRUONG DVKD", "GDTD", "PGD"]),
        "TEN_CPD": chon(["Trưởng PGD", "Phó PGD", "Trưởng phòng"]),
        "Vung mien": chon(["BAC", "TRUNG", "NAM"]),
        "CAP_PD1": chon(["DVKD", "PGD"]),
        "TENNGANH_KH": chon([f"NGANH_KH_{i}" for i in range(1, 6)]),
        "TENNGANH": chon([f"NGANH_{i}" for i in range(1, 6)]),
        "TEN_MD": chon(["MUCDICH_A", "MUCDICH_B", "MUCDICH_C"]),
        "FLAG_LHINH": chon(["LH1", "LH2"]),
        "TENLHINH2": chon(["LOAIHINH_A", "LOAIHINH_B", "LOAIHINH_C"]),
        "TENMUCDICH2": chon(["MD_A", "MD_B"]),
        "DM2": chon(["DM2_A", "DM2_B", "DM2_C"]),
        "CAP_PD2": chon(["DVKD", "PGD"]),
        "CAP_PD3": chon(["DVKD", "PGD"]),
        "PL_BDS_DS": chon(["BDS", "DS", "KHAC"]),
    }


def _sinh_tin_hieu(rng, n, so_thang, thang_bat_dau="2024-01-31"):
    """Phần dữ liệu chung mang tín hiệu cho nhãn; dùng cho cả hai dạng fixture."""
    thang_idx = rng.integers(0, so_thang, size=n)
    cac_thang = pd.date_range(thang_bat_dau, periods=so_thang, freq="ME")
    ngay_bc = pd.Series(cac_thang[thang_idx]).reset_index(drop=True)

    lai_suat = np.clip(rng.normal(10.0, 2.5, size=n), 3.0, 20.0)
    du_no = rng.lognormal(mean=12.0, sigma=0.9, size=n)  # đơn vị tiền giả lập
    dur_open = rng.integers(0, 121, size=n)
    dur_issue = np.clip(dur_open - rng.integers(0, 24, size=n), 0, None)
    dur_first = dur_open.copy()
    thoihan = rng.choice(_THOIHAN, size=n, p=[0.55, 0.30, 0.15])
    cib = rng.choice(_CIB, size=n, p=[0.5, 0.35, 0.15])
    tin_chap = rng.random(n) < 0.25
    ltv_so = np.clip(rng.normal(0.62, 0.18, size=n), 0.05, 1.10)
    tsbd = np.where(tin_chap, 0.0, du_no / np.maximum(ltv_so, 0.05))
    bds = np.where(rng.random(n) < 0.5, tsbd * rng.uniform(0.3, 1.0, size=n), 0.0)
    ds = np.maximum(tsbd - bds, 0.0)

    z_du_no = (np.log1p(du_no) - np.log1p(du_no).mean()) / np.log1p(du_no).std()
    logit = (
        -2.75
        + 0.25 * (lai_suat - 10.0)
        + 0.55 * z_du_no
        - 0.012 * dur_open
        + np.array([_RUI_RO_THOIHAN[t] for t in thoihan])
        + np.array([_RUI_RO_CIB[c] for c in cib])
        + np.where(tin_chap, 0.40, 0.8 * (ltv_so - 0.6))
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    nx = (rng.random(n) < p).astype(np.int32)

    ltv_cif = np.where(tin_chap, "tín chấp", np.round(ltv_so, 4).astype(str))
    ids = {
        "BILLING_ACCTNUM": [f"{1000000000000 + i}" for i in range(n)],
        "CIF": [f"{10000000 + (i % max(n // 3, 1)):08d}" for i in range(n)],
    }
    chinh = {
        "NGAY_BC": ngay_bc,
        "LAI_SUAT": np.round(lai_suat, 4),
        "SO_THE": rng.integers(0, 4, size=n),
        "NX": nx,
        "TSBD": np.round(tsbd, 2),
        "THOIHAN": thoihan,
        "CIB_SME_KHCN": cib,
        "LTV_CIF": pd.array(ltv_cif, dtype="string"),
        "DU_NO_ADJUST_CIF": np.round(du_no, 2),
        "BDS": np.round(bds, 2),
        "DS": np.round(ds, 2),
        "DURATION_ISSUE_DATE": dur_issue.astype(np.int32),
        "DURATION_OPEN_CIF": dur_open.astype(np.int32),
        "DURATION_FIRST_BOOK_DATE": dur_first.astype(np.int32),
    }
    return ids, chinh, thang_idx, so_thang


def tao_credit_data_gia_lap(n: int = 12000, so_thang: int = 24,
                            so_thang_test: int = 7, seed: int = 42) -> pd.DataFrame:
    """
    Tạo DataFrame đúng 36 cột của credit_data.parquet (đầu vào Step 2).

    train_test: gán theo mốc tháng chỉ phục vụ kiểm thử (xem cảnh báo đầu
    module). Các tháng cuối (so_thang_test tháng) = "test", còn lại "train" —
    tỷ lệ xấp xỉ 70/30 theo thiết kế fixture, không phải quy tắc nghiệp vụ.
    """
    rng = np.random.default_rng(seed)
    ids, chinh, thang_idx, tong_thang = _sinh_tin_hieu(rng, n, so_thang)
    phu = _cot_phu(rng, n)
    df = pd.DataFrame({**ids, **phu, **{k: v for k, v in chinh.items()}})
    moc = tong_thang - so_thang_test
    df["train_test"] = np.where(thang_idx >= moc, "test", "train")
    thu_tu = [
        "BILLING_ACCTNUM", "CIF", "CAP_1", "CAP_2", "DM", "NGAY_BC", "LAI_SUAT",
        "SO_THE", "NX", "DVKD_OLD", "TDVKD", "TSBD", "CAP_PD", "TEN_CPD",
        "THOIHAN", "CIB_SME_KHCN", "LTV_CIF", "Vung mien", "CAP_PD1",
        "TENNGANH_KH", "TENNGANH", "TEN_MD", "FLAG_LHINH", "TENLHINH2",
        "TENMUCDICH2", "DM2", "CAP_PD2", "CAP_PD3", "DU_NO_ADJUST_CIF", "BDS",
        "DS", "PL_BDS_DS", "DURATION_ISSUE_DATE", "DURATION_OPEN_CIF",
        "DURATION_FIRST_BOOK_DATE", "train_test",
    ]
    return df[thu_tu].sort_values(["NGAY_BC", "BILLING_ACCTNUM"]).reset_index(drop=True)


def tao_skv_tho_gia_lap(n: int = 6000, so_thang: int = 12, seed: int = 7) -> pd.DataFrame:
    """
    Tạo DataFrame đúng 35 cột đầu vào thô của Step 1 (các tệp SKV *.parquet):
    ngày dạng chuỗi dd/mm/yyyy, có NHOM_NO (1–5) thay cho NX.

    Dùng để kiểm thử Step 1: đầu ra phải có train_test rỗng (vì khối chia tập
    vắng mặt trong mã đã truy cập) và bị kiểm tra trước của web chặn trước
    Step 2 — đúng hành vi kỳ vọng, không phải lỗi.
    """
    rng = np.random.default_rng(seed)
    ids, chinh, _, _ = _sinh_tin_hieu(rng, n, so_thang, thang_bat_dau="2024-01-31")
    phu = _cot_phu(rng, n)
    ngay_bc = pd.Series(chinh["NGAY_BC"]).reset_index(drop=True)
    mo_cif = ngay_bc - pd.to_timedelta(chinh["DURATION_OPEN_CIF"] * 30, unit="D")
    phat_hanh = ngay_bc - pd.to_timedelta(chinh["DURATION_ISSUE_DATE"] * 30, unit="D")
    fmt = "%d/%m/%Y"
    nhom_no = np.where(chinh["NX"] == 1,
                       rng.choice([3, 4, 5], size=n),
                       rng.choice([1, 2], size=n, p=[0.85, 0.15]))
    df = pd.DataFrame({
        **ids, **phu,
        "ISSUE_DATE": phat_hanh.dt.strftime(fmt),
        "OPEN_CIF": mo_cif.dt.strftime(fmt),
        "FIRST_BOOK_DATE": mo_cif.dt.strftime(fmt),
        "NGAY_BC": ngay_bc.dt.strftime(fmt),
        "LAI_SUAT": chinh["LAI_SUAT"],
        "SO_THE": chinh["SO_THE"],
        "NHOM_NO": nhom_no.astype(int),
        "TSBD": chinh["TSBD"],
        "THOIHAN": chinh["THOIHAN"],
        "CIB_SME_KHCN": chinh["CIB_SME_KHCN"],
        "LTV_CIF": chinh["LTV_CIF"].astype(object),
        "DU_NO_ADJUST_CIF": chinh["DU_NO_ADJUST_CIF"],
        "BDS": chinh["BDS"],
        "DS": chinh["DS"],
    })
    thu_tu = [
        "BILLING_ACCTNUM", "CIF", "ISSUE_DATE", "OPEN_CIF", "CAP_1", "CAP_2",
        "DM", "NGAY_BC", "LAI_SUAT", "SO_THE", "NHOM_NO", "DVKD_OLD", "TDVKD",
        "TSBD", "CAP_PD", "TEN_CPD", "THOIHAN", "CIB_SME_KHCN", "LTV_CIF",
        "Vung mien", "CAP_PD1", "TENNGANH_KH", "TENNGANH", "TEN_MD",
        "FLAG_LHINH", "TENLHINH2", "TENMUCDICH2", "DM2", "FIRST_BOOK_DATE",
        "CAP_PD2", "CAP_PD3", "DU_NO_ADJUST_CIF", "BDS", "DS", "PL_BDS_DS",
    ]
    return df[thu_tu].reset_index(drop=True)
