# -*- coding: utf-8 -*-
"""Kiểm thử tầng kiểm tra trước: trường hợp hợp lệ, sai/thiếu, và biên."""

import pandas as pd
import pytest

from app import synthetic
from app import validation as val


@pytest.fixture(scope="module")
def df_hop_le():
    return synthetic.tao_credit_data_gia_lap(n=4000, seed=11)


# --- Trường hợp HỢP LỆ -------------------------------------------------------

def test_du_lieu_hop_le_khong_co_loi_chan(df_hop_le):
    kt = val.KetQuaKiemTra()
    kt.gop(val.kiem_tra_cot_bat_buoc(df_hop_le, val.STEP2_INPUT_COLUMNS, "schema"))
    kt.gop(val.kiem_tra_muc_tieu(df_hop_le))
    kt.gop(val.kiem_tra_chia_tap(df_hop_le))
    assert kt.ok, kt.blockers


def test_skv_gia_lap_khop_schema_step1():
    skv = synthetic.tao_skv_tho_gia_lap(n=500)
    assert list(skv.columns) == val.STEP1_INPUT_COLUMNS
    kt = val.kiem_tra_cot_bat_buoc(skv, val.STEP1_INPUT_COLUMNS, "step1")
    assert kt.ok


# --- Trường hợp SAI / THIẾU --------------------------------------------------

def test_thieu_cot_muc_tieu_bi_chan(df_hop_le):
    kt = val.kiem_tra_muc_tieu(df_hop_le.drop(columns=["NX"]))
    assert not kt.ok


def test_muc_tieu_khong_nhi_phan_bi_chan(df_hop_le):
    df = df_hop_le.copy()
    df.loc[df.index[0], "NX"] = 2
    kt = val.kiem_tra_muc_tieu(df)
    assert not kt.ok


def test_muc_tieu_mot_lop_bi_chan(df_hop_le):
    df = df_hop_le.copy()
    df["NX"] = 0
    kt = val.kiem_tra_muc_tieu(df)
    assert not kt.ok


def test_train_test_rong_bi_chan_kem_giai_thich_khoi_thieu(df_hop_le):
    """Trạng thái đúng của hai tệp parquet được cung cấp: train_test='' toàn bộ."""
    df = df_hop_le.copy()
    df["train_test"] = ""
    kt = val.kiem_tra_chia_tap(df)
    assert not kt.ok
    assert any("khối" in b or "TT14" in b for b in kt.blockers), (
        "Thông điệp phải chỉ rõ khối còn thiếu của chủ sản phẩm")


def test_train_test_nhan_la_bi_chan(df_hop_le):
    df = df_hop_le.copy()
    df.loc[df.index[0], "train_test"] = "oot"
    kt = val.kiem_tra_chia_tap(df)
    assert not kt.ok


def test_thieu_cot_schema_bi_chan(df_hop_le):
    kt = val.kiem_tra_cot_bat_buoc(df_hop_le.drop(columns=["LTV_CIF"]),
                                    val.STEP2_INPUT_COLUMNS, "schema")
    assert not kt.ok


def test_tep_sai_dinh_dang_va_tep_rong_bi_tu_choi():
    cho_phep = ("parquet", "csv", "xlsx", "txt")
    assert val.kiem_tra_tep_co_ban("du_lieu.pkl", b"\x80\x04", 100, 200, cho_phep)
    assert val.kiem_tra_tep_co_ban("du_lieu.exe", b"MZ", 100, 200, cho_phep)
    assert val.kiem_tra_tep_co_ban("du_lieu.csv", b"", 0, 200, cho_phep)
    assert val.kiem_tra_tep_co_ban("du_lieu.parquet", b"KHONGPHAI", 100, 200, cho_phep)
    assert val.kiem_tra_tep_co_ban("du_lieu.xlsx", b"KHONGPHAI", 100, 200, cho_phep)
    assert not val.kiem_tra_tep_co_ban("du_lieu.csv", b"a,b\n1,2\n", 8, 200, cho_phep)


def test_csv_hong_doc_an_toan():
    with pytest.raises(ValueError):
        val.doc_tep_an_toan(b"", "parquet")  # parquet rỗng phải ném ValueError có kiểm soát


# --- Phủ thời gian ------------------------------------------------------------

def test_phu_thoi_gian_khong_du_ky_du_bao_bi_chan(df_hop_le):
    d_min, d_max = df_hop_le["NGAY_BC"].min(), df_hop_le["NGAY_BC"].max()
    kt = val.kiem_tra_phu_thoi_gian(df_hop_le, "NGAY_BC", d_min, d_max, 12)
    assert not kt.ok  # kết thúc quan sát = ngày dữ liệu cuối -> không còn 12 tháng theo dõi


def test_phu_thoi_gian_du_khi_lui_diem_quan_sat(df_hop_le):
    d_min, d_max = df_hop_le["NGAY_BC"].min(), df_hop_le["NGAY_BC"].max()
    qs_cuoi = d_max - pd.DateOffset(months=12)
    kt = val.kiem_tra_phu_thoi_gian(df_hop_le, "NGAY_BC", d_min, qs_cuoi, 12)
    assert kt.ok, kt.blockers
    assert kt.chi_tiet["so_dong_nguy_co_cat_phai"] > 0  # vẫn đếm và cảnh báo cắt phải


def test_phu_thoi_gian_chan_khi_du_lieu_bat_dau_muon(df_hop_le):
    d_min, d_max = df_hop_le["NGAY_BC"].min(), df_hop_le["NGAY_BC"].max()
    kt = val.kiem_tra_phu_thoi_gian(df_hop_le, "NGAY_BC",
                                     d_min - pd.DateOffset(months=6),
                                     d_max - pd.DateOffset(months=12), 12)
    assert not kt.ok


# --- Biên dung lượng và xem trước an toàn --------------------------------------

def test_tep_vuot_gioi_han_dung_luong_bi_chan():
    loi = val.kiem_tra_tep_co_ban("lon.csv", b"a,b\n", 201 * 1024 * 1024, 200, ("csv",))
    assert loi and "vượt giới hạn" in loi[0]


def test_xem_truoc_che_cot_dinh_danh_va_ho_ten(df_hop_le):
    xt = val.xem_truoc_an_toan(df_hop_le, 3)
    for cot in ("BILLING_ACCTNUM", "CIF"):
        assert all(v.startswith("***") and len(v) <= 7 for v in xt[cot])
        assert set(xt[cot]) & set(df_hop_le[cot].head(3).astype(str)) == set()
    assert all(v == "***" for v in xt["TEN_CPD"])


def test_doc_csv_utf8_co_dau():
    noi_dung = "BILLING_ACCTNUM;LTV_CIF\n0000000000001;tín chấp\n".encode("utf-8")
    df = val.doc_tep_an_toan(noi_dung, "csv")
    assert df.iloc[0]["LTV_CIF"] == "tín chấp"
