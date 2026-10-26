# -*- coding: utf-8 -*-
"""
Kiểm thử đầu cuối (end-to-end) bằng dữ liệu GIẢ LẬP:

1. Step 1 chạy trên dữ liệu SKV giả lập → sinh credit_data nhưng train_test
   rỗng (khối TT14/OOT vắng mặt trong mã đã truy cập) → chính Step 2 từ chối
   đầu vào đó (mã thoát 0, không artifact) và bộ chuyển đổi phát hiện đúng.
2. Step 2 + Step 3 chạy trọn trên credit_data giả lập hợp lệ → đủ artifact
   kỳ vọng; báo cáo tổng hợp không chứa giá trị định danh.

Các kiểm thử này chứng minh đường ống kỹ thuật và cơ chế chặn; chúng không
khẳng định chất lượng mô hình hay tính tương đương với bất kỳ lần chạy nào
trên dữ liệu thật.
"""

import io

import pandas as pd
import pytest

from app import adapters, config, report as report_mod, synthetic


@pytest.fixture()
def run_dir_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    run_dir = adapters.tao_run_dir()
    hash_truoc = {s: adapters.bam_sha256_tep(p) for s, p in config.STEP_FILES.items()}
    yield run_dir
    # Mã gốc không được thay đổi sau mọi lần chạy
    hash_sau = {s: adapters.bam_sha256_tep(p) for s, p in config.STEP_FILES.items()}
    assert hash_truoc == hash_sau


def test_step1_dung_o_ranh_gioi_train_test_rong_va_step2_tu_choi(run_dir_tmp):
    df_skv = synthetic.tao_skv_tho_gia_lap(n=3000, seed=3)
    buf = io.BytesIO()
    df_skv.to_parquet(buf, index=False, engine="pyarrow")
    adapters.dat_dau_vao_step1(run_dir_tmp, [buf.getvalue()])

    kq1 = adapters.chay_step("step1", run_dir_tmp, timeout_giay=600)
    assert kq1.thanh_cong, adapters.doc_log_tail(kq1.log_path, 3000)

    dau_ra = pd.read_parquet(adapters.thu_muc_model(run_dir_tmp) / "credit_data.parquet",
                             columns=["train_test", "NX"], engine="pyarrow")
    assert (dau_ra["train_test"].astype(str).str.strip() == "").all(), (
        "Step 1 theo mã đã truy cập phải cho train_test rỗng vì khối chia tập "
        "chưa có trong tệp; nếu chủ sản phẩm bổ sung khối đó, hãy cập nhật test.")
    assert set(dau_ra["NX"].unique()) <= {0, 1}

    kq2 = adapters.chay_step("step2", run_dir_tmp, timeout_giay=600)
    assert kq2.thanh_cong is False
    assert kq2.ma_thoat == 0, "Step 2 cố ý sys.exit(0) khi validate thất bại"
    assert "credit_data_woe.parquet" in kq2.dau_ra_thieu
    assert "train_test" in adapters.doc_log_tail(kq2.log_path, 4000)


@pytest.mark.slow
def test_step2_step3_tron_goi_tren_du_lieu_gia_lap(run_dir_tmp):
    df = synthetic.tao_credit_data_gia_lap(n=12000, seed=42)
    adapters.dat_credit_data(run_dir_tmp, df)

    kq2 = adapters.chay_step("step2", run_dir_tmp, timeout_giay=1200)
    assert kq2.thanh_cong, adapters.doc_log_tail(kq2.log_path, 4000)
    woe = pd.read_parquet(adapters.thu_muc_model(run_dir_tmp) / "credit_data_woe.parquet",
                          engine="pyarrow")
    cot_woe = [c for c in woe.columns
               if c not in ("BILLING_ACCTNUM", "CIF", "NGAY_BC", "NX", "train_test")]
    assert cot_woe, "Step 2 phải giữ được ít một biến sau lọc WOE/IV trên dữ liệu kiểm thử"

    kq3 = adapters.chay_step("step3", run_dir_tmp, timeout_giay=1200)
    assert kq3.thanh_cong, adapters.doc_log_tail(kq3.log_path, 4000)

    artifacts = {a["ten"] for a in adapters.liet_ke_artifact(run_dir_tmp)}
    for ten in ("credit_data_scoring.parquet", "parameter_model.xlsx",
                "score_parameter_new.xlsx", "plot/metrics_summary.xlsx",
                "Gini_KS_result.xlsx"):
        assert ten in artifacts
    so_png = sum(1 for t in artifacts if t.endswith(".png"))
    assert so_png >= 20

    kq_doc = report_mod.doc_ket_qua_pipeline(run_dir_tmp)
    assert kq_doc["metrics_17"] is not None and len(kq_doc["metrics_17"]) >= 10
    assert kq_doc["he_so"] is not None and len(kq_doc["he_so"]) >= 2

    # Báo cáo tổng hợp không chứa giá trị định danh cấp hồ sơ
    adapters.ghi_manifest(run_dir_tmp, {
        "run_id": run_dir_tmp.name, "du_lieu_gia_lap": True, "diem_vao": "gia_lap_step2"})
    html_bao_cao = report_mod.tao_bao_cao_html(run_dir_tmp)
    mot_so_ho_so = str(woe["BILLING_ACCTNUM"].iloc[0])
    mot_cif = str(woe["CIF"].iloc[0])
    assert mot_so_ho_so not in html_bao_cao
    assert mot_cif not in html_bao_cao
    xlsx_bytes = report_mod.tao_bao_cao_excel(run_dir_tmp)
    assert mot_so_ho_so.encode() not in xlsx_bytes
