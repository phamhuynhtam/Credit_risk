# -*- coding: utf-8 -*-
"""Kiểm thử bộ chuyển đổi: tính nguyên vẹn mã gốc và thay thế đường dẫn."""

from app import adapters, config


def test_ba_tep_pipeline_khop_hash_da_truy_xuat():
    kq = adapters.kiem_tra_nguyen_ven_pipeline()
    for step in ("step1", "step2", "step3"):
        assert kq[step]["sha256"] == config.STEP_HASHES[step], (
            f"{config.STEP_FILES[step].name} lệch SHA-256 so với bản đã truy xuất; "
            "nếu chủ sản phẩm chủ đích sửa mã gốc, hãy cập nhật hashes và tài liệu đi kèm."
        )
        assert kq[step]["khop"] is True


def test_chuyen_doi_duong_dan_khong_con_windows_va_bien_dich_duoc():
    for step in ("step1", "step2", "step3"):
        ma_goc = config.STEP_FILES[step].read_text(encoding="utf-8")
        ma_chay, mo_ta = adapters.chuyen_doi_duong_dan(ma_goc, "/tmp/run_test/model", step)
        assert mo_ta, f"{step}: kỳ vọng có ít nhất một thay thế đường dẫn được ghi nhận"
        assert not adapters.con_duong_dan_windows(ma_chay), (
            f"{step}: bản sao thời gian chạy vẫn còn đường dẫn kiểu Windows")
        # Không còn dấu phân cách Windows ngay sau 'model' trong đường dẫn đã đổi
        assert "model\\SKV" not in ma_chay
        assert "model\\credit" not in ma_chay
        # Bản sao vẫn là Python hợp lệ
        compile(ma_chay, f"runtime_{step}", "exec")


def test_chuyen_doi_khong_dung_vao_tham_so_pipeline():
    """Các ngưỡng nghiệp vụ chủ chốt phải nguyên vẹn sau chuyển đổi."""
    ma2, _ = adapters.chuyen_doi_duong_dan(
        config.STEP2_FILE.read_text(encoding="utf-8"), "/tmp/r/model", "step2")
    for hang_so in ("IV_THRESHOLD               = 0.1",
                    "CORR_THRESHOLD             = 0.89",
                    "VIF_THRESHOLD              = 5",
                    "MIN_BIN_SIZE               = 0.05",
                    'TARGET_COL  = "NX"'):
        assert hang_so in ma2
    ma3, _ = adapters.chuyen_doi_duong_dan(
        config.STEP3_FILE.read_text(encoding="utf-8"), "/tmp/r/model", "step3")
    for hang_so in ('FIXED_SOLVER = "newton-cholesky"',
                    "min_score = 300", "max_score = 850", "base_odds = 50"):
        assert hang_so in ma3
