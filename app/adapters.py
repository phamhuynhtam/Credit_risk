# -*- coding: utf-8 -*-
"""
Bộ chuyển đổi (adapter) giữa ứng dụng web và mã gốc Step 1–3 trong pipeline/.

Nguyên tắc bất biến:
1. Không bao giờ sửa tệp trong pipeline/. Mỗi lần chạy dùng một BẢN SAO thời
   gian chạy đặt trong thư mục chạy riêng (runs/<run_id>/).
2. Thay đổi duy nhất trên bản sao là thay chuỗi đường dẫn cứng kiểu Windows
   (D:\\model, D:/model) bằng thư mục chạy hiện tại, cộng với đổi dấu phân
   cách trong mẫu glob của Step 1 cho phù hợp hệ điều hành đang chạy. Không
   đổi công thức, tham số, ngưỡng hay thứ tự xử lý. Danh sách thay thế được
   ghi vào run_manifest.json để truy vết.
3. Đầu vào được đặt vào thư mục chạy với tên an toàn do hệ thống sinh. Tên
   tệp và nội dung người dùng tải lên không bao giờ được chèn vào mã thực thi
   hay lệnh shell (tiến trình con được gọi dạng danh sách tham số, không qua
   shell).
4. Thành công của một bước được xác định bằng đầu ra kỳ vọng trên đĩa, không
   chỉ bằng mã thoát — vì Step 2 cố ý thoát với mã 0 ngay cả khi kiểm tra đầu
   vào thất bại (sys.exit(0) trong mã gốc).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app import config


# ---------------------------------------------------------------------------
# Kiểm tra môi trường và tính nguyên vẹn mã gốc
# ---------------------------------------------------------------------------

def pyarrow_kha_dung() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def bam_sha256_tep(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def kiem_tra_nguyen_ven_pipeline() -> dict:
    """
    Đối chiếu SHA-256 của 3 tệp pipeline/ với giá trị đã ghi khi truy xuất
    (pipeline/NGUON_GOC.md). Trả về {step: {"khop": bool, "sha256": str|None}}.
    """
    ket_qua = {}
    for step, path in config.STEP_FILES.items():
        if not path.exists():
            ket_qua[step] = {"khop": False, "sha256": None, "ly_do": "Thiếu tệp"}
            continue
        digest = bam_sha256_tep(path)
        ket_qua[step] = {"khop": digest == config.STEP_HASHES[step], "sha256": digest}
    return ket_qua


# ---------------------------------------------------------------------------
# Quản lý thư mục chạy
# ---------------------------------------------------------------------------

def tao_run_dir() -> Path:
    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
    run_dir = config.RUNS_DIR / run_id
    (run_dir / "model").mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def thu_muc_model(run_dir: Path) -> Path:
    return Path(run_dir) / "model"


def _trong_runs(path: Path) -> bool:
    try:
        Path(path).resolve().relative_to(config.RUNS_DIR.resolve())
        return True
    except Exception:
        return False


def xoa_run(run_dir: Path) -> bool:
    """Xóa một thư mục chạy. Chỉ chấp nhận đường dẫn nằm trong runs/."""
    run_dir = Path(run_dir)
    if not _trong_runs(run_dir) or run_dir.resolve() == config.RUNS_DIR.resolve():
        return False
    shutil.rmtree(run_dir, ignore_errors=True)
    return not run_dir.exists()


def don_dep_runs_cu(so_gio: float) -> int:
    """Best-effort: xóa các thư mục chạy cũ hơn số giờ cấu hình. Trả về số đã xóa."""
    if not config.RUNS_DIR.exists():
        return 0
    han = time.time() - so_gio * 3600
    da_xoa = 0
    for muc in config.RUNS_DIR.iterdir():
        try:
            if muc.is_dir() and muc.stat().st_mtime < han:
                shutil.rmtree(muc, ignore_errors=True)
                if not muc.exists():
                    da_xoa += 1
        except Exception:
            continue
    return da_xoa


# ---------------------------------------------------------------------------
# Thay thế đường dẫn trên bản sao thời gian chạy
# ---------------------------------------------------------------------------

def chuyen_doi_duong_dan(ma_nguon: str, thu_muc_model_posix: str, step: str):
    """
    Thay chuỗi đường dẫn cứng của mã gốc bằng thư mục chạy. Trả về
    (ma_da_doi, danh_sach_mo_ta_thay_the) để ghi truy vết.

    Chỉ gồm các thay thế cơ học dưới đây; ngoài ra không động vào mã:
    - Step 1: mẫu glob "\\*.parquet" -> "/*.parquet" (dấu phân cách thư mục).
    - "D:\\model\\" -> "<run>/model/"; "D:/model/" -> "<run>/model/";
      hậu tố còn lại "D:\\model" / "D:/model" -> "<run>/model".
    """
    if '"' in thu_muc_model_posix or "\\" in thu_muc_model_posix:
        raise ValueError("Đường dẫn thư mục chạy chứa ký tự không an toàn để chèn vào bản sao mã.")
    mo_ta = []
    ma = ma_nguon
    if step == "step1":
        cu = r'folder_path + "\*.parquet"'
        moi = 'folder_path + "/*.parquet"'
        if cu in ma:
            ma = ma.replace(cu, moi)
            mo_ta.append('Step 1: mẫu glob "\\*.parquet" → "/*.parquet" (dấu phân cách Linux)')
    cac_cap = [
        ("D:\\model\\", thu_muc_model_posix + "/", "chuỗi đường dẫn 'D:\\model\\' (kèm dấu phân cách cuối)"),
        ("D:/model/", thu_muc_model_posix + "/", "chuỗi đường dẫn 'D:/model/'"),
        ("D:\\model", thu_muc_model_posix, "chuỗi đường dẫn 'D:\\model' còn lại"),
        ("D:/model", thu_muc_model_posix, "chuỗi đường dẫn 'D:/model' còn lại"),
    ]
    for cu, moi, ten in cac_cap:
        so_lan = ma.count(cu)
        if so_lan:
            ma = ma.replace(cu, moi)
            mo_ta.append(f"Thay {so_lan} lần {ten} → thư mục chạy")
    return ma, mo_ta


def con_duong_dan_windows(ma: str) -> bool:
    """True nếu bản sao vẫn còn chuỗi ổ đĩa D: kiểu Windows chưa được thay."""
    return ("D:\\" in ma) or ("D:/" in ma)


# ---------------------------------------------------------------------------
# Đặt đầu vào vào thư mục chạy
# ---------------------------------------------------------------------------

def ghi_bytes_an_toan(dich: Path, noi_dung: bytes) -> None:
    dich.parent.mkdir(parents=True, exist_ok=True)
    with open(dich, "wb") as f:
        f.write(noi_dung)


def dat_dau_vao_step1(run_dir: Path, danh_sach_parquet_bytes: list) -> list:
    """
    Đặt các tệp SKV parquet vào <run>/model/SKV parquet/ với tên do hệ thống
    sinh (skv_001.parquet, ...). Step 1 gốc glob *.parquet trong thư mục này;
    thứ tự nối dữ liệu trong mã gốc vốn phụ thuộc thứ tự glob của hệ điều hành.
    Trả về danh sách tên tệp đã đặt.
    """
    dich_dir = thu_muc_model(run_dir) / "SKV parquet"
    dich_dir.mkdir(parents=True, exist_ok=True)
    ten_da_dat = []
    for i, noi_dung in enumerate(danh_sach_parquet_bytes, start=1):
        ten = f"skv_{i:03d}.parquet"
        ghi_bytes_an_toan(dich_dir / ten, noi_dung)
        ten_da_dat.append(ten)
    return ten_da_dat


def ghi_dataframe_parquet(df, dich: Path) -> None:
    """Ghi DataFrame thành parquet nén brotli (giống cách pipeline ghi đầu ra)."""
    dich.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dich, index=False, compression="brotli", engine="pyarrow")


def dat_credit_data(run_dir: Path, df) -> Path:
    dich = thu_muc_model(run_dir) / "credit_data.parquet"
    ghi_dataframe_parquet(df, dich)
    return dich


def dat_credit_data_woe_tu_bytes(run_dir: Path, noi_dung: bytes) -> Path:
    dich = thu_muc_model(run_dir) / "credit_data_woe.parquet"
    ghi_bytes_an_toan(dich, noi_dung)
    return dich


# ---------------------------------------------------------------------------
# Chạy một bước pipeline bằng tiến trình con
# ---------------------------------------------------------------------------

DAU_RA_KY_VONG = {
    "step1": ["credit_data.parquet", "credit_data1.parquet", "credit_data2.parquet"],
    "step2": ["credit_data_woe.parquet", "credit_data_woe_raw.parquet"],
    "step3": ["credit_data_scoring.parquet", "parameter_model.xlsx",
              "plot/metrics_summary.xlsx"],
}
"""Đầu ra tối thiểu phải xuất hiện trong <run>/model/ để coi bước là hoàn tất."""

DAU_HIEU_THANH_CONG_STDOUT = {
    "step1": None,
    "step2": "HOÀN TẤT",   # Step 2 thoát mã 0 cả khi lỗi nên cần dấu hiệu này
    "step3": None,
}


@dataclass
class KetQuaChay:
    step: str
    thanh_cong: bool
    ma_thoat: int | None
    thoi_gian_giay: float
    log_path: str
    dau_ra_thieu: list = field(default_factory=list)
    thong_diep: str = ""
    het_thoi_gian: bool = False


def chay_step(step: str, run_dir: Path, timeout_giay: int | None = None,
              on_line=None) -> KetQuaChay:
    """
    Tạo bản sao thời gian chạy của mã gốc (chỉ đổi đường dẫn), rồi thực thi
    bằng tiến trình con trong thư mục chạy.

    on_line: hàm callback tùy chọn nhận từng dòng log (dùng cho giao diện
    hiển thị tiến trình). Log đầy đủ được ghi vào runs/<id>/logs/<step>.log.
    """
    # Chuẩn hóa tuyệt đối: tiến trình con chạy với cwd=run_dir nên mọi đường
    # dẫn truyền cho nó phải tuyệt đối, nếu không sẽ bị nhân đôi tiền tố.
    run_dir = Path(run_dir).resolve()
    model_dir = thu_muc_model(run_dir)
    timeout = timeout_giay or config.TIMEOUT_STEP_GIAY

    tep_goc = config.STEP_FILES[step]
    ma_goc = tep_goc.read_text(encoding="utf-8")
    ma_chay, mo_ta_thay = chuyen_doi_duong_dan(ma_goc, model_dir.as_posix(), step)
    if con_duong_dan_windows(ma_chay):
        return KetQuaChay(
            step=step, thanh_cong=False, ma_thoat=None, thoi_gian_giay=0.0,
            log_path="", thong_diep=(
                "Bản sao thời gian chạy vẫn còn đường dẫn kiểu Windows chưa được "
                "chuyển đổi; ứng dụng dừng để tránh ghi sai vị trí. Đây là lỗi của "
                "bộ chuyển đổi, cần báo cho người bảo trì."))
    tep_chay = run_dir / f"runtime_{step}.py"
    tep_chay.write_text(ma_chay, encoding="utf-8")
    log_path = run_dir / "logs" / f"{step}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "MPLBACKEND": "Agg",            # matplotlib không cần màn hình (plt.show thành no-op)
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    bat_dau = time.time()
    het_gio = False
    ma_thoat = None
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"# Bộ chuyển đổi — các thay thế đường dẫn đã áp dụng cho bản sao:\n")
        for dong in mo_ta_thay:
            log_f.write(f"# - {dong}\n")
        log_f.flush()
        proc = subprocess.Popen(
            [sys.executable, "-u", str(tep_chay)],
            cwd=str(run_dir), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        try:
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                if on_line is not None:
                    try:
                        on_line(line.rstrip("\n"))
                    except Exception:
                        pass
                if time.time() - bat_dau > timeout:
                    het_gio = True
                    proc.kill()
                    break
            ma_thoat = proc.wait(timeout=30)
        except Exception:
            proc.kill()
            ma_thoat = proc.wait(timeout=30)
    thoi_gian = time.time() - bat_dau

    thieu = [ten for ten in DAU_RA_KY_VONG[step] if not (model_dir / ten).exists()]
    dau_hieu = DAU_HIEU_THANH_CONG_STDOUT.get(step)
    log_text_tail = ""
    try:
        log_text_tail = log_path.read_text(encoding="utf-8", errors="replace")[-200_000:]
    except Exception:
        pass
    thanh_cong = (not het_gio) and (ma_thoat == 0) and (not thieu)
    if dau_hieu and dau_hieu not in log_text_tail:
        thanh_cong = False

    if het_gio:
        thong_diep = (f"Bước bị dừng do vượt giới hạn thời gian {timeout} giây. "
                      "Đầu ra (nếu có) chưa hoàn chỉnh và không được dùng tiếp.")
    elif thanh_cong:
        thong_diep = "Bước hoàn tất; đủ đầu ra kỳ vọng."
    elif thieu and ma_thoat == 0:
        thong_diep = (
            "Tiến trình kết thúc với mã 0 nhưng thiếu đầu ra kỳ vọng: "
            + ", ".join(thieu)
            + ". Với Step 2, mã gốc cố ý thoát 0 khi kiểm tra đầu vào thất bại — "
              "hãy xem phần cuối nhật ký để biết điều kiện chưa đạt."
        )
    else:
        thong_diep = (
            f"Bước dừng với mã thoát {ma_thoat}. Xem phần cuối nhật ký của bước "
            "để biết lỗi (nhật ký nằm trong thư mục chạy, không chứa giá trị hồ sơ)."
        )
    return KetQuaChay(step=step, thanh_cong=thanh_cong, ma_thoat=ma_thoat,
                      thoi_gian_giay=thoi_gian, log_path=str(log_path),
                      dau_ra_thieu=thieu, thong_diep=thong_diep, het_thoi_gian=het_gio)


def doc_log_tail(log_path: str, so_ky_tu: int = 8000) -> str:
    try:
        return Path(log_path).read_text(encoding="utf-8", errors="replace")[-so_ky_tu:]
    except Exception:
        return ""


def liet_ke_artifact(run_dir: Path) -> list:
    """Liệt kê tệp trong <run>/model/ (tên tương đối, dung lượng) cho trang kết quả."""
    model_dir = thu_muc_model(Path(run_dir))
    ket_qua = []
    if not model_dir.exists():
        return ket_qua
    for tep in sorted(model_dir.rglob("*")):
        if tep.is_file():
            ket_qua.append({
                "ten": str(tep.relative_to(model_dir)),
                "duong_dan": str(tep),
                "dung_luong_kb": round(tep.stat().st_size / 1024.0, 1),
            })
    return ket_qua


# ---------------------------------------------------------------------------
# Manifest của lần chạy (truy vết, không chứa dữ liệu hồ sơ)
# ---------------------------------------------------------------------------

def ghi_manifest(run_dir: Path, thong_tin: dict) -> Path:
    dich = Path(run_dir) / "run_manifest.json"
    du_lieu = {}
    if dich.exists():
        try:
            du_lieu = json.loads(dich.read_text(encoding="utf-8"))
        except Exception:
            du_lieu = {}
    du_lieu.update(thong_tin)
    dich.write_text(json.dumps(du_lieu, ensure_ascii=False, indent=2), encoding="utf-8")
    return dich


def doc_manifest(run_dir: Path) -> dict:
    dich = Path(run_dir) / "run_manifest.json"
    if not dich.exists():
        return {}
    try:
        return json.loads(dich.read_text(encoding="utf-8"))
    except Exception:
        return {}
