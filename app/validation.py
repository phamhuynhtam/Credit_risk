# -*- coding: utf-8 -*-
"""
Tầng kiểm tra trước (pre-check) của ứng dụng web.

Vai trò: kiểm tra tệp người dùng tải lên TRƯỚC khi chạy pipeline — định dạng,
dung lượng, schema, kiểu dữ liệu, giá trị thiếu, trùng lặp, độ phủ thời gian
theo khai báo của người dùng — và tạo bản xem trước an toàn (che cột định
danh). Tầng này KHÔNG thay đổi logic của Step 1–3 và không thay thế các kiểm
tra mà bản thân Step 2/Step 3 tự thực hiện khi chạy; nó chỉ giúp người dùng
phát hiện sớm và sửa dữ liệu trước khi tốn thời gian chạy.

Nguyên tắc báo lỗi: thông báo chỉ chứa tên cột, số đếm, tỷ lệ và khoảng ngày
gộp; không đưa giá trị của từng khách hàng/hồ sơ vào thông báo, nhật ký hay
báo cáo. Ngoại lệ có chủ đích: liệt kê tối đa 10 giá trị phân biệt của hai cột
cấu trúc `NX` và `train_test` để người dùng sửa nhãn (đây là metadata cấu
trúc, không phải dữ liệu cá nhân).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# Schema kỳ vọng — chép đúng từ mã nguồn pipeline đã truy cập
# ---------------------------------------------------------------------------

STEP1_INPUT_COLUMNS = [
    "BILLING_ACCTNUM", "CIF", "ISSUE_DATE", "OPEN_CIF", "CAP_1", "CAP_2", "DM",
    "NGAY_BC", "LAI_SUAT", "SO_THE", "NHOM_NO", "DVKD_OLD", "TDVKD", "TSBD",
    "CAP_PD", "TEN_CPD", "THOIHAN", "CIB_SME_KHCN", "LTV_CIF", "Vung mien",
    "CAP_PD1", "TENNGANH_KH", "TENNGANH", "TEN_MD", "FLAG_LHINH", "TENLHINH2",
    "TENMUCDICH2", "DM2", "FIRST_BOOK_DATE", "CAP_PD2", "CAP_PD3",
    "DU_NO_ADJUST_CIF", "BDS", "DS", "PL_BDS_DS",
]
"""35 cột mà Step 1 đọc từ mỗi tệp SKV (danh sách cứng trong pd.read_parquet)."""

STEP1_DATE_COLUMNS = ["ISSUE_DATE", "OPEN_CIF", "NGAY_BC", "FIRST_BOOK_DATE"]
"""Các cột ngày Step 1 phân tích với format='%d/%m/%Y'."""

STEP2_INPUT_COLUMNS = [
    "BILLING_ACCTNUM", "CIF", "CAP_1", "CAP_2", "DM", "NGAY_BC", "LAI_SUAT",
    "SO_THE", "NX", "DVKD_OLD", "TDVKD", "TSBD", "CAP_PD", "TEN_CPD", "THOIHAN",
    "CIB_SME_KHCN", "LTV_CIF", "Vung mien", "CAP_PD1", "TENNGANH_KH",
    "TENNGANH", "TEN_MD", "FLAG_LHINH", "TENLHINH2", "TENMUCDICH2", "DM2",
    "CAP_PD2", "CAP_PD3", "DU_NO_ADJUST_CIF", "BDS", "DS", "PL_BDS_DS",
    "DURATION_ISSUE_DATE", "DURATION_OPEN_CIF", "DURATION_FIRST_BOOK_DATE",
    "train_test",
]
"""36 cột đầu ra của Step 1, cũng là dạng đầu vào Step 2 mong đợi."""

STEP3_ID_COLUMNS = ["BILLING_ACCTNUM", "CIF", "NGAY_BC"]
"""Cột định danh/thời gian Step 3 kỳ vọng có trong credit_data_woe.parquet
(cùng với NX, train_test — xem ID_COLS trong mã Step 3)."""

TARGET_COL = "NX"
SPLIT_COL = "train_test"
DATE_COL = "NGAY_BC"
TRAIN_LABEL = "train"
TEST_LABEL = "test"

# Cột chứa định danh hoặc họ tên cá nhân: luôn bị che khi xem trước và không
# bao giờ xuất hiện dưới dạng giá trị trong báo cáo/thông báo của ứng dụng.
COT_DINH_DANH = ["BILLING_ACCTNUM", "CIF", "SO_THE", "TEN_CPD"]
# Cột họ tên: che toàn bộ. Cột số định danh: chỉ giữ 4 ký tự cuối khi xem trước
# để người dùng đối chiếu dòng trong chính phiên của mình.
COT_CHE_HOAN_TOAN = ["TEN_CPD"]

CAC_DINH_DANG_NGAY = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"]
"""Định dạng ngày thử theo thứ tự khi cột ngày đang ở dạng chuỗi.
Định dạng đầu tiên là định dạng mã gốc Step 1 sử dụng."""


# ---------------------------------------------------------------------------
# Kiểm tra định dạng tệp ở mức byte
# ---------------------------------------------------------------------------

def kiem_tra_tep_co_ban(ten_tep: str, du_lieu_dau: bytes, kich_thuoc: int,
                         gioi_han_mb: int, dinh_dang_cho_phep: tuple) -> list:
    """Trả về danh sách lỗi chặn (rỗng = đạt). Không thực thi nội dung tệp."""
    loi = []
    if kich_thuoc == 0:
        loi.append("Tệp rỗng (0 byte). Hãy chọn tệp có nội dung.")
        return loi
    if kich_thuoc > gioi_han_mb * 1024 * 1024:
        loi.append(
            f"Tệp vượt giới hạn {gioi_han_mb} MB của ứng dụng "
            f"(dung lượng tệp: {kich_thuoc / 1024 / 1024:.1f} MB). "
            "Hãy chia nhỏ dữ liệu hoặc liên hệ chủ sản phẩm để nâng giới hạn có kiểm soát."
        )
    phan_mo_rong = ten_tep.rsplit(".", 1)[-1].lower() if "." in ten_tep else ""
    if phan_mo_rong not in dinh_dang_cho_phep:
        loi.append(
            f"Định dạng .{phan_mo_rong or '(không có)'} không được hỗ trợ. "
            f"Định dạng được chấp nhận: {', '.join('.' + d for d in dinh_dang_cho_phep)}. "
            "Ứng dụng không nhận tệp pickle/joblib vì lý do an toàn."
        )
    elif phan_mo_rong == "parquet" and not du_lieu_dau.startswith(b"PAR1"):
        loi.append("Tệp khai là .parquet nhưng không có chữ ký PAR1 ở đầu tệp. "
                   "Hãy kiểm tra lại tệp; ứng dụng không thử đọc khi định dạng không khớp.")
    elif phan_mo_rong == "xlsx" and not du_lieu_dau.startswith(b"PK"):
        loi.append("Tệp khai là .xlsx nhưng không phải gói ZIP/Office hợp lệ. "
                   "Hãy xuất lại tệp Excel đúng định dạng .xlsx (không phải .xls/.xlsm).")
    return loi


def phan_mo_rong_tep(ten_tep: str) -> str:
    return ten_tep.rsplit(".", 1)[-1].lower() if "." in ten_tep else ""


# ---------------------------------------------------------------------------
# Đọc tệp an toàn (không nạp đối tượng tùy ý)
# ---------------------------------------------------------------------------

def _doan_dau_phan_cach(noi_dung: bytes) -> str:
    dong_dau = noi_dung[:8192].split(b"\n", 1)[0]
    ung_vien = [",", ";", "\t", "|"]
    dem = {d: dong_dau.count(d.encode()) for d in ung_vien}
    tot_nhat = max(dem, key=dem.get)
    return tot_nhat if dem[tot_nhat] > 0 else ","


def doc_tep_an_toan(noi_dung: bytes, phan_mo_rong: str,
                    sheet_name=None, dau_phan_cach: str = "tự động") -> pd.DataFrame:
    """
    Đọc nội dung tệp thành DataFrame bằng bộ phân tích dữ liệu thuần
    (pyarrow cho parquet, openpyxl cho xlsx, parser CSV cho csv/txt).
    Ném ValueError với thông điệp an toàn khi không đọc được.
    """
    ext = phan_mo_rong.lower()
    try:
        if ext == "parquet":
            import pyarrow.parquet as pq  # chỉ nạp khi cần; UI đã kiểm tra khả dụng
            bang = pq.read_table(io.BytesIO(noi_dung))
            return bang.to_pandas()
        if ext == "xlsx":
            return pd.read_excel(io.BytesIO(noi_dung), sheet_name=sheet_name or 0,
                                 engine="openpyxl")
        if ext in ("csv", "txt"):
            sep = _doan_dau_phan_cach(noi_dung) if dau_phan_cach == "tự động" else dau_phan_cach
            loi_giai_ma = None
            for ma_hoa in ("utf-8-sig", "cp1258"):
                try:
                    return pd.read_csv(io.BytesIO(noi_dung), sep=sep,
                                       encoding=ma_hoa, low_memory=False)
                except UnicodeDecodeError as exc:  # thử mã hóa tiếp theo
                    loi_giai_ma = exc
            raise ValueError(
                "Không giải mã được tệp văn bản bằng UTF-8 hoặc CP1258. "
                "Hãy lưu lại tệp ở mã hóa UTF-8."
            ) from loi_giai_ma
    except ValueError:
        raise
    except Exception as exc:
        # Không đưa nội dung dữ liệu vào thông báo; chỉ nêu loại lỗi kỹ thuật.
        raise ValueError(
            f"Không đọc được tệp (lỗi {type(exc).__name__}). "
            "Nguyên nhân thường gặp: tệp hỏng, sai định dạng thực tế, hoặc "
            "thiếu thư viện đọc định dạng trong môi trường chạy."
        ) from exc
    raise ValueError(f"Định dạng .{ext} không được hỗ trợ.")


def liet_ke_sheet_xlsx(noi_dung: bytes) -> list:
    """Liệt kê tên sheet của tệp xlsx (chỉ đọc cấu trúc, không đọc dữ liệu)."""
    try:
        with pd.ExcelFile(io.BytesIO(noi_dung), engine="openpyxl") as xf:
            return list(xf.sheet_names)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Kết quả kiểm tra
# ---------------------------------------------------------------------------

@dataclass
class KetQuaKiemTra:
    """Kết quả gộp của một nhóm kiểm tra: lỗi chặn, cảnh báo, chi tiết gộp."""
    blockers: list = field(default_factory=list)   # có mặt bất kỳ mục nào = không được chạy bước phụ thuộc
    warnings: list = field(default_factory=list)
    chi_tiet: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.blockers) == 0

    def gop(self, other: "KetQuaKiemTra") -> "KetQuaKiemTra":
        self.blockers.extend(other.blockers)
        self.warnings.extend(other.warnings)
        self.chi_tiet.update(other.chi_tiet)
        return self


# ---------------------------------------------------------------------------
# Kiểm tra schema và kiểu
# ---------------------------------------------------------------------------

def kiem_tra_cot_bat_buoc(df: pd.DataFrame, cot_yeu_cau: list, ten_bo_schema: str) -> KetQuaKiemTra:
    kq = KetQuaKiemTra()
    thieu = [c for c in cot_yeu_cau if c not in df.columns]
    if thieu:
        kq.blockers.append(
            f"Thiếu {len(thieu)} cột bắt buộc của {ten_bo_schema}: {', '.join(thieu)}. "
            "Hãy dùng bước ánh xạ cột để ghép cột của bạn với tên cột quy trình, "
            "hoặc bổ sung cột còn thiếu trong dữ liệu nguồn."
        )
    trung_ten = sorted({c for c in df.columns if list(df.columns).count(c) > 1})
    if trung_ten:
        kq.blockers.append(
            f"Tên cột bị trùng trong tệp: {', '.join(trung_ten)}. Hãy đổi tên để mỗi cột là duy nhất."
        )
    kq.chi_tiet["cot_thieu"] = thieu
    kq.chi_tiet["so_cot"] = int(df.shape[1])
    kq.chi_tiet["so_dong"] = int(df.shape[0])
    return kq


def tong_quan_cot(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng tổng quan theo cột: kiểu dữ liệu, số thiếu, tỷ lệ thiếu, số giá trị phân biệt."""
    n = len(df)
    dong = []
    for cot in df.columns:
        s = df[cot]
        so_thieu = int(s.isna().sum())
        dong.append({
            "Cột": cot,
            "Kiểu dữ liệu": str(s.dtype),
            "Số giá trị thiếu": so_thieu,
            "Tỷ lệ thiếu (%)": round(100.0 * so_thieu / n, 2) if n else 0.0,
            "Số giá trị phân biệt": int(s.nunique(dropna=True)),
            "Cột hằng số": "Có" if s.nunique(dropna=True) <= 1 else "",
        })
    return pd.DataFrame(dong)


def kiem_tra_trung_lap(df: pd.DataFrame) -> KetQuaKiemTra:
    kq = KetQuaKiemTra()
    so_trung_toan_dong = int(df.duplicated().sum())
    kq.chi_tiet["dong_trung_toan_bo"] = so_trung_toan_dong
    if so_trung_toan_dong > 0:
        kq.warnings.append(
            f"Có {so_trung_toan_dong:,} dòng trùng hoàn toàn với dòng khác. "
            "Ứng dụng không tự xóa; hãy rà soát quy trình trích xuất dữ liệu nguồn."
        )
    khoa = [c for c in ("BILLING_ACCTNUM", DATE_COL) if c in df.columns]
    if len(khoa) == 2:
        so_trung_khoa = int(df.duplicated(subset=khoa).sum())
        kq.chi_tiet["trung_khoa_bill_ngaybc"] = so_trung_khoa
        if so_trung_khoa > 0:
            kq.warnings.append(
                f"Có {so_trung_khoa:,} dòng trùng khóa (BILLING_ACCTNUM, NGAY_BC). "
                "Nếu mỗi tài khoản chỉ xuất hiện một lần mỗi kỳ báo cáo, đây là dấu hiệu cần kiểm tra."
            )
    return kq


def thu_phan_tich_ngay(chuoi_ngay: pd.Series):
    """
    Thử phân tích một cột ngày. Trả về (series_datetime, dinh_dang_dung, ty_le_that_bai).
    Nếu cột đã là datetime thì trả về nguyên trạng.
    """
    if pd.api.types.is_datetime64_any_dtype(chuoi_ngay):
        return chuoi_ngay, "(datetime sẵn có)", 0.0
    khong_thieu = chuoi_ngay.dropna()
    if len(khong_thieu) == 0:
        return chuoi_ngay, None, 1.0
    for fmt in CAC_DINH_DANG_NGAY:
        da_parse = pd.to_datetime(chuoi_ngay, format=fmt, errors="coerce")
        ty_le_hong = float(da_parse.isna().sum() - chuoi_ngay.isna().sum()) / max(len(khong_thieu), 1)
        if ty_le_hong <= 0.001:
            return da_parse, fmt, max(ty_le_hong, 0.0)
    da_parse = pd.to_datetime(chuoi_ngay, errors="coerce")
    ty_le_hong = float((da_parse.isna() & chuoi_ngay.notna()).sum()) / max(len(khong_thieu), 1)
    return da_parse, None, ty_le_hong


# ---------------------------------------------------------------------------
# Kiểm tra biến mục tiêu và cột chia tập (theo cùng quy tắc Step 2/Step 3)
# ---------------------------------------------------------------------------

def kiem_tra_muc_tieu(df: pd.DataFrame, target_col: str = TARGET_COL) -> KetQuaKiemTra:
    kq = KetQuaKiemTra()
    if target_col not in df.columns:
        kq.blockers.append(f"Không tìm thấy cột biến mục tiêu '{target_col}'.")
        return kq
    s = df[target_col]
    so_thieu = int(s.isna().sum())
    if so_thieu > 0:
        kq.warnings.append(
            f"Cột '{target_col}' thiếu {so_thieu:,} giá trị. Step 2 sẽ loại các dòng này; "
            "Step 3 sẽ dừng lỗi nếu còn giá trị thiếu trong tệp WOE đầu vào của nó."
        )
    gia_tri = sorted({str(v) for v in s.dropna().unique().tolist()})[:10]
    kq.chi_tiet["gia_tri_muc_tieu"] = gia_tri
    dang_so = pd.to_numeric(s, errors="coerce")
    tap_hop = {int(v) for v in dang_so.dropna().unique().tolist() if float(v).is_integer()}
    if dang_so.isna().sum() > so_thieu or not tap_hop.issubset({0, 1}):
        kq.blockers.append(
            f"Cột '{target_col}' phải là nhị phân 0/1. Giá trị phân biệt phát hiện: "
            f"{', '.join(gia_tri) if gia_tri else '(rỗng)'}. "
            "Theo quy trình gốc, nhãn 1 tương ứng NHOM_NO >= 3 tại Step 1."
        )
    elif tap_hop != {0, 1}:
        kq.blockers.append(
            f"Cột '{target_col}' chỉ có một lớp ({', '.join(gia_tri)}). "
            "Mô hình PD cần cả hai lớp 0 và 1; hãy kiểm tra lại kỳ quan sát và định nghĩa nhãn."
        )
    else:
        kq.chi_tiet["ty_le_nhan_1"] = float((dang_so == 1).mean())
    return kq


def chuan_hoa_nhan_chia_tap(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def kiem_tra_chia_tap(df: pd.DataFrame, bat_buoc_co_test: bool = False) -> KetQuaKiemTra:
    """
    Kiểm tra cột train_test theo đúng quy tắc hàm
    validate_existing_train_test_split trong Step 2 (và khối đầu Step 3).
    Đây là kiểm tra trước của web; pipeline vẫn tự kiểm tra lại khi chạy.
    """
    kq = KetQuaKiemTra()
    if SPLIT_COL not in df.columns:
        kq.blockers.append(
            f"Không tìm thấy cột chia tập '{SPLIT_COL}'. Step 2 và Step 3 không tự chia tập; "
            "cột này phải có sẵn với nhãn 'train'/'test'."
        )
        return kq
    s_goc = df[SPLIT_COL]
    if int(s_goc.isna().sum()) > 0:
        kq.blockers.append(
            f"Cột '{SPLIT_COL}' có {int(s_goc.isna().sum()):,} giá trị thiếu (missing)."
        )
    s = chuan_hoa_nhan_chia_tap(s_goc.fillna(""))
    so_rong = int((s == "").sum())
    if so_rong > 0:
        kq.blockers.append(
            f"Cột '{SPLIT_COL}' có {so_rong:,} giá trị rỗng. Đây là trạng thái của đầu ra Step 1 "
            "khi chưa áp khối chia tập của chủ sản phẩm (lọc TT14, chia OOT 30%). "
            "Ứng dụng không tự điền cột này; hãy dùng tệp credit_data.parquet từ lần chạy "
            "Step 1 đầy đủ (đã gồm khối chia tập) hoặc bổ sung khối đó vào mã Step 1."
        )
    nhan = sorted(set(s.unique()) - {""})[:10]
    kq.chi_tiet["nhan_chia_tap"] = nhan
    nhan_sai = [x for x in nhan if x not in (TRAIN_LABEL, TEST_LABEL)]
    if nhan_sai:
        kq.blockers.append(
            f"Cột '{SPLIT_COL}' chứa nhãn ngoài 'train'/'test': {', '.join(nhan_sai)}."
        )
    co_train = TRAIN_LABEL in set(s.unique())
    co_test = TEST_LABEL in set(s.unique())
    if not co_train:
        kq.blockers.append(f"Không có dòng nào thuộc tập '{TRAIN_LABEL}'.")
    if not co_test:
        if bat_buoc_co_test:
            kq.blockers.append(
                f"Không có dòng nào thuộc tập '{TEST_LABEL}'. Step 3 yêu cầu cả hai tập đều có dòng."
            )
        else:
            kq.warnings.append(
                f"Không có dòng nào thuộc tập '{TEST_LABEL}'. Step 2 vẫn chạy phần train, "
                "nhưng Step 3 sẽ dừng lỗi vì yêu cầu có tập test; đồng thời không còn tập "
                "ngoài thời gian (OOT) để kiểm định."
            )
    if co_train and TARGET_COL in df.columns:
        y_train = pd.to_numeric(df.loc[s == TRAIN_LABEL, TARGET_COL], errors="coerce")
        if set(y_train.dropna().unique().tolist()) != {0, 1}:
            kq.blockers.append("Tập train không đủ cả hai lớp 0 và 1 của biến mục tiêu.")
    if co_train and co_test:
        kq.chi_tiet["so_dong_train"] = int((s == TRAIN_LABEL).sum())
        kq.chi_tiet["so_dong_test"] = int((s == TEST_LABEL).sum())
    return kq


# ---------------------------------------------------------------------------
# Kiểm tra độ phủ thời gian theo khai báo của người dùng
# ---------------------------------------------------------------------------

def _so_thang_giua(dau: pd.Timestamp, cuoi: pd.Timestamp) -> int:
    return (cuoi.year - dau.year) * 12 + (cuoi.month - dau.month)


def kiem_tra_phu_thoi_gian(df: pd.DataFrame, cot_ngay: str,
                           bat_dau_qs, ket_thuc_qs, ky_du_bao_thang: int) -> KetQuaKiemTra:
    """
    Đối chiếu khoảng ngày thực tế trong dữ liệu với khai báo:
    - bat_dau_qs / ket_thuc_qs: kỳ quan sát (observation window) người dùng khai;
    - ky_du_bao_thang: kỳ dự báo (performance horizon), đơn vị tháng.

    Nguyên tắc kiểm soát bên phải bị cắt (right-censoring): một hồ sơ quan sát
    tại thời điểm t chỉ có nhãn đầy đủ cho kỳ dự báo h tháng nếu dữ liệu còn
    phủ tới t + h. Hồ sơ nằm sau mốc (ngày dữ liệu cuối − h) được đếm và báo
    cáo là có nguy cơ bị cắt phải theo chính khai báo h đó.

    Lưu ý trung thực: kiểm tra này dựa trên khai báo của người dùng và cột
    ngày được chọn; nó không chứng minh cách khối chia tập (hiện vắng mặt
    trong Step 1) xử lý thời gian.
    """
    kq = KetQuaKiemTra()
    if cot_ngay not in df.columns:
        kq.blockers.append(f"Không tìm thấy cột ngày '{cot_ngay}' để kiểm tra phủ thời gian.")
        return kq
    ngay, fmt, ty_le_hong = thu_phan_tich_ngay(df[cot_ngay])
    if fmt is None and ty_le_hong > 0.001:
        kq.blockers.append(
            f"Cột ngày '{cot_ngay}' có {ty_le_hong * 100:.1f}% giá trị không phân tích được "
            "bằng các định dạng dd/mm/yyyy, yyyy-mm-dd, dd-mm-yyyy, yyyy/mm/dd. "
            "Pipeline gốc dùng định dạng dd/mm/yyyy cho dữ liệu thô Step 1."
        )
        return kq
    ngay_hop_le = ngay.dropna()
    if ngay_hop_le.empty:
        kq.blockers.append(f"Cột ngày '{cot_ngay}' không có giá trị hợp lệ.")
        return kq

    d_min, d_max = ngay_hop_le.min(), ngay_hop_le.max()
    span_thang = _so_thang_giua(d_min, d_max) + 1
    cac_thang = ngay_hop_le.dt.to_period("M").unique()
    thang_thieu = span_thang - len(cac_thang)
    kq.chi_tiet.update({
        "ngay_bat_dau": str(d_min.date()), "ngay_ket_thuc": str(d_max.date()),
        "so_thang_phu": int(span_thang), "so_thang_co_du_lieu": int(len(cac_thang)),
        "so_thang_trong_khoang_khong_co_du_lieu": int(max(thang_thieu, 0)),
    })
    if thang_thieu > 0:
        kq.warnings.append(
            f"Trong khoảng {d_min.date()} đến {d_max.date()} có {thang_thieu} tháng không có "
            "dòng dữ liệu nào. Hãy xác nhận đây là chủ đích trích xuất, không phải thiếu kỳ."
        )

    qs_dau = pd.Timestamp(bat_dau_qs)
    qs_cuoi = pd.Timestamp(ket_thuc_qs)
    if qs_cuoi < qs_dau:
        kq.blockers.append("Ngày kết thúc kỳ quan sát khai báo sớm hơn ngày bắt đầu.")
        return kq
    if d_min > qs_dau:
        kq.blockers.append(
            f"Dữ liệu bắt đầu {d_min.date()}, muộn hơn mốc bắt đầu kỳ quan sát đã khai báo "
            f"{qs_dau.date()}. Thiếu phủ đầu kỳ quan sát; hãy bổ sung dữ liệu hoặc sửa khai báo."
        )
    # Mốc quan sát muộn nhất còn đủ kỳ dự báo trong dữ liệu
    moc_day_du = d_max - pd.DateOffset(months=int(ky_du_bao_thang))
    if qs_cuoi > moc_day_du:
        kq.blockers.append(
            f"Kỳ quan sát khai báo kết thúc {qs_cuoi.date()} nhưng dữ liệu chỉ đến "
            f"{d_max.date()}; với kỳ dự báo {ky_du_bao_thang} tháng, mốc quan sát muộn nhất "
            f"còn quan sát trọn kỳ dự báo là {moc_day_du.date()}. Thiếu phủ thời gian cho "
            "kỳ dự báo: hãy bổ sung dữ liệu các tháng sau, rút ngắn kỳ quan sát, hoặc giảm "
            "kỳ dự báo đã khai báo."
        )
    so_cat_phai = int((ngay_hop_le > moc_day_du).sum())
    ty_le_cat = so_cat_phai / max(len(ngay_hop_le), 1)
    kq.chi_tiet["so_dong_nguy_co_cat_phai"] = so_cat_phai
    kq.chi_tiet["ty_le_cat_phai"] = round(float(ty_le_cat), 4)
    if so_cat_phai > 0:
        kq.warnings.append(
            f"{so_cat_phai:,} dòng ({ty_le_cat * 100:.1f}%) có ngày quan sát sau "
            f"{moc_day_du.date()} nên chưa đủ {ky_du_bao_thang} tháng theo dõi kết quả "
            "(nguy cơ right-censoring theo khai báo). Cách xử lý các dòng này thuộc khối "
            "chuẩn bị dữ liệu/chia tập của chủ sản phẩm trong Step 1 — khối hiện chưa có "
            "trong tệp đã truy cập, vì vậy ứng dụng không tự loại hay giữ chúng."
        )

    # Đối chiếu cohort OOT nếu đã có cột chia tập hợp lệ
    if SPLIT_COL in df.columns:
        s = chuan_hoa_nhan_chia_tap(df[SPLIT_COL].fillna(""))
        ngay_train = ngay[s == TRAIN_LABEL].dropna()
        ngay_test = ngay[s == TEST_LABEL].dropna()
        if not ngay_train.empty and not ngay_test.empty:
            kq.chi_tiet["khoang_ngay_train"] = f"{ngay_train.min().date()} → {ngay_train.max().date()}"
            kq.chi_tiet["khoang_ngay_test"] = f"{ngay_test.min().date()} → {ngay_test.max().date()}"
            if ngay_test.min() <= ngay_train.max():
                kq.warnings.append(
                    "Khoảng thời gian của tập test chồng lấn tập train. Nếu tập test được kỳ "
                    "vọng là ngoài thời gian (OOT, giai đoạn muộn hơn), hãy rà soát lại cách "
                    "chia tập; ứng dụng chỉ cảnh báo, không tự chia lại."
                )
    return kq


# ---------------------------------------------------------------------------
# Xem trước an toàn và thống kê gộp cho biểu đồ
# ---------------------------------------------------------------------------

def _che_gia_tri(v, che_hoan_toan: bool = False) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    chuoi = str(v)
    if che_hoan_toan or len(chuoi) <= 4:
        return "***"
    return "***" + chuoi[-4:]


def xem_truoc_an_toan(df: pd.DataFrame, so_dong: int = 5) -> pd.DataFrame:
    """Bản xem trước vài dòng đầu; cột định danh/họ tên được che."""
    xem = df.head(so_dong).copy()
    for cot in COT_DINH_DANH:
        if cot in xem.columns:
            xem[cot] = xem[cot].map(
                lambda v, full=(cot in COT_CHE_HOAN_TOAN): _che_gia_tri(v, full))
    return xem


def thong_ke_theo_thang(df: pd.DataFrame, cot_ngay: str = DATE_COL,
                        target_col: str = TARGET_COL) -> pd.DataFrame:
    """Thống kê gộp theo tháng: số dòng và tỷ lệ nhãn 1 (nếu có cột mục tiêu)."""
    if cot_ngay not in df.columns:
        return pd.DataFrame()
    ngay, _, _ = thu_phan_tich_ngay(df[cot_ngay])
    khung = pd.DataFrame({"Tháng": ngay.dt.to_period("M").astype(str)})
    if target_col in df.columns:
        khung["nhan"] = pd.to_numeric(df[target_col], errors="coerce")
        gop = khung.groupby("Tháng").agg(
            **{"Số dòng": ("nhan", "size"), "Tỷ lệ NX=1": ("nhan", "mean")}
        ).reset_index()
    else:
        gop = khung.groupby("Tháng").size().reset_index(name="Số dòng")
    return gop.sort_values("Tháng").reset_index(drop=True)
