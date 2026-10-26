# -*- coding: utf-8 -*-
"""
CẤU HÌNH SẢN PHẨM — chủ sản phẩm chỉnh sửa trực tiếp tại tệp này.

Mọi tên gọi, mô tả, gói giá hiển thị trên web đều lấy từ đây (ngoại trừ các
hằng số nghiệp vụ của pipeline Step 1-3, vốn nằm nguyên vẹn trong thư mục
pipeline/ và không được ứng dụng thay đổi).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Định danh sản phẩm [GIẢ ĐỊNH 3 — chủ sản phẩm đổi tên tại đây]
# ---------------------------------------------------------------------------
TEN_SAN_PHAM = "PD Logistic VN"
"""
Tên hiển thị của ứng dụng, mặc định mang tính mô tả kỹ thuật. Chủ sản phẩm sửa
trước khi giới thiệu ra bên ngoài; không dùng tên gây hiểu nhầm là sản phẩm
được cơ quan quản lý phê duyệt.
"""

TIEU_DE_PHU = "Ứng dụng web xây dựng mô hình PD bằng hồi quy logistic"
"""
Một PD là xác suất khách hàng thuộc nhóm nợ dùng để xây dựng biến mục tiêu
(default) trong một kỳ dự báo xác định.
"""

MO_TA_NGAN = (
    "Công cụ hỗ trợ chuyên viên xây dựng mô hình chấm điểm PD (xác suất vỡ nợ) "
    "theo quy trình 3 bước có sẵn: làm dữ liệu, lọc biến bằng WOE/IV và hồi quy "
    "logistic, chấm điểm và kiểm định. Ứng dụng chạy đúng mã Step 1–3 trong "
    "thư mục pipeline/ của repository; phần chưa có trong mã nguồn được chặn và "
    "ghi rõ, không tự thay thế."
)

GHI_CHU_LIEN_HE = (
    "[THÔNG TIN LIÊN HỆ DO CHỦ SẢN PHẨM CẤU HÌNH]"
)

# ---------------------------------------------------------------------------
# Gói giá — do chủ sản phẩm ĐỀ XUẤT, KHÔNG phải giá thị trường đã kiểm chứng
# ---------------------------------------------------------------------------
SO_LUOT_MIEN_PHI = 5  # 5 lượt đầu miễn phí

GIA_MOI_LUOT_VND_2026 = 18_000_000
"""
Giá cho mỗi lượt từ lượt thứ 6 trở đi: 18.000.000 VND/lượt (giá cơ sở năm 2026).

Đây là mức giá chủ sản phẩm đề xuất cho sản phẩm của mình, KHÔNG phải giá thị
trường đã được kiểm chứng và không dựa trên khảo sát giá đối thủ nào được lưu
trong repo này.
"""

NAM_GIA_CO_SO = 2026

CONG_THUC_DIEU_CHINH_GIA = "Giá năm y = Giá năm (y-1) × (1 + tỷ lệ lạm phát năm y)"
"""
Công thức điều chỉnh giá theo lạm phát do chủ sản phẩm đặt ra.

Tỷ lệ lạm phát của từng năm phải lấy từ một nguồn thống kê chính thức, đúng
năm, đúng định nghĩa chỉ số, kèm liên kết truy xuất được. Khi chưa có nguồn
đạt yêu cầu đó cho một năm cụ thể, KHÔNG điền tỷ lệ và KHÔNG suy ra giá của
năm đó; bảng giá trên web để trống các năm chưa có nguồn.
"""

GIA_THEO_NAM = {
    # year: (ty_le_lam_phat, gia_vnd, nguon_lien_ket_that)
    # Chỉ thêm dòng khi đã có nguồn lạm phát chính thức kèm liên kết thật,
    # đúng năm và đúng cách tính của chỉ số. Để trống khi chưa xác minh.
    # Ví dụ định dạng khi đã có nguồn:
    # 2027: (0.0, 0, "[CHỦ SẢN PHẨM ĐIỀN LIÊN KẾT NGUỒN LẠM PHÁT CHÍNH THỨC]"),
}
"""Hàm ý về giới hạn: ứng dụng chỉ HIỂN THỊ bảng giá. Ứng dụng không thu phí,
không xác nhận thanh toán, không tự động khóa lượt chạy theo giá; việc đếm lượt
và thu tiền (nếu có) do chủ sản phẩm thực hiện ngoài ứng dụng này.

[GIẢ ĐỊNH 1] Một "lượt" tạm hiểu là một lần chạy trọn gói đến hết Step 3 cho
một bộ dữ liệu. Chủ sản phẩm cần xác nhận định nghĩa này trước khi áp dụng
thương mại; có thể đảo ngược bằng cách sửa tài liệu/config mà không đổi mã
pipeline.
"""

THONG_TIN_THANH_TOAN = "[THÔNG TIN THANH TOÁN DO CHỦ SẢN PHẨM CẤU HÌNH]"
"""Cố ý để dạng giữ chỗ. Không nhúng số tài khoản hay thông tin thanh toán cá
nhân vào repository công khai."""

# ---------------------------------------------------------------------------
# Giới hạn tải tệp và thực thi
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = int(os.environ.get("PD_MAX_UPLOAD_MB", "200"))
"""
Dung lượng tối đa cho MỘT tệp tải lên, đơn vị MB. [GIẢ ĐỊNH 2] Giá trị mặc định
200 MB — bằng giới hạn cấu hình trong .streamlit/config.toml. Chủ sản phẩm có
thể giảm xuống tùy hạ tầng; khi đổi, phải đổi cả hai nơi (config này và
.streamlit/config.toml, khóa server.maxUploadSize) để thông báo trên web khớp
với giới hạn thực enforced.

Ứng dụng KHÔNG tuyên bố hỗ trợ mọi dung lượng; tệp vượt giới hạn bị từ chối
trước khi phân tích.
"""

MAX_FILES_STEP1 = 24
"""Số tệp SKV tối đa trong một lần tải cho Step 1 (giới hạn tài nguyên phiên)."""

MAX_PARTS_CREDIT_DATA = 8
"""Số phần tối đa khi tải credit_data dạng chia nhỏ (ví dụ credit_data1/2)."""

DINH_DANG_HO_TRO = ("parquet", "csv", "xlsx", "txt")
"""
Định dạng tệp được chấp nhận. Parquet chỉ khả dụng khi thư viện pyarrow được
cài trong môi trường (ứng dụng tự kiểm tra lúc khởi động và thông báo nếu
thiếu). Tuyệt đối không chấp nhận định dạng pickle/joblib vì nạp đối tượng tùy
ý từ tệp không đáng tin là không an toàn.
"""

TIMEOUT_STEP_GIAY = int(os.environ.get("PD_TIMEOUT_STEP_GIAY", "1800"))
"""Thời gian tối đa cho một bước pipeline (giây). Hết hạn, tiến trình bị dừng
và lần chạy được ghi nhận là chưa hoàn tất, kèm nhật ký đã có."""

SO_DONG_XEM_TRUOC = 5
"""Số dòng tối đa hiển thị ở phần xem trước (các cột định danh luôn được che)."""

# ---------------------------------------------------------------------------
# Thư mục làm việc của ứng dụng
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
RUNS_DIR = Path(os.environ.get("PD_RUN_DIR", REPO_ROOT / "runs"))
"""
Thư mục chứa các lần chạy (runs/<run_id>/). Mỗi lần chạy có thư mục riêng chứa
bản sao thời gian chạy của mã gốc, đầu vào đã đặt tên an toàn, đầu ra và nhật
ký. Thư mục này nằm trong .gitignore. Dữ liệu trong đó vẫn tồn tại trên máy
chủ cho đến khi người dùng bấm xóa dữ liệu phiên hoặc quá hạn dọn dẹp dưới
đây; ứng dụng không tuyên bố tự xóa ngoài hai cơ chế đã cài đặt đó.
"""

DON_DEP_RUN_SAU_GIO = float(os.environ.get("PD_DON_DEP_RUN_SAU_GIO", "24"))
"""Khi ứng dụng khởi động, các thư mục chạy cũ hơn số giờ này bị xóa. Cơ chế
này là best-effort (chỉ chạy khi có lượt truy cập khởi động ứng dụng) và được
mô tả đúng như vậy trong tài liệu, không phải cam kết xóa theo lịch."""

# Tên tệp mã gốc (giữ nguyên tên do chủ sản phẩm đặt)
STEP1_FILE = PIPELINE_DIR / "step 1 làm dữ liệu mô hình.py"
STEP2_FILE = PIPELINE_DIR / "step 2 lọc biến mô hình.py"
STEP3_FILE = PIPELINE_DIR / "step 3 modelling V2.py"

# Hàm băm SHA-256 kỳ vọng của mã gốc (đối chiếu pipeline/NGUON_GOC.md).
# Ứng dụng kiểm tra khi khởi động; sai lệch = mã gốc đã bị thay đổi so với
# bản đã truy xuất, ứng dụng cảnh báo và không khẳng định giữ nguyên quy trình.
STEP_HASHES = {
    "step1": "d3870cd7806f6eb2dc2e6fdee8343ba1cafea2b328ffeb52264c6e9270baee44",
    "step2": "8c2e810a0ba2a870c0d02d608da5b06ba37866a72500f9eda89d8d4432b40294",
    "step3": "6d17d71f8afdabc301e42cb720c800022b2e345230164bda03a3337219b4fd84",
}
STEP_FILES = {"step1": STEP1_FILE, "step2": STEP2_FILE, "step3": STEP3_FILE}
