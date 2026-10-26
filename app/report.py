# -*- coding: utf-8 -*-
"""
Sinh báo cáo tổng hợp một lần chạy (Excel và HTML).

Báo cáo ghi lại: đầu vào ở dạng không nhận diện cá nhân (điểm vào, số tệp,
định dạng, kích thước, số dòng/cột), cấu hình, các bước đã chạy và trạng thái,
kết quả đọc từ artifact của pipeline (chỉ số, hệ số, tham số điểm — số liệu
gộp cấp mô hình), kiểm tra đã thực hiện, cảnh báo và giới hạn, thời điểm tạo.
Phần tính được được tách bạch khỏi phần chưa kiểm định; báo cáo không chứa bất
kỳ dòng dữ liệu hồ sơ hay giá trị định danh nào.
"""

from __future__ import annotations

import html
import io
from datetime import datetime
from pathlib import Path

import pandas as pd

from app import adapters, config

# Hằng số pipeline hiển thị trong báo cáo — chép từ mã nguồn đã truy cập để
# người đọc đối chiếu. Ứng dụng không thay đổi các giá trị này; nếu chủ sản
# phẩm sửa mã gốc, hàm băm SHA-256 trong báo cáo sẽ lệch khỏi bản đã truy xuất
# và phần này cần được cập nhật tương ứng.
THAM_SO_STEP2 = [
    ("IV_THRESHOLD", "0.1"), ("CORR_THRESHOLD", "0.89"), ("VIF_THRESHOLD", "5"),
    ("MIN_BIN_SIZE", "0.05"), ("MAX_BINS / MIN_BINS", "10 / 2"),
    ("MONOTONIC_ENFORCE", "True"), ("PSI_THRESHOLD", "0.1"),
    ("MIN_BIN_COUNT", "50"), ("MIN_EVENTS_PER_BIN / MIN_NON_EVENTS_PER_BIN", "10 / 10"),
    ("IV_STABILITY_N_BOOTSTRAP", "50"), ("IV_STABILITY_CV_THRESHOLD", "0.1"),
]
THAM_SO_STEP3 = [
    ("Thuật toán", "LogisticRegression, solver newton-cholesky (cố định), random_state=42"),
    ("Lưới scaler", "StandardScaler, RobustScaler, MinMaxScaler"),
    ("Lưới C (cơ sở)", "1.0, 0.1, 10.0, 0.01; mở rộng quanh cấu hình tốt nhất, trần 80 cấu hình"),
    ("Penalty", "l2 hoặc None (tự dò khả năng hỗ trợ của solver)"),
    ("Class weight", "None, balanced, {0:1.0, 1:n0/n1} tính động trên train"),
    ("Quét ngưỡng", "0.05 → 0.92, bước 0.0005, chỉ trên train; quy tắc gap-margin"),
    ("9 cổng KPI", "Sensitivity≥0.70; MCC≥0.68; Specificity≥0.95; Precision≥0.70; Gap>0; BCE≤0.15; NPV≥0.95; F1≥0.70; Accuracy≥0.95"),
    ("Thang điểm", "min 300, max 850, base odds 50; PDO suy từ khoảng PD của train (công thức Siddiqi)"),
    ("Hạng (grade)", "10 hạng 0.AAA … 9.D chia đều khoảng điểm của train"),
    ("Hiệu chỉnh", "Platt 2 tầng (CV 10-fold trên train + lưới δ, s), chỉ fit trên train"),
]


def _doc_excel(path: Path, sheet=None):
    try:
        if not Path(path).exists():
            return None
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0,
                             engine="openpyxl")
    except Exception:
        return None


def doc_ket_qua_pipeline(run_dir: Path) -> dict:
    """Đọc các artifact bảng biểu chính (nếu có) từ thư mục model của lần chạy."""
    m = adapters.thu_muc_model(Path(run_dir))
    return {
        "metrics_17": _doc_excel(m / "plot/metrics_summary.xlsx", "Metrics_17"),
        "he_so": _doc_excel(m / "parameter_model.xlsx"),
        "tham_so_diem": _doc_excel(m / "score_parameter.xlsx"),
        "tham_so_diem_new": _doc_excel(m / "score_parameter_new.xlsx"),
        "pd_decile": _doc_excel(m / "PD decile.xlsx"),
        "pd_decile_new": _doc_excel(m / "PD decile_new.xlsx"),
        "grade_score": _doc_excel(m / "Grade_score.xlsx"),
        "grade_score_new": _doc_excel(m / "Grade_score_new.xlsx"),
        "gini_ks_summary": _doc_excel(m / "Gini_KS_result.xlsx", "Summary"),
        "psi_tong_hop": _doc_excel(m / "plot/psi_detail.xlsx", "PSI_tong_hop"),
        "iv_per_feature": _doc_excel(m / "plot/iv_detail.xlsx", "IV_per_feature"),
    }


def _bang_thong_tin_chung(manifest: dict) -> pd.DataFrame:
    dong = [
        ("Sản phẩm", config.TEN_SAN_PHAM),
        ("Mã lần chạy (run_id)", manifest.get("run_id", "")),
        ("Thời điểm tạo báo cáo", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Điểm vào dữ liệu", manifest.get("diem_vao", "")),
        ("Dữ liệu giả lập minh họa", "CÓ — kết quả chỉ minh họa kỹ thuật, không phải thực nghiệm"
         if manifest.get("du_lieu_gia_lap") else "Không (dữ liệu do người dùng tải lên)"),
    ]
    khai_bao = manifest.get("khai_bao", {}) or {}
    for khoa, nhan in [("cot_muc_tieu", "Cột mục tiêu đã khai báo"),
                       ("cot_ngay", "Cột ngày quan sát đã khai báo"),
                       ("bat_dau_qs", "Bắt đầu kỳ quan sát (khai báo)"),
                       ("ket_thuc_qs", "Kết thúc kỳ quan sát (khai báo)"),
                       ("ky_du_bao_thang", "Kỳ dự báo đã khai báo (tháng)")]:
        if khai_bao.get(khoa) not in (None, ""):
            dong.append((nhan, str(khai_bao[khoa])))
    for step, info in (manifest.get("pipeline_hash", {}) or {}).items():
        dong.append((f"SHA-256 mã gốc {step}",
                     f"{info.get('sha256') or 'thiếu tệp'} — "
                     f"{'khớp bản đã truy xuất' if info.get('khop') else 'LỆCH bản đã truy xuất'}"))
    for pkg, ver in (manifest.get("phien_ban_thu_vien", {}) or {}).items():
        dong.append((f"Thư viện {pkg}", str(ver)))
    return pd.DataFrame(dong, columns=["Hạng mục", "Giá trị"])


def _bang_cac_buoc(manifest: dict) -> pd.DataFrame:
    dong = []
    for buoc in manifest.get("cac_buoc", []) or []:
        dong.append({
            "Bước": buoc.get("step", ""),
            "Trạng thái": "Hoàn tất" if buoc.get("thanh_cong") else "Chưa hoàn tất / bị chặn",
            "Thời gian (giây)": round(float(buoc.get("thoi_gian_giay") or 0), 1),
            "Mã thoát": buoc.get("ma_thoat"),
            "Thông điệp": buoc.get("thong_diep", ""),
        })
    return pd.DataFrame(dong)


PHAN_CHUA_KIEM_DINH_MAC_DINH = [
    "Khối lọc dữ liệu TT14 và khối chia train/OOT 30% trong Step 1: vắng mặt trong mã "
    "đã truy cập; ứng dụng không tự thực hiện và không xác minh được logic của khối này.",
    "Tương đương kết quả với lần chạy gốc trên Windows: chưa xác minh (chưa đối chiếu "
    "cùng đầu vào, cùng cấu hình, bước trung gian và đầu ra với lần chạy gốc).",
    "Tuân thủ pháp luật/Thông tư 14/2025/TT-NHNN và yêu cầu của Ngân hàng Nhà nước: "
    "chưa xác minh pháp lý; cần chuyên gia pháp lý và bộ phận tuân thủ rà soát trên "
    "văn bản chính thức còn hiệu lực trước mọi tuyên bố.",
    "Chất lượng mô hình cho sử dụng thực tế (phê duyệt tín dụng, trích lập, vốn): các "
    "chỉ số trong báo cáo là số đo trên dữ liệu của lần chạy này theo định nghĩa trong "
    "mã; chúng không tự nó chứng minh mô hình đủ điều kiện vận hành. Cần thẩm định mô "
    "hình độc lập theo quy trình nội bộ của tổ chức.",
    "Bảo mật hệ thống ở môi trường triển khai cụ thể: chưa được kiểm thử xâm nhập; báo "
    "cáo này chỉ mô tả biện pháp đã cài trong mã ứng dụng, không cam kết an toàn tuyệt đối.",
]


def tao_bao_cao_excel(run_dir: Path) -> bytes:
    manifest = adapters.doc_manifest(Path(run_dir))
    kq = doc_ket_qua_pipeline(Path(run_dir))
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        _bang_thong_tin_chung(manifest).to_excel(w, sheet_name="Tong_quan", index=False)
        dau_vao = pd.DataFrame(manifest.get("dau_vao", []) or [])
        dau_vao.to_excel(w, sheet_name="Dau_vao", index=False)
        kiem_tra = manifest.get("kiem_tra", {}) or {}
        pd.DataFrame({
            "Loại": (["Lỗi chặn"] * len(kiem_tra.get("blockers", []))
                     + ["Cảnh báo"] * len(kiem_tra.get("warnings", []))),
            "Nội dung": (list(kiem_tra.get("blockers", []))
                         + list(kiem_tra.get("warnings", []))),
        }).to_excel(w, sheet_name="Kiem_tra_truoc", index=False)
        pd.DataFrame(THAM_SO_STEP2, columns=["Tham số Step 2", "Giá trị"]).to_excel(
            w, sheet_name="Cau_hinh_Step2", index=False)
        pd.DataFrame(THAM_SO_STEP3, columns=["Hạng mục Step 3", "Giá trị"]).to_excel(
            w, sheet_name="Cau_hinh_Step3", index=False)
        _bang_cac_buoc(manifest).to_excel(w, sheet_name="Cac_buoc_da_chay", index=False)
        ten_sheet = {
            "metrics_17": "Metrics_17", "he_so": "He_so_mo_hinh",
            "tham_so_diem": "Tham_so_diem", "tham_so_diem_new": "Tham_so_diem_new",
            "pd_decile": "PD_decile", "pd_decile_new": "PD_decile_new",
            "grade_score": "Grade_score", "grade_score_new": "Grade_score_new",
            "gini_ks_summary": "Gini_KS_summary", "psi_tong_hop": "PSI_tong_hop",
            "iv_per_feature": "IV_theo_bien",
        }
        for khoa, sheet in ten_sheet.items():
            df = kq.get(khoa)
            if df is not None:
                df.to_excel(w, sheet_name=sheet[:31], index=False)
        pd.DataFrame({"Nội dung chưa kiểm định": PHAN_CHUA_KIEM_DINH_MAC_DINH}).to_excel(
            w, sheet_name="Chua_kiem_dinh", index=False)
    return buf.getvalue()


def _bang_html(df, tieu_de: str) -> str:
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return f"<h3>{html.escape(tieu_de)}</h3><p><i>Không có dữ liệu (bước tạo artifact này chưa chạy hoặc chưa hoàn tất).</i></p>"
    return (f"<h3>{html.escape(tieu_de)}</h3>"
            + df.to_html(index=False, border=0, classes="bang", escape=True))


def tao_bao_cao_html(run_dir: Path) -> str:
    manifest = adapters.doc_manifest(Path(run_dir))
    kq = doc_ket_qua_pipeline(Path(run_dir))
    kiem_tra = manifest.get("kiem_tra", {}) or {}
    canh_bao_html = "".join(
        f"<li><b>[Chặn]</b> {html.escape(str(x))}</li>" for x in kiem_tra.get("blockers", []))
    canh_bao_html += "".join(
        f"<li>{html.escape(str(x))}</li>" for x in kiem_tra.get("warnings", []))
    chua_kiem_dinh = "".join(f"<li>{html.escape(x)}</li>" for x in PHAN_CHUA_KIEM_DINH_MAC_DINH)
    banner = ""
    if manifest.get("du_lieu_gia_lap"):
        banner = ("<div class='banner'>DỮ LIỆU GIẢ LẬP MINH HỌA — mọi số liệu trong báo cáo "
                  "này chỉ chứng minh đường ống kỹ thuật, không phải kết quả thực nghiệm.</div>")
    return f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Báo cáo lần chạy mô hình PD — {html.escape(str(manifest.get('run_id','')))}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:16px;line-height:1.5;color:#1a1a1a}}
.bang{{border-collapse:collapse;overflow-x:auto;display:block;max-width:100%}}
.bang th,.bang td{{border:1px solid #ccc;padding:4px 8px;text-align:left;font-size:13px}}
.banner{{background:#fff3cd;border:1px solid #e0a800;padding:10px;border-radius:6px;font-weight:600}}
h1{{font-size:20px}} h3{{margin-top:24px}}
</style></head><body>
<h1>Báo cáo lần chạy mô hình PD (hồi quy logistic)</h1>
{banner}
{_bang_html(_bang_thong_tin_chung(manifest), "1. Thông tin chung và truy vết")}
{_bang_html(pd.DataFrame(manifest.get('dau_vao', []) or []), "2. Đầu vào (không nhận diện cá nhân)")}
<h3>3. Kiểm tra trước khi chạy</h3>
<ul>{canh_bao_html or '<li>Không ghi nhận lỗi chặn hay cảnh báo nào ở tầng kiểm tra trước.</li>'}</ul>
{_bang_html(_bang_cac_buoc(manifest), "4. Các bước đã chạy")}
{_bang_html(kq.get('metrics_17'), "5. Bộ chỉ số Metrics_17 (artifact Step 3)")}
{_bang_html(kq.get('he_so'), "6. Hệ số mô hình (parameter_model.xlsx)")}
{_bang_html(kq.get('tham_so_diem_new') if kq.get('tham_so_diem_new') is not None else kq.get('tham_so_diem'), "7. Tham số thang điểm")}
{_bang_html(kq.get('pd_decile_new') if kq.get('pd_decile_new') is not None else kq.get('pd_decile'), "8. Khoảng PD theo decile (train)")}
{_bang_html(kq.get('grade_score_new') if kq.get('grade_score_new') is not None else kq.get('grade_score'), "9. Khoảng điểm theo hạng (grade, train)")}
{_bang_html(kq.get('gini_ks_summary'), "10. Tổng hợp Gini/KS theo hạng (test)")}
{_bang_html(kq.get('psi_tong_hop'), "11. PSI tổng hợp")}
{_bang_html(kq.get('iv_per_feature'), "12. IV theo biến")}
<h3>13. Phần chưa kiểm định và giới hạn</h3>
<ul>{chua_kiem_dinh}</ul>
<p><i>Báo cáo này được sinh tự động từ artifact của lần chạy. Nó không phải văn bản phê
duyệt mô hình và không thay thế thẩm định độc lập theo quy trình quản trị rủi ro của tổ chức.</i></p>
</body></html>"""
