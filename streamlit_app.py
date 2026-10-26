# -*- coding: utf-8 -*-
"""
Ứng dụng web mô hình PD bằng hồi quy logistic.

Ứng dụng là lớp giao diện + kiểm tra trước + bộ chuyển đổi quanh mã gốc
Step 1–3 trong thư mục pipeline/ (giữ nguyên từng byte, kiểm chứng bằng
SHA-256 lúc khởi động). Mọi tính toán mô hình (WOE/IV, chọn biến, hồi quy
logistic, ngưỡng, chấm điểm, kiểm định) do mã gốc thực hiện trong tiến trình
con; ứng dụng chỉ thay đường dẫn tệp cho phù hợp thư mục chạy, đặt đầu vào,
theo dõi tiến trình và trình bày artifact.

Phần vắng mặt trong mã đã truy cập — khối lọc TT14 và chia train/OOT trong
Step 1 — được chặn và nêu rõ, ứng dụng không tự viết thay.
"""

from __future__ import annotations

import io
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app import adapters, config, report as report_mod, synthetic, validation as val

st.set_page_config(page_title=f"{config.TEN_SAN_PHAM} — Mô hình PD",
                   page_icon="📊", layout="centered")

# ---------------------------------------------------------------------------
# Khởi tạo phiên và kiểm tra lúc khởi động
# ---------------------------------------------------------------------------

if "khoi_dong_xong" not in st.session_state:
    adapters.don_dep_runs_cu(config.DON_DEP_RUN_SAU_GIO)
    st.session_state["khoi_dong_xong"] = True


@st.cache_data(show_spinner=False)
def _kiem_tra_hash():
    return adapters.kiem_tra_nguyen_ven_pipeline()


HASH_KQ = _kiem_tra_hash()
CO_PYARROW = adapters.pyarrow_kha_dung()
DINH_DANG_DUNG_DUOC = tuple(
    d for d in config.DINH_DANG_HO_TRO if d != "parquet" or CO_PYARROW
)

for khoa, mac_dinh in [
    ("run_dir", None), ("diem_vao", None), ("df", None), ("skv_file_bytes", []),
    ("skv_df_xem_truoc", None), ("kiem_tra", None), ("khai_bao", {}),
    ("anh_xa_cot", {}), ("buoc", {}), ("du_lieu_gia_lap", False),
    ("dau_vao_meta", []),
]:
    st.session_state.setdefault(khoa, mac_dinh)


def _phien_ban_thu_vien() -> dict:
    phien_ban = {}
    for pkg in ("pandas", "numpy", "scipy", "sklearn", "matplotlib", "pyarrow", "openpyxl"):
        try:
            phien_ban[pkg] = __import__(pkg).__version__
        except Exception:
            phien_ban[pkg] = "không có trong môi trường"
    return phien_ban


def _dam_bao_run_dir() -> Path:
    if not st.session_state["run_dir"]:
        run_dir = adapters.tao_run_dir()
        st.session_state["run_dir"] = str(run_dir)
        adapters.ghi_manifest(run_dir, {
            "run_id": run_dir.name,
            "tao_luc": datetime.now().isoformat(timespec="seconds"),
            "pipeline_hash": HASH_KQ,
            "phien_ban_thu_vien": _phien_ban_thu_vien(),
        })
    return Path(st.session_state["run_dir"])


def _ghi_manifest_phien():
    run_dir = _dam_bao_run_dir()
    kt = st.session_state.get("kiem_tra")
    adapters.ghi_manifest(run_dir, {
        "diem_vao": st.session_state.get("diem_vao"),
        "du_lieu_gia_lap": bool(st.session_state.get("du_lieu_gia_lap")),
        "dau_vao": st.session_state.get("dau_vao_meta", []),
        "anh_xa_cot": {k: v for k, v in (st.session_state.get("anh_xa_cot") or {}).items() if k != v},
        "khai_bao": st.session_state.get("khai_bao", {}),
        "kiem_tra": ({
            "blockers": list(kt.blockers), "warnings": list(kt.warnings),
            "chi_tiet": {str(k): (v if isinstance(v, (int, float, str, bool, list)) else str(v))
                         for k, v in (kt.chi_tiet or {}).items()},
        } if kt is not None else {}),
    })


def _hien_loi_va_canh_bao(kt):
    if kt is None:
        return
    for loi in kt.blockers:
        st.error("⛔ " + loi)
    for cb in kt.warnings:
        st.warning("⚠️ " + cb)
    if kt.ok and not kt.warnings:
        st.success("Không phát hiện lỗi chặn hay cảnh báo ở tầng kiểm tra trước. "
                   "Lưu ý: bản thân Step 2/Step 3 vẫn tự kiểm tra lại khi chạy.")
    elif kt.ok:
        st.success("Không có lỗi chặn. Hãy đọc các cảnh báo ở trên trước khi chạy.")


# ---------------------------------------------------------------------------
# Điều hướng
# ---------------------------------------------------------------------------

st.sidebar.markdown(f"## {config.TEN_SAN_PHAM}")
st.sidebar.caption(config.TIEU_DE_PHU)
TRANG = st.sidebar.radio(
    "Chọn mục",
    ["Giới thiệu", "Tải dữ liệu & kiểm tra", "Chạy mô hình (Step 1–3)",
     "Kết quả & báo cáo", "Gói giá", "Hướng dẫn & triển khai"],
)

_hash_xau = [s for s, info in HASH_KQ.items() if not info.get("khop")]
if _hash_xau:
    st.sidebar.error(
        "Mã pipeline của " + ", ".join(_hash_xau) + " LỆCH hàm băm so với bản đã "
        "truy xuất từ Drive. Ứng dụng không khẳng định quy trình được giữ nguyên "
        "cho tới khi sự lệch này được giải thích."
    )
if st.session_state.get("run_dir"):
    st.sidebar.caption(f"Thư mục chạy hiện tại: `{Path(st.session_state['run_dir']).name}`")

# ===========================================================================
# TRANG GIỚI THIỆU
# ===========================================================================
if TRANG == "Giới thiệu":
    st.title(f"📊 {config.TEN_SAN_PHAM}")
    st.markdown(f"**{config.TIEU_DE_PHU}**")
    st.write(config.MO_TA_NGAN)
    st.info(
        "**PD** (Probability of Default — xác suất vỡ nợ) là xác suất một khách "
        "hàng/khoản vay rơi vào trạng thái vỡ nợ theo định nghĩa đã chọn trong "
        "một kỳ dự báo. Trong quy trình của repo này, nhãn vỡ nợ là biến `NX` = 1 "
        "khi nhóm nợ `NHOM_NO` ≥ 3 tại Step 1."
    )

    st.subheader("Trạng thái xác minh của quy trình")
    dong_hash = []
    for step in ("step1", "step2", "step3"):
        info = HASH_KQ.get(step, {})
        dong_hash.append({
            "Bước": step,
            "Tệp gốc": config.STEP_FILES[step].name,
            "SHA-256 khớp bản đã truy xuất": "Khớp" if info.get("khop") else "LỆCH/KHÔNG CÓ",
        })
    st.table(pd.DataFrame(dong_hash))
    st.error(
        "⛔ **Khối còn thiếu trong Step 1 (đã xác minh trong mã):** hai vị trí ghi chú "
        "“thêm phần lọc dữ liệu TT14 tại đây” đang trống và vùng chia tập chỉ gán "
        "`train_test = ''`. Hệ quả: đầu ra Step 1 khi chạy đúng tệp đã truy cập sẽ "
        "chưa đủ điều kiện vào Step 2 (Step 2 yêu cầu nhãn train/test có sẵn). Ứng "
        "dụng **chặn** tại ranh giới này, không tự viết khối lọc TT14 hay tự chia "
        "OOT 30% cho dữ liệu thật. Người dùng có tệp `credit_data.parquet` từ một "
        "lần chạy Step 1 đầy đủ (đã gồm khối của chủ sản phẩm) có thể vào thẳng "
        "Step 2 ở mục “Tải dữ liệu & kiểm tra”."
    )
    st.write(
        "Tệp `credit_data_woe.parquet` là **đầu ra của Step 2** theo mã nguồn "
        "(`OUTPUT_FILE_WOE`), được sinh ra khi chạy Step 2 — không cần tải riêng, "
        "trừ khi bạn muốn chạy Step 3 độc lập với tệp có sẵn."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Ứng dụng làm được**")
        st.markdown(
            "- Nhận tệp " + ", ".join("." + d for d in DINH_DANG_DUNG_DUOC)
            + f", tối đa **{config.MAX_UPLOAD_MB} MB mỗi tệp**.\n"
            "- Kiểm tra trước: schema, kiểu, giá trị thiếu, trùng lặp, độ phủ "
            "thời gian theo khai báo (kỳ quan sát, kỳ dự báo, nguy cơ bị cắt phải).\n"
            "- Chạy Step 1–3 bằng chính mã gốc trong `pipeline/`.\n"
            "- Hiển thị tiến trình, kết quả, biểu đồ; cho tải artifact và báo "
            "cáo tổng hợp (Excel/HTML)."
        )
    with col2:
        st.markdown("**Ứng dụng không làm / không tuyên bố**")
        st.markdown(
            "- Không tự lọc TT14, không tự chia train/test hay OOT cho dữ liệu thật.\n"
            "- Không tuyên bố tuân thủ Thông tư 14/2025/TT-NHNN hay được NHNN phê duyệt "
            "(chưa xác minh pháp lý — cần chuyên gia rà soát).\n"
            "- Không cam kết chất lượng mô hình đủ điều kiện vận hành; cần thẩm định độc lập.\n"
            "- Không thu phí, không xác nhận thanh toán; mục Gói giá chỉ hiển thị thông tin.\n"
            "- Không cam kết an toàn tuyệt đối hay hỗ trợ mọi dung lượng dữ liệu."
        )

    if not CO_PYARROW:
        st.warning(
            "⚠️ Môi trường hiện tại **thiếu thư viện pyarrow** nên định dạng .parquet "
            "tạm không khả dụng (CSV/XLSX/TXT vẫn dùng được). Hãy cài pyarrow theo "
            "`requirements.txt` để bật parquet."
        )

    st.subheader("Dữ liệu của bạn được xử lý ở đâu")
    st.write(
        "Dữ liệu tải lên được xử lý trên máy chủ đang chạy ứng dụng này: bản sao "
        "đầu vào, đầu ra và nhật ký nằm trong thư mục chạy riêng `runs/<run_id>/` "
        "cho tới khi bạn bấm **Xóa dữ liệu phiên** (mục Kết quả & báo cáo) hoặc "
        f"quá hạn {config.DON_DEP_RUN_SAU_GIO:g} giờ và có lượt khởi động ứng dụng "
        "kích hoạt dọn dẹp (best-effort, không phải lịch xóa cam kết). Mã ứng dụng "
        "không gửi dữ liệu tới dịch vụ bên thứ ba, AI, analytics hay email. Tuy "
        "nhiên nơi bạn **triển khai** ứng dụng (máy cá nhân, máy chủ nội bộ, hay "
        "dịch vụ công cộng) quyết định ai có thể truy cập dữ liệu: chỉ dùng dữ liệu "
        "giả lập cho demo công khai; không tải dữ liệu ngân hàng thật lên dịch vụ "
        "công cộng khi chưa được phê duyệt và đánh giá đầy đủ về lưu trữ, truy cập, "
        "hợp đồng và pháp luật. Xem mục “Hướng dẫn & triển khai”."
    )

    with st.expander("Thuật ngữ (giải thích ngắn khi gặp lần đầu)"):
        st.markdown(
            "- **PD**: xác suất vỡ nợ trong kỳ dự báo.\n"
            "- **WOE** (Weight of Evidence): logarit tỷ lệ phân phối sự kiện/không sự kiện "
            "trong một nhóm giá trị (bin); cách mã hóa biến thành số cho mô hình.\n"
            "- **IV** (Information Value): mức độ phân tách hai lớp của một biến; Step 2 giữ biến có IV ≥ 0.1.\n"
            "- **Bin / binning**: chia miền giá trị của biến thành các nhóm trước khi tính WOE.\n"
            "- **VIF**: chỉ số phóng đại phương sai, đo đa cộng tuyến; Step 2 loại biến có VIF > 5.\n"
            "- **PSI** (Population Stability Index): độ lệch phân phối giữa hai tập; ≥ 0.25 thường coi là bất ổn.\n"
            "- **OOT** (Out-of-Time): tập kiểm định ở giai đoạn thời gian muộn hơn tập huấn luyện.\n"
            "- **Right-censoring (cắt phải)**: hồ sơ chưa đủ thời gian theo dõi để biết kết quả cuối kỳ dự báo.\n"
            "- **AUC / Gini / KS**: thước đo khả năng phân biệt tốt/xấu (Gini = 2×AUC − 1; KS là khoảng cách "
            "lớn nhất giữa hai phân phối tích lũy).\n"
            "- **Hiệu chỉnh Platt**: biến đổi đơn điệu log-odds để xác suất dự báo khớp tỷ lệ quan sát hơn.\n"
            "- **PDO** (Points to Double the Odds): số điểm làm odds tăng gấp đôi trong thang điểm scorecard.\n"
            "- **Decile**: 10 nhóm bằng nhau theo phân phối PD; **Grade**: hạng điểm kiểu AAA…D."
        )

# ===========================================================================
# TRANG TẢI DỮ LIỆU & KIỂM TRA
# ===========================================================================
elif TRANG == "Tải dữ liệu & kiểm tra":
    st.title("Tải dữ liệu & kiểm tra")
    st.write(
        f"Định dạng được chấp nhận: **{', '.join('.' + d for d in DINH_DANG_DUNG_DUOC)}**. "
        f"Giới hạn: **{config.MAX_UPLOAD_MB} MB mỗi tệp**; tối đa {config.MAX_FILES_STEP1} tệp "
        f"cho Step 1 và {config.MAX_PARTS_CREDIT_DATA} phần cho credit_data chia nhỏ. "
        "Ứng dụng không nhận tệp pickle/joblib, không thực thi nội dung tệp, và không "
        "dùng tên tệp của bạn làm đường dẫn trên máy chủ."
    )

    lua_chon = st.radio(
        "Bạn sẽ bắt đầu từ điểm vào nào?",
        ["step1_skv", "step2_credit_data", "step3_woe", "gia_lap"],
        format_func=lambda x: {
            "step1_skv": "A. Dữ liệu thô SKV → chạy Step 1 (nhiều tệp)",
            "step2_credit_data": "B. credit_data (đã có train_test) → chạy Step 2",
            "step3_woe": "C. credit_data_woe → chạy Step 3 độc lập",
            "gia_lap": "D. Dữ liệu GIẢ LẬP minh họa → chạy thử Step 2–3",
        }[x],
    )

    if lua_chon == "step1_skv":
        st.warning(
            "Lưu ý ranh giới chưa xác minh: Step 1 trong mã đã truy cập chưa có khối "
            "lọc TT14 và chia OOT, nên đầu ra sẽ có `train_test` rỗng và **không chạy "
            "tiếp được Step 2**. Bạn vẫn có thể chạy Step 1 để lấy đầu ra trung gian "
            "và đối chiếu biến đổi đã xác minh."
        )
    if lua_chon == "gia_lap":
        st.warning(
            "⚠️ " + synthetic.NHAN_GIA_LAP + ". Cột train_test trong dữ liệu giả lập "
            "được gán theo mốc tháng chỉ để kiểm thử kỹ thuật — không phải khối "
            "TT14/OOT của quy trình gốc. Mọi chỉ số thu được chỉ chứng minh đường "
            "ống chạy được."
        )
        if st.button("Tạo dữ liệu giả lập (12.000 dòng)"):
            df = synthetic.tao_credit_data_gia_lap()
            st.session_state.update({
                "diem_vao": "gia_lap_step2", "df": df, "du_lieu_gia_lap": True,
                "anh_xa_cot": {},
                "dau_vao_meta": [{"vai_tro": "credit_data giả lập", "dinh_dang": "sinh trong bộ nhớ",
                                  "so_dong": int(len(df)), "so_cot": int(df.shape[1])}],
            })
            _dam_bao_run_dir()
            st.success("Đã tạo dữ liệu giả lập trong bộ nhớ phiên. Kéo xuống để khai báo và kiểm tra.")
    else:
        nhieu_tep = lua_chon in ("step1_skv", "step2_credit_data")
        tep_tai = st.file_uploader(
            "Chọn tệp dữ liệu",
            type=list(DINH_DANG_DUNG_DUOC),
            accept_multiple_files=nhieu_tep,
            help="Tên tệp của bạn không được dùng làm đường dẫn trên máy chủ; "
                 "hệ thống đặt tên an toàn riêng trong thư mục chạy.",
        )
        if tep_tai:
            danh_sach = tep_tai if isinstance(tep_tai, list) else [tep_tai]
            gioi_han_so = (config.MAX_FILES_STEP1 if lua_chon == "step1_skv"
                           else config.MAX_PARTS_CREDIT_DATA if lua_chon == "step2_credit_data" else 1)
            if len(danh_sach) > gioi_han_so:
                st.error(f"Quá giới hạn {gioi_han_so} tệp cho điểm vào này. Hãy giảm số tệp mỗi lần tải.")
                danh_sach = []
            loi_co_ban = []
            noi_dung_tep = []
            for tep in danh_sach:
                du_lieu = tep.getvalue()
                loi_co_ban += val.kiem_tra_tep_co_ban(
                    tep.name, du_lieu[:8], len(du_lieu),
                    config.MAX_UPLOAD_MB, DINH_DANG_DUNG_DUOC)
                noi_dung_tep.append((val.phan_mo_rong_tep(tep.name), du_lieu))
            if loi_co_ban:
                for loi in loi_co_ban:
                    st.error("⛔ " + loi)
            else:
                try:
                    if lua_chon == "step1_skv":
                        # Giữ bytes gốc cho parquet; định dạng khác được đọc và
                        # chuyển parquet ở bước đặt đầu vào.
                        khung_xem = None
                        meta = []
                        for ext, du_lieu in noi_dung_tep:
                            if ext == "parquet":
                                import pyarrow.parquet as pq
                                md = pq.ParquetFile(io.BytesIO(du_lieu)).metadata
                                meta.append({"vai_tro": "SKV parquet", "dinh_dang": ext,
                                             "dung_luong_mb": round(len(du_lieu) / 1024 / 1024, 2),
                                             "so_dong": int(md.num_rows), "so_cot": int(md.num_columns)})
                                if khung_xem is None:
                                    khung_xem = val.doc_tep_an_toan(du_lieu, ext).head(2000)
                            else:
                                khung = val.doc_tep_an_toan(du_lieu, ext)
                                meta.append({"vai_tro": "SKV bảng", "dinh_dang": ext,
                                             "dung_luong_mb": round(len(du_lieu) / 1024 / 1024, 2),
                                             "so_dong": int(len(khung)), "so_cot": int(khung.shape[1])})
                                if khung_xem is None:
                                    khung_xem = khung
                        st.session_state.update({
                            "diem_vao": "step1_skv", "skv_file_bytes": noi_dung_tep,
                            "skv_df_xem_truoc": khung_xem, "df": None,
                            "du_lieu_gia_lap": False, "dau_vao_meta": meta,
                        })
                    else:
                        cac_khung = [val.doc_tep_an_toan(du_lieu, ext) for ext, du_lieu in noi_dung_tep]
                        cot_chuan = list(cac_khung[0].columns)
                        if any(list(k.columns) != cot_chuan for k in cac_khung[1:]):
                            raise ValueError(
                                "Các phần dữ liệu không có cùng tập cột theo cùng thứ tự; "
                                "không nối được. Hãy tải một tệp hợp nhất hoặc đồng bộ cột."
                            )
                        khung = (pd.concat(cac_khung, axis=0, ignore_index=True)
                                 if len(cac_khung) > 1 else cac_khung[0])
                        meta = [{"vai_tro": "credit_data" if lua_chon == "step2_credit_data" else "credit_data_woe",
                                 "dinh_dang": noi_dung_tep[0][0],
                                 "dung_luong_mb": round(sum(len(d) for _, d in noi_dung_tep) / 1024 / 1024, 2),
                                 "so_dong": int(len(khung)), "so_cot": int(khung.shape[1]),
                                 "so_tep": len(noi_dung_tep)}]
                        st.session_state.update({
                            "diem_vao": lua_chon, "df": khung, "du_lieu_gia_lap": False,
                            "dau_vao_meta": meta,
                        })
                    _dam_bao_run_dir()
                    st.success("Đã đọc tệp vào bộ nhớ phiên. Kéo xuống để ánh xạ cột, khai báo và kiểm tra.")
                except ValueError as exc:
                    st.error("⛔ " + str(exc))
                except Exception:
                    st.error("⛔ Không đọc được tệp vì lỗi kỹ thuật không xác định. "
                             "Hãy thử xuất lại tệp ở đúng định dạng, hoặc giảm dung lượng.")

    # ---- Khung làm việc cho kiểm tra theo điểm vào ----
    diem_vao = st.session_state.get("diem_vao")
    khung_kiem_tra = None
    cot_yeu_cau = []
    ten_schema = ""
    if diem_vao == "step1_skv":
        khung_kiem_tra = st.session_state.get("skv_df_xem_truoc")
        cot_yeu_cau, ten_schema = val.STEP1_INPUT_COLUMNS, "schema đầu vào Step 1 (35 cột)"
        if khung_kiem_tra is not None:
            st.caption("Phần kiểm tra và xem trước dưới đây áp dụng trên tệp SKV đầu tiên "
                       "(tối đa 2.000 dòng đầu); khi chạy, Step 1 gốc xử lý mọi tệp đã tải.")
    elif diem_vao in ("step2_credit_data", "gia_lap_step2"):
        khung_kiem_tra = st.session_state.get("df")
        cot_yeu_cau, ten_schema = val.STEP2_INPUT_COLUMNS, "schema credit_data đầu vào Step 2 (36 cột)"
    elif diem_vao == "step3_woe":
        khung_kiem_tra = st.session_state.get("df")
        cot_yeu_cau = val.STEP3_ID_COLUMNS + [val.TARGET_COL, val.SPLIT_COL]
        ten_schema = "cột bắt buộc của credit_data_woe đầu vào Step 3"

    if khung_kiem_tra is not None:
        st.divider()
        st.subheader("Ánh xạ cột")
        st.write(
            "Nếu tên cột trong tệp của bạn khác tên cột quy trình, hãy ghép từng cột "
            "quy trình với cột nguồn tương ứng. Cột đã trùng tên được giữ nguyên. "
            "Bộ chuyển đổi chỉ **đổi tên cột**, không sửa giá trị nghiệp vụ."
        )
        cot_nguon = list(khung_kiem_tra.columns)
        anh_xa = dict(st.session_state.get("anh_xa_cot") or {})
        cot_con_thieu = [c for c in cot_yeu_cau if c not in cot_nguon]
        if cot_con_thieu:
            for cot_dich in cot_con_thieu:
                chon = st.selectbox(
                    f"Cột quy trình “{cot_dich}” lấy từ cột nguồn nào?",
                    ["(chưa chọn)"] + cot_nguon, key=f"map_{cot_dich}")
                if chon != "(chưa chọn)":
                    anh_xa[chon] = cot_dich
            st.session_state["anh_xa_cot"] = anh_xa
        else:
            st.success("Mọi cột bắt buộc đã có sẵn với đúng tên quy trình.")

        khung_sau_map = khung_kiem_tra.rename(columns=anh_xa) if anh_xa else khung_kiem_tra

        st.subheader("Khai báo mục tiêu và thời gian")
        if diem_vao == "step1_skv":
            st.write(
                "Biến mục tiêu của quy trình là **NX**, được Step 1 tạo từ `NHOM_NO` "
                "(NX = 1 khi NHOM_NO ≥ 3) — bạn không cần tự tạo. Kiểm tra phủ thời "
                "gian dưới đây dùng cột `NGAY_BC` của dữ liệu thô."
            )
        cot_ngay_mac_dinh = val.DATE_COL if val.DATE_COL in khung_sau_map.columns else cot_nguon[0]
        cot_ngay = st.selectbox("Cột ngày quan sát", list(khung_sau_map.columns),
                                 index=list(khung_sau_map.columns).index(cot_ngay_mac_dinh))
        ngay_parse, _, _ = val.thu_phan_tich_ngay(khung_sau_map[cot_ngay])
        d_min = ngay_parse.min() if ngay_parse.notna().any() else pd.Timestamp("2024-01-01")
        d_max = ngay_parse.max() if ngay_parse.notna().any() else pd.Timestamp("2024-12-31")
        st.caption(
            "Kỳ quan sát là giai đoạn lấy hồ sơ làm mẫu; kỳ dự báo là thời gian theo dõi "
            "kết quả sau điểm quan sát. Để mỗi hồ sơ có nhãn trọn vẹn, điểm quan sát muộn "
            "nhất phải sớm hơn ngày dữ liệu cuối ít nhất bằng kỳ dự báo (tránh bị cắt phải "
            "— right-censoring). Giá trị mặc định “Kết thúc kỳ quan sát” dưới đây đã lùi đủ "
            "kỳ dự báo so với dữ liệu; hãy sửa nếu định nghĩa nhãn của bạn khác."
        )
        c0, c1, c2 = st.columns(3)
        with c0:
            ky_du_bao = st.number_input("Kỳ dự báo (tháng)", min_value=1, max_value=60, value=12)
        qs_cuoi_goi_y = (d_max - pd.DateOffset(months=int(ky_du_bao))).date()
        with c1:
            qs_dau = st.date_input("Bắt đầu kỳ quan sát", value=d_min.date())
        with c2:
            qs_cuoi = st.date_input("Kết thúc kỳ quan sát", value=qs_cuoi_goi_y,
                                     key=f"qs_cuoi_{int(ky_du_bao)}")
        st.session_state["khai_bao"] = {
            "cot_muc_tieu": val.TARGET_COL, "cot_ngay": cot_ngay,
            "bat_dau_qs": str(qs_dau), "ket_thuc_qs": str(qs_cuoi),
            "ky_du_bao_thang": int(ky_du_bao),
        }

        if st.button("🔎 Chạy kiểm tra"):
            kt = val.KetQuaKiemTra()
            kt.gop(val.kiem_tra_cot_bat_buoc(khung_sau_map, cot_yeu_cau, ten_schema))
            kt.gop(val.kiem_tra_trung_lap(khung_sau_map))
            if diem_vao == "step1_skv":
                for cot_ngay_bat_buoc in val.STEP1_DATE_COLUMNS:
                    if cot_ngay_bat_buoc in khung_sau_map.columns:
                        _, fmt, ty_le_hong = val.thu_phan_tich_ngay(khung_sau_map[cot_ngay_bat_buoc])
                        if fmt is not None and not fmt.startswith("(") and fmt != "%d/%m/%Y":
                            kt.warnings.append(
                                f"Cột {cot_ngay_bat_buoc} phân tích được theo định dạng {fmt}, khác "
                                f"định dạng %d/%m/%Y mà Step 1 gốc dùng cho chuỗi ngày. Nếu cột đang "
                                "là chuỗi khác định dạng, Step 1 sẽ báo lỗi; bộ chuyển đổi của web "
                                "sẽ viết lại các cột ngày thành chuỗi dd/mm/yyyy khi đặt đầu vào "
                                "từ CSV/XLSX/TXT. Với tệp parquet tải lên, nội dung được giữ "
                                "nguyên như tệp gốc của hệ thống nguồn.")
                        elif ty_le_hong > 0.001:
                            kt.blockers.append(
                                f"Cột ngày {cot_ngay_bat_buoc} có {ty_le_hong * 100:.1f}% giá trị "
                                "không phân tích được; Step 1 sẽ dừng lỗi.")
                if "NHOM_NO" in khung_sau_map.columns:
                    nhom_so = pd.to_numeric(khung_sau_map["NHOM_NO"], errors="coerce")
                    kt.chi_tiet["ty_le_nhom_no_ge_3_du_kien_nx_1"] = float((nhom_so >= 3).mean())
            else:
                kt.gop(val.kiem_tra_muc_tieu(khung_sau_map))
                kt.gop(val.kiem_tra_chia_tap(khung_sau_map,
                                             bat_buoc_co_test=(diem_vao == "step3_woe")))
                if diem_vao == "step3_woe":
                    cot_so = [c for c in khung_sau_map.columns
                              if c not in val.STEP3_ID_COLUMNS + [val.TARGET_COL, val.SPLIT_COL]
                              and pd.api.types.is_numeric_dtype(khung_sau_map[c])]
                    if not cot_so:
                        kt.blockers.append(
                            "Không có cột số nào làm biến mô hình sau khi loại cột định danh, "
                            "mục tiêu và chia tập. Step 3 sẽ dừng lỗi tương tự.")
                    kt.chi_tiet["so_bien_so_step3"] = len(cot_so)
            kt.gop(val.kiem_tra_phu_thoi_gian(khung_sau_map, cot_ngay, qs_dau, qs_cuoi,
                                              int(ky_du_bao)))
            st.session_state["kiem_tra"] = kt
            if diem_vao != "step1_skv":
                st.session_state["df"] = khung_sau_map
            else:
                st.session_state["skv_df_xem_truoc"] = khung_sau_map
            _ghi_manifest_phien()
            _hien_loi_va_canh_bao(kt)
        elif st.session_state.get("kiem_tra") is not None:
            _hien_loi_va_canh_bao(st.session_state["kiem_tra"])

        st.subheader("Xem trước an toàn")
        st.caption("Chỉ vài dòng đầu; cột định danh/họ tên (BILLING_ACCTNUM, CIF, SO_THE, TEN_CPD) đã được che.")
        st.dataframe(val.xem_truoc_an_toan(khung_sau_map, config.SO_DONG_XEM_TRUOC),
                     use_container_width=True)
        with st.expander("Tổng quan theo cột (kiểu, thiếu, phân biệt)"):
            tong_quan = val.tong_quan_cot(khung_sau_map)
            st.dataframe(tong_quan, use_container_width=True)
            thieu_nhieu = tong_quan[tong_quan["Tỷ lệ thiếu (%)"] > 0].sort_values(
                "Tỷ lệ thiếu (%)", ascending=False).head(20)
            if not thieu_nhieu.empty:
                st.markdown("**Tỷ lệ thiếu theo cột (20 cột thiếu nhiều nhất)**")
                st.bar_chart(thieu_nhieu.set_index("Cột")[["Tỷ lệ thiếu (%)"]])
        theo_thang = val.thong_ke_theo_thang(khung_sau_map, cot_ngay)
        if not theo_thang.empty:
            st.markdown("**Số dòng theo tháng của cột ngày quan sát**")
            st.bar_chart(theo_thang.set_index("Tháng")[["Số dòng"]])
            if "Tỷ lệ NX=1" in theo_thang.columns and diem_vao != "step1_skv":
                st.markdown("**Tỷ lệ nhãn NX = 1 theo tháng (kiểm tra độ ổn định nhãn theo thời gian)**")
                st.line_chart(theo_thang.set_index("Tháng")[["Tỷ lệ NX=1"]])
    else:
        st.info("Hãy chọn điểm vào và tải tệp (hoặc tạo dữ liệu giả lập) để bắt đầu.")

# ===========================================================================
# TRANG CHẠY MÔ HÌNH
# ===========================================================================
elif TRANG == "Chạy mô hình (Step 1–3)":
    st.title("Chạy mô hình (Step 1–3)")
    st.write(
        "Mỗi bước chạy bằng **bản sao của mã gốc** trong `pipeline/`; bộ chuyển đổi "
        "chỉ đổi đường dẫn thư mục đầu vào/đầu ra sang thư mục chạy của phiên. "
        f"Giới hạn thời gian mỗi bước: {config.TIMEOUT_STEP_GIAY} giây. Thời gian chạy "
        "trên dữ liệu lớn (hàng triệu dòng) trong môi trường web **chưa được kiểm "
        "chứng**; hãy bắt đầu với dữ liệu giả lập hoặc tập con do bạn trích xuất."
    )
    with st.expander("Cấu hình pipeline đang áp dụng (đọc từ mã nguồn, ứng dụng không đổi)"):
        st.markdown("**Step 2**")
        st.table(pd.DataFrame(report_mod.THAM_SO_STEP2, columns=["Tham số", "Giá trị"]))
        st.markdown("**Step 3**")
        st.table(pd.DataFrame(report_mod.THAM_SO_STEP3, columns=["Hạng mục", "Giá trị"]))

    diem_vao = st.session_state.get("diem_vao")
    kt = st.session_state.get("kiem_tra")
    cac_buoc = st.session_state.get("buoc") or {}

    def _trang_thai(step):
        return "✅ Hoàn tất" if cac_buoc.get(step, {}).get("thanh_cong") else (
            "⛔ Chưa hoàn tất" if step in cac_buoc else "— Chưa chạy")

    st.table(pd.DataFrame([
        {"Bước": "Step 1 — làm dữ liệu", "Trạng thái": _trang_thai("step1")},
        {"Bước": "Step 2 — lọc biến (WOE/IV)", "Trạng thái": _trang_thai("step2")},
        {"Bước": "Step 3 — mô hình & chấm điểm", "Trạng thái": _trang_thai("step3")},
    ]))

    def _chay_va_hien(step, ten_hien_thi):
        run_dir = _dam_bao_run_dir()
        log_lines = []
        hop_trang_thai = st.status(f"Đang chạy {ten_hien_thi}…", expanded=True)
        vung_log = hop_trang_thai.empty()
        dem = [0]
        t0 = time.time()

        def nhan_log(dong: str):
            log_lines.append(dong)
            dem[0] += 1
            if dem[0] % 20 == 0:
                vung_log.code("\n".join(log_lines[-25:])
                              + f"\n… đã chạy {time.time() - t0:.0f}s")

        ket_qua = adapters.chay_step(step, run_dir, on_line=nhan_log)
        vung_log.code("\n".join(log_lines[-25:]))
        hop_trang_thai.update(
            label=(f"{ten_hien_thi}: hoàn tất sau {ket_qua.thoi_gian_giay:.0f}s"
                   if ket_qua.thanh_cong else f"{ten_hien_thi}: chưa hoàn tất"),
            state="complete" if ket_qua.thanh_cong else "error", expanded=not ket_qua.thanh_cong)
        cac_buoc_local = dict(st.session_state.get("buoc") or {})
        cac_buoc_local[step] = {
            "step": step, "thanh_cong": ket_qua.thanh_cong,
            "thoi_gian_giay": ket_qua.thoi_gian_giay, "ma_thoat": ket_qua.ma_thoat,
            "thong_diep": ket_qua.thong_diep,
        }
        st.session_state["buoc"] = cac_buoc_local
        adapters.ghi_manifest(run_dir, {"cac_buoc": list(cac_buoc_local.values())})
        if ket_qua.thanh_cong:
            st.success(ket_qua.thong_diep)
        else:
            st.error("⛔ " + ket_qua.thong_diep)
            with st.expander("Cuối nhật ký của bước"):
                st.code(adapters.doc_log_tail(ket_qua.log_path) or "(không có nhật ký)")
        return ket_qua

    st.divider()
    st.markdown("### Step 1 — Làm dữ liệu mô hình")
    if diem_vao == "step1_skv" and kt is not None and kt.ok:
        st.warning(
            "Nhắc lại: đầu ra Step 1 theo mã đã truy cập có `train_test` rỗng (khối "
            "TT14/OOT vắng mặt). Sau khi chạy, hãy đọc kiểm tra hậu kỳ bên dưới; "
            "Step 2 sẽ bị chặn tới khi có tệp credit_data đã hoàn thiện."
        )
        if st.button("▶️ Chạy Step 1"):
            run_dir = _dam_bao_run_dir()
            danh_sach_bytes = []
            loi_chuyen = None
            for ext, du_lieu in st.session_state.get("skv_file_bytes") or []:
                if ext == "parquet":
                    danh_sach_bytes.append(du_lieu)
                else:
                    try:
                        khung = val.doc_tep_an_toan(du_lieu, ext)
                        anh_xa = st.session_state.get("anh_xa_cot") or {}
                        if anh_xa:
                            khung = khung.rename(columns=anh_xa)
                        # Step 1 gốc phân tích chuỗi ngày dd/mm/yyyy: viết các cột
                        # ngày thành chuỗi đúng định dạng đó (biến đổi của bộ
                        # chuyển đổi, chỉ áp cho đầu vào không phải parquet).
                        for cot_n in val.STEP1_DATE_COLUMNS:
                            if cot_n in khung.columns:
                                ngay_da, fmt, _ = val.thu_phan_tich_ngay(khung[cot_n])
                                khung[cot_n] = pd.Series(ngay_da).dt.strftime("%d/%m/%Y")
                        buf = io.BytesIO()
                        khung.to_parquet(buf, index=False, engine="pyarrow")
                        danh_sach_bytes.append(buf.getvalue())
                    except Exception:
                        loi_chuyen = "Không chuyển được một tệp SKV sang parquet. Hãy kiểm tra lại định dạng."
                if loi_chuyen:
                    break
            if loi_chuyen:
                st.error("⛔ " + loi_chuyen)
            else:
                adapters.dat_dau_vao_step1(run_dir, danh_sach_bytes)
                kq1 = _chay_va_hien("step1", "Step 1")
                if kq1.thanh_cong:
                    # Kiểm tra hậu kỳ: cột train_test của đầu ra (chỉ đọc 1 cột)
                    try:
                        dau_ra = pd.read_parquet(
                            adapters.thu_muc_model(run_dir) / "credit_data.parquet",
                            columns=[val.SPLIT_COL], engine="pyarrow")
                        so_rong = int((dau_ra[val.SPLIT_COL].astype(str).str.strip() == "").sum())
                        if so_rong == len(dau_ra):
                            st.error(
                                f"⛔ **Bị chặn trước Step 2:** toàn bộ {len(dau_ra):,} dòng đầu ra "
                                "Step 1 có train_test rỗng — đúng như mã đã truy cập (khối "
                                "lọc TT14 + chia OOT 30% chưa có trong tệp). Ứng dụng không "
                                "tự điền train_test. Để chạy tiếp: bổ sung khối của chủ "
                                "sản phẩm vào đúng hai vị trí đã đánh dấu trong Step 1, "
                                "hoặc tải credit_data.parquet đã hoàn thiện ở điểm vào B. "
                                "Đầu ra trung gian của Step 1 vẫn tải được ở mục Kết quả.")
                        else:
                            st.info(f"Đầu ra Step 1: {len(dau_ra):,} dòng, trong đó {so_rong:,} dòng train_test rỗng.")
                    except Exception:
                        st.warning("Không đọc lại được đầu ra Step 1 để kiểm tra hậu kỳ; hãy xem artifact ở mục Kết quả.")
    elif diem_vao == "step1_skv":
        st.info("Hãy bấm “Chạy kiểm tra” ở mục Tải dữ liệu và xử lý hết lỗi chặn trước khi chạy Step 1.")
    else:
        st.caption("Step 1 chỉ khả dụng với điểm vào A (dữ liệu thô SKV).")

    st.divider()
    st.markdown("### Step 2 — Lọc biến mô hình (WOE/IV)")
    co_credit_data_kiem_tra = (diem_vao in ("step2_credit_data", "gia_lap_step2")
                                and kt is not None and kt.ok)
    if co_credit_data_kiem_tra:
        if st.button("▶️ Chạy Step 2"):
            run_dir = _dam_bao_run_dir()
            adapters.dat_credit_data(run_dir, st.session_state["df"])
            _chay_va_hien("step2", "Step 2")
    elif (st.session_state.get("buoc") or {}).get("step1", {}).get("thanh_cong"):
        st.error(
            "⛔ Step 2 đang bị chặn: đầu ra Step 1 của phiên này chưa có train_test "
            "hợp lệ do khối TT14/OOT vắng mặt trong mã đã truy cập (xem kiểm tra hậu "
            "kỳ ở trên). Không có cách chạy tiếp trung thực trong ứng dụng cho tới "
            "khi khối đó được bổ sung vào mã hoặc bạn tải credit_data đã hoàn thiện."
        )
    else:
        st.info("Step 2 cần điểm vào B (credit_data có train_test hợp lệ) đã qua kiểm tra mà không còn lỗi chặn.")

    st.divider()
    st.markdown("### Step 3 — Modelling (hồi quy logistic) & chấm điểm")
    step2_xong = bool((st.session_state.get("buoc") or {}).get("step2", {}).get("thanh_cong"))
    co_woe_truc_tiep = (diem_vao == "step3_woe" and kt is not None and kt.ok)
    if step2_xong or co_woe_truc_tiep:
        if st.button("▶️ Chạy Step 3"):
            run_dir = _dam_bao_run_dir()
            if co_woe_truc_tiep and not step2_xong:
                adapters.ghi_dataframe_parquet(
                    st.session_state["df"],
                    adapters.thu_muc_model(run_dir) / "credit_data_woe.parquet")
            _chay_va_hien("step3", "Step 3")
    else:
        st.info("Step 3 cần Step 2 hoàn tất trong phiên này, hoặc điểm vào C (credit_data_woe) đã qua kiểm tra.")

# ===========================================================================
# TRANG KẾT QUẢ & BÁO CÁO
# ===========================================================================
elif TRANG == "Kết quả & báo cáo":
    st.title("Kết quả & báo cáo")
    if not st.session_state.get("run_dir"):
        st.info("Chưa có lần chạy nào trong phiên này. Hãy tải dữ liệu và chạy mô hình trước.")
    else:
        run_dir = Path(st.session_state["run_dir"])
        if st.session_state.get("du_lieu_gia_lap"):
            st.warning("⚠️ " + synthetic.NHAN_GIA_LAP + " — số liệu dưới đây chỉ minh họa kỹ thuật.")
        kq = report_mod.doc_ket_qua_pipeline(run_dir)

        if kq.get("metrics_17") is not None:
            st.subheader("Bộ chỉ số đánh giá (Metrics_17 — artifact Step 3)")
            st.dataframe(kq["metrics_17"], use_container_width=True)
        if kq.get("he_so") is not None:
            st.subheader("Hệ số mô hình hồi quy logistic")
            he_so = kq["he_so"]
            st.dataframe(he_so, use_container_width=True)
            try:
                cot_ten = "X" if "X" in he_so.columns else he_so.columns[0]
                cot_hs = "Coefficients" if "Coefficients" in he_so.columns else he_so.columns[1]
                bang_hs = (he_so[[cot_ten, cot_hs]]
                           .rename(columns={cot_ten: "Biến", cot_hs: "Hệ số"})
                           .set_index("Biến"))
                st.markdown("**Hệ số theo biến (không gian đầu vào WOE)**")
                st.bar_chart(bang_hs)
            except Exception:
                pass
        for khoa, tieu_de in [
            ("tham_so_diem_new", "Tham số thang điểm sau hiệu chỉnh (score_parameter_new)"),
            ("tham_so_diem", "Tham số thang điểm ban đầu (score_parameter)"),
            ("pd_decile_new", "Khoảng PD theo decile trên train (mới)"),
            ("pd_decile", "Khoảng PD theo decile trên train"),
            ("grade_score_new", "Khoảng điểm theo hạng trên train (mới)"),
            ("grade_score", "Khoảng điểm theo hạng trên train"),
            ("gini_ks_summary", "Tổng hợp Gini/KS theo hạng (tập test)"),
            ("psi_tong_hop", "PSI tổng hợp train → test"),
            ("iv_per_feature", "IV theo biến"),
        ]:
            if kq.get(khoa) is not None:
                with st.expander(tieu_de):
                    st.dataframe(kq[khoa], use_container_width=True)

        thu_muc_plot = adapters.thu_muc_model(run_dir) / "plot" / "model_plot"
        cac_png = sorted(thu_muc_plot.glob("*.png")) if thu_muc_plot.exists() else []
        if cac_png:
            st.subheader("Biểu đồ kiểm định do Step 3 sinh ra")
            st.caption("Hình gốc từ artifact (dpi=150), có chú thích chỉ số trên từng hình.")
            for png in cac_png:
                with st.expander(png.name):
                    st.image(str(png), use_container_width=True)

        st.divider()
        st.subheader("Phần đã kiểm định và phần chưa kiểm định")
        st.markdown(
            "**Đã tính được trong lần chạy này:** các bảng/hình ở trên là đầu ra trực "
            "tiếp của mã Step 2/Step 3 (WOE/IV/PSI phía Step 2 nằm trong artifact và "
            "nhật ký; chỉ số phân biệt, hiệu chuẩn, ổn định phía Step 3 nằm trong "
            "Metrics_17 và thư mục plot), cùng kiểm tra trước của tầng web.\n\n"
            "**Chưa kiểm định:**")
        for muc in report_mod.PHAN_CHUA_KIEM_DINH_MAC_DINH:
            st.markdown(f"- {muc}")

        st.divider()
        st.subheader("Tải báo cáo và artifact")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Báo cáo tổng hợp (Excel)",
                data=report_mod.tao_bao_cao_excel(run_dir),
                file_name=f"bao_cao_pd_{run_dir.name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c2:
            st.download_button(
                "⬇️ Báo cáo tổng hợp (HTML)",
                data=report_mod.tao_bao_cao_html(run_dir).encode("utf-8"),
                file_name=f"bao_cao_pd_{run_dir.name}.html", mime="text/html")
        cac_artifact = adapters.liet_ke_artifact(run_dir)
        if cac_artifact:
            st.markdown("**Artifact trong thư mục model của lần chạy** "
                        "(bao gồm tệp điểm số cấp hồ sơ `credit_data_scoring.parquet` — "
                        "chỉ tải về môi trường được phê duyệt cho dữ liệu này):")
            for art in cac_artifact:
                with open(art["duong_dan"], "rb") as f:
                    du_lieu_tep = f.read()
                st.download_button(
                    f"⬇️ {art['ten']} ({art['dung_luong_kb']} KB)",
                    data=du_lieu_tep,
                    file_name=Path(art["ten"]).name,
                    key="dl_" + art["ten"])
        for step in ("step1", "step2", "step3"):
            log_path = run_dir / "logs" / f"{step}.log"
            if log_path.exists():
                with st.expander(f"Nhật ký {step} (phần cuối)"):
                    st.code(adapters.doc_log_tail(str(log_path)))

        st.divider()
        st.subheader("Xóa dữ liệu phiên")
        st.write(
            "Bấm nút dưới đây để xóa toàn bộ thư mục chạy của phiên này khỏi máy chủ "
            "(đầu vào, đầu ra, nhật ký) ngay lập tức. Hành động không hoàn tác được; "
            "hãy tải mọi báo cáo/artifact cần giữ trước khi xóa."
        )
        if st.button("🗑️ Xóa dữ liệu phiên này"):
            if adapters.xoa_run(run_dir):
                for khoa in ("run_dir", "df", "skv_file_bytes", "skv_df_xem_truoc",
                             "kiem_tra", "buoc", "dau_vao_meta", "anh_xa_cot"):
                    st.session_state[khoa] = [] if khoa in ("skv_file_bytes", "dau_vao_meta") else (
                        {} if khoa in ("buoc", "anh_xa_cot") else None)
                st.success("Đã xóa thư mục chạy của phiên khỏi máy chủ.")
            else:
                st.error("Không xóa được thư mục chạy. Người vận hành cần kiểm tra quyền trên máy chủ.")

# ===========================================================================
# TRANG GÓI GIÁ
# ===========================================================================
elif TRANG == "Gói giá":
    st.title("Gói giá")
    st.markdown(
        f"- **{config.SO_LUOT_MIEN_PHI} lượt đầu: miễn phí.**\n"
        f"- Từ lượt thứ {config.SO_LUOT_MIEN_PHI + 1}: **{config.GIA_MOI_LUOT_VND_2026:,} "
        f"VND/lượt** (giá cơ sở năm {config.NAM_GIA_CO_SO}).".replace(",", "."))
    st.info(
        "Đây là mức giá do **chủ sản phẩm đề xuất** cho sản phẩm này, **không phải "
        "giá thị trường đã được kiểm chứng** và không kèm khảo sát giá đối thủ trong "
        "repo này. Chủ sản phẩm chỉnh sửa tại `app/config.py`."
    )
    st.markdown("**Điều chỉnh theo lạm phát (chính sách của chủ sản phẩm):**")
    st.latex(r"\text{Giá năm } y = \text{Giá năm } (y-1) \times (1 + \text{tỷ lệ lạm phát năm } y)")
    dong_gia = [{"Năm": config.NAM_GIA_CO_SO, "Tỷ lệ lạm phát áp dụng": "— (năm cơ sở)",
                 "Giá (VND/lượt)": f"{config.GIA_MOI_LUOT_VND_2026:,}".replace(",", "."),
                 "Nguồn": "—"}]
    for nam, (ty_le, gia, nguon) in sorted(config.GIA_THEO_NAM.items()):
        dong_gia.append({"Năm": nam, "Tỷ lệ lạm phát áp dụng": ty_le,
                         "Giá (VND/lượt)": gia, "Nguồn": nguon})
    st.table(pd.DataFrame(dong_gia))
    if not config.GIA_THEO_NAM:
        st.write(
            "Giá các năm sau năm cơ sở **chưa được điền**: repo này chưa có tỷ lệ lạm "
            "phát từ nguồn thống kê chính thức kèm liên kết truy xuất được, đúng năm "
            "và đúng định nghĩa chỉ số. Khi đã có nguồn đạt yêu cầu, chủ sản phẩm bổ "
            "sung vào `GIA_THEO_NAM` trong `app/config.py`; ứng dụng không tự suy đoán."
        )
    st.markdown("**Thông tin thanh toán:**")
    st.code(config.THONG_TIN_THANH_TOAN)
    st.write(
        "Ứng dụng web này **không thu phí, không xác nhận thanh toán và không tự "
        "khóa lượt chạy theo giá**. Việc định nghĩa một “lượt”, đếm lượt và thanh "
        "toán (nếu triển khai thương mại) do chủ sản phẩm thực hiện ngoài ứng dụng, "
        "theo thỏa thuận với khách hàng và quy định liên quan."
    )

# ===========================================================================
# TRANG HƯỚNG DẪN & TRIỂN KHAI
# ===========================================================================
elif TRANG == "Hướng dẫn & triển khai":
    st.title("Hướng dẫn & triển khai")
    st.subheader("Cách dùng trong 5 bước")
    st.markdown(
        "1. Chuẩn bị tệp đúng một trong ba điểm vào (SKV thô / credit_data / credit_data_woe) "
        "hoặc chọn dữ liệu giả lập để chạy thử.\n"
        "2. Vào **Tải dữ liệu & kiểm tra**: tải tệp, ánh xạ cột nếu tên khác quy trình, "
        "khai báo kỳ quan sát và kỳ dự báo, bấm **Chạy kiểm tra** và sửa mọi lỗi chặn (mục đỏ).\n"
        "3. Vào **Chạy mô hình**: chạy lần lượt các bước đang mở khóa; đọc nhật ký trực tiếp.\n"
        "4. Vào **Kết quả & báo cáo**: xem chỉ số, biểu đồ, phần đã/chưa kiểm định; tải báo cáo và artifact.\n"
        "5. Bấm **Xóa dữ liệu phiên** khi xong để gỡ dữ liệu khỏi máy chủ."
    )
    st.subheader("Triển khai demo bằng điện thoại qua GitHub (tóm tắt)")
    st.markdown(
        "Ứng dụng là một app **Streamlit** (`streamlit_app.py`). Cách deploy demo không cần máy tính:\n\n"
        "1. (Khuyến nghị) Trên **github.com** bằng điện thoại, merge Pull Request của nhánh "
        "`arena/eac91acf-credit-risk` vào `main` (**Merge pull request → Confirm merge**) để deploy từ "
        "nhánh chính; hoặc bỏ qua và deploy thẳng từ nhánh phụ.\n"
        "2. Mở **share.streamlit.io/deploy**, đăng nhập GitHub của chính chủ sản phẩm, chọn "
        "**Create app → Yup, I have an app**, điền: Repository `phamhuynhtam/Credit_risk`; Branch "
        "`main` (hoặc `arena/eac91acf-credit-risk`, gõ tay nếu không thấy trong gợi ý); "
        "Main file path `streamlit_app.py`.\n"
        "3. Mở **Advanced settings → Python version: chọn 3.11** (đúng môi trường đã kiểm thử; "
        "không đổi được sau khi deploy trừ phi xóa app tạo lại), để trống ô Secrets, rồi bấm **Deploy**.\n"
        "4. Đợi vài phút cài `requirements.txt`; mở liên kết `*.streamlit.app` được cấp, chỉ kiểm thử "
        "bằng **Dữ liệu giả lập minh họa**. App ngủ sau 12 giờ không truy cập — bấm "
        "*Yes, get this app back up!* để đánh thức.\n\n"
        "Bản chi tiết từng thao tác (cấp quyền GitHub, chọn subdomain, giới hạn gói miễn phí "
        "2.7 GB RAM/50 GB đĩa, bảng xử lý lỗi build, Codespaces, chạy nội bộ cho dữ liệu thật) "
        "nằm ở `docs/HUONG_DAN_TRIEN_KHAI_DIEN_THOAI.md` trong repo. Repo này **chưa được triển khai** ở "
        "đâu; không có liên kết website nào đang hoạt động được xác minh."
    )
    st.error(
        "⛔ **Cấm dùng cho demo công khai:** không tải dữ liệu ngân hàng thật lên dịch vụ "
        "công cộng hay bên thứ ba khi chưa có phê duyệt và đánh giá đầy đủ về nơi lưu trữ, "
        "kiểm soát truy cập, xử lý dữ liệu, hợp đồng và yêu cầu pháp luật. Demo công khai "
        "chỉ dùng dữ liệu giả lập."
    )
    st.subheader("Giới hạn và rủi ro còn mở")
    st.markdown(
        "- Khối lọc TT14 + chia OOT của Step 1 vắng mặt → chuỗi chạy trọn vẹn từ dữ liệu "
        "thô tới mô hình chưa thực hiện được cho dữ liệu thật trong ứng dụng này.\n"
        "- Chưa xác minh pháp lý đối với Thông tư 14/2025/TT-NHNN; liên kết tài liệu tham "
        "chiếu không phải nguồn chính thức.\n"
        "- Chưa đo thời gian/bộ nhớ với dữ liệu hàng triệu dòng trong môi trường web; "
        f"giới hạn hiện tại là {config.MAX_UPLOAD_MB} MB/tệp và {config.TIMEOUT_STEP_GIAY}s/bước (cấu hình được).\n"
        "- Ứng dụng không có tài khoản người dùng hay phân quyền; khi triển khai nội bộ "
        "cần đặt sau cơ chế xác thực của tổ chức.\n"
        "- Thư mục chạy trên máy chủ chứa dữ liệu của phiên cho tới khi bị xóa (nút xóa "
        "thủ công hoặc dọn dẹp khi khởi động sau hạn cấu hình); bản thân việc còn tồn "
        "tại thư mục này phải được tính vào đánh giá lưu trữ/truy cập của nơi triển khai."
    )
    st.subheader("Giới thiệu sản phẩm tới ngân hàng")
    st.write(
        "Repo chứa **bản nháp** thư giới thiệu tại `docs/BAN_NHAP_EMAIL_GIOI_THIEU.md` "
        "để chủ sản phẩm tự rà soát và tự gửi theo quy định. Ứng dụng không tự gửi thư, "
        "không gửi hàng loạt, và không thu thập hay suy đoán địa chỉ người nhận."
    )
    ban_nhap_path = config.REPO_ROOT / "docs" / "BAN_NHAP_EMAIL_GIOI_THIEU.md"
    if ban_nhap_path.exists():
        with st.expander("Xem bản nháp thư giới thiệu (văn bản nháp, chưa gửi)"):
            st.markdown(ban_nhap_path.read_text(encoding="utf-8"))
