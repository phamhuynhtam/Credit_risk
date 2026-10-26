# Bản đồ Step 1–3: đầu vào, đầu ra, biến đổi, tham số, thư viện, môi trường

Tài liệu này lập từ việc **đọc trực tiếp** 3 tệp mã nguồn trong `pipeline/`
(hàm băm xem `pipeline/NGUON_GOC.md`) và kiểm tra schema 2 tệp dữ liệu đi kèm.
Phần nào chưa được kiểm chứng bằng lần chạy thực tế đều ghi rõ "chưa xác minh".

## 0. Luồng tổng thể

```
[Nhiều tệp *.parquet SKV thô]
        │  Step 1: step 1 làm dữ liệu mô hình.py
        ▼
credit_data.parquet  (+ credit_data1.parquet, credit_data2.parquet = chia đôi vận chuyển)
        │  Step 2: step 2 lọc biến mô hình.py  (credit_scoring v5)
        ▼
credit_data_woe.parquet (+ credit_data_woe_raw.parquet, thư mục mapping/, XLSX theo NGAY_BC)
        │  Step 3: step 3 modelling V2.py
        ▼
credit_data_scoring.parquet, parameter_model.xlsx, PD decile(_new).xlsx,
score_parameter(_new).xlsx, Grade_score(_new).xlsx, calibration.xlsx,
Gini_KS_result.xlsx, thư mục plot/ (metrics_summary.xlsx, model_plot/*.png, ...)
```

- Biến mục tiêu (target): `NX` = 1 nếu `NHOM_NO >= 3`, ngược lại 0 (tạo ở Step 1).
- Cột chia tập: `train_test` với nhãn `train`/`test`. **Step 2 và Step 3 không tự chia
  tập** — chúng yêu cầu cột này đã có sẵn và hợp lệ.
- `credit_data_woe.parquet`: mã nguồn Step 2 ghi rõ đây là **đầu ra** của Step 2
  (`OUTPUT_FILE_WOE`, xuất từ `df4_v2`). Đây là tệp trung gian sinh ra khi chạy Step 2,
  không cần liên kết riêng. Nội dung cụ thể chỉ xác nhận được sau lần chạy thực tế.

---

## 1. Step 1 — `step 1 làm dữ liệu mô hình.py` (14.517 byte, 555 dòng)

### Đầu vào
- Thư mục `D:\model\SKV parquet`, đọc **mọi** tệp `*.parquet` (glob), nối dọc (concat).
- Mỗi tệp phải có đúng 35 cột được liệt kê cứng trong `pd.read_parquet(..., columns=[...], engine="pyarrow")`:
  `BILLING_ACCTNUM, CIF, ISSUE_DATE, OPEN_CIF, CAP_1, CAP_2, DM, NGAY_BC, LAI_SUAT,
  SO_THE, NHOM_NO, DVKD_OLD, TDVKD, TSBD, CAP_PD, TEN_CPD, THOIHAN, CIB_SME_KHCN,
  LTV_CIF, "Vung mien", CAP_PD1, TENNGANH_KH, TENNGANH, TEN_MD, FLAG_LHINH, TENLHINH2,
  TENMUCDICH2, DM2, FIRST_BOOK_DATE, CAP_PD2, CAP_PD3, DU_NO_ADJUST_CIF, BDS, DS, PL_BDS_DS`.
- Giả định ngầm trong mã: các cột ngày ở dạng chuỗi định dạng `%d/%m/%Y`.

### Biến đổi (theo đúng mã đã truy cập)
1. `BILLING_ACCTNUM`: fillna `''`, bỏ hậu tố `.0`, đệm 0 thành 13 ký tự. `CIF`: tương tự, 8 ký tự.
2. `ISSUE_DATE, OPEN_CIF, NGAY_BC, FIRST_BOOK_DATE` → datetime với `format='%d/%m/%Y'`.
3. `LAI_SUAT < 0 → 0`.
4. `NX = 1 nếu NHOM_NO >= 3 else 0`; xóa `NHOM_NO`.
5. Điền khuyết chuỗi rỗng: `THOIHAN→'NGANHAN'`, `CAP_PD→'TRUONG DVKD'`,
   `CAP_PD1/CAP_PD2/CAP_PD3→'DVKD'`, `TEN_CPD→'Trưởng PGD'`.
6. Sau khi gộp: `OPEN_CIF = min theo CIF`; `FIRST_BOOK_DATE` trống lấy `OPEN_CIF`
   cùng dòng, rồi lấy `min theo CIF`.
7. ⚠️ **[CHƯA XÁC MINH — KHỐI THIẾU 1]** ghi chú `#thêm phần lọc dữ liệu TT14 tại đây` (trống).
8. Tạo 3 biến thời lượng (tháng, chặn dưới 0): `DURATION_ISSUE_DATE`,
   `DURATION_FIRST_BOOK_DATE`, `DURATION_OPEN_CIF` =
   `(năm(NGAY_BC)-năm(ngày gốc))*12 + (tháng(NGAY_BC)-tháng(ngày gốc))`.
9. `DU_NO_ADJUST_CIF, TSBD, BDS, DS`: fillna 0, ép float.
10. `LTV_CIF`: ép float nếu khác chuỗi `"tín chấp"` (giữ nguyên chuỗi nếu là `"tín chấp"`),
    cuối cùng ép dtype `string`.
11. ⚠️ **[CHƯA XÁC MINH — KHỐI THIẾU 2]** ghi chú `#thêm phần lọc dữ liệu TT14 tại đây #...` (trống).
12. ⚠️ **[CHƯA XÁC MINH — KHỐI THIẾU 3]** vùng `#######chỉnh từ vùng này` …
    `####### chỉnh đến vùng này` chỉ có `data_sktv['train_test'] = ''`.
    Đây là vị trí chủ sản phẩm dự định đặt khối **lọc TT14 + chia OOT 30%**.
    **Ứng dụng web không tự viết khối này.**
13. Xóa `ISSUE_DATE, FIRST_BOOK_DATE, OPEN_CIF`; sắp xếp 36 cột đầu ra.

### Đầu ra
- `D:\model\credit_data.parquet` (36 cột: 33 cột gốc đã biến đổi + 3 `DURATION_*` + `train_test`),
  nén `brotli`.
- Chia đôi theo thứ tự dòng (`ceil(n/2)`) thành `credit_data1.parquet`, `credit_data2.parquet`;
  in cảnh báo nếu có tệp > 20 MB (`MAX_SIZE_MB = 20`).

### Thư viện / môi trường
`pandas`, `numpy`, `glob`, `timeit`, `pathlib`, engine `pyarrow` (cần thêm gói `brotli`
để ghi nén brotli). Đường dẫn Windows cứng `D:\model`.

### Đã kiểm chứng trên dữ liệu được cung cấp
- `credit_data1.parquet` (764.530 dòng) + `credit_data2.parquet` (764.529 dòng) khớp đúng
  schema 36 cột đầu ra của Step 1; `train_test = ''` ở **100%** dòng → hai tệp này là đầu ra
  của Step 1 **khi chưa có** khối TT14/OOT. `NX`: tổng tỷ lệ nhãn 1 ≈ 7,05%.
  `NGAY_BC`: tệp 1 từ 31/12/2023 đến 31/05/2026 (18 tháng phân biệt);
  tệp 2 từ 30/06/2024 đến 30/09/2026 (17 tháng phân biệt).

---

## 2. Step 2 — `step 2 lọc biến mô hình.py` (100.142 byte, 3.405 dòng; "credit_scoring v5")

### Đầu vào
- `INPUT_FILE = D:\model\credit_data.parquet`, `TARGET_COL = "NX"`,
  `ID_COLS = ["BILLING_ACCTNUM", "CIF", "NGAY_BC"]`, `SPLIT_COL = "train_test"`,
  `DATE_COL = "NGAY_BC"`.

### Điều kiện chạy (mã tự kiểm tra; sai → in "LỖI" và `sys.exit(0)`, **không** tạo đầu ra)
- `NX` tồn tại, không thiếu sau khi loại dòng thiếu, chỉ gồm {0,1}.
- `train_test` tồn tại, không missing, không rỗng sau khi strip, nhãn sau chuẩn hóa
  (strip + lower) chỉ gồm {`train`, `test`}; bắt buộc có dòng `train`; tập train phải đủ cả 0 và 1.
  → Hệ quả đã kiểm chứng: hai tệp dữ liệu được cung cấp (train_test='') **bị Step 2 từ chối**.

### Tham số chính (đầu tệp, giữ nguyên)
| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `IV_THRESHOLD` | 0.1 | Ngưỡng IV tối thiểu để giữ biến |
| `CORR_THRESHOLD` | 0.89 | Ngưỡng tương quan cặp (đa cộng tuyến) |
| `VIF_THRESHOLD` | 5 | Ngưỡng VIF tối đa |
| `MIN_BIN_SIZE` | 0.05 | Tỷ lệ quan sát tối thiểu trong 1 bin |
| `MAX_BINS` / `MIN_BINS` | 10 / 2 | Số bin tối đa / tối thiểu |
| `MONOTONIC_ENFORCE` | True | Bắt buộc WOE đơn điệu theo bin |
| `PSI_THRESHOLD` | 0.1 | Ngưỡng PSI kiểm tra ổn định train→test |
| `ADAPTIVE_SMOOTH`, `MIN_SMOOTH_EPS`, `MAX_SMOOTH_EPS` | True, 0.05, 0.3 | Làm phẳng thích ứng khi bin có 0 event/non-event |
| `MIN_BIN_COUNT` | 50 | Số quan sát tối thiểu tuyệt đối trong 1 bin |
| `MIN_EVENTS_PER_BIN` / `MIN_NON_EVENTS_PER_BIN` | 10 / 10 | Tối thiểu mỗi lớp trong 1 bin |
| `IV_STABILITY_N_BOOTSTRAP` | 50 | Số lần bootstrap kiểm định ổn định IV |
| `IV_STABILITY_CV_THRESHOLD` | 0.1 | Hệ số biến thiên IV tối đa |
| `SPECIAL_FLOAT_BINNING` | `{"LTV_CIF": "tín chấp"}` | Giá trị đặc biệt tách bin riêng, phần còn lại binning số |

### Biến đổi (khối MAIN cuối tệp)
1. `load_and_validate` → `format_billing_acctnum` → `validate_existing_train_test_split`.
2. Tách `df_train` / `df_test` theo `train_test` (không chia lại).
3. Nhận diện cột "numeric-like" trên train (`parse_ratio=0.95`), ép kiểu cho cả hai tập.
4. `split_features`: tách numeric/categorical (loại ID, target, split).
5. `auto_bin_select` (chỉ trên train): thử mọi số bin từ `MAX_BINS` xuống `MIN_BINS`;
   binning số (quantile + gộp khoảng + ép đơn điệu) và binning phân loại (gộp nhóm);
   tính **WOE = ln(EventDist / NonEventDist)**, **IV = Σ (EventDist − NonEventDist) × WOE**
   với làm phẳng thích ứng khi có bin 0 sự kiện; chọn ứng viên IV cao nhất thỏa
   `min_size ≥ MIN_BIN_SIZE` và `IV ≥ IV_THRESHOLD`; kiểm định ổn định IV bằng bootstrap.
6. `woe_transform`: ánh xạ mỗi biến đã chọn thành cột `<biến>_bin_woe`.
7. `handle_multicollinearity`: loại biến theo tương quan cặp > `CORR_THRESHOLD` (giữ IV cao hơn)
   và VIF > `VIF_THRESHOLD` lặp lại.
8. `transform_test_with_train_binning`: áp binning của train cho test (không fit lại).
9. `check_woe_stability`: PSI từng biến WOE giữa train/test, ngưỡng `PSI_THRESHOLD`.

### Đầu ra
- `credit_data_woe_raw.parquet` (df4: ID + cột `_bin`, `_bin_woe` + `NX` + `train_test`).
- `credit_data_woe.parquet` (df4_v2: ID + các cột WOE đã bỏ hậu tố `_bin_woe` + `NX` + `train_test`).
- Thư mục `credit_data_woe/`: XLSX df4 của **riêng các dòng test**, tách theo từng `NGAY_BC`
  (hàm `export_df4_to_excel_by_ngay_bc` lọc `train_test == 'test'` trước khi xuất).
- Thư mục `mapping/`: XLSX bảng ánh xạ bin→WOE cho từng biến cuối cùng.

### Thư viện / môi trường
`pandas`, `numpy`, `pyarrow`, `openpyxl` (ghi XLSX), `gc`, `re`, `sys`, `os`.
Lưu ý: khi kiểm tra điều kiện đầu vào thất bại, mã gọi `sys.exit(0)` — mã thoát 0 kể cả khi lỗi.
Bộ chuyển đổi của web phát hiện thất bại bằng đầu ra kỳ vọng + dấu hiệu `HOÀN TẤT` trong stdout.

---

## 3. Step 3 — `step 3 modelling V2.py` (199.676 byte, 5.507 dòng)

Tệp gồm nhiều khối nối tiếp trong cùng một tiến trình (khối sau kế thừa biến khối trước).

### Đầu vào
- `D:\model\credit_data_woe.parquet`; `TARGET_COL="NX"`, `SPLIT_COL="train_test"`,
  `ID_COLS=["BILLING_ACCTNUM","CIF","NGAY_BC","NX","train_test"]`.
- Điều kiện: `NX` nhị phân 0/1 không thiếu; `train_test` chỉ gồm {train,test} và **cả hai
  tập đều phải có dòng**; đặc trưng = mọi cột số còn lại (ép float, fillna 0 ở tầng mảng).
- Chia tập thuần theo cột `train_test` ("KHÔNG dùng train_test_split").

### Khối A — Chọn mô hình logistic (fit/chọn/ngưỡng 100% trên train)
- `LogisticRegression(solver="newton-cholesky")` cố định; `random_state=42`, `n_jobs=1`.
- Lưới cấu hình (`GRID_BASE`): scaler ∈ {Standard, Robust, MinMax}; C ∈ {1.0, 0.1, 10.0, 0.01};
  penalty ∈ {l2, None nếu solver hỗ trợ (tự dò)}; class_weight ∈ {None, "balanced", {0:1, 1:n0/n1}};
  `max_iter=10000`, `tol=1e-4`; giới hạn mềm 36 cấu hình cơ sở, mở rộng (`GRID_EXPAND`)
  quanh hạt giống, trần cứng `MAX_CONFIGS=80`.
- Hệ số được quy đổi về không gian gốc (`ScaledLogisticWrapper.raw_coef_`, `raw_intercept_`).
- Quét ngưỡng trên train: `THRESHOLDS = 0.05 → 0.92` bước `0.0005` (1.741 điểm);
  quy tắc khoảng cách `gap ≥ GAP_FRACTION × gap_max` (`GAP_FRACTIONS` quét lại
  {0.50, 0.60, 0.65, 0.75, 0.85} không fit lại).
- **9 cổng KPI** (`KPI_GATES`): Sensitivity ≥ 0.70; MCC ≥ 0.68; Specificity ≥ 0.95;
  Precision ≥ 0.70; Gap=Recall−Precision > 0; BCE ≤ 0.15; NPV ≥ 0.95; F1 ≥ 0.70;
  Accuracy ≥ 0.95. TEST/FULL chỉ dùng kiểm chứng sớm (theo ghi chú trong mã), không vào khóa sắp xếp.
  Nếu không cấu hình nào đạt đủ cổng, mã vẫn tiếp tục theo nhánh "BEST-EFFORT" và in kết luận tương ứng.

### Khối B — Điểm số (scorecard) và hạng (grade) — tính trên train
- `probability_NX`, `log_odds = X·coef + intercept`, `probability_manual = sigmoid(log_odds)`,
  cột `diff` đối chiếu hai cách tính.
- Decile PD: ngưỡng `qcut` 10 phần theo phân phối train → cột `decile` (1 tốt nhất).
- Thang điểm: `min_score=300`, `max_score=850`, `base_odds=50`;
  `pdo = (max_score−min_score)·ln2 / ( ln((1−min_PD)/min_PD) − ln((1−max_PD)/max_PD) )`
  với min_PD/max_PD là min/max PD trên train; `factor = pdo/ln2`;
  `offset = base_score − factor·ln(base_odds)`; `score = round(offset − factor·log_odds)`.
- Grade: 10 hạng `0.AAA … 9.D` chia **đều theo khoảng điểm** min→max của train.
- Xuất: `PD decile.xlsx`, `score_parameter.xlsx`, `Grade_score.xlsx`.

### Khối C — Hiệu chỉnh Platt 2 tầng (fit chỉ trên train)
- Tầng 1: 3 class_weight × Platt trên log-odds out-of-fold (StratifiedKFold 10 fold,
  shuffle, random_state=42) + 1 neo MLE in-sample → 4 cấu hình neo (α, β).
- Tầng 2: tinh chỉnh δ ∈ [−0.15, +0.25] bước 0.01 (41 điểm) × s ∈ [0.95, 1.20] bước 0.01
  (26 điểm) = 1.066 ứng viên/neo, tổng 4.264 ứng viên; chấm theo KPI trên các tập.
- Sinh `log_odds_new = α + β·log_odds`, `probability_manual_new`, thang điểm mới
  (`score_new`, `base_score_new`, ...) và `Grade_new`; xuất bản `_new` của 3 tệp trên
  + `calibration.xlsx`.
- `evaluate_new(...)`: đánh giá KPI trên TRAIN/TEST/FULL tại ngưỡng đã chọn.

### Khối D — Xuất tham số, điểm số, báo cáo Gini/KS theo hạng
- `parameter_model.xlsx` (intercept + hệ số theo từng biến).
- `credit_data_scoring.parquet` (toàn bộ `data_group` sau chấm điểm, nén brotli).
- `gini_ks_report(...)` trên tập test theo `Grade_new`: Grouped AUC/Gini, KS, IV total,
  Model AUC/Gini, AUC grouping loss → `Gini_KS_result.xlsx` (sheet "Grade Detail", "Summary").

### Khối E — Kiểm định & biểu đồ (AUTO_APPEND)
- Thư mục `D:/model/plot/` (có nhánh dự phòng biến môi trường trong mã), thư mục con `model_plot/`,
  log `runtime_log.txt` (tee stdout).
- 17 chỉ số (`METRIC_ROWS`, sheet `Metrics_17`) gồm: AUC (kèm khoảng tin cậy DeLong/Hanley),
  Gini, KS, ECE/MCE, Brier (phân rã), hệ số hiệu chuẩn (α̂, β̂), AP, Balanced Accuracy,
  AP, ... cùng bảng `statistical_assessment.xlsx` (phương pháp, công thức, ý nghĩa, mức đánh giá).
- PSI điểm/xác suất (`psi_detail.xlsx`), IV từng biến (`iv_detail.xlsx`),
  độ lệch train–test & đường cong học (`overfitting_stability.xlsx`),
  ổn định theo `NGAY_BC` (`temporal_stability_ngaybc.xlsx`),
  đối chiếu 3 chiều console ↔ Excel ↔ chú thích biểu đồ, `output_manifest.xlsx`.
- 22 biểu đồ PNG (dpi=150): ROC, PR, KS, Lorenz/Gini, CAP, Lift, Cumulative Gain,
  phân phối điểm/xác suất, bad rate theo decile/grade, calibration, predicted vs observed,
  confusion matrix, threshold analysis, stability, PSI by bin, feature importance,
  learning curve, validation curve, temporal stability, IV by feature.

### Thư viện / môi trường
`numpy`, `pandas`, `scikit-learn` (LogisticRegression, scalers, metrics, model_selection),
`scipy` (expit, norm, ks_2samp), `matplotlib` (có `plt.show()` — cần backend không hiển thị
khi chạy trên máy chủ), `pyarrow`/`brotli`, `openpyxl`. Không có `input()`, không gọi mạng,
không `eval/exec` (đã quét mã tĩnh ngày 26/10/2026).

---

## 4. Đặc tả bộ chuyển đổi của ứng dụng web (`app/adapters.py`)

Nguyên tắc: **không sửa tệp gốc trong `pipeline/`**. Mỗi lần chạy, bộ chuyển đổi:

1. Tạo thư mục chạy riêng `runs/<run_id>/model/…` (nằm trong `.gitignore`).
2. Sao chép mã gốc vào thư mục chạy và chỉ thay thế chuỗi đường dẫn:
   - `D:\model\` và `D:/model/` → `<thu_muc_chay>/model/`; `D:\model` → `<thu_muc_chay>/model`.
   - Riêng Step 1: dòng gán `folder_path = "D:\model\SKV parquet"` →
     `folder_path = "<thu_muc_chay>/model/SKV parquet"`; mẫu glob `"\*.parquet"` → `"/*.parquet"`
     (dấu phân cách thư mục của Linux).
   - Tác động: thuần túy đổi nơi đọc/ghi tệp; **không** đổi công thức, tham số, ngưỡng,
     thứ tự xử lý. Kiểm thử `tests/test_adapters.py` xác minh bản sao sau thay thế vẫn
     biên dịch được và không còn chuỗi đường dẫn `D:`.
3. Đặt đầu vào vào đúng vị trí mã gốc mong đợi, với **tên tệp an toàn do hệ thống sinh**
   (ví dụ `skv_001.parquet`, `credit_data.parquet`); không bao giờ đưa tên tệp/nội dung
   người dùng tải lên vào mã được thực thi.
   - Nếu người dùng tải `credit_data` dạng nhiều phần (như credit_data1+2), bộ chuyển đổi
     nối dọc (concat) thành một `credit_data.parquet` — phép nghịch đảo đúng của thao tác
     chia đôi ở cuối Step 1.
   - Nếu tải CSV/XLSX/TXT: bộ chuyển đổi đổi tên cột theo ánh xạ người dùng xác nhận và
     phân tích cột ngày theo định dạng người dùng chọn (mặc định `dd/mm/yyyy` như mã gốc),
     rồi ghi parquet. Không thay đổi giá trị nghiệp vụ nào khác.
4. Chạy bằng tiến trình con (`subprocess`) với `MPLBACKEND=Agg`, thư mục làm việc là thư mục
   chạy, có giới hạn thời gian cấu hình được; thu stdout/stderr vào log của lần chạy.
5. Xác định thành công bằng **đầu ra kỳ vọng** (không chỉ mã thoát, vì Step 2 thoát 0 khi lỗi):
   - Step 1: có `credit_data.parquet` + `credit_data1.parquet` + `credit_data2.parquet`.
   - Step 2: có `credit_data_woe.parquet` + `credit_data_woe_raw.parquet` và stdout chứa `HOÀN TẤT`.
   - Step 3: có `credit_data_scoring.parquet` + `parameter_model.xlsx` + `plot/metrics_summary.xlsx`.

## 5. Ranh giới chưa xác minh & điều chưa rõ

| # | Nội dung | Trạng thái | Hệ quả trong ứng dụng |
|---|---|---|---|
| 1 | Khối lọc dữ liệu theo TT14 (2 vị trí ghi chú trong Step 1) | **Vắng mặt** trong tệp đã truy cập | Step 1 chạy theo đúng tệp đã truy cập; ứng dụng gắn nhãn đầu ra "chưa lọc TT14 — thiếu khối của chủ sản phẩm" |
| 2 | Khối chia train/OOT 30% điền `train_test` (vùng `#######chỉnh từ vùng này`) | **Vắng mặt**; tệp chỉ gán `train_test=''` | Đầu ra Step 1 không đạt điều kiện Step 2 → ứng dụng **chặn** chạy tiếp, nêu rõ khối cần bổ sung; người dùng có thể tải `credit_data.parquet` đã hoàn thiện từ lần chạy Step 1 đầy đủ ở nơi khác |
| 3 | Ý nghĩa pháp lý/điều kiện lọc của Thông tư 14/2025/TT-NHNN đối với dữ liệu | Chưa xác minh pháp lý (liên kết thuvienphapluat truy cập được ngày 26/10/2026 nhưng không phải nguồn chính thức; chưa đối chiếu văn bản gốc) | Ứng dụng không diễn giải, không tuyên bố tuân thủ; yêu cầu chuyên gia pháp lý rà soát |
| 4 | Môi trường gốc của tác giả (phiên bản thư viện khi viết code) | Không xác định (không có thông tin trong tệp) | Kiểm thử dùng môi trường khai báo trong `requirements.txt`; kết quả số có thể lệch nhỏ theo phiên bản — ghi rõ trong báo cáo kiểm thử |
| 5 | Tính tương đương kết quả giữa chạy web (Linux) và chạy gốc (Windows) | **Chưa xác minh tương đương** trên cùng dữ liệu thật vì thiếu khối #1/#2 và chưa có kết quả gốc để đối chiếu | Báo cáo của ứng dụng ghi "chưa xác minh tương đương với lần chạy gốc"; điều kiện để khẳng định: cùng đầu vào + cùng cấu hình + đối chiếu bước trung gian/đầu ra với sai số phù hợp |
| 6 | Thời gian chạy Step 2/3 trên 1,53 triệu dòng trong môi trường web | Chưa đo trên dữ liệu thật (bị chặn ở #2 trước khi chạy) | Ứng dụng công bố giới hạn dung lượng/tài nguyên đã cấu hình và cảnh báo thời gian chạy chưa kiểm chứng ở quy mô đầy đủ |
