# Báo cáo kiểm thử

Ngày thực hiện: **26/10/2026**. Môi trường: Linux (sandbox), Python 3.11.2,
thư viện theo `requirements.txt` (pandas 2.2.3, numpy 2.1.3, scipy 1.14.1,
scikit-learn 1.6.1, matplotlib 3.9.2, pyarrow 18.1.0, openpyxl 3.1.5,
brotli 1.1.0, streamlit 1.41.1), pytest 8.3.4.

Toàn bộ kiểm thử dùng **dữ liệu giả lập** sinh bởi `app/synthetic.py` (hạt giống
cố định) — không dùng dữ liệu trong hai tệp parquet của chủ sản phẩm cho kiểm
thử chạy; hai tệp đó chỉ được đọc metadata cấu trúc (xem `docs/BAN_DO_STEP_1_3.md`).

## Cách chạy lại

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -v
```

## Kết quả đã đo

```
21 passed in 22.57s
```

### Nhóm 1 — Trường hợp hợp lệ (chạy được đầu cuối)

| Kiểm thử | Nội dung | Kết quả đo được |
|---|---|---|
| `test_pipeline_e2e.py::test_step2_step3_tron_goi_tren_du_lieu_gia_lap` | 12.000 dòng credit_data giả lập → Step 2 → Step 3 trong cùng thư mục chạy | Step 2 hoàn tất ≈13,5s, giữ 3 biến (LAI_SUAT, DU_NO_ADJUST_CIF, DURATION_FIRST_BOOK_DATE); Step 3 hoàn tất ≈11,2s; 53 artifact gồm đủ 22 biểu đồ PNG, `parameter_model.xlsx`, `credit_data_scoring.parquet`, `plot/metrics_summary.xlsx`; Metrics_17 đọc được (AUC train/test ≈0,725/0,718 trên dữ liệu giả lập — **chỉ** chứng minh đường ống, không phải chất lượng mô hình) |
| Báo cáo tổng hợp | Sinh Excel 18 sheet + HTML từ lần chạy trên | Tạo thành công; khẳng định tự động: mã hồ sơ và CIF mẫu **không xuất hiện** trong cả hai dạng báo cáo |
| Giao diện Streamlit | AppTest (streamlit.testing) mở cả 6 trang | Không ngoại lệ ở cả 6 trang; hai hộp lỗi đỏ duy nhất là banner chủ đích (khối thiếu Step 1; cấm dữ liệu thật cho demo công khai) |

### Nhóm 2 — Trường hợp sai/thiếu phải bị chặn

| Kiểm thử | Kỳ vọng đã xác minh |
|---|---|
| Step 1 trên SKV giả lập, sau đó Step 2 trên chính đầu ra đó | Step 1 hoàn tất với `train_test=''` toàn bộ; **Step 2 tự từ chối** (“LỖI: Cột 'train_test' có N giá trị rỗng sau khi strip”), thoát mã 0, bộ chuyển đổi phát hiện thất bại nhờ thiếu `credit_data_woe.parquet` |
| Thiếu/không nhị phân/một lớp ở cột `NX` | Bị tầng kiểm tra trước chặn kèm hướng dẫn |
| `train_test` rỗng, nhãn lạ (“oot”) | Bị chặn; thông điệp nêu rõ khối TT14/OOT vắng mặt |
| Thiếu cột bắt buộc của schema | Bị chặn, chỉ tên cột thiếu |
| Tệp `.pkl`/`.exe`, tệp rỗng, parquet/xlsx giả chữ ký | Bị từ chối trước khi phân tích nội dung |
| Phủ thời gian không đủ (kết thúc quan sát = ngày dữ liệu cuối với kỳ dự báo 12 tháng; dữ liệu bắt đầu muộn hơn khai báo) | Bị chặn kèm mốc quan sát muộn nhất còn đủ kỳ dự báo và cách bổ sung |
| Parquet rỗng đọc qua `doc_tep_an_toan` | Ném `ValueError` có thông điệp an toàn (không lộ nội dung) |

### Nhóm 3 — Biên dữ liệu/dung lượng và an toàn hiển thị

| Kiểm thử | Kỳ vọng đã xác minh |
|---|---|
| Tệp 201 MB với giới hạn 200 MB | Bị chặn kèm dung lượng thực và giới hạn |
| Phủ thời gian đủ khi lùi điểm quan sát | Đạt kiểm tra; đồng thời vẫn đếm số dòng nguy cơ cắt phải để cảnh báo |
| Xem trước an toàn | `BILLING_ACCTNUM`/`CIF` chỉ còn `***`+4 ký tự cuối, `TEN_CPD` che toàn bộ; giá trị gốc không xuất hiện trong bản xem trước |
| CSV tiếng Việt UTF-8 (giá trị “tín chấp”) | Đọc đúng dấu phân cách `;` tự dò và giữ nguyên chuỗi có dấu |
| Nguyên vẹn mã gốc | SHA-256 của 3 tệp `pipeline/` khớp bản đã truy xuất **trước và sau** mọi lần chạy (fixture kiểm tra lại sau mỗi test chạy pipeline) |
| Thay thế đường dẫn của bộ chuyển đổi | Bản sao 3 bước không còn `D:\`/`D:/`, vẫn biên dịch được; hằng số nghiệp vụ chủ chốt nguyên vẹn (IV 0.1, CORR 0.89, VIF 5, solver newton-cholesky, thang điểm 300/850/odds 50) |

## Lỗi đã phát hiện và đã sửa trong quá trình kiểm thử

1. Chuỗi tìm kiếm đường dẫn của bộ chuyển đổi thừa một dấu `\` khiến `D:\model\`
   không khớp (đầu ra rơi vào đường dẫn sai, Step 1 glob rỗng). Đã sửa ở
   `app/adapters.py::chuyen_doi_duong_dan`; kiểm thử hồi quy:
   `test_chuyen_doi_duong_dan_khong_con_windows_va_bien_dich_duoc`.
2. Tiến trình con nhận đường dẫn kịch bản tương đối trong khi `cwd` là thư mục
   chạy → đường dẫn bị nhân đôi khi caller truyền đường dẫn tương đối. Đã chuẩn
   hóa `resolve()` trong `chay_step`.
3. Thiếu khai báo hằng số che họ tên (`COT_CHE_HOAN_TOAN`) — do kiểm thử
   `test_xem_truoc_che_cot_dinh_danh_va_ho_ten` phát hiện. Đã bổ sung.

## Phần CHƯA chạy / chưa kiểm chứng

- **Chưa** chạy Step 2/Step 3 trên hai tệp dữ liệu được cung cấp
  (`credit_data1/2.parquet`, 1,53 triệu dòng): bị chặn một cách đúng đắn ở
  `train_test` rỗng. Vì vậy thời gian chạy và bộ nhớ ở quy mô 1,53 triệu dòng
  trong môi trường web **chưa được đo**; không tuyên bố hiệu năng ở quy mô đó.
- **Chưa** đối chiếu tương đương kết quả với lần chạy gốc Windows của chủ sản
  phẩm (thiếu khối TT14/OOT và thiếu kết quả gốc để đối chiếu) → trạng thái:
  *chưa xác minh tương đương*.
- **Chưa** kiểm thử trên trình duyệt thật/thiết bị di động thật; giao diện mới
  được kiểm chứng qua AppTest kết xuất không lỗi. Mạnh yếu hiển thị trên điện
  thoại cần xác nhận thực tế trên thiết bị của chủ sản phẩm.
- **Chưa** kiểm thử xâm nhập, chưa đánh giá bảo mật môi trường triển khai cụ thể.
- **Chưa** triển khai lên bất kỳ nền tảng nào; không có website đang hoạt động.
