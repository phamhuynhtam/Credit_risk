# Nguồn gốc mã pipeline Step 1–3 (NGUON_GOC)

Thư mục `pipeline/` chứa **bản sao nguyên vẹn từng byte** của 3 tệp mã nguồn do chủ
sản phẩm cung cấp qua Google Drive. Ứng dụng web **không sửa** các tệp này; khi chạy,
bộ chuyển đổi (`app/adapters.py`) tạo một bản sao thời gian chạy trong thư mục chạy
riêng và chỉ thay thế **chuỗi đường dẫn** `D:\model` / `D:/model` bằng đường dẫn của
thư mục chạy (xem đặc tả bộ chuyển đổi trong `docs/BAN_DO_STEP_1_3.md`).

| Tệp trong repo | Google Drive File ID | Dung lượng (byte) | SHA-256 |
|---|---|---|---|
| `step 1 làm dữ liệu mô hình.py` | `1mzMVEuyGT2VneFqVLqQsfxCbrs10LoYS` | 14.517 | `d3870cd7806f6eb2dc2e6fdee8343ba1cafea2b328ffeb52264c6e9270baee44` |
| `step 2 lọc biến mô hình.py` | `1jku0FXCiwlA-2qp7DgCgU5qpsUs6hYE8` | 100.142 | `8c2e810a0ba2a870c0d02d608da5b06ba37866a72500f9eda89d8d4432b40294` |
| `step 3 modelling V2.py` | `18yxqgWuqE5mD0Ag9-pIzn1r_xQ1mcsTY` | 199.676 | `6d17d71f8afdabc301e42cb720c800022b2e345230164bda03a3337219b4fd84` |

- Ngày tải xuống để tích hợp: 26/10/2026 (qua kết nối Google Drive của tài khoản chủ sản phẩm).
- Cách kiểm tra lại tính nguyên vẹn: chạy `sha256sum pipeline/*.py` và đối chiếu bảng trên.
- Kiểm thử tự động `tests/test_adapters.py` cũng xác minh các hàm băm này để phát hiện
  mọi thay đổi ngoài ý muốn đối với mã gốc.

## Dữ liệu mẫu đi kèm (KHÔNG nằm trong repo)

| Tệp | Google Drive File ID | Dung lượng | SHA-256 | Ghi chú đã kiểm chứng |
|---|---|---|---|---|
| `credit_data1.parquet` | `1RjdzSoIhDXV6Z6fJHQo9PvxhNLQaP641` | 14.548.485 byte | `25f08a00d9cd73346a7dd990dddf6b8919cdd0c758fe071f597fb90918249512` | 764.530 dòng × 36 cột, khớp schema đầu ra Step 1; cột `train_test` = chuỗi rỗng ở **toàn bộ** dòng |
| `credit_data2.parquet` | `1grTQIEqQ0_k9-ZDhJ5eENZLJ1GFM4d_o` | 15.162.085 byte | `10becbff4dae1bd66aa8a3dc9ace082647d89b9a92c3379d03b0a477d9b9fce2` | 764.529 dòng × 36 cột; cột `train_test` = chuỗi rỗng ở **toàn bộ** dòng |

Hai tệp dữ liệu trên **không được đưa vào repo** (nguyên tắc tối thiểu hóa dữ liệu);
chỉ metadata cấu trúc (tên cột, kiểu dữ liệu, số dòng, phân phối gộp của `NX`/`train_test`,
khoảng `NGAY_BC`) được ghi lại trong `docs/BAN_DO_STEP_1_3.md`. Không có giá trị định danh
nào được trích xuất hay lưu trữ.

## Trạng thái khối còn thiếu của Step 1 (đã xác minh trong mã)

Trong `step 1 làm dữ liệu mô hình.py` bản đã truy cập:

- Có 2 vị trí ghi chú `#thêm phần lọc dữ liệu TT14 tại đây` đang **trống**.
- Khối `#######chỉnh từ vùng này` … `####### chỉnh đến vùng này` chỉ chứa
  `data_sktv['train_test'] = ''`.

Do đó đoạn thay thế của chủ sản phẩm (lọc TT14, chia OOT 30% để điền `train_test`)
**chưa có trong tệp đã truy cập**. Ứng dụng dừng Step 1 ở ranh giới này, không tự viết
thay (chi tiết trong `docs/BAN_DO_STEP_1_3.md`).
