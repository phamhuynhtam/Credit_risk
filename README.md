# Ứng dụng web mô hình PD bằng hồi quy logistic

Ứng dụng web tiếng Việt hỗ trợ chuyên viên ngân hàng và quản trị rủi ro chạy
quy trình xây dựng mô hình PD (xác suất vỡ nợ) ba bước có sẵn trong repo:
làm dữ liệu, lọc biến bằng WOE/IV, và hồi quy logistic kèm chấm điểm, hiệu
chỉnh và kiểm định. Giao diện gồm: tải tệp và kiểm tra trước (schema, kiểu,
thiếu, trùng, phủ thời gian theo kỳ quan sát và kỳ dự báo), ánh xạ cột, chạy
Step 1–3 bằng chính mã gốc, xem kết quả/biểu đồ và tải báo cáo, giới thiệu
sản phẩm, gói giá, hướng dẫn sử dụng và triển khai.

> ⚠️ **Ranh giới chưa xác minh quan trọng nhất:** trong mã Step 1 đã truy cập,
> khối lọc TT14 và khối chia train/OOT điền cột `train_test` **vắng mặt** (hai
> vị trí còn là ghi chú trống; `train_test` được gán rỗng). Hai tệp dữ liệu đi
> kèm cũng có `train_test` rỗng ở 100% dòng. Do đó chuỗi Step 1 → Step 2 cho dữ
> liệu thật bị ứng dụng **chặn** tại ranh giới này — ứng dụng không tự viết
> khối còn thiếu. Step 2 → Step 3 chạy được với tệp `credit_data` đã hoàn
> thiện từ nơi khác, hoặc với dữ liệu giả lập tích hợp để kiểm thử. Chi tiết:
> [`docs/BAN_DO_STEP_1_3.md`](docs/BAN_DO_STEP_1_3.md).

## Khởi chạy

Yêu cầu: Python 3.11 (môi trường kiểm thử của repo).

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run streamlit_app.py
```

Mở trình duyệt tới địa chỉ Streamlit in ra (mặc định `http://localhost:8501`).

Kiểm thử tự động (21 kiểm thử, dùng dữ liệu giả lập; kết quả đo trong
[`docs/BAO_CAO_KIEM_THU.md`](docs/BAO_CAO_KIEM_THU.md)):

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -v
```

## Triển khai demo qua GitHub bằng điện thoại

[![Mở trang triển khai Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy)

Nút trên mở thẳng biểu mẫu triển khai của Streamlit Community Cloud. Repo này
**chưa được triển khai**; chủ sản phẩm tự thực hiện bằng tài khoản GitHub của
mình, điền đúng 3 trường sau rồi vào **Advanced settings** chọn **Python 3.11**
trước khi bấm **Deploy**:

| Trường | Giá trị |
|---|---|
| Repository | `phamhuynhtam/Credit_risk` |
| Branch | `main` (sau khi merge Pull Request của nhánh `arena/eac91acf-credit-risk`) hoặc chính nhánh `arena/eac91acf-credit-risk` |
| Main file path | `streamlit_app.py` |

Hướng dẫn từng thao tác trên điện thoại (merge PR, cấp quyền GitHub, chọn
subdomain, giới hạn gói miễn phí, app ngủ sau 12 giờ, xử lý lỗi build,
Codespaces, chạy nội bộ cho dữ liệu thật):
[`docs/HUONG_DAN_TRIEN_KHAI_DIEN_THOAI.md`](docs/HUONG_DAN_TRIEN_KHAI_DIEN_THOAI.md).
Demo công khai **chỉ dùng dữ liệu giả lập**.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `streamlit_app.py` | Ứng dụng web (6 trang) |
| `pipeline/` | **Mã gốc Step 1–3 nguyên vẹn từng byte** + `NGUON_GOC.md` (Drive ID, SHA-256). Không sửa trực tiếp |
| `app/config.py` | Tên sản phẩm, gói giá, giới hạn tệp — nơi chủ sản phẩm chỉnh sửa |
| `app/validation.py` | Kiểm tra trước: định dạng, schema, mục tiêu/chia tập, phủ thời gian, xem trước an toàn |
| `app/adapters.py` | Bộ chuyển đổi: bản sao thời gian chạy của mã gốc (chỉ đổi đường dẫn), chạy tiến trình con, quản lý thư mục chạy |
| `app/report.py` | Báo cáo tổng hợp Excel/HTML từ artifact (không chứa dữ liệu cấp hồ sơ) |
| `app/synthetic.py` | Bộ sinh dữ liệu **giả lập** cho kiểm thử/minh họa (gắn nhãn rõ ràng) |
| `docs/` | Bản đồ Step 1–3, hướng dẫn sử dụng, hướng dẫn triển khai bằng điện thoại, báo cáo kiểm thử, bản nháp thư giới thiệu |
| `tests/` | Kiểm thử bộ chuyển đổi, tầng kiểm tra và chạy đầu cuối |
| `runs/` (gitignore) | Thư mục mỗi lần chạy của ứng dụng; xóa bằng nút “Xóa dữ liệu phiên” |

Tệp `data/gdp_data.csv` và lịch sử template GDP là phần còn lại của repo mẫu
ban đầu, không được ứng dụng này sử dụng và được giữ nguyên để không xóa dữ
liệu sẵn có khi không cần thiết.

## Nguyên tắc tích hợp

- Mã gốc trong `pipeline/` không bị sửa; ứng dụng kiểm tra SHA-256 khi khởi
  động và cảnh báo nếu lệch so với bản đã truy xuất từ Google Drive
  (ngày 26/10/2026, hàm băm trong `pipeline/NGUON_GOC.md`).
- Khi chạy, bộ chuyển đổi sao chép mã vào thư mục chạy riêng và **chỉ** thay
  chuỗi đường dẫn `D:\model`/`D:/model` bằng thư mục đó (cùng dấu phân cách
  glob của Step 1). Công thức, tham số, ngưỡng giữ nguyên; danh sách thay thế
  ghi vào nhật ký từng lần chạy.
- Tên tệp người dùng tải lên không bao giờ được dùng làm đường dẫn hay chèn
  vào mã thực thi; nội dung tệp được đọc bằng bộ phân tích dữ liệu thuần
  (pyarrow/openpyxl/CSV), không nạp pickle, không thực thi.
- Báo cáo và thông báo lỗi chỉ chứa số liệu gộp và tên cột; bản xem trước che
  cột định danh/họ tên.

## Giới hạn và tuyên bố (đọc trước khi dùng thật)

- **Chưa xác minh pháp lý**: ứng dụng không tuyên bố tuân thủ Thông tư
  14/2025/TT-NHNN hay được Ngân hàng Nhà nước phê duyệt. Liên kết văn bản tham
  chiếu không phải nguồn chính thức; cần chuyên gia pháp lý/tuân thủ rà soát.
- **Chưa xác minh tương đương** kết quả với lần chạy gốc trên Windows (thiếu
  khối TT14/OOT và kết quả gốc để đối chiếu trên cùng dữ liệu).
- **Chưa đo hiệu năng** ở quy mô 1,53 triệu dòng trong môi trường web; giới
  hạn công bố hiện tại: 200 MB/tệp, 1.800 giây/bước (cấu hình được).
- Không cam kết an toàn tuyệt đối; dữ liệu phiên tồn tại trên máy chủ tới khi
  người dùng xóa hoặc cơ chế dọn dẹp khởi động kích hoạt (xem tài liệu triển
  khai). Demo công khai chỉ dùng dữ liệu giả lập; dữ liệu ngân hàng thật chỉ
  dùng ở môi trường được phê duyệt.
- Gói giá hiển thị trong ứng dụng là đề xuất của chủ sản phẩm (5 lượt đầu
  miễn phí; 18.000.000 VND/lượt từ lượt 6, giá cơ sở 2026, điều chỉnh theo
  lạm phát có nguồn chính thức), không phải giá thị trường đã kiểm chứng;
  ứng dụng không thu phí hay xác nhận thanh toán.
