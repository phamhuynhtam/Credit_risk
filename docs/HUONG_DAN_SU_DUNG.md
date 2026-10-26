# Hướng dẫn sử dụng

Đối tượng: chuyên viên ngân hàng và quản trị rủi ro vận hành ứng dụng web mô
hình PD (hồi quy logistic) theo quy trình Step 1–3 trong repo này.

## 1. Trước khi bắt đầu

- Chuẩn bị dữ liệu ở **đúng một điểm vào** trong bảng dưới. Định dạng file:
  `.parquet`, `.csv`, `.xlsx`, `.txt`; giới hạn hiện tại **200 MB mỗi tệp**
  (chủ sản phẩm cấu hình trong `app/config.py` và `.streamlit/config.toml`).
- Chỉ tải dữ liệu thật lên môi trường đã được tổ chức của bạn phê duyệt cho
  dữ liệu đó. Bản demo công khai chỉ dùng dữ liệu giả lập.

| Điểm vào | Tệp cần có | Điều kiện bắt buộc | Chạy được |
|---|---|---|---|
| A. SKV thô | 1–24 tệp bảng có 35 cột đầu vào của Step 1 (gồm `NHOM_NO`) | Cột ngày đúng định dạng dd/mm/yyyy nếu là chuỗi | Step 1. Đầu ra **chưa** vào được Step 2 vì khối chia tập vắng mặt trong mã đã truy cập (xem mục 4) |
| B. credit_data | `credit_data.parquet` tương đương (36 cột), một tệp hoặc tối đa 8 phần nối dọc | `NX` ∈ {0,1}; `train_test` ∈ {train,test} đã điền sẵn | Step 2 → Step 3 |
| C. credit_data_woe | Tệp WOE do Step 2 sinh ra ở nơi khác | Có `NX`, `train_test` (đủ cả train và test), ít nhất 1 biến số | Step 3 |
| D. Giả lập | Không cần tệp | — | Chạy thử Step 2–3; kết quả chỉ minh họa kỹ thuật |

## 2. Tải dữ liệu & kiểm tra

1. Chọn điểm vào, tải tệp. Hệ thống kiểm tra định dạng và dung lượng ngay;
   tệp không đạt bị từ chối trước khi phân tích.
2. **Ánh xạ cột**: nếu tên cột của bạn khác tên cột quy trình (ví dụ `NGAY_BC`),
   hãy ghép từng cột. Hệ thống chỉ đổi tên cột khi đặt đầu vào, không sửa giá trị.
3. **Khai báo thời gian**: chọn cột ngày quan sát, kỳ quan sát (giai đoạn lấy
   mẫu) và kỳ dự báo (tháng theo dõi kết quả).
   - Kiểm tra phủ thời gian sẽ chặn nếu điểm quan sát muộn nhất cộng kỳ dự báo
     vượt quá ngày dữ liệu cuối — trường hợp hồ sơ chưa đủ thời gian theo dõi
     gọi là bị cắt phải (right-censoring). Khắc phục: bổ sung dữ liệu tháng sau,
     rút ngắn kỳ quan sát, hoặc khai báo kỳ dự báo đúng với định nghĩa nhãn của bạn.
4. Bấm **Chạy kiểm tra**. Xử lý hết các dòng đỏ (lỗi chặn); đọc các dòng vàng
   (cảnh báo: trùng lặp, tháng trống, chồng lấn thời gian train/test...) trước
   khi chạy vì chúng ảnh hưởng diễn giải kết quả.
5. Xem trước dữ liệu (cột định danh đã che), bảng thiếu theo cột và biểu đồ
   theo tháng để rà soát bất thường.

## 3. Chạy mô hình

- Trang **Chạy mô hình** mở khóa từng nút theo điều kiện. Trong khi chạy, nhật
  ký của chính mã Step 1–3 được hiển thị trực tiếp; mỗi bước có trần thời gian
  1.800 giây (cấu hình được).
- Một bước được coi là hoàn tất **chỉ khi đủ đầu ra kỳ vọng** xuất hiện, không
  chỉ khi tiến trình kết thúc êm — đặc biệt Step 2 sẽ kết thúc “êm” ngay cả khi
  đầu vào không đạt; ứng dụng sẽ báo chưa hoàn tất kèm nhật ký của bước.
- Không đóng phiên trình duyệt giữa bước chạy dài; kết quả từng bước được lưu
  trong thư mục chạy của phiên trên máy chủ cho tới khi bạn xóa.

## 4. Ranh giới hiện tại: khối TT14/OOT trong Step 1

Mã Step 1 đã truy cập **chưa chứa** khối lọc dữ liệu TT14 và khối chia
train/OOT điền cột `train_test` (hai vị trí vẫn là ghi chú trống; kiểm chứng
trực tiếp: đầu ra có `train_test` rỗng 100%). Hệ quả và cách xử lý:

- Điểm vào A chạy được Step 1 để đối chiếu các biến đổi đã xác minh
  (tạo `NX` từ `NHOM_NO >= 3`, 3 biến DURATION, làm sạch định danh), nhưng
  Step 2 sẽ bị chặn. Đây là hành vi đúng thiết kế của ứng dụng, không phải lỗi.
- Để chạy tiếp, cần **một trong hai**: (a) chủ sản phẩm bổ sung khối của mình
  vào đúng các vị trí đã đánh dấu trong `pipeline/step 1 làm dữ liệu mô hình.py`
  rồi cập nhật hàm băm và tài liệu đi kèm; hoặc (b) dùng điểm vào B với tệp
  credit_data từ một lần chạy Step 1 đầy đủ bên ngoài.
- Ứng dụng **không** có nút tự chia train/test cho dữ liệu thật.

## 5. Kết quả & báo cáo

- Bảng Metrics_17, hệ số mô hình, tham số thang điểm, khoảng PD theo decile,
  khoảng điểm theo grade, PSI/IV và 22 biểu đồ kiểm định đều là artifact do
  chính Step 2/Step 3 sinh ra; ứng dụng đọc và trình bày lại, không tính lại.
- Đọc kỹ mục **“Phần đã kiểm định và phần chưa kiểm định”** trên trang kết quả
  và trong báo cáo trước khi trích dẫn số liệu đi nơi khác.
- Tệp `credit_data_scoring.parquet` chứa điểm/PD **cấp từng hồ sơ** cùng cột
  định danh: chỉ tải về môi trường được phê duyệt; báo cáo tổng hợp (Excel/HTML)
  không chứa giá trị định danh.
- Sau khi tải xong mọi thứ cần giữ, bấm **Xóa dữ liệu phiên** để gỡ thư mục
  chạy khỏi máy chủ ngay lập tức.

## 6. Sự cố thường gặp

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| “train_test có N giá trị rỗng” | Tệp là đầu ra Step 1 chưa có khối chia tập | Xem mục 4 |
| “Thiếu phủ thời gian cho kỳ dự báo” | Kết thúc quan sát + kỳ dự báo > ngày dữ liệu cuối | Lùi kết thúc kỳ quan sát hoặc bổ sung dữ liệu |
| Không thấy biến nào sau Step 2 | Không biến nào đạt đồng thời IV ≥ 0.1, cỡ bin và ổn định theo ngưỡng gốc | Kiểm tra nhật ký Step 2 (lý do loại theo biến); rà soát định nghĩa nhãn và chất lượng biến; ngưỡng nằm trong mã gốc, ứng dụng không tự nới |
| Step 3 báo BEST-EFFORT | Không cấu hình nào đạt đủ 9 cổng KPI của mã gốc trên dữ liệu này | Kết quả vẫn là cấu hình tốt nhất theo quy tắc gốc; đọc bảng KPI chưa đạt trong nhật ký và đưa vào thẩm định mô hình thay vì coi là đạt chuẩn |
| Hết thời gian 1.800s | Dữ liệu lớn hoặc máy chủ yếu | Chạy trên máy mạnh hơn/máy chủ nội bộ; không có cam kết thời gian ở quy mô chưa đo (xem báo cáo kiểm thử) |
