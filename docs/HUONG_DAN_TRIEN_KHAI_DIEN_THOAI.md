# Hướng dẫn triển khai (deploy) qua GitHub bằng điện thoại — bản chi tiết

Ứng dụng là một app **Streamlit** (tệp chính `streamlit_app.py`, thư viện ghim
phiên bản trong `requirements.txt`). Tài liệu này hướng dẫn chủ sản phẩm tự
đưa bản **demo** lên Streamlit Community Cloud chỉ bằng điện thoại, theo đúng
từng thao tác trên GitHub và trên trang triển khai.

> Trạng thái xác minh của tài liệu này (26/10/2026): các bước trên Streamlit
> Community Cloud được chép từ tài liệu chính thức của Streamlit (liên kết ở
> cuối), kết hợp những gì đã tự kiểm chứng được: repo
> `phamhuynhtam/Credit_risk` đang ở chế độ **công khai**; mọi thư viện trong
> `requirements.txt` đều có wheel Linux x86_64 cho cả Python 3.11 và 3.12 trên
> PyPI (cài nhanh, không biên dịch tại chỗ). Repo này **chưa được deploy
> thật**: thao tác bấm Deploy cần tài khoản GitHub/Streamlit của chính chủ sản
> phẩm. Tên nút trên nền tảng có thể thay đổi theo thời gian.

## Chọn một trong hai đường

| Đường | Cách làm | Khi nào nên dùng |
|---|---|---|
| **A. Deploy thẳng từ nhánh tính năng** | Trỏ Streamlit vào nhánh `arena/eac91acf-credit-risk` | Muốn xem ngay, chưa muốn gộp code. Nhược điểm: nếu nhánh bị xóa sau này, app sẽ lỗi cho tới khi bạn trỏ lại (phải xóa app và deploy lại) |
| **B. Gộp về `main` rồi deploy (khuyến nghị)** | Merge Pull Request vào `main` trên GitHub, sau đó trỏ Streamlit vào `main` | Ổn định lâu dài, URL deploy không phụ thuộc nhánh phụ; đây cũng là cách các bản demo thường trỏ vào nhánh chính |

Streamlit chỉ theo dõi **đúng một nhánh** bạn đã chọn khi deploy; muốn đổi
nhánh phải xóa app và deploy lại. Vì vậy hãy chốt đường A hay B ngay từ đầu.

## Cảnh báo bắt buộc trước khi triển khai công khai

- Demo công khai **chỉ dùng dữ liệu giả lập** (điểm vào “D. Dữ liệu GIẢ LẬP
  minh họa” trong ứng dụng). Tuyệt đối không tải dữ liệu ngân hàng/khách hàng
  thật lên dịch vụ công cộng hoặc bên thứ ba khi chưa được phê duyệt và chưa
  đánh giá đầy đủ nơi lưu trữ, kiểm soát truy cập, xử lý dữ liệu, điều khoản
  hợp đồng với nhà cung cấp và yêu cầu pháp luật.
- Repo đang **công khai** ⇒ mã nguồn ai cũng xem được, và app deploy từ repo
  công khai là app công khai: ai có liên kết đều mở được. Ứng dụng không có
  tài khoản/phân quyền tích hợp.
- Dữ liệu của một phiên được ghi vào thư mục `runs/` trên máy chủ chạy ứng
  dụng cho tới khi bị xóa (nút “Xóa dữ liệu phiên” hoặc dọn dẹp best-effort
  khi khởi động sau số giờ cấu hình); trên đám mây công cộng, mọi tệp bạn tải
  lên đều đã rời khỏi thiết bị của mình — thêm một lý do chỉ dùng giả lập.

---

## Phần 1 — (Đường B) Gộp code về `main` bằng điện thoại

Bỏ qua phần này nếu đi đường A.

1. Mở **Pull Request** từ nhánh `arena/eac91acf-credit-risk` về `main`:
   - Nếu một PR cho nhánh này đã được tạo sẵn, chỉ cần mở PR đó từ trang
     **Pull requests** của repo.
   - Nếu chưa có: trên `github.com` (trình duyệt điện thoại, có thể cần bật
     “Trang web cho máy tính” nếu giao diện di động thiếu nút): repo →
     **Pull requests** → **New pull request** → `base: main` ←
     `compare: arena/eac91acf-credit-risk` → **Create pull request** → đặt
     tiêu đề → **Create pull request**.
2. Trong PR, mở thẻ **Files changed** lướt kiểm danh sách tệp (kỳ vọng: 25
   tệp của ứng dụng, **không** có tệp dữ liệu `.parquet`).
3. Kéo xuống cuối PR → **Merge pull request** → **Confirm merge**.
4. Sau khi merge xong, có thể bấm **Delete branch** để dọn nhánh phụ (chỉ làm
   khi đã chắc chắn deploy từ `main`).
5. Kiểm tra nhanh: mở repo, chuyển bộ chọn nhánh sang `main`, bảo đảm thấy
   `streamlit_app.py`, `requirements.txt`, `.streamlit/config.toml`, thư mục
   `app/`, `pipeline/`, `docs/`, `tests/`.

> Có thể thực hiện các bước merge bằng **ứng dụng GitHub** (iOS/Android):
> mở repo → Pull Requests → mở PR → Merge. Tên nút theo phiên bản ứng dụng
> có thể khác đôi chút.

## Phần 2 — Deploy trên Streamlit Community Cloud (từng thao tác)

### Bước 1. Đăng nhập và cấp quyền GitHub

1. Trên trình duyệt điện thoại, mở **share.streamlit.io** → đăng nhập bằng
   **GitHub** (đúng tài khoản sở hữu repo `phamhuynhtam`).
2. Lần đầu, GitHub sẽ hỏi cấp quyền cho Streamlit. Hãy đọc màn hình cấp
   quyền và chỉ đồng ý những gì bạn chấp nhận; quyền quản trị trên repo là
   cần thiết ở bước kết nối repo. Nếu bạn chỉ deploy repo công khai và không
   muốn cấp thêm quyền đối với repo riêng tư, hãy đi thẳng tới
   **share.streamlit.io/deploy** để vào trực tiếp biểu mẫu triển khai.
3. Nếu sau này Streamlit không nhìn thấy repo, vào GitHub → ảnh đại diện →
   **Settings** → **Applications** → kiểm tra mục Authorized OAuth Apps /
   Installed GitHub Apps có Streamlit và đã bao gồm repo này hay chưa.

### Bước 2. Tạo app mới và điền đúng 3 trường

1. Trong workspace Streamlit: **Create app** (góc trên bên phải).
2. Khi được hỏi “Do you already have an app?” → chọn **Yup, I have an app**.
3. Điền:

   | Trường | Giá trị |
   |---|---|
   | Repository | `phamhuynhtam/Credit_risk` |
   | Branch | Đường B: `main` — Đường A: `arena/eac91acf-credit-risk` |
   | Main file path | `streamlit_app.py` |

   Mẹo: danh sách gợi ý của nền tảng có thể không đầy đủ; nếu không thấy
   nhánh của mình, cứ **gõ tay đầy đủ** tên nhánh (kể cả dấu `/` trong
   `arena/eac91acf-credit-risk`). Cách khác: dùng **Paste GitHub URL** và dán
   liên kết trực tiếp tới tệp `streamlit_app.py` trên GitHub.
4. Ô **App URL** (không bắt buộc): đặt subdomain ngắn dễ nhớ, ví dụ
   `pd-logistic-vn`, nếu còn trống. Để trống thì URL sinh tự động dạng dài
   `https://<owner>-<repo>-<file>-<branch>-<hash>.streamlit.app`. Đừng đặt
   tên gây hiểu nhầm là trang của cơ quan quản lý.

### Bước 3. Advanced settings — chọn đúng phiên bản Python (quan trọng)

1. Bấm **Advanced settings** ngay tại màn deploy.
2. Mục **Python version**: chọn **3.11** — đây là phiên bản toàn bộ kiểm thử
   của repo đã chạy trên đó (Python 3.11.2, xem `docs/BAO_CAO_KIEM_THU.md`).
   - Lý do phải chọn tay: phiên bản mặc định của nền tảng thay đổi theo thời
     gian và có thể mới hơn mức các thư viện khoa học kịp hỗ trợ; tệp
     `runtime.txt` **không** có tác dụng chọn Python trên Community Cloud.
   - Phiên bản Python **không đổi được sau khi deploy**; muốn đổi phải xóa
     app và deploy lại (subdomain tùy chỉnh được dùng lại ngay sau khi xóa).
   - Nếu danh sách không còn 3.11, chọn **3.12**: mọi thư viện trong
     `requirements.txt` đã được kiểm chứng có wheel Linux cho cả 3.11 và
     3.12 (ngày 26/10/2026). Lưu ý nền tảng có thể tự nâng phiên bản Python
     khi phiên bản đang dùng hết hạn hỗ trợ bảo mật, app khi đó có thể cần
     kiểm tra lại.
3. Ô **Secrets**: **để trống**. Ứng dụng này không dùng khóa bí mật nào;
   tuyệt đối không dán dữ liệu khách hàng hay thông tin thanh toán vào đây.
4. Bấm **Save**.

### Bước 4. Deploy và theo dõi build

1. Bấm **Deploy**. Giữ nguyên màn hình; log cài đặt hiện ở khung bên (log chỉ
   hiển thị với tài khoản có quyền ghi vào repo).
2. Thời gian kỳ vọng: vài phút cho lần đầu (cài pandas, scikit-learn, scipy,
   pyarrow... đều là wheel có sẵn). Nếu một gói chuyển sang “building wheel”
   hay lỗi biên dịch, nhiều khả năng đã chọn sai phiên bản Python ở Bước 3.
3. Khi app mở ra, kiểm thử nhanh trên chính điện thoại — chỉ với dữ liệu giả
   lập:
   1. Trang **Giới thiệu** hiển thị, bảng SHA-256 báo “Khớp” cả 3 bước.
   2. **Tải dữ liệu & kiểm tra** → điểm vào **D. Dữ liệu GIẢ LẬP** →
      **Tạo dữ liệu giả lập** → **Chạy kiểm tra** (chấp nhận các cảnh báo
      vàng về cắt phải, không có lỗi đỏ).
   3. **Chạy mô hình** → **Chạy Step 2** → đợi xong → **Chạy Step 3**.
   4. **Kết quả & báo cáo**: thấy bảng Metrics_17, hệ số và biểu đồ; tải thử
      báo cáo HTML.
   5. Bấm **Xóa dữ liệu phiên** sau khi thử xong.

## Phần 3 — Vận hành sau deploy

- **Cập nhật code**: cứ push commit mới lên đúng nhánh đã chọn là app tự cập
  nhật; nếu thay đổi `requirements.txt`, nền tảng tự cài lại (vài phút).
- **App “ngủ”**: không có lượt truy cập trong **12 giờ** thì app ngủ. Bất kỳ
  người xem nào cũng đánh thức được bằng nút “Yes, get this app back up!” —
  không phải lỗi, nhưng lần mở đầu sẽ chậm. Trong workspace,
...[truncated 16515 chars]