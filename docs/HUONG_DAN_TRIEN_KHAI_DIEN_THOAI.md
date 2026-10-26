# Hướng dẫn triển khai bằng điện thoại

Ứng dụng là một app **Streamlit** (tệp chính `streamlit_app.py`, thư viện ở
`requirements.txt`). Tài liệu này hướng dẫn chủ sản phẩm đưa bản **demo** lên
mạng chỉ bằng điện thoại, đồng thời nêu rõ giới hạn của cách làm đó.

> Trạng thái hiện tại: repo này **chưa được triển khai ở đâu cả**. Dưới đây là
> hướng dẫn để chủ sản phẩm tự thực hiện khi quyết định triển khai; việc tạo
> tài khoản, công bố website và chịu trách nhiệm nội dung thuộc về chủ sản phẩm.

## Cảnh báo bắt buộc trước khi triển khai công khai

- Demo công khai **chỉ dùng dữ liệu giả lập** (nút “Dữ liệu giả lập minh họa”
  trong ứng dụng). Tuyệt đối không tải dữ liệu ngân hàng/khách hàng thật lên
  dịch vụ công cộng hoặc bên thứ ba khi chưa được phê duyệt và chưa đánh giá
  đầy đủ về nơi lưu trữ, kiểm soát truy cập, xử lý dữ liệu, điều khoản hợp
  đồng với nhà cung cấp và yêu cầu pháp luật áp dụng.
- Ứng dụng không có tài khoản/phân quyền tích hợp: ai có liên kết của bản demo
  công khai đều mở được. Vì vậy bản công khai chỉ phù hợp minh họa giao diện
  bằng dữ liệu giả lập. Dữ liệu thật phải chạy ở môi trường nội bộ có xác thực
  và kiểm soát truy cập của tổ chức.
- Dữ liệu của một phiên được ghi vào thư mục `runs/` trên máy chủ chạy ứng
  dụng cho tới khi bị xóa (nút “Xóa dữ liệu phiên” hoặc dọn dẹp best-effort
  khi khởi động sau số giờ cấu hình). Trên dịch vụ đám mây công cộng, điều này
  đồng nghĩa dữ liệu rời khỏi thiết bị của người dùng — thêm một lý do chỉ
  dùng dữ liệu giả lập ở đó.

## Cách 1 — Streamlit Community Cloud (chỉ cần điện thoại + tài khoản GitHub)

1. Trên điện thoại, mở trình duyệt tới `github.com`, đăng nhập tài khoản của
   chủ sản phẩm, mở repo này và bảo đảm nhánh sẽ triển khai chứa:
   `streamlit_app.py`, `requirements.txt`, `.streamlit/config.toml`,
   thư mục `app/` và `pipeline/`.
2. Mở `share.streamlit.io`, chọn đăng nhập bằng GitHub, cấp quyền cho
   Streamlit truy cập repo (chỉ repo này nếu có lựa chọn giới hạn).
3. Chọn **New app** → chọn repository, branch và tệp chính
   `streamlit_app.py` → **Deploy**.
4. Chờ hệ thống cài thư viện theo `requirements.txt` (vài phút lần đầu).
   Nếu build lỗi, mở mục log của app trên cùng trang để xem lỗi cài đặt.
5. Mở liên kết dạng `https://<ten-app>.streamlit.app` được cấp, kiểm tra:
   trang Giới thiệu hiển thị; chọn điểm vào “Dữ liệu giả lập minh họa” →
   tạo dữ liệu → chạy kiểm tra → chạy Step 2 → Step 3 → xem kết quả.
6. Khi không dùng nữa: vào trang quản trị app trên `share.streamlit.io` →
   Settings → Delete app (hoặc đặt app ở chế độ riêng tư nếu gói của bạn hỗ trợ).

### Giới hạn của Cách 1 (cần biết trước)

- Tài nguyên gói cộng đồng miễn phí có hạn (RAM/CPU) và có thể thay đổi theo
  chính sách nhà cung cấp; **chưa kiểm chứng** chạy Step 2/Step 3 trên dữ liệu
  hàng triệu dòng ở môi trường này. Thực tế nên kỳ vọng chỉ hợp với dữ liệu
  giả lập dung lượng vừa.
- Ứng dụng có thể “ngủ” khi không có truy cập và cần vài chục giây để thức dậy.
- Đĩa của máy chủ là tạm thời: thư mục `runs/` có thể mất khi app khởi động
  lại. Không coi đây là nơi lưu kết quả.
- Mọi điều khoản sử dụng, giới hạn và chính sách dữ liệu của nền tảng do nhà
  cung cấp quyết định; chủ sản phẩm cần đọc trước khi bấm triển khai.

## Cách 2 — Chạy trên máy tính cá nhân hoặc máy chủ nội bộ (cho dữ liệu thật)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run streamlit_app.py
```

Sau đó mở trình duyệt tới địa chỉ Streamlit in ra (mặc định cổng 8501).
Khi dùng cho dữ liệu thật trong tổ chức, tối thiểu cần: đặt ứng dụng sau cơ
chế đăng nhập của tổ chức (mạng nội bộ/VPN/reverse proxy có xác thực), giới
hạn ai truy cập thư mục `runs/`, sao lưu và xóa định kỳ theo chính sách dữ
liệu, và thẩm định mô hình độc lập trước khi dùng kết quả vào quyết định
tín dụng. Các yêu cầu này nằm ngoài phạm vi mã nguồn của repo.

## Cách 3 — GitHub Codespaces bằng điện thoại (chạy thử, không công bố)

Repo có sẵn `.devcontainer/devcontainer.json`. Trên điện thoại: mở repo trên
GitHub → **Code** → **Codespaces** → **Create codespace**; khi môi trường sẵn
sàng, cổng 8501 được chuyển tiếp theo cấu hình devcontainer và có thể mở bản
xem trước ngay trong trình duyệt điện thoại. Cách này phù hợp chạy thử nội
bộ; không dùng làm địa chỉ demo công khai lâu dài và vẫn áp dụng cảnh báo dữ
liệu thật như trên.

## Sau khi triển khai demo

- Chỉ chia sẻ liên kết kèm bối cảnh “bản minh họa kỹ thuật bằng dữ liệu giả
  lập”; không trình bày số liệu từ dữ liệu giả lập như hiệu quả mô hình.
- Không tuyên bố sản phẩm đã được Ngân hàng Nhà nước phê duyệt hay tuân thủ
  Thông tư 14/2025/TT-NHNN (chưa xác minh pháp lý).
- Theo dõi log lỗi của nền tảng; khi sửa mã, đẩy lên đúng nhánh đã triển khai
  để nền tảng tự cập nhật, rồi kiểm tra lại luồng giả lập đầu cuối.
