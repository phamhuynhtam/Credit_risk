# Bản nháp thư giới thiệu sản phẩm (CHƯA GỬI — DÀNH CHO CHỦ SẢN PHẨM TỰ RÀ SOÁT)

> Đây chỉ là **văn bản nháp** trong repo. Nó chưa được gửi đi từ bất kỳ hệ
> thống nào. Chủ sản phẩm tự rà soát nội dung, tự xác định người nhận hợp lệ
> qua kênh liên hệ chính thức của mình và tự gửi theo quy định áp dụng
> (bao gồm quy định về thư điện tử thương mại/chống thư rác và chính sách
> của tổ chức nhận). Ứng dụng và repo này không thu thập, không suy đoán
> địa chỉ thư của bất kỳ ngân hàng hay cá nhân nào.

---

**Tiêu đề đề xuất:** Giới thiệu công cụ xây dựng mô hình PD bằng hồi quy
logistic và đề xuất buổi trao đổi ngắn

Kính gửi [TÊN ĐƠN VỊ / PHÒNG BAN DO CHỦ SẢN PHẨM ĐIỀN],

Tôi là [HỌ TÊN VÀ THÔNG TIN LIÊN HỆ DO CHỦ SẢN PHẨM CẤU HÌNH]. Tôi đang phát
triển một công cụ web hỗ trợ xây dựng mô hình xác suất vỡ nợ (PD) bằng hồi quy
logistic theo quy trình ba bước: chuẩn bị dữ liệu, lọc biến bằng WOE/IV có
kiểm soát đa cộng tuyến và độ ổn định (PSI), sau đó huấn luyện hồi quy
logistic, hiệu chỉnh xác suất, xây thang điểm và sinh bộ chỉ số kiểm định
(AUC/Gini/KS, hiệu chuẩn, ổn định theo thời gian).

Điểm chúng tôi tập trung là tính truy vết của quy trình: mã mô hình giữ
nguyên một phiên bản duy nhất có hàm băm kiểm chứng, các điều kiện đầu vào
(schema, nhãn, chia tập, độ phủ kỳ quan sát và kỳ dự báo) được kiểm tra và
chặn rõ ràng trước khi chạy, và mọi lần chạy đều xuất báo cáo tách bạch phần
đã tính được khỏi phần chưa kiểm định.

Tôi cũng xin nêu thẳng các giới hạn hiện tại để Quý đơn vị cân nhắc đúng mức:

- Đây là sản phẩm đang phát triển; chưa có tuyên bố tuân thủ Thông tư
  14/2025/TT-NHNN hay phê duyệt của Ngân hàng Nhà nước, và kết quả mô hình
  vẫn cần thẩm định độc lập theo quy trình quản trị rủi ro của Quý đơn vị
  trước bất kỳ ứng dụng vận hành nào.
- Bản demo chỉ dùng dữ liệu giả lập; chúng tôi không đề nghị Quý đơn vị tải
  dữ liệu thật lên bất kỳ dịch vụ công cộng nào.

Nếu Quý đơn vị quan tâm, tôi đề xuất một buổi trao đổi ngắn (khoảng 30
phút) để trình bày quy trình trên dữ liệu giả lập và lắng nghe yêu cầu của
Quý đơn vị về quản trị mô hình. Về chi phí dùng thử theo chính sách hiện
hành của chúng tôi: 5 lượt chạy đầu miễn phí; các lượt tiếp theo áp dụng
mức 18.000.000 VND/lượt theo giá cơ sở năm 2026 do chúng tôi đề xuất và
điều chỉnh hằng năm theo lạm phát có nguồn chính thức — chi tiết sẽ được
trao đổi minh bạch khi làm việc trực tiếp.

Trân trọng cảm ơn Quý đơn vị đã dành thời gian.

[HỌ TÊN, CHỨC DANH, ĐIỆN THOẠI/EMAIL DO CHỦ SẢN PHẨM CẤU HÌNH]
