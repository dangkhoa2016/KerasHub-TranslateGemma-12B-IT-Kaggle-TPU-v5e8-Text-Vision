# Chính sách bảo mật

> 🌐 Language / Ngôn ngữ: [English](SECURITY.md) | **Tiếng Việt**

## Phạm vi dự án và triển khai

Dự án cộng đồng công khai này cung cấp dịch vụ KerasHub/TranslateGemma và các tập lệnh hỗ trợ.

Cloudflare Quick Tunnel và sổ tay Kaggle là môi trường trình diễn/phát triển, không phải nơi lưu trữ sản xuất.

Việc triển khai trong môi trường sản xuất tự chịu trách nhiệm về thiết kế mạng, vận hành và kiểm soát truy cập.

## Phiên bản được hỗ trợ

Bản vá bảo mật nhắm tới nhánh `main` hiện tại và bản phát hành công khai được gắn thẻ mới nhất.

## Báo cáo lỗ hổng bảo mật

Không đưa thông tin xác thực, mã thông báo, chi tiết có thể khai thác hoặc nội dung người dùng nhạy cảm vào issue công khai.

Dùng GitHub Security Advisories khi khả dụng để báo cáo riêng tư một lỗ hổng.

## Ranh giới bảo mật đã triển khai

Xác thực API mặc định được bật; các tuyến dịch, kết quả và thông tin yêu cầu API key.

Khởi động lại dùng mã bí mật riêng và yêu cầu tiêu đề HTTP `X-Restart-Secret`.

Phản hồi kiểm tra tình trạng cơ bản là công khai, trong khi phản hồi chi tiết yêu cầu API key.

Các giới hạn đã cấu hình bao gồm byte yêu cầu, độ dài văn bản, byte/pixel ảnh, số mã thông báo đầu ra, kích thước hàng đợi, kết quả lưu trữ và thời hạn lưu kết quả.

## Gia cố triển khai

Cơ chế kết thúc TLS, giới hạn tốc độ và cô lập đối tượng thuê phải do một triển khai sản xuất cung cấp.

Giữ xác thực API được bật khi dịch vụ có thể truy cập ngoài localhost.

Hạn chế phạm vi truy cập dịch vụ, bảo vệ thông tin xác thực và áp dụng giám sát phù hợp với môi trường triển khai.

## Dữ liệu runtime nhạy cảm

`.env` là cấu hình cục bộ tùy chọn và không có trong bản sao mã nguồn sạch.

`data/api_key.txt`, `data/restart_secret.txt` và `data/tunnel_url.txt` được tạo trong lúc chạy khi các tính năng tương ứng hoạt động.

Các tệp bí mật được tạo có quyền truy cập hạn chế và không được đưa vào kho mã hoặc chia sẻ.

- `.env`
- `data/api_key.txt`
- `data/restart_secret.txt`
- `data/tunnel_url.txt`
- giá trị bearer/API-key, thông tin xác thực đường hầm, khóa bí mật SSH, mã truy cập mô hình và nội dung yêu cầu nhạy cảm

## Trước khi chia sẻ artifact

Chạy:

```bash
python3 scripts/secret_scan.py .
python3 scripts/package_source.py /tmp/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision-v1.0.0.zip
python3 scripts/secret_scan.py /tmp/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision-v1.0.0.zip
```

Đóng gói mã nguồn sạch loại trừ cấu hình cục bộ tùy chọn, thông tin xác thực được tạo trong lúc chạy, nhật ký, trạng thái tiến trình, trạng thái đường hầm, bộ nhớ đệm và các tệp nén được tạo ra.

## Chính sách dependency

Môi trường chạy Kaggle được thiết kế để giữ nguyên bản cài đặt JAX/JAXLIB tương thích với bộ tăng tốc.

Không thêm thao tác nâng cấp JAX/JAXLIB thiếu kiểm soát vào quy trình thiết lập thông thường; bước khởi tạo `libtpu` chỉ chạy có điều kiện và dùng `--no-deps` khi gói chưa được cài đặt.
