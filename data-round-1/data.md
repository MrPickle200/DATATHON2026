Tổng quan các bảng dữ liệu

Table 1: Danh sách các file dữ liệu

# File Lớp Mô tả
1 products.csv Master Danh mục sản phẩm
2 customers.csv Master Thông tin khách hàng
3 promotions.csv Master Các chiến dịch khuyến mãi
4 geography.csv Master Danh sách mã bưu chính các vùng
5 orders.csv Transaction Thông tin đơn hàng
6 order_items.csv Transaction Chi tiết từng dòng sản phẩm trong đơn
7 payments.csv Transaction Thông tin thanh toán tương ứng 1:1 với đơn hàng
8 shipments.csv Transaction Thông tin vận chuyển
9 returns.csv Transaction Các sản phẩm bị trả lại
10 reviews.csv Transaction Đánh giá sản phẩm sau giao hàng
11 sales.csv Analytical Dữ liệu doanh thu huấn luyện
12 sample_submission.csv Analytical Định dạng file nộp bài (mẫu)
13 inventory.csv Operational Ảnh chụp tồn kho cuối tháng
14 web_traffic.csv Operational Lưu lượng truy cập website hàng ngày
Bảng Master
▶ products.csv — Danh mục sản phẩm
Cột Kiểu Mô tả
product_id int Khoá chính
product_name str Tên sản phẩm
category str Danh mục sản phẩm
segment str Phân khúc thị trường của sản phẩm
size str Kích cỡ sản phẩm
color str Nhãn màu sản phẩm
price float Giá bán lẻ
cogs float Giá vốn hàng bán

Ràng buộc: cogs < price với mọi sản phẩm.

2

DATATHON 2026 — The Gridbreakers Được tổ chức bởi VinTelligence — VinUni DS&AI Club

▶ customers.csv — Khách hàng
Cột Kiểu Mô tả
customer_id int Khoá chính
zip int Mã bưu chính (FK → geography.zip)
city str Tên thành phố của khách hàng
signup_date date Ngày đăng ký tài khoản
gender str Giới tính khách hàng (nullable)
age_group str Nhóm tuổi khách hàng (nullable)
acquisition_channel str Kênh tiếp thị khách hàng đăng ký qua (nullable)
▶ promotions.csv — Chương trình khuyến mãi
Cột Kiểu Mô tả
promo_id str Khoá chính
promo_name str Tên chiến dịch kèm năm
promo_type str Loại giảm giá: theo phần trăm hoặc số tiền cố định
discount_value float Giá trị giảm (phần trăm hoặc số tiền tùy promo_type)
start_date date Ngày bắt đầu chiến dịch
end_date date Ngày kết thúc chiến dịch
applicable_category str Danh mục áp dụng, null nếu áp dụng tất cả
promo_channel str Kênh phân phối áp dụng khuyến mãi (nullable)
stackable_flag int Cờ cho phép áp dụng đồng thời nhiều khuyến mãi
min_order_value float Giá trị đơn hàng tối thiểu để áp dụng khuyến mãi (nullable)
Công thức giảm giá:
• percentage: discount_amount = quantity × unit_price × (discount_value/100)
• fixed: discount_amount = quantity × discount_value
▶ geography.csv — Địa lý

Cột Kiểu Mô tả
zip int Khoá chính (mã bưu chính)
city str Tên thành phố
region str Vùng địa lý
district str Tên quận/huyện

Bảng Transaction
▶ orders.csv — Đơn hàng

Cột Kiểu Mô tả
order_id int Khoá chính
order_date date Ngày đặt hàng
customer_id int FK → customers.customer_id
zip int Mã bưu chính giao hàng (FK → geography.zip)
order_status str Trạng thái xử lý của đơn hàng
payment_method str Phương thức thanh toán được sử dụng
device_type str Thiết bị khách hàng dùng khi đặt hàng
order_source str Kênh marketing dẫn đến đơn hàng

3

DATATHON 2026 — The Gridbreakers Được tổ chức bởi VinTelligence — VinUni DS&AI Club

▶ order_items.csv — Chi tiết đơn hàng
Cột Kiểu Mô tả
order_id int FK → orders.order_id
product_id int FK → products.product_id
quantity int Số lượng sản phẩm đặt mua
unit_price float Đơn giá
discount_amount float Tổng số tiền giảm giá cho dòng sản phẩm này
promo_id str FK → promotions.promo_id (nullable)
promo_id_2 str FK → promotions.promo_id, khuyến mãi thứ hai (nullable)
▶ payments.csv — Thanh toán

Cột Kiểu Mô tả
order_id int FK → orders.order_id (quan hệ 1:1)
payment_method str Phương thức thanh toán
payment_value float Tổng giá trị thanh toán của đơn hàng
installments int Số kỳ trả góp

▶ shipments.csv — Vận chuyển

Cột Kiểu Mô tả
order_id int FK → orders.order_id
ship_date date Ngày gửi hàng
delivery_date date Ngày giao hàng đến tay khách
shipping_fee float Phí vận chuyển (0 nếu đơn được miễn phí)
Chỉ tồn tại cho đơn hàng có trạng thái shipped, delivered hoặc returned.
▶ returns.csv — Trả hàng

Cột Kiểu Mô tả
return_id str Khoá chính
order_id int FK → orders.order_id
product_id int FK → products.product_id
return_date date Ngày khách gửi trả hàng
return_reason str Lý do trả hàng
return_quantity int Số lượng sản phẩm trả lại
refund_amount float Số tiền hoàn lại cho khách

▶ reviews.csv — Đánh giá

Cột Kiểu Mô tả
review_id str Khoá chính
order_id int FK → orders.order_id
product_id int FK → products.product_id
customer_id int FK → customers.customer_id
review_date date Ngày khách gửi đánh giá
rating int Điểm đánh giá từ 1 đến 5
review_title str Tiêu đề đánh giá của khách hàng

4

DATATHON 2026 — The Gridbreakers Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Bảng Analytical
▶ sales.csv — Dữ liệu doanh thu

Cột Kiểu Mô tả
Date date Ngày đặt hàng
Revenue float Tổng doanh thu thuần
COGS float Tổng giá vốn hàng bán

Split File Khoảng thời gian
Train sales.csv 04/07/2012 – 31/12/2022
Test sales_test.csv 01/01/2023 – 01/07/2024

Lưu ý: Tập test sẽ không được công bố mà được dùng để đánh giá kết quả mô hình trên
Kaggle. Cấu trúc của file test sẽ giống với sample_submission.csv

Bảng Operational
▶ inventory.csv — Tồn kho

Cột Kiểu Mô tả
snapshot_date date Ngày chụp ảnh tồn kho (cuối tháng)
product_id int FK → products.product_id
stock_on_hand int Số lượng tồn kho cuối tháng
units_received int Số lượng nhập kho trong tháng
units_sold int Số lượng bán ra trong tháng
stockout_days int Số ngày hết hàng trong tháng
days_of_supply float Số ngày tồn kho có thể đáp ứng nhu cầu bán
fill_rate float Tỷ lệ đơn hàng được đáp ứng đủ từ tồn kho
stockout_flag int Cờ báo tháng có xảy ra hết hàng
overstock_flag int Cờ báo tồn kho vượt mức cần thiết
reorder_flag int Cờ báo cần tái đặt hàng sớm
sell_through_rate float Tỷ lệ hàng đã bán so với tổng hàng sẵn có
product_name str Tên sản phẩm
category str Danh mục sản phẩm
segment str Phân khúc sản phẩm
year int Năm trích từ snapshot_date
month int Tháng trích từ snapshot_date

▶ web_traffic.csv — Lưu lượng truy cập
Cột Kiểu Mô tả
date date Ngày ghi nhận lưu lượng
sessions int Tổng số phiên truy cập trong ngày
unique_visitors int Số lượt khách truy cập duy nhất
page_views int Tổng số lượt xem trang
bounce_rate float Tỷ lệ phiên chỉ xem một trang rồi thoát
avg_session_duration_sec float Thời gian trung bình mỗi phiên (giây)
traffic_source str Kênh nguồn dẫn traffic về website

5

DATATHON 2026 — The Gridbreakers Được tổ chức bởi VinTelligence — VinUni DS&AI Club

Quan hệ giữa các bảng

Table 2: Quy tắc quan hệ (Cardinality)

Quan hệ Cardinality
orders ↔ payments 1 : 1
orders ↔ shipments 1 : 0 hoặc 1 (trạng thái shipped/delivered/returned)
orders ↔ returns 1 : 0 hoặc nhiều (trạng thái returned)
orders ↔ reviews 1 : 0 hoặc nhiều (trạng thái delivered, ∼20%)
order_items ↔ promotions nhiều : 0 hoặc 1
products ↔ inventory 1 : nhiều (1 dòng/sản phẩm/tháng)