# Datathon 2026 — Dự báo Doanh thu & COGS cho E-Commerce

**Cuộc thi:** Datathon 2026  
**Mục tiêu:** Dự báo doanh thu và COGS hàng ngày, giai đoạn 01/01/2023 – 01/07/2024  
**Metric đánh giá:** MAE, RMSE, R²
**Đội thi:** Optimal
**Thành viên:** Đào Thế Việt, Phạm Mai Linh

---

## Cấu trúc thư mục

```
DATATHON2026/
├── data-round-1/
│ ├── customers.csv
│ ├── geography.csv
│ ├── inventory.csv
│ ├── order_items.csv
│ ├── orders.csv
│ ├── payments.csv
│ ├── products.csv
│ ├── promotions.csv
│ ├── returns.csv
│ ├── reviews.csv
│ ├── sales.csv
│ ├── sample_submission.csv
│ ├── shipments.csv
│ └── web_traffic.csv
│
├── Notebooks/
│ ├── datathon26-mcqs.ipynb # Giải các câu MCQ phân tích dữ liệu
│ ├── example_baseline.ipynb # Baseline đơn giản (seasonal average + trend)
│ ├── model.ipynb # Pipeline chính: stacking XGB + LGB + CatBoost + Prophet
│ └── helper.py # Feature engineering (calendar, Tết, sale seasons, HTE)
│
├── Submission/
│ └── submission.csv
│
├── requirements.txt
└── README.md
```
---

## Phương pháp

Mô hình chính (`model.ipynb`) sử dụng **stacking ensemble** gồm:

- **XGBoost** — `reg:pseudohubererror`, depth 4, 2000 estimators
- **LightGBM** — gradient boosting trên feature bảng
- **CatBoost** — xử lý tốt categorical features
- **Prophet** — bắt trend + seasonality theo tuần/năm

Meta-learner: **HuberRegressor** để robust với outlier.

### Feature engineering (`helper.py`)

- **Calendar features:** year, month, day, dayofweek
- **Tết features:** `days_to_next_tet`, `days_since_last_tet`, exponential decay effects
- **Sale season features:** 6 đợt sale mỗi năm (tháng 1, 3, 6, 7, 8, 11), `is_sale_season`, `sale_rank`, `sale_progress`, pre/post-sale decay
- **Historical Target Encoding (HTE):** encode trung bình target theo nhóm tháng/ngày

---

## Cài đặt

```bash
git clone https://github.com/MrPickle200/DATATHON2026
cd DATATHON2026
pip install -r requirements.txt
```

Các thư viện chính: `prophet`, `lightgbm`, `xgboost`, `catboost`, `scikit-learn`, `shap`, `pandas`, `numpy`

### Đăng ký Jupyter Kernel

Sau khi cài xong requirements, cần đăng ký kernel để notebook nhận đúng môi trường:

```bash
pip install ipykernel
python -m ipykernel install --user --name datathon --display-name "Python (datathon)"
```

Sau đó mở notebook và chọn kernel **"Python (datathon)"** ở góc trên bên phải (VSCode) hoặc menu **Kernel → Change Kernel** (Jupyter).

> **Windows:** Chạy file `setup.bat` để tự động hóa toàn bộ bước cài đặt.  
> **Mac/Linux:** Chạy `bash setup.sh`.

---

## Sử dụng

Chạy các notebook theo thứ tự:

```bash
jupyter notebook
```

| Notebook | Mục đích |
|---|---|
| `datathon26-mcqs.ipynb` | Phân tích và trả lời các câu hỏi MCQ |
| `example_baseline.ipynb` | Baseline đơn giản để tham khảo |
| `model.ipynb` | Pipeline huấn luyện và sinh submission.csv |

> **Lưu ý:**  Đặt toàn bộ file CSV vào đúng thư mục data-round-1/ trước khi chạy. model.ipynb import helper.py cùng thư mục — chạy notebook từ bên trong Notebooks/.

---

## Định dạng submission

```csv
date,revenue,cogs
2023-01-01,12500000,7800000
2023-01-02,13100000,8100000
```

---
