# Datathon 2026 — Dự báo Doanh thu & COGS cho E-Commerce

**Cuộc thi:** Datathon 2026  
**Mục tiêu:** Dự báo doanh thu và COGS hàng ngày, giai đoạn 01/01/2023 – 01/07/2024  
**Metric đánh giá:** MAE, RMSE, R²
**Đội thi:** Optimal
**Thành viên:** Đào Thế Việt, Phạm Mai Linh

---

## Cấu trúc thư mục

```
datathon26/
├── data-round-1/
│   ├── products.csv
│   ├── customers.csv
│   ├── promotions.csv
│   ├── geography.csv
│   ├── orders.csv
│   ├── order_items.csv
│   ├── payments.csv
│   ├── shipments.csv
│   ├── returns.csv
│   ├── reviews.csv
│   ├── sales.csv
│   ├── inventory.csv
│   └── web_traffic.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_shap_interpretation.ipynb
│
├── submission.csv
├── DATATHON26_MODELS.xlsx
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
git clone https://github.com/MrPickle200/DATATHON2026
cd datathon26

pip install -r requirements.txt
```

Các thư viện chính: `prophet`, `lightgbm`, `xgboost`, `catboost`, `scikit-learn`, `shap`, `pandas`, `numpy`

---

## Sử dụng

Chạy các notebook theo thứ tự:

```bash
jupyter notebook
```

| Notebook | Mục đích |
|---|---|
| `01_eda.ipynb` | Phân tích khám phá dữ liệu |
| `02_feature_engineering.ipynb` | Tạo feature, xử lý lag/rolling |
| `03_modeling.ipynb` | Huấn luyện model (Prophet + Stacking) |
| `04_evaluation.ipynb` | Đánh giá, backtesting, xuất `submission.csv` |
| `05_shap_interpretation.ipynb` | SHAP — phân tích feature importance |

> **Lưu ý:** Đặt toàn bộ file CSV vào đúng thư mục `data-round-1/` trước khi chạy. Các notebook phụ thuộc nhau theo thứ tự.

---

## Định dạng submission

```csv
date,revenue,cogs
2023-01-01,12500000,7800000
2023-01-02,13100000,8100000
```

---
