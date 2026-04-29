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
├── Notebooks/
│ ├── datathon26-mcqs.ipynb # Giải các câu MCQ phân tích dữ liệu
│ ├── example_baseline.ipynb # Baseline đơn giản (seasonal average + trend)
│ └── datathon26-model.ipynb # Pipeline chính: stacking XGB + LGB + CatBoost + Prophet
├── Submission/
│ └── submission.csv
└── README.md
```
---

## Phương pháp

Mô hình chính sử dụng **stacking ensemble** gồm:

- **XGBoost** — `reg:pseudohubererror`, depth 4, 2000 estimators
- **LightGBM** — gradient boosting trên feature bảng
- **CatBoost** — xử lý tốt categorical features
- **Prophet** — bắt trend + seasonality theo tuần/năm

Meta-learner: **HuberRegressor** để robust với outlier.

### Feature engineering 

- **Calendar features:** year, month, day, dayofweek, month_sin, month_cos
- **Sale season features:** 6 đợt sale mỗi năm (tháng 1, 3, 6, 7, 8, 11), `is_sale_season`, `sale_rank`, `day_to_next_sale`, `day_since_last_sale`,  
- **Peak season features:** 3 peak mỗi năm (tháng 4, 5, 6), `is_peak_season`, `peak_proximity`

---

## Sử dụng

Tạo và import notebook trong cuộc thi trên Kaggle, sau đó chạy notebook. Riêng với datathon26-model.ipynb cần chọn Kaggle GPU để tăng tốc quá trình huấn luyện

| Notebook | Mục đích |
|---|---|
| `datathon26-mcqs.ipynb` | Phân tích và trả lời các câu hỏi MCQ |
| `example_baseline.ipynb` | Baseline đơn giản để tham khảo |
| `datathon26-model.ipynb` | Pipeline huấn luyện và sinh submission.csv |


---

## Định dạng submission

```csv
date,revenue,cogs
2023-01-01,12500000,7800000
2023-01-02,13100000,8100000
```

---
