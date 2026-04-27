import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. DETERMINISTIC FEATURE ENGINEERING
# ---------------------------------------------------------
TET_DATES = pd.to_datetime([
    '2012-01-23', '2013-02-10', '2014-01-31', '2015-02-19', '2016-02-08',
    '2017-01-28', '2018-02-16', '2019-02-05', '2020-01-25', '2021-02-12',
    '2022-02-01', '2023-01-22', '2024-02-10', '2025-01-29'
])

# ──────────────────────────────────────────────
# SALE SEASON CONFIG
# profit_rank: Jan=1 (highest), Mar=2, Jun=3, Aug=4, Jul=5, Nov=6
# ──────────────────────────────────────────────
SALE_SEASONS = [
    {'month': 1,  'start_day': 30, 'duration': 30, 'profit_rank': 1},
    {'month': 3,  'start_day': 18, 'duration': 30, 'profit_rank': 2},
    {'month': 6,  'start_day': 23, 'duration': 29, 'profit_rank': 3},
    {'month': 7,  'start_day': 30, 'duration': 34, 'profit_rank': 5},
    {'month': 8,  'start_day': 30, 'duration': 32, 'profit_rank': 4},
    {'month': 11, 'start_day': 18, 'duration': 45, 'profit_rank': 6},
]

PRE_SALE_DECAY  = 7.0   # days: người mua biết trước ~1 tuần
POST_SALE_DECAY = 5.0   # days: demand hồi phục sau ~5 ngày


def _get_sale_windows(years):
    """
    Sinh ra list các (start_date, end_date, profit_rank) cho từng đợt sale,
    theo từng năm trong `years`.
    Xử lý trường hợp end_date tràn sang tháng sau (ví dụ tháng 7 + 34 ngày).
    """
    windows = []
    for year in years:
        for s in SALE_SEASONS:
            try:
                start = pd.Timestamp(year=year, month=s['month'], day=s['start_day'])
            except ValueError:
                # start_day vượt quá số ngày trong tháng → dùng ngày cuối tháng
                start = pd.Timestamp(year=year, month=s['month'], day=1) + pd.offsets.MonthEnd(0)
            end = start + pd.Timedelta(days=s['duration'] - 1)
            windows.append((start, end, s['profit_rank']))
    return windows


def add_deterministic_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # ── Basic calendar ──────────────────────────────────────────────────────
    df['year']      = df['Date'].dt.year
    df['month']     = df['Date'].dt.month
    df['day']       = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek

    # ── Tet features (giữ nguyên logic cũ, vector hoá cho nhanh) ───────────
    df['days_to_next_tet']   = df['Date'].apply(
        lambda d: (TET_DATES[TET_DATES >= d].min() - d).days
                  if (TET_DATES >= d).any() else 999
    )
    df['days_since_last_tet'] = df['Date'].apply(
        lambda d: (d - TET_DATES[TET_DATES <= d].max()).days
                  if (TET_DATES <= d).any() else 999
    )
    df['tet_pre_effect']  = np.exp(-df['days_to_next_tet']   / 10.0)
    df['tet_post_effect'] = np.exp(-df['days_since_last_tet'] / 5.0)

    # ── Sale season windows ─────────────────────────────────────────────────
    years   = df['Date'].dt.year.unique()
    windows = _get_sale_windows(years)

    # Khởi tạo
    df['is_sale_season']      = 0
    df['sale_rank']           = 0      # 0 = không sale
    df['sale_progress']       = 0.0   # 0→1 trong đợt sale
    df['days_to_next_sale']   = 999
    df['days_since_last_sale']= 999

    dates = df['Date'].values  # numpy array để vectorise

    for start, end, rank in windows:
        mask_in = (df['Date'] >= start) & (df['Date'] <= end)

        # is_sale_season & sale_rank
        df.loc[mask_in, 'is_sale_season'] = 1
        df.loc[mask_in, 'sale_rank']      = rank

        # sale_progress: 0.0 (ngày đầu) → 1.0 (ngày cuối)
        duration_days = (end - start).days
        if duration_days > 0:
            df.loc[mask_in, 'sale_progress'] = (
                (df.loc[mask_in, 'Date'] - start).dt.days / duration_days
            )

        # days_to_next_sale: cập nhật cho các ngày TRƯỚC đợt sale này
        mask_before = df['Date'] < start
        days_to = (start - df.loc[mask_before, 'Date']).dt.days
        df.loc[mask_before, 'days_to_next_sale'] = np.minimum(
            df.loc[mask_before, 'days_to_next_sale'], days_to
        )

        # days_since_last_sale: cập nhật cho các ngày SAU đợt sale này
        mask_after = df['Date'] > end
        days_since = (df.loc[mask_after, 'Date'] - end).dt.days
        df.loc[mask_after, 'days_since_last_sale'] = np.minimum(
            df.loc[mask_after, 'days_since_last_sale'], days_since
        )

    # Với ngày đang trong sale: days_to = 0, days_since = 0
    df.loc[df['is_sale_season'] == 1, 'days_to_next_sale']    = 0
    df.loc[df['is_sale_season'] == 1, 'days_since_last_sale'] = 0

    # Exponential decay (pre / post sale)
    df['sale_pre_effect']  = np.where(
        df['is_sale_season'] == 0,
        np.exp(-df['days_to_next_sale']    / PRE_SALE_DECAY),
        0.0   # trong sale thì pre-effect = 0 (đã vào rồi)
    )
    df['sale_post_effect'] = np.where(
        df['is_sale_season'] == 0,
        np.exp(-df['days_since_last_sale'] / POST_SALE_DECAY),
        0.0   # trong sale thì post-effect = 0
    )

    return df

# ---------------------------------------------------------
# 2. HISTORICAL TARGET ENCODING (HTE)
# ---------------------------------------------------------
def apply_hte(train_df, target_df, group_cols, target_col):
    mapping = train_df.groupby(group_cols)[target_col].mean().reset_index()
    mapping.rename(columns={target_col: f'hte_{target_col}'}, inplace=True)
    res = target_df.merge(mapping, on=group_cols, how='left')
    
    if res[f'hte_{target_col}'].isnull().any():
        fallback = train_df.groupby(['month'])[target_col].mean().reset_index()
        fallback.rename(columns={target_col: f'fallback_{target_col}'}, inplace=True)
        res = res.merge(fallback, on=['month'], how='left')
        res[f'hte_{target_col}'] = res[f'hte_{target_col}'].fillna(res[f'fallback_{target_col}'])
        res.drop(columns=[f'fallback_{target_col}'], inplace=True)
        
    res[f'hte_{target_col}'] = res[f'hte_{target_col}'].fillna(train_df[target_col].mean())
    return res[f'hte_{target_col}'].values