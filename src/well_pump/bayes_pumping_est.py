import pandas as pd
import numpy as np
from pathlib import Path

# ---- Paths (adjust if needed) ----
WELLS_PATH = Path("/mnt/data/wells_w_efet.csv")
FIELDS_PATH = Path("/mnt/data/fields_by_groupnumb.csv")
OUT_PATH = Path(
    "/mnt/data/per_well_pumping_estimates_fields_based_2024_2025_FIXED_capped.csv"
)

# ---- Configuration ----
YEARS = [2024, 2025]
CANAL_DAYS_BY_YEAR = {2024: 170, 2025: 50}  # your assumption
GPM_PER_HP_DEFAULT = 8.0  # fallback if GPM missing
TOTALIZER_MIN_AF = 1.0  # treat Totalizer as annual anchor if in this range
TOTALIZER_MAX_AF = 5000.0


def norm(s):
    return s.strip() if isinstance(s, str) else s


def derive_k_diam(df, diam_col_name):
    tmp = df[[diam_col_name, "GPM"]].copy()
    tmp["GPM"] = pd.to_numeric(tmp["GPM"], errors="coerce")
    tmp[diam_col_name] = pd.to_numeric(tmp[diam_col_name], errors="coerce")
    good = tmp.dropna()
    good = good[(good["GPM"] > 0) & (good[diam_col_name] > 0)]
    if len(good) >= 5:
        return np.median(good["GPM"].values / (good[diam_col_name].values ** 2))
    return None


def compute_capacity_gpm(row, diam_col=None, k_diam=None):
    gpm = pd.to_numeric(row.get("GPM"), errors="coerce")
    if pd.notna(gpm) and gpm > 0:
        return float(gpm)
    hp = pd.to_numeric(row.get("Horsepower"), errors="coerce")
    if pd.notna(hp) and hp > 0:
        return float(hp) * GPM_PER_HP_DEFAULT
    if diam_col and (k_diam is not None):
        d = pd.to_numeric(row.get(diam_col), errors="coerce")
        if pd.notna(d) and d > 0:
            return float(k_diam) * float(d) ** 2
    return np.nan


# ---- Field-side: compute GW portion per group & year (Mix logic) ----
def gw_fraction(row, year):
    ws = str(row.get("WaterSourc", "")).strip().lower()
    wd = pd.to_numeric(row.get("watering_days"), errors="coerce")
    if ws == "gw":
        return 1.0
    if ws == "mix" and pd.notna(wd) and wd > 0:
        return max(0.0, (wd - CANAL_DAYS_BY_YEAR.get(year, 0)) / float(wd))
    return 1.0


# ---- Allocate group AF to wells by capacity, with optional Totalizer anchoring ----
def allocate_group_to_wells(group_slice: pd.DataFrame, group_total_af: float):
    if not np.isfinite(group_total_af) or group_total_af <= 0 or group_slice.empty:
        return pd.Series(0.0, index=group_slice.index)

    caps = pd.to_numeric(group_slice["alloc_capacity_gpm"], errors="coerce")
    if caps.isna().all() or (caps <= 0).all():
        weights = pd.Series(1.0, index=group_slice.index)
    else:
        weights = caps.fillna(0.0).clip(lower=0.0)
        if weights.sum() == 0:
            weights = pd.Series(1.0, index=group_slice.index)
    weights = weights / weights.sum()
    alloc = weights * group_total_af

    # Totalizer anchors treated as annual AF if plausible
    tot = pd.to_numeric(group_slice.get("Totalizer"), errors="coerce")
    is_anchor = (tot >= TOTALIZER_MIN_AF) & (tot <= TOTALIZER_MAX_AF)
    if is_anchor.any():
        anchors = tot[is_anchor]
        anchor_sum = anchors.sum()
        scale = min(1.0, group_total_af / anchor_sum) if anchor_sum > 0 else 1.0
        anchored_values = anchors * scale

        # Apply anchored values by position
        for idx, val in anchored_values.items():
            alloc.iloc[idx] = val

        remaining = group_total_af - anchored_values.sum()
        non_anchor_idx = [
            i for i in range(len(alloc)) if i not in anchors.index.tolist()
        ]
        if remaining > 0 and len(non_anchor_idx) > 0:
            w_non = weights.iloc[non_anchor_idx]
            w_non = w_non / w_non.sum()
            alloc.iloc[non_anchor_idx] = w_non * remaining
        else:
            alloc.iloc[non_anchor_idx] = 0.0

    return alloc
