# Re-implement using pure shapely + rasterio (no geopandas) to avoid array-interface issues
import os, re, json, warnings
import pathlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, label
import xml.etree.ElementTree

warnings.filterwarnings("ignore", category=UserWarning)


def _coeffs_from_xml(xml_path: pathlib.Path) -> dict[int, float]:
    """Return {band_index(1-based): reflectanceCoefficient} from a Planet XML."""
    coeffs = {}
    try:
        root = xml.etree.ElementTree.parse(str(xml_path)).getroot()
        for node in root.findall(".//{*}bandSpecificMetadata"):
            b = node.find(".//{*}bandNumber")
            c = node.find(".//{*}reflectanceCoefficient")
            if b is not None and c is not None:
                try:
                    coeffs[int(b.text.strip())] = float(c.text.strip())
                except Exception:
                    pass
    except Exception:
        pass
    return coeffs


def scale_with_xml(
    masked_raster: np.ndarray,
    xml_path: str | pathlib.Path,
    band_index: int | None = None,
    out_tif: str | pathlib.Path | None = None,
    template_tif: str | pathlib.Path | None = None,
) -> np.ndarray:  # type: ignore
    """Scale a masked raster using Planet reflectance coefficients from XML.

    Parameters
    ----------
    masked_raster : np.ndarray
        Your raster as a NumPy array. NaNs are preserved. Can be 2D (single-band)
        or 3D (bands, rows, cols). If 3D, scaling is applied per-band when possible.
    xml_path : str or Path
        Path to the Planet metadata XML containing <reflectanceCoefficient> per band.
    band_index : int, optional
        1-based band index whose coefficient should be used. Required if masked_raster is 2D
        and you want a specific band's coefficient. If masked_raster is 3D and band_index is None,
        the function attempts to scale each band i with coefficient (i+1).
    out_tif : str or Path, optional
        If given, write a GeoTIFF of the scaled raster (requires template_tif for geo metadata).
    template_tif : str or Path, optional
        A GeoTIFF to copy georeferencing/transform/profile from when writing out_tif.

    Returns
    -------
    np.ndarray
        Scaled raster (float32), preserving NaNs.
    """
    coeffs = _coeffs_from_xml(pathlib.Path(xml_path))
    arr = masked_raster.astype("float32", copy=True)

    if arr.ndim == 2:
        if band_index is None:
            raise ValueError(
                "For a single-band array, provide band_index (1-based) to select a coefficient."
            )
        coeff = coeffs.get(int(band_index))
        if coeff is None:
            scaled = arr
        else:
            finite = np.isfinite(arr)
            scaled = np.full_like(arr, np.nan, dtype="float32")
            scaled[finite] = arr[finite] * float(coeff)
    elif arr.ndim == 3:
        scaled = np.full_like(arr, np.nan, dtype="float32")
        bands = arr.shape[0]
        for i in range(bands):
            c = coeffs.get(i + 1)
            if c is None:
                scaled[i] = arr[i]
            else:
                finite = np.isfinite(arr[i])
                scaled[i, finite] = arr[i, finite] * float(c)
    else:
        raise ValueError("masked_raster must be 2D or 3D.")

    if out_tif is not None:
        if template_tif is None:
            raise ValueError("template_tif is required when writing an output GeoTIFF.")
        with rasterio.open(template_tif) as tmpl:
            profile = tmpl.profile.copy()
            if scaled.ndim == 2:
                profile.update(count=1, dtype="float32", nodata=np.nan)
                with rasterio.open(out_tif, "w", **profile) as dst:
                    dst.write(scaled, 1)
            else:
                profile.update(count=scaled.shape[0], dtype="float32", nodata=np.nan)
                with rasterio.open(out_tif, "w", **profile) as dst:
                    dst.write(scaled)
    return scaled


def parse_date_from_path(path):
    base = os.path.basename(path)
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", base)
    if m:
        return pd.to_datetime(m.group(1)).date()
    js = re.sub(r"\.tif$", "_metadata.json", path)
    if os.path.exists(js):
        try:
            meta = json.load(open(js))
            acq = meta.get("properties", {}).get("acquired")
            if acq:
                return pd.to_datetime(acq).date()
        except Exception:
            pass
    return None


def get_ndvi(src, mask=None, xml=None):
    red = src.read(6).astype("float32")
    nir = src.read(8).astype("float32")
    if mask is not None:
        red = np.where(mask, red, np.nan)
        nir = np.where(mask, nir, np.nan)
    if xml is not None:
        red = scale_with_xml(red, xml_path=xml, band_index=6)
        nir = scale_with_xml(nir, xml_path=xml, band_index=8)
    den = nir + red
    ndvi = np.full_like(nir, np.nan, dtype="float32")
    valid = den != 0
    ndvi[valid] = (nir[valid] - red[valid]) / den[valid]
    return ndvi


def get_transform(src):
    transform = src.transform
    crs = src.crs
    xres = transform.a
    yres = -transform.e if transform.e < 0 else transform.e
    pixel_size = (xres, yres)
    return transform, crs, pixel_size


def read_udm_mask(udm_path: pathlib.Path):
    with rasterio.open(udm_path) as src:
        udm = src.read(1)
        mask = udm == 1
    return mask


def make_ndvi(
    composite_path: pathlib.Path,
    xml_path: pathlib.Path | None = None,
    udm_path: pathlib.Path | None = None,
):
    mask = read_udm_mask(udm_path)
    with rasterio.open(composite_path) as src:
        ndvi = get_ndvi(src, mask=mask, xml=xml_path)
        transform, crs, pixel_size = get_transform(src)
        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan)
        return ndvi, profile, transform, crs, pixel_size


def polygon_metrics(poly):
    A = poly.area
    Pp = poly.length
    if Pp == 0:
        return 0.0, 0.0
    circ = 4 * np.pi * A / (Pp * Pp)
    r_equiv = np.sqrt(A / np.pi)
    return float(circ), float(r_equiv)


def raster_mask_from_ndvi(ndvi, thresh, min_pixels):

    m = (ndvi >= thresh) & np.isfinite(ndvi)
    m = binary_opening(m, structure=np.ones((3, 3)))
    m = binary_closing(m, structure=np.ones((5, 5)))
    m = binary_fill_holes(m)
    lab, n = label(m)  # type: ignore
    if n > 0:
        cnt = np.bincount(lab.ravel())
        small = np.isin(lab, np.where(cnt < min_pixels)[0])
        m[small] = False
    return m


def zonal_ndvi(ndvi, polys, transform):
    ids = list(range(1, len(polys) + 1))
    lid = rasterize(
        shapes=[(mapping(g), pid) for g, pid in zip(polys, ids)],
        out_shape=ndvi.shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )
    rows = []
    for pid in ids:
        m = lid == pid
        if m.any():
            v = ndvi[m]
            rows.append(
                dict(
                    pivot_id=pid,
                    ndvi_mean=float(np.nanmean(v)),
                    ndvi_p50=float(np.nanpercentile(v, 50)),
                    ndvi_p90=float(np.nanpercentile(v, 90)),
                    n_pix=int(np.isfinite(v).sum()),
                )
            )
        else:
            rows.append(
                dict(
                    pivot_id=pid,
                    ndvi_mean=np.nan,
                    ndvi_p50=np.nan,
                    ndvi_p90=np.nan,
                    n_pix=0,
                )
            )
    return pd.DataFrame(rows)


# ---- Helper functions ----
def _regularize_series(s: pd.Series, freq="5D") -> pd.Series:
    """Resample to a regular grid with median aggregation, forward-fill small gaps, and smooth lightly."""
    sr = s.resample(freq).median()
    # Forward/back fill small gaps (<= 2 steps)
    sr = sr.ffill(limit=2).bfill(limit=2)
    # Light smoothing: centered rolling median then mean
    sm = sr.rolling(3, center=True, min_periods=1).median()
    sm = sm.rolling(3, center=True, min_periods=1).mean()
    return sm


def detect_alfalfa_cuts(
    series: pd.Series,
    drop_thresh: float = 0.15,
    pre_min: float = 0.55,
    recovery: float = 0.10,
    min_spacing_days: int = 18,
) -> list:
    """
    Detect likely alfalfa cut dates as sharp drops from a high NDVI followed by recovery.
    Heuristics only—tuned for irrigated alfalfa in semi-arid settings.
    """
    s = _regularize_series(series)
    if s.dropna().empty:
        return []
    cuts = []
    min_spacing = pd.Timedelta(days=min_spacing_days)

    # For each time, compare current value to prior 15 days max; drop >= drop_thresh and prior >= pre_min.
    win = 15  # days, but our frequency is ~5D, so use 3 steps either side
    # Use rolling window based on time deltas rather than fixed count
    # implement with expanding max of last 30 days sampled from s.index
    for i, t in enumerate(s.index):
        val = s.iloc[i]
        if pd.isna(val):
            continue
        # Prior window max over ~20 days
        t0 = t - pd.Timedelta(days=25)
        prior = s.loc[(s.index >= t0) & (s.index < t)]
        if prior.empty:
            continue
        prior_max = prior.max()
        # Drop size
        drop_val = prior_max - val
        if np.isnan(prior_max) or np.isnan(drop_val):
            continue
        if drop_val >= drop_thresh and prior_max >= pre_min:
            # Ensure recovery within next ~25 days
            t1 = t + pd.Timedelta(days=25)
            future = s.loc[(s.index > t) & (s.index <= t1)]
            recov = future.max() - val if not future.empty else 0.0
            if recov >= recovery:
                # Debounce: avoid marking multiple points in the same event; choose local minimum in ±10 days
                start = t - pd.Timedelta(days=10)
                end = t + pd.Timedelta(days=10)
                window = s.loc[(s.index >= start) & (s.index <= end)]
                if not window.empty:
                    local_min_time = window.idxmin()
                    # Enforce spacing
                    if not cuts or (local_min_time - cuts[-1]) >= min_spacing:
                        cuts.append(local_min_time)
    return sorted(list(dict.fromkeys(cuts)))  # unique and sorted


def classify_plot(series: pd.Series) -> dict:
    """
    Classify a single field's NDVI time series into {'alfalfa','corn','fallow','unknown'}.
    Returns features used for transparency.
    """
    s = _regularize_series(series)
    if s.dropna().empty:
        return {
            "type": "unknown",
            "cuts": [],
            "peak": np.nan,
            "median": np.nan,
            "peak_date": pd.NaT,
        }
    cuts = detect_alfalfa_cuts(series)
    peak = float(s.max())
    median = float(s.median())
    peak_date = s.idxmax()

    # Early and late season medians to help distinguish corn ramp
    # early: April–June 15; late: July 1–Sep 30
    early = s.loc[(s.index.month <= 6) & (s.index.day <= 15) | (s.index.month < 6)]
    late = s.loc[(s.index.month >= 7) & (s.index.month <= 9)]
    early_med = float(early.median()) if not early.empty else np.nan
    late_med = float(late.median()) if not late.empty else np.nan

    # Simple rules
    if (peak < 0.5 and median < 0.35) or (np.nan_to_num(peak) < 0.45):
        ptype = "fallow"
    elif len(cuts) >= 2 and peak >= 0.7:
        ptype = "alfalfa"
    elif len(cuts) <= 1 and peak >= 0.7:
        # Corn tends to have low early season and a single mid/late summer peak
        if (
            pd.notna(peak_date)
            and (peak_date.month in (7, 8, 9))
            and (np.isnan(early_med) or early_med < 0.4)
            and (np.isnan(late_med) or late_med >= 0.55)
        ):
            ptype = "corn"
        else:
            ptype = "corn"
    else:
        # Fallbacks
        ptype = "fallow" if median < 0.4 else ("alfalfa" if len(cuts) >= 2 else "corn")
    return {
        "type": ptype,
        "cuts": cuts,
        "peak": peak,
        "median": median,
        "peak_date": peak_date,
        "early_med": early_med,
        "late_med": late_med,
        "num_cuts": len(cuts),
    }


def classify_all(df_multi: pd.DataFrame, ndvi_col: str) -> pd.DataFrame:
    """Apply classification for each field_id; return a tidy dataframe with features."""
    out = []
    for fid, sdf in df_multi.groupby(level=0):
        series = sdf[ndvi_col].droplevel(0)
        feat = classify_plot(series)
        row = {
            "field_id": fid,
            "type": feat["type"],
            "num_cuts": feat["num_cuts"],
            "peak_ndvi": feat["peak"],
            "median_ndvi": feat["median"],
            "peak_date": feat["peak_date"],
            "early_med": feat["early_med"],
            "late_med": feat["late_med"],
            "cut_dates": [feat["cuts"]],
        }
        out.append(row)
    res = pd.DataFrame(out).set_index("field_id").sort_index()
    return res
