# Re-implement using pure shapely + rasterio (no geopandas) to avoid array-interface issues
import os, re, json, glob, math, warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes, rasterize
from shapely.geometry import shape as shp_shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class Params:
    red_band: int = 6
    nir_band: int = 8
    scale: float = 10000.0
    ndvi_thresh: float = 0.2
    min_pixels: int = 3000
    min_circularity: float = 0.55
    min_radius_m: float = 150.0
    max_radius_m: float = 700.0


P = Params()


def list_planet_images(folder="/mnt/data"):
    tifs = sorted(glob.glob(os.path.join(folder, "*composite.tif")))
    if not tifs:
        tifs = sorted(glob.glob(os.path.join(folder, "*.tif")))
    return tifs


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


def read_bands(path, bands=(P.red_band, P.nir_band)):
    with rasterio.open(path) as ds:
        data = ds.read(bands).astype("float32")
        prof = ds.profile
        transform = ds.transform
        crs = ds.crs
    red, nir = data[0] / P.scale, data[1] / P.scale
    return red, nir, prof, transform, crs


def compute_ndvi(red, nir):
    den = red + nir
    nd = np.where(den > 0, (nir - red) / den, np.nan)
    return np.clip(nd, -1, 1)


def connected_polygons_from_mask(mask, transform):
    polys = []
    for geom, val in shapes(
        mask.astype("uint8"), mask=mask.astype("uint8"), transform=transform
    ):
        if val == 1:
            polys.append(shp_shape(geom))
    if not polys:
        return []
    dissolved = unary_union(polys)
    if isinstance(dissolved, Polygon):
        return [dissolved]
    elif isinstance(dissolved, MultiPolygon):
        return list(dissolved.geoms)
    return []


def pixel_size_m(transform):
    return abs(transform.a)


def polygon_metrics(poly):
    A = poly.area
    Pp = poly.length
    if Pp == 0:
        return 0.0, 0.0
    circ = 4 * np.pi * A / (Pp * Pp)
    r_equiv = np.sqrt(A / np.pi)
    return float(circ), float(r_equiv)


def raster_mask_from_ndvi(ndvi, thresh, min_pixels):
    from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, label

    m = (ndvi >= thresh) & np.isfinite(ndvi)
    m = binary_opening(m, structure=np.ones((3, 3)))
    m = binary_closing(m, structure=np.ones((5, 5)))
    m = binary_fill_holes(m)
    lab, n = label(m)
    if n > 0:
        cnt = np.bincount(lab.ravel())
        small = np.isin(lab, np.where(cnt < min_pixels)[0])
        m[small] = False
    return m


def detect_pivots(first_path):
    red, nir, prof, transform, crs = read_bands(first_path)
    ndvi = compute_ndvi(red, nir)
    mask = raster_mask_from_ndvi(ndvi, P.ndvi_thresh, P.min_pixels)
    polys = connected_polygons_from_mask(mask, transform)
    if not polys:
        return [], transform, crs
    px_m = pixel_size_m(transform)
    keep = []
    for poly in polys:
        c, rpx = polygon_metrics(poly)
        rm = rpx * px_m
        if (c >= P.min_circularity) and (P.min_radius_m <= rm <= P.max_radius_m):
            keep.append(poly)
    return keep, transform, crs


def save_polygons_geojson(polys, crs, path="/mnt/data/pivot_fields.geojson"):
    # Minimal GeoJSON FeatureCollection
    feats = []
    for i, poly in enumerate(polys, start=1):
        feats.append(
            {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {"pivot_id": i},
            }
        )
    fc = {
        "type": "FeatureCollection",
        "features": feats,
        "crs": {"type": "name", "properties": {"name": str(crs)}},
    }
    with open(path, "w") as f:
        json.dump(fc, f)
    return path


def zonal_ndvi(image_path, polys):
    red, nir, prof, transform, crs = read_bands(image_path)
    ndvi = compute_ndvi(red, nir)
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
