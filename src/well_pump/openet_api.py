import geopandas as gpd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import orient


def geojson_to_lonlat_lists(
    path: str,
    include_holes: bool = False,
    enforce_ccw: bool = False,
    feature_index: int | None = None,
    *,
    simplify_tolerance: float | None = None,
    simplify_units: str = "meters",  # "meters" or "degrees"
    simplify_preserve_topology: bool = True,
):
    """
    Read a GeoJSON file and return flattened [lon, lat, lon, lat, ...] lists
    for each polygon ring (exteriors by default; holes optional).

    Parameters
    ----------
    path : str
        Path to a GeoJSON file.
    include_holes : bool
        If True, include interior rings (holes).
    enforce_ccw : bool
        If True, force exterior rings to be counterclockwise.
    feature_index : int | None
        If an int, process only that feature. If None, process all features.
    simplify_tolerance : float | None
        If provided, simplify geometry with Shapely .simplify().
        Units controlled by `simplify_units`.
    simplify_units : {"meters", "degrees"}
        Unit for `simplify_tolerance`. If "meters" (recommended), geometry is
        projected to EPSG:3857 for simplification, then returned to WGS84.
        If "degrees", simplification is done directly on WGS84 coords.
    simplify_preserve_topology : bool
        Passed to Shapely .simplify(preserve_topology=...).

    Returns
    -------
    list
        A single flat list if exactly one ring was produced;
        otherwise a list of flat lists (one per ring).
    """
    gdf = gpd.read_file(path)
    if gdf.empty:
        return []

    # Assume GeoJSON is WGS84 if undefined
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    # Limit to one feature if requested
    if feature_index is not None:
        gdf = gdf.iloc[[feature_index]]

    # Optional simplification
    if simplify_tolerance is not None:
        if simplify_units.lower() == "meters":
            gdf_metric = gdf.to_crs(3857)
            gdf_metric["geometry"] = gdf_metric.geometry.apply(
                lambda geom: (
                    geom.simplify(
                        simplify_tolerance, preserve_topology=simplify_preserve_topology
                    )
                    if geom and not geom.is_empty
                    else geom
                )
            )
            gdf = gdf_metric.to_crs(4326)
        elif simplify_units.lower() == "degrees":
            gdf_4326 = gdf.to_crs(4326)
            gdf_4326["geometry"] = gdf_4326.geometry.apply(
                lambda geom: (
                    geom.simplify(
                        simplify_tolerance, preserve_topology=simplify_preserve_topology
                    )
                    if geom and not geom.is_empty
                    else geom
                )
            )
            gdf = gdf_4326
        else:
            raise ValueError("simplify_units must be 'meters' or 'degrees'.")

    def ring_to_flat_list(ring):
        x, y = ring.xy
        coords = list(zip(x, y))[:-1]  # drop duplicated closing vertex
        out = []
        for lon, lat in coords:
            out.extend([lon, lat])
        return out

    out_lists = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue

        def handle_polygon(poly: Polygon):
            p = orient(poly, 1.0) if enforce_ccw else poly
            out_lists.append(ring_to_flat_list(p.exterior))
            if include_holes:
                for hole in p.interiors:
                    out_lists.append(ring_to_flat_list(hole))

        if isinstance(geom, Polygon):
            handle_polygon(geom)
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                if p and not p.is_empty:
                    handle_polygon(p)
        else:
            # Ignore non-polygonal features
            continue

    return out_lists[0] if len(out_lists) == 1 else out_lists
