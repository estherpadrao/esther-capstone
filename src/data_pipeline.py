"""Integrated data preparation, visualisation and PCI computation pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from itertools import cycle
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import folium
import geopandas as gpd
import pandas as pd
from branca.colormap import linear
from folium import FeatureGroup
from shapely.geometry import Point

from .accessibility import AccessibilityCalculator, MultiModalNetwork

LOGGER = logging.getLogger(__name__)

WGS84 = 4326
SF_UTM = 26910  # UTM zone 10N (NAD83) – suitable for San Francisco


def _load_vector_layer(path: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    """Load a vector dataset and return it in WGS84 coordinates."""

    gdf = gpd.read_file(path, layer=layer)  # type: ignore[arg-type]
    gdf = gdf.dropna(subset=["geometry"]).copy()
    if gdf.crs is None:
        gdf.set_crs(WGS84, inplace=True)
    else:
        gdf.to_crs(WGS84, inplace=True)
    return gdf


def _split_location_column(
    df: pd.DataFrame, *, location_column: str, lat_name: str, lon_name: str
) -> pd.DataFrame:
    """Extract latitude/longitude floats from a column like "CA (lat, lon)"."""

    if location_column not in df.columns:
        raise ValueError(
            f"Column '{location_column}' not found while parsing location coordinates."
        )

    # Extract numeric parts inside parentheses regardless of extra labels or spaces.
    extracted = df[location_column].astype(str).str.extract(
        r"\((?P<lat>[-+]?\d*\.?\d+),\s*(?P<lon>[-+]?\d*\.?\d+)\)"
    )

    if extracted.isna().all(axis=None):
        raise ValueError(
            "Could not parse latitude/longitude values from column "
            f"'{location_column}'. Ensure it looks like 'City (lat, lon)'."
        )

    df = df.copy()
    df[lat_name] = extracted["lat"].astype(float)
    df[lon_name] = extracted["lon"].astype(float)
    return df


def _normalise_column_name(
    df: pd.DataFrame, column: str, *, optional: bool = False
) -> tuple[pd.DataFrame, Optional[str]]:
    """Ensure a column exists using case-insensitive matching.

    Returns a tuple of ``(dataframe, resolved_column_name)`` where the
    dataframe may have been renamed so the resolved column matches the
    requested ``column`` exactly.  If ``optional`` is ``True`` and no match is
    found, ``None`` is returned instead of raising an error.
    """

    # First standardise obvious inconsistencies such as leading/trailing spaces.
    renamed: Dict[str, str] = {}
    for col in df.columns:
        stripped = col.strip()
        if stripped != col and stripped not in df.columns:
            renamed[col] = stripped
    if renamed:
        df = df.rename(columns=renamed)

    if column in df.columns:
        return df, column

    lowered = {col.casefold(): col for col in df.columns}
    match = lowered.get(column.casefold())
    if match is not None:
        df = df.rename(columns={match: column})
        return df, column

    if optional:
        return df, None

    raise ValueError(f"Column '{column}' not found in provided dataset.")


def _load_permits(
    path: Path,
    lat_column: str,
    lon_column: str,
    *,
    location_column: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Load a CSV file with coordinates as a GeoDataFrame.

    Supports either explicit latitude/longitude columns or a single text column
    containing a pattern such as "CA (37.77, -122.42)".  The latter is parsed
    into float columns before creating point geometries.
    """

    df = pd.read_csv(path)

    df, lat_column = _normalise_column_name(df, lat_column)
    df, lon_column = _normalise_column_name(df, lon_column)

    if location_column:
        df, resolved_location = _normalise_column_name(
            df, location_column, optional=True
        )
        # ``optional=True`` returns ``None`` when the column is not present. In
        # that scenario we must clear ``location_column`` so a missing
        # "Location 1" header does not trigger an unnecessary parsing attempt
        # (and subsequent KeyError) when explicit latitude/longitude columns are
        # already available.
        location_column = resolved_location

    if location_column and (lat_column not in df.columns or lon_column not in df.columns):
        df = _split_location_column(
            df,
            location_column=location_column,
            lat_name=lat_column,
            lon_name=lon_column,
        )

    df = df.dropna(subset=[lat_column, lon_column])
    geometry = gpd.points_from_xy(df[lon_column], df[lat_column], crs=WGS84)
    return gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84)


def _ensure_hex_index(hexes: gpd.GeoDataFrame, hex_id_column: str) -> gpd.GeoDataFrame:
    """Return a copy of the hex GeoDataFrame indexed by the requested column."""

    if hex_id_column not in hexes.columns:
        raise ValueError(f"Hexagon layer is missing column '{hex_id_column}'.")
    hexes = hexes.copy()
    if hexes.index.name != hex_id_column:
        hexes.set_index(hex_id_column, inplace=True)
    return hexes


def _compute_polygon_coverage(
    hexes: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    *,
    coverage_column: str,
) -> gpd.GeoDataFrame:
    """Attach polygon coverage proportions to the hex grid."""

    if polygons.empty:
        hexes[coverage_column] = 0.0
        return hexes

    hex_m = hexes.to_crs(SF_UTM)
    polygons_m = polygons.to_crs(SF_UTM)
    intersections = gpd.overlay(
        hex_m[["geometry"]], polygons_m[["geometry"]], how="intersection"
    )
    if intersections.empty:
        hexes[coverage_column] = 0.0
        return hexes

    intersections["_area"] = intersections.area
    aggregated = intersections.groupby(level=0)["_area"].sum()
    hex_area = hex_m.area
    coverage = aggregated.reindex(hex_area.index).fillna(0) / hex_area
    hexes[coverage_column] = coverage.astype(float)
    return hexes


def _count_points_within(
    hexes: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    *,
    target_column: str,
) -> gpd.GeoDataFrame:
    """Count how many points fall inside each hexagon."""

    if points.empty:
        hexes[target_column] = 0
        return hexes

    points = points.to_crs(hexes.crs)
    joined = gpd.sjoin(points, hexes[["geometry"]], how="left", predicate="within")
    counts = joined.groupby("index_right").size()
    hexes[target_column] = counts.reindex(hexes.index).fillna(0).astype(int)
    return hexes


def _prepare_parks(parks: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return polygons for coverage and representative points for accessibility."""

    if parks.empty:
        empty = gpd.GeoDataFrame(columns=["geometry", "mass"], geometry=[], crs=WGS84)
        return empty, empty

    polygons = parks.copy()
    parks_m = polygons.to_crs(SF_UTM)
    masses = parks_m.area.astype(float)
    points = polygons.copy()
    points["geometry"] = points.geometry.representative_point()
    points["mass"] = masses
    return polygons, points


def _prepare_point_layer(
    gdf: gpd.GeoDataFrame,
    *,
    mass_column: str | None = None,
    default_mass: float = 1.0,
) -> gpd.GeoDataFrame:
    """Return a copy of a point layer with an explicit mass column."""

    gdf = gdf.copy()
    if gdf.empty:
        gdf["mass"] = []
        return gdf

    if mass_column and mass_column in gdf.columns:
        gdf["mass"] = gdf[mass_column].fillna(default_mass).astype(float)
    else:
        gdf["mass"] = default_mass
    return gdf


def _prepare_hexes(
    hex_path: Path,
    *,
    hex_id_column: str,
    park_coverage_column: str,
    permit_column: str,
    parks_polygons: gpd.GeoDataFrame,
    permits: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Load the hexagons and attach derived indicators."""

    hexes = _load_vector_layer(hex_path)
    hexes = _ensure_hex_index(hexes, hex_id_column)
    hexes = _compute_polygon_coverage(hexes, parks_polygons, coverage_column=park_coverage_column)
    hexes = _count_points_within(hexes, permits, target_column=permit_column)
    return hexes


def _add_optional_layer(
    container: MutableMapping[str, gpd.GeoDataFrame],
    *,
    name: str,
    path: Optional[Path],
    layer: Optional[str] = None,
) -> None:
    if path is None:
        return
    container[name] = _load_vector_layer(path, layer=layer)


def _build_layer_map(
    hexes: gpd.GeoDataFrame,
    *,
    hex_value_column: Optional[str] = None,
    point_layers: Mapping[str, gpd.GeoDataFrame] = (),
    line_layers: Mapping[str, gpd.GeoDataFrame] = (),
    polygon_layers: Mapping[str, gpd.GeoDataFrame] = (),
) -> folium.Map:
    """Create a Folium map overlaying the provided layers."""

    if hexes.empty:
        raise ValueError("Hex layer is empty – nothing to plot.")

    centroid = hexes.geometry.unary_union.centroid
    fmap = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles="cartodbpositron")

    colormap = None
    value_key = hex_value_column if hex_value_column and hex_value_column in hexes.columns else None
    if value_key:
        values = hexes[value_key].replace([float("inf"), float("-inf")], pd.NA)
        valid = values.dropna()
        if not valid.empty:
            colormap = linear.YlGnBu_09.scale(valid.min(), valid.max())
            colormap.caption = f"Hexagon {value_key}"

    def style(feature: Mapping[str, object]) -> Dict[str, object]:
        color = "#ffffff"
        fill_opacity = 0.05
        if value_key and colormap is not None:
            raw_value = feature["properties"].get(value_key)
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError):
                numeric = float("nan")
            if pd.notna(numeric):
                color = colormap(numeric)
                fill_opacity = 0.6
        return {
            "color": "#444444",
            "weight": 1,
            "fillColor": color,
            "fillOpacity": fill_opacity,
        }

    folium.GeoJson(hexes, name="Hexagons", style_function=style).add_to(fmap)
    if colormap is not None:
        colormap.add_to(fmap)

    for name, layer in polygon_layers.items():
        if layer.empty:
            continue
        folium.GeoJson(
            layer,
            name=name,
            style_function=lambda _feature, color="#2b8cbe": {
                "color": color,
                "weight": 1,
                "fillColor": color,
                "fillOpacity": 0.3,
            },
        ).add_to(fmap)

    for name, layer in line_layers.items():
        if layer.empty:
            continue
        folium.GeoJson(
            layer,
            name=name,
            style_function=lambda _feature, color="#de2d26": {
                "color": color,
                "weight": 2,
            },
        ).add_to(fmap)

    point_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    color_cycle = cycle(point_colors)
    for name, layer in point_layers.items():
        color = next(color_cycle)
        if layer.empty:
            continue
        group = FeatureGroup(name=name)
        for _, row in layer.iterrows():
            geom = row.geometry
            if not isinstance(geom, Point):
                continue
            folium.CircleMarker(
                location=(geom.y, geom.x),
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.8,
            ).add_to(group)
        group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


@dataclass
class PipelineConfig:
    hex_path: Path
    hospitals_path: Path
    parks_path: Path
    schools_path: Path
    permits_path: Path
    hex_id_column: str = "hex_id"
    population_column: str = "population"
    income_column: str = "median_income"
    park_coverage_column: str = "park_coverage"
    permit_column: str = "street_permits"
    permits_lat_column: str = "latitude"
    permits_lon_column: str = "longitude"
    permits_location_column: Optional[str] = None
    hospital_mass_column: Optional[str] = None
    school_mass_column: Optional[str] = None
    park_layer: Optional[str] = None
    hospital_layer: Optional[str] = None
    school_layer: Optional[str] = None
    permits_are_points: bool = False
    bike_network_path: Optional[Path] = None
    transit_lines_path: Optional[Path] = None
    transit_stops_path: Optional[Path] = None
    additional_polygons_path: Optional[Path] = None
    output_map: Optional[Path] = None
    output_per_hex: Optional[Path] = None
    output_per_hex_geojson: Optional[Path] = None
    output_system: Optional[Path] = None


def run_pipeline(config: PipelineConfig) -> tuple[pd.DataFrame, float]:
    """Execute the full pipeline: load, visualise, compute PCI."""

    hospitals = _load_vector_layer(config.hospitals_path, layer=config.hospital_layer)
    parks_raw = _load_vector_layer(config.parks_path, layer=config.park_layer)
    schools = _load_vector_layer(config.schools_path, layer=config.school_layer)

    if config.permits_are_points:
        permits = _load_vector_layer(config.permits_path)
    else:
        permits = _load_permits(
            config.permits_path,
            config.permits_lat_column,
            config.permits_lon_column,
            location_column=config.permits_location_column,
        )

    parks_polygons, parks_points = _prepare_parks(parks_raw)
    hospitals_points = _prepare_point_layer(
        hospitals, mass_column=config.hospital_mass_column
    )
    schools_points = _prepare_point_layer(schools, mass_column=config.school_mass_column)

    hexes = _prepare_hexes(
        config.hex_path,
        hex_id_column=config.hex_id_column,
        park_coverage_column=config.park_coverage_column,
        permit_column=config.permit_column,
        parks_polygons=parks_polygons,
        permits=permits,
    )

    optional_lines: Dict[str, gpd.GeoDataFrame] = {}
    _add_optional_layer(
        optional_lines, name="Bike network", path=config.bike_network_path
    )
    _add_optional_layer(
        optional_lines, name="Transit lines", path=config.transit_lines_path
    )

    optional_points: Dict[str, gpd.GeoDataFrame] = {
        "Hospitals": hospitals_points,
        "Schools": schools_points,
        "Permits": permits,
    }
    if config.transit_stops_path:
        optional_points["Transit stops"] = _load_vector_layer(config.transit_stops_path)

    optional_polygons: Dict[str, gpd.GeoDataFrame] = {"Parks": parks_polygons}
    if config.additional_polygons_path:
        optional_polygons["Additional polygons"] = _load_vector_layer(
            config.additional_polygons_path
        )

    calculator = AccessibilityCalculator(network=MultiModalNetwork())
    per_hex, system_pci = calculator.compute_pci(
        hex_gdf=hexes,
        hospitals_gdf=hospitals_points,
        parks_gdf=parks_points,
        schools_gdf=schools_points,
        population_column=config.population_column,
        income_column=config.income_column,
        park_coverage_column=config.park_coverage_column,
        permit_column=config.permit_column,
        amenity_mass_columns={
            "hospitals": "mass",
            "parks": "mass",
            "schools": "mass",
        },
    )

    if config.output_map:
        hex_for_map = hexes.join(per_hex, how="left")
        fmap = _build_layer_map(
            hex_for_map,
            hex_value_column="pci",
            point_layers=optional_points,
            line_layers=optional_lines,
            polygon_layers=optional_polygons,
        )
        config.output_map.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(config.output_map)
        LOGGER.info("Saved interactive map to %s", config.output_map)

    if config.output_per_hex:
        config.output_per_hex.parent.mkdir(parents=True, exist_ok=True)
        per_hex.to_csv(config.output_per_hex)
        LOGGER.info("Saved per-hexagon results to %s", config.output_per_hex)

    if config.output_per_hex_geojson:
        config.output_per_hex_geojson.parent.mkdir(parents=True, exist_ok=True)
        merged = hexes.join(per_hex, how="left")
        merged.to_file(config.output_per_hex_geojson, driver="GeoJSON")
        LOGGER.info("Saved per-hexagon GeoJSON to %s", config.output_per_hex_geojson)

    if config.output_system:
        config.output_system.parent.mkdir(parents=True, exist_ok=True)
        with config.output_system.open("w", encoding="utf-8") as fp:
            json.dump({"system_pci": system_pci}, fp, indent=2)
        LOGGER.info("Saved system PCI summary to %s", config.output_system)

    return per_hex, system_pci


def _parse_args(argv: Optional[Iterable[str]] = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Run the San Francisco PCI pipeline.")
    parser.add_argument("--hex", required=True, type=Path, help="Path to the hexagon layer")
    parser.add_argument("--hospitals", required=True, type=Path, help="Path to hospitals data")
    parser.add_argument("--parks", required=True, type=Path, help="Path to parks data")
    parser.add_argument("--schools", required=True, type=Path, help="Path to schools data")
    parser.add_argument("--permits", required=True, type=Path, help="Path to permits data")
    parser.add_argument("--hex-id-column", default="hex_id")
    parser.add_argument("--population-column", default="population")
    parser.add_argument("--income-column", default="median_income")
    parser.add_argument("--park-coverage-column", default="park_coverage")
    parser.add_argument("--permit-column", default="street_permits")
    parser.add_argument("--permits-lat-column", default="latitude")
    parser.add_argument("--permits-lon-column", default="longitude")
    parser.add_argument("--permits-location-column")
    parser.add_argument("--permits-are-points", action="store_true")
    parser.add_argument("--hospital-mass-column")
    parser.add_argument("--school-mass-column")
    parser.add_argument("--hospital-layer")
    parser.add_argument("--school-layer")
    parser.add_argument("--park-layer")
    parser.add_argument("--bike-network", type=Path)
    parser.add_argument("--transit-lines", type=Path)
    parser.add_argument("--transit-stops", type=Path)
    parser.add_argument("--additional-polygons", type=Path)
    parser.add_argument("--output-map", type=Path)
    parser.add_argument("--output-per-hex", type=Path)
    parser.add_argument("--output-per-hex-geojson", type=Path)
    parser.add_argument("--output-system", type=Path)

    args = parser.parse_args(argv)
    return PipelineConfig(
        hex_path=args.hex,
        hospitals_path=args.hospitals,
        parks_path=args.parks,
        schools_path=args.schools,
        permits_path=args.permits,
        hex_id_column=args.hex_id_column,
        population_column=args.population_column,
        income_column=args.income_column,
        park_coverage_column=args.park_coverage_column,
        permit_column=args.permit_column,
        permits_lat_column=args.permits_lat_column,
        permits_lon_column=args.permits_lon_column,
        permits_location_column=args.permits_location_column,
        hospital_mass_column=args.hospital_mass_column,
        school_mass_column=args.school_mass_column,
        hospital_layer=args.hospital_layer,
        school_layer=args.school_layer,
        park_layer=args.park_layer,
        permits_are_points=args.permits_are_points,
        bike_network_path=args.bike_network,
        transit_lines_path=args.transit_lines,
        transit_stops_path=args.transit_stops,
        additional_polygons_path=args.additional_polygons,
        output_map=args.output_map,
        output_per_hex=args.output_per_hex,
        output_per_hex_geojson=args.output_per_hex_geojson,
        output_system=args.output_system,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    config = _parse_args(argv)
    run_pipeline(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

