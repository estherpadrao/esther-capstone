"""Computation of an adapted Hansen-type accessibility index for San Francisco.

This module provides utilities to build a multimodal transportation graph for
San Francisco and to evaluate a compound accessibility index (PCI) for a set of
hexagonal zones.  The PCI is defined as a weighted combination of accessibility
scores to hospitals, parks and schools, adjusted by household income, street
permits and population.

The code is designed to be data-source agnostic: callers are expected to supply
pre-processed ``geopandas`` GeoDataFrames representing the analysis hexagons and
amenities.  The computation is nevertheless robust; it performs validation,
allows custom hyperparameters, and will gracefully skip unreachable amenities
while still producing transparent diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multimodal graph construction utilities
# ---------------------------------------------------------------------------


def _default_graph_settings() -> Dict[str, Dict[str, float]]:
    """Return default per-mode configuration."""

    return {
        "walk": {"speed_kph": 5.0},
        "bike": {"speed_kph": 15.0},
        "transit": {"speed_kph": 28.0},
    }


@dataclass
class MultiModalNetwork:
    """Build and query a multimodal transportation graph for San Francisco."""

    place_name: str = "San Francisco, California, USA"
    graph_settings: Mapping[str, Mapping[str, float]] = field(
        default_factory=_default_graph_settings
    )
    transfer_penalty_min: float = 4.0
    cache_graphs: bool = True

    _graphs: MutableMapping[str, nx.MultiDiGraph] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not self.graph_settings:
            self.graph_settings = _default_graph_settings()

    def ensure_graphs(self) -> None:
        """Download and cache the per-mode graphs if they are not available."""

        if self._graphs and self.cache_graphs:
            return

        for mode in ("walk", "bike", "transit"):
            graph = self._download_graph(mode)
            graph = ox.distance.add_edge_lengths(graph)
            speed_kph = self.graph_settings.get(mode, {}).get("speed_kph", 5.0)
            speed_m_per_min = speed_kph * 1000 / 60
            for _, _, edge_data in graph.edges(data=True):
                length = edge_data.get("length")
                if length is None:
                    continue
                edge_data["travel_time"] = length / speed_m_per_min
                edge_data["mode"] = mode
            self._graphs[mode] = graph
            logger.info("Loaded %s network with %d edges", mode, graph.number_of_edges())

    def _download_graph(self, mode: str) -> nx.MultiDiGraph:
        """Download an OpenStreetMap graph for the requested travel mode."""

        if mode == "walk":
            return ox.graph_from_place(self.place_name, network_type="walk")
        if mode == "bike":
            return ox.graph_from_place(self.place_name, network_type="bike")
        if mode == "transit":
            custom_filter = (
                '["railway"~"subway|light_rail|tram|rail|station"]'
                '["service"!~"yard|spur|siding"]'
            )
            return ox.graph_from_place(
                self.place_name,
                custom_filter=custom_filter,
                simplify=True,
            )
        raise ValueError(f"Unsupported mode '{mode}'.")

    def _nearest_node(self, graph: nx.MultiDiGraph, point: Point) -> int:
        """Return the identifier of the node nearest to the provided point."""

        return ox.distance.nearest_nodes(graph, point.x, point.y)  # type: ignore[arg-type]

    def _shortest_path_cost(
        self,
        graph: nx.MultiDiGraph,
        origin: Point,
        destination: Point,
        weight: str = "travel_time",
    ) -> Optional[float]:
        """Compute the shortest-path cost between two coordinates on a graph."""

        try:
            origin_node = self._nearest_node(graph, origin)
            destination_node = self._nearest_node(graph, destination)
        except (ValueError, nx.NetworkXException) as exc:  # pragma: no cover - defensive
            logger.warning("Unable to match nodes for %s graph: %s", graph, exc)
            return None

        try:
            return nx.shortest_path_length(graph, origin_node, destination_node, weight=weight)
        except nx.NetworkXNoPath:
            return None

    def multimodal_travel_cost(
        self,
        origin: Point,
        destination: Point,
        penalise_transfers: bool = True,
    ) -> Optional[float]:
        """Return the minimal generalised travel cost across the available modes."""

        self.ensure_graphs()
        if not self._graphs:
            return None

        costs: List[float] = []
        for mode, graph in self._graphs.items():
            cost = self._shortest_path_cost(graph, origin, destination)
            if cost is None:
                continue
            if penalise_transfers and mode != "walk":
                cost += self.transfer_penalty_min
            costs.append(cost)

        if not costs:
            return None
        return min(costs)


# ---------------------------------------------------------------------------
# Hansen-type accessibility computation
# ---------------------------------------------------------------------------


def _validate_hex_population(
    hex_gdf: gpd.GeoDataFrame,
    population_column: str,
    park_coverage_column: str,
    threshold: float = 0.98,
) -> None:
    """Ensure that hexagons without population are mostly covered by parks."""

    invalid = hex_gdf[
        (hex_gdf[population_column] <= 0) & (hex_gdf[park_coverage_column] < threshold)
    ]
    if not invalid.empty:
        invalid_ids = invalid.index.tolist()
        raise ValueError(
            "Hexagons without population must have at least "
            f"{threshold:.0%} park coverage. Invalid hexagon indices: {invalid_ids}"
        )


def _population_mass(hex_row: pd.Series, population_column: str) -> float:
    """Return the population mass for the provided hexagon."""

    return float(max(hex_row[population_column], 0.0))


def _income_adjustment(
    base_cost_min: float,
    income: float,
    reference_income: float,
    elasticity: float = 1.0,
    min_income: float = 1_000.0,
) -> float:
    """Adjust travel cost so that lower-income areas experience higher costs."""

    adjusted_income = max(income, min_income)
    factor = (reference_income / adjusted_income) ** elasticity
    return base_cost_min * factor


def _as_point(geometry: BaseGeometry) -> Optional[Point]:
    """Return a representative point for an arbitrary geometry."""

    if isinstance(geometry, Point):
        return geometry
    if geometry.is_empty:
        return None
    try:
        return geometry.representative_point()
    except AttributeError:  # pragma: no cover - safety for exotic geometries
        return None


def _amenity_accessibility(
    hex_centroid: Point,
    amenity_gdf: gpd.GeoDataFrame,
    amenity_mass_column: str,
    network: MultiModalNetwork,
    income_adjuster: Callable[[float], float],
    beta: float,
) -> float:
    """Compute Hansen-type accessibility to a given amenity."""

    if amenity_gdf.empty:
        return 0.0

    score = 0.0
    for _, amenity in amenity_gdf.iterrows():
        destination = _as_point(amenity.geometry)
        if destination is None:
            continue
        base_cost = network.multimodal_travel_cost(hex_centroid, destination)
        if base_cost is None:
            continue
        adjusted_cost = income_adjuster(base_cost)
        impedance = adjusted_cost**beta
        if impedance <= 0:
            continue
        score += float(amenity.get(amenity_mass_column, 1.0)) / impedance
    return score


@dataclass
class AccessibilityCalculator:
    """Calculate the PCI for a system of hexagons."""

    network: MultiModalNetwork
    beta: float = 1.5
    permit_scaling: float = 1.0
    permit_offset: float = 1.0
    income_elasticity: float = 1.0

    def compute_pci(
        self,
        hex_gdf: gpd.GeoDataFrame,
        hospitals_gdf: gpd.GeoDataFrame,
        parks_gdf: gpd.GeoDataFrame,
        schools_gdf: gpd.GeoDataFrame,
        *,
        population_column: str = "population",
        income_column: str = "median_income",
        park_coverage_column: str = "park_coverage",
        permit_column: str = "street_permits",
        amenity_mass_columns: Mapping[str, str] | None = None,
    ) -> Tuple[pd.DataFrame, float]:
        """Compute the PCI per hexagon and for the entire system."""

        if amenity_mass_columns is None:
            amenity_mass_columns = {
                "hospitals": "mass",
                "parks": "mass",
                "schools": "mass",
            }

        amenity_layers = {
            "hospitals": hospitals_gdf,
            "parks": parks_gdf,
            "schools": schools_gdf,
        }

        required_cols = {
            population_column,
            income_column,
            park_coverage_column,
            permit_column,
        }
        missing = required_cols - set(hex_gdf.columns)
        if missing:
            raise ValueError(f"Hexagon GeoDataFrame is missing columns: {sorted(missing)}")

        for amenity_name, gdf in amenity_layers.items():
            mass_col = amenity_mass_columns.get(amenity_name, "mass")
            if mass_col not in gdf.columns and not gdf.empty:
                raise ValueError(
                    f"Amenity layer '{amenity_name}' is missing column '{mass_col}'."
                )

        _validate_hex_population(hex_gdf, population_column, park_coverage_column)

        reference_income = float(hex_gdf[income_column].median())
        if not np.isfinite(reference_income) or reference_income <= 0:
            raise ValueError("Invalid reference income computed from hexagons.")

        per_hexagon_records: List[Dict[str, float]] = []
        weights = {
            "hospitals": 0.5,
            "parks": 0.2,
            "schools": 0.3,
        }

        for idx, hex_row in hex_gdf.iterrows():
            centroid: Point = hex_row.geometry.centroid
            income = float(hex_row[income_column])
            population_mass = _population_mass(hex_row, population_column)

            def income_adjuster(base_cost: float) -> float:
                return _income_adjustment(
                    base_cost,
                    income=income,
                    reference_income=reference_income,
                    elasticity=self.income_elasticity,
                )

            scores: Dict[str, float] = {}
            for amenity_name, gdf in amenity_layers.items():
                scores[amenity_name] = _amenity_accessibility(
                    centroid,
                    gdf,
                    amenity_mass_columns.get(amenity_name, "mass"),
                    self.network,
                    income_adjuster,
                    self.beta,
                )

            hospital_score = scores.get("hospitals", 0.0)
            park_score = scores.get("parks", 0.0)
            school_score = scores.get("schools", 0.0)

            weighted_access = (
                weights["hospitals"] * hospital_score
                + weights["parks"] * park_score
                + weights["schools"] * school_score
            )

            permit_count = float(hex_row[permit_column])
            permit_multiplier = np.log1p(permit_count * self.permit_scaling) + self.permit_offset
            pci_value = weighted_access * permit_multiplier

            per_hexagon_records.append(
                {
                    "hex_id": idx,
                    "population": population_mass,
                    "income": income,
                    "hospital_access": hospital_score,
                    "park_access": park_score,
                    "school_access": school_score,
                    "weighted_access": weighted_access,
                    "street_permits": permit_count,
                    "permit_multiplier": permit_multiplier,
                    "pci": pci_value,
                }
            )

        per_hexagon_df = pd.DataFrame(per_hexagon_records).set_index("hex_id")
        if per_hexagon_df.empty:
            system_pci = float("nan")
        else:
            weights_array = per_hexagon_df["population"].to_numpy()
            pci_array = per_hexagon_df["pci"].to_numpy()
            valid_mask = np.isfinite(pci_array)
            if not np.any(valid_mask):
                system_pci = float("nan")
            else:
                pci_array = pci_array[valid_mask]
                weights_array = weights_array[valid_mask]
                if np.all(weights_array == 0):
                    system_pci = float(np.mean(pci_array))
                else:
                    system_pci = float(np.average(pci_array, weights=weights_array))

        return per_hexagon_df, system_pci


__all__ = [
    "AccessibilityCalculator",
    "MultiModalNetwork",
]
