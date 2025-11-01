# esther-capstone

Esther's Capstone Code, Data and Outputs

## Accessibility analysis toolkit

The `src/accessibility.py` module provides a reusable implementation of a
multimodal, income-adjusted Hansen accessibility index tailored to San
Francisco.  It exposes two entry points:

* `MultiModalNetwork` downloads walking, cycling and transit graphs from
  OpenStreetMap via `osmnx` and computes travel times between arbitrary
  origins and destinations.
* `AccessibilityCalculator` consumes the network together with hexagon-level
  socio-demographic data and amenity layers (hospitals, parks and schools) to
  produce per-hexagon and system-wide accessibility scores.  The calculation
  enforces the 98% park-coverage rule for empty hexagons, adjusts travel costs
  by income, applies the required amenity weights (0.5 hospitals, 0.2 parks,
  0.3 schools), and scales the final index by street-permit activity.

Both classes operate on `geopandas` GeoDataFrames and can therefore be used in
Jupyter notebooks or scripts without further adaptation.

### Example usage

```python
import geopandas as gpd

from src.accessibility import AccessibilityCalculator, MultiModalNetwork

# Load your prepared GeoDataFrames (examples shown with placeholder paths)
hexes = gpd.read_file("data/hexagons.geojson").set_index("hex_id")
hospitals = gpd.read_file("data/hospitals.geojson")
parks = gpd.read_file("data/parks.geojson")
schools = gpd.read_file("data/schools.geojson")

network = MultiModalNetwork()
calculator = AccessibilityCalculator(network=network)

per_hex_pci, system_pci = calculator.compute_pci(
    hex_gdf=hexes,
    hospitals_gdf=hospitals,
    parks_gdf=parks,
    schools_gdf=schools,
    population_column="population",        # customise if your column names differ
    income_column="median_income",
    park_coverage_column="park_coverage",
    permit_column="street_permits",
)

print(per_hex_pci[["hospital_access", "park_access", "school_access", "pci"]])
print("System-wide PCI:", system_pci)
```

`compute_pci` returns a tuple containing a per-hexagon `pandas.DataFrame` with
the intermediate and final indicators, and the overall (population-weighted)
PCI for the complete study area.  You can further post-process or visualise the
returned table within notebooks to replicate previous analyses.
