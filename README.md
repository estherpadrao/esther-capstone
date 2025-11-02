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

### End-to-end pipeline and mapping

If you prefer a scripted workflow that mirrors the Colab notebook, the
`src/data_pipeline.py` module stitches the data preparation, visualisation and
PCI calculation steps together.  It expects local copies of the files you were
previously uploading in Colab (hex grid, hospitals, parks, schools, transit
layers, permits CSV, etc.) and produces:

* an interactive Folium map overlaying the hexagons with points, lines and
  polygons so you can validate spatial joins and coverage;
* a per-hexagon CSV/GeoJSON containing the full set of calculated indicators;
* a JSON summary with the system-wide PCI.

Run the pipeline from the repository root once your datasets are stored on disk:

```bash
python -m src.data_pipeline \
  --hex data/hexagons.geojson \
  --hospitals data/hospitals.geojson \
  --parks data/parks.geojson \
  --schools data/schools.geojson \
  --permits data/permits.csv \
  --permits-lat-column latitude \
  --permits-lon-column longitude \
  --bike-network data/bike_routes.geojson \
  --transit-lines data/transit_lines.geojson \
  --transit-stops data/transit_stops.geojson \
  --output-map outputs/sf_layers.html \
  --output-per-hex outputs/pci_per_hex.csv \
  --output-per-hex-geojson outputs/pci_per_hex.geojson \
  --output-system outputs/system_pci.json
```

Adjust the command line arguments to match your filenames and column naming
conventions (for instance if your hexagon file stores the population column
under a different name).  The CLI offers flags for datasets stored in
multi-layer GeoPackages (`--hospital-layer`, `--park-layer`, etc.) and for
permits already stored as spatial formats (`--permits-are-points`).

The generated HTML map colours the hexagons by the computed PCI and includes
separate toggleable layers for hospitals, schools, permits, transit lines, bike
lanes and park polygons, providing a quick visual QA that every geometry falls
where expected.
