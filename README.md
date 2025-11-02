# esther-capstone

Esther's Capstone Code, Data and Outputs

## Getting started

The repository targets Python 3.10+ and relies on the scientific geo stack
(`geopandas`, `osmnx`, `folium`, etc.).  Install the dependencies before running
any scripts:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Colab tip:** add a notebook cell with `!pip install -r requirements.txt`
> after cloning or uploading the repository so the environment matches the
> local setup.

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

### Using the pipeline in Google Colab

1. Clone or upload the repository into your Colab workspace:

   ```python
   !git clone https://github.com/<your-org>/esther-capstone.git
   %cd esther-capstone
   ```

2. Install the dependencies in a new cell:

   ```python
   !pip install -r requirements.txt
   ```

3. Stage your data so the notebook can read it.  Pick the option that matches
   where the files currently live:

   * **Already tracked in GitHub** – If your datasets are committed to this
     repository (or another public repo), cloning in step 1 already downloaded
     them.  You can confirm with `!ls data`.
   * **Stored on your computer** – Upload them straight from the browser:

     ```python
     from google.colab import files
     from pathlib import Path

     uploaded = files.upload()  # select your GeoJSON/CSV layers when prompted

     data_dir = Path("data")
     data_dir.mkdir(exist_ok=True)

     # Map the uploaded filenames to the ones used by the pipeline.
     # Update the keys to match the values shown in `uploaded.keys()`.
     rename_map = {
         "Active_Street-Use_Permits_20251031.csv": "permits.csv",
         "Health_Care_Facilities_20251030.geojson": "hospitals.geojson",
         "Recreation_and_Parks_Properties_20251031.geojson": "parks.geojson",
         "Schools_20251031.csv": "schools.csv",
         "MTA_Bike_Network_Linear_Features_20251031.geojson": "bike_network.geojson",
         "Muni_Simple_Routes_20251031.geojson": "transit_lines.geojson",
         "Muni_Stops_20251031.geojson": "transit_stops.geojson",
         # add your hexagon grid file and any other layers here
     }

     for original_name, target_name in rename_map.items():
         source = Path(original_name)
         if source.exists():
             source.rename(data_dir / target_name)
         else:
             print(f"Skipping {original_name!r} because it was not uploaded in this session")
     ```

     This approach avoids shell `mv` errors when Colab appends suffixes like
     `" (1)"` to uploaded files (e.g., `"Muni_Simple_Routes_20251031 (1).geojson"`).
     Update the `rename_map` keys to match the exact names reported in
     `uploaded.keys()` before running the cell.

     If any amenity dataset only exists as a CSV with latitude/longitude
     columns (for example, the `Schools_20251031.csv` download), convert it to a
     GeoJSON file after uploading so the pipeline can read it with GeoPandas:

     ```python
     import geopandas as gpd
     import pandas as pd

     df = pd.read_csv("data/schools.csv")
     geometry = gpd.points_from_xy(df["Longitude"], df["Latitude"], crs=4326)
     schools_gdf = gpd.GeoDataFrame(df, geometry=geometry)
     schools_gdf.to_file("data/schools.geojson", driver="GeoJSON")
     ```

     Adjust the column names (`"Longitude"`, `"Latitude"`) to match the headers
     in your CSV, then point the pipeline to `data/schools.geojson` instead of
     the CSV path.

   * **Hosted in a different GitHub project or URL** – Download them into the
     `data/` folder with `wget` (or clone the data-only repo and copy the
     relevant files):

     ```python
     !mkdir -p data
     !wget -O data/hexagons.geojson https://raw.githubusercontent.com/<user>/<repo>/main/hexagons.geojson
     !wget -O data/hospitals.geojson https://raw.githubusercontent.com/<user>/<repo>/main/hospitals.geojson
     # repeat for the remaining layers
     ```

4. Run the pipeline directly from a notebook cell, reusing the same arguments
   exposed by the CLI:

   ```python
   from pathlib import Path
   from src import PipelineConfig, run_pipeline

   config = PipelineConfig(
       hex_path=Path("data/hexagons.geojson"),
       hospitals_path=Path("data/hospitals.geojson"),
       parks_path=Path("data/parks.geojson"),
       schools_path=Path("data/schools.geojson"),
       permits_path=Path("data/permits.csv"),
       bike_network_path=Path("data/bike_network.geojson"),
       transit_lines_path=Path("data/transit_lines.geojson"),
       transit_stops_path=Path("data/transit_stops.geojson"),
       output_map=Path("outputs/sf_layers.html"),
       output_per_hex=Path("outputs/pci_per_hex.csv"),
       output_per_hex_geojson=Path("outputs/pci_per_hex.geojson"),
       output_system=Path("outputs/system_pci.json"),
   )

   # Override any paths whose filenames you chose not to rename, e.g.:
   # config = config.replace(
   #     hex_path=Path("data/sf_polygon.geojson"),
   #     schools_path=Path("data/Schools_20251031.csv"),
   # )

   per_hex, system_pci = run_pipeline(config)
   per_hex.head()
   system_pci
   ```

This Colab workflow mirrors the command-line invocation while keeping all
intermediate outputs accessible for inspection within the notebook session.
