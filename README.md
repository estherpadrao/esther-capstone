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
