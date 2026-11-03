# Demand-placement population input

`tract_population_2020.csv.gz` contains two columns:

- `tract_geoid`: the 11-digit 2020 Census tract GEOID
- `population`: 2020 Decennial Census P1 total population (`P1_001N`)

It was extracted from the nationwide `DECENNIALPL2020.P1` ZIP downloaded from
data.census.gov on 2026-11-02. The original ZIP SHA-256 is
`77ab655f48365b8e539b508b96cee0d1a03792a6aaa17595b0f8e6e8e71a3ebc`.
The normalized uncompressed CSV contains 85,395 unique tract rows and has
SHA-256 `575af0cbfd600ef557a5218c58ed74fffaa4332911d05dfb4ff8585e8767c2b9`.

This input is used only to place Demand-comparator expansion nodes. Outage
epicenters are sampled uniformly within the selected county polygon.
