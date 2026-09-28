# Geography lock — EAGLE-I FIPS ↔ county shp ↔ tracts

## What was already matched (outage pipeline)

EAGLE-I events carry **county FIPS** and **customers out**. Cleaning joins:

1. Outages → county FIPS  
2. `MCC.csv` → total customers per county (same FIPS)  
3. Coverage `theta_s` by state-year  
4. Impact radius uses MCC + county **area** from the county shapefile  

So the county shapefile **must use the same FIPS vintage as EAGLE-I/MCC**, not “whatever year is newest.”

## What we verified in this repo

| Source | CT FIPS |
|--------|---------|
| `cleaned_outages_2018_2023.csv` (incl. 2023) | legacy 8: `09001`…`09015` |
| `MCC.csv` | same 8 |
| `tl_2021_us_county` | same 8 |

**Never use** TIGER county files from 2022+ for this project: CT planning-region equivalents (`09110`–`09190`) will not join EAGLE-I. Unit **Connecticut = CT+RI** makes a silent FIPS miss especially costly.

Code guard: `assert_county_geoids_match_eagle_i()` in `compute_impact_radius.py` (called when loading county polygons / MCC geometry).

## Tract epicenters (Batch1 / §4.3)

| Layer | Vintage | Role |
|-------|---------|------|
| Tract polygons | **TIGER 2020** | Epicenter support; stable 2020–2029 |
| Tract population | **2020 PL P1_001N** (or ACS 2019–2023 on 2020 GEOID) | Weights inside county |
| `county_fips` on tracts | First 5 of 2020 tract GEOID | Must equal EAGLE-I county FIPS |

Tracts are **not** re-aggregated to CT planning regions. Station-KDE is disallowed.

## Practical rule

- County shp: **tl_2021** (locked)  
- Tract: **2020** + 2020 census (or ACS on 2020 tracts)  
- Customers / radius: MCC + EAGLE-I FIPS (already aligned)  
