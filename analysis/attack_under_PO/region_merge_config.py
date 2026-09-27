"""
Hypernode threshold (50) and fixed state merges for Bayesian outage + network attack.

Step 1: states with < MIN_HYPERNODES hypernodes merge with a neighbor (user-specified pairs).
Step 2: validate via validate_region_merge.py (re-run MC, stability metrics).
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_HYPERNODES = 50

# Contiguous US adjacency (abbr) — for suggesting merges not in the fixed plan
STATE_NEIGHBORS: dict[str, tuple[str, ...]] = {
    "AL": ("MS", "TN", "GA", "FL"),
    "AZ": ("CA", "NV", "UT", "CO", "NM"),
    "AR": ("MO", "TN", "MS", "LA", "TX", "OK"),
    "CA": ("OR", "NV", "AZ"),
    "CO": ("WY", "NE", "KS", "OK", "NM", "AZ", "UT"),
    "CT": ("NY", "MA", "RI"),
    "DE": ("MD", "NJ", "PA"),
    "DC": ("MD", "VA"),
    "FL": ("GA", "AL"),
    "GA": ("FL", "AL", "TN", "NC", "SC"),
    "ID": ("WA", "OR", "NV", "UT", "WY", "MT"),
    "IL": ("WI", "IN", "KY", "MO", "IA"),
    "IN": ("MI", "OH", "KY", "IL"),
    "IA": ("MN", "WI", "IL", "MO", "NE", "SD"),
    "KS": ("NE", "MO", "OK", "CO"),
    "KY": ("IL", "IN", "OH", "WV", "VA", "TN", "MO"),
    "LA": ("TX", "AR", "MS"),
    "ME": ("NH"),
    "MD": ("PA", "DE", "VA", "WV"),
    "MA": ("NY", "VT", "NH", "CT", "RI"),
    "MI": ("WI", "IN", "OH"),
    "MN": ("WI", "IA", "SD", "ND"),
    "MS": ("LA", "AR", "TN", "AL"),
    "MO": ("IA", "IL", "KY", "TN", "AR", "OK", "KS", "NE"),
    "MT": ("ID", "WY", "SD", "ND"),
    "NE": ("SD", "IA", "MO", "KS", "CO", "WY"),
    "NV": ("OR", "ID", "UT", "AZ", "CA"),
    "NH": ("ME", "VT", "MA"),
    "NJ": ("NY", "PA", "DE"),
    "NM": ("CO", "OK", "TX", "AZ"),
    "NY": ("VT", "MA", "CT", "NJ", "PA"),
    "NC": ("VA", "TN", "GA", "SC"),
    "ND": ("MN", "SD", "MT"),
    "OH": ("MI", "PA", "WV", "KY", "IN"),
    "OK": ("KS", "MO", "AR", "TX", "NM", "CO"),
    "OR": ("WA", "ID", "NV", "CA"),
    "PA": ("NY", "NJ", "DE", "MD", "WV", "OH"),
    "RI": ("CT", "MA"),
    "SC": ("NC", "GA"),
    "SD": ("ND", "MN", "IA", "NE", "WY"),
    "TN": ("KY", "VA", "NC", "GA", "AL", "MS", "AR", "MO"),
    "TX": ("NM", "OK", "AR", "LA"),
    "UT": ("ID", "WY", "CO", "AZ", "NV"),
    "VT": ("NY", "NH", "MA"),
    "VA": ("MD", "WV", "KY", "TN", "NC"),
    "WA": ("ID", "OR"),
    "WV": ("OH", "PA", "MD", "VA", "KY"),
    "WI": ("MN", "IA", "IL", "MI"),
    "WY": ("MT", "SD", "NE", "CO", "UT", "ID"),
}

ABBR_TO_STATE_NAME: dict[str, str] = {
    "AL": "Alabama",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}


@dataclass(frozen=True)
class RegionSpec:
    region_id: str
    display_name: str
    member_abbr: tuple[str, ...]

    @property
    def member_states(self) -> tuple[str, ...]:
        return tuple(ABBR_TO_STATE_NAME[a] for a in self.member_abbr)


# Paper Step 1: five fixed adjacent pairs (all members typically Small in S/L strata).
# Use audit_hypernode_counts.py to compare |V| vs MIN_HYPERNODES (=50).
REGION_MERGE_PLAN: tuple[RegionSpec, ...] = (
    RegionSpec("chesapeake", "Chesapeake region", ("DE", "MD")),
    RegionSpec("southern_new_england", "Southern New England", ("RI", "CT")),
    RegionSpec("northern_new_england", "Northern New England", ("VT", "NH")),
    RegionSpec("northern_mountain", "Northern Mountain", ("WY", "MT")),
    RegionSpec("dakotas", "Dakotas", ("ND", "SD")),
)

REGION_BY_ID: dict[str, RegionSpec] = {r.region_id: r for r in REGION_MERGE_PLAN}

# States absorbed into a region (no longer analyzed alone)
MERGED_AWAY_ABBR: frozenset[str] = frozenset(
    abbr for spec in REGION_MERGE_PLAN for abbr in spec.member_abbr
)

# Paper residual narrative (JTG)
DEEP_SOUTH_STATES: tuple[str, ...] = (
    "Louisiana",
    "Mississippi",
    "Alabama",
)
SIZE_ARTIFACT_LABELS: tuple[str, ...] = (
    "Chesapeake region",
    "Southern New England",
    "Northern New England",
    "Northern Mountain",
    "Dakotas",
)


def suggest_neighbor_merge(small_abbr: str, n_nodes_by_abbr: dict[str, int]) -> str | None:
    """
    If a state is below threshold and not in the fixed plan, merge with adjacent
    neighbor that has the smallest hypernode count (tie-break: lexicographic abbr).
    """
    neighbors = STATE_NEIGHBORS.get(small_abbr, ())
    if not neighbors:
        return None
    candidates = [
        nb for nb in neighbors if nb not in MERGED_AWAY_ABBR and nb != small_abbr
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda a: (n_nodes_by_abbr.get(a, 10**9), a))


def analysis_units_after_merge(
    n_nodes_by_abbr: dict[str, int],
    *,
    threshold: int = MIN_HYPERNODES,
) -> dict[str, list[str]]:
    """
    Return mapping unit_id -> list of abbr.
    unit_id is either a state abbr or a region_id from REGION_MERGE_PLAN.
    """
    units: dict[str, list[str]] = {}
    covered: set[str] = set()

    for spec in REGION_MERGE_PLAN:
        units[spec.region_id] = list(spec.member_abbr)
        covered.update(spec.member_abbr)

    for abbr, n in sorted(n_nodes_by_abbr.items()):
        if abbr in covered:
            continue
        if n >= threshold:
            units[abbr] = [abbr]
            covered.add(abbr)
            continue
        partner = suggest_neighbor_merge(abbr, n_nodes_by_abbr)
        if partner and partner not in covered:
            units[f"{abbr}_{partner}"] = [abbr, partner]
            covered.update([abbr, partner])
        elif partner:
            units[abbr] = [abbr]
            covered.add(abbr)
        else:
            units[abbr] = [abbr]
            covered.add(abbr)

    return units
