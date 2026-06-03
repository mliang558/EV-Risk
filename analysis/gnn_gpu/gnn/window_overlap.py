"""Disk overlap utilities for sliding-window sampling (max overlap cap)."""

from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = math.pi / 180.0
    lat1r, lon1r = lat1 * r, lon1 * r
    lat2r, lon2r = lat2 * r, lon2 * r
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def disk_overlap_fraction(r1_km: float, r2_km: float, center_dist_km: float) -> float:
    """
    Fraction of the *smaller* disk covered by intersection with the other disk.
    Returns 0 if disks are disjoint, 1 if one center inside the other.
    """
    r1, r2, d = float(r1_km), float(r2_km), float(center_dist_km)
    if r1 <= 0 or r2 <= 0:
        return 0.0
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return 1.0
    r_small = min(r1, r2)
    area_small = math.pi * r_small * r_small

    def lens(r_a: float, r_b: float, dist: float) -> float:
        # intersection area contribution from circle A
        return r_a * r_a * math.acos((dist * dist + r_a * r_a - r_b * r_b) / (2 * dist * r_a + 1e-15)) - (
            dist * math.sqrt(max(0.0, r_a * r_a - ((dist * dist + r_a * r_a - r_b * r_b) / (2 * dist + 1e-15)) ** 2))
            / 2
        )

    inter = lens(r1, r2, d) + lens(r2, r1, d)
    return float(min(1.0, inter / area_small))


def min_center_sep_km(r1_km: float, r2_km: float, max_overlap_frac: float = 0.5) -> float:
    """Lower bound on center distance so overlap fraction <= max_overlap_frac (equal-R approximation)."""
    r_max = max(float(r1_km), float(r2_km))
    if r_max <= 0:
        return 0.0
    lo, hi = 0.0, 2.0 * r_max
    for _ in range(40):
        mid = (lo + hi) / 2
        if disk_overlap_fraction(r1_km, r2_km, mid) <= max_overlap_frac:
            hi = mid
        else:
            lo = mid
    return hi


def windows_overlap(
    lat1: float,
    lon1: float,
    r1_km: float,
    lat2: float,
    lon2: float,
    r2_km: float,
    *,
    max_overlap_frac: float = 0.5,
) -> bool:
    d = haversine_km(lat1, lon1, lat2, lon2)
    return disk_overlap_fraction(r1_km, r2_km, d) > max_overlap_frac


def sep_frac_for_max_overlap(max_overlap_frac: float = 0.5, radius_km: float = 10.0) -> float:
    """Equivalent center separation / R for equal-radius disks (documentation helper)."""
    return min_center_sep_km(radius_km, radius_km, max_overlap_frac) / max(radius_km, 1e-9)


def windows_conflict(
    lat1: float,
    lon1: float,
    r1_km: float,
    stratum1: str,
    lat2: float,
    lon2: float,
    r2_km: float,
    stratum2: str,
    *,
    max_overlap_frac: float = 0.5,
    cross_stratum_mode: str = "area",
) -> bool:
    """
    area: max overlap on smaller disk <= max_overlap_frac (strict).
    center: cross-stratum only if smaller disk's center lies inside larger disk.
    """
    same = str(stratum1) == str(stratum2)
    d = haversine_km(lat1, lon1, lat2, lon2)
    if same or cross_stratum_mode == "area":
        return disk_overlap_fraction(r1_km, r2_km, d) > max_overlap_frac
    r_hi = max(float(r1_km), float(r2_km))
    return d < r_hi
