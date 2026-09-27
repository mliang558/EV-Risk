#!/usr/bin/env python3
"""
Batch-1 counterfactual scenarios (2023 network × pooled outages).

Design: one-at-a-time (OAT) dose sweep with common random numbers.
  - Network, outage draws, and seeds fixed across scenarios within a sim.
  - Only one intervention factor changes at a time.
  - Shared dose grid for CF-D and CF-S: 0 (baseline), 10, 20, 30, 50%.

CF-D placement rules (coverage expansion, not within-cluster densify):
  CF-D      = NEVI corridor (main) — every 50 mi on interstate, >10 km from existing
  CF-D-pop  = population-weighted in unit counties
  CF-D-null = uniform spatial null, 10 seed replicates

Does NOT touch the original Monte Carlo pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from counterfactual_batch1.densify_network import NULL_REPLICATES

# Shared intensity grid for dose-response (percent). Baseline = 0 is separate.
DOSE_PCTS: tuple[int, ...] = (10, 20, 30, 50)


@dataclass(frozen=True)
class Scenario:
    """One counterfactual / robustness setting."""

    key: str
    family: str  # baseline | CF-D | CF-D-pop | CF-D-null | CF-S | CF-T | U | K | CF-D-within
    radius_scale: float = 1.0
    duration_scale: float = 1.0
    urban_radius_scale: float | None = None  # if set, only urban counties use this
    densify_pct: float | None = None  # fraction of |V| new hypernodes
    densify_mode: str | None = None  # "nevi" | "pop" | "uniform" | "within"
    densify_seed: int = 42
    densify_replicate: int | None = None  # null control only
    network_variant: str = "base_10km"  # base_10km | densify_* | k5km
    # Dose for OAT curves: 0 = baseline; else % intensity
    dose_pct: float = 0.0
    description: str = ""


def batch1_scenarios() -> list[Scenario]:
    """
    Batch 1 matrix (2023 × pooled), sharing one CRN event draw per sim.

    Primary OAT pair: CF-D (NEVI corridor coverage expansion) vs CF-S.
    CF-D-pop / CF-D-null are robustness / null controls on the same dose grid.
    Within-cluster densification is beyond model resolution (optional CF-D-within).
    """
    out: list[Scenario] = [
        Scenario(
            key="baseline",
            family="baseline",
            network_variant="base_10km",
            dose_pct=0.0,
            description="OAT reference: 2023 net, pooled outages, no intervention",
        ),
    ]
    for pct in DOSE_PCTS:
        frac = pct / 100.0
        # --- main: NEVI corridor ---
        out.append(
            Scenario(
                key=f"CF-D_nevi_p{pct}",
                family="CF-D",
                densify_pct=frac,
                densify_mode="nevi",
                network_variant=f"densify_nevi_p{pct}",
                dose_pct=float(pct),
                description=(
                    f"NEVI corridor coverage: +{pct}% hypernodes every 50 mi on "
                    f"interstate, excluding sites <10 km from existing (main CF-D)"
                ),
            )
        )
        # --- demand-oriented ---
        out.append(
            Scenario(
                key=f"CF-D_pop_p{pct}",
                family="CF-D-pop",
                densify_pct=frac,
                densify_mode="pop",
                network_variant=f"densify_pop_p{pct}",
                dose_pct=float(pct),
                description=(
                    f"Population-weighted coverage: +{pct}% hypernodes in unit "
                    f"counties, >10 km from existing"
                ),
            )
        )
        # --- spatial null (10 replicates) ---
        for r in range(NULL_REPLICATES):
            out.append(
                Scenario(
                    key=f"CF-D_null_p{pct}_r{r}",
                    family="CF-D-null",
                    densify_pct=frac,
                    densify_mode="uniform",
                    densify_seed=42,
                    densify_replicate=r,
                    network_variant=f"densify_uniform_p{pct}_r{r}",
                    dose_pct=float(pct),
                    description=(
                        f"Uniform spatial null: +{pct}% hypernodes, replicate {r}/"
                        f"{NULL_REPLICATES - 1}"
                    ),
                )
            )
        # optional diagnostic
        out.append(
            Scenario(
                key=f"CF-D_within_p{pct}",
                family="CF-D-within",
                densify_pct=frac,
                densify_mode="within",
                network_variant=f"densify_within_p{pct}",
                dose_pct=float(pct),
                description=(
                    f"Diagnostic: within-cluster add (beyond 10 km resolution; "
                    f"topology fixed → ΔL≈0 by construction)"
                ),
            )
        )
    for pct in DOSE_PCTS:
        scale = 1.0 - pct / 100.0
        out.append(
            Scenario(
                key=f"CF-S_m{pct}",
                family="CF-S",
                radius_scale=scale,
                dose_pct=float(pct),
                description=f"OAT severity: impact radius R_c −{pct}%",
            )
        )
        out.append(
            Scenario(
                key=f"CF-T_m{pct}",
                family="CF-T",
                duration_scale=scale,
                dose_pct=float(pct),
                description=f"OAT duration: T_i −{pct}%",
            )
        )
    for mult, tag in ((1.5, "1p5"), (2.0, "2")):
        out.append(
            Scenario(
                key=f"U_x{tag}",
                family="U",
                urban_radius_scale=mult,
                dose_pct=float("nan"),
                description=f"Urban counties: R_c × {mult} (not on CF-D/CF-S dose grid)",
            )
        )
    out.append(
        Scenario(
            key="K_5km",
            family="K",
            network_variant="k5km",
            dose_pct=float("nan"),
            description="5 km clustering network (not on CF-D/CF-S dose grid)",
        )
    )
    return out


def filter_scenarios(
    scenarios: list[Scenario],
    families: set[str] | None,
    keys: set[str] | None = None,
) -> list[Scenario]:
    out = scenarios
    if families:
        fam = set(families)
        # Do NOT auto-include CF-D-within / CF-D-pop / CF-D-null with CF-D.
        out = [s for s in out if s.family in fam or s.key == "baseline"]
        if not any(s.key == "baseline" for s in out):
            out = [scenarios[0]] + out
    if keys:
        keep = keys | {"baseline"}
        out = [s for s in out if s.key in keep]
    seen: set[str] = set()
    uniq: list[Scenario] = []
    for s in out:
        if s.key in seen:
            continue
        seen.add(s.key)
        uniq.append(s)
    return uniq


def dose_grid_families() -> tuple[str, ...]:
    """Families that share DOSE_PCTS for dose-response plots."""
    return ("CF-D", "CF-D-pop", "CF-D-null", "CF-D-within", "CF-S", "CF-T")
