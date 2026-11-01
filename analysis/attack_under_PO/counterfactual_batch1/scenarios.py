"""Intervention scenarios for the 2023 canonical network."""

from __future__ import annotations

from dataclasses import dataclass

DOSE_PCTS = (0, 10, 20, 30, 50)


@dataclass(frozen=True)
class Scenario:
    key: str
    family: str
    dose_pct: float = 0.0
    affected_customer_scale: float | None = None
    densify_pct: float | None = None
    demand_pct: float | None = None
    network_variant: str = "base_10km"
    description: str = ""


def batch1_scenarios() -> list[Scenario]:
    """Baseline plus two expansion strategies and severity reduction."""
    scenarios = [
        Scenario(
            key="baseline",
            family="baseline",
            description="2023 network with pooled 2018-2023 outages",
        )
    ]

    for pct in DOSE_PCTS:
        scenarios.append(
            Scenario(
                key=f"CF-D_nevi_p{pct}",
                family="CF-D",
                dose_pct=float(pct),
                densify_pct=pct / 100 if pct else None,
                network_variant="base_10km" if pct == 0 else f"nevi_p{pct}",
                description=(
                    "No corridor expansion"
                    if pct == 0
                    else f"Target {pct}% more hypernodes at eligible Interstate sites"
                ),
            )
        )
        scenarios.append(
            Scenario(
                key=f"Demand_p{pct}",
                family="Demand",
                dose_pct=float(pct),
                demand_pct=pct / 100 if pct else None,
                network_variant="base_10km" if pct == 0 else f"demand_p{pct}",
                description=(
                    "No demand-targeted expansion"
                    if pct == 0
                    else "Add the same number of nodes as the corridor scenario, "
                    "prioritizing high-population census tracts"
                ),
            )
        )
        scenarios.append(
            Scenario(
                key=f"CF-S_m{pct}",
                family="CF-S",
                dose_pct=float(pct),
                affected_customer_scale=1 - pct / 100,
                description=f"Reduce affected customers by {pct}%",
            )
        )
    return scenarios
