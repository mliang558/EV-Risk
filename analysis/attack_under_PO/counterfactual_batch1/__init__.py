"""
Counterfactual Batch 1 — new package (does not overwrite existing MC pipeline).

Batch 1: 2023 network × pooled outages, CRN across CF-D / CF-S / CF-T / U / K.
"""

from counterfactual_batch1.scenarios import batch1_scenarios, filter_scenarios

__all__ = ["batch1_scenarios", "filter_scenarios"]
