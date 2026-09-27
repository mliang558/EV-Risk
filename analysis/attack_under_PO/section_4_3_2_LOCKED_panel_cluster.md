# Section 4.3.2 — locked panel-cluster coefficients (SUPPLEMENTARY)

> **Role:** Supplementary to Section 4.3 main analysis
> (`section_4_3_exposure_structure.py`: \(L_{\mathrm{event}}=P(\mathrm{hit})\times E[\mathrm{loss}\mid\mathrm{hit}]\)).
> Prefer the exposure×structure decomposition for the primary claim that outage
> footprint exposure and network structure contribute separately; keep these
> regressions as robustness / covariate checks.

Generated: 2026-06-22T20:27:48.541657+00:00

## L_event — log|V| (primary spec, Steps 1–3)

| Step | β | SE | p | R² |
|------|---|----|----|-----|
| 1 | -0.3523 | 0.3201 | 0.2770 | 0.030 |
| 2 | 0.1320 | 0.2603 | 0.6147 | 0.251 |
| 3 | -0.0166 | 0.6554 | 0.9799 | 0.252 |

## Step 1 structural — density & λ₂ (panel cluster)

| Outcome | Variable | β | SE | p |
|---------|----------|---|----|----|
| L_cum | density | -4.0398 | 7.2930 | 0.5824 |
| L_cum | λ₂ | 1.9681 | 3.9560 | 0.6213 |
| L_event | density | -2.0772 | 4.4745 | 0.6448 |
| L_event | λ₂ | 0.6867 | 2.0812 | 0.7430 |

## Outage λ — Step 2 (panel cluster)

| Outcome | β | SE | p |
|---------|---|----|----|
| L_cum | -5.18e-05 | 4.74e-05 | 0.2805 |
| L_event | -1.04e-04 | 3.52e-05 | 0.0049 |

## Outage severity — Step 2 (panel cluster)

| Outcome | β | SE | p |
|---------|---|----|----|
| L_cum | 1.88e-04 | 6.41e-05 | 0.0053 |
| L_event | 1.20e-04 | 3.44e-05 | 0.0011 |

## VIF — outage exposure (Step 2 covariate set)

| Sample | λ VIF | severity VIF | (stale: 2.6, 1.4) |
| 45 pooled means | 2.27 | 1.15 | |
| 270 panel (L_cum) | 2.07 | 1.13 | |
| 270 panel (L_event) | 2.07 | 1.13 | |

**Note:** Do not cross-reference Section 4.3.3 residual corroboration until that section is revised.