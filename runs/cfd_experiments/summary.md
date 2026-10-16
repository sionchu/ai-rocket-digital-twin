# CFD vs ideal-1D (inviscid wedge CFD, OpenFOAM version in cfd_summary.json)

| engine | thr | Pa [Pa] | F CFD [N] | F 1D [N] | dF | mdot CFD | mdot 1D | dmdot | Isp CFD | Isp 1D | valid CFD/1D |
|---|---|---|---|---|---|---|---|---|---|---|---|
| conical_A | 1.0 | 101325 | 19773 | 19846 | -0.37% | 8.470 | 8.527 | -0.67% | 238.1 | 237.3 | True/True |
| conical_A | 0.6 | 101325 | 11071 | 11113 | -0.38% | 5.082 | 5.116 | -0.67% | 222.1 | 221.5 | True/True |
| conical_A | 1.0 | 50000 | 20778 | 20852 | -0.36% | 8.470 | 8.527 | -0.67% | 250.2 | 249.4 | True/True |
| bell_B | 1.0 | 101325 | 19513 | 19886 | -1.88% | 8.469 | 8.527 | -0.68% | 234.9 | 237.8 | True/True |
| bell_B | 0.6 | 101325 | 10914 | 11137 | -2.00% | 5.081 | 5.116 | -0.68% | 219.0 | 222.0 | True/True |
| bell_B | 1.0 | 50000 | 20517 | 20892 | -1.79% | 8.469 | 8.527 | -0.68% | 247.0 | 249.8 | True/True |

| engine | Pe CFD (area avg) | Pe 1D | GCI stream thrust (fine) | observed order p | asymptotic ratio |
|---|---|---|---|---|---|
| conical_A | 133182 | 131398 | 0.135% | 0.92 | 1.000 |
| bell_B | 136862 | 131398 | 0.089% | 1.51 | 1.001 |

## Flight with 1D map vs CFD map (PD baseline, Python reference integrator)

| map | engine | scenario | result | physics valid | fuel [kg] | mean throttle | pos RMS / td pos err [m] |
|---|---|---|---|---|---|---|---|
| CFD:ideal1d | conical_A | hover | completed | True | 104.06 | 0.6101 | 0.214 |
| CFD:ideal1d | conical_A | landing | landed | True | 77.40 | 0.6401 | 0.289 |
| CFD:ideal1d | bell_B | hover | completed | True | 103.84 | 0.6089 | 0.214 |
| CFD:ideal1d | bell_B | landing | landed | True | 77.24 | 0.6388 | 0.289 |
| CFD:cfd | conical_A | hover | completed | True | 103.71 | 0.6122 | 0.214 |
| CFD:cfd | conical_A | landing | landed | True | 77.14 | 0.6422 | 0.289 |
| CFD:cfd | bell_B | hover | completed | True | 104.88 | 0.6192 | 0.213 |
| CFD:cfd | bell_B | landing | landed | True | 78.02 | 0.6496 | 0.293 |

## Landing suite (seeds 1000+)

| map | engine | success | success & valid | fuel mean [kg] | td pos err P95 [m] |
|---|---|---|---|---|---|
| ideal1d | conical_A | 30/30 | 30/30 | 78.85 | 0.71 |
| ideal1d | bell_B | 30/30 | 30/30 | 78.69 | 0.70 |
| cfd | conical_A | 30/30 | 30/30 | 78.58 | 0.71 |
| cfd | bell_B | 30/30 | 30/30 | 79.47 | 0.71 |
