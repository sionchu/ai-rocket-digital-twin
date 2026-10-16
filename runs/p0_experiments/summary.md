# P0 experiment summary

Label: **ideal-1D map (not CFD-verified), Python RK4 reference integrator, PD baseline**. Commit `d1f964b`.

## E0 open-loop sensitivity (throttle 1.0, gimbal_y 0.035 rad, sea level, full tanks)

| engine | F [N] | mdot [kg/s] | engine mass [kg] | CoM z [m] | Ixx [kg m2] | a_z [m/s2] | alpha_y [rad/s2] | map valid |
|---|---|---|---|---|---|---|---|---|
| conical_A | 19846 | 8.527 | 47.19 | -0.2276 | 4016.0 | 6.6234 | -0.47942 | True |
| bell_B | 19886 | 8.527 | 46.79 | -0.2266 | 4012.0 | 6.6618 | -0.48104 | True |
| bell_C | 19690 | 8.527 | 48.70 | -0.2316 | 4031.1 | 6.4738 | -0.47320 | True |

## E1 (engine-A hardware mass, map swapped) and E2 (full engine swap)

| exp | engine | scenario | result | physics valid | fuel [kg] | mean throttle | pos RMS / td pos err [m] | td vz [m/s] |
|---|---|---|---|---|---|---|---|---|
| E1 | conical_A | hover | completed | True (0 invalid steps) | 104.06 | 0.6101 | 0.214 | - |
| E1 | conical_A | landing | landed | True (0 invalid steps) | 77.40 | 0.6401 | 0.289 | -0.814 |
| E1 | bell_B | hover | completed | True (0 invalid steps) | 103.87 | 0.6090 | 0.214 | - |
| E1 | bell_B | landing | landed | True (0 invalid steps) | 77.26 | 0.6389 | 0.289 | -0.814 |
| E1 | bell_C | hover | completed | False (4000 invalid steps) | 108.65 | 0.6371 | 0.213 | - |
| E1 | bell_C | landing | landed | False (2415 invalid steps) | 80.49 | 0.6657 | 0.296 | -0.830 |
| E2 | conical_A | hover | completed | True (0 invalid steps) | 104.06 | 0.6101 | 0.214 | - |
| E2 | conical_A | landing | landed | True (0 invalid steps) | 77.40 | 0.6401 | 0.289 | -0.814 |
| E2 | bell_B | hover | completed | True (0 invalid steps) | 103.84 | 0.6089 | 0.214 | - |
| E2 | bell_B | landing | landed | True (0 invalid steps) | 77.24 | 0.6388 | 0.289 | -0.814 |
| E2 | bell_C | hover | completed | False (4000 invalid steps) | 108.76 | 0.6377 | 0.214 | - |
| E2 | bell_C | landing | landed | False (2405 invalid steps) | 80.57 | 0.6663 | 0.295 | -0.830 |

## S3 randomized landing suite, PD baseline, nominal (seeds 1000–1029)

| engine | success | Wilson 95% | success & physics-valid | td pos err median/P95/max [m] | td |vz| P95 | td vh P95 | max td tilt [deg] | fuel mean/P95 [kg] | failure reasons |
|---|---|---|---|---|---|---|---|---|---|
| conical_A | 30/30 | [0.89, 1.00] | 30/30 | 0.36/0.71/0.84 | 0.83 | 0.75 | 6.45 | 78.8/93.1 | - |
| bell_B | 30/30 | [0.89, 1.00] | 30/30 | 0.36/0.70/0.84 | 0.83 | 0.76 | 6.45 | 78.7/93.0 | - |
| bell_C | 30/30 | [0.89, 1.00] | 0/30 | 0.37/0.71/0.84 | 0.83 | 0.76 | 6.50 | 82.1/97.1 | - |

## S3 randomized landing suite, PD baseline, rcs_01 failed (seeds 1000–1029)

| engine | success | Wilson 95% | success & physics-valid | td pos err median/P95/max [m] | td |vz| P95 | td vh P95 | max td tilt [deg] | fuel mean/P95 [kg] | failure reasons |
|---|---|---|---|---|---|---|---|---|---|
| conical_A | 19/30 | [0.46, 0.78] | 19/30 | 0.57/2.10/3.62 | 1.16 | 2.50 | 14.47 | 76.5/92.8 | horizontal_speed, position_error, tip_over, tip_over_or_loss_of_control, touchdown_tilt |
| bell_B | 19/30 | [0.46, 0.78] | 19/30 | 0.57/2.10/3.59 | 1.16 | 2.51 | 14.47 | 76.4/92.6 | horizontal_speed, position_error, tip_over, tip_over_or_loss_of_control, touchdown_tilt |
| bell_C | 19/30 | [0.46, 0.78] | 0/30 | 0.57/2.17/3.63 | 1.15 | 2.52 | 14.49 | 79.6/96.6 | horizontal_speed, position_error, tip_over, tip_over_or_loss_of_control, touchdown_tilt |
