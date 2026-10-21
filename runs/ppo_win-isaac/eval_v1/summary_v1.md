| 시나리오 | PD 기본 v0 → v1 | PD 재튜닝 v1 | RL 기본 v0(r3) → v1(v1_nominal) | RL 시나리오 v0 → v1 | v1 실패 사유 (PD 기본 / RL 시나리오) |
|---|---|---|---|---|---|
| nominal | 60 → 60/60 | 60/60 | 57 → 60/60 | - → - | landed 60 / - |
| wind | 43 → 41/60 | 44/60 | 20 → 27/60 | 54 → 56/60 | landed 41, landing_limits 19 / landed 56, landing_limits 4 |
| rcs_fail | 60 → 60/60 | 60/60 | 27 → 26/60 | 57 → 35/60 | landed 60 / landed 35, landing_limits 25 |
| noise | 60 → 60/60 | 60/60 | 60 → 60/60 | 17 → 6/60 | landed 60 / landed 6, landing_limits 54 |
| lateral | 5 → 37/60 | 60/60 | 28 → 13/60 | 52 → 30/60 | landed 37, landing_limits 13, tip_over_or_loss_of_control 10 / landed 30, landing_limits 29, tip_over_or_loss_of_control 1 |

| 배치 (RL, nominal) | v0 | v1 |
|---|---|---|
| tri_diff | 60/60 | 43/60 |
| quad_diff | 60/60 | 17/60 |
| coldgas | 59/60 | 41/60 |

wind 재튜닝 PD try1(`wind_try1_8seeds.json`, 실패 기록): 38/60
