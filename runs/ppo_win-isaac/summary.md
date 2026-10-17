# PPO landing (G4), label `win-isaac`

Environment: Windows 11, RTX 4070 Ti 12 GB (driver 610.88), Isaac Lab 3.0.0 (`v3.0.0-beta2.patch1-5-g418ca31b4`), Isaac Sim 6.0.1-rc.7, PhysX, rsl-rl-lib 5.0.1, torch 2.10.0+cu128, kit python 3.12.13.
Env: `rocket_dt.isaac_env.rocket_env.VtvlEnv` (ADR-0009). Engine conical_A, ideal-1D map (hash b337fd547e15ef5c), fixed mass at full propellant, PhysX foot-sphere contact.
Every number below comes from this PC. The PD baseline numbers come from `cloud-v1912` (Python RK4 reference) and are a **different simulator** (variable mass, penalty contact), so the comparison is indicative, not like-for-like. PD inside the Isaac env was NOT_RUN.

## Training runs (4096 envs, 32 steps/env/iteration, MLP 128x128 ELU, empirical obs normalisation)
| run | start | config | iterations | wall time |
|---|---|---|---|---|
| r0_base | scratch | shaping "basic", lr 3e-4, entropy 0.005, γ 0.995, init std 0.5 | 0–149 | 3,481 s (CFD was running in WSL alongside) |
| r1_pos1 | r0 model_149 | φ_pos 1 m, success bonus 20 | 150–349 | 1,528 s |
| r2_pos2_ent | r0 model_149 | φ_pos 2 m, entropy 0.002, lr 2e-4 | 150–349 | 1,522 s (parallel with r1) |
| r3_guided | r0 model_149 | shaping "guided" (v_h reference = PD horizontal law), φ_pos 2 m, φ_tilt 0.3 rad, success bonus 20 | 150–399 | 1,415 s |

Curves: `curves.png`, `curves.json` (from the console logs). TensorBoard event files and checkpoints are in `train/<run>/`. Commits recorded in the manifests predate a rebase: r0 `4798bbd-dirty` (env code from `6ac8004`, uncommitted then), r1/r2 `5de6026` = `6ac8004`, r3 `337ccb3` = `b2dd5d1`.
VRAM: about 2.1 GB per 4096-env process (nvidia-smi delta). torch peak allocation 227 MiB.

## Checkpoint selection (seeds 1000–1029, deterministic policy, LANDING_LIMITS, 3 s settle)
| checkpoint | success | Wilson 95% | td pos err median/P95 [m] | td \|vz\| P95 | fuel mean [kg] |
|---|---|---|---|---|---|
| r0 model_100 | 0/30 | [0.00, 0.11] | 5.73 / 16.81 | 4.13 | 68.0 |
| r0 model_149 | 0/30 | [0.00, 0.11] | 4.91 / 10.51 | 1.73 | 79.9 |
| r1 model_200 / 250 / 300 / 348 | 3 / 7 / 7 / 5 | best [0.12, 0.41] | 2.27 / 4.63 (300) | 1.58 (300) | 78.5 (300) |
| r2 model_200 / 300 / 348 | 1 / 5 / 0 | best [0.07, 0.34] | 3.46 / 7.78 (300) | 0.91 (300) | 87.1 (300) |
| r3 model_300 / 350 | 23 / 23 | [0.59, 0.88] | 1.43 / 2.46 , 1.14 / 2.32 | 1.12 , 1.38 | 84.2 , 83.7 |
| **r3 model_398** | **29/30** | **[0.83, 0.99]** | **0.64 / 1.43** | **1.97** | **73.8** |

The checkpoint was chosen on these same seeds (selection bias), so it was re-checked on held-out seeds 2000–2029.

## G4 result: r3 model_398 vs PD baseline (conical_A, 1D map)
| | success | Wilson 95% | physics-valid | td pos err median/P95/max [m] | td \|vz\| P95 | td vh P95 | max td tilt | fuel mean/P95 [kg] |
|---|---|---|---|---|---|---|---|---|
| PPO, seeds 1000–1029 (`win-isaac`) | 29/30 | [0.83, 0.99] | 30/30 | 0.64 / 1.43 / 1.58 | 1.97 | 0.75 | 3.2° | 73.8 / 87.0 |
| PPO, held-out seeds 2000–2029 (`win-isaac`) | 28/30 | [0.79, 0.98] | 30/30 | 0.70 / 1.32 / 1.60 | 1.96 | 0.82 | 3.5° | 71.4 / 86.9 |
| PD baseline, seeds 1000–1029 (`cloud-v1912`, Python reference) | 30/30 | [0.89, 1.00] | 30/30 | 0.36 / 0.71 / 0.84 | 0.83 | 0.75 | 6.45° | 78.8 / 93.1 |

Failures (3 of 60): touchdown |vz| 2.03–2.04 m/s (limit 2.0) in two, horizontal speed 1.01 m/s (limit 1.0) in one. The policy lands at a vertical speed right on the limit (P95 1.97), while PD lands with margin (P95 0.83). It uses 6–9% less fuel than PD, but the simulators differ (fixed vs decreasing mass).

## E3: same policy, other engines, no retraining (each engine's own map and mass, seeds 1000–1029)
| engine | success | Wilson 95% | success & physics-valid | td \|vz\| P95 | td pos err median | fuel mean [kg] | PD baseline (`cloud-v1912`) |
|---|---|---|---|---|---|---|---|
| conical_A | 29/30 | [0.83, 0.99] | 29/30 | 1.97 | 0.64 | 73.8 | 30/30 |
| bell_B | 20/30 | [0.49, 0.81] | 20/30 | 2.08 | 0.93 | 68.0 | 30/30 |
| bell_C | 5/30 | [0.07, 0.34] | 0/30 | 1.35 | 2.30 | 99.1 | 30/30 (0/30 physics-valid) |

bell_B: 8 of 10 failures are touchdown |vz| 2.00–2.13 m/s. bell_B's 1D thrust is only 0.2–0.6% different from A's, but the policy has no vertical-speed margin, so it crosses the limit. bell_C: all 25 failures are position error 1.8–3.5 m, and every episode uses separation-risk map cells (2,400–3,400 invalid steps), as PD does.
Conclusion: PPO matches the PD baseline on its training engine but is **not robust to an engine swap**. PD is robust because it inverts the map explicitly (`throttle_for_thrust`).

## Videos
`videos/late_r3_model398.mp4`, `videos/early_r0_model50.mp4`: 3 episodes each (seeds 1000–1002), fixed world camera, Kit RTX capture, every second control step (25 fps, 1280x720). late 3.5 MB (3/3 landed in this run), early 2.6 MB (0/3; touchdown |vz| up to 5.7 m/s, position error about 10 m). The vehicle is small in frame (camera about 100 m away so the whole 80 m descent is visible).
