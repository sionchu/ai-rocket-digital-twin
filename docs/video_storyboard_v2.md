# Video v2 storyboard — "Can Isaac do Roadster-style thruster control?"

Goal: one narrative video (~3–4 min) instead of separate HUD clips.
Question: the 2026 Tesla Roadster demo uses compressed-gas (cold-gas) thrusters for
attitude/lift. Can thruster-based thrust + attitude control be built and learned in Isaac Sim/Lab?
Every shot carries its env label (cloud-v1912 / wsl-v2606 / win-isaac). Flames are visual proxies.

## Visual style (replaces current HUD)
- No glow, gradients, emoji, rounded "cards", or rainbow palettes. Black or very dark grey background.
- One typeface (Inter or DejaVu Sans), white/grey text, thin 1 px rules, small caps labels.
- A single accent colour (amber) for "active/now"; red only for failure. Numbers tabular-aligned.
- Telemetry as a thin strip at the bottom (T+, ALT, VZ, THR, Pe/Pa), SpaceX-webcast style.
- Per-engine row: three small engine glyphs (A conical / B bell / C bell-short) and RCS ring; the active one lit.

## Sections
| # | Section | Content | Source | Owner |
|---|---|---|---|---|
| 1 | Motivation | Roadster cold-gas thrusters (text card, sources) → the question | text | cloud |
| 2 | Three engines | Contours A/B/C side by side, CFD Mach field each, thrust vs altitude curves, Pe/Pa & separation band | CFD + nozzle1d (cloud-v1912, wsl-v2606 match) | cloud |
| 3 | Isaac setup | force/torque smoke test (errors ≤0.26%), vehicle + RCS layout | win-isaac | cloud (from logs) |
| 4 | Training in parallel | 4×4 tiled env grid at early checkpoints (r0 model_0/50: crashes, tip-overs), then late (r3 398, m1 697) | win-isaac renders | PC |
| 5 | Reward curves | animated curves for r0→r3, m1, m2 from tfevents, annotated with what changed each run | tfevents in repo | cloud |
| 6 | Failure gallery | r3 on bell_C (5/30): crash/tilt/fuel-out clips; bell_C separation; rcs_01 failure | win-isaac | PC renders, cloud edit |
| 7 | Engine swap | same seed, same controller (PD and PPO-m1), A vs B vs C side by side | win-isaac | PC renders |
| 8 | Scoreboard | success per engine/controller (PD 60/60 each; m1 56/57/57; r3 57/42/9), fuel, touchdown vz | eval json | cloud |
| 9 | Answer | Yes in sim; PD still ≥ PPO; bell_C physics-invalid; limits (fixed mass ADR-0009, 1D nozzle, no aero) | ledger | cloud |

## PC render list (raw, no HUD; push mp4 + per-frame telemetry json)
1. `grid_r0_m0`, `grid_r0_m50`, `grid_r3_m398`, `grid_m1_m697`: 16 envs tiled, mixed engines, 10 s each, one fixed overview camera.
2. Failure clips: 3 failed r3 episodes on bell_C (different failure modes), 1 m2 fuel-heavy episode.
3. Engine swap: PD and PPO-m1, seed 1004, conical_A / bell_B / bell_C, chase cam with Flow plume.
