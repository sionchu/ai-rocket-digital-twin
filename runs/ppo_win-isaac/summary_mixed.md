# Mixed-engine PPO and Isaac PD baseline (label `win-isaac`)

Same Isaac env for every row (VtvlEnv, Isaac Lab 3.0.0 + Isaac Sim 6.0.1 PhysX, ideal-1D maps, fixed mass per ADR-0009, LANDING_LIMITS with 3 s settle, deterministic policy). Each engine is evaluated with its own USD mass properties and its own map.
- **PD (Isaac):** the unchanged `CascadedPD` (G3) drives the env through the same normalised action interface as PPO (`src/rocket_dt/isaac_env/pd_policy.py`). Its internal mass model sees the propellant decreasing, while the Isaac body mass stays fixed.
- **m1:** fine-tuned from r3 model_398. Engine sampled uniformly from {A, B, C} per env at reset. Iterations 400–697, 10,210 s wall time, running alongside other jobs.
- **m2:** the same mix trained from scratch. Iterations 0–499, 2,012 s wall time.
- **Shared m1/m2 settings:** guided shaping (PD horizontal law, disclosed in ADR-0009), and a touchdown margin penalty of 4 per m/s of |vz| above 1.2 m/s.
- **Mass during training:** conical_A's mass properties for all envs (A/B/C differ by ≤ 1.5 kg, 4 mm CoM, 0.4% inertia).
- **Observations:** unchanged, no engine ID. The only engine cue is `engine_available_thrust_norm`.

| controller | engine | seeds 1000–1029 | seeds 2000–2029 | pooled 60 (Wilson 95%) | physics-valid | td \|vz\| P95 | td pos err median / P95 [m] | fuel mean [kg] |
|---|---|---|---|---|---|---|---|---|
| PD (Isaac) | conical_A | 30/30 | 30/30 | 60/60 [0.94, 1.00] | 60/60 | 1.30 | 0.30–0.33 / 0.66–0.68 | 70.3 |
| PD (Isaac) | bell_B | 30/30 | 30/30 | 60/60 [0.94, 1.00] | 60/60 | 1.30 | 0.30–0.33 / 0.66–0.68 | 70.2 |
| PD (Isaac) | bell_C | 30/30 | 30/30 | 60/60 [0.94, 1.00] | 0/60 | 1.32 | 0.30–0.34 / 0.65–0.67 | 72.8 |
| PPO r3 (A only) | conical_A | 29/30 | 28/30 | 57/60 [0.86, 0.98] | 60/60 | 1.97 | 0.64–0.70 / 1.32–1.43 | 72.6 |
| PPO r3 (A only) | bell_B | 20/30 | 22/30 | 42/60 [0.57, 0.80] | 60/60 | 2.08 | 0.85–0.93 / 1.73–2.13 | 66.9 |
| PPO r3 (A only) | bell_C | 5/30 | 4/30 | 9/60 [0.08, 0.26] | 0/60 | 1.39 | 2.10–2.30 / 3.23–3.35 | 98.5 |
| PPO m1 (mixed, ft) | conical_A | 28/30 | 28/30 | 56/60 [0.84, 0.97] | 60/60 | 1.18 | 1.07–1.23 / 1.96–2.02 | 84.9 |
| PPO m1 (mixed, ft) | bell_B | 28/30 | 29/30 | 57/60 [0.86, 0.98] | 60/60 | 1.26 | 0.88–1.16 / 1.91–2.01 | 81.9 |
| PPO m1 (mixed, ft) | bell_C | 29/30 | 28/30 | 57/60 [0.86, 0.98] | 0/60 | 0.94 | 1.14–1.16 / 1.70–2.01 | 90.5 |
| PPO m2 (mixed, scratch) | conical_A | 28/30 | 29/30 | 57/60 [0.86, 0.98] | 60/60 | 0.41 | 0.99–1.00 / 1.95–1.97 | 106.9 |
| PPO m2 (mixed, scratch) | bell_B | 29/30 | 30/30 | 59/60 [0.91, 1.00] | 60/60 | 0.43 | 0.82–0.85 / 1.85–1.88 | 101.3 |
| PPO m2 (mixed, scratch) | bell_C | 26/30 | 24/30 | 50/60 [0.72, 0.91] | 0/60 | 0.50 | 1.38–1.53 / 2.38–2.55 | 105.2 |

Reading:
- PD in the same Isaac simulator lands 180/180 with margin (|vz| P95 1.30 m/s, position P95 0.68 m). It stays the reference: PPO does not beat it on any metric except, for m2, touchdown speed.
- Mixed training removes the engine-swap failure. bell_C goes from 9/60 (r3) to 57/60 (m1), and bell_B from 42/60 to 57/60. The cost is 1/60 on conical_A (57 to 56) and about 12 kg more fuel (84.9 against 72.6 kg).
- The touchdown margin works: |vz| P95 is 0.94–1.26 m/s for m1 and about 0.4–0.5 m/s for m2, against 1.97–2.08 for r3. The remaining m1 failures are position error just over 2 m.
- m2 descends very slowly: 101–107 kg fuel against 70–73 kg for PD. bell_C drops to 50/60. Best by success: m1, which is used for the videos.
- bell_C is 0/60 physics-valid for every controller. It flies on separation-risk cells of its 1D map at sea level (Pe/Pa < 0.4 at throttle < 0.65). Its successes are not physical evidence.

Videos (`videos/`, HUD from `scripts/render_mission_hud.py`, raw renders from `scripts/isaac_mission_video.py`):
- `pd_vs_ppo_m1_conical_A_s1004.mp4`
- `ppo_m1_bell_C_s1004_separation.mp4`
