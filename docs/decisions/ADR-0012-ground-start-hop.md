# ADR-0012 Ground-start hop with engine ignition (`win-isaac`, physics v1)

- Status: accepted for experiment v1 (2026-10-22). Behind `scenario="hop"`; every other scenario is unchanged
  (regression: PD nominal v1, seeds 1000-1007, same numbers to the last digit as before the change).
- Why: the v0/v1 episodes start in powered descent with the engine already running, so ignition and liftoff never
  happen. The design doc scope includes vertical takeoff, hover and landing.

## Mission
- Start: standing upright on the launch pad at the env origin, at rest, full propellant, engine not yet lit.
- Ignition: the sequencer commands it at t = 0.5 s. Main thrust and propellant flow are 0 until t = 0.8 s, then
  scale with 1 - exp(-(t - 0.8) / 0.2 s). The 0.3 s delay and 0.2 s rise time are assumptions (no engine test data).
- Hover point: height U(15, 25) m above a landing pad at radius U(6, 14) m, random bearing (eval: seeded per
  episode, `scenarios.eval_scenarios("hop")`). Wind as nominal (U(0, 300) N equivalent, ADR-0011).
- Hover hold: the CoM stays within 1.5 m of the hover point at under 1.0 m/s for 1.0 s. Then the target moves to
  the landing pad and the landing logic of ADR-0009 applies. Without the hold the target moves at t = 25 s and the
  episode cannot succeed.
- Touchdown detection is armed once the feet have been above 1.0 m.
- Success = hover hold done and the ADR-0005 landing limits on the second pad.

## Policy interface
- Observation and action spaces are unchanged: `rel` points to the current target (hover point, then pad). So the
  hop policy is a fine-tune of v1_nominal. Reward: the guided potential of ADR-0009 in the landing phase; in the
  hover phase a 3-D version (distance to the hover point, velocity error against the PD-style law capped at 3 m/s,
  tilt, rate). No shaping reward on the step where the target switches.
- PD baseline: the existing CascadedPD in "hover" mode to the hover point, then "descent" to the pad.
