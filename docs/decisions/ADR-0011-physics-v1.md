# ADR-0011 Physics v1: aero, variable mass, CFD map, re-tuned PD

- Status: accepted for experiment v1 (2026-10-21). Supersedes the "fixed mass" simplification of ADR-0009 for
  runs flagged `--physics v1`; `--physics v0` reproduces ADR-0009/0010 exactly (all new code is behind flags).
- Why: the ADR-0010 conclusion ("RL beats PD outside PD's envelope") rested on four simplifications the user
  flagged: no aerodynamics, fixed mass, 1D nozzle maps, and a PD that was never re-tuned per scenario.

## 1. Aerodynamics (`src/rocket_dt/dynamics/aero.py`, `configs/aero_v0.yaml`)
- Viscous-crossflow strip theory (Allen & Perkins, NACA TR-1048): 14 strips over the 7 m body (cylinder r 0.6 m
  + 1 m tangent ogive), sectional crossflow Cd 1.2; axial Cd 1.0 tail-first / 0.3 nose-first on the base area.
  The moment about the CoM and the rate damping follow from the strip positions (no separate Cm or Cmq).
- Wind is a velocity with a 1/7 power-law profile referenced to 10 m. The v0 wind force F maps to the speed that
  gives F broadside at sea level: 300 N -> 7.1 m/s, 1500 N -> 15.9 m/s. Gusts scale speed by sqrt(1 + a sin), so
  the force gust matches v0.
- Assumptions, not measured: coefficients are textbook values for a bare cylinder; legs, fins, plume-flow
  interaction, compressibility and the unsteady wake are not modelled.

## 2. Variable mass (`src/rocket_dt/dynamics/varmass.py`, `MassModel.batched`, `MassModel.com_z_rates`)
- Isaac Lab 3.0 does not apply run-time mass changes on the GPU pipeline (ADR-0009), so the PhysX body keeps
  the full-propellant m0, I0, c0 and the env applies a compensating wrench so that the true CoM obeys m a = F and
  the attitude obeys I w_dot + w x I w = tau with the true m(t), I(t), c(t) (derivation in the module docstring,
  including the CoM drift terms 2 w x d_dot + d_ddot). `_state()` reports the true CoM.
- Verified (cloud, `tests/test_varmass_aero.py`): a fixed-mass rigid body driven by the compensated wrench
  reproduces the variable-mass reference integrator over a 6 s, 36 kg burn with tumbling (stress case) to
  1.9 cm (dt 1/200 s; 0.9 cm at 1/400 s, i.e. discretisation), against 1.08 m without compensation.
  Torch == numpy for the new code is tested in `tests/test_torch_models.py` (runs where torch exists: PC).
- Not corrected: contact forces act on the m0/I0 body (post-touchdown settle only; touchdown metrics are
  taken before contact). Initial c0-point velocity ignores the CoM drift rate at reset (~1 cm/s).

## 3. Thrust map
- `--physics v1` uses the CFD map (`runs/cfd/maps/conical_A_cfd.json`, cloud-v1912, L3, QA-passed) for engine A.
  bell_C has no CFD map (separation not resolved), so v1 runs use conical_A only.

## 4. Re-tuned PD (`scripts/tune_pd.py`)
- PDGains now include the descent profile (vz_slope, vz_min, vz_max). Cross-entropy search over 10 gains per
  scenario on the Python reference integrator with physics v1, tuning seeds 5000+, then a held-out check on
  seeds 1000.. and 2000.. (`runs/pd_tuning/<scenario>.json`, label cloud-v1912). The Isaac numbers come from
  `ppo_eval.py --controller pd --physics v1 --pd_gains runs/pd_tuning/<scenario>.json` on the PC.

## 5. Re-run plan (PC, win-isaac)
- RL: nominal fine-tune from r3 under v1, scenario fine-tunes from it, layouts b-d from scratch under v1.
- Evaluation: same 60 seeds (1000-1029, 2000-2029) for default PD, tuned PD, nominal RL, scenario RL.
- Failed runs are reported as they are.
