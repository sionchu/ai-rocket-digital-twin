"""Isaac Lab DirectRLEnv skeleton for the VTVL task (status: NOT_RUN, BLOCKED: no GPU).

Target pin (ADR-0001): Isaac Lab v2.3.2 + Isaac Sim 5.1.0 + PhysX, rsl-rl-lib 3.1.2,
modelled on Isaac-Quadcopter-Direct-v0 (thrust via permanent_wrench_composer).
This file is a structural contract, never executed in this repository so far.
Every API call below must be checked against the installed release before the
first smoke test (scripts/isaac_force_smoke.py); nothing here is verified.

Single-integrator rule: in this env PhysX integrates the rigid body. The Python
reference integrator is used only offline to compare one-step responses.
"""
from __future__ import annotations

try:  # Isaac Lab is not installable here; keep the module importable for docs/tests.
    import torch
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # type: ignore
    ISAAC_AVAILABLE = True
except Exception:  # pragma: no cover
    ISAAC_AVAILABLE = False
    DirectRLEnv = object  # type: ignore
    DirectRLEnvCfg = object  # type: ignore

# Observation contract (design doc C2), fixed order, SI units, body frame unless noted.
OBSERVATION_SCHEMA_V1 = (
    ("relative_pad_position_body", 3), ("linear_velocity_body", 3), ("gravity_vector_body", 3),
    ("angular_velocity_body", 3), ("altitude", 1), ("vertical_velocity", 1), ("normalized_fuel", 1),
    ("engine_available_thrust_norm", 1), ("throttle_actual", 1), ("gimbal_actual", 2), ("previous_action", 7),
)
# Action contract: [throttle, gimbal_x, gimbal_y] in [-1, 1] + 4 RCS duty in [0, 1] (layout gimbal2_rcs4_roll_v0).
ACTION_SCHEMA_V1 = (("throttle", 1), ("gimbal", 2), ("rcs_duty", 4))
OBS_DIM = sum(n for _, n in OBSERVATION_SCHEMA_V1)
ACT_DIM = sum(n for _, n in ACTION_SCHEMA_V1)


class VtvlEnv(DirectRLEnv):  # type: ignore[misc]
    """Sketch of the per-step flow; method bodies outline the contract, not tested code."""

    def _pre_physics_step(self, actions):
        # clamp/scale actions -> ActuatorModel commands (throttle lag, gimbal rate, RCS min-on)
        raise NotImplementedError("NOT_RUN: implement against installed Isaac Lab release")

    def _apply_action(self):
        # 1) actuator step (dt = sim dt) 2) PropulsionAdapter.thruster_forces per env
        # 3) wrench_bridge.batched_wrench(force_b, pos_b, com_b)
        # 4) self.robot.permanent_wrench_composer.set_forces_and_torques(
        #        forces=F, torques=tau, body_ids=self._body_id, is_global=False)   # positions=None
        # 5) update body mass/CoM/inertia from MassModel when propellant changes (rate-limited)
        raise NotImplementedError("NOT_RUN")

    def _get_observations(self):
        raise NotImplementedError("NOT_RUN")

    def _get_rewards(self):
        raise NotImplementedError("NOT_RUN")

    def _get_dones(self):
        raise NotImplementedError("NOT_RUN")

    def _reset_idx(self, env_ids):
        raise NotImplementedError("NOT_RUN")
