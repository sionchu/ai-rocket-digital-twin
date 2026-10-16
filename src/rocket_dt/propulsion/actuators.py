"""Actuator dynamics: commands are requests; these models produce actual states.

- Main engine: ignition flag, throttle clamped to the map range, first-order lag.
- Gimbal: angle and rate limited, per axis.
- RCS: on-time commands per control period, valve delay, minimum on-time
  quantization, per-thruster failure. Per physics step the model returns the
  exact on-fraction of the step so short impulses are not lost to dt.
"""
from __future__ import annotations

import numpy as np

from rocket_dt.propulsion.adapter import ActuatorState
from rocket_dt.schemas import EngineDesign, VehicleConfig


class ActuatorModel:
    def __init__(self, vehicle: VehicleConfig, engine: EngineDesign, throttle0: float = 1.0):
        me = vehicle.layout.main_engine
        self.tmin, self.tmax = engine.throttle_min, engine.throttle_max
        self.tau = me.throttle_time_constant_s
        self.gmax, self.grate = me.max_gimbal_rad, me.max_gimbal_rate_rad_s
        self.rcs = vehicle.layout.rcs
        n = len(self.rcs)
        self.delay = np.array([t.valve_delay_s for t in self.rcs])
        self.min_on = np.array([t.min_on_time_s for t in self.rcs])
        self.failed = np.array([t.failed for t in self.rcs], dtype=bool)
        self.state = ActuatorState(True, float(np.clip(throttle0, self.tmin, self.tmax)), np.zeros(2), np.zeros(n))
        self.throttle_cmd = self.state.throttle
        self.gimbal_cmd = np.zeros(2)
        self.windows: list[list[tuple[float, float]]] = [[] for _ in range(n)]
        self.rcs_impulse_cmd_ns = np.zeros(n)  # bookkeeping for tests

    def command(self, t_now: float, engine_on: bool, throttle: float, gimbal_rad, rcs_on_time_s) -> None:
        self.state.engine_on = bool(engine_on)
        self.throttle_cmd = float(np.clip(throttle, self.tmin, self.tmax))
        self.gimbal_cmd = np.clip(np.asarray(gimbal_rad, float), -self.gmax, self.gmax)
        on = np.asarray(rcs_on_time_s, float) if rcs_on_time_s is not None else np.zeros(len(self.rcs))
        for k, dur in enumerate(on):
            if dur <= 0 or self.failed[k]:
                continue
            dur = 0.0 if dur < 0.5 * self.min_on[k] else max(dur, self.min_on[k])  # min impulse bit
            if dur > 0:
                start = t_now + self.delay[k]
                self.windows[k].append((start, start + dur))
                self.rcs_impulse_cmd_ns[k] += dur * self.rcs[k].max_thrust_n

    def step(self, t: float, dt: float) -> ActuatorState:
        s = self.state
        a = 1.0 - np.exp(-dt / self.tau) if self.tau > 0 else 1.0
        s.throttle = float(np.clip(s.throttle + a * (self.throttle_cmd - s.throttle), self.tmin, self.tmax))
        dg = np.clip(self.gimbal_cmd - s.gimbal_rad, -self.grate * dt, self.grate * dt)
        s.gimbal_rad = np.clip(s.gimbal_rad + dg, -self.gmax, self.gmax)
        frac = np.zeros(len(self.rcs))
        for k, wins in enumerate(self.windows):
            keep = []
            for (t0, t1) in wins:
                overlap = max(0.0, min(t1, t + dt) - max(t0, t))
                frac[k] += overlap / dt
                if t1 > t + dt:
                    keep.append((t0, t1))
            self.windows[k] = keep
        frac[self.failed] = 0.0
        s.rcs_fraction = np.clip(frac, 0.0, 1.0)
        return s
