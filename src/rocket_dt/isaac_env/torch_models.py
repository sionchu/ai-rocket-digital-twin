"""Batched torch mirrors of the propulsion models for the Isaac Lab env (one row per env).

Each class mirrors a numpy reference and is checked against it in tests/test_torch_models.py:
- TorchPerfMap      <- propulsion.performance_map.EnginePerformanceMap.lookup (bounded bilinear)
- TorchActuators    <- propulsion.actuators.ActuatorModel (throttle lag, gimbal angle/rate limit,
                       RCS valve delay + minimum on-time, exact on-fraction per physics step)
- gimbal_direction  <- propulsion.adapter.gimbal_direction
- isa_pressure      <- dynamics.frames.isa_pressure (no range check; altitude clamped to >= 0 by callers)
Out-of-box map queries cannot raise on the GPU: they are clamped to the box and reported as invalid.
"""
from __future__ import annotations

import numpy as np
import torch

from rocket_dt.dynamics.frames import G0
from rocket_dt.propulsion.performance_map import EnginePerformanceMap
from rocket_dt.schemas import EngineDesign, VehicleConfig


def isa_pressure(alt_m: torch.Tensor) -> torch.Tensor:
    return 101325.0 * (1.0 - 2.25577e-5 * alt_m) ** 5.25588


def gimbal_direction(gimbal: torch.Tensor) -> torch.Tensor:
    """(N, 2) gimbal (delta_x, delta_y) -> (N, 3) unit thrust direction, body frame."""
    cx, sx = torch.cos(gimbal[:, 0]), torch.sin(gimbal[:, 0])
    cy, sy = torch.cos(gimbal[:, 1]), torch.sin(gimbal[:, 1])
    return torch.stack([cx * sy, -sx, cx * cy], dim=-1)


class TorchPerfMap:
    """One map, or several maps on the same (throttle, pa) grid stacked so each env can use its own (idx)."""

    def __init__(self, m: EnginePerformanceMap | list[EnginePerformanceMap], device):
        ms = m if isinstance(m, list) else [m]
        for o in ms[1:]:
            if not (np.array_equal(o.throttle, ms[0].throttle) and np.array_equal(o.ambient_pressure_pa, ms[0].ambient_pressure_pa)):
                raise ValueError("stacked maps must share the throttle / ambient pressure grid")
        f = dict(dtype=torch.float32, device=device)
        self.thr = torch.tensor(ms[0].throttle, **f)
        self.pa = torch.tensor(ms[0].ambient_pressure_pa, **f)
        self.F = torch.tensor(np.stack([o.values["thrust_n"] for o in ms]), **f)
        self.mdot = torch.tensor(np.stack([o.values["mass_flow_kg_s"] for o in ms]), **f)
        self.valid = torch.tensor(np.stack([o.valid for o in ms]), dtype=torch.bool, device=device)

    @staticmethod
    def _locate(axis, v):
        inside = (v >= axis[0] - 1e-6) & (v <= axis[-1] + 1e-6)
        vc = v.clamp(axis[0], axis[-1])
        i = (torch.searchsorted(axis, vc.contiguous()) - 1).clamp(0, len(axis) - 2)
        t = ((vc - axis[i]) / (axis[i + 1] - axis[i])).clamp(0.0, 1.0)
        return i, t, inside

    def lookup(self, throttle: torch.Tensor, pa: torch.Tensor, idx: torch.Tensor | None = None):
        """Returns thrust [N], mass flow [kg/s], valid (bool) for each env; idx selects the stacked map."""
        i, s, in_t = self._locate(self.thr, throttle)
        j, t, in_p = self._locate(self.pa, pa)
        k = torch.zeros_like(i) if idx is None else idx
        w = torch.stack([(1 - s) * (1 - t), (1 - s) * t, s * (1 - t), s * t], dim=-1)

        def corners(a):
            return torch.stack([a[k, i, j], a[k, i, j + 1], a[k, i + 1, j], a[k, i + 1, j + 1]], dim=-1)

        f = (w * corners(self.F)).sum(-1)
        md = (w * corners(self.mdot)).sum(-1)
        ok = (corners(self.valid) | (w < 1e-12)).all(-1) & in_t & in_p
        return f, md, ok


class TorchActuators:
    """Vectorised ActuatorModel. RCS windows are kept in two slots (current + previous command),
    which is exact while valve_delay + on_time <= 2 control periods (asserted)."""

    def __init__(self, vehicle: VehicleConfig, engine: EngineDesign, num_envs: int, control_dt: float, device):
        me = vehicle.layout.main_engine
        f = dict(dtype=torch.float32, device=device)
        self.n, self.device = num_envs, device
        self.tmin, self.tmax = engine.throttle_min, engine.throttle_max
        self.tau = me.throttle_time_constant_s
        self.gmax, self.grate = me.max_gimbal_rad, me.max_gimbal_rate_rad_s
        rcs = vehicle.layout.rcs
        self.k = len(rcs)
        self.delay = torch.tensor([t.valve_delay_s for t in rcs], **f)
        self.min_on = torch.tensor([t.min_on_time_s for t in rcs], **f)
        self.failed = torch.tensor([t.failed for t in rcs], dtype=torch.bool, device=device)
        self.fmax = torch.tensor([t.max_thrust_n for t in rcs], **f)
        self.dir_b = torch.tensor([t.force_direction_body for t in rcs], **f)
        self.pos_b = torch.tensor([t.position_body_m for t in rcs], **f)
        self.mdot_full = self.fmax / (torch.tensor([t.isp_s for t in rcs], **f) * G0)
        if float((self.delay + control_dt).max()) > 2 * control_dt + 1e-9:
            raise NotImplementedError("RCS valve delay longer than one control period needs more window slots")
        self.engine_on = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.throttle = torch.zeros(num_envs, **f)
        self.throttle_cmd = torch.zeros(num_envs, **f)
        self.gimbal = torch.zeros(num_envs, 2, **f)
        self.gimbal_cmd = torch.zeros(num_envs, 2, **f)
        self.win = torch.zeros(num_envs, self.k, 2, 2, **f)  # [env, thruster, slot, (start, end)]
        self.frac = torch.zeros(num_envs, self.k, **f)

    def reset(self, ids, throttle0: float):
        self.engine_on[ids] = True
        self.throttle[ids] = min(max(throttle0, self.tmin), self.tmax)
        self.throttle_cmd[ids] = self.throttle[ids]
        self.gimbal[ids] = 0.0
        self.gimbal_cmd[ids] = 0.0
        self.win[ids] = 0.0
        self.frac[ids] = 0.0

    def command(self, t_now: torch.Tensor, throttle: torch.Tensor, gimbal: torch.Tensor, rcs_on_time: torch.Tensor):
        """t_now (N,), throttle (N,), gimbal (N, 2) [rad], rcs_on_time (N, K) [s]. Engine-off envs ignore it."""
        on = self.engine_on
        self.throttle_cmd = torch.where(on, throttle.clamp(self.tmin, self.tmax), torch.full_like(throttle, self.tmin))
        self.gimbal_cmd = torch.where(on[:, None], gimbal.clamp(-self.gmax, self.gmax), torch.zeros_like(gimbal))
        dur = torch.where(rcs_on_time < 0.5 * self.min_on, torch.zeros_like(rcs_on_time),
                          torch.maximum(rcs_on_time, self.min_on))
        dur = torch.where(on[:, None] & ~self.failed[None, :], dur, torch.zeros_like(dur))
        start = t_now[:, None] + self.delay[None, :]
        self.win[:, :, 1] = self.win[:, :, 0]
        self.win[:, :, 0, 0] = start
        self.win[:, :, 0, 1] = torch.where(dur > 0, start + dur, start)

    def step(self, t: torch.Tensor, dt: float):
        a = 1.0 - torch.exp(torch.tensor(-dt / self.tau)) if self.tau > 0 else torch.tensor(1.0)
        self.throttle = (self.throttle + a * (self.throttle_cmd - self.throttle)).clamp(self.tmin, self.tmax)
        dg = (self.gimbal_cmd - self.gimbal).clamp(-self.grate * dt, self.grate * dt)
        self.gimbal = (self.gimbal + dg).clamp(-self.gmax, self.gmax)
        t0, t1 = t[:, None, None], (t + dt)[:, None, None]
        overlap = (torch.minimum(self.win[..., 1], t1) - torch.maximum(self.win[..., 0], t0)).clamp(min=0.0)
        frac = (overlap.sum(-1) / dt).clamp(0.0, 1.0)
        self.frac = torch.where(self.failed[None, :], torch.zeros_like(frac), frac)
        return self.throttle, self.gimbal, self.frac
