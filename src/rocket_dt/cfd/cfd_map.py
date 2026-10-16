"""CFD-derived EnginePerformanceMap (same interface as the ideal-1D map).

From one converged inviscid run at nominal Pc:
    F(thr, Pa) = thr * S_nom - Pa * A_exit,  mdot = thr * mdot_nom,  Isp = F / (mdot g0)
valid only while the nozzle flows full; the 1D separation criterion on the
CFD exit pressure (Pe scales with thr) marks invalid cells. Grid uncertainty
(GCI on stream thrust) is stored in provenance.
"""
from __future__ import annotations

import numpy as np

from rocket_dt.dynamics.frames import G0
from rocket_dt.propulsion.nozzle1d import SEPARATION_PE_PA_RATIO
from rocket_dt.propulsion.performance_map import EnginePerformanceMap, VALUE_KEYS
from rocket_dt.schemas import EngineDesign


def build_cfd_map(design: EngineDesign, finest: dict, gci: dict, n_throttle: int = 13,
                  pa_range=(50_000.0, 105_000.0), n_pa: int = 12) -> EnginePerformanceMap:
    from rocket_dt.cfd.post import window_stats
    fin = window_stats(finest["history"]) if len(finest["history"]) > 1 else finest["history"][-1]
    s_nom, mdot_nom = fin["stream_thrust_n"], fin["mdot_out"]
    pe_nom, ae = fin["exit_p_avg_pa"], fin["exit_area_m2"]
    thr = np.linspace(design.throttle_min, design.throttle_max, n_throttle)
    pa = np.linspace(*pa_range, n_pa)
    T, P = np.meshgrid(thr, pa, indexing="ij")
    F = T * s_nom - P * ae
    md = T * mdot_nom
    vals = {"thrust_n": F, "mass_flow_kg_s": md, "isp_s": F / (md * G0)}
    valid = ((T * pe_nom) / P >= SEPARATION_PE_PA_RATIO) & (F > 0)
    prov = {"source": "cfd_rhoCentralFoam_inviscid_wedge", "engine_design_hash": design.content_hash(),
            "cfd_case": finest["case"], "cfd_level": finest["level"], "openfoam_version": finest["openfoam_version"],
            "qa_status": finest["qa"]["status"], "gci_stream_thrust": gci.get("stream_thrust"),
            "assumptions": "inviscid, perfect gas, slip walls, attached flow; Pe-scaling with throttle",
            "nominal": {"stream_thrust_n": s_nom, "mdot_kg_s": mdot_nom, "exit_p_avg_pa": pe_nom, "exit_area_m2": ae}}
    assert set(vals) == set(VALUE_KEYS)
    return EnginePerformanceMap(design.engine_design_id, thr, pa, vals, valid, prov)
