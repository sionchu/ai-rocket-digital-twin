"""OpenFOAM rhoCentralFoam case generator: 2D axisymmetric wedge, nozzle interior.

Geometry comes from propulsion.geometry.build_contour (the canonical contour).
Domain: chamber inlet -> nozzle exit plane, one cell thick, +/- WEDGE_HALF_DEG
about the x axis, axis faces collapsed (blockMesh drops them, real wedge BCs).
Physics: inviscid (mu = 0), calorically perfect gas with the design's gamma and
molar mass, frozen flow, slip walls. Total pressure / temperature inlet,
supersonic outlet. Because the inviscid perfect-gas solution scales with Pc at
fixed T0 while the nozzle runs full, one run at nominal Pc gives F(throttle, Pa)
for every attached-flow point; separation is NOT modelled (validity rules kept).
Written for ESI OpenFOAM dictionaries (tested against the version recorded in
each run manifest).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rocket_dt.propulsion.geometry import build_contour
from rocket_dt.schemas import EngineDesign

WEDGE_HALF_DEG = 2.5
BASE_CELLS = {"nx_conv": 40, "nx_div": 80, "nr": 24}
REFINEMENT_RATIO = 1.5  # between successive levels (GCI needs r > 1.3)
# KT Courant number in rhoCentralFoam includes c|Sf| on wedge faces, which is
# largest next to the axis although no flux crosses them; 0.4 keeps run time
# practical (ADR-0006). Stability is checked by the QA (finite, converged).
MAX_CO = 0.4
HEADER = """FoamFile
{{
    version 2.0; format ascii; class {cls}; object {obj};
}}
"""


def level_cells(level: int) -> dict:
    f = REFINEMENT_RATIO ** (level - 1)
    return {k: int(round(v * f)) for k, v in BASE_CELLS.items()}


def _pt(x, r, sign):
    a = np.radians(WEDGE_HALF_DEG)
    return f"({x:.9g} {r * np.cos(a):.9g} {sign * r * np.sin(a):.9g})"


def block_mesh_dict(d: EngineDesign, level: int) -> str:
    c = build_contour(d)
    x, r, it = c.x_m, c.r_m, c.throat_index
    n = level_cells(level)
    xs = [x[0], x[it], x[-1]]
    rs = [r[0], r[it], r[-1]]
    verts = []
    for xi, ri in zip(xs, rs):  # per station: axis(back), axis(front) collapsed -> one point each side
        verts += [_pt(xi, 0.0, -1), _pt(xi, ri, -1), _pt(xi, ri, 1), _pt(xi, 0.0, 1)]
    # vertex ids per station s: a_b=4s, w_b=4s+1, w_f=4s+2, a_f=4s+3
    def blk(s, nx):
        a0b, w0b, w0f, a0f = 4 * s, 4 * s + 1, 4 * s + 2, 4 * s + 3
        a1b, w1b, w1f, a1f = a0b + 4, w0b + 4, w0f + 4, a0f + 4
        return f"hex ({a0b} {a1b} {w1b} {w0b} {a0f} {a1f} {w1f} {w0f}) ({nx} {n['nr']} 1) simpleGrading (1 1 1)"

    def edge(s, sign, lo, hi):
        pts = " ".join(_pt(xx, rr, sign) for xx, rr in zip(x[lo + 1:hi], r[lo + 1:hi]))
        v0, v1 = (4 * s + 1, 4 * s + 5) if sign < 0 else (4 * s + 2, 4 * s + 6)
        return f"polyLine {v0} {v1} ({pts})"

    edges = [edge(0, -1, 0, it), edge(0, 1, 0, it), edge(1, -1, it, len(x) - 1), edge(1, 1, it, len(x) - 1)]
    return HEADER.format(cls="dictionary", obj="blockMeshDict") + f"""
scale 1;
vertices
(
    {chr(10).join('    ' + v for v in verts)}
);
blocks
(
    {blk(0, n['nx_conv'])}
    {blk(1, n['nx_div'])}
);
edges
(
    {chr(10).join('    ' + e for e in edges)}
);
boundary
(
    inlet  {{ type patch; faces ((0 3 2 1)); }}
    outlet {{ type patch; faces ((8 9 10 11)); }}
    wall   {{ type wall;  faces ((1 2 6 5) (5 6 10 9)); }}
    front  {{ type wedge; faces ((3 7 6 2) (7 11 10 6)); }}
    back   {{ type wedge; faces ((0 1 5 4) (4 5 9 8)); }}
    axis   {{ type empty; faces ((0 4 7 3) (4 8 11 7)); }}
);
mergePatchPairs ();
"""


def _field(name, cls, dim, internal, bcs):
    body = "\n".join(f"    {p} {{ {v} }}" for p, v in bcs.items())
    return HEADER.format(cls=cls, obj=name) + f"""
dimensions {dim};
internalField uniform {internal};
boundaryField
{{
{body}
    front {{ type wedge; }}
    back  {{ type wedge; }}
    axis  {{ type empty; }}
}}
"""


def write_case(d: EngineDesign, out: Path, level: int = 1, ambient_pressure_pa: float = 101325.0,
               throttle: float = 1.0, end_time_s: float = 3.0e-3,
               write_interval_s: float = 1e-4) -> Path:
    g, R = d.gas.gamma, d.gas.r_specific
    cp = g * R / (g - 1.0)
    pc = d.chamber_pressure_pa * throttle
    tc = d.chamber_temperature_k
    case = Path(out) / f"{d.engine_design_id}_L{level}"
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "constant").mkdir(exist_ok=True)
    (case / "0").mkdir(exist_ok=True)
    (case / "system/blockMeshDict").write_text(block_mesh_dict(d, level))
    wedge_factor = 360.0 / (2 * WEDGE_HALF_DEG)
    (case / "system/controlDict").write_text(HEADER.format(cls="dictionary", obj="controlDict") + f"""
application rhoCentralFoam;
startFrom latestTime; startTime 0; stopAt endTime; endTime {end_time_s};
deltaT 1e-9; writeControl adjustableRunTime; writeInterval {write_interval_s};
purgeWrite 0; writeFormat ascii; writePrecision 10; timeFormat general; timePrecision 8;
runTimeModifiable true; adjustTimeStep yes; maxCo {MAX_CO}; maxDeltaT 1;
// Exit-plane integrals are computed by rocket_dt.cfd.post from the written fields
// (function-object dictionaries hit an IOstream 'sha1' digest error in the v1912 Debian build).
""")
    (case / "system/fvSchemes").write_text(HEADER.format(cls="dictionary", obj="fvSchemes") + """
fluxScheme Kurganov;
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(tauMC) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes
{
    default linear;
    reconstruct(rho) vanLeer;
    reconstruct(U) vanLeerV;
    reconstruct(T) vanLeer;
}
snGradSchemes { default corrected; }
""")
    (case / "system/fvSolution").write_text(HEADER.format(cls="dictionary", obj="fvSolution") + """
solvers
{
    "(rho|rhoU|rhoE)" { solver diagonal; }
    U { solver smoothSolver; smoother GaussSeidel; nSweeps 2; tolerance 1e-09; relTol 0.01; }
    e { $U; tolerance 1e-10; relTol 0; }
}
""")
    (case / "constant/thermophysicalProperties").write_text(
        HEADER.format(cls="dictionary", obj="thermophysicalProperties") + f"""
thermoType
{{
    type hePsiThermo; mixture pureMixture; transport const; thermo hConst;
    equationOfState perfectGas; specie specie; energy sensibleInternalEnergy;
}}
mixture
{{
    specie {{ molWeight {d.gas.molar_mass_kg_mol * 1000.0:.9g}; }}
    thermodynamics {{ Cp {cp:.9g}; Hf 0; }}
    transport {{ mu 0; Pr 1; }}
}}
""")
    (case / "constant/turbulenceProperties").write_text(
        HEADER.format(cls="dictionary", obj="turbulenceProperties") + "\nsimulationType laminar;\n")
    p0_init = 0.05 * pc
    (case / "0/p").write_text(_field("p", "volScalarField", "[1 -1 -2 0 0 0 0]", f"{p0_init:.9g}", {
        "inlet": f"type totalPressure; p0 uniform {pc:.9g}; psi thermo:psi; gamma {g}; value uniform {pc:.9g};",
        "outlet": f"type waveTransmissive; field p; psi thermo:psi; gamma {g}; fieldInf {p0_init:.9g}; lInf 0.05; value uniform {p0_init:.9g};",
        "wall": "type zeroGradient;"}))
    (case / "0/T").write_text(_field("T", "volScalarField", "[0 0 0 1 0 0 0]", f"{tc:.9g}", {
        "inlet": f"type totalTemperature; T0 uniform {tc:.9g}; gamma {g}; psi thermo:psi; value uniform {tc:.9g};",
        "outlet": f"type inletOutlet; inletValue uniform {tc:.9g}; value uniform {tc:.9g};",
        "wall": "type zeroGradient;"}))
    (case / "0/U").write_text(_field("U", "volVectorField", "[0 1 -1 0 0 0 0]", "(0 0 0)", {
        "inlet": "type pressureInletOutletVelocity; value uniform (0 0 0);",
        "outlet": "type zeroGradient;",
        "wall": "type slip;"}))
    meta = {"engine_design_id": d.engine_design_id, "engine_design_hash": d.content_hash(), "level": level,
            "cells": level_cells(level), "refinement_ratio": REFINEMENT_RATIO, "wedge_half_deg": WEDGE_HALF_DEG,
            "wedge_to_full_factor": wedge_factor, "chamber_pressure_pa": pc, "chamber_temperature_k": tc,
            "throttle": throttle, "gamma": g, "r_specific": R, "cp": cp, "model": "inviscid_perfect_gas_slip_walls",
            "max_co": MAX_CO, "end_time_s": end_time_s, "exit_area_m2": d.exit_area_m2, "throat_area_m2": d.throat_area_m2,
            "ambient_pressure_note": "outlet supersonic; Pa enters only in post-processing (Pe - Pa) Ae"}
    (case / "case_meta.json").write_text(json.dumps(meta, indent=1))
    return case
