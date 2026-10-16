from pathlib import Path

import pytest

from rocket_dt.schemas import load_engine, load_vehicle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def vehicle():
    return load_vehicle(ROOT / "configs/vehicle_v0.yaml")


@pytest.fixture(scope="session")
def engines():
    return {n: load_engine(ROOT / f"configs/engines/{n}.yaml") for n in ("conical_A", "bell_B", "bell_C")}
