from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ScatteringCondition:
    """Represents a condition for a scattering calculation."""

    energy: float
    theta: float
    phi: float


def parse_scattering_condition(line: str) -> ScatteringCondition:
    energy, theta, phi = map(float, line.split())
    return ScatteringCondition(energy, theta, phi)


def load_scattering_conditions(file: Path) -> list[ScatteringCondition]:
    return [parse_scattering_condition(cond) for cond in file.read_text().splitlines()]
