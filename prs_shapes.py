"""Sechs Punktwolken mit EXAKT gleichem r, aber verschiedener Form (n = 40).

Jede Form hat einen Formparameter t, bei dem r(t) monoton ist; der Zielwert wird per Bisektion getroffen (Genauigkeit ~1e-15). Die Grundfelder sind zwei exakt
unkorrelierte, standardisierte Stichproben (Gram-Schmidt), damit die Formen nicht vom Zufall der Ziehung abhängen. Nicht jedes Ziel ist für jede Form erreichbar
(Bereich je Form über `reachable_range`); die App zeigt dann „nicht erreichbar“ statt eines falschen Bildes."""

import math
from dataclasses import dataclass

import numpy as np

import prs_constants as C
from prs_algorithm import pearson_r
from prs_scenario import VRng


def orth_pair(seed: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Zwei exakt unkorrelierte Stichproben der Länge m mit Mittel 0 und Standardabweichung 1."""
    rng = VRng(seed)
    a = rng.normal(m)
    b = rng.normal(m)
    a = (a - a.mean()) / a.std()
    b = b - b.mean()
    b = b - float(a @ b) / float(a @ a) * a
    b = b / b.std()
    return a, b


def _wolke(t, a, b):
    return a, t * a + b                                       # r = t / √(1 + t²)


def _hebelpunkt(t, a, b):
    return np.append(a, t), np.append(b, t)                   # 39 unkorrelierte Punkte und ein Punkt bei (t, t)


def _gegenausreisser(t, a, b):
    y = 0.97 * a + math.sqrt(1.0 - 0.97**2) * b               # Masse mit r = 0,97 und ein Punkt bei (t, −t)
    return np.append(a, t), np.append(y, -t)


def _zwei_gruppen(t, a, b):
    g = np.r_[np.zeros(len(a) // 2), np.ones(len(a) - len(a) // 2)]
    return a + t * g, b + t * g                               # zwei Gruppen ohne Zusammenhang innen, um t entlang der Diagonale verschoben


def _parabel(t, a, b):
    return a, a * a + t * a + 0.3 * b                         # U-Form mit Schräge: r wächst mit t


def _kurve(t, a, b):
    return a, np.exp(0.5 * a) + t * b                         # monoton gekrümmt, Rauschen t: r fällt mit t


@dataclass(frozen=True)
class Shape:
    label: str
    func: object
    n_base: int          # Länge der Grundfelder (die Hebelformen fügen einen Punkt hinzu)
    t_lo: float
    t_hi: float
    blurb: str


SHAPES = {
    "wolke": Shape("Wolke", _wolke, C.SHAPE_N, 0.0, 50.0, "gewöhnliche elliptische Wolke"),
    "hebelpunkt": Shape("Hebelpunkt", _hebelpunkt, C.SHAPE_N - 1, 0.0, 1e4, "39 unkorrelierte Punkte und ein einzelner Punkt weit draußen"),
    "gegenausreisser": Shape("Gegenausreißer", _gegenausreisser, C.SHAPE_N - 1, 0.0, 1e4, "enger Zusammenhang, den ein einzelner Punkt gegen den Trend senkt"),
    "zwei_gruppen": Shape("Zwei Gruppen", _zwei_gruppen, C.SHAPE_N, 0.0, 1e3, "zwei Gruppen ohne Zusammenhang innen, gegeneinander verschoben"),
    "parabel": Shape("Parabel mit Schräge", _parabel, C.SHAPE_N, 0.0, 1e3, "U-förmige Beziehung mit schwacher Steigung"),
    "kurve": Shape("Gekrümmte Kurve", _kurve, C.SHAPE_N, 1e-6, 1e3, "streng steigend, aber stark gekrümmt (Exponentialkurve mit Rauschen)"),
}


@dataclass(frozen=True)
class ShapeResult:
    key: str
    target: float
    t: float
    x: tuple[float, ...]
    y: tuple[float, ...]
    r: float


def _r_at(shape: Shape, t: float, a, b) -> float:
    x, y = shape.func(t, a, b)
    return pearson_r(x, y)


def reachable_range(key: str, seed: int) -> tuple[float, float]:
    """Kleinster und größter erreichbarer Wert von r für diese Form und diesen Seed (die Enden des Formparameters)."""
    shape = SHAPES[key]
    a, b = orth_pair(seed, shape.n_base)
    r1, r2 = _r_at(shape, shape.t_lo, a, b), _r_at(shape, shape.t_hi, a, b)
    return min(r1, r2), max(r1, r2)


def solve_shape(key: str, target: float, seed: int) -> ShapeResult | None:
    """Punktwolke dieser Form mit r = target; None, wenn das Ziel außerhalb des erreichbaren Bereichs liegt."""
    shape = SHAPES[key]
    a, b = orth_pair(seed, shape.n_base)
    lo, hi = shape.t_lo, shape.t_hi
    r_lo, r_hi = _r_at(shape, lo, a, b), _r_at(shape, hi, a, b)
    if not min(r_lo, r_hi) - 1e-12 <= target <= max(r_lo, r_hi) + 1e-12:
        return None
    increasing = r_hi >= r_lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (_r_at(shape, mid, a, b) < target) == increasing:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    x, y = shape.func(t, a, b)
    return ShapeResult(key, target, t, tuple(map(float, x)), tuple(map(float, y)), pearson_r(x, y))
