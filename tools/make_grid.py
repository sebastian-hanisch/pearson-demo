"""Rechnet das Raster der Messreihen vor (E2 und E5) und schreibt data/pearson_grid.json.

    python tools/make_grid.py            # volles Raster (7 Familien x 7 Größen x 3 ρ, vier Seeds), mehrere Prozesse
    python tools/make_grid.py --quick    # kleines Probe-Raster für Tests (nicht speichern)

Das Raster ist deterministisch (SplitMix64, feste Seeds); tests/test_grid.py rechnet eine feste Probezelle nach und vergleicht sie mit der Datei."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import prs_constants as C  # noqa: E402
import prs_evaluation as ev  # noqa: E402


def _coverage(args):
    return ev.grid_coverage_cell(*args)


def _size(args):
    return ev.grid_size_cell(*args)


def _law(args):
    nu, rho = args
    return ev.grid_law_cell(float("inf") if nu == "inf" else nu, rho)


def main() -> None:
    quick = "--quick" in sys.argv
    families = list(C.FAMILIES)
    ns = (10, 30) if quick else C.GRID_NS
    cov_tasks = [(f, n, rho) for f in families for n in ns for rho in C.GRID_RHOS]
    size_tasks = [(f, n) for f in C.SIZE_FAMILIES for n in C.SIZE_NS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=14) as pool:
        cov = list(pool.map(_coverage, cov_tasks, chunksize=1))
        size = list(pool.map(_size, size_tasks, chunksize=1))
        law_tasks = [(nu, rho) for nu in (() if quick else (*C.LAW_NUS, "inf")) for rho in C.GRID_RHOS]
        law = list(pool.map(_law, law_tasks, chunksize=1))
    grid = {
        "meta": {"boot_b": C.GRID_BOOT, "seeds": list(C.GRID_SEEDS), "rhos": list(C.GRID_RHOS), "ns": list(ns), "seconds": round(time.time() - t0)},
        "coverage": cov,
        "size": size,
        "law": law,
        "probe": ev.grid_probe(),
    }
    out = ROOT / "data" / ("pearson_grid_quick.json" if quick else "pearson_grid.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out} geschrieben: {len(cov)} Abdeckungszellen, {len(size)} Fehlalarmzellen, {grid['meta']['seconds']} s")


if __name__ == "__main__":
    main()
