"""Auswertung: Einzeldatensatz, Wiederholungsexperimente, Messreihen (E1 bis E6), Zugriff auf das vorgerechnete Raster und alle Verdikte als reine Funktionen."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import prs_algorithm as A
import prs_constants as C
import prs_shapes as SH
from prs_constants import LEV_DIRS
from prs_scenario import VRng, draw_pairs, make_dataset, substream_seeds, true_r

GRID_PATH = Path(__file__).resolve().parent / "data" / "pearson_grid.json"


# --- Einzeldatensatz -------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    family: str = "normal"
    rho: float = C.DEFAULT_RHO
    n: int = C.DEFAULT_N
    lev_count: int = C.DEFAULT_LEV_COUNT
    lev_k: float = C.DEFAULT_LEV_K
    lev_dir: str = "gleichläufig"
    boot_b: int = C.DEFAULT_B
    reps: int = C.DEFAULT_REPS
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    x: tuple[float, ...]
    y: tuple[float, ...]
    lever: tuple[int, ...]
    true_r: float             # wahre Korrelation der erzeugenden Verteilung (ohne Hebelpunkte)
    r: float
    r2: float
    slope: float
    spearman: float
    p: float
    intervals: dict           # Name -> (unten, oben)
    influence: tuple[float, ...]
    top: int                  # Index der einflussreichsten Beobachtung
    covers: dict              # Name -> Intervall enthält die wahre Korrelation


def interval_names() -> tuple[str, ...]:
    return ("Fisher-z", "Perzentil-Bootstrap", "BCa-Bootstrap")


def analyse(s: Settings) -> Analysis:
    ds = make_dataset(s.family, s.n, s.rho, s.seed, s.lev_count, s.lev_k, LEV_DIRS[s.lev_dir])
    x, y = np.array(ds.x), np.array(ds.y)
    r = A.pearson_r(x, y)
    rb = A.bootstrap_replicates(x, y, s.boot_b, VRng(s.seed + 1))
    intervals = {
        "Fisher-z": A.fisher_interval(r, s.n),
        "Perzentil-Bootstrap": A.percentile_interval(rb),
        "BCa-Bootstrap": A.bca_interval(x, y, rb, r),
    }
    infl = A.influence(x, y)
    top = int(np.argmax(np.abs(infl)))
    covers = {k: lo <= ds.true_r <= hi for k, (lo, hi) in intervals.items()}
    return Analysis(s, ds.x, ds.y, ds.lever, ds.true_r, r, r * r, A.slope(x, y), A.spearman_r(x, y), A.t_test_p(r, s.n), intervals,
                    tuple(map(float, infl)), top, covers)


# --- Wiederholungen: Abdeckung der drei Intervalle (E2) ---------------------------------------------------------------------------------------------

def affordable_reps(reps: int, boot_b: int, n: int, budget: int = C.REPS_BUDGET) -> int:
    """So viele Wiederholungen, wie das Budget Wiederholungen · B · n erlaubt (mindestens eine)."""
    return max(1, min(reps, budget // max(1, boot_b * n)))


def coverage_runs(family: str, n: int, rho: float, runs: int, boot_b: int, seed: int) -> dict:
    """Abdeckung und mittlere Breite von Fisher-, Perzentil- und BCa-Intervall über `runs` unabhängige Stichproben (jede mit eigenem Teilstrom)."""
    target = true_r(family, rho)
    hits = {"f": 0, "p": 0, "b": 0}
    width = {"f": 0.0, "p": 0.0, "b": 0.0}
    for sub in substream_seeds(seed, runs):
        rng = VRng(sub)
        X, Y = draw_pairs(family, 1, n, rho, rng)
        x, y = X[0], Y[0]
        r = A.pearson_r(x, y)
        rb = A.bootstrap_replicates(x, y, boot_b, rng)
        for key, (lo, hi) in (("f", A.fisher_interval(r, n)), ("p", A.percentile_interval(rb)), ("b", A.bca_interval(x, y, rb, r))):
            hits[key] += lo <= target <= hi
            width[key] += hi - lo
    return {"runs": runs, "true_r": target, **{k: hits[k] / runs for k in hits}, **{f"w{k}": width[k] / runs for k in width}}


def coverage_experiment(s: Settings) -> dict:
    """E2 live: Wiederholungen bei den aktuellen Einstellungen (ohne Hebelpunkte), im Budget."""
    runs = affordable_reps(s.reps, s.boot_b, s.n)
    out = coverage_runs(s.family, s.n, s.rho, runs, s.boot_b, s.seed + 7)
    out["requested"] = s.reps
    out["boot_b"] = s.boot_b
    return out


# --- E3: das Gesetz hinter dem Versagen ---------------------------------------------------------------------------------------------------------

def law_correlations(nu: float, rho: float, n: int, runs: int, seed: int, chunk: int = 50) -> np.ndarray:
    """`runs` Korrelationen aus Stichproben der Größe n mit t(ν)-Rändern und gemeinsamer Skala (ν = ∞: Normalverteilung), in Blöcken, damit der Speicher klein bleibt."""
    family = "normal" if math.isinf(nu) else f"t{int(nu)}"
    rng = VRng(seed)
    parts = []
    for start in range(0, runs, chunk):
        X, Y = draw_pairs(family, min(chunk, runs - start), n, rho, rng)
        parts.append(A.pearson_rows(X, Y))
    return np.concatenate(parts)


def law_row(rs: np.ndarray, nu: float, rho: float, n: int) -> dict:
    """Aus den Korrelationen: gemessener Varianzfaktor n·Var(r)/(1−ρ²)² und Abdeckung des Fisher-Intervalls, jeweils gegen die Vorhersage."""
    cover = float(np.mean([lo <= rho <= hi for lo, hi in (A.fisher_interval(float(r), n) for r in rs)]))
    return {"nu": nu, "rho": rho, "n": n, "runs": len(rs), "predicted": A.variance_factor(nu), "measured": float(n * rs.var(ddof=1) / (1 - rho * rho) ** 2),
            "cov_predicted": A.predicted_coverage(nu), "cov_measured": cover}


def variance_law(nus=C.LAW_NUS, rho: float = 0.5, n: int = 1000, runs: int = 400, seed: int = 11) -> list[dict]:
    """Live-Fassung von E3: Varianz von r bei t(ν) mit gemeinsamer Skala gegen 1 + κ = (ν − 2)/(ν − 4), und die Fisher-Abdeckung gegen ihre Vorhersage."""
    return [law_row(law_correlations(nu, rho, n, runs, seed * 1000 + int(nu)), nu, rho, n) for nu in nus]


def grid_law_cell(nu: float, rho: float, n: int = C.LAW_N, seeds=C.GRID_SEEDS, runs_per_seed: int = C.LAW_RUNS_PER_SEED) -> dict:
    """Zelle des vorgerechneten Varianzgesetzes: Mittel über die Seeds, Standardabweichung des Varianzfaktors über die Seeds."""
    rows = [law_row(law_correlations(nu, rho, n, runs_per_seed, cell_seed("normal", n, rho, s) + int(0 if math.isinf(nu) else nu) * 17), nu, rho, n) for s in seeds]
    out = dict(rows[0])
    out["runs"] = runs_per_seed * len(seeds)
    out["measured"] = float(np.mean([r["measured"] for r in rows]))
    out["measured_sd"] = float(np.std([r["measured"] for r in rows], ddof=1))
    out["cov_measured"] = float(np.mean([r["cov_measured"] for r in rows]))
    out["nu"] = "inf" if math.isinf(nu) else nu
    return out


# --- E4: ein einzelner Punkt -----------------------------------------------------------------------------------------------------------------

def leverage_grid(ns=C.LEVERAGE_NS, ks=C.LEVERAGE_KS, runs: int = 1000, seed: int = 21) -> list[dict]:
    """Anteil „signifikant“ (t-Test, α = 0,05) und mittleres r, wenn in n unabhängigen Standardnormalpaaren ein Punkt bei (k, k) steht."""
    rows = []
    for n in ns:
        rng = VRng(seed * 10_000 + n)
        X, Y = draw_pairs("normal", runs, n, 0.0, rng)
        for k in ks:
            X2, Y2 = X.copy(), Y.copy()
            if k:
                X2[:, 0], Y2[:, 0] = k, k
            rs = A.pearson_rows(X2, Y2)
            rows.append({"n": n, "k": k, "share": float(A.rejects_rows(rs, n).mean()), "mean_r": float(rs.mean()), "runs": runs})
    return rows


def leverage_threshold(n: int, q: float, runs: int = 500, seed: int = 31, step: float = 0.25, kmax: float = 30.0) -> float | None:
    """Kleinstes k (Schritt 0,25), ab dem in mindestens einem Anteil q der Läufe der t-Test anschlägt; None, wenn bis kmax nicht erreicht."""
    rng = VRng(seed * 10_000 + n)
    X, Y = draw_pairs("normal", runs, n, 0.0, rng)
    for k in np.arange(0.0, kmax + step, step):
        X2, Y2 = X.copy(), Y.copy()
        X2[:, 0], Y2[:, 0] = k, k
        if A.rejects_rows(A.pearson_rows(X2, Y2), n).mean() >= q:
            return float(k)
    return None


# --- E5: unkorreliert ist nicht unabhängig ------------------------------------------------------------------------------------------------------

def size_runs(family: str, n: int, runs: int, seed: int) -> float:
    """Fehlalarmrate des t-Tests bei wahrem ρ = 0 (nur der t-Test; Bootstrap und Permutation stehen im vorgerechneten Raster)."""
    X, Y = draw_pairs(family, runs, n, 0.0, VRng(seed))
    return float(A.rejects_rows(A.pearson_rows(X, Y), n).mean())


def plan_power(rho: float, family: str, runs: int = 2000, seed: int = 41) -> dict:
    """Teststärke beim Planwert n (Fisher-Formel für 80 %); nicht fehlalarmbereinigt, nur zusammen mit der Fehlalarmrate lesbar. Nur Familien mit wahrem r = ρ."""
    if true_r(family, rho) != rho:
        raise ValueError("die Familie hat ein wahres r ungleich ρ")
    n = A.plan_n(rho)
    X, Y = draw_pairs(family, runs, n, rho, VRng(seed * 1000 + n))
    return {"rho": rho, "n": n, "family": family, "power": float(A.rejects_rows(A.pearson_rows(X, Y), n).mean()), "runs": runs}


# --- E1: gleiches r, andere Bilder --------------------------------------------------------------------------------------------------------------

def shapes_table(target: float, seeds=C.SHAPE_SEEDS) -> list[dict]:
    """Je Form: in wie vielen Seeds lösbar, Spearman, Steigung, größter Einfluss einer Beobachtung (Mittel, kleinster, größter Wert) und der erreichbare Bereich (Seed 1)."""
    rows = []
    for key, shape in SH.SHAPES.items():
        sp, sl, inf = [], [], []
        for seed in seeds:
            res = SH.solve_shape(key, target, seed)
            if res is None:
                continue
            x, y = np.array(res.x), np.array(res.y)
            sp.append(A.spearman_r(x, y))
            sl.append(A.slope(x, y))
            inf.append(float(np.abs(A.influence(x, y)).max()))
        lo, hi = SH.reachable_range(key, seeds[0])
        row = {"key": key, "label": shape.label, "solved": len(sp), "of": len(seeds), "range": (lo, hi)}
        if sp:
            row.update(spearman=(float(np.mean(sp)), min(sp), max(sp)), slope=(float(np.mean(sl)), min(sl), max(sl)), influence=(float(np.mean(inf)), min(inf), max(inf)))
        rows.append(row)
    return rows


# --- E6: was bringt r? -----------------------------------------------------------------------------------------------------------------------

def utility_rows(rs=C.UTILITY_RS, n: int = 200_000, seed: int = 5) -> list[dict]:
    """RMSE-Verhältnis und Vorzeichen-Trefferquote an einer Normalverteilung der Größe n, gegen die Formeln √(1−r²) und ½ + arcsin(r)/π."""
    rows = []
    for r in rs:
        rng = VRng(seed * 100 + int(round(r * 100)))
        X, Y = draw_pairs("normal", 1, n, r, rng)
        x, y = X[0], Y[0]
        b = A.slope(x, y)
        resid = (y - y.mean()) - b * (x - x.mean())
        rows.append({"r": r, "rmse_measured": float(resid.std() / y.std()), "rmse_formula": A.rmse_ratio(r), "r2": r * r,
                     "sign_measured": float(np.mean((x > 0) == (y > 0))), "sign_formula": A.sign_hit(r)})
    return rows


# --- Zellen des vorgerechneten Rasters (tools/make_grid.py ruft sie auf) ---------------------------------------------------------------------------

def cell_seed(family: str, n: int, rho: float, s: int) -> int:
    """Eigener Seed je Zelle und Plan-Seed s (verschieden für jede Familie, jedes n und ρ)."""
    return s * 1_000_003 + n * 101 + int(round(rho * 10)) * 7 + list(C.ALL_FAMILIES).index(family) * 1013


def grid_coverage_cell(family: str, n: int, rho: float, seeds=C.GRID_SEEDS, runs_per_seed: int | None = None, boot_b: int = C.GRID_BOOT) -> dict:
    """Abdeckung und Breite der drei Intervalle in einer Zelle: Mittel über die Plan-Seeds mit Standardabweichung über die Seeds."""
    rps = runs_per_seed or (1000 if n >= 1000 else 500)
    per = [coverage_runs(family, n, rho, rps, boot_b, cell_seed(family, n, rho, s)) for s in seeds]
    row = {"family": family, "n": n, "rho": rho, "true_r": per[0]["true_r"], "runs": rps * len(seeds), "seeds": len(seeds), "boot_b": boot_b}
    for key, name in (("f", "fisher"), ("p", "perc"), ("b", "bca")):
        vals = [p[key] for p in per]
        row[name] = float(np.mean(vals))
        row[name + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        row["w_" + name] = float(np.mean([p["w" + key] for p in per]))
    return row


def grid_size_cell(family: str, n: int, seeds=C.GRID_SEEDS, t_runs: int = 2000, bp_runs: int = 500, boot_b: int = C.GRID_BOOT, perm_b: int = 199) -> dict:
    """Fehlalarmrate bei wahrem ρ = 0: t-Test (Mittel über die Seeds), Perzentil-Bootstrap (liegt 0 außerhalb?) und Permutationstest (p ≤ 0,05) auf bp_runs Läufen."""
    t_rates = [size_runs(family, n, t_runs, cell_seed(family, n, 0.0, s)) for s in seeds]
    boot = perm = 0
    for sub in substream_seeds(cell_seed(family, n, 0.0, 999), bp_runs):
        rng = VRng(sub)
        X, Y = draw_pairs(family, 1, n, 0.0, rng)
        x, y = X[0], Y[0]
        lo, hi = A.percentile_interval(A.bootstrap_replicates(x, y, boot_b, rng))
        boot += not (lo <= 0.0 <= hi)
        perm += A.permutation_p(x, y, perm_b, rng) <= 0.05
    return {"family": family, "n": n, "t": float(np.mean(t_rates)), "t_sd": float(np.std(t_rates, ddof=1)), "t_runs": t_runs * len(seeds),
            "boot": boot / bp_runs, "perm": perm / bp_runs, "bp_runs": bp_runs}


def grid_probe() -> dict:
    """Kleine feste Zelle, die der Test nachrechnet: ändert sich der Erzeuger, stimmt sie nicht mehr mit dem gespeicherten Raster überein."""
    return coverage_runs("t5", 20, 0.5, 25, 100, 12345)


# --- Vorgerechnetes Raster (tools/make_grid.py) --------------------------------------------------------------------------------------------------

def load_grid(path: Path = GRID_PATH) -> dict | None:
    """Das gespeicherte Raster oder None, wenn die Datei fehlt."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def grid_cell(grid: dict, family: str, n: int, rho: float) -> dict | None:
    for row in grid["coverage"]:
        if row["family"] == family and row["n"] == n and abs(row["rho"] - rho) < 1e-9:
            return row
    return None


def grid_size_row(grid: dict, family: str, n: int) -> dict | None:
    for row in grid["size"]:
        if row["family"] == family and row["n"] == n:
            return row
    return None


def coverage_series(grid: dict, family: str, rho: float) -> list[dict]:
    """Zeilen des Rasters für eine Familie und ein ρ, nach n sortiert."""
    return sorted((r for r in grid["coverage"] if r["family"] == family and abs(r["rho"] - rho) < 1e-9), key=lambda r: r["n"])


# --- Verdikte (reine Funktionen: Art, Text) -----------------------------------------------------------------------------------------------------

def verdict_main(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: Wo liegt die wahre Korrelation, und welche Intervalle fangen sie?"""
    s = a.settings
    fam = C.ALL_FAMILIES[s.family]
    names = interval_names()
    hit = [k for k in names if a.covers[k]]
    miss = [k for k in names if not a.covers[k]]
    base = f"Gemessen r = **{a.r:.2f}** bei n = {s.n}; wahr ist **{a.true_r:.2f}** ({fam})."
    if a.lever:
        base += f" {len(a.lever)} Hebelpunkt(e) bei k = {s.lev_k:g}."
    if not miss:
        return "success", f"{base} Alle drei Intervalle enthalten den wahren Wert. Ein einzelner Datensatz zeigt aber nur einen Treffer; wie oft sie treffen, zeigt erst die Wiederholung unten."
    if not hit:
        return "warning", f"{base} Kein Intervall enthält den wahren Wert. Das kommt bei 95 %-Intervallen in etwa jedem zwanzigsten Datensatz vor, bei schweren Rändern oder Hebelpunkten deutlich öfter."
    return "info", f"{base} Es enthalten ihn {', '.join(hit)}; es verfehlen ihn {', '.join(miss)}."


def verdict_grid_series(series: list[dict], label: str) -> tuple[str, str]:
    """E2 (Raster): Abdeckung der drei Intervalle bei der kleinsten und der größten Stichprobe der Reihe. Warnung, wenn Fisher-z bei der größten unter 90 % liegt."""
    first, last = series[0], series[-1]
    text = (f"{label}, ρ = {last['rho']:g}: Fisher-z deckt bei n = {first['n']} zu **{100 * first['fisher']:.1f} %** und bei n = {last['n']} zu **{100 * last['fisher']:.1f} %**; "
            f"Perzentil-Bootstrap {100 * first['perc']:.1f} % und {100 * last['perc']:.1f} %, BCa {100 * first['bca']:.1f} % und {100 * last['bca']:.1f} %.")
    if last["fisher"] < 0.90:
        return "warning", f"{text} Das Fisher-Intervall hält sein Versprechen von 95 % hier nicht."
    return "success", f"{text} Das Fisher-Intervall hält hier annähernd sein Versprechen."


def verdict_coverage(res: dict) -> tuple[str, str]:
    """E2: hält das Intervall, was es verspricht? Toleranz: drei Standardfehler der Abdeckung."""
    runs = res["runs"]
    se = math.sqrt(0.95 * 0.05 / runs)
    names = {"f": "Fisher-z", "p": "Perzentil-Bootstrap", "b": "BCa-Bootstrap"}
    parts = ", ".join(f"{names[k]} **{100 * res[k]:.1f} %**" for k in names)
    off = [names[k] for k in names if abs(res[k] - 0.95) > 3 * se]
    text = f"Abdeckung der 95 %-Intervalle über {runs} Wiederholungen: {parts}."
    if not off:
        return "success", f"{text} Alle drei liegen innerhalb von drei Standardfehlern (±{100 * 3 * se:.1f} Punkte) um 95 %."
    return "warning", f"{text} Außerhalb von ±{100 * 3 * se:.1f} Punkten um 95 %: {', '.join(off)}."


def verdict_law(rows: list[dict]) -> tuple[str, str]:
    """E3: trifft 1 + κ den gemessenen Varianzfaktor? Genannt wird die größte relative Abweichung über die endlichen Zeilen mit ν ≥ 6 (ν = 5 konvergiert langsam und wird gesondert gezeigt)."""
    finite = [r for r in rows if r["nu"] != "inf" and r["nu"] >= 6 and math.isfinite(r["predicted"])]
    worst = max(finite, key=lambda r: abs(r["measured"] / r["predicted"] - 1))
    dev = abs(worst["measured"] / worst["predicted"] - 1)
    text = (f"Der gemessene Varianzfaktor weicht von (ν − 2)/(ν − 4) um höchstens **{100 * dev:.0f} %** ab (größte Abweichung bei ν = {worst['nu']}: "
            f"{worst['measured']:.2f} gegen {worst['predicted']:.2f}).")
    return ("success" if dev <= 0.15 else "info"), text


def verdict_leverage(rows: list[dict], n: int, k: float) -> tuple[str, str]:
    """E4: Anteil „signifikant“ in der Zelle (n, k)."""
    cell = next(r for r in rows if r["n"] == n and r["k"] == k)
    if k == 0:
        return "info", f"Ohne Hebelpunkt schlägt der Test bei n = {n} in {100 * cell['share']:.1f} % der {cell['runs']} Läufe an (erwartet 5 %)."
    kind = "warning" if cell["share"] >= 0.5 else "info"
    return kind, (f"Ein Punkt bei {k:g} Standardabweichungen macht bei n = {n} in **{100 * cell['share']:.1f} %** der {cell['runs']} Läufe aus Unabhängigkeit einen "
                  f"„signifikanten“ Befund (mittleres r = {cell['mean_r']:.2f}).")


def verdict_size(family: str, n: int, rate: float, runs: int) -> tuple[str, str]:
    """E5: Fehlalarmrate bei wahrem ρ = 0; Toleranz drei Standardfehler um 5 %."""
    se = math.sqrt(0.05 * 0.95 / runs)
    name = C.ALL_FAMILIES[family]
    if rate > 0.05 + 3 * se:
        return "warning", f"Bei wahrem ρ = 0 schlägt der Test ({name}, n = {n}) in **{100 * rate:.1f} %** der {runs} Läufe an statt in 5 %."
    return "success", f"Bei wahrem ρ = 0 schlägt der Test ({name}, n = {n}) in {100 * rate:.1f} % der {runs} Läufe an, innerhalb von drei Standardfehlern um 5 %."


def verdict_shapes(rows: list[dict], target: float) -> tuple[str, str]:
    """E1: Spannweite von Spearman und Steigung über die lösbaren Formen."""
    ok = [r for r in rows if "spearman" in r]
    if len(ok) < 2:
        return "warning", f"Für r = {target:.2f} ist höchstens eine Form erreichbar; Ziel senken."
    sp = [r["spearman"][0] for r in ok]
    sl = [r["slope"][0] for r in ok]
    return "info", (f"Alle {len(ok)} erreichbaren Formen haben r = {target:.2f}, aber ein mittleres Spearman von **{min(sp):.2f} bis {max(sp):.2f}** und eine mittlere Steigung "
                    f"von **{min(sl):.2f} bis {max(sl):.2f}**.")


def verdict_utility(rows: list[dict]) -> tuple[str, str]:
    """E6: was bringt r = 0,5? Der Fehler sinkt um 1 − √(1 − r²)."""
    row = min(rows, key=lambda r: abs(r["r"] - 0.5))
    cut = 100 * (1 - row["rmse_measured"])
    return "info", (f"Bei r = {row['r']:.2f} erklärt die Gerade zwar {100 * row['r2']:.0f} % der Varianz, verkleinert aber den Vorhersagefehler (in den Einheiten von y) nur um "
                    f"**{cut:.0f} %** (Formel: {100 * (1 - row['rmse_formula']):.0f} %); das Vorzeichen von y stimmt in {100 * row['sign_measured']:.0f} % der Paare.")
