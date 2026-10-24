"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Ergebnis und Intervalle der Hauptansicht. Kein Gedankenstrich und kein Euro-Zeichen (Kernschriften)."""

from fpdf import FPDF

import prs_constants as C
import prs_evaluation as ev


def _text(value: str) -> str:
    """Zeichen, die die Kernschriften nicht kennen, ersetzen (Gedankenstrich, Pfeile, Fettdruck-Markierung, griechische und mathematische Zeichen)."""
    table = {"–": "-", "—": "-", "→": "->", "**": "", "€": "EUR", "·": "*", "≤": "<=", "≥": ">=", "−": "-", "×": "x", "„": '"', "“": '"', "ρ": "rho", "σ": "sigma",
             "ν": "nu", "κ": "kappa", "²": "^2", "√": "sqrt ", "…": "...", "Φ": "Phi"}
    for old, new in table.items():
        value = value.replace(old, new)
    return value.encode("latin-1", "replace").decode("latin-1")


def _para(pdf: FPDF, height: float, text: str) -> None:
    """Absatz über die volle Breite; danach steht der Cursor wieder am linken Rand."""
    pdf.multi_cell(0, height, _text(text), new_x="LMARGIN", new_y="NEXT")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _text("Pearson: eine Zahl, viele Daten"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, f"Familie: {C.ALL_FAMILIES[s.family]}   rho: {s.rho:.2f}   n: {s.n}   Hebelpunkte: {len(a.lever)}   Zufalls-Seed: {s.seed}")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Ergebnis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, ev.verdict_main(a)[1])
    _para(pdf, 5.5, f"r = {a.r:.3f}, r^2 = {a.r2:.3f}, Steigung = {a.slope:.3f}, Spearman = {a.spearman:.3f}, p (t-Test) = {a.p:.4f}")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "95 %-Intervalle", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for name, (lo, hi) in a.intervals.items():
        _para(pdf, 5.5, f"{name}: [{lo:.3f}; {hi:.3f}]   enthaelt den wahren Wert ({a.true_r:.3f}): {'ja' if a.covers[name] else 'nein'}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    _para(pdf, 4.5, "Simulierte Daten mit bekannter wahrer Korrelation. Sebastian Hanisch - Operations Research und Machine Learning.")
    return bytes(pdf.output())
