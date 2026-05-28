"""
LEIS Analysis Tool v4 — Correct VAMAS Parser
==============================================
Root cause of all previous x-axis errors:

The VAMAS block header says:
    abscissa_start     = 100 eV
    abscissa_increment = 0.5 eV
    num_ord_values     = 5602
    num_corresponding_variables = 2  (counts + Transmission)

num_ord_values = 5602 is the TOTAL number of stored values, NOT the number of
data points.  Because num_corresponding_variables = 2, the file stores 2 values
per point — counts and Transmission — interleaved:
    line 0: counts[0], line 1: transmission[0],
    line 2: counts[1], line 3: transmission[1], ...

So real num_pts = 5602 / 2 = 2801, and the x axis is:
    100.0,  100.5,  101.0, ... 1500.0 eV   (NO offset required)

Previous versions read all 5602 lines as single-column data, giving a fake
100–2900.5 eV axis that was 2× too wide and required a spurious ~1238 eV
offset to partially align the Ta peak.
"""

import sys
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QGroupBox,
    QDoubleSpinBox, QCheckBox, QMessageBox, QDialog, QSplitter,
    QScrollArea, QGridLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt

# ── Physical constants ────────────────────────────────────────────────────────
E0   = 1447.0   # He+ beam energy (eV)
M_HE = 4.0026   # He mass (amu)

ATOMIC_MASSES = {
    'C':  12.011, 'N':  14.007, 'O':  15.999,
    'Si': 28.086, 'Nb': 92.906, 'Ta': 180.948,
}

def calc_ke(m2, alpha_deg, e0=E0, m1=M_HE):
    """
    LEIS binary-collision kinematic peak energy for any continuous angle.
      alpha_deg : He+ incidence angle relative to surface normal (degrees)
      theta     = 180 - alpha  (scattering angle)
    """
    theta = np.radians(180.0 - alpha_deg)
    term  = (np.sqrt(max(m2**2 - (m1 * np.sin(theta))**2, 0.0))
             + m1 * np.cos(theta)) / (m1 + m2)
    return e0 * term**2

def theoretical_ke(alpha_deg):
    """Return {element: KE (eV)} computed analytically for any float angle."""
    return {elem: calc_ke(mass, alpha_deg) for elem, mass in ATOMIC_MASSES.items()}

# Approximate ISS RSF for He+ (~1.5 keV) — adjust from known standards if available
DEFAULT_RSF = {
    'C':  1.00, 'N':  1.28, 'O':  1.65,
    'Si': 3.10, 'Nb': 6.20, 'Ta': 9.85,
}

# Default Shirley integration windows (eV) per element
DEFAULT_WINDOWS = {
    'C':  (430, 520),
    'N':  (520, 600),
    'O':  (600, 690),
    'Si': (860, 970),
    'Nb': (1195, 1290),
    'Ta': (1290, 1410),
}

ELEM_COLORS = {
    'C':  '#CC78BC', 'N':  '#0173B2', 'O':  '#DE8F05',
    'Si': '#029E73', 'Nb': '#D55E00', 'Ta': '#CC0000',
}
CYCLE_COLORS = [
    '#0173B2', '#029E73', '#DE8F05', '#CC78BC',
    '#D55E00', '#949494', '#56B4E9', '#E69F00',
]

DISPLAY_LO, DISPLAY_HI = 200.0, 1500.0


# ── Helpers ───────────────────────────────────────────────────────────────────
def shirley_bg(x, y, n_iter=15):
    """Iterative Shirley background between the endpoints of x, y."""
    y = np.array(y, dtype=float)
    I_L = np.mean(y[:3]);  I_R = np.mean(y[-3:])
    bg   = np.full(len(y), I_R)
    diff = I_L - I_R
    for _ in range(n_iter):
        sig = np.maximum(y - bg, 0)
        cum = np.cumsum(sig[::-1])[::-1]
        tot = cum[0]
        if tot == 0: break
        new_bg = I_R + diff * (cum / tot)
        if np.max(np.abs(new_bg - bg)) < 1e-4:
            bg = new_bg; break
        bg = new_bg
    return bg


# ── VAMAS parser ──────────────────────────────────────────────────────────────
class VmsParser:
    """
    Correct VAMAS parser for SpecsLab2 ISS/LEIS files.

    Key insight: num_ord_values (line pc+8) counts ALL stored values across ALL
    ordinate variables.  With num_corresponding_variables = 2 (counts +
    Transmission), the data is interleaved:
        counts[0], transmission[0], counts[1], transmission[1], ...
    Real num_pts = num_ord_values // num_corresponding_variables.
    The x axis is then:  x[i] = abscissa_start + i * abscissa_increment
    which for this instrument gives exactly 100–1500 eV with no offset needed.
    """

    SIGNAL_MODES = {'pulse counting', 'analog', 'pulse_counting', 'pulsecounting'}

    @staticmethod
    def _num_exp_vars(lines):
        try:
            n_c = int(lines[5].strip())
            return int(lines[6 + n_c + 3].strip())
        except Exception:
            return 4

    @staticmethod
    def parse(path):
        with open(path, 'r', encoding='latin-1') as f:
            lines = [l.rstrip('\r\n') for l in f.readlines()]

        nev     = VmsParser._num_exp_vars(lines)
        pc_list = [i for i, l in enumerate(lines)
                   if l.strip().lower() in VmsParser.SIGNAL_MODES]

        cycles = []
        for ci, pc in enumerate(pc_list):
            try:
                start_ke   = float(lines[pc - 7].strip())  # abscissa_start
                step_ke    = float(lines[pc - 6].strip())  # abscissa_increment
                num_vars   = int(lines[pc - 5].strip())    # num_corresponding_variables
                dwell      = float(lines[pc + 1].strip())  # signal_collection_time (s)
                total_vals = int(lines[pc + 8].strip())    # num_ord_values (ALL values)
                num_pts    = total_vals // num_vars         # real data points

                d_start = pc + 9 + nev   # skip experiment variable values
                raw = []
                for k in range(total_vals):
                    try:    raw.append(float(lines[d_start + k].strip()))
                    except: raw.append(0.0)

                counts = np.array(raw[0::num_vars])        # first variable = counts
                # transmission = np.array(raw[1::num_vars])  # available if needed

                ke = np.array([start_ke + i * step_ke for i in range(num_pts)])
                cycles.append({
                    'ke':     ke,
                    'counts': counts,
                    'dwell':  dwell,
                    'label':  f"Cycle {ci + 1}",
                })
            except Exception as e:
                print(f"Warning: skipped cycle at line {pc}: {e}")

        return cycles


# ── Canvas ────────────────────────────────────────────────────────────────────
class Canvas(FigureCanvasQTAgg):
    def __init__(self, w=10, h=7, dpi=100, parent=None):
        fig = Figure(figsize=(w, h), dpi=dpi, tight_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


# ── Per-element region widget (used in Composition dialog) ────────────────────
class ElementRegionWidget(QGroupBox):
    def __init__(self, elem, color, parent=None):
        super().__init__(elem, parent)
        self.elem = elem
        self.setStyleSheet(
            f"QGroupBox::title{{color:{color}; font-weight:bold; font-size:11pt;}}")
        lay = QGridLayout(self)

        lay.addWidget(QLabel("Include:"), 0, 0)
        self.include_cb = QCheckBox(); self.include_cb.setChecked(True)
        lay.addWidget(self.include_cb, 0, 1)

        lay.addWidget(QLabel("RSF:"), 0, 2)
        self.rsf_spin = QDoubleSpinBox()
        self.rsf_spin.setRange(0.01, 99); self.rsf_spin.setDecimals(3)
        self.rsf_spin.setValue(DEFAULT_RSF.get(elem, 1.0))
        lay.addWidget(self.rsf_spin, 0, 3)

        lay.addWidget(QLabel("Start (eV):"), 1, 0)
        self.lo_spin = QDoubleSpinBox()
        self.lo_spin.setRange(100, 1500); self.lo_spin.setDecimals(1)
        lo, hi = DEFAULT_WINDOWS.get(elem, (400, 700))
        self.lo_spin.setValue(lo)
        lay.addWidget(self.lo_spin, 1, 1)

        lay.addWidget(QLabel("End (eV):"), 1, 2)
        self.hi_spin = QDoubleSpinBox()
        self.hi_spin.setRange(100, 1500); self.hi_spin.setDecimals(1)
        self.hi_spin.setValue(hi)
        lay.addWidget(self.hi_spin, 1, 3)

        self.result_label = QLabel("—")
        self.result_label.setStyleSheet(
            f"color:{color}; font-size:10pt; font-weight:bold;")
        lay.addWidget(self.result_label, 2, 0, 1, 4)


# ── Composition Analysis Dialog ───────────────────────────────────────────────
class CompositionDialog(QDialog):
    def __init__(self, cycles, offset=0.0, parent=None):
        super().__init__(parent)
        self.cycles = cycles
        self.offset = offset
        self.setWindowTitle("LEIS Composition Analysis")
        self.resize(1120, 740)
        self._build_ui()
        self._recalculate()

    def _build_ui(self):
        outer = QVBoxLayout(self)

        # Cycle selector
        row = QHBoxLayout()
        row.addWidget(QLabel("<b>Analyse cycle:</b>"))
        self.cycle_cb = QComboBox()
        self.cycle_cb.addItems([c['label'] for c in self.cycles])
        self.cycle_cb.currentIndexChanged.connect(self._recalculate)
        row.addWidget(self.cycle_cb)
        offset_lbl = QLabel(
            f"  (spectrum offset: {self.offset:+.1f} eV — set in main window)")
        offset_lbl.setStyleSheet("color: #666; font-style: italic;")
        row.addWidget(offset_lbl)
        row.addStretch()
        outer.addLayout(row)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # Left: per-element controls
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_w  = QWidget(); left_lay = QVBoxLayout(left_w)
        self.elem_widgets = {}
        for elem, color in ELEM_COLORS.items():
            w = ElementRegionWidget(elem, color)
            w.include_cb.stateChanged.connect(self._recalculate)
            w.rsf_spin.valueChanged.connect(self._recalculate)
            w.lo_spin.valueChanged.connect(self._recalculate)
            w.hi_spin.valueChanged.connect(self._recalculate)
            self.elem_widgets[elem] = w
            left_lay.addWidget(w)
        left_lay.addStretch()
        left_scroll.setWidget(left_w)
        splitter.addWidget(left_scroll)

        # Right: spectrum + table
        right_w  = QWidget(); right_lay = QVBoxLayout(right_w)
        self.canvas = Canvas(w=6, h=5, dpi=90)
        right_lay.addWidget(self.canvas, stretch=2)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Element", "RSF", "Net Area", "Norm. Area", "at%", "Peak (eV)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        right_lay.addWidget(self.table, stretch=1)
        splitter.addWidget(right_w)
        splitter.setSizes([380, 740])

        # Buttons
        btn_row = QHBoxLayout()
        recalc = QPushButton("⟳  Recalculate")
        recalc.setStyleSheet(
            "background:#029E73;color:white;font-weight:bold;padding:9px;")
        recalc.clicked.connect(self._recalculate)
        export = QPushButton("Export CSV")
        export.setStyleSheet("background:#0173B2;color:white;padding:9px;")
        export.clicked.connect(self._export)
        btn_row.addWidget(recalc); btn_row.addWidget(export); btn_row.addStretch()
        outer.addLayout(btn_row)

    def _current_data(self):
        c = self.cycles[self.cycle_cb.currentIndex()]
        return c['ke'] + self.offset, c['counts'] / c['dwell']   # KE (eV) + offset, CPS

    def _recalculate(self):
        x, y = self._current_data()
        results = []
        for elem, w in self.elem_widgets.items():
            if not w.include_cb.isChecked():
                w.result_label.setText("excluded"); continue
            lo, hi = w.lo_spin.value(), w.hi_spin.value()
            rsf    = w.rsf_spin.value()
            mask   = (x >= lo) & (x <= hi)
            if mask.sum() < 5:
                w.result_label.setText("no data in range"); continue
            xw, yw = x[mask], y[mask]
            bg   = shirley_bg(xw, yw)
            net  = np.maximum(yw - bg, 0)
            area = float(np.trapezoid(net, xw))
            peak = xw[np.argmax(net)]
            results.append({'elem': elem, 'rsf': rsf, 'area': area,
                             'norm': area / rsf, 'peak': peak})
            w.result_label.setText(
                f"area = {area:.0f}   peak @ {peak:.1f} eV")

        total = sum(r['norm'] for r in results)
        for r in results:
            r['at_pct'] = 100.0 * r['norm'] / total if total > 0 else 0.0
        self._update_table(results)
        self._update_chart(results, x, y)

    def _update_table(self, results):
        self.table.setRowCount(0)
        for r in sorted(results, key=lambda z: -z['at_pct']):
            row = self.table.rowCount(); self.table.insertRow(row)
            vals = [r['elem'], f"{r['rsf']:.3f}", f"{r['area']:.0f}",
                    f"{r['norm']:.0f}", f"{r['at_pct']:.2f}%", f"{r['peak']:.1f}"]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 4:
                    item.setFont(QFont("Arial", 10, QFont.Bold))
                    s = int(min(r['at_pct'] / 100.0, 1.0) * 160)
                    item.setBackground(QColor(230 - s, 240, 255))
                self.table.setItem(row, col, item)

    def _update_chart(self, results, x_full, y_full):
        ax = self.canvas.ax; ax.clear()
        mask = (x_full >= DISPLAY_LO) & (x_full <= DISPLAY_HI)
        xd, yd = x_full[mask], y_full[mask]
        ax.plot(xd, yd, color='#bbbbbb', lw=1.0, zorder=1)

        for r in results:
            w     = self.elem_widgets[r['elem']]
            lo, hi = w.lo_spin.value(), w.hi_spin.value()
            color  = ELEM_COLORS.get(r['elem'], '#333')
            m      = (x_full >= lo) & (x_full <= hi)
            xw, yw = x_full[m], y_full[m]
            bg     = shirley_bg(xw, yw)
            net    = np.maximum(yw - bg, 0)
            ax.fill_between(xw, bg, bg + net, alpha=0.45, color=color, zorder=2)
            ax.plot(xw, bg, color=color, lw=1.2, ls='--', zorder=3)
            ax.text(r['peak'], (bg + net).max() * 1.03,
                    f"{r['elem']}\n{r['at_pct']:.1f}%",
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color=color, fontfamily='Arial')

        ax.set_xlabel("Kinetic Energy (eV)",
                      fontsize=11, fontfamily='Arial', fontweight='bold')
        ax.set_ylabel("CPS", fontsize=11, fontfamily='Arial', fontweight='bold')
        ax.set_title(f"Shirley BG — {self.cycle_cb.currentText()}",
                     fontsize=12, fontfamily='Arial', fontweight='bold')
        ax.set_xlim(DISPLAY_LO, DISPLAY_HI)
        ax.grid(True, alpha=0.25)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily('Arial'); lbl.set_fontsize(9)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "LEIS_composition.csv", "CSV Files (*.csv)")
        if not path: return
        try:
            with open(path, 'w') as f:
                f.write(f"# {self.cycle_cb.currentText()}\n")
                f.write("Element,RSF,Net Area,Norm. Area,at%,Peak (eV)\n")
                for r in range(self.table.rowCount()):
                    f.write(",".join(
                        self.table.item(r, c).text()
                        for c in range(self.table.columnCount())) + "\n")
            QMessageBox.information(self, "Saved", path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# ── Main window ───────────────────────────────────────────────────────────────
class LEISAnalysisTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LEIS Analysis Tool v4 — TaCN/Nb")
        self.resize(1300, 820)
        self.cycles = []
        self._init_ui()

    def _init_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        outer   = QHBoxLayout(central)

        # Plot
        left = QVBoxLayout()
        self.canvas  = Canvas(w=11, h=7)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        left.addWidget(self.toolbar)
        left.addWidget(self.canvas)
        outer.addLayout(left, 8)

        # Controls
        right = QVBoxLayout(); right.setSpacing(8)

        load_btn = QPushButton("1. Load .vms File")
        load_btn.setStyleSheet("font-weight:bold;padding:10px;font-size:11pt;")
        load_btn.clicked.connect(self._load)
        right.addWidget(load_btn)

        # Kinematics + offset
        ang_grp = QGroupBox("2. Kinematics & Calibration")
        ang_lay = QGridLayout()

        ang_lay.addWidget(QLabel("Incidence angle (°):"), 0, 0)
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(30.0, 70.0)
        self.angle_spin.setDecimals(1)       # continuous, not discrete
        self.angle_spin.setSingleStep(0.5)
        self.angle_spin.setValue(50.0)
        self.angle_spin.setToolTip(
            "He+ incidence angle relative to surface normal (30–70°).\n"
            "Marker positions update continuously using the exact kinematic formula.")
        self.angle_spin.valueChanged.connect(self._update_plot)
        ang_lay.addWidget(self.angle_spin, 0, 1)

        ang_lay.addWidget(QLabel("Energy offset (eV):"), 1, 0)
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-200.0, 200.0)
        self.offset_spin.setDecimals(1)
        self.offset_spin.setSingleStep(1.0)
        self.offset_spin.setValue(0.0)
        self.offset_spin.setToolTip(
            "Shift the displayed spectrum along the KE axis.\n"
            "Use to correct for inelastic losses, charging, or instrumental offsets\n"
            "(the email notes real peaks are typically shifted tens of eV below\n"
            "the ideal kinematic position).\n"
            "Positive value shifts spectrum to higher KE.")
        self.offset_spin.valueChanged.connect(self._update_plot)
        ang_lay.addWidget(self.offset_spin, 1, 1)

        ang_grp.setLayout(ang_lay)
        right.addWidget(ang_grp)

        # Cycle display
        cyc_grp = QGroupBox("3. Cycle Display")
        cyc_lay = QVBoxLayout()
        self.all_cb = QCheckBox("Overlay all cycles")
        self.all_cb.stateChanged.connect(self._update_plot)
        cyc_lay.addWidget(self.all_cb)
        self.cycle_combo = QComboBox()
        self.cycle_combo.currentIndexChanged.connect(self._update_plot)
        cyc_lay.addWidget(QLabel("Single cycle:"))
        cyc_lay.addWidget(self.cycle_combo)
        cyc_grp.setLayout(cyc_lay)
        right.addWidget(cyc_grp)

        # Element markers
        mk_grp = QGroupBox("4. Element Markers")
        mk_lay = QVBoxLayout()
        self.elem_cbs = {}
        for elem, color in ELEM_COLORS.items():
            cb = QCheckBox(f"  {elem}"); cb.setChecked(True)
            cb.setStyleSheet(f"color:{color}; font-weight:bold;")
            cb.stateChanged.connect(self._update_plot)
            self.elem_cbs[elem] = cb; mk_lay.addWidget(cb)
        mk_grp.setLayout(mk_lay)
        right.addWidget(mk_grp)

        # Display
        disp_grp = QGroupBox("5. Display Options")
        disp_lay = QVBoxLayout()
        self.norm_cb = QCheckBox("Normalise to max = 1")
        self.norm_cb.stateChanged.connect(self._update_plot)
        disp_lay.addWidget(self.norm_cb)
        self.theory_cb = QCheckBox("Show theoretical positions")
        self.theory_cb.setChecked(True)
        self.theory_cb.stateChanged.connect(self._update_plot)
        disp_lay.addWidget(self.theory_cb)
        self.uncert_cb = QCheckBox("Show ±5° uncertainty bands")
        self.uncert_cb.stateChanged.connect(self._update_plot)
        disp_lay.addWidget(self.uncert_cb)
        disp_grp.setLayout(disp_lay)
        right.addWidget(disp_grp)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken)
        right.addWidget(sep)

        comp_btn = QPushButton("⚗  Composition Analysis…")
        comp_btn.setStyleSheet(
            "background:#0173B2;color:white;font-weight:bold;"
            "padding:12px;font-size:11pt;border-radius:4px;")
        comp_btn.clicked.connect(self._open_composition)
        right.addWidget(comp_btn)

        exp_btn = QPushButton("Export corrected CSV")
        exp_btn.clicked.connect(self._export_csv)
        right.addWidget(exp_btn)

        right.addStretch()
        outer.addLayout(right, 2)

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open VAMAS File", "", "*.vms;;All Files (*)")
        if not path: return
        try:
            self.cycles = VmsParser.parse(path)
            if not self.cycles:
                with open(path, 'r', encoding='latin-1') as f:
                    lns = [l.rstrip('\r\n') for l in f.readlines()]
                found = [(i, repr(l)) for i, l in enumerate(lns)
                         if any(k in l.lower()
                                for k in ('pulse','analog','counting','signal'))][:8]
                diag = "\n".join(f"  line {i}: {r}" for i, r in found) or "  (none)"
                QMessageBox.warning(self, "Parse error",
                    f"No cycles found.\nSignal-mode candidates:\n{diag}")
                return
            self.cycle_combo.blockSignals(True)
            self.cycle_combo.clear()
            self.cycle_combo.addItems([c['label'] for c in self.cycles])
            self.cycle_combo.blockSignals(False)
            self.cycle_combo.setCurrentIndex(0)
            self._update_plot()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{e}")

    def _update_plot(self):
        if not self.cycles: return
        ax = self.canvas.ax; ax.clear()
        alpha    = self.angle_spin.value()          # float, e.g. 50.3
        offset   = self.offset_spin.value()         # eV shift applied to displayed KE
        show_all = self.all_cb.isChecked()
        to_plot  = (list(range(len(self.cycles)))
                    if show_all else [max(0, self.cycle_combo.currentIndex())])

        for ci in to_plot:
            c  = self.cycles[ci]
            x  = c['ke'] + offset                  # apply user offset
            y  = c['counts'] / c['dwell']
            mask = (x >= DISPLAY_LO) & (x <= DISPLAY_HI)
            xd, yd = x[mask], y[mask]
            if self.norm_cb.isChecked() and yd.max() > 0:
                yd = yd / yd.max()
            color = CYCLE_COLORS[ci % len(CYCLE_COLORS)]
            alpha_plot = 0.85 if len(to_plot) == 1 else max(0.4, 1.0 - ci * 0.12)
            ax.plot(xd, yd, color=color, lw=1.6, alpha=alpha_plot, label=c['label'])

        # Marker positions computed analytically for the exact angle chosen
        theory    = theoretical_ke(alpha)
        theory_lo = theoretical_ke(max(alpha - 5.0, 30.0))  # clamped to valid range
        theory_hi = theoretical_ke(min(alpha + 5.0, 70.0))
        ymax = ax.get_ylim()[1] or 1

        for elem, cb in self.elem_cbs.items():
            if not cb.isChecked(): continue
            pos   = theory.get(elem)
            color = ELEM_COLORS[elem]
            if pos is None: continue
            if self.theory_cb.isChecked():
                ax.axvline(pos, color=color, ls='--', lw=1.1, alpha=0.8)
                ax.text(pos + 4, ymax * 0.93, elem, color=color,
                        fontsize=10, fontweight='bold', rotation=90,
                        va='top', ha='left', fontfamily='Arial')
            if self.uncert_cb.isChecked():
                ax.axvspan(theory_lo.get(elem, pos),
                           theory_hi.get(elem, pos),
                           alpha=0.07, color=color)

        offset_str = (f"  |  spectrum offset {offset:+.1f} eV" if offset != 0 else
                      "  |  no spectrum offset")
        ax.set_xlabel("Kinetic Energy (eV)",
                      fontsize=13, fontfamily='Arial', fontweight='bold')
        ax.set_ylabel(
            "CPS" if not self.norm_cb.isChecked() else "Normalised Intensity",
            fontsize=13, fontfamily='Arial', fontweight='bold')
        ax.set_title(
            f"LEIS — He⁺ {E0:.0f} eV,  {alpha:.1f}° incidence{offset_str}",
            fontsize=12, fontfamily='Arial', fontweight='bold')
        ax.set_xlim(DISPLAY_LO, DISPLAY_HI)
        ax.grid(True, alpha=0.25)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily('Arial'); lbl.set_fontsize(11)
        if len(to_plot) > 1:
            ax.legend(fontsize=10, framealpha=0.85)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def _open_composition(self):
        if not self.cycles:
            QMessageBox.warning(self, "No data", "Load a .vms file first."); return
        dlg = CompositionDialog(self.cycles, offset=self.offset_spin.value(), parent=self)
        dlg.show()

    def _export_csv(self):
        if not self.cycles:
            QMessageBox.warning(self, "No data", "Load a .vms file first."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "LEIS_data.csv", "CSV Files (*.csv)")
        if not path: return
        try:
            headers = ["KE_eV"] + [c['label'] for c in self.cycles]
            ref_ke  = self.cycles[0]['ke']
            mask    = (ref_ke >= DISPLAY_LO) & (ref_ke <= DISPLAY_HI)
            with open(path, 'w') as f:
                f.write(",".join(headers) + "\n")
                for i, ke in enumerate(ref_ke[mask]):
                    row = [f"{ke:.1f}"]
                    for c in self.cycles:
                        idx = np.argmin(np.abs(c['ke'] - ke))
                        row.append(f"{c['counts'][idx]/c['dwell']:.1f}")
                    f.write(",".join(row) + "\n")
            QMessageBox.information(self, "Saved", path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    gui = LEISAnalysisTool()
    gui.show()
    sys.exit(app.exec_())