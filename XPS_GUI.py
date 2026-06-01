"""
XPS Analysis Tool V6 - Composition Analysis, lmfit Line Shapes + Fitting,
Doublet Constraints.
"""

import sys
import os
import json
import copy
import re
import numpy as np
import lineshapes

_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
_CNUM = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
_CONSTRAINT_HELP = (
    "Constraint options (one per box):\n"
    "  • Equality / expression:  P1+1.9   0.75*P1   2*P1-P2\n"
    "  • Numeric range:           282:286   or   [282, 286]\n"
    "  • One-sided bound:         >282    <286\n"
    "  • Inequality vs a peak:    >P1    <P1    >P1+0.5\n"
)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QComboBox, QScrollArea, QFileDialog, QMessageBox,
                             QGridLayout, QGroupBox, QDoubleSpinBox, QShortcut,
                             QDialog, QTableWidget, QTableWidgetItem, QTextEdit,
                             QHeaderView, QSplitter, QFrame, QCheckBox)
from PyQt5.QtGui import QKeySequence, QColor, QFont
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

COLORS = {
    'raw': '#0173B2', 'background': '#DE8F05', 'fit': '#029E73',
    'peaks': ['#CC78BC','#CA9161','#949494','#ECE133','#56B4E9',
              '#E69F00','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7','#999999']
}

LINE_SHAPES = list(lineshapes.LINE_SHAPES)

DEFAULT_ASF = {
    'C':0.296,'N':0.477,'O':0.711,'F':1.000,'Si':0.339,'P':0.486,
    'S':0.666,'Cl':0.891,'Ta':3.082,'Hf':2.639,'Ti':2.001,
    'W':3.523,'Al':0.234,'Mo':2.217,'Cr':1.192,'Fe':1.900,
}
REGION_ELEMENT_MAP = {
    'c1s':'C','c 1s':'C','n1s':'N','n 1s':'N','o1s':'O','o 1s':'O',
    'si2p':'Si','si 2p':'Si','ta4f':'Ta','ta 4f':'Ta','hf4f':'Hf',
    'hf 4f':'Hf','ti2p':'Ti','ti 2p':'Ti','w4f':'W','w 4f':'W',
    'al2p':'Al','al 2p':'Al','mo3d':'Mo','mo 3d':'Mo',
    'cr2p':'Cr','cr 2p':'Cr','fe2p':'Fe','fe 2p':'Fe',
}

def guess_element(region_name):
    nl = region_name.lower().replace('_',' ')
    for k, v in REGION_ELEMENT_MAP.items():
        if k in nl: return v
    return None

class UndoManager:
    def __init__(self, limit=15):
        self.limit = limit; self.stack = []; self.is_undoing = False
    def push_state(self, state):
        if self.is_undoing: return
        self.stack.append(copy.deepcopy(state))
        if len(self.stack) > self.limit: self.stack.pop(0)
    def pop_state(self):
        if not self.stack: return None
        self.is_undoing = True
        s = self.stack.pop(); self.is_undoing = False; return s

class XPSPlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.yscale = 1.0          # counts per axis unit of the last plot
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_spectrum(self, data, region_name, background_subtracted=False,
                      background_data=None, peaks_data=None, show_peaks=False,
                      peak_handles=None):
        self.axes.clear()
        sd = sorted(data, key=lambda x: x['be'], reverse=True)
        be_vals = [d['be'] for d in sd]

        raw_vals = ([d.get('countsSubtracted', 0) for d in sd]
                    if background_subtracted else [d['counts'] for d in sd])
        mv = max(raw_vals) if raw_vals else 1
        if mv <= 0: mv = 1
        exp = int(np.floor(np.log10(mv))); exp = max(exp, 0)
        sf = 10**exp
        self.yscale = sf
        ylabel = f'Counts/s (×10$^{{{exp}}}$)' if exp else 'Counts/s'
        sc = lambda arr: [x/sf for x in arr]

        if background_subtracted:
            self.axes.plot(be_vals, sc([d.get('countsSubtracted',0) for d in sd]),
                           'o', color=COLORS['raw'], markersize=4, label='Data', alpha=0.7)
            if show_peaks and peaks_data:
                pcolors = peaks_data.get('colors')
                for idx, peak in enumerate(peaks_data['peaks']):
                    pbe = [p['be'] for p in peak]
                    pc  = sc([p['counts'] for p in peak])
                    col = (pcolors[idx] if pcolors and idx < len(pcolors)
                           else COLORS['peaks'][idx % len(COLORS['peaks'])])
                    self.axes.fill_between(pbe, 0, pc, alpha=0.4, color=col,
                                           label=peaks_data['names'][idx])
                sb = peaks_data['sum']
                self.axes.plot([p['be'] for p in sb],
                               sc([p['counts'] for p in sb]),
                               '--', color='k', linewidth=2.5, label='Total Fit')
        else:
            self.axes.plot(be_vals, sc([d['counts'] for d in sd]),
                           '-', color=COLORS['raw'], linewidth=2.5, label='Raw Data')
            if background_data:
                bgm = {d['be']: d['counts'] for d in background_data}
                bgbe, bgc = [], []
                for be in be_vals:
                    v = bgm.get(be, 0)
                    if v > 0: bgbe.append(be); bgc.append(v/sf)
                if bgbe:
                    self.axes.plot(bgbe, bgc, '-', color=COLORS['background'],
                                   linewidth=2.5, label='Shirley BG')

        fa = {'family':'Arial','fontsize':20,'fontweight':'bold'}
        ft = {'family':'Arial','fontsize':24,'fontweight':'bold'}
        self.axes.set_xlabel('Binding Energy (eV)', **fa)
        self.axes.set_ylabel(ylabel, **fa)
        self.axes.set_title(region_name, **ft, pad=15)
        for lbl in self.axes.get_xticklabels() + self.axes.get_yticklabels():
            lbl.set_fontname('Arial'); lbl.set_fontsize(18)
        self.axes.grid(True, alpha=0.3)
        import matplotlib.font_manager as fm
        self.axes.legend(loc='upper right', prop=fm.FontProperties(family='Arial', size=16),
                         frameon=True, framealpha=0.9)
        self.axes.set_xlim(max(be_vals), min(be_vals))

        # Interactive drag handles (apex circle = move; square = width).
        if peak_handles:
            for h in peak_handles:
                col = h.get('color', '#333')
                if h['role'] == 'move':
                    self.axes.plot([h['x']], [h['y'] / sf], marker='o', ms=11,
                                   mfc='white', mec=col, mew=2.0, zorder=10,
                                   linestyle='none')
                else:
                    self.axes.plot([h['x']], [h['y'] / sf], marker='s', ms=8,
                                   mfc=col, mec='white', mew=1.2, zorder=10,
                                   linestyle='none')
        self.fig.tight_layout(); self.draw()


class PeakWidget(QWidget):
    def __init__(self, peak_number, color, parent=None, callback=None):
        super().__init__(parent)
        self.peak_number = peak_number
        self.color = color
        self.callback = callback
        self.label = ""
        self._build()

    def set_color(self, color):
        """Recolor the widget border + badge (used to give a doublet one color)."""
        self.color = color
        self.setStyleSheet(
            f"border-left:5px solid {color}; background:#f9f9f9;"
            "border-radius:4px; margin-bottom:5px;")
        self.label_badge.setStyleSheet(
            f"background:{color};color:white;font-weight:bold;border-radius:3px;padding:3px;")

    def _build(self):
        layout = QGridLayout()
        self.setStyleSheet(
            f"border-left:5px solid {self.color}; background:#f9f9f9;"
            "border-radius:4px; margin-bottom:5px;")

        self.label_badge = QLabel(self.label or "P?")
        self.label_badge.setAlignment(Qt.AlignCenter)
        self.label_badge.setFixedWidth(40)
        self.label_badge.setStyleSheet(
            f"background:{self.color};color:white;font-weight:bold;border-radius:3px;padding:3px;")
        layout.addWidget(self.label_badge, 0, 0)
        layout.addWidget(QLabel("Name:"), 0, 1)
        self.name_input = QLineEdit(f"Peak {self.peak_number}")
        self.name_input.editingFinished.connect(self._save)
        layout.addWidget(self.name_input, 0, 2, 1, 2)

        def dspin(lo, hi, dec, step, val=0.0):
            w = QDoubleSpinBox()
            w.setRange(lo, hi); w.setDecimals(dec)
            w.setSingleStep(step); w.setValue(val)
            w.editingFinished.connect(self._save); return w

        # Position + Fix Toggle
        self.position_input = dspin(0, 9999999, 4, 0.01)
        layout.addWidget(QLabel("Pos (eV):"), 1, 0)
        layout.addWidget(self.position_input, 1, 1)
        self.pos_constraint = QLineEdit()
        self.pos_constraint.setPlaceholderText("P1+1.9 | 282:286 | >P1")
        self.pos_constraint.editingFinished.connect(self._save)
        layout.addWidget(QLabel("Constraint:"), 1, 2)
        layout.addWidget(self.pos_constraint, 1, 3)
        self.fix_pos = QCheckBox("Fix")
        self.fix_pos.stateChanged.connect(self._save)
        layout.addWidget(self.fix_pos, 1, 4)

        # Area + Fix Toggle
        self.area_input = dspin(0, 9999999, 4, 10)
        layout.addWidget(QLabel("Area:"), 2, 0)
        layout.addWidget(self.area_input, 2, 1)
        self.area_constraint = QLineEdit()
        self.area_constraint.setPlaceholderText("0.75*P1 | >0 | <P1")
        self.area_constraint.editingFinished.connect(self._save)
        layout.addWidget(QLabel("Constraint:"), 2, 2)
        layout.addWidget(self.area_constraint, 2, 3)
        self.fix_area = QCheckBox("Fix")
        self.fix_area.stateChanged.connect(self._save)
        layout.addWidget(self.fix_area, 2, 4)

        # Sigma + Fix Toggle
        self.sigma_input = dspin(0.0001, 9999999, 4, 0.01, 0.5)
        layout.addWidget(QLabel("\u03c3 (eV):"), 3, 0)
        layout.addWidget(self.sigma_input, 3, 1)
        self.sigma_constraint = QLineEdit()
        self.sigma_constraint.setPlaceholderText("P1 | 0.2:0.6 | <P1")
        self.sigma_constraint.editingFinished.connect(self._save)
        layout.addWidget(QLabel("Constraint:"), 3, 2)
        layout.addWidget(self.sigma_constraint, 3, 3)
        self.fix_sigma = QCheckBox("Fix")
        self.fix_sigma.stateChanged.connect(self._save)
        layout.addWidget(self.fix_sigma, 3, 4)

        # Line shape
        layout.addWidget(QLabel("Shape:"), 4, 0)
        self.shape_family = QComboBox()
        for fam in lineshapes.FAMILIES:
            self.shape_family.addItem(lineshapes.FAMILY_LABELS[fam], fam)
        self.shape_family.setCurrentIndex(self.shape_family.findData('V'))
        self.shape_family.currentIndexChanged.connect(self._on_family_changed)
        layout.addWidget(self.shape_family, 4, 1)

        params_holder = QWidget()
        phl = QHBoxLayout(params_holder)
        phl.setContentsMargins(0, 0, 0, 0); phl.setSpacing(4)

        def _spin(lo, hi, dec, step, val):
            sp = QDoubleSpinBox(); sp.setRange(lo, hi); sp.setDecimals(dec)
            sp.setSingleStep(step); sp.setValue(val)
            sp.editingFinished.connect(self._on_param_changed)
            return sp

        # Shape parameters + Fix toggles
        self.gamma_lbl    = QLabel("\u03b3:")
        self.gamma_spin   = _spin(0.0, 10000.0, 4, 0.05, 0.30)
        self.fix_gamma    = QCheckBox("Fix")
        self.fix_gamma.stateChanged.connect(self._save)

        self.fraction_lbl = QLabel("frac:")
        self.fraction_spin= _spin(0.0, 1.0, 3, 0.05, 0.50)
        self.fix_fraction = QCheckBox("Fix")
        self.fix_fraction.stateChanged.connect(self._save)

        self.skew_lbl     = QLabel("skew:")
        self.skew_spin    = _spin(-10.0, 10.0, 3, 0.1, 0.0)
        self.fix_skew     = QCheckBox("Fix")
        self.fix_skew.stateChanged.connect(self._save)

        for w in (self.gamma_lbl, self.gamma_spin, self.fix_gamma, 
                  self.fraction_lbl, self.fraction_spin, self.fix_fraction, 
                  self.skew_lbl, self.skew_spin, self.fix_skew):
            phl.addWidget(w)
        phl.addStretch()
        layout.addWidget(params_holder, 4, 2, 1, 3)
        self._update_shape_params()

        self.readout_lbl = QLabel("FWHM: \u2013    Height: \u2013")
        self.readout_lbl.setStyleSheet("color:#555;font-style:italic;")
        layout.addWidget(self.readout_lbl, 5, 0, 1, 5)

        self.remove_btn = QPushButton("Remove Peak")
        self.remove_btn.setStyleSheet("background:#ffcccc;color:#cc0000;border:none;padding:4px;")
        layout.addWidget(self.remove_btn, 6, 0, 1, 5)
        self.setLayout(layout)
        for sp in (self.position_input, self.area_input, self.sigma_input):
            sp.editingFinished.connect(self.update_readouts)
        self.update_readouts()

    def _on_family_changed(self, *_):
        self._update_shape_params(); self.update_readouts(); self._save()

    def _on_param_changed(self, *_):
        self.update_readouts(); self._save()

    def _current_family(self):
        return self.shape_family.currentData() or 'V'

    def _update_shape_params(self):
        names = lineshapes.extra_param_names(self._current_family())
        
        show_g = 'gamma' in names
        self.gamma_lbl.setVisible(show_g)
        self.gamma_spin.setVisible(show_g)
        self.fix_gamma.setVisible(show_g)

        show_f = 'fraction' in names
        self.fraction_lbl.setVisible(show_f)
        self.fraction_spin.setVisible(show_f)
        self.fix_fraction.setVisible(show_f)

        show_s = 'skew' in names
        self.skew_lbl.setVisible(show_s)
        self.skew_spin.setVisible(show_s)
        self.fix_skew.setVisible(show_s)

    def _extras(self):
        fam = self._current_family()
        names = lineshapes.extra_param_names(fam)
        out = {}
        if 'gamma' in names:    out['gamma'] = self.gamma_spin.value()
        if 'fraction' in names: out['fraction'] = self.fraction_spin.value()
        if 'skew' in names:     out['skew'] = self.skew_spin.value()
        return out

    def _shape_string(self):
        return lineshapes.shape_string(self._current_family(), self._extras())

    def _apply_shape_string(self, s):
        parsed = lineshapes.parse_lineshape(s)
        fam, extras = parsed['family'], parsed['extras']
        guarded = [self.shape_family, self.gamma_spin, self.fraction_spin, self.skew_spin]
        for w in guarded: w.blockSignals(True)
        idx = self.shape_family.findData(fam)
        if idx >= 0: self.shape_family.setCurrentIndex(idx)
        if 'gamma' in extras:    self.gamma_spin.setValue(extras['gamma'])
        if 'fraction' in extras: self.fraction_spin.setValue(extras['fraction'])
        if 'skew' in extras:     self.skew_spin.setValue(extras['skew'])
        for w in guarded: w.blockSignals(False)
        self._update_shape_params()

    def update_readouts(self):
        try:
            fwhm, height = lineshapes.fwhm_height(
                self._shape_string(), self.position_input.value(),
                self.sigma_input.value(), self.area_input.value())
            self.readout_lbl.setText(f"FWHM: {fwhm:.4g} eV    Height: {height:.4g}")
        except Exception:
            self.readout_lbl.setText("FWHM: \u2013    Height: \u2013")

    def set_label(self, text):
        self.label = text
        self.label_badge.setText(text)

    def _save(self):
        if self.callback: self.callback()

    def get_parameters(self):
        return {
            'label': self.label,
            'name': self.name_input.text(),
            'family': self._current_family(),
            'position': self.position_input.value(),
            'positionConstraint': self.pos_constraint.text(),
            'fix_pos': self.fix_pos.isChecked(),
            'area': self.area_input.value(),
            'areaConstraint': self.area_constraint.text(),
            'fix_area': self.fix_area.isChecked(),
            'sigma': self.sigma_input.value(),
            'sigmaConstraint': self.sigma_constraint.text(),
            'fix_sigma': self.fix_sigma.isChecked(),
            'gamma': self.gamma_spin.value(),
            'fix_gamma': self.fix_gamma.isChecked(),
            'fraction': self.fraction_spin.value(),
            'fix_fraction': self.fix_fraction.isChecked(),
            'skew': self.skew_spin.value(),
            'fix_skew': self.fix_skew.isChecked(),
            'lineShape': self._shape_string(),
        }

    def set_parameters(self, p):
        self.name_input.setText(p.get('name', ''))
        self.position_input.setValue(p.get('position', 0.0))
        self.pos_constraint.setText(p.get('positionConstraint', ''))
        self.fix_pos.setChecked(p.get('fix_pos', False))
        
        self.area_input.setValue(p.get('area', 0.0))
        self.area_constraint.setText(p.get('areaConstraint', ''))
        self.fix_area.setChecked(p.get('fix_area', False))
        
        if 'sigma' in p: self.sigma_input.setValue(p['sigma'])
        elif 'fwhm' in p: self.sigma_input.setValue(float(p['fwhm']) / 2.35482)
        
        self.sigma_constraint.setText(p.get('sigmaConstraint', p.get('fwhmConstraint', '')))
        self.fix_sigma.setChecked(p.get('fix_sigma', False))
        
        if p.get('label'): self.set_label(p['label'])
        self._apply_shape_string(p.get('lineShape', 'V(0.3)'))
        
        for key, spin in (('gamma', self.gamma_spin), ('fraction', self.fraction_spin), ('skew', self.skew_spin)):
            if key in p:
                spin.blockSignals(True); spin.setValue(p[key]); spin.blockSignals(False)
                
        self.fix_gamma.setChecked(p.get('fix_gamma', False))
        self.fix_fraction.setChecked(p.get('fix_fraction', False))
        self.fix_skew.setChecked(p.get('fix_skew', False))
        
        self.update_readouts()


class ExcludePeakWidget(QWidget):
    changed = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(2,2,2,2)
        self.name_edit  = QLineEdit("Peak"); self.name_edit.setFixedWidth(70)
        self.pos_spin   = QDoubleSpinBox(); self.pos_spin.setRange(0,9999); self.pos_spin.setDecimals(4); self.pos_spin.setFixedWidth(90)
        self.sigma_spin  = QDoubleSpinBox(); self.sigma_spin.setRange(0.01,99); self.sigma_spin.setDecimals(4); self.sigma_spin.setValue(0.85); self.sigma_spin.setFixedWidth(70)
        self.shape_cb   = QComboBox(); self.shape_cb.setEditable(True); self.shape_cb.addItems(LINE_SHAPES); self.shape_cb.setFixedWidth(110)
        self.area_label = QLabel("0.00"); self.area_label.setFixedWidth(80); self.area_label.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        self.remove_btn = QPushButton("✕"); self.remove_btn.setFixedWidth(26); self.remove_btn.setStyleSheet("color:#c00;font-weight:bold;border:none;")
        for w,lbl in [(self.name_edit,"Name"),(self.pos_spin,"Pos(eV)"),(self.sigma_spin,"\u03c3"),(self.shape_cb,"Shape")]:
            lay.addWidget(QLabel(lbl)); lay.addWidget(w)
        lay.addWidget(QLabel("Area:")); lay.addWidget(self.area_label)
        lay.addWidget(self.remove_btn); lay.addStretch()
        self.pos_spin.valueChanged.connect(self.changed)
        self.sigma_spin.valueChanged.connect(self.changed)
        self.shape_cb.currentIndexChanged.connect(self.changed)

    def get_params(self):
        return {'name':self.name_edit.text(),'position':self.pos_spin.value(),
                'sigma':self.sigma_spin.value(),'lineShape':self.shape_cb.currentText()}
    def set_area_display(self, a): self.area_label.setText(f"{a:.2f}")


class RegionCompositionWidget(QGroupBox):
    recalculate_requested = pyqtSignal()
    def __init__(self, region_name, region_data, calc_fn, parent=None):
        super().__init__(region_name, parent)
        self.region_name  = region_name
        self.region_data  = region_data
        self.calc_fn      = calc_fn
        self.excl_widgets = []
        self._build_ui(); self._detect()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Element:"))
        self.elem_edit = QLineEdit(); self.elem_edit.setFixedWidth(45)
        self.elem_edit.textChanged.connect(self.recalculate_requested)
        r1.addWidget(self.elem_edit)
        r1.addWidget(QLabel("ASF:"))
        self.asf_spin = QDoubleSpinBox(); self.asf_spin.setRange(0.001,99)
        self.asf_spin.setDecimals(4); self.asf_spin.setFixedWidth(80)
        self.asf_spin.valueChanged.connect(self.recalculate_requested)
        r1.addWidget(self.asf_spin)
        r1.addWidget(QLabel("Include:"))
        self.include_cb = QCheckBox(); self.include_cb.setChecked(True)
        self.include_cb.stateChanged.connect(self.recalculate_requested)
        r1.addWidget(self.include_cb); r1.addStretch(); lay.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("BG Start:"))
        self.start_spin = QDoubleSpinBox(); self.start_spin.setRange(0,9999); self.start_spin.setDecimals(4); self.start_spin.setFixedWidth(80); r2.addWidget(self.start_spin)
        r2.addWidget(QLabel("End:"))
        self.end_spin   = QDoubleSpinBox(); self.end_spin.setRange(0,9999);   self.end_spin.setDecimals(4); self.end_spin.setFixedWidth(80); r2.addWidget(self.end_spin)
        rb = QPushButton("Recalc BG"); rb.setFixedWidth(85)
        rb.clicked.connect(self._recalc_bg_emit); r2.addWidget(rb); r2.addStretch(); lay.addLayout(r2)
        eh = QHBoxLayout(); eh.addWidget(QLabel("<b>Exclude peaks:</b>"))
        ae = QPushButton("+ Add"); ae.setFixedWidth(60)
        ae.clicked.connect(self._add_excl); eh.addWidget(ae); eh.addStretch(); lay.addLayout(eh)
        self.excl_container = QVBoxLayout(); lay.addLayout(self.excl_container)
        r3 = QHBoxLayout()
        self.raw_lbl  = QLabel("Raw: \u2014")
        self.excl_lbl = QLabel("Excl: \u2014")
        self.net_lbl  = QLabel("<b>Net: \u2014</b>")
        self.atpct_lbl= QLabel("<b>at%: \u2014</b>")
        self.atpct_lbl.setStyleSheet("color:#0173B2;font-size:13pt;")
        for w in (self.raw_lbl, self.excl_lbl, self.net_lbl, self.atpct_lbl):
            r3.addWidget(w)
        r3.addStretch(); lay.addLayout(r3)

    def _detect(self):
        elem = guess_element(self.region_name)
        if elem:
            self.elem_edit.setText(elem)
            self.asf_spin.setValue(DEFAULT_ASF.get(elem, 1.0))
        else:
            self.elem_edit.setText("?"); self.asf_spin.setValue(1.0)
        data = self.region_data['original']['data']
        be_vals = [d['be'] for d in data]
        if be_vals:
            self.start_spin.setValue(max(be_vals))
            self.end_spin.setValue(min(be_vals))
        if self.region_data.get('backgroundData'):
            active = [d for d in self.region_data['backgroundData'] if d['counts']>0]
            if active:
                abs_ = [d['be'] for d in active]
                self.start_spin.setValue(max(abs_)); self.end_spin.setValue(min(abs_))

    def _add_excl(self):
        w = ExcludePeakWidget()
        w.changed.connect(self._on_excl_changed)
        w.remove_btn.clicked.connect(lambda: self._rm_excl(w))
        self.excl_container.addWidget(w); self.excl_widgets.append(w)
        self.recalculate_requested.emit()

    def _rm_excl(self, w):
        self.excl_widgets = [x for x in self.excl_widgets if x is not w]
        w.deleteLater(); self.recalculate_requested.emit()

    def _on_excl_changed(self): self.recalculate_requested.emit()

    def _recalc_bg_emit(self):
        self._recompute_bg(); self.recalculate_requested.emit()

    def _recompute_bg(self):
        data = self.region_data['original']['data']
        s = max(self.start_spin.value(), self.end_spin.value())
        e = min(self.start_spin.value(), self.end_spin.value())
        p = self.parent()
        while p and not hasattr(p, 'calculate_shirley_background'): p = p.parent()
        if p is None: return
        bg = p.calculate_shirley_background(data, s, e)
        if bg is None: return
        self.region_data['backgroundSubtracted'] = True
        self.region_data['backgroundData'] = bg
        bgm = {b['be']:b['counts'] for b in bg}
        dc = copy.deepcopy(data)
        for d in dc:
            d['countsSubtracted'] = 0 if (d['be']>s or d['be']<e) else max(0, d['counts']-bgm.get(d['be'],0))
        self.region_data['original']['data'] = dc

    def compute_net_area(self):
        elem    = self.elem_edit.text().strip()
        asf     = self.asf_spin.value()
        included= self.include_cb.isChecked()
        data    = self.region_data['original']['data']
        s = max(self.start_spin.value(), self.end_spin.value())
        e = min(self.start_spin.value(), self.end_spin.value())
        if self.region_data.get('backgroundSubtracted'):
            pts = [(d['be'], d.get('countsSubtracted',0)) for d in data if e<=d['be']<=s]
        else:
            pts = [(d['be'], d['counts']) for d in data if e<=d['be']<=s]
        if len(pts) < 2: return 0,0,0,asf,elem,included
        pts.sort(key=lambda x: x[0])
        be_arr  = np.array([p[0] for p in pts])
        cnt_arr = np.array([p[1] for p in pts])
        raw_area = float(_trapz(cnt_arr, be_arr))
        excl_area = 0.0
        for ew in self.excl_widgets:
            p = ew.get_params()
            sig = max(p['sigma'], 1e-6)
            pv = self.calc_fn(be_arr, p['position'], sig, 1.0, p['lineShape'])
            norm = float(_trapz(pv, be_arr))
            if norm > 0:
                mask = np.abs(be_arr-p['position']) <= 6*sig
                if mask.sum() >= 2:
                    scale = (np.dot(cnt_arr[mask], pv[mask]) / np.dot(pv[mask], pv[mask]))
                    scale = max(scale, 0)
                    ea = float(_trapz(scale*pv[mask], be_arr[mask]))
                    ew.set_area_display(ea); excl_area += ea
                else: ew.set_area_display(0.0)
        net = max(raw_area - abs(excl_area), 0.0)
        return raw_area, excl_area, net, asf, elem, included


class CompositionWindow(QDialog):
    def __init__(self, region_data, calc_fn, shirley_fn, parent=None):
        super().__init__(parent)
        self.region_data    = region_data
        self.calc_fn        = calc_fn
        self.shirley_fn     = shirley_fn
        self.region_widgets = {}
        self.setWindowTitle("XPS Composition Analysis")
        self.resize(1100, 820)
        self._build_ui(); self._populate()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        title = QLabel("Atomic Composition Calculator")
        title.setStyleSheet("font-size:17pt;font-weight:bold;color:#0173B2;padding:6px;")
        outer.addWidget(title)
        desc = QLabel("Set element / ASF per region; optionally add exclude-peaks.")
        desc.setWordWrap(True); outer.addWidget(desc)
        splitter = QSplitter(Qt.Horizontal); outer.addWidget(splitter, stretch=1)
        lscroll = QScrollArea(); lscroll.setWidgetResizable(True)
        lscroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._cards_widget = QWidget(); self._cards_layout = QVBoxLayout(self._cards_widget)
        self._cards_layout.addStretch()
        lscroll.setWidget(self._cards_widget); splitter.addWidget(lscroll)
        rw = QWidget(); rl = QVBoxLayout(rw)
        rl.addWidget(QLabel("<b>Composition Results</b>"))
        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(["Region","Element","Net Area","ASF","at%"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setAlternatingRowColors(True)
        rl.addWidget(self.results_table, stretch=1)
        self.bar_canvas = XPSPlotCanvas(self, width=5, height=4, dpi=80)
        rl.addWidget(self.bar_canvas, stretch=1)
        splitter.addWidget(rw); splitter.setSizes([600, 500])
        br = QHBoxLayout()
        recalc = QPushButton("\u27f3  Recalculate All")
        recalc.setStyleSheet("background:#029E73;color:white;font-weight:bold;padding:10px;font-size:12pt;")
        recalc.clicked.connect(self.recalculate_all)
        exp = QPushButton("Export CSV")
        exp.setStyleSheet("background:#0173B2;color:white;padding:10px;font-size:12pt;")
        exp.clicked.connect(self.export_csv)
        br.addWidget(recalc); br.addWidget(exp); br.addStretch()
        outer.addLayout(br)

    def _populate(self):
        for name, data in self.region_data.items():
            w = RegionCompositionWidget(name, data, self.calc_fn, parent=self)
            w.recalculate_requested.connect(self.recalculate_all)
            self._cards_layout.insertWidget(self._cards_layout.count()-1, w)
            self.region_widgets[name] = w
        self.recalculate_all()

    def recalculate_all(self):
        results = []
        for name, w in self.region_widgets.items():
            raw, excl, net, asf, elem, included = w.compute_net_area()
            w.raw_lbl.setText(f"Raw: {raw:.1f}")
            w.excl_lbl.setText(f"Excl: {excl:.1f}")
            w.net_lbl.setText(f"<b>Net: {net:.1f}</b>")
            if included and asf > 0:
                results.append({'region':name,'elem':elem,'net':net,'asf':asf,'norm':net/asf})
            else:
                w.atpct_lbl.setText("<b>at%: excluded</b>")
        total = sum(r['norm'] for r in results)
        self.results_table.setRowCount(0)
        for r in results:
            pct = 100.0*r['norm']/total if total > 0 else 0.0
            self.region_widgets[r['region']].atpct_lbl.setText(f"<b>at%: {pct:.2f}%</b>")
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            for col, txt in enumerate([r['region'], r['elem'],
                                        f"{r['net']:.1f}", f"{r['asf']:.4f}", f"{pct:.2f}%"]):
                item = QTableWidgetItem(txt); item.setTextAlignment(Qt.AlignCenter)
                if col == 4:
                    item.setFont(QFont("Arial",11,QFont.Bold))
                    item.setBackground(QColor(230-int(min(pct/100,1)*180), 240, 255))
                self.results_table.setItem(row, col, item)
        self._draw_bar(results, total)

    def _draw_bar(self, results, total):
        ax = self.bar_canvas.axes; ax.clear()
        if not results or total == 0: self.bar_canvas.draw(); return
        labels  = [f"{r['elem']}\n({r['region']})" for r in results]
        pcts    = [100.0*r['norm']/total for r in results]
        cols    = [COLORS['peaks'][i%len(COLORS['peaks'])] for i in range(len(results))]
        bars = ax.bar(labels, pcts, color=cols, edgecolor='k', linewidth=0.6)
        for bar, v in zip(bars, pcts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                    f"{v:.1f}%", ha='center', va='bottom',
                    fontname='Arial', fontsize=9, fontweight='bold')
        ax.set_ylabel("Atomic %", fontname='Arial', fontsize=11, fontweight='bold')
        ax.set_title("Elemental Composition", fontname='Arial', fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(pcts)*1.18)
        self.bar_canvas.fig.tight_layout(); self.bar_canvas.draw()

    def export_csv(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Export", "composition.csv", "CSV (*.csv)")
        if not fn: return
        try:
            rows = [[self.results_table.item(r,c).text() for c in range(5)] for r in range(self.results_table.rowCount())]
            with open(fn,'w') as f:
                f.write("Region,Element,Net Area,ASF,at%\n")
                for row in rows: f.write(",".join(row)+"\n")
            QMessageBox.information(self,"Exported",f"Saved:\n{fn}")
        except Exception as e: QMessageBox.critical(self,"Error",str(e))


class XPSAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.regions = []; self.region_data = {}
        self.current_region = None
        self.peak_widgets = []
        self.undo_manager = UndoManager(limit=15)
        self.region_templates = {}
        self._switching_region = False
        self.calibration_offset = 0.0
        self.last_dir = os.path.expanduser("~")   # remembered load/save folder
        self.edit_mode = False                    # interactive drag on plot
        self._drag = None
        self._handles = []
        self._build_ui()
        self.canvas.mpl_connect('button_press_event', self.on_plot_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_plot_motion)
        self.canvas.mpl_connect('button_release_event', self.on_plot_release)
        qs = QShortcut(QKeySequence("Ctrl+Z"), self)
        qs.activated.connect(self.undo_action)

    def _build_ui(self):
        self.setWindowTitle('XPS Analysis Tool V6')
        self.resize(1350, 880)
        cw = QWidget(); self.setCentralWidget(cw)
        ml = QHBoxLayout(); cw.setLayout(ml)

        ll = QVBoxLayout()
        tc = QHBoxLayout()
        lb = QPushButton('Load .XY File'); lb.setStyleSheet("font-weight:bold;padding:8px;")
        lb.clicked.connect(self.load_file); tc.addWidget(lb)
        self.region_combo = QComboBox()
        self.region_combo.currentTextChanged.connect(self.change_region)
        tc.addWidget(self.region_combo); tc.addStretch(); ll.addLayout(tc)
        self.canvas  = XPSPlotCanvas(self, width=10, height=8)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        ll.addWidget(self.toolbar); ll.addWidget(self.canvas)
        ml.addLayout(ll, 7)

        rs = QScrollArea(); rs.setWidgetResizable(True); rs.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        rw = QWidget(); rl = QVBoxLayout(); rw.setLayout(rl)

        bg_grp = QGroupBox("1. Background Subtraction")
        bgl = QVBoxLayout(); rgl = QGridLayout()
        rgl.addWidget(QLabel("Start BE (eV):"), 0, 0)
        self.bg_start = QDoubleSpinBox(); self.bg_start.setRange(0,9999999); self.bg_start.setDecimals(4); self.bg_start.setSingleStep(0.0001)
        rgl.addWidget(self.bg_start, 0, 1)
        rgl.addWidget(QLabel("End BE (eV):"), 1, 0)
        self.bg_end   = QDoubleSpinBox(); self.bg_end.setRange(0,9999999); self.bg_end.setDecimals(4); self.bg_end.setSingleStep(0.0001)
        rgl.addWidget(self.bg_end, 1, 1); bgl.addLayout(rgl)
        self.bg_btn = QPushButton("Calculate Background")
        self.bg_btn.setStyleSheet("background:#029E73;color:white;padding:10px;font-weight:bold;")
        self.bg_btn.clicked.connect(self.toggle_background); bgl.addWidget(self.bg_btn)
        bg_grp.setLayout(bgl); rl.addWidget(bg_grp)

        self.peak_grp = QGroupBox("2. Peak Fitting"); self.peak_grp.setEnabled(False)
        pkl = QVBoxLayout()
        
        btn_lay = QHBoxLayout()
        add_btn = QPushButton("+ Add Peak"); add_btn.clicked.connect(self.add_peak_action)
        add_dbl_btn = QPushButton("+ Add Doublet"); add_dbl_btn.clicked.connect(self.add_doublet_action)
        btn_lay.addWidget(add_btn); btn_lay.addWidget(add_dbl_btn)
        pkl.addLayout(btn_lay)
        
        self.peaks_container = QVBoxLayout(); pkl.addLayout(self.peaks_container)
        self.show_btn = QPushButton("Preview Fit"); self.show_btn.setCheckable(True)
        self.show_btn.clicked.connect(self.toggle_show_peaks); pkl.addWidget(self.show_btn)
        self.drag_btn = QPushButton("Drag peaks on plot"); self.drag_btn.setCheckable(True)
        self.drag_btn.setToolTip(
            "When on, drag a peak's apex (circle) to move it (position + height),\n"
            "or drag the square half-max handles to change its width (\u03c3).\n"
            "Only parameters that are free (not fixed, not constrained) can be dragged.")
        self.drag_btn.toggled.connect(self.toggle_edit_mode); pkl.addWidget(self.drag_btn)
        self.weight_chk = QCheckBox("Poisson weighting (\u03c7\u00b2\u22481 for a good fit)")
        self.weight_chk.setChecked(True); pkl.addWidget(self.weight_chk)
        
        fbtn_lay = QHBoxLayout()
        self.fit_btn = QPushButton("Fit (lmfit)")
        self.fit_btn.setStyleSheet("background:#7B3FBF;color:white;padding:10px;font-weight:bold;")
        self.fit_btn.clicked.connect(self.fit_region)
        self.report_btn = QPushButton("Show Fit Report")
        self.report_btn.setEnabled(False)
        self.report_btn.clicked.connect(self.show_fit_report)
        fbtn_lay.addWidget(self.fit_btn); fbtn_lay.addWidget(self.report_btn)
        pkl.addLayout(fbtn_lay)
        
        self.fit_result_lbl = QLabel(""); self.fit_result_lbl.setStyleSheet("color:#333;font-style:italic;")
        self.fit_result_lbl.setWordWrap(True); pkl.addWidget(self.fit_result_lbl)
        self.peak_grp.setLayout(pkl); rl.addWidget(self.peak_grp)

        cal_grp = QGroupBox("3. Energy Calibration")
        cal = QVBoxLayout(); ref = QGridLayout()
        ref.addWidget(QLabel("Peak now at (eV):"), 0, 0)
        self.cal_measured = QDoubleSpinBox(); self.cal_measured.setRange(0, 9999999); self.cal_measured.setDecimals(4); self.cal_measured.setSingleStep(0.01)
        ref.addWidget(self.cal_measured, 0, 1)
        ref.addWidget(QLabel("Should be (eV):"), 1, 0)
        self.cal_target = QDoubleSpinBox(); self.cal_target.setRange(0, 9999999); self.cal_target.setDecimals(4); self.cal_target.setSingleStep(0.01); self.cal_target.setValue(284.8)
        ref.addWidget(self.cal_target, 1, 1); cal.addLayout(ref)
        cal_ref_btn = QPushButton("Calibrate to Reference"); cal_ref_btn.setStyleSheet("background:#0173B2;color:white;padding:8px;font-weight:bold;")
        cal_ref_btn.clicked.connect(self.calibrate_to_reference); cal.addWidget(cal_ref_btn)
        mo = QHBoxLayout()
        mo.addWidget(QLabel("Total offset (eV):"))
        self.cal_offset_spin = QDoubleSpinBox(); self.cal_offset_spin.setRange(-9999, 9999); self.cal_offset_spin.setDecimals(4); self.cal_offset_spin.setSingleStep(0.01)
        mo.addWidget(self.cal_offset_spin)
        cal_apply_btn = QPushButton("Apply"); cal_apply_btn.setFixedWidth(70); cal_apply_btn.clicked.connect(self.apply_manual_offset); mo.addWidget(cal_apply_btn)
        cal_reset_btn = QPushButton("Reset"); cal_reset_btn.setFixedWidth(70); cal_reset_btn.clicked.connect(lambda: self._set_calibration_offset(0.0)); mo.addWidget(cal_reset_btn)
        cal.addLayout(mo)
        self.calib_offset_label = QLabel("Current offset: +0.0000 eV"); self.calib_offset_label.setStyleSheet("font-weight:bold;color:#0173B2;")
        cal.addWidget(self.calib_offset_label); cal_grp.setLayout(cal); rl.addWidget(cal_grp)

        cond_lay = QHBoxLayout()
        save_cond_btn = QPushButton("Save Fit Conditions")
        save_cond_btn.setToolTip("Save the current peaks, constraints and BG window "
                                 "to a .json recipe you can reload for this material.")
        save_cond_btn.clicked.connect(self.save_fit_conditions)
        load_cond_btn = QPushButton("Load Fit Conditions")
        load_cond_btn.setToolTip("Load a previously saved material recipe (.json) "
                                 "and rebuild its peaks here.")
        load_cond_btn.clicked.connect(self.load_fit_conditions)
        cond_lay.addWidget(save_cond_btn); cond_lay.addWidget(load_cond_btn)
        rl.addLayout(cond_lay)

        exp_btn = QPushButton("Export Results (CSV)"); exp_btn.clicked.connect(self.export_data); rl.addWidget(exp_btn)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken); rl.addWidget(sep)
        comp_btn = QPushButton("\u2697  Composition Analysis…"); comp_btn.setStyleSheet("background:#0173B2;color:white;padding:12px;font-weight:bold;font-size:12pt;border-radius:4px;")
        comp_btn.clicked.connect(self.open_composition); rl.addWidget(comp_btn)

        rl.addStretch(); rs.setWidget(rw); ml.addWidget(rs, 3)

    def show_fit_report(self):
        if not hasattr(self, 'last_fit_report') or not self.last_fit_report: return
        msg = QDialog(self); msg.setWindowTitle("lmfit Fit Report"); msg.resize(600, 600)
        lay = QVBoxLayout(msg)
        txt = QTextEdit(); txt.setReadOnly(True); txt.setFont(QFont("Courier", 10))
        txt.setText(self.last_fit_report)
        lay.addWidget(txt)
        msg.exec_()

    def save_state(self):
        if not self.current_region: return
        r = self.region_data[self.current_region]
        self.undo_manager.push_state({
            'peaks': [w.get_parameters() for w in self.peak_widgets],
            'bg': {'isSubtracted': r['backgroundSubtracted'], 'startBE': self.bg_start.value(), 'endBE': self.bg_end.value(),
                   'backgroundData': copy.deepcopy(r.get('backgroundData')), 'regionData': copy.deepcopy(r['original']['data'])},
            'region': self.current_region
        })

    def undo_action(self):
        st = self.undo_manager.pop_state()
        if not st: return
        if self.current_region != st['region']: self.region_combo.setCurrentText(st['region'])
        r = self.region_data[self.current_region]
        self.bg_start.setValue(st['bg']['startBE'])
        self.bg_end.setValue(st['bg']['endBE'])
        r['backgroundSubtracted'] = st['bg']['isSubtracted']
        r['backgroundData']       = st['bg']['backgroundData']
        r['original']['data']     = st['bg']['regionData']
        self._sync_bg_btn(r['backgroundSubtracted'])
        for w in self.peak_widgets: w.deleteLater()
        self.peak_widgets.clear()
        for p in st['peaks']: self._create_peak_from_params(p)
        self.update_plot()

    def _sync_bg_btn(self, subtracted):
        if subtracted:
            self.bg_btn.setText("Reset / Remove Background"); self.bg_btn.setStyleSheet("background:#d32f2f;color:white;padding:10px;")
            self.peak_grp.setEnabled(True)
        else:
            self.bg_btn.setText("Calculate Background"); self.bg_btn.setStyleSheet("background:#029E73;color:white;padding:10px;font-weight:bold;")
            self.peak_grp.setEnabled(False)

    @staticmethod
    def _norm_region(name): return name.lower().replace('_', ' ').strip()

    def _capture_region_state(self):
        if not self.current_region or not self.region_data.get(self.current_region): return
        r = self.region_data[self.current_region]
        peaks = [w.get_parameters() for w in self.peak_widgets]
        r['peaks'] = peaks; r['bgStart'] = self.bg_start.value(); r['bgEnd'] = self.bg_end.value()
        self.region_templates[self._norm_region(self.current_region)] = {
            'peaks': copy.deepcopy(peaks), 'bgStart': self.bg_start.value(), 'bgEnd': self.bg_end.value()}

    def _restore_region_state(self, name):
        for w in self.peak_widgets: w.deleteLater()
        self.peak_widgets.clear()
        r = self.region_data.get(name)
        if r is None: return
        if r.get('bgStart') is not None:
            self.bg_start.setValue(r['bgStart']); self.bg_end.setValue(r['bgEnd'])
        else: self.auto_set_bounds()
        for p in r.get('peaks', []): self._create_peak_from_params(p)
        self._sync_bg_btn(r.get('backgroundSubtracted', False))

    def _set_calibration_offset(self, new_offset):
        if not self.region_data: return
        delta = new_offset - self.calibration_offset
        if abs(delta) < 1e-9: self._update_calib_label(); return
        self._capture_region_state()
        for name, r in self.region_data.items():
            exc = r['original']['metadata'].get('excitationEnergy', 1486.6)
            for d in r['original']['data']: d['be'] = (exc - d['ke']) + new_offset
            if r.get('backgroundData'):
                for b in r['backgroundData']: b['be'] = b['be'] + delta
            for p in r.get('peaks', []): p['position'] = p.get('position', 0.0) + delta
            if r.get('bgStart') is not None:
                r['bgStart'] = r['bgStart'] + delta; r['bgEnd'] = r['bgEnd'] + delta
        self.calibration_offset = new_offset
        self._switching_region = True; self._restore_region_state(self.current_region); self._switching_region = False
        self.update_plot(); self._update_calib_label()

    def calibrate_to_reference(self):
        if not self.region_data: QMessageBox.warning(self, "No Data", "Load an .XY file first."); return
        measured = self.cal_measured.value(); target = self.cal_target.value()
        if measured <= 0: QMessageBox.warning(self, "Invalid", "Enter the current (measured) BE of your reference peak."); return
        self._set_calibration_offset(self.calibration_offset + (target - measured))

    def apply_manual_offset(self):
        if not self.region_data: QMessageBox.warning(self, "No Data", "Load an .XY file first."); return
        self._set_calibration_offset(self.cal_offset_spin.value())

    def _update_calib_label(self):
        self.calib_offset_label.setText(f"Current offset: {self.calibration_offset:+.4f} eV")
        self.cal_offset_spin.blockSignals(True); self.cal_offset_spin.setValue(self.calibration_offset); self.cal_offset_spin.blockSignals(False)

    def load_file(self):
        fn, _ = QFileDialog.getOpenFileName(self,"Open XY File", self.last_dir,"XY Files (*.xy);;All Files (*)")
        if fn:
            self._remember_dir(fn)
            try: self.parse_xy_file(fn)
            except Exception as e: QMessageBox.critical(self,"Error",str(e))

    def parse_xy_file(self, filename):
        self.regions = []; cr = None; meta = {}; data = []; in_data = False
        with open(filename,'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('# Region:'):
                    if cr and data: self.regions.append({'name':cr,'metadata':meta.copy(),'data':list(data)})
                    cr = line.split(':',1)[1].strip(); meta = {'region':cr}; data = []; in_data = False
                elif line.startswith('# Excitation Energy:'): meta['excitationEnergy'] = float(line.split(':')[1].strip())
                elif line.startswith('# ColumnLabels:'): in_data = True
                elif in_data and line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            ke = float(parts[0]); counts = float(parts[1]); be = meta.get('excitationEnergy',1486.6) - ke
                            data.append({'ke':ke,'be':be,'counts':counts})
                        except: continue
        if cr and data: self.regions.append({'name':cr,'metadata':meta,'data':data})

        self._capture_region_state()
        self._switching_region = True
        for w in self.peak_widgets: w.deleteLater()
        self.peak_widgets.clear()
        self.calibration_offset = 0.0
        self._update_calib_label()

        self.region_data = {}
        for r in self.regions:
            st = {'original':r, 'backgroundSubtracted':False, 'backgroundRange':None, 'backgroundData':None,
                  'peaks':[], 'showPeaks':False, 'bgStart':None, 'bgEnd':None}
            tmpl = self.region_templates.get(self._norm_region(r['name']))
            if tmpl:
                st['peaks'] = copy.deepcopy(tmpl['peaks']); st['bgStart'] = tmpl['bgStart']; st['bgEnd'] = tmpl['bgEnd']
            self.region_data[r['name']] = st

        self.region_combo.clear()
        self.region_combo.addItems([r['name'] for r in self.regions])
        self._switching_region = False
        if self.regions:
            self.current_region = self.regions[0]['name']
            self._restore_region_state(self.current_region)
            self.update_plot()

    def auto_set_bounds(self):
        if not self.current_region: return
        bv = [d['be'] for d in self.region_data[self.current_region]['original']['data']]
        if bv: self.bg_start.setValue(max(bv)); self.bg_end.setValue(min(bv))

    def calculate_shirley_background(self, data, start_be, end_be):
        sd = sorted(data, key=lambda x: x['be'], reverse=True)
        si = next((i for i,d in enumerate(sd) if d['be']<=start_be), 0)
        ei = next((i for i,d in enumerate(sd) if d['be']<=end_be),   len(sd)-1)
        if si >= ei: return None
        sub = sd[si:ei+1]; n = len(sub); I_l = sub[0]['counts']; I_r = sub[-1]['counts']
        bg = np.full(n, I_r); counts = np.array([d['counts'] for d in sub]); diff = I_l - I_r
        for _ in range(15):
            sig = np.maximum(counts - bg, 0); cum = np.cumsum(sig[::-1])[::-1]; tot = cum[0]
            if tot == 0: break
            nbg = I_r + diff*(cum/tot)
            if np.max(np.abs(nbg-bg)) < 1e-5: bg = nbg; break
            bg = nbg
        full = []
        for i,d in enumerate(sd): full.append({'be':d['be'],'counts': bg[i-si] if si<=i<=ei else 0})
        return full

    def toggle_background(self): self.save_state(); self._exec_toggle_bg()

    def _exec_toggle_bg(self):
        if not self.current_region: return
        r = self.region_data[self.current_region]
        if r['backgroundSubtracted']:
            r['backgroundSubtracted'] = False; self._sync_bg_btn(False); self.update_plot(); return
        v1 = self.bg_start.value(); v2 = self.bg_end.value()
        s = max(v1,v2); e = min(v1,v2)
        bg = self.calculate_shirley_background(r['original']['data'], s, e)
        if bg:
            r['backgroundSubtracted'] = True; r['backgroundData'] = bg; bgm = {b['be']:b['counts'] for b in bg}
            dc = [d.copy() for d in r['original']['data']]
            for d in dc: d['countsSubtracted'] = 0 if (d['be']>s or d['be']<e) else max(0, d['counts']-bgm.get(d['be'],0))
            self.region_data[self.current_region]['original']['data'] = dc
            self._sync_bg_btn(True); self.update_plot(); self._capture_region_state()

    def _next_label(self):
        used = {w.label for w in self.peak_widgets if getattr(w, 'label', '')}
        n = 1
        while f"P{n}" in used: n += 1
        return f"P{n}"

    def _create_peak_instance(self):
        if len(self.peak_widgets) >= 15: return None
        col = COLORS['peaks'][len(self.peak_widgets)%len(COLORS['peaks'])]
        w = PeakWidget(len(self.peak_widgets)+1, col, callback=self.save_state)
        w.set_label(self._next_label())
        w.remove_btn.clicked.connect(lambda: self.remove_peak_action(w))
        self.peaks_container.addWidget(w); self.peak_widgets.append(w)
        w.position_input.setValue((self.bg_start.value()+self.bg_end.value())/2)
        return w

    def add_peak_action(self):
        self.save_state(); self._create_peak_instance()

    def add_doublet_action(self):
        self.save_state()
        p1 = self._create_peak_instance()
        if not p1: return
        p1.name_input.setText(f"Peak {p1.peak_number} (3/2)")
        
        p2 = self._create_peak_instance()
        if not p2: return
        p2.name_input.setText(f"Peak {p2.peak_number} (1/2)")
        p2.set_color(p1.color)        # doublet partners share one colour
        
        p2.area_constraint.setText(f"0.5 * {p1.label}")
        p2.pos_constraint.setText(f"{p1.label} + 3.0")
        p2.sigma_constraint.setText(f"{p1.label}")
        
        self.update_plot()

    def remove_peak_action(self, w): self.save_state(); self._remove_peak(w)
    def _create_peak_from_params(self, params):
        w = self._create_peak_instance()
        w.set_label(params.get('label') or w.label)
        w.set_parameters(params)

    def _remove_peak(self, w): self.peak_widgets.remove(w); w.deleteLater(); self.update_plot()

    def apply_constraints(self, params_list):
        out = [p.copy() for p in params_list]
        _safe = {"__builtins__": {}}
        label_map = {p.get('label'): p for p in params_list if p.get('label')}
        for p in out:
            for field, ckey in [('position', 'positionConstraint'), ('area', 'areaConstraint'), ('sigma', 'sigmaConstraint')]:
                expr = p.get(ckey, '').strip()
                if not expr: continue
                try:
                    e = expr
                    for lbl, ref in label_map.items(): e = re.sub(r'\b' + re.escape(lbl) + r'\b', f"({ref[field]})", e)
                    p[field] = float(eval(e, _safe))
                except Exception: pass
        return out

    def calculate_peak_shape(self, be_arr, pos, sigma, area, shape_str):
        return lineshapes.evaluate(shape_str, be_arr, pos, sigma, area)

    def _translate_constraint(self, expr, label_map, suffix):
        e = expr
        for lbl, prefix in label_map.items(): e = re.sub(r'\b' + re.escape(lbl) + r'\b', f"{prefix}{suffix}", e)
        return e

    def _parse_constraint(self, text):
        t = (text or '').strip()
        if not t: return {'kind': 'none'}
        has_cmp = any(c in t for c in '<>:[') or '..' in t
        if not has_cmp: return {'kind': 'expr', 'raw': t}
        m = re.match(r'^([<>])=?\s*(.+)$', t)
        if m and re.search(r'\bP\d+\b', m.group(2)): return {'kind': 'ineq', 'op': m.group(1), 'ref': m.group(2).strip()}
        rng = re.match(r'^\[?\s*(' + _CNUM + r')\s*(?:,|:|\.\.)\s*(' + _CNUM + r')\s*\]?$', t)
        lo = hi = None
        if rng: lo, hi = float(rng.group(1)), float(rng.group(2))
        else:
            for mm in re.finditer(r'([<>])=?\s*(' + _CNUM + r')', t):
                v = float(mm.group(2))
                if mm.group(1) == '>': lo = v
                else: hi = v
        if lo is not None and hi is not None and lo > hi: lo, hi = hi, lo
        if lo is None and hi is None: return {'kind': 'none'}
        return {'kind': 'bound', 'lo': lo, 'hi': hi}

    def fit_region(self):
        if not self.current_region or not self.peak_widgets: QMessageBox.warning(self, "Nothing to fit", "Add at least one peak first."); return
        r = self.region_data[self.current_region]
        if not r.get('backgroundSubtracted'): QMessageBox.warning(self, "Subtract background first", "Calculate the Shirley background before fitting."); return

        s_be = max(self.bg_start.value(), self.bg_end.value())
        e_be = min(self.bg_start.value(), self.bg_end.value())
        pts = [(d['be'], d.get('countsSubtracted', 0.0), d.get('counts', 0.0)) for d in r['original']['data'] if e_be <= d['be'] <= s_be]
        if len(pts) < 5: QMessageBox.warning(self, "Not enough data", "Too few points in the background window to fit."); return
        pts.sort(key=lambda t: t[0])
        x = np.array([p[0] for p in pts], dtype=float); y = np.array([p[1] for p in pts], dtype=float); raw = np.array([p[2] for p in pts], dtype=float)
        xmin, xmax = float(x.min()), float(x.max())
        weights = 1.0 / np.sqrt(np.clip(raw, 1.0, None)) if self.weight_chk.isChecked() else None

        self.save_state()
        widgets = list(self.peak_widgets); label_map = {}
        for i, w in enumerate(widgets): label_map[w.label or f"P{i+1}"] = f"{w.label or f'P{i+1}'}_"

        comp = None; wpar = [w.get_parameters() for w in widgets]; prefixes = []
        for i, (w, p) in enumerate(zip(widgets, wpar)):
            prefix = label_map.get(w.label) or f"P{i+1}_"
            prefixes.append(prefix)
            fam = p.get('family') or lineshapes.split_lineshape(p['lineShape'])[0]
            model = lineshapes.make_model(fam, prefix)
            comp = model if comp is None else comp + model
        params = comp.make_params()

        seed_vals = {}
        for prefix, p in zip(prefixes, wpar):
            fam = p.get('family') or lineshapes.split_lineshape(p['lineShape'])[0]
            
            params[prefix + 'center'].set(value=p['position'], min=xmin - 2.0, max=xmax + 2.0, vary=not p.get('fix_pos', False))
            params[prefix + 'amplitude'].set(value=max(p['area'], 0.0), min=0.0, vary=not p.get('fix_area', False))
            params[prefix + 'sigma'].set(value=max(p['sigma'], 1e-4), min=1e-4, vary=not p.get('fix_sigma', False))
            
            seed_vals[prefix + 'center'] = p['position']; seed_vals[prefix + 'amplitude'] = max(p['area'], 0.0); seed_vals[prefix + 'sigma'] = max(p['sigma'], 1e-4)
            names = lineshapes.extra_param_names(fam)
            
            # Now explicitly applying the fix status to shape parameters
            if 'gamma' in names:
                gmax = 1.0 if fam == 'DS' else None
                params[prefix + 'gamma'].set(value=max(p['gamma'], 0.0), min=0.0, max=gmax, vary=not p.get('fix_gamma', False), expr='')
            if 'fraction' in names: 
                params[prefix + 'fraction'].set(value=float(np.clip(p['fraction'], 0, 1)), min=0.0, max=1.0, vary=not p.get('fix_fraction', False))
            if 'skew' in names: 
                params[prefix + 'skew'].set(value=p['skew'], vary=not p.get('fix_skew', False))

        _seed_env = {"__builtins__": {}}
        for prefix, p in zip(prefixes, wpar):
            for ckey, suffix in (('positionConstraint', 'center'), ('areaConstraint', 'amplitude'), ('sigmaConstraint', 'sigma')):
                pname = prefix + suffix
                spec = self._parse_constraint(p.get(ckey))
                try:
                    if spec['kind'] == 'expr': params[pname].set(expr=self._translate_constraint(spec['raw'], label_map, suffix))
                    elif spec['kind'] == 'bound':
                        kw, cur = {}, params[pname].value
                        if spec['lo'] is not None: kw['min'] = spec['lo']; cur = max(cur, spec['lo'])
                        if spec['hi'] is not None: kw['max'] = spec['hi']; cur = min(cur, spec['hi'])
                        params[pname].set(value=cur, vary=True, expr='', **kw)
                    elif spec['kind'] == 'ineq':
                        ref = self._translate_constraint(spec['ref'], label_map, suffix)
                        try: refval = float(eval(ref, _seed_env, seed_vals))
                        except Exception: refval = seed_vals.get(pname, params[pname].value)
                        step = 1.0 if suffix == 'amplitude' else 0.05
                        gap = (seed_vals.get(pname, params[pname].value) - refval if spec['op'] == '>' else refval - seed_vals.get(pname, params[pname].value))
                        dname = pname + ('_dlo' if spec['op'] == '>' else '_dhi')
                        params.add(dname, value=max(gap, step), min=0.0)
                        sign = '+' if spec['op'] == '>' else '-'
                        params[pname].set(expr=f"{ref} {sign} {dname}")
                except Exception: pass

        try: result = comp.fit(y, params, x=x, weights=weights)
        except Exception as ex: QMessageBox.critical(self, "Fit failed", str(ex)); return

        for prefix, w in zip(prefixes, widgets):
            rp = result.params
            for spin, name in ((w.position_input, 'center'), (w.area_input, 'amplitude'), (w.sigma_input, 'sigma'), (w.gamma_spin, 'gamma'), (w.fraction_spin, 'fraction'), (w.skew_spin, 'skew')):
                if (prefix + name) in rp:
                    spin.blockSignals(True); spin.setValue(float(rp[prefix + name].value)); spin.blockSignals(False)
            w.update_readouts()

        mode = "Poisson-weighted" if weights is not None else "unweighted"
        self.fit_result_lbl.setText(f"Fit {'OK' if result.success else 'did not converge'} \u2014 reduced \u03c7\u00b2 = {result.redchi:.4g} ({mode}), {result.nfev} evals, {result.nvarys} free params.")
        
        self.last_fit_report = result.fit_report()
        self.report_btn.setEnabled(True)

        self.show_btn.setChecked(True); self.update_plot()

    def toggle_show_peaks(self): self.update_plot()

    # ── interactive dragging on the plot ───────────────────────────────────────
    def toggle_edit_mode(self, on):
        self.edit_mode = bool(on)
        if on and not self.show_btn.isChecked():
            self.show_btn.setChecked(True)
        self._drag = None
        self.update_plot()

    def _nearest_handle(self, event, thresh_px=14):
        if not self._handles or event.x is None:
            return None
        trans = self.canvas.axes.transData
        best, best_d = None, thresh_px
        for h in self._handles:
            px, py = trans.transform((h['x'], h['y'] / self.canvas.yscale))
            d = ((px - event.x) ** 2 + (py - event.y) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = h, d
        return best

    def on_plot_press(self, event):
        if (not self.edit_mode or event.button != 1
                or event.inaxes is not self.canvas.axes):
            return
        if getattr(self.toolbar, 'mode', ''):     # pan/zoom active -> ignore
            return
        h = self._nearest_handle(event)
        if h is None:
            return
        self.save_state()
        self._drag = {'i': h['i'], 'role': h['role']}

    def on_plot_motion(self, event):
        if (not self._drag or event.inaxes is not self.canvas.axes
                or event.xdata is None):
            return
        w = self.peak_widgets[self._drag['i']]
        shape = w._shape_string()
        if self._drag['role'] == 'move':
            new_be = float(event.xdata)
            new_h = max(float(event.ydata) * self.canvas.yscale, 0.0)
            h1 = lineshapes.fwhm_height(shape, new_be, w.sigma_input.value(), 1.0)[1]
            amp = new_h / h1 if h1 > 0 else w.area_input.value()
            for spin, val in ((w.position_input, new_be), (w.area_input, amp)):
                spin.blockSignals(True); spin.setValue(val); spin.blockSignals(False)
        else:
            center = w.position_input.value()
            target_fwhm = 2.0 * abs(float(event.xdata) - center)
            cur_fwhm = lineshapes.fwhm_height(
                shape, center, w.sigma_input.value(), w.area_input.value())[0]
            if cur_fwhm > 0 and target_fwhm > 0:
                new_sigma = max(w.sigma_input.value() * target_fwhm / cur_fwhm, 1e-4)
                w.sigma_input.blockSignals(True); w.sigma_input.setValue(new_sigma)
                w.sigma_input.blockSignals(False)
        w.update_readouts()
        self.update_plot()

    def on_plot_release(self, event):
        if self._drag is not None:
            self._drag = None
            self.update_plot()

    # ── file-dialog directory memory ───────────────────────────────────────────
    def _start_path(self, filename):
        return os.path.join(self.last_dir, filename) if self.last_dir else filename

    def _remember_dir(self, fn):
        if fn:
            self.last_dir = os.path.dirname(fn)

    # ── fitting conditions (material recipes) ──────────────────────────────────
    def save_fit_conditions(self):
        if not self.peak_widgets:
            QMessageBox.warning(self, "Nothing to save", "Add at least one peak first.")
            return
        default = f"{self.current_region or 'material'}_conditions.json"
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Fit Conditions", self._start_path(default),
            "Fit Conditions (*.json);;All Files (*)")
        if not fn:
            return
        if not fn.lower().endswith('.json'):
            fn += '.json'
        recipe = {
            'format': 'xps_fit_conditions',
            'version': 1,
            'material': os.path.splitext(os.path.basename(fn))[0],
            'region': self.current_region,
            'bgStart': self.bg_start.value(),
            'bgEnd': self.bg_end.value(),
            'poissonWeighting': self.weight_chk.isChecked(),
            'peaks': [w.get_parameters() for w in self.peak_widgets],
        }
        try:
            with open(fn, 'w') as f:
                json.dump(recipe, f, indent=2)
            self._remember_dir(fn)
            QMessageBox.information(self, "Saved", f"Fit conditions saved:\n{fn}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def load_fit_conditions(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Load Fit Conditions", self.last_dir,
            "Fit Conditions (*.json);;All Files (*)")
        if not fn:
            return
        try:
            with open(fn, 'r') as f:
                recipe = json.load(f)
            peaks = recipe.get('peaks', [])
            if not peaks:
                QMessageBox.warning(self, "Empty recipe", "No peaks found in that file.")
                return
            self._remember_dir(fn)
            self.save_state()
            # clear existing peaks
            for w in list(self.peak_widgets):
                w.deleteLater()
            self.peak_widgets.clear()
            # optionally restore the background window
            if recipe.get('bgStart') is not None:
                self.bg_start.setValue(recipe['bgStart'])
            if recipe.get('bgEnd') is not None:
                self.bg_end.setValue(recipe['bgEnd'])
            if 'poissonWeighting' in recipe:
                self.weight_chk.setChecked(bool(recipe['poissonWeighting']))
            # rebuild peaks
            for p in peaks:
                if len(self.peak_widgets) >= 15:
                    break
                self._create_peak_from_params(p)
            self.peak_grp.setEnabled(True)
            self.update_plot()
            mat = recipe.get('material', os.path.basename(fn))
            QMessageBox.information(
                self, "Loaded",
                f"Loaded {len(peaks)} peak(s) from '{mat}'.\n"
                "Set your background window and Fit when ready.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


    def update_plot(self):
        if not self.current_region: return
        r = self.region_data[self.current_region]
        peaks_info = None
        if self.show_btn.isChecked() and self.peak_widgets:
            all_p = [w.get_parameters() for w in self.peak_widgets]
            cp    = self.apply_constraints(all_p)
            data  = r['original']['data']
            be_arr= np.array([d['be'] for d in data])
            s_be = max(self.bg_start.value(), self.bg_end.value())
            e_be = min(self.bg_start.value(), self.bg_end.value())
            in_region = (be_arr <= s_be) & (be_arr >= e_be)
            
            pres, names = [], []
            for p in cp:
                ya = self.calculate_peak_shape(be_arr, p['position'], p['sigma'], p['area'], p['lineShape'])
                ya = np.where(in_region, ya, 0.0)
                pres.append([{'be':be,'counts':y} for be,y in zip(be_arr,ya)])
                names.append(p['name'])
            sum_res = [{'be':data[i]['be'], 'counts':sum(pres[j][i]['counts'] for j in range(len(pres)))} for i in range(len(data))]
            colors = [w.color for w in self.peak_widgets]
            peaks_info = {'peaks':pres,'sum':sum_res,'names':names,'colors':colors}

        handles = []
        if self.edit_mode and peaks_info and r['backgroundSubtracted']:
            all_p = [w.get_parameters() for w in self.peak_widgets]
            cp = self.apply_constraints(all_p)
            for i, p in enumerate(cp):
                wp = all_p[i]
                fwhm, height = lineshapes.fwhm_height(
                    p['lineShape'], p['position'], p['sigma'], p['area'])
                if height <= 0:
                    continue
                col = self.peak_widgets[i].color
                move_free = not (wp.get('positionConstraint', '').strip()
                                 or wp.get('areaConstraint', '').strip()
                                 or wp.get('fix_pos') or wp.get('fix_area'))
                width_free = not (wp.get('sigmaConstraint', '').strip()
                                  or wp.get('fix_sigma'))
                if move_free:
                    handles.append({'i': i, 'role': 'move',
                                    'x': p['position'], 'y': height, 'color': col})
                if width_free and fwhm > 0:
                    hw = fwhm / 2.0
                    handles.append({'i': i, 'role': 'width',
                                    'x': p['position'] - hw, 'y': height / 2, 'color': col})
                    handles.append({'i': i, 'role': 'width',
                                    'x': p['position'] + hw, 'y': height / 2, 'color': col})
        self._handles = handles

        self.canvas.plot_spectrum(
            r['original']['data'], r['original']['metadata']['region'], background_subtracted=r['backgroundSubtracted'],
            background_data=r.get('backgroundData'), peaks_data=peaks_info, show_peaks=self.show_btn.isChecked(),
            peak_handles=handles)

    def change_region(self, name):
        if not name or self._switching_region: return
        self._capture_region_state()
        self._switching_region = True; self.current_region = name; self._restore_region_state(name); self._switching_region = False
        self.update_plot()

    def open_composition(self):
        if not self.region_data: QMessageBox.warning(self,"No Data","Load an .XY file first."); return
        dlg = CompositionWindow(self.region_data, self.calculate_peak_shape, self.calculate_shirley_background, parent=self)
        dlg.show()

    def export_data(self):
        if not self.current_region: return
        r = self.region_data[self.current_region]
        sd = sorted(r['original']['data'], key=lambda x: x['ke'])
        fn, _ = QFileDialog.getSaveFileName(self,"Save Results", self._start_path(f"{self.current_region}_fit.csv"),"CSV (*.csv)")
        if not fn: return
        self._remember_dir(fn)
        try:
            be_arr = np.array([d['be'] for d in sd])
            s_be  = max(self.bg_start.value(), self.bg_end.value()); e_be  = min(self.bg_start.value(), self.bg_end.value())
            in_region = (be_arr <= s_be) & (be_arr >= e_be)

            all_p  = [w.get_parameters() for w in self.peak_widgets]
            cp     = self.apply_constraints(all_p)
            pres, pnames = [], []
            summary = []      # per-peak (label, name, shape, position, area, fwhm, height, sigma, gamma, fraction, skew)
            for p in cp:
                pnames.append(p['name'])
                ya = self.calculate_peak_shape(be_arr, p['position'], p['sigma'], p['area'], p['lineShape'])
                ya = np.where(in_region, ya, 0.0)
                pres.append(ya.tolist())
                fwhm, height = lineshapes.fwhm_height(p['lineShape'], p['position'], p['sigma'], p['area'])
                summary.append([
                    p.get('label',''), p.get('name',''), p.get('lineShape',''),
                    f"{p['position']:.4f}", f"{p['area']:.4f}", f"{fwhm:.4f}",
                    f"{height:.4f}", f"{p['sigma']:.4f}",
                    f"{p.get('gamma',0):.4f}", f"{p.get('fraction',0):.4f}", f"{p.get('skew',0):.4f}"])

            hdrs = ["KE (eV)","BE (eV)","Raw Counts","Background","Bg-subtracted"]
            if pres: hdrs += ["Total Fit"] + [f"{n} (Fit)" for n in pnames]
            bgm   = {d['be']:d['counts'] for d in r['backgroundData']} if r.get('backgroundData') else {}

            with open(fn,'w') as f:
                f.write(f"# Region: {self.current_region}\n")
                f.write(f"# Energy calibration offset (eV): {self.calibration_offset:+.4f}\n")
                f.write(f"# Background window (BE eV): {e_be:.4f} to {s_be:.4f}\n")
                if summary:
                    f.write("#\n# Peak Parameters\n")
                    f.write("Label,Name,Line Shape,Position (eV),Area,FWHM (eV),Height,sigma,gamma,fraction,skew\n")
                    for row in summary:
                        f.write(",".join(str(c) for c in row) + "\n")
                f.write("#\n# Spectrum Data\n")
                f.write(",".join(hdrs)+"\n")
                for i, d in enumerate(sd):
                    bv = bgm.get(d['be'],0)
                    sv = 0 if (d['be']>s_be or d['be']<e_be) else max(0, d['counts']-bv)
                    row = [f"{d['ke']:.4f}",f"{d['be']:.4f}",f"{d['counts']:.4f}", f"{bv:.4f}",f"{sv:.4f}"]
                    if pres:
                        tf = sum(pres[j][i] for j in range(len(pres)))
                        row.append(f"{tf:.4f}")
                        for j in range(len(pres)): row.append(f"{pres[j][i]:.4f}")
                    f.write(",".join(row)+"\n")
            QMessageBox.information(self,"Exported",f"Saved:\n{fn}")
        except Exception as e: QMessageBox.critical(self,"Error",str(e))

def main():
    app = QApplication(sys.argv); app.setStyle('Fusion'); gui = XPSAnalysisGUI(); gui.show(); sys.exit(app.exec_())
if __name__ == '__main__': main()